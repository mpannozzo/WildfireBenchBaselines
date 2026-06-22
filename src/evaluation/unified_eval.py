"""
src/evaluation/unified_eval.py

Unified evaluation harness for WildfireSpreadBench.

The discriminative Lightning baselines (ResNet U-Net, ConvLSTM, UTAE,
Logistic Regression, Persistence) are evaluated through
evaluate_lightning_module(), which reuses each model's own get_pred_and_gt
path so temporal flattening, doy features, and tiled inference all match the
training-time code. This guarantees all numbers in the results tables come
from identical data, thresholds, and metric implementations.

evaluate_model() is a generic, model-agnostic harness: pass any
predict_fn(x0) -> probability map and it computes the same metrics. It is
provided for evaluating non-Lightning models (e.g. a future diffusion or
flow-matching baseline) through the identical metric code.

predict_fn signature
--------------------
    predict_fn(x0: Tensor) -> Tensor
    Input:  x0  — conditioning tensor already on `device`. As yielded by the
                  loader this is [B, T, C, H, W]; predict_fn is responsible for
                  any reshaping (e.g. squeezing the leading time dim).
    Output: probability map in [0, 1], shape [B, 1, H, W]

max_batches: optional cap on the number of eval batches, useful for keeping
periodic in-training eval affordable. Final eval should pass None to use the
entire test set.

Metrics
-------
    AP, F1, Precision, Recall, IoU. AP is the primary, threshold-free metric.

Baselines (WildfireSpreadTS, 12-fold mean, "All" feature set)
-------------------------------------------------------------
    Persistence:         AP ~ 0.193
    Logistic Regression: AP ~ 0.279
    ResNet18 U-Net:      AP ~ 0.328
    ConvLSTM:            AP ~ 0.306
    UTAE (vegetation):   AP ~ 0.372
  NOTE: these are 12-fold means. Single-fold (fold 0, test=2021) numbers
  run noticeably higher because 2021 is an easy test year (see suppl. Fig 2).
"""

import numpy as np
import torch
import wandb
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    predict_fn,
    eval_loader,
    device,
    model_name: str = "model",
    epoch=None,
    threshold: float = 0.5,
    wandb_log: bool = True,
    verbose: bool = True,
    max_batches=None,
) -> dict:
    """
    Run inference over the eval loader and compute all benchmark metrics.

    eval_loader yields (x0, x1). x0 = conditioning (as stored by the dataset,
    i.e. possibly [B, T, C, H, W]); x1 = binary target. predict_fn is
    responsible for any reshaping its model needs.

    max_batches: if set, evaluate only the first `max_batches` batches. Used to
    keep DDPM periodic eval affordable; final eval should pass None.

    Returns dict with keys: ap, f1, precision, recall, iou (all Python floats).
    """
    tag = f"epoch_{epoch}" if epoch is not None else "final"

    all_targets = []
    all_scores  = []

    with torch.no_grad():
        for step, (x0, x1) in enumerate(eval_loader):
            if max_batches is not None and step >= max_batches:
                break
            x0 = x0.to(device)
            x1 = x1.to(device)

            prob = predict_fn(x0)          # [B, 1, H, W], values in [0, 1]

            all_targets.append(x1.cpu().numpy().flatten())
            all_scores.append(prob.cpu().numpy().flatten())

            if verbose and step % 50 == 0:
                print(f"  [{model_name}] {step}/{len(eval_loader)} batches...")

    targets_flat = np.concatenate(all_targets)
    scores_flat  = np.concatenate(all_scores)
    binary_flat  = (scores_flat >= threshold).astype(np.float32)

    ap = float(average_precision_score(targets_flat, scores_flat))

    p, r, f1, _ = precision_recall_fscore_support(
        targets_flat, binary_flat,
        labels=[1], average=None, zero_division=0,
    )
    precision = float(p[0])
    recall    = float(r[0])
    f1_score  = float(f1[0])

    tp = float(np.sum((binary_flat == 1) & (targets_flat == 1)))
    fp = float(np.sum((binary_flat == 1) & (targets_flat == 0)))
    fn = float(np.sum((binary_flat == 0) & (targets_flat == 1)))
    iou = tp / (tp + fp + fn + 1e-8)

    results = dict(ap=ap, f1=f1_score, precision=precision, recall=recall, iou=iou)

    if verbose:
        _print_results(model_name, tag, results, threshold)
    if wandb_log:
        _log_to_wandb(model_name, tag, epoch, results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_results(model_name: str, tag: str, results: dict, threshold: float):
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Model:     {model_name}  ({tag})")
    print(f"  Threshold: {threshold}")
    print(f"  ---")
    print(f"  AP        (threshold-free): {results['ap']:.4f}")
    print(f"  F1        (thr={threshold}):       {results['f1']:.4f}")
    print(f"  Precision (thr={threshold}):       {results['precision']:.4f}")
    print(f"  Recall    (thr={threshold}):       {results['recall']:.4f}")
    print(f"  IoU       (thr={threshold}):       {results['iou']:.4f}")
    print(f"  ---")
    print(f"  Persistence baseline AP (12-fold mean): 0.193")
    print(f"  ResNet18 U-Net AP (12-fold mean):       ~0.328")
    print(f"  (Single-fold test=2021 runs higher; see suppl. Fig 2.)")
    print(f"{sep}\n")


def _log_to_wandb(model_name: str, tag: str, epoch, results: dict):
    prefix = f"eval/{model_name}"
    log_dict = {f"{prefix}/{k}": v for k, v in results.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    wandb.log(log_dict)


# ---------------------------------------------------------------------------
# Lightning model evaluation (discriminative baselines)
# ---------------------------------------------------------------------------

def _move_batch_to_device(batch, device):
    if len(batch) == 3:
        x, y, doys = batch
        return x.to(device), y.to(device), doys.to(device)
    x, y = batch
    return x.to(device), y.to(device)


def _collect_lightning_predictions(
    pl_module,
    eval_loader,
    device,
    model_name: str,
    verbose: bool,
    max_batches=None,
):
    """Run inference via the Lightning module's own get_pred_and_gt path."""
    pl_module.eval()

    all_targets = []
    all_scores = []

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            if max_batches is not None and step >= max_batches:
                break

            batch = _move_batch_to_device(batch, device)
            y_hat, y = pl_module.get_pred_and_gt(batch)
            prob = torch.sigmoid(y_hat)

            all_targets.append(y.cpu().numpy().flatten())
            all_scores.append(prob.cpu().numpy().flatten())

            if verbose and step % 50 == 0:
                print(f"  [{model_name}] {step}/{len(eval_loader)} batches...")

    return all_targets, all_scores


def evaluate_lightning_module(
    pl_module,
    eval_loader,
    device,
    model_name: str = "model",
    epoch=None,
    threshold: float = 0.5,
    wandb_log: bool = True,
    verbose: bool = True,
    max_batches=None,
) -> dict:
    """
    Evaluate a PyTorch Lightning BaseModel subclass through the unified harness.

    Uses get_pred_and_gt so temporal flattening, doy features, and tiled
    inference all match the training-time code path.
    """
    tag = f"epoch_{epoch}" if epoch is not None else "final"

    all_targets, all_scores = _collect_lightning_predictions(
        pl_module=pl_module,
        eval_loader=eval_loader,
        device=device,
        model_name=model_name,
        verbose=verbose,
        max_batches=max_batches,
    )

    targets_flat = np.concatenate(all_targets)
    scores_flat = np.concatenate(all_scores)
    binary_flat = (scores_flat >= threshold).astype(np.float32)

    ap = float(average_precision_score(targets_flat, scores_flat))

    p, r, f1, _ = precision_recall_fscore_support(
        targets_flat, binary_flat,
        labels=[1], average=None, zero_division=0,
    )
    precision = float(p[0])
    recall = float(r[0])
    f1_score = float(f1[0])

    tp = float(np.sum((binary_flat == 1) & (targets_flat == 1)))
    fp = float(np.sum((binary_flat == 1) & (targets_flat == 0)))
    fn = float(np.sum((binary_flat == 0) & (targets_flat == 1)))
    iou = tp / (tp + fp + fn + 1e-8)

    results = dict(ap=ap, f1=f1_score, precision=precision, recall=recall, iou=iou)

    if verbose:
        _print_results(model_name, tag, results, threshold)
    if wandb_log:
        _log_to_wandb(model_name, tag, epoch, results)

    return results


def evaluate_lightning_model(
    model, datamodule, device, model_name: str, wandb_log: bool = True,
) -> dict:
    """
    Evaluate a PyTorch Lightning BaseModel subclass through the unified harness,
    giving discriminative models identical metric computation to generative ones.
    """
    model.to(device)

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    return evaluate_lightning_module(
        pl_module=model,
        eval_loader=test_loader,
        device=device,
        model_name=model_name,
        epoch=None,
        wandb_log=wandb_log,
    )
