"""Binary segmentation metric helpers for burn scar evaluation."""

import numpy as np


def binary_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Burn-class metrics with -1 (no data) pixels excluded.

    pred: (H,W) integer array, 1 = burn predicted.
    true: (H,W) integer array, 1 = burn, 0 = not burned, -1 = ignore.
    """
    valid = true != -1
    p = pred[valid] == 1
    t = true[valid] == 1
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def dnbr_mask(dnbr: np.ndarray, threshold: float = 0.27) -> np.ndarray:
    """Binary burned mask from dNBR (>= threshold), NaN pixels excluded."""
    return np.isfinite(dnbr) & (dnbr >= threshold)


def aggregate_metrics(results: list[dict]) -> dict[str, float]:
    """Mean of each float metric across results."""
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
