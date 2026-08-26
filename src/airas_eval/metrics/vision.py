"""Computer-vision metrics on label masks and images, pure functions.

Segmentation metrics take integer class masks of identical shape. Image-quality
metrics take float arrays in a stated value range. Model-dependent scores (FID,
LPIPS, ...) are out of scope for the pure core by design.
"""

from typing import Any

import numpy as np

ArrayLike = Any


def _paired_masks(
    predicted: ArrayLike, reference: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(predicted)
    y_true = np.asarray(reference)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: {y_pred.shape} vs {y_true.shape}")
    if y_pred.size == 0:
        raise ValueError("cannot compute a metric on empty masks")
    return y_pred, y_true


def pixel_accuracy(predicted: ArrayLike, reference: ArrayLike) -> float:
    y_pred, y_true = _paired_masks(predicted, reference)
    return float(np.mean(y_pred == y_true))


def binary_iou(predicted: ArrayLike, reference: ArrayLike) -> float:
    """Jaccard index of the foreground (nonzero) region."""
    y_pred, y_true = _paired_masks(predicted, reference)
    pred_fg = y_pred.astype(bool)
    true_fg = y_true.astype(bool)
    union = np.logical_or(pred_fg, true_fg).sum()
    if union == 0:
        raise ValueError("IoU is undefined when both masks are empty")
    return float(np.logical_and(pred_fg, true_fg).sum() / union)


def dice_coefficient(predicted: ArrayLike, reference: ArrayLike) -> float:
    y_pred, y_true = _paired_masks(predicted, reference)
    pred_fg = y_pred.astype(bool)
    true_fg = y_true.astype(bool)
    total = pred_fg.sum() + true_fg.sum()
    if total == 0:
        raise ValueError("Dice is undefined when both masks are empty")
    return float(2.0 * np.logical_and(pred_fg, true_fg).sum() / total)


def mean_iou(
    predicted: ArrayLike, reference: ArrayLike, num_classes: int | None = None
) -> float:
    """Mean per-class IoU over classes present in reference or prediction."""
    y_pred, y_true = _paired_masks(predicted, reference)
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    ious = []
    for cls in classes:
        pred_c = y_pred == cls
        true_c = y_true == cls
        union = np.logical_or(pred_c, true_c).sum()
        if union == 0:
            continue
        ious.append(float(np.logical_and(pred_c, true_c).sum() / union))
    if not ious:
        raise ValueError("mIoU is undefined: no class present in either mask")
    return float(np.mean(ious))


def psnr(predicted: ArrayLike, reference: ArrayLike, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB; identical images raise (infinite PSNR)."""
    y_pred, y_true = _paired_masks(
        np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    )
    mse = float(np.mean((y_pred - y_true) ** 2))
    if mse == 0.0:
        raise ValueError("PSNR is infinite for identical images")
    return float(10.0 * np.log10(data_range**2 / mse))
