"""
Supervised OOD detection metrics.

All primitive functions accept raw score and label arrays so they can be used
independently of ``ScoreBank``.  The ``evaluate`` / ``evaluate_by_class``
convenience wrappers take a ``ScoreBank`` directly.

All metrics follow the library convention: **higher score = more OOD**,
**ood_labels: 0 = ID, 1 = OOD**.

Implemented with NumPy only (no sklearn required).
"""

from typing import Dict, Tuple

import numpy as np

from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

from oodkit.evaluation.score_bank import ScoreBank


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal rule; NumPy 2+ uses ``trapezoid``, NumPy 1.x uses ``trapz``."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# ------------------------------------------------------------------
# Curve primitives
# ------------------------------------------------------------------

def roc_curve(
    scores: ArrayLike,
    ood_labels: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the ROC curve for a single detector.

    Args:
        scores: Per-sample OOD scores, shape ``(n_samples,)``.  Higher = more OOD.
        ood_labels: Ground-truth labels, shape ``(n_samples,)``.  0=ID, 1=OOD.

    Returns:
        ``(fpr, tpr)`` — two arrays of length ``n_samples + 1`` including the
        (0, 0) origin point.
    """
    s = to_numpy(scores).ravel().astype(np.float64)
    y = to_numpy(ood_labels).ravel().astype(np.int64)
    _check_binary_labels(y)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ood_labels must contain both ID (0) and OOD (1) samples.")

    desc_idx = np.argsort(-s, kind="stable")
    y_sorted = y[desc_idx]

    tp = np.concatenate([[0], np.cumsum(y_sorted)])
    fp = np.concatenate([[0], np.cumsum(1 - y_sorted)])

    tpr = tp / n_pos
    fpr = fp / n_neg
    return fpr, tpr


def pr_curve(
    scores: ArrayLike,
    ood_labels: ArrayLike,
    positive: str = "ood",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Precision-Recall curve.

    Args:
        scores: Per-sample OOD scores, shape ``(n_samples,)``.
        ood_labels: Ground-truth labels, shape ``(n_samples,)``.  0=ID, 1=OOD.
        positive: Which class is treated as positive.  ``"ood"`` (default) or
            ``"id"``.

    Returns:
        ``(precision, recall)`` arrays.  The curve starts at recall=0 with a
        sentinel precision value of 1.0.
    """
    s = to_numpy(scores).ravel().astype(np.float64)
    y = to_numpy(ood_labels).ravel().astype(np.int64)
    _check_binary_labels(y)

    if positive == "id":
        s = -s
        y = 1 - y

    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive samples found for the requested 'positive' class.")

    desc_idx = np.argsort(-s, kind="stable")
    y_sorted = y[desc_idx]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    precision = tp / (tp + fp)
    recall = tp / n_pos

    # Prepend (recall=0, precision=1) sentinel
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


# ------------------------------------------------------------------
# Scalar metrics
# ------------------------------------------------------------------

def auroc(scores: ArrayLike, ood_labels: ArrayLike) -> float:
    """Area under the ROC curve (AUROC).

    Args:
        scores: Per-sample OOD scores, shape ``(n_samples,)``.
        ood_labels: Ground-truth labels ``(n_samples,)``, 0=ID, 1=OOD.

    Returns:
        AUROC in ``[0, 1]``.  0.5 = random, 1.0 = perfect.
    """
    fpr, tpr = roc_curve(scores, ood_labels)
    return _trapz(tpr, fpr)


def fpr_at_tpr(
    scores: ArrayLike,
    ood_labels: ArrayLike,
    tpr: float = 0.95,
) -> float:
    """False-positive rate at a fixed true-positive rate (FPR@TPR).

    The standard OOD benchmark metric is FPR@95TPR.

    Args:
        scores: Per-sample OOD scores, shape ``(n_samples,)``.
        ood_labels: Ground-truth labels, 0=ID, 1=OOD.
        tpr: Target TPR threshold (default ``0.95``).

    Returns:
        FPR at the lowest score threshold that achieves ``>= tpr`` TPR.
        Returns ``1.0`` if the target TPR is never reached.
    """
    fpr_arr, tpr_arr = roc_curve(scores, ood_labels)
    # Find the first point where TPR >= target
    idx = np.searchsorted(tpr_arr, tpr, side="left")
    if idx >= len(fpr_arr):
        return 1.0
    return float(fpr_arr[idx])


def aupr(
    scores: ArrayLike,
    ood_labels: ArrayLike,
    positive: str = "ood",
) -> float:
    """Area under the Precision-Recall curve (AUPR).

    Args:
        scores: Per-sample OOD scores, shape ``(n_samples,)``.
        ood_labels: Ground-truth labels, 0=ID, 1=OOD.
        positive: Class treated as positive — ``"ood"`` (default) or ``"id"``.

    Returns:
        AUPR in ``[0, 1]``.
    """
    precision, recall = pr_curve(scores, ood_labels, positive=positive)
    return _trapz(precision, recall)


def detection_error(scores: ArrayLike, ood_labels: ArrayLike) -> float:
    """Minimum detection error over all thresholds.

    Defined as ``min_t 0.5 * (FPR(t) + FNR(t))``.

    Args:
        scores: Per-sample OOD scores, shape ``(n_samples,)``.
        ood_labels: Ground-truth labels, 0=ID, 1=OOD.

    Returns:
        Minimum balanced error rate in ``[0, 1]``.  0 = perfect separation.
    """
    fpr_arr, tpr_arr = roc_curve(scores, ood_labels)
    fnr_arr = 1.0 - tpr_arr
    return float(np.min(0.5 * (fpr_arr + fnr_arr)))


# ------------------------------------------------------------------
# MetricsTable
# ------------------------------------------------------------------

class MetricsTable:
    """Lightweight table of per-detector evaluation metrics.

    Thin wrapper around ``dict[detector_name, dict[metric_name, float]]``
    that adds pretty-printing and dict-style access.  No pandas required.

    Access patterns::

        table["MSP"]["auroc"]          # single value
        table.data                     # full raw dict
        print(table)                   # aligned ASCII table
        table.to_dict()                # copy of the raw dict

    Attributes:
        data: Raw dict mapping detector name → metric name → float value.
    """

    _METRIC_ORDER = ("auroc", "fpr95", "aupr_ood", "aupr_id", "det_err")
    _METRIC_LABELS = {
        "auroc": "AUROC",
        "fpr95": "FPR@95",
        "aupr_ood": "AUPR(OOD)",
        "aupr_id": "AUPR(ID)",
        "det_err": "DetErr",
    }

    def __init__(self, data: Dict[str, Dict[str, float]]) -> None:
        self.data = data

    def __getitem__(self, detector: str) -> Dict[str, float]:
        """Index by detector name.

        Args:
            detector: Name of the detector.

        Returns:
            Dict of metric name → float.

        Raises:
            KeyError: If the detector is not in the table.
        """
        return self.data[detector]

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of the raw data dict."""
        return {det: dict(metrics) for det, metrics in self.data.items()}

    def __repr__(self) -> str:
        if not self.data:
            return "MetricsTable(empty)"

        col_labels = [self._METRIC_LABELS.get(m, m) for m in self._METRIC_ORDER]
        det_col = max(len(d) for d in self.data) + 2
        val_col = max(len(c) for c in col_labels) + 2

        header = f"{'Detector':<{det_col}}" + "".join(f"{c:>{val_col}}" for c in col_labels)
        sep = "-" * len(header)

        rows = [header, sep]
        for det, metrics in self.data.items():
            row = f"{det:<{det_col}}"
            for m in self._METRIC_ORDER:
                v = metrics.get(m)
                row += f"{v:>{val_col}.4f}" if v is not None else f"{'—':>{val_col}}"
            rows.append(row)

        return "\n".join(rows)


# ------------------------------------------------------------------
# Bank-level convenience
# ------------------------------------------------------------------

def evaluate(bank: ScoreBank) -> MetricsTable:
    """Compute standard OOD metrics for all detectors in a ``ScoreBank``.

    Computes AUROC, FPR@95TPR, AUPR with OOD as positive class, AUPR with ID
    as positive class, and detection error for each detector.

    Args:
        bank: A ``ScoreBank`` with at least one detector and ``ood_labels``.

    Returns:
        A ``MetricsTable`` with rows for each detector.

    Raises:
        ValueError: If the bank has no OOD labels or no detectors.
    """
    if not bank.has_ood_labels:
        raise ValueError(
            "evaluate() requires OOD ground-truth labels. "
            "Pass ood_labels when constructing the ScoreBank."
        )
    if not bank.detectors:
        raise ValueError("ScoreBank has no detectors. Add scores with bank.add() first.")

    results: Dict[str, Dict[str, float]] = {}
    for det in bank.detectors:
        s = bank.scores_for(det)
        results[det] = {
            "auroc": auroc(s, bank.ood_labels),
            "fpr95": fpr_at_tpr(s, bank.ood_labels, tpr=0.95),
            "aupr_ood": aupr(s, bank.ood_labels, positive="ood"),
            "aupr_id": aupr(s, bank.ood_labels, positive="id"),
            "det_err": detection_error(s, bank.ood_labels),
        }
    return MetricsTable(results)


def evaluate_by_class(bank: ScoreBank) -> Dict[int, MetricsTable]:
    """Evaluate each detector per class.

    Internally calls ``bank.by_class(c)`` for each class and runs
    ``evaluate`` on the resulting slice.

    Args:
        bank: A ``ScoreBank`` with ``ood_labels`` and ``class_labels``.

    Returns:
        Dict mapping class label (int) → ``MetricsTable``.

    Raises:
        ValueError: If the bank lacks class labels or OOD labels.
    """
    if not bank.has_class_labels:
        raise ValueError(
            "evaluate_by_class() requires class_labels. "
            "Pass class_labels when constructing the ScoreBank."
        )
    return {int(c): evaluate(bank.by_class(int(c))) for c in bank.classes}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _check_binary_labels(y: np.ndarray) -> None:
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"ood_labels must contain only 0 (ID) and 1 (OOD), got values: {unique}"
        )
