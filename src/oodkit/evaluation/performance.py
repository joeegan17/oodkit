"""
OOD score vs downstream model performance analysis.

Answers the question: do higher OOD scores actually correspond to model
failure?  Samples are partitioned into equal-frequency bins by OOD score and
the mean of a user-provided per-sample metric is computed in each bin.

For class-conditional analysis, slice the bank first::

    bank_cls0 = bank.by_class(0)
    curves = score_vs_metric(bank_cls0, "accuracy")
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from oodkit.evaluation.score_bank import ScoreBank


@dataclass
class PerformanceCurve:
    """Result of a single ``score_vs_metric`` computation.

    Attributes:
        bin_edges: Score percentile boundaries, shape ``(n_bins + 1,)``.
        bin_centers: Midpoint of each bin, shape ``(n_bins,)``.
        mean_metric: Mean metric value per bin, shape ``(n_bins,)``.
        n_samples: Number of samples per bin, shape ``(n_bins,)``.
        detector: Name of the detector whose scores were binned.
        metric_name: Name of the user-provided metric.
    """

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    mean_metric: np.ndarray
    n_samples: np.ndarray
    detector: str
    metric_name: str


def score_vs_metric(
    bank: ScoreBank,
    metric_name: str,
    detector: Optional[str] = None,
    n_bins: int = 10,
) -> Union[PerformanceCurve, Dict[str, PerformanceCurve]]:
    """Partition samples by OOD score and compute mean metric per partition.

    Uses equal-frequency (quantile) binning so every bin has a comparable
    number of samples.

    For class-conditional analysis, call ``bank.by_class(c)`` first and pass
    the resulting slice to this function.

    Args:
        bank: A ``ScoreBank`` with at least one detector and the requested
            metric in ``sample_metrics``.
        metric_name: Name of the per-sample metric to analyze (must have been
            added via ``bank.add_metric``).
        detector: If given, compute only for that detector and return a single
            ``PerformanceCurve``.  If ``None`` (default), compute for all
            detectors and return a ``dict``.
        n_bins: Number of equal-frequency bins (default ``10``).

    Returns:
        A single ``PerformanceCurve`` when ``detector`` is specified, or a
        ``dict`` mapping detector name → ``PerformanceCurve`` when
        ``detector=None``.

    Raises:
        ValueError: If ``n_bins < 1`` or the bank has no detectors.
        KeyError: If ``metric_name`` or ``detector`` are not in the bank.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if not bank.detectors:
        raise ValueError("ScoreBank has no detectors. Add scores with bank.add() first.")

    metric_values = bank.metric_for(metric_name)

    if detector is not None:
        return _compute_curve(bank.scores_for(detector), metric_values, n_bins, detector, metric_name)

    return {
        det: _compute_curve(bank.scores_for(det), metric_values, n_bins, det, metric_name)
        for det in bank.detectors
    }


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _compute_curve(
    scores: np.ndarray,
    metric: np.ndarray,
    n_bins: int,
    detector: str,
    metric_name: str,
) -> PerformanceCurve:
    """Build a single PerformanceCurve from score and metric arrays."""
    n = len(scores)
    # Clamp n_bins so we don't create empty bins
    n_bins = min(n_bins, n)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(scores, percentiles)

    # Digitize to bin indices (1-based from np.digitize; clamp to n_bins)
    bin_idx = np.digitize(scores, bin_edges[1:-1])  # 0 … n_bins-1

    mean_metric = np.empty(n_bins, dtype=np.float64)
    counts = np.empty(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        mean_metric[b] = float(metric[mask].mean()) if counts[b] > 0 else np.nan

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return PerformanceCurve(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        mean_metric=mean_metric,
        n_samples=counts,
        detector=detector,
        metric_name=metric_name,
    )
