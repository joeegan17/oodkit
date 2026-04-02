"""
Cross-detector comparison utilities.

All functions that compare scores across detectors are **robust to scale
differences by default** — they operate on ranks internally, so users never
need to call ``normalize_scores`` first.

``normalize_scores`` is provided as a utility for users who want scaled scores
for their own analysis, but no function in this module or ``plots.py`` requires
calling it first.
"""

import numpy as np

from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy
from oodkit.evaluation.score_bank import ScoreBank


def rank_samples(
    bank: ScoreBank,
    detector: str,
    top_k: int = 20,
    direction: str = "ood",
) -> np.ndarray:
    """Return indices of the top-k samples ranked by OOD score.

    Args:
        bank: A ``ScoreBank`` with at least one detector.
        detector: Which detector's scores to rank by.
        top_k: Number of sample indices to return.
        direction: ``"ood"`` (default) returns highest-scoring = most OOD
            samples; ``"id"`` returns lowest-scoring = most ID-like samples.

    Returns:
        Integer index array of length ``min(top_k, n_samples)``, sorted so
        the strongest examples come first.

    Raises:
        ValueError: If ``direction`` is not ``"ood"`` or ``"id"``.
        KeyError: If ``detector`` is not in the bank.
    """
    if direction not in ("ood", "id"):
        raise ValueError(f"direction must be 'ood' or 'id', got {direction!r}")

    scores = bank.scores_for(detector)
    k = min(top_k, len(scores))

    if direction == "ood":
        idx = np.argsort(-scores, kind="stable")
    else:
        idx = np.argsort(scores, kind="stable")

    return idx[:k]


def disagreements(
    bank: ScoreBank,
    det_a: str,
    det_b: str,
    top_k: int = 20,
) -> np.ndarray:
    """Return sample indices where two detectors disagree most.

    Disagreement is measured as absolute rank difference, so the comparison is
    fully robust to scale — a score gap of 0.1 (MSP) and 500 (Mahalanobis)
    are both expressed as positions in the sorted order before comparing.

    Args:
        bank: A ``ScoreBank`` with both detectors present.
        det_a: Name of the first detector.
        det_b: Name of the second detector.
        top_k: Number of indices to return.

    Returns:
        Integer index array of length ``min(top_k, n_samples)`` sorted by
        descending rank disagreement (largest disagreement first).

    Raises:
        KeyError: If either detector is not in the bank.
    """
    s_a = bank.scores_for(det_a)
    s_b = bank.scores_for(det_b)

    rank_a = _to_ranks(s_a)
    rank_b = _to_ranks(s_b)

    rank_diff = np.abs(rank_a.astype(np.float64) - rank_b.astype(np.float64))
    k = min(top_k, len(rank_diff))
    return np.argsort(-rank_diff, kind="stable")[:k]


def score_correlation(
    bank: ScoreBank,
    method: str = "spearman",
) -> np.ndarray:
    """Pairwise correlation matrix across all detectors.

    Spearman (default) is rank-based and inherently scale-invariant — no
    normalization needed before calling this.

    Args:
        bank: A ``ScoreBank`` with at least two detectors.
        method: ``"spearman"`` (rank-based, default) or ``"pearson"``
            (raw scores — sensitive to scale differences between detectors).

    Returns:
        Symmetric correlation matrix, shape ``(n_detectors, n_detectors)``.
        Row/column order matches ``bank.detectors``.

    Raises:
        ValueError: If ``method`` is unknown or the bank has fewer than 2
            detectors.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    if len(bank.detectors) < 2:
        raise ValueError("score_correlation requires at least 2 detectors in the bank.")

    score_matrix = np.stack(
        [bank.scores_for(det) for det in bank.detectors], axis=0
    )  # (n_detectors, n_samples)

    if method == "spearman":
        score_matrix = np.apply_along_axis(_to_ranks, 1, score_matrix).astype(np.float64)

    return np.corrcoef(score_matrix)


def normalize_scores(
    bank: ScoreBank,
    method: str = "standardize",
) -> ScoreBank:
    """Return a new ``ScoreBank`` with normalized scores.

    This is a utility for users who want scaled scores for custom downstream
    analysis.  It is **not required** by any function in ``compare``,
    ``performance``, or ``plots`` — those handle scale differences internally
    where needed.

    The original ``bank`` is never modified.

    Args:
        bank: Source ``ScoreBank``.
        method: ``"standardize"`` (zero mean, unit variance, default) or
            ``"minmax"`` (scale to ``[0, 1]``).

    Returns:
        A new ``ScoreBank`` with the same structure but normalized score arrays.
        OOD labels, class labels, and sample metrics are copied as-is.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in ("standardize", "minmax"):
        raise ValueError(f"method must be 'standardize' or 'minmax', got {method!r}")

    normalized: dict = {}
    for det in bank.detectors:
        s = bank.scores_for(det).copy()
        if method == "standardize":
            std = s.std()
            s = (s - s.mean()) / (std if std > 0 else 1.0)
        else:
            lo, hi = s.min(), s.max()
            s = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        normalized[det] = s

    return ScoreBank(
        scores=normalized,
        ood_labels=bank.ood_labels,
        class_labels=bank.class_labels,
        sample_metrics={name: bank.metric_for(name) for name in bank.metric_names},
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _to_ranks(arr: np.ndarray) -> np.ndarray:
    """Convert a 1D float array to integer ranks (0-based, ascending)."""
    order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    return ranks
