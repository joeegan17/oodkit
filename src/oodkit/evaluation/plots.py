"""
Matplotlib-based visualization for OOD evaluation.

All functions accept a ``ScoreBank`` and an optional ``ax`` argument:
- If ``ax`` is provided, the plot is drawn onto that axes (for embedding in
  user subplot layouts).
- If ``ax`` is ``None``, a new figure and axes are created and returned.

Functions that produce multi-panel figures (e.g. ``score_distributions``)
accept an optional ``fig`` / ``axes`` pair instead.

Cross-detector score comparisons (``correlation_heatmap``) use Spearman
ranks internally — scale differences between detectors are handled
automatically.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from oodkit.evaluation.compare import rank_samples, score_correlation
from oodkit.evaluation.metrics import aupr, auroc, pr_curve, roc_curve
from oodkit.evaluation.performance import PerformanceCurve, score_vs_metric
from oodkit.evaluation.score_bank import ScoreBank


# ------------------------------------------------------------------
# ROC / PR curves
# ------------------------------------------------------------------

def roc_curves(
    bank: ScoreBank,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Overlay ROC curves for all detectors.

    Each curve's legend entry includes the AUROC value.

    Args:
        bank: A ``ScoreBank`` with ``ood_labels`` and at least one detector.
        ax: Optional axes to draw on.  If ``None``, a new figure is created.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the bank has no OOD labels or no detectors.
    """
    _require_ood_labels(bank, "roc_curves")
    fig, ax = _get_ax(ax)

    for det in bank.detectors:
        s = bank.scores_for(det)
        fpr, tpr = roc_curve(s, bank.ood_labels)
        auc = auroc(s, bank.ood_labels)
        ax.plot(fpr, tpr, label=f"{det} (AUROC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def pr_curves(
    bank: ScoreBank,
    positive: str = "ood",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Overlay Precision-Recall curves for all detectors.

    Each curve's legend entry includes the AUPR value.

    Args:
        bank: A ``ScoreBank`` with ``ood_labels`` and at least one detector.
        positive: Which class is positive — ``"ood"`` (default) or ``"id"``.
        ax: Optional axes to draw on.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the bank has no OOD labels or no detectors.
    """
    _require_ood_labels(bank, "pr_curves")
    fig, ax = _get_ax(ax)

    for det in bank.detectors:
        s = bank.scores_for(det)
        precision, recall = pr_curve(s, bank.ood_labels, positive=positive)
        ap = aupr(s, bank.ood_labels, positive=positive)
        ax.plot(recall, precision, label=f"{det} (AUPR={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves (positive='{positive}')")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Score distributions
# ------------------------------------------------------------------

def score_distributions(
    bank: ScoreBank,
    n_bins: int = 40,
    axes: Optional[np.ndarray] = None,
) -> Figure:
    """ID vs OOD score histograms, one subplot per detector.

    When ``ood_labels`` are present, each subplot overlays two filled
    histograms (ID in blue, OOD in red).  Without labels, a single
    histogram of all scores is shown.

    Args:
        bank: A ``ScoreBank`` with at least one detector.
        n_bins: Number of histogram bins per detector (default ``40``).
        axes: Optional pre-created axes array (one per detector).

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the bank has no detectors.
    """
    if not bank.detectors:
        raise ValueError("ScoreBank has no detectors.")

    n_dets = len(bank.detectors)
    if axes is None:
        ncols = min(n_dets, 3)
        nrows = (n_dets + ncols - 1) // ncols
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    else:
        axes_arr = np.asarray(axes).reshape(-1)
        fig = axes_arr.flat[0].get_figure()
        axes_arr = axes_arr.reshape(1, -1)

    flat_axes = axes_arr.ravel()

    for i, det in enumerate(bank.detectors):
        ax = flat_axes[i]
        s = bank.scores_for(det)

        if bank.has_ood_labels:
            id_scores = s[bank.ood_labels == 0]
            ood_scores = s[bank.ood_labels == 1]
            lo, hi = s.min(), s.max()
            bins = np.linspace(lo, hi, n_bins + 1)
            ax.hist(id_scores, bins=bins, alpha=0.55, color="steelblue", label="ID", density=True)
            ax.hist(ood_scores, bins=bins, alpha=0.55, color="tomato", label="OOD", density=True)
            ax.legend(fontsize=8)
        else:
            ax.hist(s, bins=n_bins, density=True, color="steelblue", alpha=0.75)

        ax.set_title(det, fontsize=10)
        ax.set_xlabel("OOD Score")
        ax.set_ylabel("Density")

    # Hide any unused axes
    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle("Score Distributions", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Performance curve
# ------------------------------------------------------------------

def performance_curve(
    bank: ScoreBank,
    metric_name: str,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """OOD score percentile bin vs mean metric, one line per detector.

    Useful for checking whether higher OOD scores correlate with model
    failure: a falling curve indicates the detector is meaningful.

    Args:
        bank: A ``ScoreBank`` with at least one detector and ``metric_name``
            in its sample metrics.
        metric_name: Name of the per-sample metric to plot.
        n_bins: Number of equal-frequency bins (default ``10``).
        ax: Optional axes to draw on.

    Returns:
        The matplotlib ``Figure``.
    """
    curves = score_vs_metric(bank, metric_name=metric_name, n_bins=n_bins)
    if isinstance(curves, PerformanceCurve):
        curves = {curves.detector: curves}

    fig, ax = _get_ax(ax)

    for det, curve in curves.items():
        ax.plot(curve.bin_centers, curve.mean_metric, marker="o", label=det)

    ax.set_xlabel("OOD Score (bin center)")
    ax.set_ylabel(f"Mean {metric_name}")
    ax.set_title(f"Performance vs OOD Score  [{metric_name}]")
    ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------------

def correlation_heatmap(
    bank: ScoreBank,
    method: str = "spearman",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Heatmap of pairwise detector score correlations.

    Spearman (default) is rank-based and scale-invariant — no normalization
    needed before calling this function.

    Args:
        bank: A ``ScoreBank`` with at least 2 detectors.
        method: ``"spearman"`` (default) or ``"pearson"``.
        ax: Optional axes to draw on.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If fewer than 2 detectors are in the bank.
    """
    corr = score_correlation(bank, method=method)
    detectors = bank.detectors

    fig, ax = _get_ax(ax)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(detectors)))
    ax.set_yticks(range(len(detectors)))
    ax.set_xticklabels(detectors, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(detectors, fontsize=8)

    for i in range(len(detectors)):
        for j in range(len(detectors)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title(f"Score Correlation ({method.capitalize()})")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Rank grid
# ------------------------------------------------------------------

def rank_grid(
    bank: ScoreBank,
    detector: str,
    images: Optional[List] = None,
    top_k: int = 16,
    direction: str = "ood",
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Grid showing the top-k samples by OOD score.

    Supports two modes:
    - **Image mode:** pass PIL images (or paths) via ``images`` — shows a
      thumbnail grid with score and class label annotations.
    - **Text mode:** no ``images`` — shows a text table of index, score, and
      class (if available).

    Args:
        bank: A ``ScoreBank`` with at least one detector.
        detector: Which detector's scores to rank by.
        images: Optional list of PIL ``Image`` objects or file-path strings,
            aligned with the samples in the bank.  If ``None``, text mode is
            used.
        top_k: Number of samples to display (default ``16``).
        direction: ``"ood"`` (highest scores, default) or ``"id"`` (lowest).
        figsize: Optional ``(width, height)`` for the figure.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        KeyError: If ``detector`` is not in the bank.
        ValueError: If ``direction`` is not ``"ood"`` or ``"id"``.
    """
    if direction not in ("ood", "id"):
        raise ValueError(f"direction must be 'ood' or 'id', got {direction!r}")

    indices = rank_samples(bank, detector, top_k=top_k, direction=direction)
    scores = bank.scores_for(detector)

    if images is not None:
        return _rank_grid_images(indices, scores, bank, images, detector, direction, figsize)
    return _rank_grid_text(indices, scores, bank, detector, direction, figsize)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_ax(ax: Optional[plt.Axes]) -> Tuple[Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.get_figure(), ax


def _require_ood_labels(bank: ScoreBank, fn_name: str) -> None:
    if not bank.has_ood_labels:
        raise ValueError(
            f"{fn_name}() requires OOD ground-truth labels. "
            "Pass ood_labels when constructing the ScoreBank."
        )
    if not bank.detectors:
        raise ValueError(f"{fn_name}() requires at least one detector in the ScoreBank.")


def _rank_grid_images(
    indices: np.ndarray,
    scores: np.ndarray,
    bank: ScoreBank,
    images: List,
    detector: str,
    direction: str,
    figsize: Optional[Tuple[float, float]],
) -> Figure:
    """Render a thumbnail grid with score / class annotations."""
    k = len(indices)
    ncols = min(k, 4)
    nrows = (k + ncols - 1) // ncols
    fs = figsize or (3 * ncols, 3.5 * nrows)
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=fs, squeeze=False)

    for pos, sample_idx in enumerate(indices):
        ax = axes_arr[pos // ncols][pos % ncols]
        img = images[sample_idx]
        if isinstance(img, str):
            from PIL import Image as PILImage
            img = PILImage.open(img)
        ax.imshow(img)
        ax.axis("off")
        label_str = ""
        if bank.has_class_labels:
            label_str = f"cls={bank.class_labels[sample_idx]} "
        ax.set_title(f"{label_str}score={scores[sample_idx]:.3f}", fontsize=7)

    for j in range(len(indices), nrows * ncols):
        axes_arr[j // ncols][j % ncols].set_visible(False)

    title_dir = "Most OOD" if direction == "ood" else "Most ID-like"
    fig.suptitle(f"Top-{k} {title_dir}  [{detector}]", fontsize=11)
    fig.tight_layout()
    return fig


def _rank_grid_text(
    indices: np.ndarray,
    scores: np.ndarray,
    bank: ScoreBank,
    detector: str,
    direction: str,
    figsize: Optional[Tuple[float, float]],
) -> Figure:
    """Render a text table of sample index, score, and class."""
    k = len(indices)
    fs = figsize or (5, 0.4 * k + 1.2)
    fig, ax = plt.subplots(figsize=fs)
    ax.axis("off")

    headers = ["Rank", "Sample", "Score"]
    if bank.has_class_labels:
        headers.append("Class")

    rows = []
    for rank, sample_idx in enumerate(indices):
        row = [str(rank + 1), str(int(sample_idx)), f"{scores[sample_idx]:.4f}"]
        if bank.has_class_labels:
            row.append(str(int(bank.class_labels[sample_idx])))
        rows.append(row)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    title_dir = "Most OOD" if direction == "ood" else "Most ID-like"
    ax.set_title(f"Top-{k} {title_dir}  [{detector}]", fontsize=11, pad=12)
    fig.tight_layout()
    return fig
