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

from typing import Any, List, Optional, Tuple

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
    kind: str = "hist",
    standardize: bool = False,
    n_points: int = 256,
    bandwidth: str = "scott",
    axes: Optional[np.ndarray] = None,
) -> Figure:
    """ID vs OOD score distributions, one subplot per detector.

    When ``ood_labels`` are present, each subplot overlays two distributions
    (ID in blue, OOD in red). Without labels, a single distribution of all
    scores is shown.

    Args:
        bank: A ``ScoreBank`` with at least one detector.
        n_bins: Number of histogram bins per detector (used when
            ``kind="hist"``). Default ``40``.
        kind: ``"hist"`` (default) for filled histograms, ``"kde"`` for a
            Gaussian KDE via :class:`scipy.stats.gaussian_kde`.
        standardize: When ``True``, z-score each detector's scores against its
            own ID pool (``ood_labels == 0``) before plotting. The ID
            distribution is then centered at 0 with unit variance, so x-axes
            carry the same "ID std" meaning across detectors — useful for
            cross-detector visual comparison. A faint vertical line at
            ``x=0`` is drawn as the ID reference. Requires ``ood_labels``.
        n_points: KDE evaluation grid size (used when ``kind="kde"``).
        bandwidth: Forwarded to ``gaussian_kde(bw_method=...)``.
        axes: Optional pre-created axes array (one per detector).

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the bank has no detectors, ``kind`` is unknown, or
            ``standardize=True`` without ``ood_labels``.
    """
    if not bank.detectors:
        raise ValueError("ScoreBank has no detectors.")
    if kind not in ("hist", "kde"):
        raise ValueError(f"kind must be 'hist' or 'kde', got {kind!r}")
    if standardize and not bank.has_ood_labels:
        raise ValueError(
            "standardize=True requires ood_labels so the ID pool can be "
            "identified (ood_labels == 0). Pass ood_labels at construction."
        )

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
    i = -1
    for i, det in enumerate(bank.detectors):
        ax = flat_axes[i]
        s = bank.scores_for(det)

        if standardize:
            assert bank.ood_labels is not None
            id_mask = bank.ood_labels == 0
            if id_mask.sum() < 2:
                raise ValueError(
                    f"standardize=True: detector {det!r} has <2 ID samples; "
                    "cannot z-score."
                )
            id_mean = float(np.mean(s[id_mask]))
            id_std = float(np.std(s[id_mask]))
            if id_std <= 0:
                raise ValueError(
                    f"standardize=True: detector {det!r} ID score std is "
                    "non-positive; cannot z-score."
                )
            s = (s - id_mean) / id_std

        if bank.has_ood_labels:
            assert bank.ood_labels is not None
            id_scores = s[bank.ood_labels == 0]
            ood_scores = s[bank.ood_labels == 1]
            if kind == "hist":
                lo, hi = float(s.min()), float(s.max())
                bins = np.linspace(lo, hi, n_bins + 1)
                ax.hist(id_scores, bins=bins, alpha=0.55, color="steelblue",
                        label="ID", density=True)
                ax.hist(ood_scores, bins=bins, alpha=0.55, color="tomato",
                        label="OOD", density=True)
            else:
                _plot_kde(ax, id_scores, ood_scores, n_points=n_points,
                          bandwidth=bandwidth)
            ax.legend(fontsize=8)
        else:
            if kind == "hist":
                ax.hist(s, bins=n_bins, density=True, color="steelblue", alpha=0.75)
            else:
                _plot_kde(ax, s, None, n_points=n_points, bandwidth=bandwidth)

        if standardize:
            ax.axvline(0.0, color="0.5", linestyle=":", linewidth=0.8)

        ax.set_title(det, fontsize=10)
        ax.set_xlabel("OOD Score (ID z-score)" if standardize else "OOD Score")
        ax.set_ylabel("Density")

    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)

    title = "Score Distributions"
    if standardize:
        title += " (ID-standardized)"
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def _plot_kde(
    ax: plt.Axes,
    id_scores: np.ndarray,
    ood_scores: Optional[np.ndarray],
    *,
    n_points: int,
    bandwidth: str,
) -> None:
    """Overlay Gaussian KDE(s) for ID (and optional OOD) on ``ax``."""
    from scipy.stats import gaussian_kde

    if ood_scores is not None and ood_scores.size > 0:
        lo = float(min(id_scores.min(), ood_scores.min()))
        hi = float(max(id_scores.max(), ood_scores.max()))
    else:
        lo, hi = float(id_scores.min()), float(id_scores.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    xs = np.linspace(lo - pad, hi + pad, n_points)

    if id_scores.size >= 2:
        kde_id = gaussian_kde(id_scores, bw_method=bandwidth)
        ys = kde_id(xs)
        ax.plot(xs, ys, color="steelblue", linewidth=1.6, label="ID")
        ax.fill_between(xs, ys, color="steelblue", alpha=0.20)
    if ood_scores is not None and ood_scores.size >= 2:
        kde_ood = gaussian_kde(ood_scores, bw_method=bandwidth)
        ys = kde_ood(xs)
        ax.plot(xs, ys, color="tomato", linewidth=1.6, linestyle="--", label="OOD")
        ax.fill_between(xs, ys, color="tomato", alpha=0.20)


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
    images: Optional[Any] = None,
    top_k: int = 16,
    direction: str = "ood",
    rank_range: Optional[Tuple[int, int]] = None,
    class_name: Optional[Any] = None,
    group: Optional[str] = None,
    truth: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Grid showing samples ranked by OOD score with optional filters.

    Supports two modes:
    - **Image mode:** pass a sequence indexable as ``images[i]`` returning a
      PIL image or file path, aligned with the samples in the bank. The
      sequence can be any list-like, including custom on-the-fly loaders
      (e.g. chip croppers for object detection) — only ``__getitem__`` is
      required.
    - **Text mode:** no ``images`` — shows a text table of index, score, and
      any available metadata (class, group, ID/OOD).

    Filters narrow the candidate pool *before* ranking, so "top 8" reflects
    the filtered subset rather than the full bank.

    Args:
        bank: A ``ScoreBank`` with at least one detector.
        detector: Which detector's scores to rank by.
        images: Optional sample-aligned sequence (PIL, path, or custom
            loader). If ``None``, text mode is used.
        top_k: Number of samples to display when ``rank_range`` is not set
            (default ``16``).
        direction: ``"ood"`` (highest scores, default) or ``"id"`` (lowest).
        rank_range: Optional ``(start, end)`` half-open slice into the
            ranking — e.g. ``(16, 24)`` for ranks 17–24. Overrides ``top_k``.
        class_name: Restrict to one class. Accepts an integer label, or a
            string name when ``bank.class_names`` is set.
        group: Restrict to one group tag (requires ``bank.groups``).
        truth: ``"id"`` / ``"ood"`` / ``None`` — restrict by ground-truth
            label (requires ``bank.ood_labels``).
        figsize: Optional ``(width, height)`` for the figure.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        KeyError: If ``detector`` is not in the bank, or ``class_name`` is a
            string and not in ``class_names``.
        ValueError: If ``direction`` is not ``"ood"``/``"id"``, ``truth`` is
            invalid, a filter references unavailable metadata, or the
            filtered pool is empty.
    """
    if direction not in ("ood", "id"):
        raise ValueError(f"direction must be 'ood' or 'id', got {direction!r}")
    if truth is not None and truth not in ("id", "ood"):
        raise ValueError(f"truth must be 'id', 'ood', or None, got {truth!r}")

    filtered_bank, filter_map = _apply_rank_filters(
        bank, class_name=class_name, group=group, truth=truth
    )

    if rank_range is None:
        sub_indices = rank_samples(
            filtered_bank, detector, top_k=top_k, direction=direction
        )
        start = 0
    else:
        sub_indices = rank_samples(
            filtered_bank, detector, direction=direction, rank_range=rank_range
        )
        start = max(0, int(rank_range[0]))

    if sub_indices.size == 0:
        raise ValueError(
            "No samples remain after filtering; relax filters or widen rank_range."
        )

    indices = filter_map[sub_indices]
    scores = bank.scores_for(detector)

    if images is not None:
        return _rank_grid_images(
            indices, scores, bank, images, detector, direction, figsize,
            class_name=class_name, group=group, truth=truth, rank_start=start,
        )
    return _rank_grid_text(
        indices, scores, bank, detector, direction, figsize,
        class_name=class_name, group=group, truth=truth, rank_start=start,
    )


def _apply_rank_filters(
    bank: ScoreBank,
    *,
    class_name: Optional[Any],
    group: Optional[str],
    truth: Optional[str],
) -> Tuple[ScoreBank, np.ndarray]:
    """Apply filters and return ``(filtered_bank, filter_map)``.

    ``filter_map[j]`` is the original sample index corresponding to the
    ``j``-th sample in ``filtered_bank``.
    """
    n = bank.n_samples
    mask = np.ones(n, dtype=bool)

    if class_name is not None:
        if not bank.has_class_labels:
            raise ValueError(
                "class_name filter requires class_labels on the ScoreBank."
            )
        if isinstance(class_name, str):
            if not bank.has_class_names:
                raise ValueError(
                    "class_name filter with a string requires class_names on "
                    "the ScoreBank."
                )
            names = bank.class_names
            assert names is not None
            if class_name not in names:
                raise KeyError(
                    f"class name {class_name!r} not in class_names; "
                    f"available: {names}"
                )
            cls_int = names.index(class_name)
        else:
            cls_int = int(class_name)
        assert bank.class_labels is not None
        mask &= bank.class_labels == cls_int

    if group is not None:
        if not bank.has_groups:
            raise ValueError(
                "group filter requires groups on the ScoreBank."
            )
        assert bank.groups is not None
        mask &= bank.groups == group

    if truth is not None:
        if not bank.has_ood_labels:
            raise ValueError(
                "truth filter requires ood_labels on the ScoreBank."
            )
        assert bank.ood_labels is not None
        want = 0 if truth == "id" else 1
        mask &= bank.ood_labels == want

    filter_map = np.nonzero(mask)[0].astype(np.intp)
    if filter_map.size == 0:
        raise ValueError(
            f"No samples match filters "
            f"(class_name={class_name!r}, group={group!r}, truth={truth!r})."
        )

    filtered = bank.subset(filter_map)
    return filtered, filter_map


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
    images: Any,
    detector: str,
    direction: str,
    figsize: Optional[Tuple[float, float]],
    *,
    class_name: Optional[Any] = None,
    group: Optional[str] = None,
    truth: Optional[str] = None,
    rank_start: int = 0,
) -> Figure:
    """Render a thumbnail grid with score / class / group / ID-OOD annotations."""
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
        ax.set_title(
            _format_sample_title(
                bank, int(sample_idx), float(scores[sample_idx]),
                rank_start + pos + 1,
            ),
            fontsize=8,
        )

    for j in range(len(indices), nrows * ncols):
        axes_arr[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        _format_rank_suptitle(
            detector, direction, k, rank_start,
            class_name=class_name, group=group, truth=truth,
        ),
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _format_sample_title(
    bank: ScoreBank, sample_idx: int, score: float, rank: int,
) -> str:
    """Build ``'class | group | ID/OOD\\nscore=... (rank N)'`` skipping absent fields."""
    parts: List[str] = []
    if bank.has_class_labels:
        assert bank.class_labels is not None
        cls_int = int(bank.class_labels[sample_idx])
        if bank.has_class_names:
            names = bank.class_names
            assert names is not None
            if 0 <= cls_int < len(names):
                parts.append(names[cls_int])
            else:
                parts.append(str(cls_int))
        else:
            parts.append(f"cls={cls_int}")
    if bank.has_groups:
        assert bank.groups is not None
        parts.append(str(bank.groups[sample_idx]))
    if bank.has_ood_labels:
        assert bank.ood_labels is not None
        parts.append("OOD" if int(bank.ood_labels[sample_idx]) == 1 else "ID")
    head = " | ".join(parts) if parts else ""
    tail = f"score={score:.3f} (rank {rank})"
    return f"{head}\n{tail}" if head else tail


def _format_rank_suptitle(
    detector: str, direction: str, k: int, rank_start: int,
    *,
    class_name: Optional[Any],
    group: Optional[str],
    truth: Optional[str],
) -> str:
    """Build the figure-level title for rank_grid."""
    sort_word = "Most OOD" if direction == "ood" else "Most ID-like"
    end = rank_start + k
    rank_str = (
        f"ranks {rank_start + 1}-{end}"
        if rank_start > 0 else f"top {k} {sort_word}"
    )
    filt = []
    if class_name is not None:
        filt.append(f"class={class_name}")
    if group is not None:
        filt.append(f"group={group}")
    if truth is not None:
        filt.append(f"truth={truth}")
    filt_str = f"  [{', '.join(filt)}]" if filt else ""
    return f"{detector}: {rank_str}{filt_str}"


def _rank_grid_text(
    indices: np.ndarray,
    scores: np.ndarray,
    bank: ScoreBank,
    detector: str,
    direction: str,
    figsize: Optional[Tuple[float, float]],
    *,
    class_name: Optional[Any] = None,
    group: Optional[str] = None,
    truth: Optional[str] = None,
    rank_start: int = 0,
) -> Figure:
    """Render a text table of sample index, score, and class."""
    k = len(indices)
    fs = figsize or (5, 0.4 * k + 1.2)
    fig, ax = plt.subplots(figsize=fs)
    ax.axis("off")

    headers = ["Rank", "Sample", "Score"]
    if bank.has_class_labels:
        headers.append("Class")
    if bank.has_groups:
        headers.append("Group")
    if bank.has_ood_labels:
        headers.append("Truth")

    names = bank.class_names
    rows = []
    for pos, sample_idx in enumerate(indices):
        rank_no = rank_start + pos + 1
        row = [str(rank_no), str(int(sample_idx)), f"{scores[sample_idx]:.4f}"]
        if bank.has_class_labels:
            assert bank.class_labels is not None
            ci = int(bank.class_labels[sample_idx])
            if names is not None and 0 <= ci < len(names):
                row.append(names[ci])
            else:
                row.append(str(ci))
        if bank.has_groups:
            assert bank.groups is not None
            row.append(str(bank.groups[sample_idx]))
        if bank.has_ood_labels:
            assert bank.ood_labels is not None
            row.append("OOD" if int(bank.ood_labels[sample_idx]) == 1 else "ID")
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

    ax.set_title(
        _format_rank_suptitle(
            detector, direction, k, rank_start,
            class_name=class_name, group=group, truth=truth,
        ),
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    return fig
