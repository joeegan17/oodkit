"""
Pool chip-level OOD scores into image-level scores.

Used with :class:`oodkit.data.chip_dataset.ChipDataset` / the ``chip_to_image``
metadata written by ``Embedder.extract``. Detectors stay OD-agnostic — pooling
lives here as a pure post-processing step over score arrays.

Images with **zero chips** in the stream (e.g. after subsampling or because the
caller pre-filtered tiny boxes) yield ``NaN`` in the output. Downstream
``ScoreBank`` consumers must drop or impute NaN rows before adding the vector.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_SUPPORTED_METHODS = ("mean", "max", "topk_mean")


def pool_image_scores(
    chip_scores: np.ndarray,
    chip_to_image: np.ndarray,
    method: str = "mean",
    k: int = 3,
    n_images: Optional[int] = None,
) -> np.ndarray:
    """Aggregate chip-level OOD scores to one score per image.

    Args:
        chip_scores: ``(N_chips,)`` float array of per-chip OOD scores (higher
            = more OOD, matching detector conventions).
        chip_to_image: ``(N_chips,)`` int array with values in
            ``[0, n_images)``; maps each chip to its parent image.
        method: ``"mean"``, ``"max"``, or ``"topk_mean"``. ``"topk_mean"`` takes
            the mean of the top ``k`` chip scores per image (or all of them if
            the image has fewer than ``k`` chips).
        k: Number of top chips per image for ``"topk_mean"``. Ignored otherwise.
            Must be ``>= 1``.
        n_images: Override for the length of the output. Defaults to
            ``chip_to_image.max() + 1``. Pass this when some images have no
            chips in ``chip_to_image`` but should still appear as ``NaN`` rows
            (e.g. after filtering small boxes or after ``frac < 1.0``
            subsampling).

    Returns:
        ``(n_images,)`` float64 array. Images with no chips in
        ``chip_to_image`` are set to ``NaN``.

    Raises:
        ValueError: On unknown ``method``, shape mismatch, ``k < 1``,
            ``n_images`` too small, or negative image indices.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {_SUPPORTED_METHODS}, got {method!r}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    scores = np.asarray(chip_scores, dtype=np.float64).ravel()
    c2i = np.asarray(chip_to_image, dtype=np.int64).ravel()
    if scores.shape[0] != c2i.shape[0]:
        raise ValueError(
            "chip_scores and chip_to_image length mismatch: "
            f"{scores.shape[0]} vs {c2i.shape[0]}"
        )
    if c2i.size and c2i.min() < 0:
        raise ValueError("chip_to_image must be non-negative")

    if n_images is None:
        n_out = int(c2i.max()) + 1 if c2i.size else 0
    else:
        if n_images < 0:
            raise ValueError("n_images must be non-negative")
        if c2i.size and int(c2i.max()) >= n_images:
            raise ValueError(
                "n_images is smaller than max(chip_to_image) + 1: "
                f"{n_images} vs {int(c2i.max()) + 1}"
            )
        n_out = int(n_images)

    out = np.full(n_out, np.nan, dtype=np.float64)
    if n_out == 0 or scores.size == 0:
        return out

    order = np.argsort(c2i, kind="stable")
    sorted_c2i = c2i[order]
    sorted_scores = scores[order]
    unique_ids, starts, counts = np.unique(
        sorted_c2i, return_index=True, return_counts=True
    )

    if method == "mean":
        sums = np.add.reduceat(sorted_scores, starts)
        out[unique_ids] = sums / counts
        return out

    if method == "max":
        out[unique_ids] = np.maximum.reduceat(sorted_scores, starts)
        return out

    for image_id, start, count in zip(unique_ids, starts, counts):
        group = sorted_scores[start : start + count]
        if count <= k:
            out[image_id] = float(group.mean())
        else:
            top = np.partition(group, -k)[-k:]
            out[image_id] = float(top.mean())
    return out


__all__ = ["pool_image_scores"]
