"""Helpers for fitting OOD detectors on trusted sample subsets.

These utilities keep detector APIs generic: callers decide which samples are
trusted ID examples, pass a mask, and the helper fits an existing detector on
that slice. A common dataset-auditing mask is "model predicted the label
correctly, optionally above a confidence threshold."
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from oodkit.data.features import Features
from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy


def _features_length(features: Features) -> int:
    lengths = []
    if features.logits is not None:
        logits = to_numpy(features.logits)
        if logits.ndim == 0:
            raise ValueError("features.logits must have at least one dimension")
        lengths.append(int(logits.shape[0]))
    if features.embeddings is not None:
        embeddings = to_numpy(features.embeddings)
        if embeddings.ndim == 0:
            raise ValueError("features.embeddings must have at least one dimension")
        lengths.append(int(embeddings.shape[0]))
    if len(set(lengths)) != 1:
        raise ValueError(f"Features fields disagree on sample count: {lengths}")
    return lengths[0]


def _as_bool_mask(mask: ArrayLike, n_samples: int) -> np.ndarray:
    mask_np = np.asarray(to_numpy(mask)).reshape(-1)
    if mask_np.shape[0] != n_samples:
        raise ValueError(
            f"mask length must match features ({n_samples}), got {mask_np.shape[0]}"
        )
    if mask_np.dtype == np.bool_:
        return mask_np.astype(bool, copy=False)
    unique = np.unique(mask_np)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("mask must be boolean or contain only 0/1 values")
    return mask_np.astype(bool)


def subset_features(features: Features, mask: ArrayLike) -> Features:
    """Return a ``Features`` object restricted to samples where ``mask`` is true.

    Args:
        features: Logits and/or embeddings aligned along the first axis.
        mask: Boolean or 0/1 vector with length ``n_samples``.

    Returns:
        A new ``Features`` instance containing sliced arrays.

    Raises:
        ValueError: If feature fields have inconsistent lengths or ``mask`` is
            invalid.
    """
    n_samples = _features_length(features)
    mask_bool = _as_bool_mask(mask, n_samples)
    logits = None
    embeddings = None
    if features.logits is not None:
        logits = to_numpy(features.logits)[mask_bool]
    if features.embeddings is not None:
        embeddings = to_numpy(features.embeddings)[mask_bool]
    return Features(logits=logits, embeddings=embeddings)


def fit_detector_on_mask(
    detector: BaseDetector,
    features: Features,
    mask: ArrayLike,
    *,
    y: Optional[ArrayLike] = None,
    min_samples: int = 1,
    **fit_kwargs: object,
) -> BaseDetector:
    """Fit an OOD detector using only samples selected by ``mask``.

    The detector object is mutated and returned, matching the existing detector
    ``fit`` contract. Scores can still be computed on the full ``features``
    afterwards with ``detector.score(features)``.

    Args:
        detector: Any already-constructed OODKit detector.
        features: Full feature set.
        mask: Boolean or 0/1 vector; true samples form the fit pool.
        y: Optional labels aligned with ``features``. When supplied, labels are
            sliced by ``mask`` and passed to ``detector.fit(..., y=y_fit)``.
        min_samples: Minimum number of selected samples required before fitting.
        **fit_kwargs: Additional keyword arguments forwarded to ``detector.fit``.

    Returns:
        The fitted detector.

    Raises:
        ValueError: If the mask is invalid, selects too few samples, or ``y`` is
            misaligned.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    n_samples = _features_length(features)
    mask_bool = _as_bool_mask(mask, n_samples)
    n_selected = int(mask_bool.sum())
    if n_selected < min_samples:
        raise ValueError(
            f"mask selected {n_selected} samples, but min_samples={min_samples}"
        )

    fit_features = subset_features(features, mask_bool)
    if y is None:
        detector.fit(fit_features, **fit_kwargs)
        return detector

    y_np = np.asarray(to_numpy(y)).reshape(-1)
    if y_np.shape[0] != n_samples:
        raise ValueError(f"y length must match features ({n_samples}), got {y_np.shape[0]}")
    detector.fit(fit_features, y=y_np[mask_bool], **fit_kwargs)
    return detector


def correct_prediction_mask(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    min_confidence: Optional[float] = None,
) -> np.ndarray:
    """Mask samples whose argmax prediction matches the label.

    Args:
        logits: Class logits, shape ``(n_samples, n_classes)``.
        labels: Integer labels, shape ``(n_samples,)``.
        min_confidence: Optional softmax confidence floor for the predicted
            class. When set, a sample must be both correct and at least this
            confident to be selected.

    Returns:
        Boolean mask, shape ``(n_samples,)``.

    Raises:
        ValueError: If shapes are invalid or ``min_confidence`` is outside
            ``[0, 1]``.
    """
    logits_np = np.asarray(to_numpy(logits), dtype=np.float64)
    labels_np = np.asarray(to_numpy(labels)).reshape(-1)
    if logits_np.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {logits_np.shape}")
    if labels_np.shape[0] != logits_np.shape[0]:
        raise ValueError(
            f"labels length must match logits ({logits_np.shape[0]}), got {labels_np.shape[0]}"
        )
    pred = np.argmax(logits_np, axis=1)
    mask = pred == labels_np.astype(pred.dtype, copy=False)

    if min_confidence is not None:
        conf_floor = float(min_confidence)
        if conf_floor < 0.0 or conf_floor > 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        confidence = probs[np.arange(logits_np.shape[0]), pred]
        mask &= confidence >= conf_floor

    return mask.astype(bool, copy=False)


__all__ = [
    "correct_prediction_mask",
    "fit_detector_on_mask",
    "subset_features",
]
