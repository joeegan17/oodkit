"""
Combine multiple :class:`~oodkit.embeddings.result.EmbeddingResult` blocks for evaluation.

Typical use: concatenate in-distribution and OOD extraction outputs, build
binary ``ood_labels`` for :class:`~oodkit.evaluation.score_bank.ScoreBank`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from oodkit.embeddings.result import EmbeddingResult
from oodkit.utils.array import to_numpy


def ood_labels_from_counts(n_id: int, n_ood: int) -> np.ndarray:
    """Build ``(n_id + n_ood,)`` array: zeros then ones.

    Args:
        n_id: Number of in-distribution samples (labeled ``0``).
        n_ood: Number of OOD samples (labeled ``1``).

    Returns:
        Array of dtype ``int64``, values ``{0, 1}``.
    """
    if n_id < 0 or n_ood < 0:
        raise ValueError("n_id and n_ood must be non-negative")
    return np.concatenate(
        [np.zeros(n_id, dtype=np.int64), np.ones(n_ood, dtype=np.int64)],
    )


def ood_labels_from_blocks(lengths: Sequence[int], flags: Sequence[int]) -> np.ndarray:
    """Build ``ood_labels`` from consecutive blocks.

    Each block ``i`` has length ``lengths[i]`` and constant label ``flags[i]``
    (``0`` = ID, ``1`` = OOD).

    Args:
        lengths: Positive sample counts per block.
        flags: ``0`` or ``1`` per block; same length as ``lengths``.

    Returns:
        Concatenated per-sample labels, shape ``(sum(lengths),)``.
    """
    if len(lengths) != len(flags):
        raise ValueError("lengths and flags must have the same length")
    parts: List[np.ndarray] = []
    for n, f in zip(lengths, flags):
        if n < 0:
            raise ValueError("lengths must be non-negative")
        if f not in (0, 1):
            raise ValueError(f"flags must be 0 or 1, got {f!r}")
        parts.append(np.full(n, f, dtype=np.int64))
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts, axis=0)


def _merge_metadata(parts: Sequence[EmbeddingResult]) -> Dict:
    merged: Dict = {}
    for r in parts:
        for key, val in r.metadata.items():
            if isinstance(val, list):
                merged.setdefault(key, []).extend(val)
            elif key not in merged:
                merged[key] = val
    return merged


def concatenate_embedding_results(
    results: Sequence[EmbeddingResult],
    ood_flags: Sequence[int],
) -> Tuple[EmbeddingResult, np.ndarray]:
    """Stack several ``EmbeddingResult`` objects along the sample axis.

    Use ``ood_flags`` with one entry per **result** (not per sample): all samples
    from ``results[i]`` receive label ``ood_flags[i]`` (``0`` = ID, ``1`` = OOD).

    Args:
        results: Non-empty sequence of extraction outputs with compatible shapes.
        ood_flags: Same length as ``results``; each value ``0`` or ``1``.

    Returns:
        ``(combined, ood_labels)`` where ``ood_labels`` has shape
        ``(total_samples,)`` and matches the row order of ``combined.embeddings``.

    Raises:
        ValueError: On length mismatch, inconsistent logits/labels presence, or
            embedding dimension mismatch.
    """
    if not results:
        raise ValueError("results must be non-empty")
    if len(results) != len(ood_flags):
        raise ValueError("results and ood_flags must have the same length")
    for f in ood_flags:
        if f not in (0, 1):
            raise ValueError(f"ood_flags must be 0 or 1, got {f!r}")

    ood_parts: List[np.ndarray] = []
    emb_parts: List[np.ndarray] = []
    logit_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    has_logits = results[0].logits is not None
    has_labels = results[0].labels is not None
    emb_dim: Union[int, None] = None

    for r, flag in zip(results, ood_flags):
        e = to_numpy(r.embeddings).astype(np.float32, copy=False)
        if e.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {e.shape}")
        if emb_dim is None:
            emb_dim = e.shape[1]
        elif e.shape[1] != emb_dim:
            raise ValueError("All embeddings must share the same feature dimension")

        n = e.shape[0]
        ood_parts.append(np.full(n, flag, dtype=np.int64))
        emb_parts.append(e)

        if (r.logits is not None) != has_logits:
            raise ValueError("All EmbeddingResult objects must either have logits or omit logits")
        if has_logits:
            assert r.logits is not None
            lg = to_numpy(r.logits).astype(np.float32, copy=False)
            if lg.shape[0] != n:
                raise ValueError("logits and embeddings length mismatch")
            logit_parts.append(lg)

        if (r.labels is not None) != has_labels:
            raise ValueError("All EmbeddingResult objects must either have labels or omit labels")
        if has_labels:
            assert r.labels is not None
            lb = to_numpy(r.labels).astype(np.int64, copy=False).ravel()
            if lb.shape[0] != n:
                raise ValueError("labels and embeddings length mismatch")
            label_parts.append(lb)

    embeddings = np.concatenate(emb_parts, axis=0)
    logits = np.concatenate(logit_parts, axis=0) if has_logits else None
    labels = np.concatenate(label_parts, axis=0) if has_labels else None
    ood_labels = np.concatenate(ood_parts, axis=0)

    combined = EmbeddingResult(
        embeddings=embeddings,
        logits=logits,
        labels=labels,
        metadata=_merge_metadata(results),
    )
    return combined, ood_labels
