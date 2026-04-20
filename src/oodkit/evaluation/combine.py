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


_CHIP_META_KEYS = ("chip_to_image", "boxes")


def _merge_metadata(parts: Sequence[EmbeddingResult]) -> Dict:
    """Merge per-block ``EmbeddingResult.metadata`` dicts.

    - List-valued entries (e.g. ``image_paths``) are concatenated in block order.
    - ``chip_to_image`` arrays are concatenated with per-block offsets so
      downstream pooling can treat the combined result as one contiguous chip
      stream with unique image indices. Offset per block equals the running
      count of unique images in earlier blocks.
    - ``boxes`` arrays are vertically concatenated.
    - For ``chip_to_image`` / ``boxes`` we enforce all-or-none: if any block has
      the key, every block must.
    - Other scalar / array entries keep the first block's value (unchanged
      legacy behavior for non-chip metadata).
    """
    merged: Dict = {}
    for r in parts:
        for key, val in r.metadata.items():
            if key in _CHIP_META_KEYS:
                continue
            if isinstance(val, list):
                merged.setdefault(key, []).extend(val)
            elif key not in merged:
                merged[key] = val

    chip_keys_present = [
        [k in r.metadata for r in parts] for k in _CHIP_META_KEYS
    ]
    for k, presence in zip(_CHIP_META_KEYS, chip_keys_present):
        if any(presence) and not all(presence):
            raise ValueError(
                f"All EmbeddingResult objects must either provide metadata "
                f"{k!r} or omit it; got a mix."
            )

    if all(k in parts[0].metadata for k in _CHIP_META_KEYS):
        chip_to_image_parts: List[np.ndarray] = []
        boxes_parts: List[np.ndarray] = []
        offset = 0
        for r in parts:
            c2i = np.asarray(r.metadata["chip_to_image"], dtype=np.int64).ravel()
            b = np.asarray(r.metadata["boxes"], dtype=np.float64)
            if b.ndim != 2 or b.shape[1] != 4:
                raise ValueError(
                    f"metadata['boxes'] must have shape (N, 4), got {b.shape}"
                )
            if c2i.shape[0] != b.shape[0]:
                raise ValueError(
                    "metadata['chip_to_image'] and metadata['boxes'] length mismatch: "
                    f"{c2i.shape[0]} vs {b.shape[0]}"
                )
            chip_to_image_parts.append(c2i + offset)
            boxes_parts.append(b)
            if c2i.size > 0:
                offset += int(c2i.max()) + 1
        merged["chip_to_image"] = np.concatenate(chip_to_image_parts, axis=0)
        merged["boxes"] = np.concatenate(boxes_parts, axis=0)

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
