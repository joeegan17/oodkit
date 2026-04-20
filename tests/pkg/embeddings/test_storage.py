"""Tests for ``oodkit.embeddings.storage.load_embeddings``."""

import json
from pathlib import Path

import numpy as np
import pytest

from oodkit.embeddings.storage import load_embeddings


def _write_shard(
    root: Path,
    *,
    n_samples: int,
    embed_dim: int,
    has_logits: bool = False,
    has_labels: bool = False,
    has_image_paths: bool = False,
    has_chip_to_image: bool = False,
    has_boxes: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "embeddings.npy", np.arange(n_samples * embed_dim, dtype=np.float32).reshape(n_samples, embed_dim))
    if has_logits:
        np.save(root / "logits.npy", np.zeros((n_samples, 3), dtype=np.float32))
    if has_labels:
        np.save(root / "labels.npy", np.arange(n_samples, dtype=np.int64))
    if has_image_paths:
        (root / "image_paths.json").write_text(
            json.dumps([f"/img/{i}.png" for i in range(n_samples)])
        )
    if has_chip_to_image:
        np.save(root / "chip_to_image.npy", np.arange(n_samples, dtype=np.int64) // 2)
    if has_boxes:
        boxes = np.tile(np.array([[0, 0, 10, 10]], dtype=np.float64), (n_samples, 1))
        np.save(root / "boxes.npy", boxes)

    manifest = {
        "n_samples": n_samples,
        "embed_dim": embed_dim,
        "has_logits": has_logits,
        "has_labels": has_labels,
        "has_image_paths": has_image_paths,
        "has_chip_to_image": has_chip_to_image,
        "has_boxes": has_boxes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_load_embeddings_basic(tmp_path: Path):
    _write_shard(tmp_path, n_samples=4, embed_dim=3, has_labels=True)
    res = load_embeddings(tmp_path)
    assert res.embeddings.shape == (4, 3)
    assert res.labels is not None and res.labels.shape == (4,)
    assert "chip_to_image" not in res.metadata
    assert "boxes" not in res.metadata


def test_load_embeddings_with_chip_metadata(tmp_path: Path):
    _write_shard(
        tmp_path,
        n_samples=6,
        embed_dim=2,
        has_image_paths=True,
        has_chip_to_image=True,
        has_boxes=True,
    )
    res = load_embeddings(tmp_path)
    assert res.embeddings.shape == (6, 2)
    assert len(res.metadata["image_paths"]) == 6
    np.testing.assert_array_equal(
        res.metadata["chip_to_image"], [0, 0, 1, 1, 2, 2]
    )
    assert res.metadata["boxes"].shape == (6, 4)


def test_load_embeddings_subsampling_preserves_chip_metadata(tmp_path: Path):
    _write_shard(
        tmp_path,
        n_samples=10,
        embed_dim=2,
        has_image_paths=True,
        has_chip_to_image=True,
        has_boxes=True,
    )
    res = load_embeddings(tmp_path, frac=0.5, seed=0)
    n = res.embeddings.shape[0]
    assert n == 5
    assert res.metadata["chip_to_image"].shape == (n,)
    assert res.metadata["boxes"].shape == (n, 4)
    assert len(res.metadata["image_paths"]) == n


def test_load_embeddings_missing_chip_flags_skipped(tmp_path: Path):
    _write_shard(tmp_path, n_samples=2, embed_dim=2)
    res = load_embeddings(tmp_path)
    assert "chip_to_image" not in res.metadata
    assert "boxes" not in res.metadata


def test_load_embeddings_invalid_frac(tmp_path: Path):
    _write_shard(tmp_path, n_samples=2, embed_dim=2)
    with pytest.raises(ValueError, match="frac"):
        load_embeddings(tmp_path, frac=0.0)
