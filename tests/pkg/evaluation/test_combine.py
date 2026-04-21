"""Tests for ``oodkit.evaluation.combine``."""

import numpy as np
import pytest

from oodkit.embeddings.result import EmbeddingResult
from oodkit.evaluation.combine import (
    concatenate_embedding_results,
    ood_labels_from_blocks,
    ood_labels_from_counts,
)


def test_ood_labels_from_counts():
    y = ood_labels_from_counts(3, 2)
    np.testing.assert_array_equal(y, np.array([0, 0, 0, 1, 1]))


def test_ood_labels_from_counts_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ood_labels_from_counts(-1, 1)


def test_ood_labels_from_blocks():
    y = ood_labels_from_blocks([2, 1, 3], [0, 1, 0])
    np.testing.assert_array_equal(y, np.array([0, 0, 1, 0, 0, 0]))


def test_ood_labels_from_blocks_empty():
    y = ood_labels_from_blocks([], [])
    assert y.shape == (0,)


def test_ood_labels_from_blocks_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        ood_labels_from_blocks([1], [0, 1])


def test_concatenate_embedding_results_basic():
    r_id = EmbeddingResult(
        embeddings=np.zeros((2, 4), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        metadata={"image_paths": ["/a", "/b"]},
    )
    r_ood = EmbeddingResult(
        embeddings=np.ones((3, 4), dtype=np.float32),
        labels=np.array([5, 5, 5], dtype=np.int64),
        metadata={"image_paths": ["/c", "/d", "/e"]},
    )
    comb, ood = concatenate_embedding_results([r_id, r_ood], [0, 1])
    assert comb.embeddings.shape == (5, 4)
    assert comb.labels is not None and comb.labels.shape == (5,)
    np.testing.assert_array_equal(ood, np.array([0, 0, 1, 1, 1]))
    assert comb.metadata["image_paths"] == ["/a", "/b", "/c", "/d", "/e"]


def test_concatenate_embedding_results_with_logits():
    r1 = EmbeddingResult(
        embeddings=np.zeros((1, 2), dtype=np.float32),
        logits=np.zeros((1, 10), dtype=np.float32),
    )
    r2 = EmbeddingResult(
        embeddings=np.ones((1, 2), dtype=np.float32),
        logits=np.ones((1, 10), dtype=np.float32),
    )
    comb, ood = concatenate_embedding_results([r1, r2], [0, 1])
    assert comb.logits is not None and comb.logits.shape == (2, 10)
    np.testing.assert_array_equal(ood, np.array([0, 1]))


def test_concatenate_embedding_results_logits_mismatch_raises():
    r1 = EmbeddingResult(embeddings=np.zeros((1, 2), dtype=np.float32), logits=np.zeros((1, 3)))
    r2 = EmbeddingResult(embeddings=np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="logits"):
        concatenate_embedding_results([r1, r2], [0, 1])


def test_concatenate_embedding_results_emb_dim_mismatch_raises():
    r1 = EmbeddingResult(embeddings=np.zeros((1, 2)))
    r2 = EmbeddingResult(embeddings=np.ones((1, 3)))
    with pytest.raises(ValueError, match="feature dimension"):
        concatenate_embedding_results([r1, r2], [0, 1])


def test_concatenate_embedding_results_chip_metadata_offsets():
    r_id = EmbeddingResult(
        embeddings=np.zeros((3, 4), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0, 0, 1], dtype=np.int64),
            "boxes": np.array(
                [[0, 0, 10, 10], [5, 5, 20, 20], [0, 0, 8, 8]], dtype=np.float64
            ),
        },
    )
    r_ood = EmbeddingResult(
        embeddings=np.ones((2, 4), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0, 1], dtype=np.int64),
            "boxes": np.array([[1, 1, 9, 9], [2, 2, 12, 12]], dtype=np.float64),
        },
    )
    comb, ood = concatenate_embedding_results([r_id, r_ood], [0, 1])
    np.testing.assert_array_equal(
        comb.metadata["chip_to_image"], [0, 0, 1, 2, 3]
    )
    assert comb.metadata["boxes"].shape == (5, 4)
    np.testing.assert_array_equal(
        comb.metadata["boxes"][-1], [2, 2, 12, 12]
    )
    np.testing.assert_array_equal(ood, [0, 0, 0, 1, 1])


def test_concatenate_embedding_results_chip_metadata_all_or_none():
    r1 = EmbeddingResult(
        embeddings=np.zeros((1, 2), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0], dtype=np.int64),
            "boxes": np.zeros((1, 4), dtype=np.float64),
        },
    )
    r2 = EmbeddingResult(embeddings=np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="chip_to_image"):
        concatenate_embedding_results([r1, r2], [0, 1])


def test_concatenate_embedding_results_od_list_metadata():
    r_id = EmbeddingResult(
        embeddings=np.zeros((2, 4), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0, 0], dtype=np.int64),
            "boxes": np.zeros((2, 4), dtype=np.float64),
            "object_ids": ["a_0", "a_1"],
            "group": ["id", "id"],
            "image_ids": ["a", "a"],
        },
    )
    r_ood = EmbeddingResult(
        embeddings=np.ones((1, 4), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0], dtype=np.int64),
            "boxes": np.zeros((1, 4), dtype=np.float64),
            "object_ids": ["b_0"],
            "group": ["cartoon"],
            "image_ids": ["b"],
        },
    )
    comb, _ = concatenate_embedding_results([r_id, r_ood], [0, 1])
    assert comb.metadata["object_ids"] == ["a_0", "a_1", "b_0"]
    assert comb.metadata["group"] == ["id", "id", "cartoon"]
    assert comb.metadata["image_ids"] == ["a", "a", "b"]


def test_concatenate_embedding_results_od_list_all_or_none():
    r1 = EmbeddingResult(
        embeddings=np.zeros((1, 2), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0], dtype=np.int64),
            "boxes": np.zeros((1, 4), dtype=np.float64),
            "group": ["cartoon"],
        },
    )
    r2 = EmbeddingResult(
        embeddings=np.ones((1, 2), dtype=np.float32),
        metadata={
            "chip_to_image": np.array([0], dtype=np.int64),
            "boxes": np.zeros((1, 4), dtype=np.float64),
        },
    )
    with pytest.raises(ValueError, match="'group'"):
        concatenate_embedding_results([r1, r2], [0, 1])
