"""Tests for ``oodkit.embeddings.result.EmbeddingResult``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.embeddings.result import EmbeddingResult


def test_to_features_with_embeddings_only():
    emb = np.random.randn(5, 8).astype(np.float64)
    result = EmbeddingResult(embeddings=emb)
    feat = result.to_features()
    assert isinstance(feat, Features)
    np.testing.assert_array_equal(feat.embeddings, emb)
    assert feat.logits is None


def test_to_features_with_logits():
    emb = np.random.randn(3, 4).astype(np.float64)
    logits = np.random.randn(3, 2).astype(np.float64)
    result = EmbeddingResult(embeddings=emb, logits=logits)
    feat = result.to_features()
    np.testing.assert_array_equal(feat.embeddings, emb)
    np.testing.assert_array_equal(feat.logits, logits)


def test_labels_and_metadata():
    emb = np.zeros((4, 2), dtype=np.float64)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    meta = {"image_paths": ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"]}
    result = EmbeddingResult(embeddings=emb, labels=labels, metadata=meta)
    np.testing.assert_array_equal(result.labels, labels)
    assert result.metadata["image_paths"][0] == "/a.jpg"


def test_default_metadata_is_empty_dict():
    result = EmbeddingResult(embeddings=np.zeros((1, 2)))
    assert result.metadata == {}
    assert result.logits is None
    assert result.labels is None
