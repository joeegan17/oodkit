"""Tests for ``oodkit.data.features.Features``."""

import numpy as np
import pytest

from oodkit.data.features import Features


def test_features_requires_at_least_one_field():
    with pytest.raises(ValueError, match="At least one"):
        Features()


def test_features_logits_only():
    logits = np.zeros((2, 3))
    f = Features(logits=logits)
    assert f.logits is logits
    assert f.embeddings is None


def test_features_embeddings_only():
    emb = np.zeros((2, 5))
    f = Features(embeddings=emb)
    assert f.embeddings is emb
    assert f.logits is None


def test_features_both():
    logits = np.ones((1, 2))
    emb = np.ones((1, 4))
    f = Features(logits=logits, embeddings=emb)
    assert f.logits is logits
    assert f.embeddings is emb


def test_features_fixture_shapes(features_both, toy_dims):
    """Shared fixtures produce consistent dimensions."""
    assert features_both.logits.shape == (toy_dims.n_samples, toy_dims.n_classes)
    assert features_both.embeddings.shape == (toy_dims.n_samples, toy_dims.n_features)
