"""Tests for ``oodkit.detectors.wdiscood.WDiscOOD``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.wdiscood import WDiscOOD


def test_wdiscood_fit_requires_y(features_embeddings_only):
    d = WDiscOOD()
    with pytest.raises(TypeError):
        d.fit(features_embeddings_only)


def test_wdiscood_fit_rejects_none_y(embeddings_two_cluster_train):
    d = WDiscOOD()
    with pytest.raises(ValueError, match="requires y"):
        d.fit(Features(embeddings=embeddings_two_cluster_train), y=None)


def test_wdiscood_fit_requires_embeddings(features_logits_only):
    d = WDiscOOD()
    with pytest.raises(ValueError, match="embeddings"):
        d.fit(features_logits_only, y=np.array([0, 1]))


def test_wdiscood_score_before_fit_raises(features_embeddings_only):
    d = WDiscOOD()
    with pytest.raises(RuntimeError, match="not fitted"):
        d.score(features_embeddings_only)


def test_wdiscood_fit_y_shape_mismatch(embeddings_two_cluster_train):
    d = WDiscOOD()
    with pytest.raises(ValueError, match="y must"):
        d.fit(Features(embeddings=embeddings_two_cluster_train), y=np.array([0, 1]))


def test_wdiscood_single_class_raises():
    X = np.ones((4, 3), dtype=np.float64)
    y = np.zeros(4, dtype=np.int64)
    d = WDiscOOD()
    with pytest.raises(ValueError, match="at least 2 distinct"):
        d.fit(Features(embeddings=X), y=y)


def test_wdiscood_n_discriminants_out_of_range(embeddings_two_cluster_train, labels_two_cluster_train):
    d = WDiscOOD(n_discriminants=2)
    with pytest.raises(ValueError, match="n_discriminants"):
        d.fit(
            Features(embeddings=embeddings_two_cluster_train),
            y=labels_two_cluster_train,
        )


def test_wdiscood_fit_score_predict_two_cluster(
    embeddings_two_cluster_train,
    labels_two_cluster_train,
    embeddings_near_far_queries,
):
    d = WDiscOOD(n_discriminants=1, ridge=1e-3, alpha=1.0)
    train = Features(embeddings=embeddings_two_cluster_train)
    assert d.fit(train, y=labels_two_cluster_train) is d

    scores = d.score(Features(embeddings=embeddings_near_far_queries))
    assert scores.shape == (2,)
    assert scores.dtype == np.float64
    assert scores[0] < scores[1]

    threshold = float((scores[0] + scores[1]) / 2.0)
    labels = d.predict(Features(embeddings=embeddings_near_far_queries), threshold=threshold)
    np.testing.assert_array_equal(labels, np.array([0, 1]))


def test_wdiscood_torch_embeddings(embeddings_two_cluster_train, labels_two_cluster_train):
    torch = pytest.importorskip("torch")
    d = WDiscOOD(n_discriminants=1)
    d.fit(
        Features(embeddings=torch.tensor(embeddings_two_cluster_train)),
        y=torch.tensor(labels_two_cluster_train),
    )
    scores = d.score(Features(embeddings=torch.tensor([[1.0, 1.0]], dtype=torch.float64)))
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)


def test_wdiscood_default_n_discriminants_uses_c_minus_one():
    """Three classes in 4D; k defaults to C-1 = 2."""
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.standard_normal((5, 4)) + np.array([3.0, 0.0, 0.0, 0.0]),
            rng.standard_normal((5, 4)) + np.array([-3.0, 0.0, 0.0, 0.0]),
            rng.standard_normal((5, 4)) + np.array([0.0, 4.0, 0.0, 0.0]),
        ]
    )
    y = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=np.int64)
    d = WDiscOOD()
    d.fit(Features(embeddings=X), y=y)
    assert d.k_fitted_ == 2
