"""Tests for ``oodkit.detectors.mahalanobis.Mahalanobis``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.mahalanobis import Mahalanobis


def test_mahalanobis_eps_must_be_positive():
    with pytest.raises(ValueError, match="eps"):
        Mahalanobis(eps=0.0)


def test_mahalanobis_fit_requires_embeddings(features_logits_only):
    d = Mahalanobis()
    with pytest.raises(ValueError, match="embeddings"):
        d.fit(features_logits_only)


def test_mahalanobis_score_before_fit_raises(features_embeddings_only):
    d = Mahalanobis()
    with pytest.raises(RuntimeError, match="not fitted"):
        d.score(features_embeddings_only)


def test_mahalanobis_fit_y_shape_mismatch(embeddings_two_cluster_train):
    d = Mahalanobis()
    with pytest.raises(ValueError, match="y must"):
        d.fit(Features(embeddings=embeddings_two_cluster_train), y=np.array([0, 1]))


def test_mahalanobis_fit_score_predict_class_conditional(
    embeddings_two_cluster_train,
    labels_two_cluster_train,
    embeddings_near_far_queries,
):
    d = Mahalanobis(eps=1e-6)
    train = Features(embeddings=embeddings_two_cluster_train)
    assert d.fit(train, y=labels_two_cluster_train) is d

    scores = d.score(Features(embeddings=embeddings_near_far_queries))
    assert scores.shape == (2,)
    assert scores[0] < scores[1]

    threshold = float((scores[0] + scores[1]) / 2.0)
    labels = d.predict(Features(embeddings=embeddings_near_far_queries), threshold=threshold)
    np.testing.assert_array_equal(labels, np.array([0, 1]))


def test_mahalanobis_fit_without_y_single_gaussian(
    embeddings_two_cluster_train,
    embeddings_near_far_queries,
    capsys,
):
    d = Mahalanobis()
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    out = capsys.readouterr().out
    assert "y is None" in out
    scores = d.score(Features(embeddings=embeddings_near_far_queries))
    assert scores.shape == (2,)


def test_mahalanobis_predict_requires_threshold(
    embeddings_two_cluster_train,
    labels_two_cluster_train,
):
    d = Mahalanobis().fit(Features(embeddings=embeddings_two_cluster_train), y=labels_two_cluster_train)
    with pytest.raises(TypeError):
        d.predict(Features(embeddings=embeddings_two_cluster_train))


def test_mahalanobis_torch_embeddings(embeddings_two_cluster_train, labels_two_cluster_train):
    torch = pytest.importorskip("torch")
    d = Mahalanobis()
    d.fit(
        Features(embeddings=torch.tensor(embeddings_two_cluster_train)),
        y=torch.tensor(labels_two_cluster_train),
    )
    scores = d.score(Features(embeddings=torch.tensor([[1.0, 1.0]], dtype=torch.float64)))
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
