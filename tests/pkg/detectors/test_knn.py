"""Tests for ``oodkit.detectors.knn.KNN``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.knn import KNN


def test_knn_k_must_be_positive():
    with pytest.raises(ValueError, match="k must"):
        KNN(k=0)


def test_knn_backend_validation():
    with pytest.raises(ValueError, match="backend"):
        KNN(k=3, backend="bad")  # type: ignore[arg-type]


def test_knn_metric_must_be_non_empty():
    with pytest.raises(ValueError, match="metric"):
        KNN(k=3, metric="")


def test_knn_fit_requires_embeddings(features_logits_only):
    d = KNN(k=1)
    with pytest.raises(ValueError, match="embeddings"):
        d.fit(features_logits_only)


def test_knn_score_before_fit_raises(features_embeddings_only):
    d = KNN(k=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        d.score(features_embeddings_only)


def test_knn_fit_rejects_k_gt_n_train(embeddings_two_cluster_train):
    d = KNN(k=100)
    with pytest.raises(ValueError, match="cannot exceed"):
        d.fit(Features(embeddings=embeddings_two_cluster_train))


def test_knn_score_predict_brute_backend(embeddings_two_cluster_train, embeddings_near_far_queries):
    d = KNN(k=2, backend="brute")
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    scores = d.score(Features(embeddings=embeddings_near_far_queries))
    assert scores.shape == (2,)
    assert scores[0] < scores[1]

    threshold = float((scores[0] + scores[1]) / 2.0)
    labels = d.predict(Features(embeddings=embeddings_near_far_queries), threshold=threshold)
    np.testing.assert_array_equal(labels, np.array([0, 1]))


def test_knn_auto_backend_smoke(embeddings_two_cluster_train, embeddings_near_far_queries):
    d = KNN(k=2, backend="auto")
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    scores = d.score(Features(embeddings=embeddings_near_far_queries))
    assert scores.shape == (2,)


def test_knn_predict_requires_threshold(embeddings_two_cluster_train):
    d = KNN(k=2, backend="brute").fit(Features(embeddings=embeddings_two_cluster_train))
    with pytest.raises(TypeError):
        d.predict(Features(embeddings=embeddings_two_cluster_train))


def test_knn_brute_non_euclidean_metric_prints_warning(embeddings_two_cluster_train, capsys):
    d = KNN(k=2, backend="brute", metric="manhattan")
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    out = capsys.readouterr().out
    assert "manual approach uses euclidean distance only" in out


def test_knn_torch_embeddings(embeddings_two_cluster_train):
    torch = pytest.importorskip("torch")
    d = KNN(k=2, backend="brute")
    d.fit(Features(embeddings=torch.tensor(embeddings_two_cluster_train)))
    scores = d.score(Features(embeddings=torch.tensor([[1.0, 1.0]], dtype=torch.float64)))
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
