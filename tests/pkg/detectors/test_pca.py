"""Tests for ``oodkit.detectors.pca.PCA``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.pca import PCA


def test_pca_kernel_invalid():
    with pytest.raises(ValueError, match="kernel"):
        PCA(kernel="bad")  # type: ignore[arg-type]


def test_pca_linear_fit_score(embeddings_two_cluster_train):
    d = PCA(kernel="linear", n_components=1)
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    s = d.score(Features(embeddings=embeddings_two_cluster_train[:2]))
    assert s.shape == (2,)
    assert np.all(s >= 0)


def test_pca_cosine_fit_score(embeddings_two_cluster_train):
    d = PCA(kernel="cosine", n_components=1)
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    s = d.score(Features(embeddings=embeddings_two_cluster_train[:2]))
    assert s.shape == (2,)


def test_pca_rff_cosine_smoke(embeddings_two_cluster_train):
    d = PCA(kernel="rff_cosine", n_components=1, rff_dim=32, random_state=0)
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    s = d.score(Features(embeddings=embeddings_two_cluster_train[:2]))
    assert s.shape == (2,)


def test_pca_off_manifold_higher_score():
    train_emb = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 0.0]], dtype=np.float64)
    d = PCA(kernel="linear", n_components=1)
    d.fit(Features(embeddings=train_emb))
    on = Features(embeddings=np.array([[1.5, 0.0]]))
    off = Features(embeddings=np.array([[1.5, 4.0]]))
    assert d.score(off)[0] > d.score(on)[0]


def test_pca_n_components_fitted_after_fit(embeddings_two_cluster_train):
    d = PCA(n_components=1)
    d.fit(Features(embeddings=embeddings_two_cluster_train))
    assert d.n_components_fitted_ == 1


def test_pca_torch(embeddings_two_cluster_train):
    torch = pytest.importorskip("torch")
    d = PCA(n_components=1)
    d.fit(Features(embeddings=torch.tensor(embeddings_two_cluster_train)))
    s = d.score(Features(embeddings=torch.tensor([[0.0, 0.0]])))
    assert isinstance(s, np.ndarray)
