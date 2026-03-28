"""Tests for ``oodkit.detectors.pca_fusion.PCAFusion``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.pca_fusion import PCAFusion


def test_pca_fusion_temperature_eps_validation():
    with pytest.raises(ValueError, match="temperature"):
        PCAFusion(temperature=0.0)
    with pytest.raises(ValueError, match="eps"):
        PCAFusion(eps=0.0)


def test_pca_fusion_fit_requires_two_features():
    d = PCAFusion(n_components=1)
    X = np.array([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="at least 2 feature"):
        d.fit(Features(embeddings=X))


def test_pca_fusion_fit_requires_two_samples():
    d = PCAFusion(n_components=1)
    X = np.array([[0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="at least 2 samples"):
        d.fit(Features(embeddings=X))


def test_pca_fusion_score_before_fit(features_embeddings_only):
    d = PCAFusion(n_components=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        d.score(
            Features(
                embeddings=features_embeddings_only.embeddings[:1],
                logits=np.zeros((1, 3)),
            )
        )


def test_pca_fusion_score_requires_logits(features_embeddings_only):
    d = PCAFusion(n_components=1)
    d.fit(features_embeddings_only)
    with pytest.raises(ValueError, match="logits"):
        d.score(Features(embeddings=features_embeddings_only.embeddings))


def test_pca_fusion_n_components_invalid():
    d = PCAFusion(n_components=5)
    X = np.random.default_rng(0).standard_normal((10, 3))
    with pytest.raises(ValueError, match="n_components"):
        d.fit(Features(embeddings=X))


def test_pca_fusion_pct_variance_invalid():
    d = PCAFusion(n_components=None, pct_variance=1.5)
    X = np.random.default_rng(0).standard_normal((10, 3))
    with pytest.raises(ValueError, match="pct_variance"):
        d.fit(Features(embeddings=X))


def test_pca_fusion_fit_score_predict_high_residual_more_ood():
    """Off-manifold point should score higher (more OOD) than on-manifold with same logits."""
    train_emb = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 0.0]], dtype=np.float64)
    logits_train = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    d = PCAFusion(n_components=1, temperature=1.0)
    d.fit(Features(embeddings=train_emb))

    logits_q = np.array([[0.0, 0.0]], dtype=np.float64)
    on_manifold = Features(embeddings=np.array([[1.5, 0.0]]), logits=logits_q)
    off_manifold = Features(embeddings=np.array([[1.5, 4.0]]), logits=logits_q)

    s_on = d.score(on_manifold)[0]
    s_off = d.score(off_manifold)[0]
    assert s_off > s_on

    mid = (s_on + s_off) / 2.0
    assert d.predict(on_manifold, threshold=mid)[0] == 0
    assert d.predict(off_manifold, threshold=mid)[0] == 1


def test_pca_fusion_default_pct_variance_fits():
    d = PCAFusion()
    X = np.random.default_rng(1).standard_normal((20, 4))
    d.fit(Features(embeddings=X))
    assert hasattr(d, "n_components_fitted_")
    assert 1 <= d.n_components_fitted_ < 4


def test_pca_fusion_predict_requires_threshold():
    d = PCAFusion(n_components=1)
    d.fit(Features(embeddings=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])))
    f = Features(
        embeddings=np.array([[0.5, 0.0]]),
        logits=np.zeros((1, 2)),
    )
    with pytest.raises(TypeError):
        d.predict(f)


def test_pca_fusion_torch():
    torch = pytest.importorskip("torch")
    train_emb = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    logits = torch.zeros(3, 2)
    d = PCAFusion(n_components=1)
    d.fit(Features(embeddings=train_emb))
    scores = d.score(
        Features(
            embeddings=torch.tensor([[1.0, 2.0]]),
            logits=torch.zeros(1, 2),
        )
    )
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
