"""Tests for ``oodkit.detectors.vim.ViM``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.vim import ViM


def test_vim_compute_origin_shape(vim_origin_W, vim_origin_b):
    o = ViM.compute_origin(vim_origin_W, vim_origin_b)
    assert o.shape == (3,)


def test_vim_compute_origin_bad_W_ndim():
    with pytest.raises(ValueError, match="W must"):
        ViM.compute_origin(np.zeros(3), np.zeros(2))


def test_vim_compute_origin_b_shape_mismatch():
    W = np.zeros((2, 3))
    with pytest.raises(ValueError, match="b must"):
        ViM.compute_origin(W, np.zeros(3))


def test_vim_center_features(center_X_and_o):
    xc = ViM.center_features(center_X_and_o.X, center_X_and_o.o)
    np.testing.assert_allclose(xc, center_X_and_o.expected)


def test_vim_center_features_bad_shapes():
    with pytest.raises(ValueError, match="X must"):
        ViM.center_features(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="o must"):
        ViM.center_features(np.zeros((2, 4)), np.zeros(3))


def test_vim_get_residual_projector_D_bounds(residual_Xc_bad_D):
    with pytest.raises(ValueError, match="D must"):
        ViM.get_residual_projector(residual_Xc_bad_D, D=10)


def test_vim_score_before_fit_raises(vim_linear_pack):
    vim = ViM(vim_linear_pack.W, vim_linear_pack.b, vim_linear_pack.D)
    f = Features(
        logits=vim_linear_pack.logits[:2],
        embeddings=vim_linear_pack.embeddings[:2],
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        vim.score(f)


def test_vim_fit_score_predict_end_to_end(vim_linear_pack):
    p = vim_linear_pack
    vim = ViM(p.W, p.b, p.D)
    train = Features(logits=p.logits, embeddings=p.embeddings)
    assert vim.fit(train) is vim
    test = Features(logits=p.logits[:2], embeddings=p.embeddings[:2])
    scores = vim.score(test)
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
    labels = vim.predict(test, threshold=0.5)
    assert labels.shape == (2,)


def test_vim_compute_alpha_and_vim_score(
    logits_alpha_vim_score,
    residual_norms_alpha_vim,
):
    alpha = ViM.compute_alpha(logits_alpha_vim_score, residual_norms_alpha_vim)
    assert alpha > 0
    s = ViM.compute_vim_score(
        logits_alpha_vim_score,
        residual_norms_alpha_vim,
        alpha,
    )
    assert s.shape == (2,)
    assert np.all((s >= 0) & (s <= 1))


def test_vim_torch_W_b_and_features(vim_linear_pack):
    torch = pytest.importorskip("torch")
    p = vim_linear_pack
    vim = ViM(torch.tensor(p.W), torch.tensor(p.b), p.D)
    train = Features(
        logits=torch.tensor(p.logits),
        embeddings=torch.tensor(p.embeddings),
    )
    vim.fit(train)
    scores = vim.score(
        Features(
            logits=torch.tensor(p.logits[:1]),
            embeddings=torch.tensor(p.embeddings[:1]),
        )
    )
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
