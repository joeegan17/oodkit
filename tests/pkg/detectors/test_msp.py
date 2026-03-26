"""Tests for ``oodkit.detectors.msp.MSP``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.msp import MSP


def test_msp_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        MSP(temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        MSP(temperature=-1.0)


def test_msp_fit_is_noop(features_logits_zeros):
    msp = MSP()
    assert msp.fit(features_logits_zeros) is msp


def test_msp_score_requires_logits(features_embeddings_only):
    msp = MSP()
    with pytest.raises(ValueError, match="logits"):
        msp.score(features_embeddings_only)


def test_msp_score_rejects_wrong_logit_ndim(logits_1d_invalid):
    msp = MSP()
    f = Features(logits=logits_1d_invalid)
    with pytest.raises(ValueError, match="shape"):
        msp.score(f)


def test_msp_score_shape_and_range(logits_sharp, logits_uniform_1x3):
    msp = MSP(temperature=1.0)
    scores = msp.score(Features(logits=logits_sharp))
    assert scores.shape == (1,)
    assert scores[0] < -0.99

    scores_u = msp.score(Features(logits=logits_uniform_1x3))
    assert scores_u.shape == (1,)
    np.testing.assert_allclose(scores_u[0], -1.0 / 3.0, rtol=1e-5)


def test_msp_temperature_scales_logits(logits_temperature_compare):
    msp1 = MSP(temperature=1.0)
    msp2 = MSP(temperature=2.0)
    f = Features(logits=logits_temperature_compare)
    s1 = msp1.score(f)
    s2 = msp2.score(f)
    assert not np.allclose(s1, s2)


def test_msp_predict(logits_msp_predict_batch):
    msp = MSP()
    f = Features(logits=logits_msp_predict_batch)
    labels = msp.predict(f, threshold=-0.5)
    assert labels.shape == (2,)
    assert np.issubdtype(labels.dtype, np.integer)


def test_msp_torch_logits():
    torch = pytest.importorskip("torch")
    msp = MSP()
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    scores = msp.score(Features(logits=logits))
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
