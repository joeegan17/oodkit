"""Tests for ``oodkit.detectors.energy.Energy``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.energy import Energy


def test_energy_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        Energy(temperature=0)


def test_energy_fit_is_noop(features_logits_zeros):
    e = Energy()
    assert e.fit(features_logits_zeros) is e


def test_energy_score_requires_logits(features_embeddings_only):
    e = Energy()
    with pytest.raises(ValueError, match="logits"):
        e.score(features_embeddings_only)


def test_energy_score_uniform_logits(logits_zeros_small):
    e = Energy(temperature=1.0)
    C = logits_zeros_small.shape[1]
    scores = e.score(Features(logits=logits_zeros_small))
    assert scores.shape == (logits_zeros_small.shape[0],)
    expected = -1.0 * np.log(C)
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_energy_predict_requires_threshold(features_logits_only):
    e = Energy()
    one = Features(logits=features_logits_only.logits[:1])
    s0 = e.score(one)[0]
    assert e.predict(one, threshold=s0 - 10.0)[0] == 1
    assert e.predict(one, threshold=s0 + 10.0)[0] == 0


def test_energy_torch_logits():
    torch = pytest.importorskip("torch")
    e = Energy()
    logits = torch.zeros(1, 3)
    scores = e.score(Features(logits=logits))
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
