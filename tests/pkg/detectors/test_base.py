"""Tests for ``oodkit.detectors.base.BaseDetector``."""

import numpy as np
import pytest

from oodkit.data.features import Features
from oodkit.detectors.base import BaseDetector


def test_base_detector_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseDetector()


def test_base_detector_predict_not_implemented_by_default(features_logits_only):
    class MinimalDetector(BaseDetector):
        def fit(self, features, **kwargs):
            return self

        def score(self, features, **kwargs):
            return np.array([0.0])

    d = MinimalDetector()
    one_row = Features(logits=features_logits_only.logits[:1])
    with pytest.raises(NotImplementedError):
        d.predict(one_row)
