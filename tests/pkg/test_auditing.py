import numpy as np
import pytest

from oodkit.auditing import (
    correct_prediction_mask,
    fit_detector_on_mask,
    subset_features,
)
from oodkit.data.features import Features


class RecordingDetector:
    def fit(self, features, y=None, **kwargs):
        self.features_ = features
        self.y_ = y
        self.kwargs_ = kwargs
        return self

    def score(self, features):
        return np.zeros(features.embeddings.shape[0], dtype=float)


def test_subset_features_slices_logits_and_embeddings(features_both):
    mask = np.array([1, 0, 1, 0, 1])

    out = subset_features(features_both, mask)

    np.testing.assert_array_equal(out.logits, features_both.logits[[0, 2, 4]])
    np.testing.assert_array_equal(
        out.embeddings, features_both.embeddings[[0, 2, 4]]
    )


def test_fit_detector_on_mask_slices_features_and_labels(features_both):
    detector = RecordingDetector()
    labels = np.array([0, 1, 2, 3, 4])
    mask = np.array([True, False, True, False, False])

    returned = fit_detector_on_mask(
        detector,
        features_both,
        mask,
        y=labels,
        min_samples=2,
        answer=42,
    )

    assert returned is detector
    np.testing.assert_array_equal(detector.features_.logits, features_both.logits[[0, 2]])
    np.testing.assert_array_equal(
        detector.features_.embeddings, features_both.embeddings[[0, 2]]
    )
    np.testing.assert_array_equal(detector.y_, np.array([0, 2]))
    assert detector.kwargs_ == {"answer": 42}


def test_fit_detector_on_mask_rejects_too_few_samples(features_embeddings_only):
    with pytest.raises(ValueError, match="selected 0 samples"):
        fit_detector_on_mask(
            RecordingDetector(),
            features_embeddings_only,
            np.zeros(features_embeddings_only.embeddings.shape[0], dtype=bool),
        )


def test_fit_detector_on_mask_rejects_misaligned_labels(features_embeddings_only):
    mask = np.ones(features_embeddings_only.embeddings.shape[0], dtype=bool)
    with pytest.raises(ValueError, match="y length"):
        fit_detector_on_mask(
            RecordingDetector(),
            features_embeddings_only,
            mask,
            y=np.array([0, 1]),
        )


def test_correct_prediction_mask_without_confidence():
    logits = np.array(
        [
            [4.0, 1.0],
            [0.0, 3.0],
            [2.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 0])

    mask = correct_prediction_mask(logits, labels)

    np.testing.assert_array_equal(mask, np.array([True, False, True]))


def test_correct_prediction_mask_with_confidence_floor():
    logits = np.array(
        [
            [4.0, 1.0],
            [1.1, 1.0],
            [0.0, 3.0],
        ]
    )
    labels = np.array([0, 0, 1])

    mask = correct_prediction_mask(logits, labels, min_confidence=0.8)

    np.testing.assert_array_equal(mask, np.array([True, False, True]))


def test_correct_prediction_mask_validates_shapes():
    with pytest.raises(ValueError, match="logits must have shape"):
        correct_prediction_mask(np.zeros(3), np.zeros(3))

    with pytest.raises(ValueError, match="labels length"):
        correct_prediction_mask(np.zeros((3, 2)), np.zeros(2))

    with pytest.raises(ValueError, match="min_confidence"):
        correct_prediction_mask(np.zeros((3, 2)), np.zeros(3), min_confidence=1.5)


def test_subset_features_rejects_mismatched_feature_lengths():
    features = Features(logits=np.zeros((3, 2)), embeddings=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="disagree"):
        subset_features(features, np.ones(3, dtype=bool))
