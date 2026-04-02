"""Tests for ``oodkit.evaluation.ScoreBank``."""

import numpy as np
import pytest

from oodkit.evaluation.score_bank import ScoreBank


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_empty_construction():
    bank = ScoreBank()
    assert bank.detectors == []
    assert not bank.has_ood_labels
    assert not bank.has_class_labels
    assert bank.metric_names == []


def test_n_samples_raises_when_empty():
    with pytest.raises(ValueError, match="empty"):
        _ = ScoreBank().n_samples


def test_add_single_detector():
    bank = ScoreBank()
    scores = np.array([0.1, 0.5, 0.9])
    bank.add("MSP", scores)
    assert bank.detectors == ["MSP"]
    assert bank.n_samples == 3
    np.testing.assert_array_equal(bank.scores_for("MSP"), scores)


def test_add_chaining():
    bank = ScoreBank()
    result = bank.add("A", np.ones(4)).add("B", np.zeros(4))
    assert result is bank
    assert set(bank.detectors) == {"A", "B"}


def test_construction_with_scores_dict():
    scores = {"MSP": np.array([0.1, 0.2]), "Energy": np.array([1.0, 2.0])}
    bank = ScoreBank(scores=scores)
    assert set(bank.detectors) == {"MSP", "Energy"}
    assert bank.n_samples == 2


def test_add_metric_chaining():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    result = bank.add_metric("acc", np.array([0.8, 0.6, 0.4]))
    assert result is bank
    assert bank.metric_names == ["acc"]
    np.testing.assert_array_almost_equal(bank.metric_for("acc"), [0.8, 0.6, 0.4])


def test_ood_labels_stored():
    labels = np.array([0, 0, 1, 1])
    bank = ScoreBank(ood_labels=labels)
    bank.add("MSP", np.array([0.1, 0.2, 0.8, 0.9]))
    assert bank.has_ood_labels
    np.testing.assert_array_equal(bank.ood_labels, labels)


def test_class_labels_stored():
    cls = np.array([0, 1, 0, 1])
    bank = ScoreBank(class_labels=cls)
    bank.add("MSP", np.ones(4))
    assert bank.has_class_labels
    np.testing.assert_array_equal(bank.class_labels, cls)
    np.testing.assert_array_equal(bank.classes, [0, 1])


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_length_mismatch_raises():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    with pytest.raises(ValueError, match="Length mismatch"):
        bank.add("Energy", np.ones(5))


def test_ood_labels_length_mismatch():
    bank = ScoreBank(ood_labels=np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="Length mismatch"):
        bank.add("MSP", np.ones(5))


def test_scores_for_missing_raises():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    with pytest.raises(KeyError):
        bank.scores_for("Energy")


def test_metric_for_missing_raises():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    with pytest.raises(KeyError):
        bank.metric_for("accuracy")


# ------------------------------------------------------------------
# Slicing
# ------------------------------------------------------------------

def test_subset():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    ood = np.array([0, 0, 1, 1])
    bank = ScoreBank(ood_labels=ood)
    bank.add("MSP", scores)

    sub = bank.subset([1, 3])
    assert sub.n_samples == 2
    np.testing.assert_array_equal(sub.scores_for("MSP"), [2.0, 4.0])
    np.testing.assert_array_equal(sub.ood_labels, [0, 1])


def test_by_class():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cls = np.array([0, 1, 0, 1, 0, 1])
    bank = ScoreBank(class_labels=cls)
    bank.add("Det", scores)

    cls0 = bank.by_class(0)
    assert cls0.n_samples == 3
    np.testing.assert_array_equal(cls0.scores_for("Det"), [1.0, 3.0, 5.0])


def test_by_class_requires_class_labels():
    bank = ScoreBank()
    bank.add("MSP", np.ones(4))
    with pytest.raises(ValueError, match="class_labels"):
        bank.by_class(0)


def test_subset_preserves_sample_metrics():
    bank = ScoreBank()
    bank.add("MSP", np.array([0.1, 0.5, 0.9]))
    bank.add_metric("acc", np.array([0.9, 0.5, 0.1]))

    sub = bank.subset([0, 2])
    np.testing.assert_array_almost_equal(sub.metric_for("acc"), [0.9, 0.1])


# ------------------------------------------------------------------
# Repr
# ------------------------------------------------------------------

def test_repr_non_empty():
    bank = ScoreBank(ood_labels=np.array([0, 1]))
    bank.add("MSP", np.ones(2))
    r = repr(bank)
    assert "ScoreBank" in r
    assert "MSP" in r
    assert "ood_labels" in r
