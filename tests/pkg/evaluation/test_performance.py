"""Tests for ``oodkit.evaluation.performance``."""

import numpy as np
import pytest

from oodkit.evaluation.performance import PerformanceCurve, score_vs_metric
from oodkit.evaluation.score_bank import ScoreBank


@pytest.fixture
def simple_bank():
    """10 samples with ascending OOD scores and descending accuracy."""
    n = 10
    bank = ScoreBank()
    bank.add("det", np.linspace(0, 1, n))
    bank.add_metric("acc", np.linspace(1, 0, n))
    return bank


# ------------------------------------------------------------------
# score_vs_metric basics
# ------------------------------------------------------------------

def test_single_detector_returns_curve(simple_bank):
    curve = score_vs_metric(simple_bank, "acc", detector="det")
    assert isinstance(curve, PerformanceCurve)


def test_all_detectors_returns_dict(simple_bank):
    result = score_vs_metric(simple_bank, "acc")
    assert isinstance(result, dict)
    assert "det" in result


def test_curve_fields_populated(simple_bank):
    curve = score_vs_metric(simple_bank, "acc", detector="det", n_bins=5)
    assert len(curve.bin_edges) == 6
    assert len(curve.bin_centers) == 5
    assert len(curve.mean_metric) == 5
    assert len(curve.n_samples) == 5


def test_curve_detector_and_metric_names(simple_bank):
    curve = score_vs_metric(simple_bank, "acc", detector="det")
    assert curve.detector == "det"
    assert curve.metric_name == "acc"


# ------------------------------------------------------------------
# Binning behaviour
# ------------------------------------------------------------------

def test_total_samples_sum(simple_bank):
    curve = score_vs_metric(simple_bank, "acc", detector="det", n_bins=5)
    assert int(curve.n_samples.sum()) == simple_bank.n_samples


def test_mean_metric_decreasing_trend(simple_bank):
    """Higher OOD scores → lower accuracy in our fixture."""
    curve = score_vs_metric(simple_bank, "acc", detector="det", n_bins=5)
    valid = ~np.isnan(curve.mean_metric)
    vals = curve.mean_metric[valid]
    # First bin should have higher mean acc than last bin
    assert vals[0] > vals[-1]


def test_bin_edges_monotone(simple_bank):
    curve = score_vs_metric(simple_bank, "acc", detector="det", n_bins=4)
    assert np.all(np.diff(curve.bin_edges) >= 0)


# ------------------------------------------------------------------
# Multi-detector
# ------------------------------------------------------------------

def test_multi_detector_all_present():
    bank = ScoreBank()
    bank.add("det_a", np.linspace(0, 1, 20))
    bank.add("det_b", np.linspace(1, 0, 20))
    bank.add_metric("iou", np.random.default_rng(0).uniform(0, 1, 20))
    result = score_vs_metric(bank, "iou")
    assert set(result.keys()) == {"det_a", "det_b"}


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_n_bins_must_be_positive():
    bank = ScoreBank()
    bank.add("det", np.ones(5))
    bank.add_metric("acc", np.ones(5))
    with pytest.raises(ValueError, match="n_bins"):
        score_vs_metric(bank, "acc", n_bins=0)


def test_missing_metric_raises():
    bank = ScoreBank()
    bank.add("det", np.ones(5))
    with pytest.raises(KeyError):
        score_vs_metric(bank, "nonexistent")


def test_missing_detector_raises():
    bank = ScoreBank()
    bank.add("det", np.ones(5))
    bank.add_metric("acc", np.ones(5))
    with pytest.raises(KeyError):
        score_vs_metric(bank, "acc", detector="ghost")


def test_empty_bank_raises():
    bank = ScoreBank()
    bank.add_metric("acc", np.ones(3))
    with pytest.raises(ValueError, match="no detectors"):
        score_vs_metric(bank, "acc")


# ------------------------------------------------------------------
# Class-conditional (via bank.by_class)
# ------------------------------------------------------------------

def test_by_class_then_score_vs_metric():
    bank = ScoreBank(class_labels=np.array([0, 0, 0, 1, 1, 1]))
    bank.add("det", np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    bank.add_metric("acc", np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))

    cls0_bank = bank.by_class(0)
    curve = score_vs_metric(cls0_bank, "acc", detector="det", n_bins=3)
    assert curve.n_samples.sum() == 3
