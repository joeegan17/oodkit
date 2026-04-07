"""Tests for ``oodkit.evaluation.compare``."""

import numpy as np
import pytest

from oodkit.evaluation.compare import (
    disagreements,
    normalize_scores,
    rank_samples,
    score_correlation,
    _to_ranks,
)
from oodkit.evaluation.score_bank import ScoreBank


@pytest.fixture
def two_detector_bank():
    """Two detectors on 6 samples with known score ordering."""
    # det_a: ascending; det_b: descending — perfectly opposite
    bank = ScoreBank()
    bank.add("det_a", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    bank.add("det_b", np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    return bank


# ------------------------------------------------------------------
# _to_ranks
# ------------------------------------------------------------------

def test_to_ranks_basic():
    ranks = _to_ranks(np.array([3.0, 1.0, 2.0]))
    np.testing.assert_array_equal(ranks, [2, 0, 1])


def test_to_ranks_ties():
    ranks = _to_ranks(np.array([1.0, 1.0, 2.0]))
    assert ranks[2] == 2


# ------------------------------------------------------------------
# rank_samples
# ------------------------------------------------------------------

def test_rank_samples_ood_returns_highest(two_detector_bank):
    idx = rank_samples(two_detector_bank, "det_a", top_k=3, direction="ood")
    # det_a is ascending, so highest scores are last: indices 5, 4, 3
    assert idx[0] == 5


def test_rank_samples_id_returns_lowest(two_detector_bank):
    idx = rank_samples(two_detector_bank, "det_a", top_k=3, direction="id")
    assert idx[0] == 0


def test_rank_samples_top_k_clamps():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    idx = rank_samples(bank, "MSP", top_k=100)
    assert len(idx) == 3


def test_rank_samples_invalid_direction():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    with pytest.raises(ValueError, match="direction"):
        rank_samples(bank, "MSP", direction="sideways")


# ------------------------------------------------------------------
# disagreements
# ------------------------------------------------------------------

def test_disagreements_opposite_detectors_all_disagree(two_detector_bank):
    """When det_a and det_b have perfectly opposite rankings, all samples disagree."""
    idx = disagreements(two_detector_bank, "det_a", "det_b", top_k=6)
    assert len(idx) == 6


def test_disagreements_identical_detectors_no_disagreement():
    bank = ScoreBank()
    bank.add("det_a", np.array([1.0, 2.0, 3.0]))
    bank.add("det_b", np.array([1.0, 2.0, 3.0]))
    idx = disagreements(bank, "det_a", "det_b", top_k=3)
    # Rank differences should all be zero; order is arbitrary but length correct
    assert len(idx) == 3


def test_disagreements_scale_irrelevant():
    """Same rank ordering at very different scales → zero disagreement."""
    bank = ScoreBank()
    bank.add("det_a", np.array([0.1, 0.2, 0.3]))
    bank.add("det_b", np.array([100.0, 200.0, 300.0]))
    # All rank diffs = 0 even though raw scores differ enormously
    idx = disagreements(bank, "det_a", "det_b", top_k=3)
    scores_a = bank.scores_for("det_a")
    scores_b = bank.scores_for("det_b")
    from oodkit.evaluation.compare import _to_ranks
    rank_diff = np.abs(_to_ranks(scores_a)[idx] - _to_ranks(scores_b)[idx])
    np.testing.assert_array_equal(rank_diff, 0)


# ------------------------------------------------------------------
# score_correlation
# ------------------------------------------------------------------

def test_score_correlation_shape(two_detector_bank):
    corr = score_correlation(two_detector_bank)
    assert corr.shape == (2, 2)


def test_score_correlation_diagonal_is_one(two_detector_bank):
    corr = score_correlation(two_detector_bank)
    np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0])


def test_score_correlation_opposite_is_minus_one(two_detector_bank):
    corr = score_correlation(two_detector_bank, method="spearman")
    # det_a and det_b are perfectly inverted in rank → Spearman = -1
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-10)


def test_score_correlation_invalid_method():
    bank = ScoreBank()
    bank.add("a", np.ones(3))
    bank.add("b", np.ones(3))
    with pytest.raises(ValueError, match="method"):
        score_correlation(bank, method="cosine")


def test_score_correlation_requires_two_detectors():
    bank = ScoreBank()
    bank.add("only", np.ones(3))
    with pytest.raises(ValueError, match="2 detectors"):
        score_correlation(bank)


def test_score_correlation_spearman_scale_invariant():
    """Same rank pattern at very different scales → identical Spearman matrix."""
    bank1 = ScoreBank()
    bank1.add("a", np.array([1.0, 2.0, 3.0]))
    bank1.add("b", np.array([4.0, 5.0, 6.0]))

    bank2 = ScoreBank()
    bank2.add("a", np.array([100.0, 200.0, 300.0]))
    bank2.add("b", np.array([4000.0, 5000.0, 6000.0]))

    np.testing.assert_array_almost_equal(
        score_correlation(bank1, "spearman"),
        score_correlation(bank2, "spearman"),
    )


# ------------------------------------------------------------------
# normalize_scores
# ------------------------------------------------------------------

def test_normalize_standardize_zero_mean_unit_std():
    bank = ScoreBank()
    bank.add("MSP", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    norm = normalize_scores(bank, method="standardize")
    s = norm.scores_for("MSP")
    # ScoreBank stores float32; standardized mean is ~0 within float32 noise (~1e-7)
    assert s.mean() == pytest.approx(0.0, abs=1e-6)
    assert s.std() == pytest.approx(1.0, abs=1e-6)


def test_normalize_minmax_range():
    bank = ScoreBank()
    bank.add("MSP", np.array([2.0, 4.0, 6.0]))
    norm = normalize_scores(bank, method="minmax")
    s = norm.scores_for("MSP")
    assert s.min() == pytest.approx(0.0)
    assert s.max() == pytest.approx(1.0)


def test_normalize_does_not_mutate_original():
    scores = np.array([1.0, 2.0, 3.0])
    bank = ScoreBank()
    bank.add("MSP", scores.copy())
    _ = normalize_scores(bank, method="standardize")
    np.testing.assert_array_equal(bank.scores_for("MSP"), scores)


def test_normalize_preserves_ood_labels():
    labels = np.array([0, 1, 0])
    bank = ScoreBank(ood_labels=labels)
    bank.add("MSP", np.array([0.1, 0.9, 0.2]))
    norm = normalize_scores(bank)
    np.testing.assert_array_equal(norm.ood_labels, labels)


def test_normalize_invalid_method():
    bank = ScoreBank()
    bank.add("MSP", np.ones(3))
    with pytest.raises(ValueError, match="method"):
        normalize_scores(bank, method="sigmoid")
