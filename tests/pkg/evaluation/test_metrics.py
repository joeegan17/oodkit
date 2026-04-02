"""Tests for ``oodkit.evaluation.metrics``."""

import numpy as np
import pytest

from oodkit.evaluation.metrics import (
    MetricsTable,
    aupr,
    auroc,
    detection_error,
    evaluate,
    evaluate_by_class,
    fpr_at_tpr,
    pr_curve,
    roc_curve,
)
from oodkit.evaluation.score_bank import ScoreBank


# Perfect-separation fixture: first half ID, second half OOD with higher scores
@pytest.fixture
def perfect_bank():
    """4 ID samples with low scores, 4 OOD with high scores — perfect AUROC=1."""
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.85, 0.95, 0.8])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bank = ScoreBank(ood_labels=labels, class_labels=np.array([0, 0, 1, 1, 0, 0, 1, 1]))
    bank.add("perfect", scores)
    return bank


@pytest.fixture
def random_bank():
    """Shuffled scores — AUROC ~0.5."""
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) > 0.5).astype(int)
    bank = ScoreBank(ood_labels=labels)
    bank.add("random", scores)
    return bank


# ------------------------------------------------------------------
# roc_curve
# ------------------------------------------------------------------

def test_roc_curve_perfect(perfect_bank):
    fpr, tpr = roc_curve(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels)
    assert fpr[0] == 0.0
    assert tpr[0] == 0.0
    assert fpr[-1] == 1.0
    assert tpr[-1] == 1.0


def test_roc_curve_requires_both_classes():
    with pytest.raises(ValueError, match="both ID"):
        roc_curve(np.ones(5), np.ones(5, dtype=int))


def test_roc_curve_invalid_labels():
    with pytest.raises(ValueError, match="0 .* 1"):
        roc_curve(np.ones(5), np.array([0, 1, 2, 0, 1]))


# ------------------------------------------------------------------
# auroc
# ------------------------------------------------------------------

def test_auroc_perfect(perfect_bank):
    val = auroc(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_auroc_random(random_bank):
    val = auroc(random_bank.scores_for("random"), random_bank.ood_labels)
    assert 0.3 < val < 0.7


def test_auroc_inverted():
    # Inverted scores → AUROC = 0
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([0, 0, 0, 1, 1])
    val = auroc(scores, labels)
    assert val == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# fpr_at_tpr
# ------------------------------------------------------------------

def test_fpr_at_tpr_perfect(perfect_bank):
    val = fpr_at_tpr(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels, tpr=0.95)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_fpr_at_tpr_default_is_95(perfect_bank):
    val = fpr_at_tpr(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels)
    assert val == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# aupr
# ------------------------------------------------------------------

def test_aupr_perfect_ood_positive(perfect_bank):
    val = aupr(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels, positive="ood")
    assert val == pytest.approx(1.0, abs=1e-6)


def test_aupr_perfect_id_positive(perfect_bank):
    val = aupr(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels, positive="id")
    assert val == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------
# pr_curve
# ------------------------------------------------------------------

def test_pr_curve_starts_at_recall_zero(perfect_bank):
    precision, recall = pr_curve(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels)
    assert recall[0] == 0.0
    assert precision[0] == 1.0


def test_pr_curve_ends_at_recall_one(perfect_bank):
    precision, recall = pr_curve(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels)
    assert recall[-1] == pytest.approx(1.0)


# ------------------------------------------------------------------
# detection_error
# ------------------------------------------------------------------

def test_detection_error_perfect(perfect_bank):
    val = detection_error(perfect_bank.scores_for("perfect"), perfect_bank.ood_labels)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_detection_error_range(random_bank):
    val = detection_error(random_bank.scores_for("random"), random_bank.ood_labels)
    assert 0.0 <= val <= 0.5


# ------------------------------------------------------------------
# MetricsTable
# ------------------------------------------------------------------

def test_metrics_table_getitem():
    data = {
        "MSP": {
            "auroc": 0.9,
            "fpr95": 0.1,
            "aupr_ood": 0.85,
            "aupr_id": 0.82,
            "det_err": 0.08,
        }
    }
    table = MetricsTable(data)
    assert table["MSP"]["auroc"] == pytest.approx(0.9)


def test_metrics_table_to_dict():
    data = {"MSP": {"auroc": 0.9}}
    table = MetricsTable(data)
    d = table.to_dict()
    assert d["MSP"]["auroc"] == pytest.approx(0.9)
    assert d is not data  # copy


def test_metrics_table_repr():
    data = {
        "MSP": {
            "auroc": 0.9,
            "fpr95": 0.1,
            "aupr_ood": 0.8,
            "aupr_id": 0.78,
            "det_err": 0.07,
        }
    }
    table = MetricsTable(data)
    r = repr(table)
    assert "MSP" in r
    assert "AUROC" in r


def test_metrics_table_empty_repr():
    assert "empty" in repr(MetricsTable({}))


# ------------------------------------------------------------------
# evaluate
# ------------------------------------------------------------------

def test_evaluate_returns_metrics_table(perfect_bank):
    table = evaluate(perfect_bank)
    assert isinstance(table, MetricsTable)
    assert "perfect" in table.data


def test_evaluate_all_metrics(perfect_bank):
    table = evaluate(perfect_bank)
    row = table["perfect"]
    assert "auroc" in row
    assert "fpr95" in row
    assert "aupr_ood" in row
    assert "aupr_id" in row
    assert "det_err" in row


def test_evaluate_requires_ood_labels():
    bank = ScoreBank()
    bank.add("MSP", np.ones(4))
    with pytest.raises(ValueError, match="ood_labels"):
        evaluate(bank)


def test_evaluate_requires_detectors():
    bank = ScoreBank(ood_labels=np.array([0, 1]))
    with pytest.raises(ValueError, match="no detectors"):
        evaluate(bank)


def test_evaluate_perfect_values(perfect_bank):
    table = evaluate(perfect_bank)
    row = table["perfect"]
    assert row["auroc"] == pytest.approx(1.0, abs=1e-6)
    assert row["fpr95"] == pytest.approx(0.0, abs=1e-6)
    assert row["aupr_ood"] == pytest.approx(1.0, abs=1e-6)
    assert row["aupr_id"] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------
# evaluate_by_class
# ------------------------------------------------------------------

def test_evaluate_by_class_keys(perfect_bank):
    result = evaluate_by_class(perfect_bank)
    assert set(result.keys()) == {0, 1}


def test_evaluate_by_class_each_is_table(perfect_bank):
    result = evaluate_by_class(perfect_bank)
    for table in result.values():
        assert isinstance(table, MetricsTable)


def test_evaluate_by_class_requires_class_labels():
    bank = ScoreBank(ood_labels=np.array([0, 1, 0, 1]))
    bank.add("MSP", np.array([0.1, 0.9, 0.2, 0.8]))
    with pytest.raises(ValueError, match="class_labels"):
        evaluate_by_class(bank)
