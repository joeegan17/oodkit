"""Minimal smoke tests for ``oodkit.evaluation.plots`` additions.

These exercise the new behavior (KDE / standardize modes and the enhanced
``rank_grid`` filters) without asserting visual content.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from oodkit.evaluation.plots import rank_grid, score_distributions
from oodkit.evaluation.score_bank import ScoreBank


def _toy_bank(n: int = 20) -> ScoreBank:
    rng = np.random.default_rng(0)
    ood = np.array([0] * (n // 2) + [1] * (n - n // 2))
    cls = np.array([0, 1] * (n // 2))
    groups = np.array(
        ["id"] * (n // 2) + (["cartoon", "tattoo"] * (n - n // 2))[: n - n // 2],
        dtype=object,
    )
    scores = rng.normal(size=n).astype(np.float32)
    bank = ScoreBank(
        scores={"Energy": scores, "ViM": scores + 0.1},
        ood_labels=ood,
        class_labels=cls,
        class_names=["cat", "dog"],
        groups=groups,
    )
    return bank


def test_score_distributions_kde_smoke():
    bank = _toy_bank()
    fig = score_distributions(bank, kind="kde")
    assert fig is not None
    plt.close(fig)


def test_score_distributions_standardize_smoke():
    bank = _toy_bank()
    fig = score_distributions(bank, kind="kde", standardize=True)
    assert fig is not None
    plt.close(fig)


def test_score_distributions_standardize_requires_ood_labels():
    bank = ScoreBank()
    bank.add("MSP", np.random.rand(10).astype(np.float32))
    with pytest.raises(ValueError, match="ood_labels"):
        score_distributions(bank, kind="kde", standardize=True)


def test_score_distributions_invalid_kind():
    bank = _toy_bank()
    with pytest.raises(ValueError, match="kind"):
        score_distributions(bank, kind="violin")


def test_rank_grid_filters_text_mode():
    bank = _toy_bank()
    fig = rank_grid(
        bank, "ViM",
        rank_range=(0, 3),
        class_name="cat",
        group="cartoon",
        truth="ood",
    )
    assert fig is not None
    plt.close(fig)


def test_rank_grid_filter_class_name_by_string_requires_class_names():
    bank = ScoreBank(class_labels=[0, 1, 0])
    bank.add("MSP", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    with pytest.raises(ValueError, match="class_names"):
        rank_grid(bank, "MSP", class_name="cat")


def test_rank_grid_empty_filter_raises():
    bank = _toy_bank()
    with pytest.raises(ValueError, match="No samples match"):
        rank_grid(bank, "ViM", class_name="cat", group="tattoo", truth="id")


def test_rank_grid_image_mode_with_custom_loader():
    bank = _toy_bank(n=10)

    class _Loader:
        def __getitem__(self, idx):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    fig = rank_grid(
        bank, "Energy", images=_Loader(),
        rank_range=(0, 4), direction="ood",
    )
    assert fig is not None
    plt.close(fig)
