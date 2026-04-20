"""Tests for ``oodkit.evaluation.pooling.pool_image_scores``."""

import numpy as np
import pytest

from oodkit.evaluation.pooling import pool_image_scores


def test_pool_mean_basic():
    scores = np.array([1.0, 3.0, 4.0, 2.0])
    c2i = np.array([0, 0, 1, 1])
    out = pool_image_scores(scores, c2i, method="mean")
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_pool_max_basic():
    scores = np.array([1.0, 3.0, 4.0, 2.0])
    c2i = np.array([0, 0, 1, 1])
    out = pool_image_scores(scores, c2i, method="max")
    np.testing.assert_allclose(out, [3.0, 4.0])


def test_pool_topk_mean_uses_top_k_per_image():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c2i = np.array([0, 0, 0, 1, 1])
    out = pool_image_scores(scores, c2i, method="topk_mean", k=2)
    np.testing.assert_allclose(out, [2.5, 4.5])


def test_pool_topk_mean_fewer_than_k_chips_uses_all():
    scores = np.array([1.0, 2.0, 5.0])
    c2i = np.array([0, 0, 1])
    out = pool_image_scores(scores, c2i, method="topk_mean", k=3)
    np.testing.assert_allclose(out, [1.5, 5.0])


def test_pool_unordered_chip_to_image():
    scores = np.array([1.0, 10.0, 2.0, 20.0])
    c2i = np.array([0, 1, 0, 1])
    out = pool_image_scores(scores, c2i, method="mean")
    np.testing.assert_allclose(out, [1.5, 15.0])


def test_pool_zero_chip_image_yields_nan():
    scores = np.array([1.0, 2.0, 3.0])
    c2i = np.array([0, 0, 2])
    out = pool_image_scores(scores, c2i, method="mean")
    assert out.shape == (3,)
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [1.5, 3.0])


def test_pool_n_images_override_appends_nan_rows():
    scores = np.array([1.0, 2.0])
    c2i = np.array([0, 0])
    out = pool_image_scores(scores, c2i, method="mean", n_images=3)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(1.5)
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_pool_n_images_too_small_raises():
    scores = np.array([1.0, 2.0])
    c2i = np.array([0, 1])
    with pytest.raises(ValueError, match="n_images"):
        pool_image_scores(scores, c2i, method="mean", n_images=1)


def test_pool_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        pool_image_scores(np.zeros(1), np.zeros(1, dtype=np.int64), method="sum")


def test_pool_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        pool_image_scores(np.zeros(2), np.zeros(3, dtype=np.int64))


def test_pool_negative_index_raises():
    with pytest.raises(ValueError, match="non-negative"):
        pool_image_scores(np.zeros(2), np.array([-1, 0]))


def test_pool_k_must_be_positive():
    with pytest.raises(ValueError, match="k"):
        pool_image_scores(np.zeros(1), np.zeros(1, dtype=np.int64), method="topk_mean", k=0)


def test_pool_empty_inputs_return_empty_array():
    out = pool_image_scores(np.array([]), np.array([], dtype=np.int64))
    assert out.shape == (0,)


def test_pool_empty_inputs_with_n_images_all_nan():
    out = pool_image_scores(
        np.array([]), np.array([], dtype=np.int64), n_images=3
    )
    assert out.shape == (3,)
    assert np.all(np.isnan(out))


def test_pool_reexport_from_evaluation():
    from oodkit.evaluation import pool_image_scores as exported

    assert exported is pool_image_scores
