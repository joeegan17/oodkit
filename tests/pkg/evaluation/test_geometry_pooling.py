"""Tests for geometry-aware object-detection image pooling."""

import numpy as np
import pytest

from oodkit.evaluation import GeometryAwarePooler


def _id_scenes():
    scores = []
    c2i = []
    boxes = []
    labels = []
    sizes = []
    for image_id in range(4):
        scores.extend([0.0, 0.0])
        c2i.extend([image_id, image_id])
        boxes.extend([[10, 10, 30, 30], [60, 10, 80, 30]])
        labels.extend([0, 1])
        sizes.extend([[100, 100], [100, 100]])
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(c2i, dtype=np.int64),
        np.asarray(boxes, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sizes, dtype=np.float64),
    )


def _fit_pooler(**kwargs):
    pooler = GeometryAwarePooler(**kwargs)
    pooler.fit(*_id_scenes())
    return pooler


def _score_one(pooler, scores, boxes, labels, *, n_images=1):
    c2i = np.zeros(len(scores), dtype=np.int64)
    sizes = np.tile(np.array([[100, 100]], dtype=np.float64), (len(scores), 1))
    final, comp = pooler.score(
        np.asarray(scores, dtype=np.float64),
        c2i,
        np.asarray(boxes, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        sizes,
        n_images=n_images,
        return_components=True,
    )
    return final, comp


def test_geometry_pooler_basic_fit_score():
    pooler = _fit_pooler()
    final, comp = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [60, 10, 80, 30]],
        [0, 1],
    )
    assert final.shape == (1,)
    assert np.isfinite(final[0])
    assert set(comp) == {
        "node",
        "size",
        "layout",
        "cooccurrence",
        "interaction",
        "final",
    }


def test_unusual_object_size_raises_size_component():
    pooler = _fit_pooler()
    _, normal = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [60, 10, 80, 30]],
        [0, 1],
    )
    _, unusual = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 70, 70], [60, 10, 80, 30]],
        [0, 1],
    )
    assert unusual["size"][0] > normal["size"][0]


def test_unusual_layout_raises_layout_component():
    pooler = _fit_pooler()
    _, normal = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [60, 10, 80, 30]],
        [0, 1],
    )
    _, unusual = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [10, 70, 30, 90]],
        [0, 1],
    )
    assert unusual["layout"][0] > normal["layout"][0]


def test_rare_class_pair_raises_cooccurrence_component():
    pooler = _fit_pooler()
    _, normal = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [60, 10, 80, 30]],
        [0, 1],
    )
    _, rare = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [60, 10, 80, 30]],
        [0, 2],
    )
    assert rare["cooccurrence"][0] > normal["cooccurrence"][0]


def test_high_chip_score_with_unusual_relation_raises_interaction_component():
    pooler = _fit_pooler()
    _, normal = _score_one(
        pooler,
        [0.0, 0.0],
        [[10, 10, 30, 30], [60, 10, 80, 30]],
        [0, 1],
    )
    _, unusual = _score_one(
        pooler,
        [5.0, 0.0],
        [[10, 10, 30, 30], [10, 70, 30, 90]],
        [0, 1],
    )
    assert unusual["interaction"][0] > normal["interaction"][0]


def test_single_object_image_uses_node_and_size_only():
    pooler = _fit_pooler()
    final, comp = _score_one(pooler, [1.0], [[10, 10, 30, 30]], [0])
    assert np.isfinite(final[0])
    assert comp["layout"][0] == pytest.approx(0.0)
    assert comp["cooccurrence"][0] == pytest.approx(0.0)
    assert comp["interaction"][0] == pytest.approx(0.0)


def test_empty_images_return_nan():
    pooler = _fit_pooler()
    out = pooler.score(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
        np.zeros((0, 4), dtype=np.float64),
        np.array([], dtype=np.int64),
        np.zeros((0, 2), dtype=np.float64),
        n_images=2,
    )
    assert out.shape == (2,)
    assert np.all(np.isnan(out))


def test_invalid_inputs_raise_clear_errors():
    pooler = _fit_pooler()
    with pytest.raises(ValueError, match="image_sizes"):
        pooler.score(
            np.zeros(1),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 4), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 3), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="positive width"):
        pooler.score(
            np.zeros(1),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 4), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.ones((1, 2), dtype=np.float64),
        )


def test_pair_cap_is_honored():
    pooler = _fit_pooler(max_pairs=1)
    centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    pairs = pooler._select_pairs(np.arange(4), centers)
    assert len(pairs) <= 1


def test_geometry_pooler_reexport_from_evaluation():
    from oodkit.evaluation import GeometryAwarePooler as exported

    assert exported is GeometryAwarePooler
