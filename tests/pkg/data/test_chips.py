"""Tests for ``oodkit.data.chips`` box utilities and square chipping."""

import numpy as np
import pytest

from oodkit.data.chips import (
    crop_chip,
    crop_chips,
    filter_small_boxes,
    square_chip_regions,
    to_xyxy,
)


def test_to_xyxy_passthrough():
    boxes = np.array([[1.0, 2.0, 5.0, 8.0]])
    out = to_xyxy(boxes, fmt="xyxy")
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, boxes)
    assert out is not boxes


def test_to_xyxy_from_xywh():
    boxes = np.array([[10.0, 20.0, 4.0, 6.0]])
    out = to_xyxy(boxes, fmt="xywh")
    np.testing.assert_array_equal(out, [[10.0, 20.0, 14.0, 26.0]])


def test_to_xyxy_from_cxcywh():
    boxes = np.array([[12.0, 23.0, 4.0, 6.0]])
    out = to_xyxy(boxes, fmt="cxcywh")
    np.testing.assert_array_equal(out, [[10.0, 20.0, 14.0, 26.0]])


def test_to_xyxy_invalid_fmt():
    with pytest.raises(ValueError, match="fmt must be one of"):
        to_xyxy(np.zeros((1, 4)), fmt="ltrb")


def test_to_xyxy_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        to_xyxy(np.zeros((1, 3)))


def test_filter_small_boxes_drops_short_longest_side():
    boxes = np.array(
        [
            [0.0, 0.0, 30.0, 30.0],
            [0.0, 0.0, 10.0, 40.0],
            [0.0, 0.0, 5.0, 5.0],
        ]
    )
    filtered, kept = filter_small_boxes(boxes, min_side=25.0)
    np.testing.assert_array_equal(kept, [0, 1])
    np.testing.assert_array_equal(filtered, boxes[[0, 1]])


def test_filter_small_boxes_empty_result():
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
    filtered, kept = filter_small_boxes(boxes, min_side=10.0)
    assert filtered.shape == (0, 4)
    assert kept.shape == (0,)


def test_square_chip_regions_equal_side_box():
    boxes = np.array([[10.0, 10.0, 30.0, 30.0]])
    regions = square_chip_regions(boxes, min_chip_size=1)
    np.testing.assert_array_equal(regions, [[10, 10, 30, 30]])


def test_square_chip_regions_rectangular_uses_longest_side():
    boxes = np.array([[10.0, 20.0, 50.0, 30.0]])
    regions = square_chip_regions(boxes, min_chip_size=1)
    side = regions[0, 2] - regions[0, 0]
    assert side == 40
    assert regions[0, 3] - regions[0, 1] == 40
    cx = (regions[0, 0] + regions[0, 2]) / 2.0
    cy = (regions[0, 1] + regions[0, 3]) / 2.0
    assert cx == pytest.approx(30.0, abs=1.0)
    assert cy == pytest.approx(25.0, abs=1.0)


def test_square_chip_regions_promotes_to_min_chip_size():
    boxes = np.array([[10.0, 10.0, 14.0, 14.0]])
    regions = square_chip_regions(boxes, min_chip_size=25)
    side = regions[0, 2] - regions[0, 0]
    assert side == 25
    assert regions[0, 3] - regions[0, 1] == 25


def test_square_chip_regions_allows_out_of_bounds():
    boxes = np.array([[0.0, 0.0, 4.0, 4.0]])
    regions = square_chip_regions(boxes, min_chip_size=25)
    x1, y1, x2, y2 = regions[0]
    assert x1 < 0 and y1 < 0
    assert x2 - x1 == 25 and y2 - y1 == 25


def test_square_chip_regions_invalid_min_size():
    with pytest.raises(ValueError, match="min_chip_size"):
        square_chip_regions(np.zeros((1, 4)), min_chip_size=0)


def _make_gradient_image(h: int, w: int, channels: int = 0) -> np.ndarray:
    base = np.arange(h * w, dtype=np.uint8).reshape(h, w) % 255
    if channels == 0:
        return base
    return np.stack([base, base, base][:channels], axis=-1)


def test_crop_chip_interior_rgb():
    img = _make_gradient_image(50, 50, channels=3)
    box = np.array([20.0, 20.0, 30.0, 30.0])
    chip = crop_chip(img, box, min_chip_size=10, fill=0)
    assert chip.shape == (10, 10, 3)
    assert chip.dtype == img.dtype
    np.testing.assert_array_equal(chip, img[20:30, 20:30])


def test_crop_chip_zero_pads_top_left_corner():
    img = np.ones((50, 50), dtype=np.uint8) * 7
    box = np.array([0.0, 0.0, 4.0, 4.0])
    chip = crop_chip(img, box, min_chip_size=25, fill=0)
    assert chip.shape == (25, 25)
    pad = np.zeros((25, 25), dtype=np.uint8)
    assert chip[0, 0] == 0
    assert chip[-1, -1] == 7
    assert (chip == 7).sum() + (chip == 0).sum() == chip.size
    assert (chip == 0).sum() > 0


def test_crop_chip_zero_pads_bottom_right_corner():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 9
    box = np.array([46.0, 46.0, 50.0, 50.0])
    chip = crop_chip(img, box, min_chip_size=25, fill=0)
    assert chip.shape == (25, 25, 3)
    assert (chip == 0).sum() > 0
    assert (chip == 9).sum() > 0


def test_crop_chip_grayscale():
    img = _make_gradient_image(40, 40)
    box = np.array([10.0, 10.0, 20.0, 20.0])
    chip = crop_chip(img, box, min_chip_size=10)
    assert chip.ndim == 2
    assert chip.shape == (10, 10)


def test_crop_chip_invalid_image_shape():
    with pytest.raises(ValueError, match="image"):
        crop_chip(np.zeros((3, 3, 3, 3)), np.array([0, 0, 1, 1]))


def test_crop_chips_batch():
    img = _make_gradient_image(50, 50, channels=3)
    boxes = np.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [30.0, 30.0, 36.0, 38.0],
        ]
    )
    chips = crop_chips(img, boxes, min_chip_size=10)
    assert len(chips) == 2
    assert chips[0].shape == (10, 10, 3)
    assert chips[1].shape == (10, 10, 3)
