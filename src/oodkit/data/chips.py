"""
Bounding box utilities and square chipping for object-detection OOD workflows.

Pure NumPy. No torch, no PIL. Downstream phases (``ChipDataset``, COCO contrib,
pooling) build on these primitives.

Chipping rule:

- Every chip is **square**.
- Side = ``max(max(box_w, box_h), min_chip_size)`` (default ``min_chip_size=25``).
- Centered on the box center; objects are never stretched or squeezed.
- Regions that extend past the image edge are **zero-padded** at crop time.
"""

from typing import Tuple

import numpy as np

_SUPPORTED_FORMATS = ("xyxy", "xywh", "cxcywh")


def to_xyxy(boxes: np.ndarray, fmt: str = "xyxy") -> np.ndarray:
    """Convert a box array to canonical ``xyxy`` format.

    Args:
        boxes: Shape ``(N, 4)`` numeric array (any float or int dtype).
        fmt: Input format, one of ``"xyxy"``, ``"xywh"`` (COCO; top-left + w/h),
            or ``"cxcywh"`` (center + w/h).

    Returns:
        ``float64`` array of shape ``(N, 4)`` in ``xyxy`` order.

    Raises:
        ValueError: If ``fmt`` is not supported or ``boxes`` is not ``(N, 4)``.
    """
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"fmt must be one of {_SUPPORTED_FORMATS}, got {fmt!r}"
        )
    b = np.asarray(boxes, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {b.shape}")

    if fmt == "xyxy":
        return b.copy()

    if fmt == "xywh":
        x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1)

    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)


def filter_small_boxes(
    boxes_xyxy: np.ndarray,
    min_side: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop boxes whose longest side is strictly less than ``min_side``.

    Args:
        boxes_xyxy: Shape ``(N, 4)`` in ``xyxy`` format.
        min_side: Threshold on the longest side (``max(w, h)``) in pixels.

    Returns:
        A tuple ``(filtered_boxes, kept_idx)`` where ``filtered_boxes`` has
        shape ``(K, 4)`` and ``kept_idx`` has shape ``(K,)`` ``int64`` with the
        original indices of retained boxes.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape (N, 4), got {b.shape}")

    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    longest = np.maximum(w, h)
    mask = longest >= float(min_side)
    kept_idx = np.nonzero(mask)[0].astype(np.int64)
    return b[mask], kept_idx


def square_chip_regions(
    boxes_xyxy: np.ndarray,
    min_chip_size: int = 25,
) -> np.ndarray:
    """Integer xyxy regions of the square chip for each input box.

    Each chip is square with side ``max(longest_box_side, min_chip_size)``,
    centered on the box center. Regions are returned in pixel coordinates and
    may extend past image boundaries (no clipping here; ``crop_chip`` handles
    edges).

    Args:
        boxes_xyxy: Shape ``(N, 4)`` float or int array in ``xyxy`` format.
        min_chip_size: Minimum side length of the resulting chip, in pixels.

    Returns:
        ``int64`` array of shape ``(N, 4)`` in ``xyxy`` order. For each row,
        ``x2 - x1 == y2 - y1`` exactly.

    Raises:
        ValueError: If ``min_chip_size`` is not a positive integer or shape is wrong.
    """
    if int(min_chip_size) < 1:
        raise ValueError(f"min_chip_size must be >= 1, got {min_chip_size}")
    b = np.asarray(boxes_xyxy, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape (N, 4), got {b.shape}")

    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    longest = np.maximum(w, h)
    side_float = np.maximum(longest, float(min_chip_size))
    side = np.ceil(side_float).astype(np.int64)

    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    x1 = np.floor(cx - side_float / 2.0).astype(np.int64)
    y1 = np.floor(cy - side_float / 2.0).astype(np.int64)
    x2 = x1 + side
    y2 = y1 + side
    return np.stack([x1, y1, x2, y2], axis=1)


def crop_chip(
    image: np.ndarray,
    box_xyxy: np.ndarray,
    min_chip_size: int = 25,
    fill: int = 0,
) -> np.ndarray:
    """Crop a single square chip from an image array.

    The chip is determined by :func:`square_chip_regions` applied to a single
    box; out-of-bounds pixels are filled with ``fill`` (default ``0``).

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` array.
        box_xyxy: Shape ``(4,)`` in ``xyxy`` format.
        min_chip_size: Minimum chip side (forwarded to ``square_chip_regions``).
        fill: Scalar pad value for out-of-bounds regions (applied to all channels).

    Returns:
        Chip array of shape ``(S, S)`` or ``(S, S, C)`` with the same dtype as
        ``image``, where ``S >= min_chip_size``.

    Raises:
        ValueError: If ``image`` or ``box_xyxy`` shapes are invalid.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"image must be (H, W) or (H, W, C), got shape {img.shape}")

    box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
    if box.shape != (4,):
        raise ValueError(f"box_xyxy must have shape (4,), got {box.shape}")

    region = square_chip_regions(box[None, :], min_chip_size=min_chip_size)[0]
    x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
    side = x2 - x1

    H, W = img.shape[0], img.shape[1]

    src_x1 = max(x1, 0)
    src_y1 = max(y1, 0)
    src_x2 = min(x2, W)
    src_y2 = min(y2, H)

    out_shape = (side, side) if img.ndim == 2 else (side, side, img.shape[2])
    chip = np.full(out_shape, fill, dtype=img.dtype)

    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        chip[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    return chip


def crop_chips(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    min_chip_size: int = 25,
    fill: int = 0,
) -> list:
    """Crop multiple square chips from a single image.

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` array.
        boxes_xyxy: Shape ``(N, 4)`` in ``xyxy`` format.
        min_chip_size: Minimum chip side.
        fill: Scalar pad value for out-of-bounds regions.

    Returns:
        List of ``N`` chip arrays. Sizes may differ per box; each chip side is
        ``max(longest_box_side, min_chip_size)`` pixels.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape (N, 4), got {boxes.shape}")
    return [
        crop_chip(image, boxes[i], min_chip_size=min_chip_size, fill=fill)
        for i in range(boxes.shape[0])
    ]
