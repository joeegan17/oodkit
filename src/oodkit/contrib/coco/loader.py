"""
Parse a COCO-format JSON into :class:`oodkit.data.chip_dataset.ChipImageAnn`.

Pure Python / NumPy — no torch. Keeps responsibilities clean: this module
understands the COCO JSON schema (images + annotations + categories) and
produces per-image records in oodkit's canonical box format (``xyxy``). The
:class:`ChipDataset` then handles image decoding and chip cropping.

COCO native bbox format is ``xywh`` (top-left + width/height); this loader
converts to ``xyxy`` upfront so callers never have to think about it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from oodkit.contrib.coco.category_table import CocoCategoryTable
from oodkit.data.chip_dataset import ChipImageAnn
from oodkit.data.chips import to_xyxy


def load_coco(
    ann_path: Union[str, Path],
    image_root: Union[str, Path],
    *,
    group: Optional[str] = None,
    category_table: Optional[CocoCategoryTable] = None,
    include_empty_images: bool = False,
    min_box_side: float = 0.0,
) -> List[ChipImageAnn]:
    """Parse a COCO JSON into :class:`ChipImageAnn` records.

    Args:
        ann_path: Path to ``instances_*.json``.
        image_root: Directory containing the image files referenced by
            ``images[].file_name``. Each annotation's ``image_path`` is built as
            ``image_root / file_name``.
        group: Optional group tag applied to every resulting annotation. Useful
            for COCO-O domains (e.g. ``"cartoon"``) so downstream analysis can
            split results by domain via ``EmbeddingResult.metadata["group"]``.
        category_table: Contiguous-label table. When ``None``, one is built from
            the JSON's ``categories`` list. Passing a shared table across splits
            keeps label spaces aligned between COCO-train, COCO-val, and COCO-O.
        include_empty_images: If ``True``, images with no annotations are kept
            with an empty ``boxes`` array and dropped later by
            :class:`ChipDataset` (which requires at least one box). Default is
            ``False`` so they are filtered here.
        min_box_side: Discard boxes whose width or height (in ``xyxy``) is below
            this many pixels. ``0`` keeps all boxes.

    Returns:
        List of :class:`ChipImageAnn` in canonical ``xyxy`` box format, ready
        for :class:`ChipDataset` with ``box_format="xyxy"``.
    """
    p = Path(ann_path)
    if not p.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {p}")
    image_root_p = Path(image_root)
    if not image_root_p.is_dir():
        raise FileNotFoundError(f"COCO image root is not a directory: {image_root_p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if category_table is None:
        category_table = CocoCategoryTable.from_categories(data["categories"])

    images: List[Mapping[str, Any]] = list(data.get("images", []))
    id_to_image: Dict[int, Mapping[str, Any]] = {int(img["id"]): img for img in images}

    ann_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        ann_by_image[int(ann["image_id"])].append(ann)

    out: List[ChipImageAnn] = []
    for image_id, img in id_to_image.items():
        raw_anns = ann_by_image.get(image_id, [])
        if not raw_anns and not include_empty_images:
            continue

        file_name = str(img["file_name"])
        image_path = str(image_root_p / file_name)
        image_id_str = Path(file_name).stem

        boxes_xywh: List[List[float]] = []
        labels: List[int] = []
        for a in raw_anns:
            bbox = a.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            boxes_xywh.append([float(v) for v in bbox])
            labels.append(category_table.category_id_to_idx[int(a["category_id"])])

        if not boxes_xywh and not include_empty_images:
            continue

        boxes_arr = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
        if boxes_arr.size:
            boxes_arr = to_xyxy(boxes_arr, fmt="xywh")
        labels_arr: Optional[np.ndarray] = (
            np.asarray(labels, dtype=np.int64) if labels else None
        )

        if min_box_side > 0 and boxes_arr.size:
            w = boxes_arr[:, 2] - boxes_arr[:, 0]
            h = boxes_arr[:, 3] - boxes_arr[:, 1]
            keep = (w >= min_box_side) & (h >= min_box_side)
            if not np.any(keep) and not include_empty_images:
                continue
            boxes_arr = boxes_arr[keep]
            if labels_arr is not None:
                labels_arr = labels_arr[keep]

        out.append(
            ChipImageAnn(
                image_path=image_path,
                boxes=boxes_arr if boxes_arr.size else np.zeros((0, 4), dtype=np.float64),
                labels=labels_arr,
                group=group,
                image_id=image_id_str,
            )
        )

    return out


def collect_category_tables(
    ann_paths: Sequence[Union[str, Path]],
) -> CocoCategoryTable:
    """Build a single :class:`CocoCategoryTable` from the union of several COCO JSONs.

    Useful when ID (train/val) and OOD (COCO-O) splits all follow the same 80-class
    schema but have been produced by different tools. The first JSON's categories
    are taken as-is and later JSONs must be consistent (same ``id → name``).
    """
    if not ann_paths:
        raise ValueError("ann_paths is empty")
    tables = [CocoCategoryTable.from_coco_json(p) for p in ann_paths]
    ref = tables[0]
    for other in tables[1:]:
        if other.idx_to_category_id != ref.idx_to_category_id or other.idx_to_name != ref.idx_to_name:
            raise ValueError(
                "Incompatible COCO category tables across annotation files; "
                "pass matching schemas or build a single table manually."
            )
    return ref


__all__ = ["collect_category_tables", "load_coco"]
