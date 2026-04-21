"""Tests for ``oodkit.contrib.coco.loader`` (no torch)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from oodkit.contrib.coco.category_table import CocoCategoryTable
from oodkit.contrib.coco.loader import collect_category_tables, load_coco


def _write_coco_json(
    path: Path,
    *,
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
        )
    )


def _default_coco(tmp_path: Path) -> Path:
    ann_path = tmp_path / "instances.json"
    _write_coco_json(
        ann_path,
        images=[
            {"id": 1, "file_name": "00001.jpg"},
            {"id": 2, "file_name": "00002.jpg"},
            {"id": 3, "file_name": "empty.jpg"},
        ],
        annotations=[
            {"id": 10, "image_id": 1, "category_id": 17, "bbox": [0, 0, 20, 30]},
            {"id": 11, "image_id": 1, "category_id": 18, "bbox": [5, 5, 10, 10]},
            {
                "id": 12,
                "image_id": 2,
                "category_id": 17,
                "bbox": [1, 1, 8, 8],
                "iscrowd": 1,
            },
            {"id": 13, "image_id": 2, "category_id": 18, "bbox": [2, 2, 3, 3]},
        ],
        categories=[
            {"id": 17, "name": "cat"},
            {"id": 18, "name": "dog"},
        ],
    )
    (tmp_path / "images").mkdir()
    for fname in ("00001.jpg", "00002.jpg", "empty.jpg"):
        (tmp_path / "images" / fname).write_bytes(b"")
    return ann_path


def test_load_coco_basic(tmp_path: Path):
    ann_path = _default_coco(tmp_path)
    anns = load_coco(ann_path, tmp_path / "images")
    assert len(anns) == 2
    a1, a2 = anns
    assert a1.image_id == "00001"
    assert a1.boxes.shape == (2, 4)
    np.testing.assert_array_equal(a1.boxes[0], [0, 0, 20, 30])
    np.testing.assert_array_equal(a1.boxes[1], [5, 5, 15, 15])
    assert a1.labels is not None
    np.testing.assert_array_equal(a1.labels, [0, 1])
    assert a2.boxes.shape == (1, 4)
    np.testing.assert_array_equal(a2.labels, [1])


def test_load_coco_applies_group(tmp_path: Path):
    ann_path = _default_coco(tmp_path)
    anns = load_coco(ann_path, tmp_path / "images", group="cartoon")
    assert all(a.group == "cartoon" for a in anns)


def test_load_coco_include_empty_images(tmp_path: Path):
    ann_path = _default_coco(tmp_path)
    anns = load_coco(ann_path, tmp_path / "images", include_empty_images=True)
    ids = sorted(a.image_id for a in anns)
    assert ids == ["00001", "00002", "empty"]
    empty_ann = next(a for a in anns if a.image_id == "empty")
    assert empty_ann.boxes.shape == (0, 4)


def test_load_coco_shared_category_table(tmp_path: Path):
    ann_path = _default_coco(tmp_path)
    table = CocoCategoryTable.from_categories(
        [{"id": 17, "name": "cat"}, {"id": 18, "name": "dog"}]
    )
    anns = load_coco(ann_path, tmp_path / "images", category_table=table)
    assert anns[0].labels is not None
    assert set(anns[0].labels.tolist()) <= {0, 1}


def test_load_coco_min_box_side(tmp_path: Path):
    ann_path = _default_coco(tmp_path)
    anns = load_coco(ann_path, tmp_path / "images", min_box_side=5)
    per_image = {a.image_id: a for a in anns}
    assert "00002" not in per_image
    a1 = per_image["00001"]
    assert a1.boxes.shape == (2, 4)


def test_load_coco_missing_annotations_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_coco(tmp_path / "nope.json", tmp_path)


def test_load_coco_missing_image_root_raises(tmp_path: Path):
    ann_path = _default_coco(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_coco(ann_path, tmp_path / "does_not_exist")


def test_collect_category_tables_requires_consistency(tmp_path: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"categories": [{"id": 1, "name": "x"}]}))
    b.write_text(json.dumps({"categories": [{"id": 1, "name": "y"}]}))
    with pytest.raises(ValueError, match="Incompatible"):
        collect_category_tables([a, b])


def test_collect_category_tables_happy(tmp_path: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    cats = [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}]
    a.write_text(json.dumps({"categories": cats}))
    b.write_text(json.dumps({"categories": cats}))
    table = collect_category_tables([a, b])
    assert table.names() == ["x", "y"]
