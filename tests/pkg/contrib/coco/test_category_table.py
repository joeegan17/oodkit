"""Tests for ``oodkit.contrib.coco.category_table`` (no torch)."""

import json
from pathlib import Path

import pytest

from oodkit.contrib.coco.category_table import CocoCategoryTable


def test_from_categories_sorts_and_remaps():
    cats = [
        {"id": 18, "name": "dog"},
        {"id": 1, "name": "person"},
        {"id": 17, "name": "cat"},
    ]
    t = CocoCategoryTable.from_categories(cats)
    assert t.num_classes == 3
    assert t.idx_to_category_id == (1, 17, 18)
    assert t.idx_to_name == ("person", "cat", "dog")
    assert t.category_id_to_idx == {1: 0, 17: 1, 18: 2}


def test_map_category_ids():
    t = CocoCategoryTable.from_categories(
        [{"id": 1, "name": "a"}, {"id": 5, "name": "b"}]
    )
    assert t.map_category_ids([5, 1, 5]) == [1, 0, 1]


def test_map_unknown_category_id_raises():
    t = CocoCategoryTable.from_categories([{"id": 1, "name": "a"}])
    with pytest.raises(KeyError):
        t.map_category_ids([42])


def test_from_categories_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        CocoCategoryTable.from_categories([])


def test_from_categories_conflicting_name_raises():
    with pytest.raises(ValueError, match="two names"):
        CocoCategoryTable.from_categories(
            [{"id": 1, "name": "a"}, {"id": 1, "name": "b"}]
        )


def test_from_coco_json(tmp_path: Path):
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 3, "name": "c"},
                    {"id": 1, "name": "a"},
                ]
            }
        )
    )
    t = CocoCategoryTable.from_coco_json(p)
    assert t.names() == ["a", "c"]


def test_from_coco_json_missing_categories_raises(tmp_path: Path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"images": []}))
    with pytest.raises(ValueError, match="categories"):
        CocoCategoryTable.from_coco_json(p)


def test_from_coco_json_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CocoCategoryTable.from_coco_json(tmp_path / "missing.json")
