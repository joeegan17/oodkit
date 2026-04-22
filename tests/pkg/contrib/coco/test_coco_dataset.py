"""Tests for ``oodkit.contrib.coco.dataset`` (requires torch + torchvision + PIL)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from oodkit.contrib.coco.category_table import CocoCategoryTable
from oodkit.contrib.coco.dataset import (
    coco_chip_dataset,
    coco_id_chip_datasets,
    coco_ood_chip_datasets,
)


class _FakeProcessor:
    def __call__(self, images=None, return_tensors=None):
        del images, return_tensors
        return {"pixel_values": torch.ones(1, 3, 4, 4, dtype=torch.float32)}


def _save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(path)


def _write_coco(
    ann_path: Path,
    images_dir: Path,
    *,
    file_names: List[str],
    annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
) -> None:
    images = [
        {"id": i + 1, "file_name": fname} for i, fname in enumerate(file_names)
    ]
    for fname in file_names:
        _save_image(images_dir / fname)
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
        )
    )


def test_coco_chip_dataset_end_to_end(tmp_path: Path):
    ann_path = tmp_path / "ann.json"
    images_dir = tmp_path / "images"
    _write_coco(
        ann_path,
        images_dir,
        file_names=["00001.jpg", "00002.jpg"],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 17, "bbox": [0, 0, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 17, "bbox": [5, 5, 10, 10]},
            {"id": 3, "image_id": 2, "category_id": 18, "bbox": [2, 2, 15, 15]},
        ],
        categories=[{"id": 17, "name": "cat"}, {"id": 18, "name": "dog"}],
    )

    ds = coco_chip_dataset(ann_path, images_dir, _FakeProcessor(), group="cartoon")
    assert len(ds) == 3
    assert ds.class_names == ["cat", "dog"]
    assert ds.groups is not None
    assert list(ds.groups) == ["cartoon", "cartoon", "cartoon"]
    assert list(ds.image_ids) == ["00001", "00001", "00002"]
    assert list(ds.object_ids) == [
        "00001_cat_cartoon_0",
        "00001_cat_cartoon_1",
        "00002_dog_cartoon_0",
    ]


def test_coco_chip_dataset_raises_when_all_images_empty(tmp_path: Path):
    ann_path = tmp_path / "ann.json"
    images_dir = tmp_path / "images"
    _write_coco(
        ann_path,
        images_dir,
        file_names=["00001.jpg"],
        annotations=[],
        categories=[{"id": 1, "name": "thing"}],
    )
    with pytest.raises(ValueError, match="No annotations usable"):
        coco_chip_dataset(ann_path, images_dir, _FakeProcessor())


def test_coco_id_chip_datasets(tmp_path: Path):
    train_ann = tmp_path / "coco_annotations" / "instances_train2017.json"
    val_ann = tmp_path / "coco_annotations" / "instances_val2017.json"
    _write_coco(
        train_ann,
        tmp_path / "coco_train",
        file_names=["t_00001.jpg"],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
        categories=[{"id": 1, "name": "x"}],
    )
    _write_coco(
        val_ann,
        tmp_path / "coco_val",
        file_names=["v_00001.jpg"],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
        categories=[{"id": 1, "name": "x"}],
    )

    out = coco_id_chip_datasets(tmp_path, _FakeProcessor())
    assert set(out) == {"train", "val"}
    assert out["train"].class_names == ["x"]
    assert out["val"].class_names == ["x"]


def test_coco_ood_chip_datasets_tags_per_domain(tmp_path: Path):
    table = CocoCategoryTable.from_categories(
        [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}]
    )
    for domain in ("cartoon", "tattoo"):
        _write_coco(
            tmp_path / domain / "annotations" / "instances_val2017.json",
            tmp_path / domain / "images",
            file_names=[f"{domain}_1.jpg"],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [0, 0, 10, 10],
                }
            ],
            categories=[{"id": 1, "name": "x"}, {"id": 2, "name": "y"}],
        )

    out = coco_ood_chip_datasets(tmp_path, _FakeProcessor(), category_table=table)
    assert set(out) == {"cartoon", "tattoo"}
    for name, ds in out.items():
        assert ds.groups is not None
        assert list(ds.groups) == [name]
        assert ds.object_ids[0] == f"{name}_1_y_{name}_0"
