"""Tests for ``oodkit.contrib.coco.discovery`` (no torch)."""

from pathlib import Path

import pytest

from oodkit.contrib.coco.discovery import discover_coco_id, discover_coco_ood


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _build_id_layout(root: Path) -> None:
    _touch(root / "coco_annotations" / "instances_train2017.json")
    _touch(root / "coco_annotations" / "instances_val2017.json")
    (root / "coco_train").mkdir(parents=True, exist_ok=True)
    (root / "coco_val").mkdir(parents=True, exist_ok=True)


def _build_ood_domain(root: Path, name: str) -> None:
    _touch(root / name / "annotations" / "instances_val2017.json")
    (root / name / "images").mkdir(parents=True, exist_ok=True)


def test_discover_coco_id_happy(tmp_path: Path):
    _build_id_layout(tmp_path)
    paths = discover_coco_id(tmp_path)
    assert paths.train_ann.name == "instances_train2017.json"
    assert paths.val_images.is_dir()
    as_pairs = paths.as_pairs()
    assert set(as_pairs) == {"train", "val"}


def test_discover_coco_id_missing_annotation_raises(tmp_path: Path):
    _build_id_layout(tmp_path)
    (tmp_path / "coco_annotations" / "instances_val2017.json").unlink()
    with pytest.raises(FileNotFoundError, match="instances_val2017"):
        discover_coco_id(tmp_path)


def test_discover_coco_id_missing_images_raises(tmp_path: Path):
    _build_id_layout(tmp_path)
    # remove val images dir
    import shutil

    shutil.rmtree(tmp_path / "coco_val")
    with pytest.raises(FileNotFoundError, match="image directory"):
        discover_coco_id(tmp_path)


def test_discover_coco_ood_discovers_all(tmp_path: Path):
    _build_ood_domain(tmp_path, "cartoon")
    _build_ood_domain(tmp_path, "tattoo")
    (tmp_path / "incomplete").mkdir()
    out = discover_coco_ood(tmp_path)
    names = [d.name for d in out]
    assert names == ["cartoon", "tattoo"]


def test_discover_coco_ood_only_preserves_order_and_raises_on_missing(tmp_path: Path):
    _build_ood_domain(tmp_path, "cartoon")
    _build_ood_domain(tmp_path, "tattoo")
    out = discover_coco_ood(tmp_path, only=["tattoo", "cartoon"])
    assert [d.name for d in out] == ["tattoo", "cartoon"]

    with pytest.raises(FileNotFoundError, match="weather"):
        discover_coco_ood(tmp_path, only=["weather"])


def test_discover_coco_ood_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_coco_ood(tmp_path / "nope")
