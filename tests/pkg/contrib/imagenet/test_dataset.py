"""Tests for ``SynsetImageDataset`` (requires torch + torchvision)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from oodkit.contrib.imagenet.dataset import SynsetImageDataset, imagenet_variant_dataset
from oodkit.contrib.imagenet.synset_table import SynsetTable


class _FakeProcessor:
    """Minimal stand-in for a HuggingFace image processor."""

    def __call__(self, images=None, return_tensors=None):
        del return_tensors
        _ = images
        return {"pixel_values": torch.ones(1, 3, 2, 2, dtype=torch.float32)}


def _write_mapping(path: Path) -> None:
    path.write_text(
        "n01498041 stingray\n"
        "n01531178 goldfinch\n"
        "n01534433 junco\n",
        encoding="utf-8",
    )


def _save_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(path)


def test_synset_image_dataset_order_and_imgs(tmp_path: Path):
    root = tmp_path / "root"
    _save_dummy_image(root / "n01531178" / "a.png")
    _save_dummy_image(root / "n01498041" / "z.png")
    _save_dummy_image(root / "n01498041" / "b.png")

    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(m)
    table = SynsetTable.from_file(m)

    ds = SynsetImageDataset(root, table, _FakeProcessor(), strict=True)
    assert len(ds) == 3
    assert len(ds.imgs) == 3
    assert ds.targets == [lbl for _, lbl in ds.imgs]
    assert ds.classes[0] == "stingray"
    x, y = ds[0]
    assert x.shape == (3, 2, 2)
    assert y in (0, 1, 2)
    desc = ds.sample_descriptor(0)
    assert "image_id" in desc and "wnid" in desc and "canonical_idx" in desc


def test_synset_image_dataset_strict_unknown_folder(tmp_path: Path):
    root = tmp_path / "root"
    (root / "n01498041").mkdir(parents=True)
    _save_dummy_image(root / "n01498041" / "x.png")
    (root / "junk_folder").mkdir()

    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(m)
    table = SynsetTable.from_file(m)

    with pytest.raises(ValueError, match="Unknown synset"):
        SynsetImageDataset(root, table, _FakeProcessor(), strict=True)


def test_synset_image_dataset_lenient_skips_unknown(tmp_path: Path):
    root = tmp_path / "root"
    _save_dummy_image(root / "n01498041" / "x.png")
    (root / "junk_folder").mkdir()

    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(m)
    table = SynsetTable.from_file(m)

    with pytest.warns(UserWarning, match="Unknown synset"):
        ds = SynsetImageDataset(root, table, _FakeProcessor(), strict=False)
    assert len(ds) == 1


def test_imagenet_variant_dataset_factory(tmp_path: Path):
    root = tmp_path / "root"
    _save_dummy_image(root / "n01534433" / "only.png")
    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(m)

    ds = imagenet_variant_dataset(root, m, _FakeProcessor())
    assert len(ds) == 1
    _path, y = ds.imgs[0]
    assert y == 2


def test_synset_image_dataset_labels_match_canonical(tmp_path: Path):
    root = tmp_path / "root"
    _save_dummy_image(root / "n01531178" / "g.png")

    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(m)
    table = SynsetTable.from_file(m)

    ds = SynsetImageDataset(root, table, _FakeProcessor())
    assert ds.imgs[0][1] == 1


def test_synset_image_dataset_empty_raises(tmp_path: Path):
    root = tmp_path / "root"
    (root / "n01498041").mkdir(parents=True)
    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(m)
    table = SynsetTable.from_file(m)

    with pytest.raises(ValueError, match="No images"):
        SynsetImageDataset(root, table, _FakeProcessor())
