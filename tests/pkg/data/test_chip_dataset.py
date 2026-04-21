"""Tests for ``oodkit.data.chip_dataset.ChipDataset`` (requires torch + PIL)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from oodkit.data.chip_dataset import (
    ChipDataset,
    ChipImageAnn,
    make_chip_annotations,
)


class _FakeProcessor:
    """Records chip sizes and returns a deterministic tensor."""

    def __init__(self) -> None:
        self.seen_sizes: list = []

    def __call__(self, images=None, return_tensors=None):
        del return_tensors
        if images is not None and hasattr(images, "size"):
            self.seen_sizes.append(images.size)
        return {"pixel_values": torch.ones(1, 3, 4, 4, dtype=torch.float32)}


def _save_image(path: Path, color=(10, 20, 30), size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def test_chip_dataset_labeled_flow(tmp_path: Path):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _save_image(img_a)
    _save_image(img_b)

    anns = [
        ChipImageAnn(
            image_path=str(img_a),
            boxes=np.array([[0, 0, 20, 20], [10, 10, 30, 30], [40, 40, 60, 60]]),
            labels=np.array([1, 2, 3]),
        ),
        ChipImageAnn(
            image_path=str(img_b),
            boxes=np.array([[5, 5, 25, 25]]),
            labels=np.array([7]),
        ),
    ]
    proc = _FakeProcessor()
    ds = ChipDataset(anns, proc, min_chip_size=25)

    assert len(ds) == 4
    assert ds.image_paths == [str(img_a), str(img_b)]
    np.testing.assert_array_equal(ds.chip_to_image, [0, 0, 0, 1])
    assert ds.boxes.shape == (4, 4)
    assert ds.labels is not None
    np.testing.assert_array_equal(ds.labels, [1, 2, 3, 7])

    assert len(ds.imgs) == 4
    assert ds.imgs[0] == (str(img_a), 1)
    assert ds.imgs[-1] == (str(img_b), 7)
    assert ds.targets == [1, 2, 3, 7]

    pixel_values, label = ds[0]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (3, 4, 4)
    assert label == 1

    assert proc.seen_sizes[0] == (25, 25)


def test_chip_dataset_unlabeled_returns_tensor_only(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)

    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0, 0, 20, 20], [10, 10, 30, 30]]),
        )
    ]
    proc = _FakeProcessor()
    ds = ChipDataset(anns, proc, min_chip_size=25)

    assert len(ds) == 2
    assert ds.labels is None
    assert ds.imgs[0][1] == -1

    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 4, 4)


def test_chip_dataset_rejects_mixed_label_presence(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)

    anns = [
        ChipImageAnn(image_path=str(img), boxes=np.zeros((1, 4)), labels=np.array([0])),
        ChipImageAnn(image_path=str(img), boxes=np.zeros((1, 4))),
    ]
    with pytest.raises(ValueError, match="All annotations"):
        ChipDataset(anns, _FakeProcessor())


def test_chip_dataset_label_length_mismatch_raises(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.zeros((2, 4)),
            labels=np.array([0]),
        )
    ]
    with pytest.raises(ValueError, match="labels has shape"):
        ChipDataset(anns, _FakeProcessor())


def test_chip_dataset_requires_annotations():
    with pytest.raises(ValueError, match="at least one"):
        ChipDataset([], _FakeProcessor())


def test_chip_dataset_requires_boxes(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    anns = [ChipImageAnn(image_path=str(img), boxes=np.zeros((0, 4)))]
    with pytest.raises(ValueError, match="zero boxes"):
        ChipDataset(anns, _FakeProcessor())


def test_chip_dataset_converts_xywh_boxes(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[10.0, 20.0, 4.0, 6.0]]),
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor(), box_format="xywh")
    np.testing.assert_array_equal(ds.boxes, [[10.0, 20.0, 14.0, 26.0]])


def test_chip_dataset_min_chip_size_sets_processor_size(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img, size=(128, 128))
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[10.0, 10.0, 50.0, 20.0]]),
        )
    ]
    proc = _FakeProcessor()
    ds = ChipDataset(anns, proc, min_chip_size=25)
    _ = ds[0]
    assert proc.seen_sizes[0] == (40, 40)


def test_chip_dataset_sample_descriptor(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            labels=np.array([5]),
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor())
    d = ds.sample_descriptor(0)
    assert d["image_index"] == 0
    assert d["image_path"] == str(img)
    assert d["label"] == 5
    assert len(d["box_xyxy"]) == 4


def test_make_chip_annotations_from_dicts(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    records = [
        {
            "image_path": str(img),
            "boxes": [[0, 0, 10, 10], [5, 5, 15, 15]],
            "labels": [1, 2],
        }
    ]
    anns = make_chip_annotations(records)
    assert len(anns) == 1
    assert isinstance(anns[0], ChipImageAnn)
    assert anns[0].labels is not None
    np.testing.assert_array_equal(anns[0].labels, [1, 2])


def test_make_chip_annotations_without_labels(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    records = [{"image_path": str(img), "boxes": [[0, 0, 10, 10]]}]
    anns = make_chip_annotations(records)
    assert anns[0].labels is None


def test_chip_dataset_image_ids_default_to_stem(tmp_path: Path):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _save_image(img_a)
    _save_image(img_b)
    anns = [
        ChipImageAnn(image_path=str(img_a), boxes=np.array([[0, 0, 10, 10]])),
        ChipImageAnn(image_path=str(img_b), boxes=np.array([[0, 0, 10, 10]])),
    ]
    ds = ChipDataset(anns, _FakeProcessor())
    assert list(ds.image_ids) == ["a", "b"]
    assert ds.groups is None


def test_chip_dataset_explicit_image_id_is_respected(tmp_path: Path):
    img = tmp_path / "00067.jpg"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0, 0, 10, 10]]),
            image_id="custom-id",
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor())
    assert list(ds.image_ids) == ["custom-id"]


def test_chip_dataset_groups_propagate(tmp_path: Path):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _save_image(img_a)
    _save_image(img_b)
    anns = [
        ChipImageAnn(
            image_path=str(img_a),
            boxes=np.array([[0, 0, 10, 10], [5, 5, 20, 20]]),
            group="cartoon",
        ),
        ChipImageAnn(
            image_path=str(img_b),
            boxes=np.array([[0, 0, 10, 10]]),
            group="tattoo",
        ),
    ]
    ds = ChipDataset(anns, _FakeProcessor())
    assert ds.groups is not None
    assert list(ds.groups) == ["cartoon", "cartoon", "tattoo"]


def test_chip_dataset_rejects_mixed_group_presence(tmp_path: Path):
    img = tmp_path / "a.png"
    _save_image(img)
    anns = [
        ChipImageAnn(image_path=str(img), boxes=np.zeros((1, 4)), group="cartoon"),
        ChipImageAnn(image_path=str(img), boxes=np.zeros((1, 4))),
    ]
    with pytest.raises(ValueError, match="group"):
        ChipDataset(anns, _FakeProcessor())


def test_chip_dataset_object_ids_unlabeled(tmp_path: Path):
    img = tmp_path / "00067.jpg"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]]),
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor())
    assert list(ds.object_ids) == ["00067_0", "00067_1"]


def test_chip_dataset_object_ids_labeled_with_class_names(tmp_path: Path):
    img = tmp_path / "00067.jpg"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array(
                [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]]
            ),
            labels=np.array([0, 0, 1]),
        )
    ]
    ds = ChipDataset(
        anns,
        _FakeProcessor(),
        class_names=["cat", "dog"],
    )
    assert list(ds.object_ids) == [
        "00067_cat_0",
        "00067_cat_1",
        "00067_dog_0",
    ]


def test_chip_dataset_object_ids_with_group_and_class(tmp_path: Path):
    img = tmp_path / "00067.jpg"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]]),
            labels=np.array([0, 0]),
            group="cartoon",
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor(), class_names=["cat"])
    assert list(ds.object_ids) == [
        "00067_cat_cartoon_0",
        "00067_cat_cartoon_1",
    ]


def test_chip_dataset_object_ids_without_class_names_stringifies_label(tmp_path: Path):
    img = tmp_path / "00067.jpg"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0, 0, 10, 10]]),
            labels=np.array([7]),
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor())
    assert list(ds.object_ids) == ["00067_7_0"]


def test_chip_dataset_object_ids_counter_is_per_image(tmp_path: Path):
    img_a = tmp_path / "00001.jpg"
    img_b = tmp_path / "00002.jpg"
    _save_image(img_a)
    _save_image(img_b)
    anns = [
        ChipImageAnn(
            image_path=str(img_a),
            boxes=np.array([[0, 0, 10, 10]]),
            labels=np.array([0]),
        ),
        ChipImageAnn(
            image_path=str(img_b),
            boxes=np.array([[0, 0, 10, 10]]),
            labels=np.array([0]),
        ),
    ]
    ds = ChipDataset(anns, _FakeProcessor(), class_names=["cat"])
    assert list(ds.object_ids) == ["00001_cat_0", "00002_cat_0"]


def test_chip_dataset_sample_descriptor_includes_group_and_object_id(tmp_path: Path):
    img = tmp_path / "00067.jpg"
    _save_image(img)
    anns = [
        ChipImageAnn(
            image_path=str(img),
            boxes=np.array([[0, 0, 10, 10]]),
            labels=np.array([0]),
            group="cartoon",
        )
    ]
    ds = ChipDataset(anns, _FakeProcessor(), class_names=["cat"])
    d = ds.sample_descriptor(0)
    assert d["image_id"] == "00067"
    assert d["object_id"] == "00067_cat_cartoon_0"
    assert d["group"] == "cartoon"
    assert d["class_name"] == "cat"
