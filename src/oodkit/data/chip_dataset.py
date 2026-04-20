"""
PyTorch ``Dataset`` that yields per-box square chips for object-detection OOD.

Wraps :mod:`oodkit.data.chips` with image decoding (PIL / torchvision) and a
HuggingFace image processor so chips can flow straight into
:meth:`oodkit.embeddings.embedder.Embedder.extract` /
:meth:`oodkit.embeddings.embedder.Embedder.fit`.

Requires the optional ML stack (``torch``, ``torchvision``) like
``oodkit.embeddings``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from oodkit.data.chips import crop_chip, to_xyxy

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision.datasets.folder import default_loader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ChipDataset requires torch and torchvision. "
        'Install with: pip install "oodkit[ml]"'
    ) from exc


@dataclass
class ChipImageAnn:
    """Per-image annotation consumed by :class:`ChipDataset`.

    Args:
        image_path: Path to the source image file. Decoded lazily via ``loader``.
        boxes: ``(K, 4)`` float array of boxes for this image, in the format
            specified by ``ChipDataset(box_format=...)``.
        labels: Optional ``(K,)`` integer array of per-box class labels. Either
            all annotations carry labels or none do; mixed is rejected.
    """

    image_path: str
    boxes: np.ndarray
    labels: Optional[np.ndarray] = None


class ChipDataset(Dataset):
    """Flatten per-image boxes into a stream of square chips.

    Each ``__getitem__`` decodes the parent image once, crops the square chip
    with :func:`oodkit.data.chips.crop_chip` (longest-side square, centered on
    the box, zero-padded at image edges, ``min_chip_size`` default 25), then
    passes the chip through ``processor`` so it can be consumed by any
    backbone-agnostic ``Embedder`` pipeline.

    Attributes:
        chip_to_image: ``(N_chips,)`` int64 array mapping each chip to its
            source-image index in ``image_paths``.
        boxes: ``(N_chips, 4)`` float64 array of boxes in canonical ``xyxy``
            format, aligned with the chip order (useful for Phase 4 pooling and
            visualisation).
        labels: ``(N_chips,)`` int64 array of per-chip class labels, or ``None``
            if the dataset is unlabeled.
        image_paths: List of parent image paths, length = number of unique
            source images.
        imgs: List of ``(parent_image_path, label)`` per chip. Mirrors the
            attribute ``torchvision.datasets.ImageFolder`` exposes so
            ``Embedder.extract`` automatically records per-chip parent paths in
            ``EmbeddingResult.metadata["image_paths"]``. When the dataset is
            unlabeled the second element is ``-1``.

    Sample format:
        - Labeled datasets yield ``(chip_tensor, int_label)``.
        - Unlabeled datasets yield ``chip_tensor`` alone.

        This matches the convention used by
        :class:`oodkit.contrib.imagenet.dataset.SynsetImageDataset` and is what
        ``Embedder._dataset_has_labels`` keys on.

    Args:
        annotations: Per-image annotations; see :class:`ChipImageAnn`.
        processor: HuggingFace image processor, called as
            ``processor(images=pil_img, return_tensors="pt")``.
        box_format: Incoming box format, forwarded to
            :func:`oodkit.data.chips.to_xyxy` (``"xyxy"``, ``"xywh"``, or
            ``"cxcywh"``).
        min_chip_size: Minimum chip side in pixels. Forwarded to
            :func:`oodkit.data.chips.crop_chip`.
        fill: Pad value for out-of-bounds regions of a chip (default ``0``).
        image_mode: PIL mode to convert loaded images to before cropping.
            ``"RGB"`` matches most vision backbones.
        loader: PIL loader. Defaults to ``torchvision.datasets.folder.default_loader``.
    """

    def __init__(
        self,
        annotations: Sequence[ChipImageAnn],
        processor: Any,
        *,
        box_format: str = "xyxy",
        min_chip_size: int = 25,
        fill: int = 0,
        image_mode: str = "RGB",
        loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        if len(annotations) == 0:
            raise ValueError("ChipDataset requires at least one ChipImageAnn")

        self.processor = processor
        self.box_format = box_format
        self.min_chip_size = int(min_chip_size)
        self.fill = int(fill)
        self.image_mode = image_mode
        self.loader = loader if loader is not None else default_loader

        has_labels_vec = [ann.labels is not None for ann in annotations]
        if any(has_labels_vec) and not all(has_labels_vec):
            raise ValueError(
                "All annotations must either carry labels or none may; "
                "mixed labeled/unlabeled annotations are not supported."
            )
        self._has_labels = bool(has_labels_vec[0])

        image_paths: List[str] = []
        boxes_per_image: List[np.ndarray] = []
        labels_per_image: List[np.ndarray] = []
        chip_to_image_chunks: List[np.ndarray] = []

        for img_idx, ann in enumerate(annotations):
            boxes_xyxy = to_xyxy(np.asarray(ann.boxes), fmt=box_format)
            n = boxes_xyxy.shape[0]
            image_paths.append(str(ann.image_path))
            boxes_per_image.append(boxes_xyxy)
            chip_to_image_chunks.append(
                np.full((n,), img_idx, dtype=np.int64)
            )
            if self._has_labels:
                lab = np.asarray(ann.labels, dtype=np.int64).reshape(-1)
                if lab.shape[0] != n:
                    raise ValueError(
                        f"annotation {img_idx}: labels has shape {lab.shape} "
                        f"but boxes has {n} rows"
                    )
                labels_per_image.append(lab)

        self.image_paths: List[str] = image_paths
        self.boxes: np.ndarray = (
            np.concatenate(boxes_per_image, axis=0)
            if boxes_per_image
            else np.zeros((0, 4), dtype=np.float64)
        )
        self.chip_to_image: np.ndarray = (
            np.concatenate(chip_to_image_chunks, axis=0)
            if chip_to_image_chunks
            else np.zeros((0,), dtype=np.int64)
        )
        self.labels: Optional[np.ndarray] = (
            np.concatenate(labels_per_image, axis=0)
            if self._has_labels and labels_per_image
            else None
        )

        if self.boxes.shape[0] == 0:
            raise ValueError("ChipDataset received zero boxes across all annotations")

        parent_paths = [self.image_paths[i] for i in self.chip_to_image.tolist()]
        if self._has_labels:
            assert self.labels is not None
            self.imgs: List[Tuple[str, int]] = list(
                zip(parent_paths, self.labels.tolist())
            )
            self.targets: List[int] = self.labels.tolist()
        else:
            self.imgs = [(p, -1) for p in parent_paths]

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        img_idx = int(self.chip_to_image[index])
        path = self.image_paths[img_idx]
        pil_img = self.loader(path)
        if self.image_mode is not None and pil_img.mode != self.image_mode:
            pil_img = pil_img.convert(self.image_mode)

        image_np = np.asarray(pil_img)
        chip_np = crop_chip(
            image_np,
            self.boxes[index],
            min_chip_size=self.min_chip_size,
            fill=self.fill,
        )
        chip_pil = _numpy_to_pil(chip_np)

        pixel_values = self.processor(
            images=chip_pil, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        if self._has_labels:
            assert self.labels is not None
            return pixel_values, int(self.labels[index])
        return pixel_values

    def sample_descriptor(self, index: int) -> dict:
        """Per-chip ids useful in notebooks / debugging."""
        img_idx = int(self.chip_to_image[index])
        out = {
            "index": int(index),
            "image_index": img_idx,
            "image_path": self.image_paths[img_idx],
            "box_xyxy": self.boxes[index].tolist(),
        }
        if self._has_labels:
            assert self.labels is not None
            out["label"] = int(self.labels[index])
        return out


def _numpy_to_pil(array: np.ndarray) -> Any:
    """NumPy → PIL conversion; dtype/mode are inferred by PIL."""
    from PIL import Image

    return Image.fromarray(array)


def make_chip_annotations(
    records: Sequence[Any],
    *,
    path_key: str = "image_path",
    boxes_key: str = "boxes",
    labels_key: str = "labels",
) -> List[ChipImageAnn]:
    """Small helper to build :class:`ChipImageAnn` list from dict-like records.

    Accepts either :class:`ChipImageAnn` instances (returned unchanged) or
    mappings with ``path_key`` / ``boxes_key`` / ``labels_key`` entries. Useful
    for constructing datasets from COCO-like records without forcing callers to
    import the dataclass.

    Args:
        records: Sequence of ``ChipImageAnn`` or mapping-like records.
        path_key: Key for the parent image path.
        boxes_key: Key for the per-image boxes array.
        labels_key: Key for the per-image labels array (optional per record).

    Returns:
        A list of :class:`ChipImageAnn` ready for :class:`ChipDataset`.
    """
    out: List[ChipImageAnn] = []
    for r in records:
        if isinstance(r, ChipImageAnn):
            out.append(r)
            continue
        if not hasattr(r, "__getitem__"):
            raise TypeError(
                f"record must be ChipImageAnn or mapping, got {type(r).__name__}"
            )
        path = r[path_key]
        boxes = np.asarray(r[boxes_key])
        labels = None
        try:
            labels_raw = r[labels_key]
        except (KeyError, TypeError):
            labels_raw = None
        if labels_raw is not None:
            labels = np.asarray(labels_raw, dtype=np.int64)
        out.append(ChipImageAnn(image_path=str(path), boxes=boxes, labels=labels))
    return out


__all__ = ["ChipDataset", "ChipImageAnn", "make_chip_annotations"]
