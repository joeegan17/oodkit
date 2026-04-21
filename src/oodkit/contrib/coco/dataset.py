"""
High-level builders that turn COCO / COCO-O directories into
:class:`oodkit.data.chip_dataset.ChipDataset` instances.

These helpers are intentionally thin wrappers over
:func:`oodkit.contrib.coco.loader.load_coco` +
:class:`oodkit.data.chip_dataset.ChipDataset`; they exist so demo notebooks read
like three or four lines instead of thirty. Requires the ``[ml]`` extras.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from oodkit.contrib.coco.category_table import CocoCategoryTable
from oodkit.contrib.coco.discovery import (
    CocoOodDomainPaths,
    discover_coco_id,
    discover_coco_ood,
)
from oodkit.contrib.coco.loader import load_coco
from oodkit.data.chip_dataset import ChipDataset


def coco_chip_dataset(
    ann_path: Union[str, Path],
    image_root: Union[str, Path],
    processor: Any,
    *,
    group: Optional[str] = None,
    category_table: Optional[CocoCategoryTable] = None,
    min_chip_size: int = 25,
    min_box_side: float = 0.0,
    fill: int = 0,
    image_mode: str = "RGB",
    loader: Optional[Callable[..., Any]] = None,
) -> ChipDataset:
    """Build a :class:`ChipDataset` from a single COCO-format JSON.

    Args:
        ann_path: Path to ``instances_*.json``.
        image_root: Directory holding the images referenced in the JSON.
        processor: HuggingFace image processor used by :class:`ChipDataset`.
        group: Optional group tag (e.g. ``"cartoon"``). Propagates via
            ``EmbeddingResult.metadata["group"]``.
        category_table: Shared category table so ID/OOD splits use matching
            contiguous labels. When ``None`` one is built from this JSON.
        min_chip_size: Minimum square-chip side (pixels).
        min_box_side: Drop boxes with either side < this threshold (pixels).
        fill: Pad value for out-of-image chip regions.
        image_mode: PIL mode for decoded images (default ``"RGB"``).
        loader: Optional PIL loader override.

    Returns:
        Ready-to-use :class:`ChipDataset`; class names come from
        ``category_table`` when available so ``object_ids`` are human-readable.
    """
    if category_table is None:
        category_table = CocoCategoryTable.from_coco_json(ann_path)

    annotations = load_coco(
        ann_path,
        image_root,
        group=group,
        category_table=category_table,
        min_box_side=min_box_side,
    )
    if not annotations:
        raise ValueError(
            f"No annotations usable from {ann_path} (image_root={image_root}); "
            "every image was empty or below min_box_side."
        )

    return ChipDataset(
        annotations,
        processor=processor,
        box_format="xyxy",
        min_chip_size=min_chip_size,
        fill=fill,
        image_mode=image_mode,
        loader=loader,
        class_names=category_table.names(),
    )


def coco_ood_chip_datasets(
    ood_root: Union[str, Path],
    processor: Any,
    *,
    category_table: CocoCategoryTable,
    only: Optional[Sequence[str]] = None,
    min_chip_size: int = 25,
    min_box_side: float = 0.0,
    fill: int = 0,
    image_mode: str = "RGB",
    loader: Optional[Callable[..., Any]] = None,
) -> Dict[str, ChipDataset]:
    """Build one :class:`ChipDataset` per COCO-O domain, keyed by domain name.

    Each dataset's annotations carry ``group=<domain_name>`` so downstream
    analysis can slice scores per domain (cartoon / tattoo / weather / ...).

    Args:
        ood_root: COCO-O root directory (see :func:`discover_coco_ood`).
        processor: HuggingFace image processor.
        category_table: Shared table so OOD labels match ID labels. Must be
            provided explicitly (COCO-O labels are a subset of COCO's).
        only: Optional whitelist of domain names (preserves order).
        min_chip_size: Minimum chip side in pixels.
        min_box_side: Drop boxes smaller than this before chip creation.
        fill: Pad value for out-of-image chip regions.
        image_mode: PIL image mode.
        loader: Optional PIL loader override.

    Returns:
        ``{domain_name: ChipDataset, ...}`` in discovery order.
    """
    domains: List[CocoOodDomainPaths] = discover_coco_ood(
        ood_root,
        only=list(only) if only is not None else None,
    )
    out: Dict[str, ChipDataset] = {}
    for dom in domains:
        out[dom.name] = coco_chip_dataset(
            dom.ann,
            dom.images,
            processor,
            group=dom.name,
            category_table=category_table,
            min_chip_size=min_chip_size,
            min_box_side=min_box_side,
            fill=fill,
            image_mode=image_mode,
            loader=loader,
        )
    return out


def coco_id_chip_datasets(
    root: Union[str, Path],
    processor: Any,
    *,
    category_table: Optional[CocoCategoryTable] = None,
    min_chip_size: int = 25,
    min_box_side: float = 0.0,
    fill: int = 0,
    image_mode: str = "RGB",
    loader: Optional[Callable[..., Any]] = None,
) -> Dict[str, ChipDataset]:
    """Build ``{"train": ChipDataset, "val": ChipDataset}`` from a COCO ID root.

    The category table is built from the train JSON when not supplied so both
    splits share a consistent 0..K-1 label space.
    """
    paths = discover_coco_id(root)
    if category_table is None:
        category_table = CocoCategoryTable.from_coco_json(paths.train_ann)

    return {
        split: coco_chip_dataset(
            ann,
            images,
            processor,
            category_table=category_table,
            min_chip_size=min_chip_size,
            min_box_side=min_box_side,
            fill=fill,
            image_mode=image_mode,
            loader=loader,
        )
        for split, (ann, images) in paths.as_pairs().items()
    }


__all__ = [
    "coco_chip_dataset",
    "coco_id_chip_datasets",
    "coco_ood_chip_datasets",
]
