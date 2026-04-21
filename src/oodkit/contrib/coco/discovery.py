"""
Path discovery helpers for COCO and COCO-O layouts.

Pure Python: no torch, no image decoding. Just small dataclasses that resolve a
few well-known directory shapes into explicit ``(annotations, images)`` pairs so
callers can feed them to :func:`load_coco`. Optional overrides make it easy to
adapt to non-standard local layouts without rewriting the whole loader.

Expected layouts
----------------

COCO ID (train / val)::

    <root>/
      coco_annotations/
        instances_train2017.json
        instances_val2017.json
      coco_train/    # image files
      coco_val/      # image files

COCO-O (per-domain)::

    <ood_root>/
      cartoon/
        annotations/instances_val2017.json
        images/       # image files (renamed from val2017/)
      tattoo/...
      weather/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class CocoIdPaths:
    """Resolved paths for the COCO 2017 train/val splits."""

    train_ann: Path
    train_images: Path
    val_ann: Path
    val_images: Path

    def as_pairs(self) -> Dict[str, Tuple[Path, Path]]:
        """Return ``{"train": (ann, images), "val": (ann, images)}``."""
        return {
            "train": (self.train_ann, self.train_images),
            "val": (self.val_ann, self.val_images),
        }


@dataclass(frozen=True)
class CocoOodDomainPaths:
    """Resolved ``(annotations, images)`` for a single COCO-O domain."""

    name: str
    ann: Path
    images: Path


def discover_coco_id(
    root: Union[str, Path],
    *,
    annotations_dir: str = "coco_annotations",
    train_images_dir: str = "coco_train",
    val_images_dir: str = "coco_val",
    train_ann_name: str = "instances_train2017.json",
    val_ann_name: str = "instances_val2017.json",
) -> CocoIdPaths:
    """Resolve COCO train/val annotation + image-directory paths.

    Raises:
        FileNotFoundError: If any expected sub-path is missing.
    """
    r = Path(root)
    train_ann = r / annotations_dir / train_ann_name
    val_ann = r / annotations_dir / val_ann_name
    train_images = r / train_images_dir
    val_images = r / val_images_dir

    for p in (train_ann, val_ann):
        if not p.is_file():
            raise FileNotFoundError(f"COCO ID annotation file not found: {p}")
    for p in (train_images, val_images):
        if not p.is_dir():
            raise FileNotFoundError(f"COCO ID image directory not found: {p}")

    return CocoIdPaths(
        train_ann=train_ann,
        train_images=train_images,
        val_ann=val_ann,
        val_images=val_images,
    )


def discover_coco_ood(
    root: Union[str, Path],
    *,
    only: Optional[List[str]] = None,
    annotations_subdir: str = "annotations",
    annotations_name: str = "instances_val2017.json",
    images_subdir: str = "images",
) -> List[CocoOodDomainPaths]:
    """Resolve per-domain paths for a COCO-O style directory.

    Every immediate subdirectory of ``root`` that contains both the expected
    annotations file and the expected images directory is returned; missing or
    mismatched domains are skipped silently **unless** named explicitly in
    ``only`` (then a :class:`FileNotFoundError` is raised).

    Args:
        root: Top-level COCO-O directory.
        only: Optional whitelist of domain names (``["cartoon", "tattoo"]``).
            When provided, the result preserves the listed order and missing
            domains raise.
        annotations_subdir: Subdir (under each domain) containing the JSON.
        annotations_name: Filename of the per-domain annotations JSON.
        images_subdir: Subdir (under each domain) containing images. Default
            ``"images"`` matches the renamed COCO-O layout used in this repo.

    Returns:
        List of :class:`CocoOodDomainPaths`, sorted alphabetically by ``name``
        (or in the order of ``only`` when provided).
    """
    r = Path(root)
    if not r.is_dir():
        raise FileNotFoundError(f"COCO-O root is not a directory: {r}")

    def _resolve(name: str) -> Optional[CocoOodDomainPaths]:
        domain_root = r / name
        ann = domain_root / annotations_subdir / annotations_name
        images = domain_root / images_subdir
        if not (ann.is_file() and images.is_dir()):
            return None
        return CocoOodDomainPaths(name=name, ann=ann, images=images)

    if only is not None:
        out: List[CocoOodDomainPaths] = []
        for name in only:
            resolved = _resolve(name)
            if resolved is None:
                raise FileNotFoundError(
                    f"COCO-O domain {name!r} is incomplete under {r}: "
                    f"expected {annotations_subdir}/{annotations_name} and {images_subdir}/"
                )
            out.append(resolved)
        return out

    discovered: List[CocoOodDomainPaths] = []
    for child in sorted(r.iterdir()):
        if not child.is_dir():
            continue
        resolved = _resolve(child.name)
        if resolved is not None:
            discovered.append(resolved)
    return discovered


__all__ = [
    "CocoIdPaths",
    "CocoOodDomainPaths",
    "discover_coco_id",
    "discover_coco_ood",
]
