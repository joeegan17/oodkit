"""
COCO / COCO-O object-detection helpers (optional).

Pure-Python pieces (no torch) are importable directly:

    from oodkit.contrib.coco import (
        CocoCategoryTable,
        load_coco,
        discover_coco_id,
        discover_coco_ood,
    )

Chip-dataset builders require the ``[ml]`` extras and are lazy-loaded:

    from oodkit.contrib.coco import (
        coco_chip_dataset,
        coco_id_chip_datasets,
        coco_ood_chip_datasets,
    )

Directory layout the discovery helpers expect:

- **COCO ID**::

    <root>/
      coco_annotations/
        instances_train2017.json
        instances_val2017.json
      coco_train/   # image files
      coco_val/     # image files

- **COCO-O**::

    <ood_root>/
      cartoon/
        annotations/instances_val2017.json
        images/        # image files (renamed from val2017/)
      tattoo/...
      weather/...

See :mod:`oodkit.contrib.coco.discovery` for knobs if a layout differs.
"""

from oodkit.contrib.coco.category_table import CocoCategoryTable
from oodkit.contrib.coco.discovery import (
    CocoIdPaths,
    CocoOodDomainPaths,
    discover_coco_id,
    discover_coco_ood,
)
from oodkit.contrib.coco.loader import load_coco

_LAZY = {"coco_chip_dataset", "coco_id_chip_datasets", "coco_ood_chip_datasets"}

__all__ = [
    "CocoCategoryTable",
    "CocoIdPaths",
    "CocoOodDomainPaths",
    "coco_chip_dataset",
    "coco_id_chip_datasets",
    "coco_ood_chip_datasets",
    "discover_coco_id",
    "discover_coco_ood",
    "load_coco",
]


def __getattr__(name: str):
    if name in _LAZY:
        from oodkit.contrib.coco import dataset as _dataset

        return getattr(_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | {"__doc__", "__file__"})
