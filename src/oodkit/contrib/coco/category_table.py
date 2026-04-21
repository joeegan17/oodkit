"""
COCO category ↔ canonical index mapping.

COCO stores class ids as a sparse set (1..90 with gaps for the 80 classes). This
module exposes a small table that remaps them to contiguous indices ``0..K-1``
so downstream code (``ChipDataset.labels``, classifier heads, etc.) can stay
dense.

No PyTorch dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


@dataclass(frozen=True)
class CocoCategoryTable:
    """Contiguous-index view of a COCO ``categories`` list.

    Attributes:
        idx_to_category_id: ``(K,)`` tuple of raw COCO ``category_id`` values,
            sorted ascending. Index into this tuple = contiguous label.
        idx_to_name: ``(K,)`` tuple of human-readable class names aligned with
            ``idx_to_category_id``.
        category_id_to_idx: Reverse map from raw COCO id → contiguous index.
    """

    idx_to_category_id: Tuple[int, ...]
    idx_to_name: Tuple[str, ...]
    category_id_to_idx: Mapping[int, int]

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_category_id)

    @classmethod
    def from_categories(
        cls, categories: Sequence[Mapping[str, Any]]
    ) -> "CocoCategoryTable":
        """Build a table from a COCO ``categories`` list (each entry has ``id`` + ``name``)."""
        if not categories:
            raise ValueError("categories is empty")
        seen: Dict[int, str] = {}
        for cat in categories:
            cid = int(cat["id"])
            name = str(cat["name"])
            if cid in seen and seen[cid] != name:
                raise ValueError(
                    f"COCO category id {cid} maps to two names: "
                    f"{seen[cid]!r} and {name!r}"
                )
            seen[cid] = name
        ordered_ids = sorted(seen.keys())
        ordered_names = tuple(seen[cid] for cid in ordered_ids)
        cid_to_idx = {cid: i for i, cid in enumerate(ordered_ids)}
        return cls(
            idx_to_category_id=tuple(ordered_ids),
            idx_to_name=ordered_names,
            category_id_to_idx=cid_to_idx,
        )

    @classmethod
    def from_coco_json(cls, path: Union[str, Path]) -> "CocoCategoryTable":
        """Build a table from the ``categories`` array in a COCO JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "categories" not in data:
            raise ValueError(f"{p} does not contain a 'categories' field")
        return cls.from_categories(data["categories"])

    def names(self) -> List[str]:
        """Return ``idx_to_name`` as a list (useful for ``ChipDataset(class_names=...)``)."""
        return list(self.idx_to_name)

    def map_category_ids(self, raw_ids: Sequence[int]) -> List[int]:
        """Remap a sequence of raw COCO category ids to contiguous indices."""
        try:
            return [self.category_id_to_idx[int(c)] for c in raw_ids]
        except KeyError as exc:
            raise KeyError(
                f"COCO category id {exc.args[0]} is not present in this table"
            ) from exc


__all__ = ["CocoCategoryTable"]
