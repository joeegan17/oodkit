"""
ImageNet-1k synset metadata from ``LOC_synset_mapping.txt``.

Canonical class index ``0 … n-1`` follows **line order in the mapping file**
(one synset per line: ``<wnid> <human-readable name>``). That order matches
lexicographically sorted WordNet IDs for the standard ILSVRC 1000-class list.

This module has **no PyTorch dependency**.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union


def _parse_line(line: str) -> Tuple[str, str]:
    line = line.strip()
    if not line or line.startswith("#"):
        raise ValueError(f"empty or comment line: {line!r}")
    parts = line.split(None, 1)
    if len(parts) != 2:
        raise ValueError(f"expected '<wnid> <name>', got: {line!r}")
    wnid, name = parts[0], parts[1].strip()
    if len(wnid) != 9 or not wnid.startswith("n") or not wnid[1:].isdigit():
        raise ValueError(f"unexpected synset id format: {wnid!r}")
    return wnid, name


def _read_mapping_lines(path: Union[str, Path]) -> List[Tuple[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Synset mapping file not found: {p}")
    rows: List[Tuple[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            try:
                rows.append(_parse_line(s))
            except ValueError as exc:
                raise ValueError(f"{p}:{lineno}: {exc}") from exc
    if not rows:
        raise ValueError(f"No synset rows parsed from {p}")
    return rows


@dataclass(frozen=True)
class RootValidation:
    """Output of :meth:`SynsetTable.validate_root`."""

    root: Path
    """Directory checked (one subfolder per synset)."""

    present_wnids: Tuple[str, ...]
    """Synset folder names under ``root`` that exist in the mapping table."""

    unknown_folders: Tuple[str, ...]
    """Immediate subdirectories of ``root`` that are not valid ``wnid`` keys."""

    missing_wnids: Tuple[str, ...]
    """Mapping ``wnid``s with no folder under ``root`` (optional completeness check)."""


class SynsetTable:
    """Maps ILSVRC synset id (``wnid``) to canonical class index and name.

    Indices are ``0 … n_classes-1`` in **file line order** (non-empty,
    non-comment lines) in ``LOC_synset_mapping.txt``.

    Args:
        rows: Sequence of ``(wnid, human_name)`` in canonical order.

    Raises:
        ValueError: If duplicate ``wnid`` or empty ``rows``.
    """

    def __init__(self, rows: List[Tuple[str, str]]) -> None:
        if not rows:
            raise ValueError("SynsetTable requires at least one (wnid, name) row")
        wnid_to_idx: Dict[str, int] = {}
        idx_to_wnid: List[str] = []
        idx_to_name: List[str] = []
        wnid_to_name: Dict[str, str] = {}
        for idx, (wnid, name) in enumerate(rows):
            if wnid in wnid_to_idx:
                raise ValueError(f"duplicate wnid in mapping: {wnid!r}")
            wnid_to_idx[wnid] = idx
            wnid_to_name[wnid] = name
            idx_to_wnid.append(wnid)
            idx_to_name.append(name)
        self._wnid_to_idx = wnid_to_idx
        self._wnid_to_name = wnid_to_name
        self._idx_to_wnid = tuple(idx_to_wnid)
        self._idx_to_name = tuple(idx_to_name)
        self._assert_sorted_wnids()

    def _assert_sorted_wnids(self) -> None:
        wnids = list(self._idx_to_wnid)
        sorted_wnids = sorted(wnids)
        if wnids != sorted_wnids:
            raise ValueError(
                "LOC_synset_mapping.txt lines must be in lexicographic wnid order "
                f"(found order != sorted); first mismatch near {wnids[:3]!r} ..."
            )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SynsetTable":
        """Load from ``LOC_synset_mapping.txt``."""
        return cls(_read_mapping_lines(path))

    @property
    def n_classes(self) -> int:
        """Number of classes (length of the mapping)."""
        return len(self._idx_to_wnid)

    @property
    def wnid_to_idx(self) -> Dict[str, int]:
        """Synset id to canonical index."""
        return dict(self._wnid_to_idx)

    @property
    def wnid_to_name(self) -> Dict[str, str]:
        """Synset id to human-readable name."""
        return dict(self._wnid_to_name)

    @property
    def idx_to_wnid(self) -> Tuple[str, ...]:
        """Canonical index to synset id."""
        return self._idx_to_wnid

    @property
    def idx_to_name(self) -> Tuple[str, ...]:
        """Canonical index to human-readable name (same order as ``classes`` list)."""
        return self._idx_to_name

    def idx_for_wnid(self, wnid: str) -> int:
        """Return canonical index for ``wnid``.

        Raises:
            KeyError: If ``wnid`` is not in the table.
        """
        return self._wnid_to_idx[wnid]

    def name_for_idx(self, idx: int) -> str:
        """Human-readable name for canonical ``idx``."""
        return self._idx_to_name[idx]

    def wnid_for_idx(self, idx: int) -> str:
        """Synset id for canonical ``idx``."""
        return self._idx_to_wnid[idx]

    def validate_root(
        self,
        root: Union[str, Path],
        *,
        check_missing: bool = False,
    ) -> RootValidation:
        """Compare a directory tree (ImageFolder-style) to this table.

        Immediate subdirectories of ``root`` are treated as class folders. Names
        that are not in the mapping appear in ``unknown_folders``. If
        ``check_missing`` is True, every table ``wnid`` without a folder is listed
        in ``missing_wnids``.

        Args:
            root: Root directory (e.g. one ImageNet-O variant folder).
            check_missing: If True, report table synsets absent under ``root``.

        Returns:
            :class:`RootValidation` (informational; does not raise).
        """
        r = Path(root)
        if not r.is_dir():
            raise FileNotFoundError(f"Dataset root is not a directory: {r}")

        subdirs = sorted(p.name for p in r.iterdir() if p.is_dir())
        unknown: List[str] = []
        present: List[str] = []
        wnid_set: Set[str] = set(self._wnid_to_idx)
        for name in subdirs:
            if name in wnid_set:
                present.append(name)
            else:
                unknown.append(name)

        missing: List[str] = []
        if check_missing:
            have = set(present)
            for w in self._idx_to_wnid:
                if w not in have:
                    missing.append(w)

        return RootValidation(
            root=r.resolve(),
            present_wnids=tuple(sorted(present)),
            unknown_folders=tuple(unknown),
            missing_wnids=tuple(missing),
        )
