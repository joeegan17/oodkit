"""Tests for ``oodkit.contrib.imagenet.synset_table`` (no torch)."""

from pathlib import Path

import pytest

from oodkit.contrib.imagenet.synset_table import SynsetTable


def _write_mapping(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_synset_table_from_file_order_and_lookup(tmp_path: Path):
    p = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(
        p,
        [
            "n01498041 stingray",
            "n01531178 goldfinch",
            "n01534433 junco",
        ],
    )
    t = SynsetTable.from_file(p)
    assert t.n_classes == 3
    assert t.idx_for_wnid("n01498041") == 0
    assert t.idx_for_wnid("n01534433") == 2
    assert t.name_for_idx(1) == "goldfinch"
    assert t.wnid_for_idx(1) == "n01531178"
    assert t.name_for_idx(2) == "junco"
    assert list(t.idx_to_wnid) == ["n01498041", "n01531178", "n01534433"]


def test_synset_table_duplicate_wnid_raises(tmp_path: Path):
    p = tmp_path / "bad.txt"
    _write_mapping(
        p,
        [
            "n01498041 stingray",
            "n01498041 duplicate",
        ],
    )
    with pytest.raises(ValueError, match="duplicate wnid"):
        SynsetTable.from_file(p)


def test_synset_table_unsorted_raises(tmp_path: Path):
    p = tmp_path / "bad_order.txt"
    _write_mapping(
        p,
        [
            "n01531178 goldfinch",
            "n01498041 stingray",
        ],
    )
    with pytest.raises(ValueError, match="lexicographic"):
        SynsetTable.from_file(p)


def test_synset_table_skips_blank_and_comment(tmp_path: Path):
    p = tmp_path / "m.txt"
    _write_mapping(
        p,
        [
            "",
            "# comment",
            "n01498041 stingray",
            "",
            "n01531178 goldfinch",
        ],
    )
    t = SynsetTable.from_file(p)
    assert t.n_classes == 2


def test_validate_root_unknown_and_missing(tmp_path: Path):
    root = tmp_path / "data"
    (root / "n01498041").mkdir(parents=True)
    (root / "not_a_wnid").mkdir()
    (root / "n01531178").mkdir()

    m = tmp_path / "LOC_synset_mapping.txt"
    _write_mapping(
        m,
        [
            "n01498041 stingray",
            "n01531178 goldfinch",
            "n01534433 junco",
        ],
    )
    t = SynsetTable.from_file(m)
    v = t.validate_root(root, check_missing=False)
    assert set(v.present_wnids) == {"n01498041", "n01531178"}
    assert v.unknown_folders == ("not_a_wnid",)
    assert v.missing_wnids == ()

    v2 = t.validate_root(root, check_missing=True)
    assert v2.missing_wnids == ("n01534433",)
