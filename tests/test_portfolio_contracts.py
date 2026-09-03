"""Artifact IO contract: atomic writes, corrupt-read safety, stable hashing."""

import json

import pytest

from tradingagents.portfolio.contracts import (
    atomic_write_json,
    read_json,
    stable_hash,
)


@pytest.mark.unit
def test_atomic_write_leaves_no_temp_file(tmp_path):
    target = tmp_path / "sub" / "artifact.json"
    atomic_write_json(target, {"a": 1})
    assert target.exists()
    assert read_json(target) == {"a": 1}
    # No leftover temp file next to the artifact.
    assert list(target.parent.glob("*.tmp")) == []


@pytest.mark.unit
def test_read_json_missing_and_corrupt_return_none(tmp_path):
    assert read_json(tmp_path / "nope.json") is None
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not valid", encoding="utf-8")
    assert read_json(corrupt) is None


@pytest.mark.unit
def test_stable_hash_is_order_independent_and_deterministic():
    a = stable_hash({"x": 1, "y": [1, 2]})
    b = stable_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert len(a) == 16
    assert a != stable_hash({"x": 1, "y": [2, 1]})


@pytest.mark.unit
def test_atomic_write_overwrites_existing(tmp_path):
    target = tmp_path / "a.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 2}
