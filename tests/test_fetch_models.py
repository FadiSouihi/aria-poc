"""Install-path guards for tools/fetch_models.py.

The weights are not in git, so this script *is* the install path on a new
machine. A typo in a destination or a second source for one entry would only
show up as a broken fresh install, so the registry is checked here without any
network access.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fetch_models as fm  # noqa: E402


def test_every_entry_has_exactly_one_source():
    for model in fm.MODELS:
        sources = [
            bool(model.url),
            bool(model.hf_file),
            bool(model.hf_snapshot),
            bool(model.ultralytics),
        ]
        assert sum(sources) == 1, f"{model.dest} has {sum(sources)} sources"


def test_destinations_are_unique_relative_and_inside_the_repo():
    seen = set()
    for model in fm.MODELS:
        dest = pathlib.PurePosixPath(model.dest)
        assert not dest.is_absolute(), model.dest
        assert ".." not in dest.parts, model.dest
        assert model.dest not in seen, f"duplicate destination {model.dest}"
        seen.add(model.dest)
        # Everything must land under a directory git deliberately ignores.
        assert dest.parts[0] in {"weights", "data"}, model.dest


def test_every_entry_documents_what_it_is_and_its_licence():
    for model in fm.MODELS:
        assert len(model.what) > 15, model.dest
        assert model.license, model.dest


def test_pinned_sizes_are_plausible():
    for model in fm.MODELS:
        assert model.size > 0, model.dest


def test_check_reports_missing_wrong_size_and_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "ROOT", tmp_path)
    stub = fm.Model(key="k", dest="weights/stub.onnx", what="a stub model entry",
                    license="MIT", size=100, url="https://example.invalid/stub.onnx")
    assert fm.check(stub) == "missing"

    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "stub.onnx").write_bytes(b"x" * 50)
    assert fm.check(stub) == "wrong-size"

    (tmp_path / "weights" / "stub.onnx").write_bytes(b"x" * 100)
    assert fm.check(stub) == "ok"


def test_check_detects_a_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "ROOT", tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "stub.onnx").write_bytes(b"x" * 10)
    stub = fm.Model(key="k", dest="weights/stub.onnx", what="a stub model entry",
                    license="MIT", size=10, sha256="0" * 64,
                    url="https://example.invalid/stub.onnx")
    assert fm.check(stub) == "wrong-hash"


def test_cli_list_and_unknown_key(capsys):
    assert fm.main(["--list"]) == 0
    assert "silero_vad.onnx" in capsys.readouterr().out
    assert fm.main(["--only", "not-a-key"]) == 2


def test_keys_match_the_documented_groups():
    assert {m.key for m in fm.MODELS} == {
        "vision", "vad", "turn", "face", "voiceprint", "whisper"
    }