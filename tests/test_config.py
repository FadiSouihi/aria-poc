"""Contract tests: profile loading, inheritance, dotted access."""
import pytest

from aria.core.config import load_config


def test_profile_inheritance_and_dotted_access(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "logging:\n  level: INFO\n  console: pretty\nruntime:\n  duration: 0\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: base.yaml\nlogging:\n  level: DEBUG\nextra: 7\n",
        encoding="utf-8",
    )
    cfg = load_config(child)
    assert cfg.get("logging.level") == "DEBUG"          # child overrides
    assert cfg.get("logging.console") == "pretty"       # parent value kept
    assert cfg.get("runtime.duration") == 0             # deep merge
    assert cfg.get("extra") == 7
    assert cfg.source == str(child)


def test_missing_required_key_raises():
    from aria.core.config import Config

    with pytest.raises(KeyError):
        Config({"a": 1}).require("b.c")
