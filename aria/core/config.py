"""Profile-based configuration with inheritance, dotted access, and defaults.

Profiles are YAML files. A profile may extend another via ``extends: path``
(child keys override parent keys recursively). Services read their config with
``service.config.get("key", default)``; the schema check in
:class:`aria.core.service.Service` validates service sections.
"""
from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, Mapping

import yaml


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Immutable-ish view over a config dict with dotted access."""

    def __init__(self, data: Dict[str, Any], source: str = "<memory>") -> None:
        self._data: Dict[str, Any] = data
        self.source = source

    # -- access ----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Missing required config key '{dotted}' in {self.source}")
        return value

    def section(self, dotted: str) -> Dict[str, Any]:
        value = self.get(dotted, {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Config(source={self.source!r}, keys={sorted(self._data)})"


def load_config(path: str | pathlib.Path) -> Config:
    """Load a YAML profile, resolving ``extends`` chains (parents first)."""
    path = pathlib.Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    parent_name = data.pop("extends", None)
    if parent_name:
        parent_path = path.parent / parent_name
        parent = load_config(parent_path)
        data = _deep_merge(parent.raw, data)
    return Config(data, str(path))
