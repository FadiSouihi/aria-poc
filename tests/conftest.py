"""Pytest bootstrap: make the repo root importable as ``aria`` and expose
shared helpers without requiring package-relative imports in tests."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tests"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
