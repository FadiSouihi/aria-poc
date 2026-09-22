"""Run the full test suite:  python run_tests.py  (extra args pass to pytest)."""
import pathlib
import sys

import pytest

if __name__ == "__main__":
    default_target = str(pathlib.Path(__file__).resolve().parent / "tests")
    args = sys.argv[1:] or ["-q", default_target]
    raise SystemExit(pytest.main(args))
