"""Structured JSON logging with correlation context and ID masking.

Design (ROADMAP §9):
- one logger per service; every line is a single JSON object with
  ``ts / level / service / msg`` plus any keyword fields and the bound
  correlation context (``session_id``, ``track_id``, ``utterance_id``);
- a masking filter redacts long digit runs (placeholder for the shared
  NFR-07 policy layer) so logs never become a data-leak channel;
- file sink is rotating; console sink is pretty (dev) or json/quiet.

Usage:
    log = get_logger("aria.perception.camera")
    log.info("Frame captured", seq=n, source="fake")
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aria.core.context import snapshot

_MASK_PATTERNS = (re.compile(r"\b\d{6,}\b"),)
_MASK_TOKEN = "[REDACTED-ID]"

_configured = False


def _mask(text: str) -> str:
    out = text
    for pattern in _MASK_PATTERNS:
        out = pattern.sub(_MASK_TOKEN, out)
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": record.name,
            "msg": _mask(record.getMessage()),
        }
        fields: Optional[Dict[str, Any]] = getattr(record, "fields", None)
        if fields:
            payload["fields"] = {k: _mask(v) if isinstance(v, str) else v for k, v in fields.items()}
        ctx = getattr(record, "event_context", None) or snapshot()
        if ctx:
            payload["context"] = ctx
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class StructLogger:
    """Thin wrapper adding keyword fields to standard log calls."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, fields: Dict[str, Any], exc_info: bool = False) -> None:
        self._logger.log(level, msg, extra={"fields": fields}, exc_info=exc_info)

    def debug(self, msg: str, **fields: Any) -> None:
        self._log(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._log(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._log(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields, exc_info=True)

    def exception(self, msg: str, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields, exc_info=True)

    @property
    def raw(self) -> logging.Logger:
        return self._logger


def get_logger(name: str) -> StructLogger:
    """Return a structured logger; call ``setup_logging`` once at startup."""
    return StructLogger(logging.getLogger(name))


def setup_logging(
    level: str = "INFO",
    console: str = "pretty",
    dir_name: str = "logs",
    max_bytes_mb: int = 5,
    backups: int = 3,
    levels: Optional[Dict[str, str]] = None,
) -> None:
    """Configure the root ``aria`` logger (idempotent; clears old handlers)."""
    global _configured
    root = logging.getLogger("aria")
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(logging.DEBUG)  # handlers filter; module levels refine below
    fmt_json = JsonFormatter()

    log_dir = pathlib.Path(dir_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "aria.log",
        maxBytes=int(max_bytes_mb * 1024 * 1024),
        backupCount=backups,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt_json)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    if console == "pretty":
        console_handler: Optional[logging.Handler] = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        console_handler.setLevel(logging.DEBUG)
        root.addHandler(console_handler)
    elif console == "json":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt_json)
        console_handler.setLevel(logging.DEBUG)
        root.addHandler(console_handler)
    # "quiet" → no console output; file sink still active.

    root.setLevel(_parse_level(level))
    for module, mod_level in (levels or {}).items():
        logging.getLogger(module).setLevel(_parse_level(mod_level))
    _configured = True


def _parse_level(name: str) -> int:
    return getattr(logging, str(name).upper(), logging.INFO)
