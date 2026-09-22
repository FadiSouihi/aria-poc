"""Config-driven plugin registry: instantiate, validate, wire, lifecycle.

Service declarations in a profile:

    services:
      camera:
        class: "aria.perception.camera:CameraService"   # module:ClassName
        enabled: true
        config: {fps: 15}

Adding a capability = a class + a config entry; removing = the ``enabled``
flag (or deleting the entry). Nothing else changes.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional

from aria.core.config import Config
from aria.core.telemetry.logging import get_logger


def import_class(class_path: str) -> type:
    """Import ``module:ClassName``."""
    if ":" not in class_path:
        raise ValueError(f"class must be 'module:ClassName', got {class_path!r}")
    module_name, cls_name = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, cls_name)
    except AttributeError as exc:
        raise ImportError(f"Module {module_name!r} has no class {cls_name!r}") from exc


class Registry:
    def __init__(self, bus, config: Config) -> None:
        self.bus = bus
        self.config = config
        self.services: Dict[str, object] = {}
        self.disabled: Dict[str, str] = {}
        self.start_errors: Dict[str, str] = {}
        self.restart_counts: Dict[str, int] = {}
        self.log = get_logger("aria.core.registry")

    # -- build -----------------------------------------------------------
    def build(self) -> List[object]:
        """Instantiate every enabled service declared in the profile."""
        services_cfg = self.config.section("services")
        if not services_cfg:
            self.log.warning("No services declared in config profile")
        for key, spec in services_cfg.items():
            if not isinstance(spec, dict) or "class" not in spec:
                self.log.error("Invalid service spec (missing 'class')", service=key)
                continue
            if not spec.get("enabled", True):
                self.disabled[key] = "disabled by config"
                self.log.info("Service disabled", service=key)
                continue
            try:
                cls = import_class(spec["class"])
                svc = cls()
                svc.attach(self.bus, Config(dict(spec.get("config") or {}), f"services.{key}"))
            except Exception:
                self.log.exception("Failed to build service", service=key, cls=spec.get("class"))
                continue
            if hasattr(svc, "set_registry"):  # watchdog gets the registry back-reference
                svc.set_registry(self)
            self.services[key] = svc
            self.log.info(
                "Service built", service=key, cls=spec["class"], produces=svc.produces, consumes=svc.consumes
            )
        return list(self.services.values())

    # -- lookup ----------------------------------------------------------
    def get(self, name: str) -> object:
        return self.services[name]

    def names(self) -> List[str]:
        return list(self.services)

    # -- lifecycle -------------------------------------------------------
    async def start_all(self) -> None:
        for name, svc in self.services.items():
            try:
                await svc.start()
            except Exception as exc:
                self.start_errors[name] = repr(exc)
                self.log.exception("Service failed to start", service=name)
        if self.start_errors:
            self.log.warning("Some services failed to start", failed=self.start_errors)

    async def stop_all(self) -> None:
        for name in reversed(list(self.services)):
            try:
                await self.services[name].stop()
            except Exception:
                self.log.exception("Service failed to stop cleanly", service=name)

    async def restart(self, name: str, reason: str = "") -> None:
        """Restart one service in place — the watchdog's recovery path."""
        svc = self.services.get(name)
        if svc is None:
            raise KeyError(f"Unknown service {name!r}")
        self.log.info("Restarting service", service=name, reason=reason)
        try:
            await svc.stop()
        except Exception:
            self.log.exception("Stop during restart failed (continuing)", service=name)
        await svc.start()
        self.restart_counts[name] = self.restart_counts.get(name, 0) + 1
