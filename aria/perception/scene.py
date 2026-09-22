"""SceneManagerService — per-track state machine over vision tracks (ROADMAP §7).

The unified single/group logic: there is no "mode" for one vs many people —
every track gets its own FSM and the scene is just the set of active tracks.

  PRESENCE (first seen) → NEAR (close/central for ``near_dwell_s``)
  → ENGAGED (keeps dwelling for ``engage_dwell_s``) → (absent ≥
  ``departed_after_s``) → removed with a departure log.

Consumes ``SceneTick`` (raw vision tracks) + ``FaceMatched`` (identities) and
publishes ``TrackStates`` — the enriched per-track snapshot that the debug
HUD (and, in Phase 3, the dialogue layer) consumes. Transitions are logged,
which gives the NFR-01-style interaction timeline for free.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from aria.core.events import Event
from aria.core.service import Service

_SCHEMA = {
    "near_area_frac": (0.06, (int, float)),   # bbox area ≥ 6% of frame ⇒ "close"
    "central_frac": (0.60, (int, float)),     # center inside the middle 60% band ⇒ "central"
    "near_dwell_s": (1.0, (int, float)),
    "engage_dwell_s": (2.0, (int, float)),
    "departed_after_s": (10.0, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class SceneManagerService(Service):
    name = "perception.scene"
    produces = ("TrackStates",)
    consumes = ("SceneTick", "FaceMatched")
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._subs = []
        self.tracks: dict = {}   # track_id -> record
        self._frame_size = (480, 640)

    async def on_start(self) -> None:
        self._subs.append(self.bus.subscribe("SceneTick", self._on_scenetick, policy="drop_oldest", maxsize=2))
        self._subs.append(self.bus.subscribe("FaceMatched", self._on_face_matched, policy="drop_oldest", maxsize=4))

    async def on_stop(self) -> None:
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()

    # -- helpers -----------------------------------------------------------
    def _near_enough(self, bbox, area_frac: float) -> bool:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        fh, fw = self._frame_size
        central = abs(cx - fw / 2) <= (fw * float(self.config.get("central_frac", 0.6)) / 2) and \
            abs(cy - fh / 2) <= (fh * float(self.config.get("central_frac", 0.6)) / 2)
        return area_frac >= float(self.config.get("near_area_frac", 0.06)) or central

    def _transition(self, tid: int, rec: dict, new_state: Optional[str]) -> None:
        if new_state and rec["state"] != new_state:
            self.log.info("Track state transition", track_id=tid, from_state=rec["state"], to_state=new_state)
            rec["state"] = new_state

    # -- handlers ----------------------------------------------------------
    async def _on_scenetick(self, event: Event) -> None:
        now = time.monotonic()
        seen = set()
        out = []
        for track in event.payload.get("tracks", []):
            tid = int(track["id"])
            seen.add(tid)
            bbox = list(track["bbox"])
            x1, y1, x2, y2 = bbox
            fh, fw = self._frame_size
            area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / max(fw * fh, 1)
            rec = self.tracks.get(tid)
            if rec is None:
                rec = self.tracks[tid] = {
                    "state": "PRESENCE", "first_ts": now, "near_since": None,
                    "engaged_since": None, "identity": None, "bbox": bbox,
                    "last_seen": now,
                }
                self.log.info("Track present in scene", track_id=tid, state="PRESENCE")
            rec["bbox"] = bbox
            rec["last_seen"] = now
            near = self._near_enough(bbox, area_frac)
            if near:
                if rec["state"] == "PRESENCE":
                    rec["near_since"] = rec["near_since"] or now
                    if now - rec["near_since"] >= float(self.config.get("near_dwell_s", 1.0)):
                        self._transition(tid, rec, "NEAR")
                        rec["engaged_since"] = now  # engage dwell starts at NEAR entry
                elif rec["state"] == "NEAR":
                    rec["engaged_since"] = rec["engaged_since"] or now
                    if now - rec["engaged_since"] >= float(self.config.get("engage_dwell_s", 2.0)):
                        self._transition(tid, rec, "ENGAGED")
                # ENGAGED stays ENGAGED while near
            else:
                rec["near_since"] = None
                rec["engaged_since"] = None
                if rec["state"] == "ENGAGED":
                    self._transition(tid, rec, "NEAR")  # stepping away demotes engagement
            out.append({
                "id": tid, "bbox": bbox, "state": rec["state"],
                "identity": rec.get("identity"), "area_frac": round(area_frac, 4),
            })
        # scene-level departure: tracks gone from ticks for departed_after_s
        departed = [tid for tid, rec in self.tracks.items() if tid not in seen and
                    now - rec["last_seen"] >= float(self.config.get("departed_after_s", 10.0))]
        for tid in departed:
            self.log.info("Track departed scene", track_id=tid)
            self.tracks.pop(tid, None)
        self.metrics.observe("scene.active_tracks", len(seen))
        self.metrics.inc("scene.ticks")
        await self.bus.publish(Event("TrackStates", {"tracks": out}))

    async def _on_face_matched(self, event: Event) -> None:
        tid = event.payload.get("track_id")
        rec = self.tracks.get(tid)
        if rec is not None:
            rec["identity"] = event.payload.get("identity")
            self.log.info("Scene track identity updated", track_id=tid, identity=rec["identity"])
