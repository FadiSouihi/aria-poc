"""Typed publish/subscribe with bounded queues and explicit overflow policy.

Design (ROADMAP §6.1):
- subscriptions are per event name with a per-subscriber bounded queue;
- overflow policy is explicit per subscription:
    ``drop_new``    - drop the incoming event (telemetry-style, lossy OK);
    ``drop_oldest`` - evict the oldest queued event (high-rate frames);
    ``block``       - await until space (events that must not be lost);
- every publish is counted, tapped into the timeline, and the correlation
  context snapshot is stamped onto the event when the publisher did not;
- a failing subscriber callback is logged and counted, never propagated to
  the publisher.

Lifecycle: create the bus inside the event loop, call ``await bus.start()``
before services publish (pre-start publishes simply buffer in queues).
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Awaitable, Callable, Dict, Iterable, List, Optional

from aria.core.context import snapshot
from aria.core.events import Event, REGISTRY
from aria.core.telemetry.logging import get_logger
from aria.core.telemetry.metrics import Metrics

Subscriber = Callable[[Event], Awaitable[None]]
Tap = Callable[[Event], None]

POLICIES = ("drop_new", "drop_oldest", "block")


class _Subscription:
    __slots__ = ("event", "callback", "policy", "queue", "dropped", "task")

    def __init__(self, event: str, callback: Subscriber, policy: str, maxsize: int) -> None:
        self.event = event
        self.callback = callback
        self.policy = policy
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.task: Optional[asyncio.Task] = None


class EventBus:
    def __init__(self, metrics: Optional[Metrics] = None, maxsize: int = 256) -> None:
        self.metrics = metrics or Metrics()
        self._default_maxsize = int(maxsize)
        self._subs: Dict[str, List[_Subscription]] = {}
        self._taps: List[Tap] = []
        self._started = False
        self.published = 0
        self.dropped = 0
        self._warned_unknown: set = set()
        self.log = get_logger("aria.core.event_bus")

    # -- wiring ----------------------------------------------------------
    def subscribe(
        self,
        event: str,
        callback: Subscriber,
        policy: str = "drop_new",
        maxsize: Optional[int] = None,
    ) -> _Subscription:
        if policy not in POLICIES:
            raise ValueError(f"Unknown overflow policy {policy!r}; expected one of {POLICIES}")
        sub = _Subscription(event, callback, policy, maxsize if maxsize is not None else self._default_maxsize)
        self._subs.setdefault(event, []).append(sub)
        if self._started:
            sub.task = asyncio.get_running_loop().create_task(
                self._dispatch(sub), name=f"bus:{event}"
            )
        self.log.debug("Subscribed", event=event, policy=policy)
        return sub

    def attach_tap(self, tap: Tap) -> None:
        """Taps see every event synchronously in the publish path (keep cheap)."""
        self._taps.append(tap)

    def unsubscribe(self, sub: "_Subscription") -> None:
        """Remove a subscription (services must do this in ``on_stop`` so a
        restart does not create duplicate subscriptions)."""
        subs = self._subs.get(sub.event)
        if subs and sub in subs:
            subs.remove(sub)
        if sub.task is not None:
            sub.task.cancel()
            sub.task = None

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self._started = True
        for subs in self._subs.values():
            for sub in subs:
                if sub.task is None:
                    sub.task = asyncio.get_running_loop().create_task(
                        self._dispatch(sub), name=f"bus:{sub.event}"
                    )

    async def stop(self) -> None:
        self._started = False
        tasks = [sub.task for subs in self._subs.values() for sub in subs if sub.task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- publishing ------------------------------------------------------
    async def publish(self, event: Event) -> None:
        if not event.context:
            event = replace(event, context=snapshot())
        if event.name not in REGISTRY and event.name not in self._warned_unknown:
            self._warned_unknown.add(event.name)
            self.metrics.inc("bus.unregistered_events")
            self.log.warning("Unregistered event name (add it to events.REGISTRY)", event=event.name)

        self.published += 1
        self.metrics.inc("bus.published")
        for tap in self._taps:
            try:
                tap(event)
            except Exception:  # telemetry must never break the pipeline
                self.log.exception("Timeline tap failed", event=event.name)

        for sub in self._subs.get(event.name, []):
            await self._enqueue(sub, event)

    async def _enqueue(self, sub: _Subscription, event: Event) -> None:
        queue = sub.queue
        if sub.policy == "block":
            await queue.put(event)
            return
        if queue.full():
            if sub.policy == "drop_oldest":
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - raced empty
                    pass
                self.dropped += 1
                sub.dropped += 1
                self.metrics.inc("bus.dropped")
            else:  # drop_new
                self.dropped += 1
                sub.dropped += 1
                self.metrics.inc("bus.dropped")
                return
        queue.put_nowait(event)

    async def _dispatch(self, sub: _Subscription) -> None:
        while True:
            event = await sub.queue.get()
            try:
                await sub.callback(event)
            except Exception:
                self.metrics.inc("bus.callback_errors")
                self.log.exception("Subscriber callback failed", event=sub.event)

    # -- introspection ---------------------------------------------------
    def snapshot_stats(self) -> dict:
        return {
            "published": self.published,
            "dropped": self.dropped,
            "subscriptions": {name: len(subs) for name, subs in self._subs.items()},
            "unregistered_seen": sorted(self._warned_unknown),
        }
