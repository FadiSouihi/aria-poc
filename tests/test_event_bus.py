"""Contract tests: EventBus pub/sub, backpressure policies, taps, context."""
import asyncio

from aria.core.context import bind
from aria.core.event_bus import EventBus
from aria.core.events import Event
from helpers import Collector, make_bus, run


def test_pubsub_delivers_in_order_to_every_subscriber():
    async def scenario():
        bus = make_bus(maxsize=16)
        await bus.start()
        c1, c2 = Collector(), Collector()
        bus.subscribe("Frame", c1)
        bus.subscribe("Frame", c2)
        for i in range(5):
            await bus.publish(Event("Frame", {"seq": i}))
        await c1.wait_for(5)
        await c2.wait_for(5)
        await bus.stop()
        assert [e.payload["seq"] for e in c1.events] == [0, 1, 2, 3, 4]
        assert len(c2.events) == 5
    run(scenario())


def test_backpressure_drop_new_keeps_oldest_delivered():
    async def scenario():
        bus = EventBus(maxsize=4)
        await bus.start()
        got = []

        async def slow_consumer(event):
            got.append(event.payload["seq"])
            await asyncio.sleep(0.001)

        bus.subscribe("Frame", slow_consumer, policy="drop_new", maxsize=4)
        for i in range(100):
            await bus.publish(Event("Frame", {"seq": i}))
        await asyncio.sleep(0.2)
        await bus.stop()
        # 1 processed + full queue of 4 delivered; the rest dropped at enqueue.
        assert got == [0, 1, 2, 3]
        assert bus.dropped == 96
        assert got.__len__() + bus.dropped == 100
        assert bus.metrics.counter("bus.dropped") == 96
    run(scenario())


def test_backpressure_drop_oldest_keeps_newest():
    async def scenario():
        bus = EventBus(maxsize=4)
        await bus.start()
        got = []

        async def consumer(event):
            got.append(event.payload["seq"])
            await asyncio.sleep(0.001)

        bus.subscribe("Frame", consumer, policy="drop_oldest", maxsize=4)
        for i in range(100):
            await bus.publish(Event("Frame", {"seq": i}))
        await asyncio.sleep(0.2)
        await bus.stop()
        assert got == [96, 97, 98, 99]  # newest survive
        assert got.__len__() + bus.dropped == 100
    run(scenario())


def test_block_policy_never_drops():
    async def scenario():
        bus = EventBus(maxsize=2)
        await bus.start()
        got = []

        async def consumer(event):
            got.append(event.payload["seq"])
            await asyncio.sleep(0.001)

        bus.subscribe("Frame", consumer, policy="block", maxsize=2)
        for i in range(5):
            await bus.publish(Event("Frame", {"seq": i}))
        await asyncio.sleep(0.2)
        await bus.stop()
        assert got == [0, 1, 2, 3, 4]
        assert bus.dropped == 0
    run(scenario())


def test_unregistered_event_names_are_counted_and_warned_once():
    async def scenario():
        bus = make_bus()
        await bus.start()
        await bus.publish(Event("NotInRegistry", {}))
        await bus.publish(Event("NotInRegistry", {}))
        await bus.stop()
        assert bus.metrics.counter("bus.unregistered_events") == 1  # warned once per name
        assert bus.snapshot_stats()["unregistered_seen"] == ["NotInRegistry"]
    run(scenario())


def test_taps_see_every_event_synchronously():
    async def scenario():
        bus = make_bus()
        await bus.start()
        seen = []
        bus.attach_tap(seen.append)
        await bus.publish(Event("Frame", {"seq": 1}))
        await bus.publish(Event("SessionOpened", {"who": "x"}))
        assert [e.name for e in seen] == ["Frame", "SessionOpened"]  # no await needed
        await bus.stop()
    run(scenario())


def test_events_carry_correlation_context():
    async def scenario():
        bus = make_bus()
        await bus.start()
        col = Collector()
        bus.subscribe("Frame", col)
        with bind(utterance_id="u9", session_id="s1"):
            await bus.publish(Event("Frame", {}))
        await col.wait_for(1)
        assert col.events[0].context == {"session_id": "s1", "utterance_id": "u9"}
        await bus.stop()
    run(scenario())
