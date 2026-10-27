"""The contract every event transport passes: the Postgres outbox, the in-memory one, the next one.

Each test runs once per transport in `TRANSPORTS`:

- `memory`: `personalos.events.memory.InMemoryEventTransport`;
- `outbox-sqlite`: `EventOutbox` over SQLite, the conditional-UPDATE claim;
- `outbox-postgres`: `EventOutbox` over a scratch schema on the configured
  Postgres, the `SELECT ... FOR UPDATE SKIP LOCKED` claim. Skipped when no
  Postgres is reachable (CI runs one).

A new transport (NATS JetStream, Redis Streams) joins by adding a builder to
`TRANSPORTS`; it is done when this file passes for it unchanged. What the
suite asserts, and what it deliberately does not, is written up in
`docs/EVENT_TRANSPORT.md` -- in particular:

- **Ordering** is asserted only for one consumer settling one event at a
  time: ready events come out in publish order, and an event backing off
  does not hold up the ones behind it. Nothing is asserted across concurrent
  consumers, and handlers must not depend on order.
- **Delivery is at-least-once.** An unsettled lease lapses and the event
  comes back; an acked one never does. Exactly-once *effects* are the
  consumer's job (`IdempotentConsumer`), shown here by the dispatcher test
  that redelivers into a deduplicating handler.
- **Publish deduplicates** an immediate re-publish of the same `dedupe_key`.
  That is the minimum a broker with a dedupe window also meets; consumers
  still deduplicate.
- **Nack** redelivers after backoff, dead-letters a spent or terminal
  failure, and quarantines a poison one; a stale delivery settles nothing.
"""

import asyncio
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from apps.worker.outbox_dispatcher import OutboxDispatcher
from personalos.config import settings
from personalos.domain.errors import ProviderPermanentFailure, RetryableFailure, ValidationFailed
from personalos.domain.event_transport import (
    EventConsumer,
    EventPublisher,
    EventTransport,
    FailureOutcome,
)
from personalos.domain.events import EventEnvelope
from personalos.domain.retry_policy import EVENT_DELIVERY
from personalos.events.memory import InMemoryEventTransport
from personalos.persistence.event_outbox import EventOutbox
from personalos.persistence.models import (
    Base,
    OutboxEventModel,
    UserModel,
    WorkflowModel,
    WorkflowRunModel,
)

T0 = datetime(2026, 10, 27, 9, 0)
VISIBILITY_TIMEOUT = timedelta(minutes=5)
MAX_ATTEMPTS = 3
#: Every jittered backoff at the bottom of its window, so delays are exact.
NO_JITTER: Callable[[], float] = lambda: 0.0  # noqa: E731


class Clock:
    """A clock the test moves by hand, so leases lapse and backoffs pass without waiting."""

    def __init__(self, now: datetime = T0):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@dataclass
class Harness:
    """One transport under test, and the clock it reads."""

    transport: EventTransport
    clock: Clock

    def lapse_lease(self) -> None:
        self.clock.advance(VISIBILITY_TIMEOUT + timedelta(seconds=1))


def _options(clock: Clock) -> dict:
    return {
        "visibility_timeout": VISIBILITY_TIMEOUT,
        "retry_policy": EVENT_DELIVERY.with_max_attempts(MAX_ATTEMPTS),
        "clock": clock,
        "rand": NO_JITTER,
    }


def _memory(clock: Clock, tmp_path) -> Iterator[EventTransport]:
    yield InMemoryEventTransport(**_options(clock))


def _outbox_sqlite(clock: Clock, tmp_path) -> Iterator[EventTransport]:
    # A busy timeout so racing consumers queue on SQLite's file lock instead of
    # failing with "database is locked".
    engine = create_engine(f"sqlite:///{tmp_path / 'contract.db'}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    try:
        yield EventOutbox(sessionmaker(bind=engine), **_options(clock))
    finally:
        engine.dispose()


def _outbox_postgres(clock: Clock, tmp_path) -> Iterator[EventTransport]:
    engine = create_engine(settings.database_url, pool_size=10)
    if engine.dialect.name != "postgresql":
        pytest.skip("the Postgres outbox contract run needs a Postgres DATABASE_URL")
    try:
        with engine.connect():
            pass
    except OperationalError:
        engine.dispose()
        pytest.skip(f"No reachable Postgres at {settings.database_url!r}")
    schema = f"transport_contract_{uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    scoped = engine.execution_options(schema_translate_map={None: schema})
    # Dead-lettering looks up the workflow an event belongs to, so the
    # workflow tables are needed alongside the outbox.
    Base.metadata.create_all(
        scoped,
        tables=[
            OutboxEventModel.__table__,
            UserModel.__table__,
            WorkflowModel.__table__,
            WorkflowRunModel.__table__,
        ],
    )
    try:
        yield EventOutbox(sessionmaker(bind=scoped), **_options(clock))
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        engine.dispose()


TRANSPORTS = {
    "memory": _memory,
    "outbox-sqlite": _outbox_sqlite,
    "outbox-postgres": _outbox_postgres,
}


@pytest.fixture(params=sorted(TRANSPORTS))
def harness(request, tmp_path) -> Iterator[Harness]:
    clock = Clock()
    builder = TRANSPORTS[request.param](clock, tmp_path)
    yield Harness(next(builder), clock)
    builder.close()


def _envelope(dedupe_key: str | None = None, **overrides) -> EventEnvelope:
    fields = {
        "type": "application.created",
        "source": "contract",
        "occurred_at": T0 - timedelta(hours=1),
        "dedupe_key": dedupe_key or f"fact:{uuid4()}",
        "payload": {"application_id": str(uuid4())},
    }
    return EventEnvelope(**{**fields, **overrides})


# --- Shape -----------------------------------------------------------------------


def test_the_transport_implements_both_ports(harness: Harness):
    assert isinstance(harness.transport, EventPublisher)
    assert isinstance(harness.transport, EventConsumer)


def test_nothing_published_means_nothing_to_claim(harness: Harness):
    assert harness.transport.claim() is None


def test_the_envelope_is_delivered_exactly_as_published(harness: Harness):
    envelope = _envelope(
        subject_ref=f"application:{uuid4()}",
        correlation_id=uuid4(),
        raw_ref="gmail:msg-1",
        payload={"status": "applied", "nested": {"n": 1, "tags": ["a", "b"]}},
    )
    published = harness.transport.publish(envelope)

    delivery = harness.transport.claim()

    assert published.event_id == envelope.event_id and not published.duplicate
    assert delivery is not None
    assert delivery.envelope == envelope
    assert delivery.attempt == 1


# --- Dedupe ----------------------------------------------------------------------


def test_republishing_a_fact_stores_nothing_and_delivers_it_once(harness: Harness):
    first = _envelope("fact:1")
    again = _envelope("fact:1", payload={"application_id": "different copy, same fact"})

    harness.transport.publish(first)
    duplicate = harness.transport.publish(again)

    assert duplicate.duplicate
    assert duplicate.event_id == first.event_id
    delivery = harness.transport.claim()
    assert delivery.envelope == first
    assert harness.transport.ack(delivery)
    assert harness.transport.claim() is None


def test_distinct_facts_are_delivered_separately_even_with_equal_payloads(harness: Harness):
    payload = {"application_id": "same"}
    harness.transport.publish(_envelope("fact:a", payload=payload))
    harness.transport.publish(_envelope("fact:b", payload=payload))

    keys = set()
    while (delivery := harness.transport.claim()) is not None:
        keys.add(delivery.envelope.dedupe_key)
        harness.transport.ack(delivery)

    assert keys == {"fact:a", "fact:b"}


def test_a_fact_republished_after_it_was_delivered_is_not_delivered_again(harness: Harness):
    harness.transport.publish(_envelope("fact:1"))
    harness.transport.ack(harness.transport.claim())

    assert harness.transport.publish(_envelope("fact:1")).duplicate
    assert harness.transport.claim() is None


# --- At-least-once ---------------------------------------------------------------


def test_an_acked_event_is_never_delivered_again(harness: Harness):
    harness.transport.publish(_envelope())
    assert harness.transport.ack(harness.transport.claim())

    harness.lapse_lease()

    assert harness.transport.claim() is None


def test_a_leased_event_is_not_handed_to_anyone_else_while_the_lease_holds(harness: Harness):
    harness.transport.publish(_envelope())
    assert harness.transport.claim() is not None

    harness.clock.advance(VISIBILITY_TIMEOUT - timedelta(seconds=1))

    assert harness.transport.claim() is None


def test_an_unsettled_event_is_redelivered_once_its_lease_lapses(harness: Harness):
    envelope = _envelope()
    harness.transport.publish(envelope)
    harness.transport.claim()  # ...and the consumer dies without settling it.

    harness.lapse_lease()
    redelivery = harness.transport.claim()

    assert redelivery is not None
    assert redelivery.envelope == envelope
    assert redelivery.attempt == 2
    assert harness.transport.ack(redelivery)


def test_an_event_whose_leases_keep_lapsing_is_dead_lettered(harness: Harness):
    harness.transport.publish(_envelope())
    for _ in range(MAX_ATTEMPTS):
        assert harness.transport.claim() is not None
        harness.lapse_lease()

    # The claim that would be attempt MAX_ATTEMPTS + 1 parks the event instead.
    assert harness.transport.claim() is None
    harness.lapse_lease()
    assert harness.transport.claim() is None


# --- Nack, retry, dead-letter ----------------------------------------------------


def test_a_nacked_event_is_redelivered_after_its_backoff(harness: Harness):
    envelope = _envelope()
    harness.transport.publish(envelope)

    failure = harness.transport.nack(harness.transport.claim(), RetryableFailure("blip"))

    assert failure.outcome is FailureOutcome.RETRY_SCHEDULED
    assert failure.delay and failure.delay > 0
    harness.clock.advance(timedelta(seconds=failure.delay) - timedelta(milliseconds=10))
    assert harness.transport.claim() is None

    harness.clock.advance(timedelta(milliseconds=20))
    redelivery = harness.transport.claim()
    assert redelivery is not None
    assert redelivery.envelope == envelope
    assert redelivery.attempt == 2


def test_a_nack_waits_at_least_as_long_as_the_error_s_retry_after(harness: Harness):
    harness.transport.publish(_envelope())

    failure = harness.transport.nack(harness.transport.claim(), RetryableFailure(retry_after=90))

    assert failure.outcome is FailureOutcome.RETRY_SCHEDULED
    assert failure.delay >= 90
    harness.clock.advance(timedelta(seconds=89))
    assert harness.transport.claim() is None


def test_successive_failures_back_off_further_each_time(harness: Harness):
    harness.transport.publish(_envelope())
    delays = []
    for _ in range(MAX_ATTEMPTS - 1):
        failure = harness.transport.nack(harness.transport.claim(), RetryableFailure())
        delays.append(failure.delay)
        harness.clock.advance(timedelta(seconds=failure.delay + 1))

    assert delays == sorted(delays) and delays[0] < delays[-1]


def test_the_last_allowed_failure_dead_letters_the_event(harness: Harness):
    harness.transport.publish(_envelope())
    outcomes = []
    for _ in range(MAX_ATTEMPTS):
        failure = harness.transport.nack(harness.transport.claim(), RetryableFailure())
        outcomes.append(failure.outcome)
        harness.clock.advance(timedelta(seconds=(failure.delay or 0) + 1))

    assert outcomes == [FailureOutcome.RETRY_SCHEDULED] * (MAX_ATTEMPTS - 1) + [
        FailureOutcome.DEAD_LETTERED
    ]
    harness.lapse_lease()
    assert harness.transport.claim() is None


def test_a_terminal_failure_dead_letters_without_a_retry(harness: Harness):
    harness.transport.publish(_envelope())

    failure = harness.transport.nack(
        harness.transport.claim(), ProviderPermanentFailure(provider="gmail")
    )

    assert failure.outcome is FailureOutcome.DEAD_LETTERED
    harness.lapse_lease()
    assert harness.transport.claim() is None


def test_a_poison_event_is_quarantined_on_its_first_failure(harness: Harness):
    harness.transport.publish(_envelope())

    failure = harness.transport.nack(harness.transport.claim(), ValidationFailed("bad payload"))

    assert failure.outcome is FailureOutcome.QUARANTINED
    harness.lapse_lease()
    assert harness.transport.claim() is None


def test_an_explicit_dead_letter_parks_the_event_whatever_its_budget(harness: Harness):
    harness.transport.publish(_envelope())

    assert harness.transport.dead_letter(harness.transport.claim(), "operator says stop")

    harness.lapse_lease()
    assert harness.transport.claim() is None


def test_an_event_failing_does_not_hold_up_the_ones_behind_it(harness: Harness):
    harness.transport.publish(_envelope("fact:first"))
    harness.transport.publish(_envelope("fact:second"))

    harness.transport.nack(harness.transport.claim(), RetryableFailure())
    behind = harness.transport.claim()

    assert behind is not None and behind.envelope.dedupe_key == "fact:second"


# --- Stale deliveries -----------------------------------------------------------


def test_a_delivery_reclaimed_by_another_consumer_cannot_be_settled_by_the_first(
    harness: Harness,
):
    harness.transport.publish(_envelope())
    stale = harness.transport.claim()
    harness.lapse_lease()
    current = harness.transport.claim()

    assert harness.transport.ack(stale) is False
    assert harness.transport.nack(stale, RetryableFailure()).outcome is FailureOutcome.CLAIM_LOST
    assert harness.transport.dead_letter(stale, "too late") is False
    # The current holder's lease is untouched by any of that.
    assert harness.transport.ack(current) is True


def test_a_lapsed_lease_nobody_reclaimed_can_still_be_settled(harness: Harness):
    harness.transport.publish(_envelope())
    delivery = harness.transport.claim()

    harness.lapse_lease()

    assert harness.transport.ack(delivery) is True
    assert harness.transport.claim() is None


# --- Ordering --------------------------------------------------------------------


def test_one_consumer_sees_ready_events_in_publish_order(harness: Harness):
    keys = [f"fact:{n}" for n in range(5)]
    for key in keys:
        harness.transport.publish(_envelope(key))

    seen = []
    while (delivery := harness.transport.claim()) is not None:
        seen.append(delivery.envelope.dedupe_key)
        harness.transport.ack(delivery)

    assert seen == keys


# --- Concurrency -----------------------------------------------------------------


def test_concurrent_consumers_never_hold_the_same_event_at_once(harness: Harness):
    rows = 30
    for n in range(rows):
        harness.transport.publish(_envelope(f"race:{n}"))
    seen: list[str] = []
    lock = threading.Lock()

    def consume():
        while (delivery := harness.transport.claim()) is not None:
            with lock:
                seen.append(delivery.envelope.dedupe_key)
            harness.transport.ack(delivery)

    workers = [threading.Thread(target=consume) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)

    assert sorted(seen) == sorted(f"race:{n}" for n in range(rows))


# --- The dispatcher, over any transport -------------------------------------------


def test_the_dispatcher_delivers_and_acks_through_any_transport(harness: Harness):
    harness.transport.publish(_envelope("fact:1"))
    harness.transport.publish(_envelope("fact:2", type="something.else"))
    handled: list[str] = []

    async def handler(envelope: EventEnvelope):
        handled.append(envelope.dedupe_key)

    dispatcher = OutboxDispatcher(harness.transport, {"application.created": handler})
    delivered = asyncio.run(dispatcher.drain())

    # `something.else` has no handler and no default: acknowledged, not stuck.
    assert delivered == 2
    assert handled == ["fact:1"]
    harness.lapse_lease()
    assert harness.transport.claim() is None


def test_a_redelivery_into_a_deduplicating_handler_applies_the_effect_once(harness: Harness):
    harness.transport.publish(_envelope("fact:1"))
    applied: list[str] = []
    consumed: set[str] = set()

    async def handler(envelope: EventEnvelope):
        # The inbox pattern in miniature: the effect and the record of having
        # applied it are one step, keyed on the fact rather than the copy.
        if envelope.dedupe_key not in consumed:
            consumed.add(envelope.dedupe_key)
            applied.append(envelope.dedupe_key)

    # The first delivery applies the effect, then its worker dies before acking.
    first = harness.transport.claim()
    asyncio.run(handler(first.envelope))
    harness.lapse_lease()

    delivered = asyncio.run(OutboxDispatcher(harness.transport, {}, default=handler).drain())

    assert delivered == 1
    assert applied == ["fact:1"]


def test_a_handler_failure_goes_back_through_nack(harness: Harness):
    harness.transport.publish(_envelope())
    calls = 0

    async def flaky(envelope: EventEnvelope):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableFailure("first time unlucky")

    dispatcher = OutboxDispatcher(harness.transport, {}, default=flaky)
    assert asyncio.run(dispatcher.process_next())
    assert asyncio.run(dispatcher.process_next()) is False  # backing off

    harness.lapse_lease()
    assert asyncio.run(dispatcher.process_next())
    assert calls == 2
    harness.lapse_lease()
    assert harness.transport.claim() is None
