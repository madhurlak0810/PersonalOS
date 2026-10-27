"""An event transport that lives in one process's memory, for fast unit tests.

`InMemoryEventTransport` is an `EventPublisher` and an `EventConsumer`
(`personalos.domain.event_transport`) with the outbox's semantics and none of
its storage: publish deduplicates on `dedupe_key`, a claim is a lease that
lapses after `visibility_timeout`, a nack backs off or parks the event as
`classify_failure` says, and a stale delivery settles nothing. It passes the
same contract suite as the Postgres outbox (`tests/contracts`), which is the
point of it: the ports are not Postgres-shaped, and a handler or dispatcher
tested against this transport behaves the same against the real one.

What it does not have is durability. Everything it holds is gone with the
process, and two processes share nothing -- it is a test double, not a
broker. Thread-safe, so concurrent consumers in one process can race on it.
"""

import random
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID

from personalos.domain.event_transport import (
    Delivery,
    DeliveryFailure,
    FailureOutcome,
    Published,
    classify_failure,
    lapsed_lease_error,
)
from personalos.domain.events import EventEnvelope
from personalos.domain.failures import describe_exception
from personalos.domain.redaction import redact_text
from personalos.domain.retry_policy import EVENT_DELIVERY, RetryPolicy

#: The outbox's default lease, so tests written against either agree.
DEFAULT_VISIBILITY_TIMEOUT = timedelta(minutes=5)


class InMemoryEventStatus(str, Enum):
    """Where one held event is in its life; the outbox's statuses, minus the legacy one."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DISPATCHED = "dispatched"
    DEAD_LETTER = "dead_letter"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class InMemoryDelivery(Delivery):
    """A delivery from an `InMemoryEventTransport`."""

    #: The lease's token: settling is conditional on it still being current.
    lease: int


@dataclass
class HeldEvent:
    """One event the transport holds, and the state of its delivery."""

    envelope: EventEnvelope
    #: Publish order; claims take the lowest ready one first.
    sequence: int
    status: InMemoryEventStatus = InMemoryEventStatus.PENDING
    attempts: int = 0
    lease: int | None = None
    claimed_at: datetime | None = None
    available_at: datetime | None = None
    last_error: str | None = None


class InMemoryEventTransport:
    """Publishes, claims and settles events in memory, with the outbox's semantics."""

    def __init__(
        self,
        *,
        exclude_types: Sequence[str] = (),
        visibility_timeout: timedelta = DEFAULT_VISIBILITY_TIMEOUT,
        retry_policy: RetryPolicy = EVENT_DELIVERY,
        max_attempts: int | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
        rand: Callable[[], float] = random.random,
    ):
        """Initialize empty; the arguments mean what they mean for `EventOutbox`."""
        if max_attempts is not None:
            retry_policy = retry_policy.with_max_attempts(max_attempts)
        self.exclude_types = frozenset(exclude_types)
        self.visibility_timeout = visibility_timeout
        self.retry_policy = retry_policy
        self.clock = clock
        self.rand = rand
        self._lock = threading.Lock()
        self._by_key: dict[str, HeldEvent] = {}
        self._sequence = 0
        self._leases = 0

    @property
    def max_attempts(self) -> int:
        """Deliveries an event gets before it is dead-lettered."""
        return self.retry_policy.max_attempts

    def publish(self, envelope: EventEnvelope) -> Published:
        """Hold `envelope` for delivery, unless its `dedupe_key` is already held."""
        with self._lock:
            existing = self._by_key.get(envelope.dedupe_key)
            if existing is not None:
                return Published(event_id=existing.envelope.event_id, duplicate=True)
            self._sequence += 1
            self._by_key[envelope.dedupe_key] = HeldEvent(envelope, self._sequence)
            return Published(event_id=envelope.event_id)

    def claim(self) -> InMemoryDelivery | None:
        """Lease the oldest ready event, dead-lettering any whose leases kept lapsing."""
        with self._lock:
            now = self.clock()
            while True:
                held = self._next_ready(now)
                if held is None:
                    return None
                held.attempts += 1
                self._leases += 1
                held.status = InMemoryEventStatus.IN_PROGRESS
                held.lease = self._leases
                held.claimed_at = now
                held.available_at = None
                if held.attempts > self.max_attempts:
                    self._park(
                        held,
                        InMemoryEventStatus.DEAD_LETTER,
                        lapsed_lease_error(held.attempts, self.visibility_timeout),
                    )
                    continue
                return InMemoryDelivery(
                    envelope=held.envelope, attempt=held.attempts, lease=held.lease
                )

    def ack(self, delivery: Delivery) -> bool:
        """Mark the delivery done. False if its lease had lapsed and moved on."""
        with self._lock:
            held = self._current(delivery)
            if held is None:
                return False
            held.status = InMemoryEventStatus.DISPATCHED
            held.lease = None
            return True

    def nack(self, delivery: Delivery, error: BaseException) -> DeliveryFailure:
        """Record a failed delivery: back off and retry, dead-letter, or quarantine."""
        plan = classify_failure(error, delivery.attempt, self.retry_policy, rand=self.rand)
        with self._lock:
            held = self._current(delivery)
            if held is None:
                return DeliveryFailure(FailureOutcome.CLAIM_LOST, plan.error)
            if plan.outcome is FailureOutcome.QUARANTINED:
                self._park(held, InMemoryEventStatus.QUARANTINED, plan.error)
            elif plan.outcome is FailureOutcome.DEAD_LETTERED:
                self._park(held, InMemoryEventStatus.DEAD_LETTER, plan.error)
            else:
                held.status = InMemoryEventStatus.PENDING
                held.lease = None
                held.claimed_at = None
                held.available_at = self.clock() + timedelta(seconds=plan.delay or 0.0)
                held.last_error = plan.error
            return plan

    def dead_letter(self, delivery: Delivery, reason: BaseException | str) -> bool:
        """Park a claimed event for an operator now, whatever its retry budget says."""
        text = reason if isinstance(reason, str) else describe_exception(reason)
        with self._lock:
            held = self._current(delivery)
            if held is None:
                return False
            self._park(held, InMemoryEventStatus.DEAD_LETTER, text)
            return True

    def requeue(self, event_id: UUID) -> bool:
        """Put a parked event back in line with a fresh budget. False if it is not parked."""
        with self._lock:
            for held in self._by_key.values():
                if held.envelope.event_id == event_id and held.status in _PARKED:
                    held.status = InMemoryEventStatus.PENDING
                    held.attempts = 0
                    held.lease = None
                    held.claimed_at = None
                    held.available_at = None
                    return True
            return False

    # ------------------------------------------------------------------
    # Inspection, for tests
    # ------------------------------------------------------------------

    def held(self, dedupe_key: str) -> HeldEvent | None:
        """The state of the event held under `dedupe_key`, if any."""
        with self._lock:
            return self._by_key.get(dedupe_key)

    def events(self, status: InMemoryEventStatus | None = None) -> list[HeldEvent]:
        """Held events in publish order; only those in `status` when given."""
        with self._lock:
            ordered = sorted(self._by_key.values(), key=lambda held: held.sequence)
            return [held for held in ordered if status is None or held.status is status]

    # ------------------------------------------------------------------
    # Internals; callers hold the lock
    # ------------------------------------------------------------------

    def _next_ready(self, now: datetime) -> HeldEvent | None:
        lapsed_before = now - self.visibility_timeout
        ready = [
            held
            for held in self._by_key.values()
            if held.envelope.type not in self.exclude_types
            and (
                (
                    held.status is InMemoryEventStatus.PENDING
                    and (held.available_at is None or held.available_at <= now)
                )
                or (
                    held.status is InMemoryEventStatus.IN_PROGRESS
                    and held.claimed_at is not None
                    and held.claimed_at <= lapsed_before
                )
            )
        ]
        return min(ready, key=lambda held: held.sequence, default=None)

    def _current(self, delivery: Delivery) -> HeldEvent | None:
        if not isinstance(delivery, InMemoryDelivery):
            raise TypeError(
                f"this transport cannot settle a {type(delivery).__name__} it did not hand out"
            )
        held = self._by_key.get(delivery.envelope.dedupe_key)
        if (
            held is None
            or held.status is not InMemoryEventStatus.IN_PROGRESS
            or held.lease != delivery.lease
        ):
            return None
        return held

    @staticmethod
    def _park(held: HeldEvent, status: InMemoryEventStatus, error: str) -> None:
        held.status = status
        held.lease = None
        held.available_at = None
        held.last_error = redact_text(error)


_PARKED = frozenset({InMemoryEventStatus.DEAD_LETTER, InMemoryEventStatus.QUARANTINED})


__all__ = [
    "DEFAULT_VISIBILITY_TIMEOUT",
    "HeldEvent",
    "InMemoryDelivery",
    "InMemoryEventStatus",
    "InMemoryEventTransport",
]
