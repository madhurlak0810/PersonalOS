"""The ports every event transport implements, and the rules they share.

Producers publish `EventEnvelope`s through an `EventPublisher`; workers take
them from an `EventConsumer` and settle each delivery. Nothing that publishes
or handles an event knows what carries it -- the Postgres outbox
(`personalos.persistence.event_outbox.EventOutbox`) today, the in-memory
transport (`personalos.events.memory.InMemoryEventTransport`) in unit tests,
and a NATS JetStream or Redis Streams adapter later. Both implementations
pass the same contract suite (`tests/contracts/test_event_transport.py`);
`docs/EVENT_TRANSPORT.md` is the prose version of it, and of what changes
when the transport does.

The contract, in the order a delivery meets it:

- **Publish deduplicates on `dedupe_key`.** Publishing a fact that is already
  in the transport stores nothing and reports `duplicate=True`. That is a
  courtesy to producers, not the correctness guarantee: brokers bound their
  dedupe window, so consumers still deduplicate (`IdempotentConsumer`).
- **Delivery is at-least-once, under a lease.** `claim` hands out one ready
  event and leases it for the consumer's visibility timeout. An event that is
  neither acked nor nacked before the lease lapses is delivered again, with
  `attempt` one higher. An acked event is never delivered again.
- **Settling is conditional on the lease.** `ack`, `nack` and `dead_letter`
  on a delivery that has since been claimed again -- its lease lapsed and
  another consumer took the event -- change nothing and say so (`False`,
  `FailureOutcome.CLAIM_LOST`). A lapsed lease nobody has reclaimed yet can
  still be settled by its holder.
- **A nack follows the retry policy.** `classify_failure` decides, the same
  way for every transport: a poison event is quarantined at once, a terminal
  failure or a spent budget is dead-lettered, anything else is redelivered no
  sooner than the policy's backoff (or the error's `Retry-After`).
- **Ordering is FIFO among ready events, for one consumer, and nothing more.**
  A consumer that claims and settles one event at a time sees events in
  publish order. Concurrent consumers, retries and lease lapses all reorder;
  a handler that needs an order must establish it from the payload.

Pure declarations and decisions, like the rest of `domain`; nothing here
reads or writes.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from personalos.domain.errors import PersonalOSError, ValidationFailed
from personalos.domain.events import EventEnvelope
from personalos.domain.failures import describe_exception
from personalos.domain.retry_policy import RetryPolicy, retry_after_seconds


@dataclass(frozen=True)
class Published:
    """What one `publish` did."""

    #: The id the transport holds the fact under: this envelope's `event_id`,
    #: or, for a duplicate, the id of the copy published first.
    event_id: UUID
    #: True when the fact was already in the transport and nothing was stored.
    duplicate: bool = False


@dataclass(frozen=True)
class Delivery:
    """An event a consumer now holds the lease on.

    Transports subclass it to carry their own receipt (a row id and claim
    stamp, a stream entry id); only the transport that handed a delivery out
    can settle it.
    """

    envelope: EventEnvelope
    #: Which delivery of this event this is, counting from 1.
    attempt: int


class FailureOutcome(str, Enum):
    """What a `nack` did with a failed delivery."""

    #: Back in line, claimable again after `delay` seconds.
    RETRY_SCHEDULED = "retry_scheduled"
    #: Out of attempts (or failed in a way no retry fixes); parked for an operator.
    DEAD_LETTERED = "dead_lettered"
    #: Malformed; parked for an operator without a single retry.
    QUARANTINED = "quarantined"
    #: The lease had lapsed and another consumer claimed the event; nothing written.
    CLAIM_LOST = "claim_lost"


@dataclass(frozen=True)
class DeliveryFailure:
    """The outcome of recording one failed delivery."""

    outcome: FailureOutcome
    #: The sanitized error the event now carries.
    error: str
    #: Seconds until the event is claimable again, for a scheduled retry.
    delay: float | None = None


@runtime_checkable
class EventPublisher(Protocol):
    """Where producers hand events to the transport."""

    def publish(self, envelope: EventEnvelope) -> Published:
        """Make `envelope` deliverable, unless its `dedupe_key` already is."""


@runtime_checkable
class EventConsumer(Protocol):
    """Where workers take events from the transport, and settle them.

    A consumer *is* the subscription: which events it sees (the outbox's
    `exclude_types`, a JetStream filter subject, a Redis consumer group) is
    fixed when it is built, and `claim` pulls from that subscription.
    """

    def claim(self) -> Delivery | None:
        """Lease the next ready event, or `None` when none is ready now."""

    def ack(self, delivery: Delivery) -> bool:
        """Mark the delivery done for good. False if the event was claimed again since."""

    def nack(self, delivery: Delivery, error: BaseException) -> DeliveryFailure:
        """Record a failed delivery; retry, dead-letter or quarantine it per `classify_failure`."""

    def dead_letter(self, delivery: Delivery, reason: BaseException | str) -> bool:
        """Park the event for an operator without another attempt. False if claimed again since."""


@runtime_checkable
class EventTransport(EventPublisher, EventConsumer, Protocol):
    """A transport that both publishes and consumes: what the contract suite tests."""


def is_poison(error: BaseException) -> bool:
    """True when no redelivery could make this failure go away: the event is malformed."""
    return isinstance(error, ValidationFailed)


def is_terminal(error: BaseException) -> bool:
    """True for a classified failure the error itself says not to retry.

    A policy refusal, a revoked integration: delivering again gets the same
    answer. An unclassified exception is not terminal -- it may be a database
    blip -- and gets the policy's retries.
    """
    return isinstance(error, PersonalOSError) and not error.retryable and not is_poison(error)


def classify_failure(
    error: BaseException,
    attempt: int,
    policy: RetryPolicy,
    *,
    rand: Callable[[], float] = random.random,
) -> DeliveryFailure:
    """What a nack of delivery `attempt` should do, before the transport does it.

    A poison event is quarantined on its first failure. A failure the error
    itself calls terminal, or one on the last attempt the policy allows, is
    dead-lettered. Anything else is retried after the policy's backoff, or
    the error's `Retry-After` if longer.
    """
    text = describe_exception(error)
    if is_poison(error):
        return DeliveryFailure(FailureOutcome.QUARANTINED, text)
    if is_terminal(error) or not policy.allows_retry(attempt):
        return DeliveryFailure(FailureOutcome.DEAD_LETTERED, text)
    delay = policy.delay(attempt, retry_after=retry_after_seconds(error), rand=rand)
    return DeliveryFailure(FailureOutcome.RETRY_SCHEDULED, text, delay)


def lapsed_lease_error(attempts: int, visibility_timeout: timedelta) -> str:
    """Why an event whose every lease lapsed unsettled is being dead-lettered."""
    return (
        f"the event's lease lapsed {attempts - 1} times without the delivery "
        f"being settled (the worker crashed or the handler outran the "
        f"{visibility_timeout} visibility timeout)"
    )


__all__ = [
    "Delivery",
    "DeliveryFailure",
    "EventConsumer",
    "EventPublisher",
    "EventTransport",
    "FailureOutcome",
    "Published",
    "classify_failure",
    "is_poison",
    "is_terminal",
    "lapsed_lease_error",
]
