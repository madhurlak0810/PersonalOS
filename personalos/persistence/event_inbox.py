"""Idempotent consumers: an event applied once, however often it is delivered.

The outbox delivers at least once (`personalos.persistence.event_outbox`), so
every consumer that changes state has to make a second delivery harmless. The
general way to do that is an inbox: in the same transaction as the effect,
insert `(consumer, dedupe_key)` into `consumed_events`, whose unique
constraint means the effect and the inbox row commit together or not at all.

- A redelivery after a committed effect finds the inbox row and does nothing.
- A redelivery after a crash *before* the commit finds no row, because the
  effect never landed either, and applies it -- for the first time.
- Two deliveries racing each other both run the effect, but only one can
  commit: the other's inbox insert violates the constraint and its whole
  transaction, effect included, is rolled back.

Keyed on `dedupe_key`, never `event_id`: a producer that retries mints a new
`event_id` for the same fact.

`consumer` namespaces the key, so two consumers of one event each apply it
once rather than the first to run consuming it for both.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.events import EventEnvelope
from personalos.persistence.models import ConsumedEventModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: The effect a consumer applies: given its session and the event, stage the
#: change (and, if it publishes follow-on events, those too, via
#: `stage_event`) without committing.
Effect = Callable[[Session, EventEnvelope], T]


@dataclass(frozen=True)
class Consumption(Generic[T]):
    """What one delivery to a consumer did."""

    #: True when this delivery applied the effect; False for a duplicate.
    applied: bool
    #: The effect's return value, when it was applied.
    result: T | None = None


class IdempotentConsumer:
    """Applies an event's effect at most once per `(consumer, dedupe_key)`."""

    def __init__(self, session_factory: Callable[[], Session], consumer: str):
        """Initialize with a session factory and the consumer's stable name."""
        if not consumer.strip():
            raise ValueError("a consumer needs a name")
        self.session_factory = session_factory
        self.consumer = consumer.strip()

    def consume(self, envelope: EventEnvelope, effect: Effect[T]) -> Consumption[T]:
        """Apply `effect` for `envelope` unless this consumer already has.

        Exceptions from `effect` roll the transaction back and propagate, so
        the event is neither applied nor marked consumed and the outbox will
        offer it again.
        """
        session = self.session_factory()
        try:
            if self._seen(session, envelope.dedupe_key):
                return self._duplicate(envelope)
            result = effect(session, envelope)
            session.add(
                ConsumedEventModel(
                    consumer=self.consumer,
                    dedupe_key=envelope.dedupe_key,
                    event_id=envelope.event_id,
                    event_type=envelope.type,
                )
            )
            try:
                session.commit()
            except IntegrityError:
                # Lost the race to a concurrent delivery of the same fact,
                # whose commit is the one that counts. Everything staged here
                # -- the effect included -- goes with this rollback.
                session.rollback()
                if not self._seen(session, envelope.dedupe_key):
                    raise
                return self._duplicate(envelope)
            return Consumption(applied=True, result=result)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def has_consumed(self, dedupe_key: str) -> bool:
        """Whether this consumer has applied the event with `dedupe_key`."""
        session = self.session_factory()
        try:
            return self._seen(session, dedupe_key)
        finally:
            session.close()

    def _seen(self, session: Session, dedupe_key: str) -> bool:
        return (
            session.query(ConsumedEventModel.id)
            .filter(
                ConsumedEventModel.consumer == self.consumer,
                ConsumedEventModel.dedupe_key == dedupe_key,
            )
            .first()
            is not None
        )

    def _duplicate(self, envelope: EventEnvelope) -> Consumption:
        logger.info(
            "%s already consumed %s (%s); delivery of event %s is a no-op",
            self.consumer,
            envelope.dedupe_key,
            envelope.type,
            envelope.event_id,
        )
        return Consumption(applied=False)


__all__ = ["Consumption", "Effect", "IdempotentConsumer"]
