"""Publishing events through `outbox_events`, and handing them to a dispatcher.

**Publishing is part of the producer's transaction.** `stage_event` takes the
producer's own `Session` and adds the outbox row to it without committing, so
the domain mutation and the event are one commit: if it lands, both exist; if
it rolls back, neither does. There is no window in which a crash leaves a
mutation with no event, or an event announcing a mutation that never happened
-- which is the entire reason the outbox is a table and not a message broker
call made after the commit.

**Dispatch is at-least-once.** `EventOutbox.claim` takes the oldest claimable
row -- `SELECT ... FOR UPDATE SKIP LOCKED` on Postgres, so any number of
dispatchers poll the same table without blocking on or double-claiming a row
-- and stamps the claim. A dispatcher that dies before `ack` leaves the row
IN_PROGRESS with a stale claim; once that claim is older than
`visibility_timeout` the row is claimable again. A row can therefore be
*delivered* more than once, and consumers deduplicate on the envelope's
`dedupe_key` (`personalos.persistence.event_inbox`) so that it is *applied*
once.

A failed delivery is not put straight back in line: the row stays claimed
until its lease runs out, which spaces retries by the visibility timeout
instead of spinning on a poison event. After `max_attempts` claims it is
marked FAILED and left for an operator.

Rows staged before envelopes existed (plain `type` + payload, written by
`OutboxEventRepository.create` directly) are read as envelopes too, with the
row's own id, type and creation time standing in for the fields they lack.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.events import ENVELOPE_KEY, EventEnvelope
from personalos.domain.models import OutboxEventStatus
from personalos.persistence.models import OutboxEventModel
from personalos.persistence.repositories import OutboxEventRepository

logger = logging.getLogger(__name__)

#: How long a claim is trusted before the row is presumed abandoned. Longer
#: than any one handler should take; a handler that hands real work to a graph
#: does so by queueing a workflow command, which returns at once.
DEFAULT_VISIBILITY_TIMEOUT = timedelta(minutes=5)

#: Claims before a row is given up on and marked FAILED.
DEFAULT_MAX_ATTEMPTS = 5

#: `EventEnvelope.source` for a row staged without an envelope.
LEGACY_SOURCE = "outbox"


def stage_event(session: Session, envelope: EventEnvelope) -> OutboxEventModel:
    """Add `envelope` to the outbox inside the caller's open transaction.

    Does not commit: the caller commits it together with the mutation it
    describes. Staging an event whose `dedupe_key` is already in the outbox
    returns the existing row and stages nothing, so a producer that re-runs
    after a crash does not fail its whole transaction on the duplicate; the
    unique constraint on `dedupe_key` still backs this up against a racing
    producer, whose commit then fails as a unit.
    """
    outbox = OutboxEventRepository(session)
    existing = outbox.get_by_dedupe_key(envelope.dedupe_key)
    if existing is not None:
        logger.debug(
            "event %s (%s) is already in the outbox as %s",
            envelope.dedupe_key,
            envelope.type,
            existing.id,
        )
        return existing
    return outbox.create(
        id=envelope.event_id,
        type=envelope.type,
        payload={ENVELOPE_KEY: envelope.model_dump(mode="json")},
        dedupe_key=envelope.dedupe_key,
        commit=False,
    )


def envelope_from_row(row: OutboxEventModel) -> EventEnvelope:
    """The envelope an outbox row carries, or one built around a legacy row."""
    stored = (row.payload_json or {}).get(ENVELOPE_KEY)
    if isinstance(stored, dict):
        return EventEnvelope.model_validate(stored)
    return EventEnvelope(
        event_id=row.id,
        type=row.type,
        source=LEGACY_SOURCE,
        occurred_at=row.created_at,
        received_at=row.created_at,
        dedupe_key=row.dedupe_key or f"outbox:{row.id}",
        payload=row.payload_json or {},
    )


@dataclass(frozen=True)
class ClaimedEvent:
    """An event a dispatcher now holds the lease on."""

    outbox_id: UUID
    envelope: EventEnvelope
    #: Which claim of this row this is, counting from 1.
    attempt: int
    #: The claim's token: settling a row is conditional on it, so a dispatcher
    #: whose lease expired cannot settle a row another one has since claimed.
    claimed_at: datetime


class EventOutbox:
    """Claims enveloped events for dispatch and settles them afterwards.

    Each call is its own short transaction from `session_factory`, for the
    reason `WorkflowCommandQueue` gives: a claim made inside a caller's
    long-lived transaction is invisible to the next dispatcher until it
    commits, and so excludes nobody.

    `exclude_types` keeps this dispatcher off rows another consumer owns --
    `workflow.command` rows are run by the workflow-command worker, under
    workflow leases this lease would otherwise duplicate.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        exclude_types: Sequence[str] = (),
        visibility_timeout: timedelta = DEFAULT_VISIBILITY_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.session_factory = session_factory
        self.exclude_types = tuple(exclude_types)
        self.visibility_timeout = visibility_timeout
        self.max_attempts = max_attempts
        self.clock = clock

    def publish(self, envelope: EventEnvelope) -> OutboxEventModel:
        """Stage and commit an event on its own, for a producer with no mutation to pair it with."""
        session = self.session_factory()
        try:
            row = stage_event(session, envelope)
            session.commit()
            return row
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def claim(self) -> ClaimedEvent | None:
        """Claim the oldest claimable event, or `None` if there is none right now."""
        session = self.session_factory()
        try:
            row = OutboxEventRepository(session).claim_next(
                exclude_types=self.exclude_types,
                reclaim_after=self.visibility_timeout,
                now=self.clock(),
            )
            if row is None:
                return None
            if row.attempts > 1:
                logger.warning(
                    "redelivering outbox event %s (%s), attempt %d: an earlier claim "
                    "was never settled",
                    row.id,
                    row.type,
                    row.attempts,
                )
            return ClaimedEvent(
                outbox_id=row.id,
                envelope=envelope_from_row(row),
                attempt=row.attempts,
                claimed_at=row.claimed_at,
            )
        finally:
            session.close()

    def ack(self, claimed: ClaimedEvent) -> bool:
        """Mark a claimed event delivered. False if the claim had already lapsed and moved on."""
        return self._settle(
            claimed,
            {
                OutboxEventModel.status: OutboxEventStatus.DISPATCHED.value,
                OutboxEventModel.dispatched_at: self.clock(),
            },
        )

    def nack(self, claimed: ClaimedEvent, *, permanent: bool = False) -> bool:
        """Record that delivering a claimed event failed. True if it was given up on.

        Below `max_attempts` nothing is written: the claim is left to expire,
        and the row comes back after the visibility timeout. At the limit --
        or at once, for a `permanent` failure no redelivery could fix, such as
        a malformed event -- it is marked FAILED.
        """
        if not permanent and claimed.attempt < self.max_attempts:
            return False
        settled = self._settle(claimed, {OutboxEventModel.status: OutboxEventStatus.FAILED.value})
        if settled:
            logger.error(
                "giving up on outbox event %s (%s) after %d attempts",
                claimed.outbox_id,
                claimed.envelope.type,
                claimed.attempt,
            )
        return settled

    def _settle(self, claimed: ClaimedEvent, values: dict) -> bool:
        session = self.session_factory()
        try:
            updated = (
                session.query(OutboxEventModel)
                .filter(
                    OutboxEventModel.id == claimed.outbox_id,
                    OutboxEventModel.status == OutboxEventStatus.IN_PROGRESS.value,
                    OutboxEventModel.claimed_at == claimed.claimed_at,
                )
                .update(values, synchronize_session=False)
            )
            session.commit()
            if updated == 0:
                logger.info(
                    "outbox event %s was claimed again before this dispatcher settled it",
                    claimed.outbox_id,
                )
            return updated == 1
        finally:
            session.close()


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_VISIBILITY_TIMEOUT",
    "LEGACY_SOURCE",
    "ClaimedEvent",
    "EventOutbox",
    "envelope_from_row",
    "stage_event",
]
