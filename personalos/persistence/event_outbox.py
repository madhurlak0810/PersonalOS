"""Publishing events through `outbox_events`, and handing them to a dispatcher.

**Publishing is part of the producer's transaction.** `stage_event` takes the
producer's own `Session` and adds the outbox row to it without committing, so
the domain mutation and the event are one commit: if it lands, both exist; if
it rolls back, neither does. There is no window in which a crash leaves a
mutation with no event, or an event announcing a mutation that never happened
-- which is the entire reason the outbox is a table and not a message broker
call made after the commit.

**Dispatch is at-least-once, under a lease.** `EventOutbox.claim` takes the
oldest claimable row -- `SELECT ... FOR UPDATE SKIP LOCKED` on Postgres, so
any number of dispatchers poll the same table without blocking on or
double-claiming a row -- and stamps the claim. A dispatcher that dies before
settling leaves the row IN_PROGRESS with a stale claim; once that claim is
older than `visibility_timeout` the row is claimable again. A row can
therefore be *delivered* more than once, and consumers deduplicate on the
envelope's `dedupe_key` (`personalos.persistence.event_inbox`) so that it is
*applied* once.

**A failed delivery backs off, then dead-letters.** `nack` puts the row back
to PENDING with an `available_at` set by the retry policy -- exponential
backoff with jitter, never sooner than a `Retry-After` the handler's error
carried (`personalos.domain.retry_policy`). Once the policy's attempts are
spent the row moves to DEAD_LETTER with its last error, redacted. So does a
row whose claims keep lapsing: a handler that takes its worker down every
time it runs is given up on after the same number of attempts, rather than
killing workers forever.

**A poison event is quarantined at once.** A row whose stored envelope does
not meet the schema -- or whose handler rejects the payload as invalid -- would
fail the same way on every delivery, so it is not retried: it moves to
QUARANTINED on its first failure. Both dead-lettered and quarantined rows keep
the workflow they belong to, when it can be told, and come back on an
operator's `requeue`.

**It is one implementation of the transport ports.** `EventOutbox` is an
`EventPublisher` and an `EventConsumer` (`personalos.domain.event_transport`),
and passes the same contract suite as the in-memory transport. Nothing outside
the persistence modules that implement the outbox touches `outbox_events`
directly -- `tests/architecture` enforces it -- so the transport can change
without its producers and handlers changing; `docs/EVENT_TRANSPORT.md` says
what would.

Rows staged before envelopes existed (plain `type` + payload, written by
`OutboxEventRepository.create` directly) are read as envelopes too, with the
row's own id, type and creation time standing in for the fields they lack.
"""

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.event_transport import (
    Delivery,
    DeliveryFailure,
    FailureOutcome,
    Published,
    classify_failure,
    is_poison,
    is_terminal,
    lapsed_lease_error,
)
from personalos.domain.events import ENVELOPE_KEY, EventEnvelope
from personalos.domain.failures import describe_exception
from personalos.domain.models import OutboxEventStatus
from personalos.domain.redaction import redact_text
from personalos.domain.retry_policy import EVENT_DELIVERY, RetryPolicy
from personalos.persistence.models import OutboxEventModel, WorkflowModel, WorkflowRunModel
from personalos.persistence.repositories import OutboxEventRepository

logger = logging.getLogger(__name__)

#: How long a claim is trusted before the row is presumed abandoned. Longer
#: than any one handler should take; a handler that hands real work to a graph
#: does so by queueing a workflow command, which returns at once.
DEFAULT_VISIBILITY_TIMEOUT = timedelta(minutes=5)

#: Claims before a row is given up on and dead-lettered.
DEFAULT_MAX_ATTEMPTS = EVENT_DELIVERY.max_attempts

#: `EventEnvelope.source` for a row staged without an envelope.
LEGACY_SOURCE = "outbox"

#: Rows an operator may put back in line with `requeue`.
PARKED_STATUSES = (
    OutboxEventStatus.DEAD_LETTER.value,
    OutboxEventStatus.QUARANTINED.value,
    OutboxEventStatus.FAILED.value,
)

#: How many rows one `claim` call parks (quarantines, or dead-letters for
#: lapsed leases) before it gives up for this poll.
_PARK_LIMIT = 16


def stage_event(session: Session, envelope: EventEnvelope) -> OutboxEventModel:
    """Add `envelope` to the outbox inside the caller's open transaction.

    Does not commit: the caller commits it together with the mutation it
    describes. Staging an event whose `dedupe_key` is already in the outbox
    returns the existing row and stages nothing, so a producer that re-runs
    after a crash does not fail its whole transaction on the duplicate; the
    unique constraint on `dedupe_key` still backs this up against a racing
    producer, whose commit then fails as a unit.
    """
    return _stage(session, envelope)[0]


def stage_raw_event(
    session: Session,
    *,
    type: str,
    payload: dict[str, Any],
    dedupe_key: str | None = None,
    skip_duplicate: bool = False,
) -> OutboxEventModel | None:
    """Add an un-enveloped event to the outbox inside the caller's open transaction.

    For producers that predate `EventEnvelope` and whose rows consumers still
    read as a bare payload; the dispatcher reads such a row as an envelope
    built around it (`envelope_from_row`). New producers use `stage_event`.

    A `dedupe_key` already in the outbox fails the caller's commit with an
    `IntegrityError`, unless `skip_duplicate`, which stages nothing and
    returns `None`. Does not commit.
    """
    outbox = OutboxEventRepository(session)
    if skip_duplicate and dedupe_key is not None and outbox.get_by_dedupe_key(dedupe_key):
        return None
    return outbox.create(type=type, payload=payload, dedupe_key=dedupe_key, commit=False)


def _stage(session: Session, envelope: EventEnvelope) -> tuple[OutboxEventModel, bool]:
    """(the row holding `envelope`'s fact, whether it was already there)."""
    outbox = OutboxEventRepository(session)
    existing = outbox.get_by_dedupe_key(envelope.dedupe_key)
    if existing is not None:
        logger.debug(
            "event %s (%s) is already in the outbox as %s",
            envelope.dedupe_key,
            envelope.type,
            existing.id,
        )
        return existing, True
    row = outbox.create(
        id=envelope.event_id,
        type=envelope.type,
        payload={ENVELOPE_KEY: envelope.model_dump(mode="json")},
        dedupe_key=envelope.dedupe_key,
        commit=False,
    )
    return row, False


def envelope_from_row(row: OutboxEventModel) -> EventEnvelope:
    """The envelope an outbox row carries, or one built around a legacy row.

    Raises `ValueError` (a pydantic `ValidationError`) when the stored
    envelope does not meet the schema.
    """
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
class ClaimedEvent(Delivery):
    """An outbox row a dispatcher now holds the lease on."""

    outbox_id: UUID
    #: The claim's token: settling a row is conditional on it, so a dispatcher
    #: whose lease expired cannot settle a row another one has since claimed.
    claimed_at: datetime


class EventOutbox:
    """The Postgres outbox as an event transport: publishes, claims and settles events.

    Each call is its own short transaction from `session_factory`, for the
    reason `WorkflowCommandQueue` gives: a claim made inside a caller's
    long-lived transaction is invisible to the next dispatcher until it
    commits, and so excludes nobody.

    `exclude_types` keeps this dispatcher off rows another consumer owns --
    `workflow.command` rows are run by the workflow-command worker, under
    workflow leases this lease would otherwise duplicate.

    `retry_policy` is the budget and backoff for failed deliveries;
    `max_attempts`, when given, overrides its budget.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        exclude_types: Sequence[str] = (),
        visibility_timeout: timedelta = DEFAULT_VISIBILITY_TIMEOUT,
        retry_policy: RetryPolicy = EVENT_DELIVERY,
        max_attempts: int | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
        rand: Callable[[], float] = random.random,
    ):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        if max_attempts is not None:
            retry_policy = retry_policy.with_max_attempts(max_attempts)
        self.session_factory = session_factory
        self.exclude_types = tuple(exclude_types)
        self.visibility_timeout = visibility_timeout
        self.retry_policy = retry_policy
        self.clock = clock
        self.rand = rand

    @property
    def max_attempts(self) -> int:
        """Deliveries a row gets before it is dead-lettered."""
        return self.retry_policy.max_attempts

    def publish(self, envelope: EventEnvelope) -> Published:
        """Stage and commit an event on its own, for a producer with no mutation to pair it with.

        A producer that has a mutation to pair the event with calls
        `stage_event` inside its own transaction instead.
        """
        session = self.session_factory()
        try:
            row, duplicate = _stage(session, envelope)
            event_id = row.id
            session.commit()
            return Published(event_id=event_id, duplicate=duplicate)
        except IntegrityError:
            # A racing publisher committed the same fact between our lookup
            # and our insert; theirs is the copy that counts.
            session.rollback()
            existing = OutboxEventRepository(session).get_by_dedupe_key(envelope.dedupe_key)
            if existing is None:
                raise
            return Published(event_id=existing.id, duplicate=True)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def claim(self) -> ClaimedEvent | None:
        """Claim the oldest claimable event, or `None` if there is none right now.

        A row whose envelope cannot be read is quarantined here, and one whose
        earlier claims all lapsed unsettled past the attempt budget is
        dead-lettered here; either way the next row is tried.
        """
        for _ in range(_PARK_LIMIT):
            session = self.session_factory()
            try:
                row = OutboxEventRepository(session).claim_next(
                    exclude_types=self.exclude_types,
                    reclaim_after=self.visibility_timeout,
                    now=self.clock(),
                )
                if row is None:
                    return None
                outbox_id, row_type, attempts, claimed_at = (
                    row.id,
                    row.type,
                    row.attempts,
                    row.claimed_at,
                )
                try:
                    envelope = envelope_from_row(row)
                except ValueError as exc:
                    error = describe_exception(exc)
                    raw = (row.payload_json or {}).get(ENVELOPE_KEY)
                    envelope = None
            finally:
                session.close()

            if envelope is None:
                logger.error(
                    "outbox event %s (%s) does not meet the event schema; quarantining it: %s",
                    outbox_id,
                    row_type,
                    error,
                )
                self._park(
                    outbox_id,
                    claimed_at,
                    OutboxEventStatus.QUARANTINED,
                    error,
                    workflow_id=_workflow_hint(raw if isinstance(raw, dict) else {}),
                )
                continue

            claimed = ClaimedEvent(
                outbox_id=outbox_id, envelope=envelope, attempt=attempts, claimed_at=claimed_at
            )
            if attempts > self.max_attempts:
                # Every earlier claim lapsed without being settled: the handler
                # keeps taking its worker down, or outlives the lease.
                self._dead_letter(claimed, lapsed_lease_error(attempts, self.visibility_timeout))
                continue
            if attempts > 1:
                logger.warning(
                    "redelivering outbox event %s (%s), attempt %d of %d",
                    outbox_id,
                    row_type,
                    attempts,
                    self.max_attempts,
                )
            return claimed
        return None

    def ack(self, claimed: Delivery) -> bool:
        """Mark a claimed event delivered. False if the claim had already lapsed and moved on."""
        return self._settle(
            _own(claimed),
            {
                OutboxEventModel.status: OutboxEventStatus.DISPATCHED.value,
                OutboxEventModel.dispatched_at: self.clock(),
            },
        )

    def nack(self, claimed: Delivery, error: BaseException) -> DeliveryFailure:
        """Record a failed delivery: back off and retry, dead-letter, or quarantine.

        `classify_failure` decides which, the same way for every transport.
        """
        claimed = _own(claimed)
        plan = classify_failure(error, claimed.attempt, self.retry_policy, rand=self.rand)
        text = plan.error
        if plan.outcome is FailureOutcome.QUARANTINED:
            settled = self._park_claim(claimed, OutboxEventStatus.QUARANTINED, text)
            logger.error(
                "outbox event %s (%s) is malformed; quarantined without a retry: %s",
                claimed.outbox_id,
                claimed.envelope.type,
                text,
            )
            return plan if settled else DeliveryFailure(FailureOutcome.CLAIM_LOST, text)

        if plan.outcome is FailureOutcome.DEAD_LETTERED:
            settled = self._dead_letter(claimed, text)
            return plan if settled else DeliveryFailure(FailureOutcome.CLAIM_LOST, text)

        delay = plan.delay
        settled = self._settle(
            claimed,
            {
                OutboxEventModel.status: OutboxEventStatus.PENDING.value,
                OutboxEventModel.claimed_at: None,
                OutboxEventModel.available_at: self.clock() + timedelta(seconds=delay),
                OutboxEventModel.last_error: text,
            },
        )
        if not settled:
            return DeliveryFailure(FailureOutcome.CLAIM_LOST, text)
        logger.warning(
            "delivering outbox event %s (%s) failed on attempt %d of %d; retrying in %.1fs: %s",
            claimed.outbox_id,
            claimed.envelope.type,
            claimed.attempt,
            self.max_attempts,
            delay,
            text,
        )
        return plan

    def dead_letter(self, claimed: Delivery, reason: BaseException | str) -> bool:
        """Park a claimed event for an operator now, whatever its retry budget says."""
        text = reason if isinstance(reason, str) else describe_exception(reason)
        return self._dead_letter(_own(claimed), text)

    def requeue(self, outbox_id: UUID) -> OutboxEventModel | None:
        """Put a dead-lettered or quarantined event back in line, with a fresh budget.

        `None` when the row does not exist or is not parked -- it is pending,
        being delivered, or was delivered. Its `last_error` is kept until a
        delivery succeeds or fails again.
        """
        session = self.session_factory()
        try:
            updated = (
                session.query(OutboxEventModel)
                .filter(
                    OutboxEventModel.id == outbox_id,
                    OutboxEventModel.status.in_(PARKED_STATUSES),
                )
                .update(
                    {
                        OutboxEventModel.status: OutboxEventStatus.PENDING.value,
                        OutboxEventModel.attempts: 0,
                        OutboxEventModel.claimed_at: None,
                        OutboxEventModel.available_at: None,
                        OutboxEventModel.failed_at: None,
                    },
                    synchronize_session=False,
                )
            )
            session.commit()
            if updated == 0:
                return None
            row = session.get(OutboxEventModel, outbox_id)
            session.expunge(row)
            logger.info("outbox event %s (%s) requeued by an operator", outbox_id, row.type)
            return row
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dead_letter(self, claimed: ClaimedEvent, error: str) -> bool:
        settled = self._park_claim(claimed, OutboxEventStatus.DEAD_LETTER, error)
        if settled:
            logger.error(
                "dead-lettering outbox event %s (%s) after %d attempts: %s",
                claimed.outbox_id,
                claimed.envelope.type,
                claimed.attempt,
                error,
            )
        return settled

    def _park_claim(self, claimed: ClaimedEvent, status: OutboxEventStatus, error: str) -> bool:
        envelope = claimed.envelope
        return self._park(
            claimed.outbox_id,
            claimed.claimed_at,
            status,
            error,
            workflow_id=_workflow_hint(
                {
                    "payload": envelope.payload,
                    "correlation_id": envelope.correlation_id,
                }
            ),
        )

    def _park(
        self,
        outbox_id: UUID,
        claimed_at: datetime,
        status: OutboxEventStatus,
        error: str,
        *,
        workflow_id: tuple[UUID | None, UUID | None],
    ) -> bool:
        session = self.session_factory()
        try:
            values: dict[Any, Any] = {
                OutboxEventModel.status: status.value,
                OutboxEventModel.last_error: redact_text(error),
                OutboxEventModel.failed_at: self.clock(),
                OutboxEventModel.available_at: None,
            }
            linked = _registered_workflow(session, *workflow_id)
            if linked is not None:
                values[OutboxEventModel.workflow_id] = linked
            return self._update_claim(session, outbox_id, claimed_at, values)
        finally:
            session.close()

    def _settle(self, claimed: ClaimedEvent, values: dict) -> bool:
        session = self.session_factory()
        try:
            return self._update_claim(session, claimed.outbox_id, claimed.claimed_at, values)
        finally:
            session.close()

    @staticmethod
    def _update_claim(
        session: Session, outbox_id: UUID, claimed_at: datetime, values: dict
    ) -> bool:
        updated = (
            session.query(OutboxEventModel)
            .filter(
                OutboxEventModel.id == outbox_id,
                OutboxEventModel.status == OutboxEventStatus.IN_PROGRESS.value,
                OutboxEventModel.claimed_at == claimed_at,
            )
            .update(values, synchronize_session=False)
        )
        session.commit()
        if updated == 0:
            logger.info(
                "outbox event %s was claimed again before this dispatcher settled it", outbox_id
            )
        return updated == 1


def _own(delivery: Delivery) -> ClaimedEvent:
    if not isinstance(delivery, ClaimedEvent):
        raise TypeError(f"the outbox cannot settle a {type(delivery).__name__} it did not hand out")
    return delivery


def _workflow_hint(envelope: dict[str, Any]) -> tuple[UUID | None, UUID | None]:
    """(workflow id named in the payload, correlation id) of a possibly malformed envelope."""
    payload = envelope.get("payload")
    named = payload.get("workflow_id") if isinstance(payload, dict) else None
    return _as_uuid(named), _as_uuid(envelope.get("correlation_id"))


def _as_uuid(value: Any) -> UUID | None:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value)) if value else None
    except (TypeError, ValueError):
        return None


def _registered_workflow(
    session: Session, named: UUID | None, correlation_id: UUID | None
) -> UUID | None:
    """The registered workflow an event belongs to: the one it names, else its correlation's."""
    if named is not None and session.get(WorkflowModel, named) is not None:
        return named
    if correlation_id is None:
        return None
    row = (
        session.query(WorkflowRunModel.workflow_id)
        .filter(WorkflowRunModel.correlation_id == correlation_id)
        .order_by(WorkflowRunModel.created_at.asc())
        .first()
    )
    return row[0] if row else None


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_VISIBILITY_TIMEOUT",
    "LEGACY_SOURCE",
    "PARKED_STATUSES",
    "ClaimedEvent",
    "DeliveryFailure",
    "EventOutbox",
    "FailureOutcome",
    "envelope_from_row",
    "is_poison",
    "is_terminal",
    "stage_event",
    "stage_raw_event",
]
