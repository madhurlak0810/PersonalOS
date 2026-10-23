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

**A failed delivery backs off, then dead-letters.** `fail` puts the row back
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

Rows staged before envelopes existed (plain `type` + payload, written by
`OutboxEventRepository.create` directly) are read as envelopes too, with the
row's own id, type and creation time standing in for the fields they lack.
"""

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.errors import PersonalOSError, ValidationFailed
from personalos.domain.events import ENVELOPE_KEY, EventEnvelope
from personalos.domain.failures import describe_exception
from personalos.domain.models import OutboxEventStatus
from personalos.domain.redaction import redact_text
from personalos.domain.retry_policy import EVENT_DELIVERY, RetryPolicy, retry_after_seconds
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


class FailureOutcome(str, Enum):
    """What `EventOutbox.fail` did with a failed delivery."""

    #: Back in line, claimable again after `delay` seconds.
    RETRY_SCHEDULED = "retry_scheduled"
    #: Out of attempts (or failed in a way no retry fixes); parked for an operator.
    DEAD_LETTERED = "dead_lettered"
    #: Malformed; parked for an operator without a single retry.
    QUARANTINED = "quarantined"
    #: The claim had lapsed and another dispatcher holds the row; nothing written.
    CLAIM_LOST = "claim_lost"


@dataclass(frozen=True)
class DeliveryFailure:
    """The outcome of recording one failed delivery."""

    outcome: FailureOutcome
    #: The sanitized error the row now carries.
    error: str
    #: Seconds until the row is claimable again, for a scheduled retry.
    delay: float | None = None


class EventOutbox:
    """Claims enveloped events for dispatch and settles them afterwards.

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
                self._dead_letter(
                    claimed,
                    f"the event's lease lapsed {attempts - 1} times without the delivery "
                    f"being settled (the worker crashed or the handler outran the "
                    f"{self.visibility_timeout} visibility timeout)",
                )
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

    def ack(self, claimed: ClaimedEvent) -> bool:
        """Mark a claimed event delivered. False if the claim had already lapsed and moved on."""
        return self._settle(
            claimed,
            {
                OutboxEventModel.status: OutboxEventStatus.DISPATCHED.value,
                OutboxEventModel.dispatched_at: self.clock(),
            },
        )

    def fail(self, claimed: ClaimedEvent, error: BaseException) -> DeliveryFailure:
        """Record a failed delivery: back off and retry, dead-letter, or quarantine.

        A poison event (`is_poison`) is quarantined on its first failure. A
        failure the error itself calls terminal, or one on the last attempt the
        policy allows, is dead-lettered. Anything else is put back in line
        after the policy's backoff, or the error's `Retry-After` if longer.
        """
        text = describe_exception(error)
        if is_poison(error):
            settled = self._park_claim(claimed, OutboxEventStatus.QUARANTINED, text)
            logger.error(
                "outbox event %s (%s) is malformed; quarantined without a retry: %s",
                claimed.outbox_id,
                claimed.envelope.type,
                text,
            )
            return DeliveryFailure(
                FailureOutcome.QUARANTINED if settled else FailureOutcome.CLAIM_LOST, text
            )

        if is_terminal(error) or not self.retry_policy.allows_retry(claimed.attempt):
            settled = self._dead_letter(claimed, text)
            return DeliveryFailure(
                FailureOutcome.DEAD_LETTERED if settled else FailureOutcome.CLAIM_LOST, text
            )

        delay = self.retry_policy.delay(
            claimed.attempt, retry_after=retry_after_seconds(error), rand=self.rand
        )
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
        return DeliveryFailure(FailureOutcome.RETRY_SCHEDULED, text, delay)

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
]
