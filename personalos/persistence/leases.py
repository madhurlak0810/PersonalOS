"""Exclusive leases on a workflow, so two workers cannot resume the same process.

A durable checkpointer makes a run resumable; it does not make resuming it
*safe*. Two workers that both pick up the same `workflow_id` load the same
checkpoint, compute the same next step, and run it twice -- and the second one
is a duplicate of a real side effect, not a retry of a lost one. So a resume
takes a lease first, and a worker that cannot get the lease does not run.

The lease is a row in `workflow_leases`, one per workflow, and correctness rests
on two database primitives rather than on the code being careful:

1. **A unique constraint on `workflow_id`.** Two workers racing from no lease at
   all both insert; one gets an `IntegrityError`. This is the part that holds on
   SQLite, where `SELECT ... FOR UPDATE` is silently a no-op.
2. **A conditional single-statement `UPDATE`.** Taking over an expired lease is
   an `UPDATE ... WHERE id = ? AND lease_token = ?` whose affected-row count is
   checked: two workers that both see the same expired lease issue the same
   guarded update, and exactly one of them updates a row.

`SELECT ... FOR UPDATE` is still taken where the dialect supports it, because on
Postgres it turns the race into a wait instead of a retry -- but nothing here
depends on it.

Leases expire. A worker killed with `kill -9` never releases, and a lease that
could only be released by its holder would strand that workflow permanently;
one that expires is takeable after `ttl`. The cost of expiry is that a stalled
holder may wake up believing it still holds the lease, which is what
`WorkflowLease.token` is for: every takeover mints a new token, and release and
renewal are rejected unless the token still matches, so the straggler cannot
cancel its successor's lease.
"""

import logging
import os
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from personalos.domain.errors import ErrorCode, PersonalOSError
from personalos.domain.workflow import WorkflowLease
from personalos.persistence.models import WorkflowLeaseModel

logger = logging.getLogger(__name__)

#: Why an acquisition was refused, carried in `WorkflowLeaseUnavailable.details`.
#: Worth distinguishing: `HELD` means the exclusion rule did its job, while
#: `LOCKED` means the database serialized two writers and we inferred exclusion
#: from that. Both are correct answers to "may I run?", but only the first
#: proves the lease itself is working -- which is exactly what a test asserting
#: mutual exclusion has to pin down, or it will pass on a broken lease whenever
#: SQLite happens to be busy.
LEASE_REASON_HELD = "held"
LEASE_REASON_INSERT_RACE = "lost_insert_race"
LEASE_REASON_TAKEOVER_RACE = "lost_takeover_race"
LEASE_REASON_LOCKED = "database_locked"

#: Refusals that came from the lease rule rather than from storage contention.
LEASE_EXCLUSION_REASONS = frozenset(
    {LEASE_REASON_HELD, LEASE_REASON_INSERT_RACE, LEASE_REASON_TAKEOVER_RACE}
)

#: How long a lease is held before another worker may take it over. Long enough
#: that an ordinary super-step -- including a tool call over the network -- does
#: not need to renew, short enough that a workflow orphaned by a hard kill is
#: resumable within a few minutes rather than needing an operator.
DEFAULT_LEASE_TTL_SECONDS = 300


class WorkflowLeaseUnavailable(PersonalOSError):
    """Another worker holds the lease on this workflow.

    Retryable, and not an error in the caller: the workflow is being worked on,
    and the right response is to come back later, not to proceed anyway.
    """

    code = ErrorCode.RETRYABLE
    http_status = 409
    retryable = True
    default_message = "another worker holds the lease on this workflow"


class WorkflowLeaseLost(PersonalOSError):
    """This worker's lease was taken over while it was working.

    Raised on renewal or release when the stored token no longer matches. Not
    retryable: the work this lease authorized is no longer this worker's to do,
    and whatever it has in flight must stop rather than be retried.
    """

    code = ErrorCode.IDEMPOTENCY_CONFLICT
    http_status = 409
    default_message = "this worker's lease on the workflow was taken over"


def default_owner() -> str:
    """Identify this worker in a lease row: `host:pid`.

    Enough to find the process holding a stale lease, which is the only
    question anyone asks of this field.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


class WorkflowLeaseStore:
    """Acquires, renews and releases the lease on a workflow.

    Takes a session factory rather than a session, like
    `SqlOperationStore`: each operation is its own short transaction, so a
    lease is visible to other workers the moment it is taken and is not tied to
    whatever transaction the caller had open.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Initialize with a session factory, a lease duration and a clock.

        `clock` is injected so expiry can be tested without sleeping.
        """
        if ttl_seconds <= 0:
            raise ValueError("lease ttl_seconds must be positive")
        self.session_factory = session_factory
        self.ttl = timedelta(seconds=ttl_seconds)
        self.clock = clock

    def acquire(
        self,
        workflow_id: UUID,
        *,
        owner: str | None = None,
        thread_id: str | None = None,
    ) -> WorkflowLease:
        """Take the lease on a workflow, or raise `WorkflowLeaseUnavailable`.

        Raises rather than blocking or returning `None`: "someone else is
        running this workflow" is a control-flow fact the caller has to handle,
        exactly as `PolicyDenied` is one level up -- and a caller that got
        `None` back could carry on by accident, which is the failure this whole
        module exists to prevent.
        """
        owner = owner or default_owner()
        now = self.clock()
        expires_at = now + self.ttl

        session = self.session_factory()
        try:
            row = self._locked_row(session, workflow_id)

            if row is None:
                lease = WorkflowLease(
                    workflow_id=workflow_id,
                    owner=owner,
                    token=uuid4(),
                    acquired_at=now,
                    expires_at=expires_at,
                    thread_id=thread_id,
                )
                session.add(
                    WorkflowLeaseModel(
                        workflow_id=workflow_id,
                        thread_id=thread_id,
                        owner=owner,
                        lease_token=lease.token,
                        acquired_at=now,
                        expires_at=expires_at,
                        released_at=None,
                    )
                )
                try:
                    session.commit()
                except IntegrityError:
                    # Lost the insert race on `workflow_leases.workflow_id`.
                    session.rollback()
                    raise WorkflowLeaseUnavailable(
                        f"workflow {workflow_id} was leased by another worker while "
                        f"'{owner}' was acquiring it",
                        details={"reason": LEASE_REASON_INSERT_RACE},
                    ) from None
                return lease

            if self._is_held(row, now):
                raise WorkflowLeaseUnavailable(
                    f"workflow {workflow_id} is leased by '{row.owner}' until "
                    f"{row.expires_at.isoformat()}",
                    details={"reason": LEASE_REASON_HELD, "holder": row.owner},
                )

            # Takeover of a released or expired lease. Guarded on the token we
            # just read, so a concurrent takeover of the same row updates zero
            # rows here and is refused rather than granted in parallel.
            token = uuid4()
            updated = (
                session.query(WorkflowLeaseModel)
                .filter(
                    WorkflowLeaseModel.id == row.id,
                    WorkflowLeaseModel.lease_token == row.lease_token,
                )
                .update(
                    {
                        "owner": owner,
                        "thread_id": thread_id,
                        "lease_token": token,
                        "acquired_at": now,
                        "expires_at": expires_at,
                        "released_at": None,
                        "updated_at": now,
                    },
                    synchronize_session=False,
                )
            )
            if not updated:
                session.rollback()
                raise WorkflowLeaseUnavailable(
                    f"workflow {workflow_id} was leased by another worker while "
                    f"'{owner}' was taking over its expired lease",
                    details={"reason": LEASE_REASON_TAKEOVER_RACE},
                )
            session.commit()
            if row.released_at is None:
                logger.warning(
                    "took over expired lease on workflow %s from '%s' (expired %s)",
                    workflow_id,
                    row.owner,
                    row.expires_at.isoformat(),
                )
            return WorkflowLease(
                workflow_id=workflow_id,
                owner=owner,
                token=token,
                acquired_at=now,
                expires_at=expires_at,
                thread_id=thread_id,
            )
        except OperationalError as exc:
            # SQLite serializes writers with a lock rather than a queue, so a
            # genuine simultaneous acquisition surfaces as "database is locked".
            # That is contention, not corruption: the other worker got there
            # first, which is the same answer the unique constraint gives.
            session.rollback()
            raise WorkflowLeaseUnavailable(
                f"workflow {workflow_id} is being leased by another worker: {exc}",
                details={"reason": LEASE_REASON_LOCKED},
            ) from exc
        finally:
            session.close()

    def renew(self, lease: WorkflowLease) -> WorkflowLease:
        """Extend a lease this worker still holds, or raise `WorkflowLeaseLost`."""
        now = self.clock()
        expires_at = now + self.ttl
        session = self.session_factory()
        try:
            updated = (
                session.query(WorkflowLeaseModel)
                .filter(
                    WorkflowLeaseModel.workflow_id == lease.workflow_id,
                    WorkflowLeaseModel.lease_token == lease.token,
                    WorkflowLeaseModel.released_at.is_(None),
                )
                .update({"expires_at": expires_at, "updated_at": now}, synchronize_session=False)
            )
            session.commit()
            if not updated:
                raise WorkflowLeaseLost(
                    f"lease {lease.token} on workflow {lease.workflow_id} is no longer "
                    f"held by '{lease.owner}'"
                )
            return lease.model_copy(update={"expires_at": expires_at})
        finally:
            session.close()

    def release(self, lease: WorkflowLease) -> bool:
        """Release a lease, returning whether this worker still held it.

        A stale release is a no-op returning `False`, not an error: a worker
        whose lease expired and was taken over is entitled to try to clean up,
        and its attempt must not cancel its successor's hold.
        """
        now = self.clock()
        session = self.session_factory()
        try:
            updated = (
                session.query(WorkflowLeaseModel)
                .filter(
                    WorkflowLeaseModel.workflow_id == lease.workflow_id,
                    WorkflowLeaseModel.lease_token == lease.token,
                    WorkflowLeaseModel.released_at.is_(None),
                )
                .update({"released_at": now, "updated_at": now}, synchronize_session=False)
            )
            session.commit()
            if not updated:
                logger.warning(
                    "release of lease %s on workflow %s ignored: no longer held by '%s'",
                    lease.token,
                    lease.workflow_id,
                    lease.owner,
                )
            return bool(updated)
        finally:
            session.close()

    def current(self, workflow_id: UUID) -> WorkflowLease | None:
        """The lease currently held on a workflow, or `None` if it is free."""
        now = self.clock()
        session = self.session_factory()
        try:
            row = (
                session.query(WorkflowLeaseModel)
                .filter(WorkflowLeaseModel.workflow_id == workflow_id)
                .first()
            )
            if row is None or not self._is_held(row, now):
                return None
            return WorkflowLease(
                workflow_id=row.workflow_id,
                owner=row.owner,
                token=row.lease_token,
                acquired_at=row.acquired_at,
                expires_at=row.expires_at,
                thread_id=row.thread_id,
            )
        finally:
            session.close()

    @contextmanager
    def hold(
        self,
        workflow_id: UUID,
        *,
        owner: str | None = None,
        thread_id: str | None = None,
    ) -> Iterator[WorkflowLease]:
        """Hold the lease for the duration of a block, releasing on the way out.

        Releases even when the block raised: a failed resume that kept its lease
        until expiry would block the retry that is meant to follow it.
        """
        lease = self.acquire(workflow_id, owner=owner, thread_id=thread_id)
        try:
            yield lease
        finally:
            self.release(lease)

    def _locked_row(self, session: Session, workflow_id: UUID) -> WorkflowLeaseModel | None:
        """Read the lease row, locking it where the dialect supports locking."""
        query = session.query(WorkflowLeaseModel).filter(
            WorkflowLeaseModel.workflow_id == workflow_id
        )
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            query = query.with_for_update()
        return query.first()

    @staticmethod
    def _is_held(row: WorkflowLeaseModel, now: datetime) -> bool:
        return row.released_at is None and row.expires_at > now


__all__ = [
    "DEFAULT_LEASE_TTL_SECONDS",
    "LEASE_REASON_HELD",
    "LEASE_REASON_INSERT_RACE",
    "LEASE_REASON_TAKEOVER_RACE",
    "LEASE_REASON_LOCKED",
    "LEASE_EXCLUSION_REASONS",
    "WorkflowLeaseUnavailable",
    "WorkflowLeaseLost",
    "WorkflowLeaseStore",
    "default_owner",
]
