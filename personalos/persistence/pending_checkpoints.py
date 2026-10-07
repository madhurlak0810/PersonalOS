"""Storage for durable, conditional waits, and the guarded close that makes them safe.

`personalos.domain.checkpoints` says what a pending checkpoint *is* and how to
decide about one; this is where it lives between the run that scheduled it and
the sweep that acts on it -- days or weeks later, in a process that did not
exist when it was created.

Two properties are the reason this is a module and not three lines of ORM:

**Scheduling is idempotent, by `dedupe_key`.** The branch that schedules a
follow-up runs again whenever the graph re-enters it (a second recruiter
message, a resumed run replaying its last super-step), and each pass proposes
the same reminder. `schedule` returns the existing wait rather than stacking a
second one, so re-entry is free. That is the same trade
`OutboxEventRepository.create` makes on its own `dedupe_key`, and the unique
constraint -- not the read-then-write -- is what actually enforces it.

**Closing is a guarded, single-statement `UPDATE`.** `close` is
`UPDATE ... WHERE id = ? AND status = 'pending'`, and its affected-row count is
returned. Two monitors sweeping the same due checkpoint both issue it; exactly
one updates a row, and only that one goes on to start the graph path. This is
the same primitive `personalos.persistence.leases` rests on and it holds for
the same reason: it does not depend on `SELECT ... FOR UPDATE`, which SQLite
silently ignores.

Sessions come from an injected factory and each call is its own short
transaction, matching `SqlOperationStore` and `WorkflowLeaseStore`: a checkpoint
closed inside some caller's long-running transaction would be invisible to the
second monitor that is about to decide about it.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.checkpoints import (
    CheckpointCondition,
    ConditionKind,
    PendingCheckpoint,
    PendingCheckpointStatus,
)
from personalos.domain.job_search import FollowUpKind
from personalos.persistence.models import PendingCheckpointModel

logger = logging.getLogger(__name__)

#: How many checkpoints one sweep takes at a time. A bound rather than "all of
#: them" so a backlog is worked through in paced batches instead of one
#: transaction that starts every stalled workflow at once.
DEFAULT_SWEEP_LIMIT = 100


class PendingCheckpointStore:
    """Schedules, finds and closes the pending checkpoints in `pending_checkpoints`.

    Takes a session factory rather than a session, for the same reason
    `WorkflowLeaseStore` does: it is called from a sweep loop that may run for
    a long time, and a borrowed session would tie every close to whatever
    transaction the caller had open -- which is precisely what would make the
    guarded `UPDATE` stop excluding anybody.

    `clock` is injected so expiry and trigger behaviour can be tested by moving
    time rather than by waiting for it.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory
        self.clock = clock

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def schedule(self, checkpoint: PendingCheckpoint) -> PendingCheckpoint:
        """Store a wait, or return the one already stored under its `dedupe_key`.

        Idempotent in the way a replayed super-step needs: the graph branch that
        schedules follow-ups runs again on every resume through it, and a second
        row would mean the candidate gets nudged twice about the same silence.
        The existing row wins whole -- including its original `trigger_at` --
        because the first schedule is the one that dated the wait correctly,
        and re-dating it on every replay would push the trigger forever into the
        future.
        """
        session = self.session_factory()
        try:
            existing = self._row_for_key(session, checkpoint.dedupe_key)
            if existing is not None:
                return _to_domain(existing)

            session.add(_to_row(checkpoint))
            try:
                session.commit()
            except IntegrityError:
                # Lost the insert race on `dedupe_key`: another process
                # scheduled the same wait first, and its row is the one both
                # must use.
                session.rollback()
                winner = self._row_for_key(session, checkpoint.dedupe_key)
                if winner is None:  # pragma: no cover - a unique violation implies a row
                    raise
                return _to_domain(winner)
            return checkpoint
        finally:
            session.close()

    def close(
        self,
        checkpoint: PendingCheckpoint | UUID,
        status: PendingCheckpointStatus,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        """Close a checkpoint, returning whether this caller was the one that did.

        The exclusion primitive. The `WHERE status = 'pending'` clause is not
        hygiene: it is what makes two monitors that both selected the same due
        checkpoint disagree about who owns it, with the database as the
        arbiter. A caller that gets `False` back has lost that race and must not
        act -- the winner is already starting the graph path.

        Returns a bool rather than raising, unlike `WorkflowLeaseStore.acquire`,
        because losing here is the ordinary outcome of two monitors overlapping
        and the right response is to move on to the next checkpoint.
        """
        if status == PendingCheckpointStatus.PENDING:
            raise ValueError("close requires a terminal status; 'pending' is not one")
        checkpoint_id = (
            checkpoint if isinstance(checkpoint, UUID) else checkpoint.checkpoint_id
        )
        now = now or self.clock()
        session = self.session_factory()
        try:
            updated = (
                session.query(PendingCheckpointModel)
                .filter(
                    PendingCheckpointModel.id == checkpoint_id,
                    PendingCheckpointModel.status == PendingCheckpointStatus.PENDING.value,
                )
                .update(
                    {
                        "status": status.value,
                        "closed_at": now,
                        "closed_reason": reason,
                        "updated_at": now,
                    },
                    synchronize_session=False,
                )
            )
            session.commit()
            if not updated:
                logger.info(
                    "checkpoint %s was already closed by someone else; not %s",
                    checkpoint_id,
                    status.value,
                )
            return bool(updated)
        finally:
            session.close()

    def supersede(
        self,
        checkpoint: PendingCheckpoint,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> PendingCheckpoint:
        """Store a wait and cancel every other open wait of its kind on its application.

        For a wait whose date follows something that can move -- the reminder
        for an interview. `schedule` deliberately keeps the first row's
        `trigger_at`, which is right for a replayed step and wrong for a moved
        interview, so a re-dated wait arrives under a *new* `dedupe_key` and
        this closes the ones it replaces. They are closed `cancelled`, with the
        reason, rather than deleted: "why did the Tuesday reminder never
        fire?" has an answer.

        The new wait is stored first, so a crash between the two steps leaves
        one reminder too many rather than none.
        """
        now = now or self.clock()
        stored = self.schedule(checkpoint)
        session = self.session_factory()
        try:
            stale = [
                row.id
                for row in session.query(PendingCheckpointModel)
                .filter(
                    PendingCheckpointModel.application_id == checkpoint.application_id,
                    PendingCheckpointModel.kind == checkpoint.kind.value,
                    PendingCheckpointModel.status == PendingCheckpointStatus.PENDING.value,
                    PendingCheckpointModel.dedupe_key != stored.dedupe_key,
                )
                .all()
            ]
        finally:
            session.close()
        for checkpoint_id in stale:
            self.close(checkpoint_id, PendingCheckpointStatus.CANCELLED, reason=reason, now=now)
        return stored

    def resolve_matching(
        self,
        *,
        condition_kind: ConditionKind,
        subject_id: UUID,
        reason: str,
        occurred_at: datetime | None = None,
        now: datetime | None = None,
    ) -> list[UUID]:
        """Close every open checkpoint this event has just made unnecessary.

        The event-driven half of resolution, and the reason a checkpoint can be
        closed *before* its trigger without anyone sweeping it: when a recruiter
        reply lands, every wait that existed only because no reply had landed
        is answered, and closing them then keeps the table honest about what is
        still outstanding.

        It is an optimization, not the mechanism. Nothing depends on this being
        called: a checkpoint nobody resolved early is re-evaluated at trigger
        time and closed silently then, which is what makes the guarantee hold
        even when the event that resolved it happened somewhere this process
        never saw. `occurred_at` respects each checkpoint's `condition.since`,
        so an event older than the wait does not resolve it.
        """
        now = now or self.clock()
        session = self.session_factory()
        try:
            rows = (
                session.query(PendingCheckpointModel)
                .filter(
                    PendingCheckpointModel.status == PendingCheckpointStatus.PENDING.value,
                    PendingCheckpointModel.condition_kind == condition_kind.value,
                    PendingCheckpointModel.condition_subject_id == subject_id,
                )
                .all()
            )
            candidates = [
                row.id
                for row in rows
                if occurred_at is None
                or row.condition_since is None
                or occurred_at >= row.condition_since
            ]
        finally:
            session.close()

        # Closed one at a time through the guarded `close`, not as one bulk
        # UPDATE: each close has to race a monitor that may be firing that same
        # checkpoint right now, and only the guarded statement settles that.
        return [
            checkpoint_id
            for checkpoint_id in candidates
            if self.close(
                checkpoint_id, PendingCheckpointStatus.RESOLVED, reason=reason, now=now
            )
        ]

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, checkpoint_id: UUID) -> PendingCheckpoint | None:
        """One checkpoint by id, or `None`."""
        session = self.session_factory()
        try:
            row = (
                session.query(PendingCheckpointModel)
                .filter(PendingCheckpointModel.id == checkpoint_id)
                .first()
            )
            return _to_domain(row) if row is not None else None
        finally:
            session.close()

    def due(
        self, *, now: datetime | None = None, limit: int = DEFAULT_SWEEP_LIMIT
    ) -> list[PendingCheckpoint]:
        """Every open checkpoint that has something to decide, soonest first.

        "Something to decide" is `trigger_at <= now` **or** `expires_at <= now`.
        The second disjunct looks redundant -- an expiry is always after a
        trigger -- and is kept because the query is the only thing standing
        between a checkpoint and sitting `pending` forever, and it should not
        depend on that invariant holding in rows written by an older version of
        the code.

        Deliberately does not evaluate any condition: that is the caller's job,
        and doing it here would mean evaluating conditions for checkpoints that
        are not yet actionable, which is exactly the creation-time evaluation
        this whole design exists to avoid.
        """
        now = now or self.clock()
        session = self.session_factory()
        try:
            rows = (
                session.query(PendingCheckpointModel)
                .filter(
                    PendingCheckpointModel.status == PendingCheckpointStatus.PENDING.value,
                    (PendingCheckpointModel.trigger_at <= now)
                    | (PendingCheckpointModel.expires_at <= now),
                )
                # id breaks ties so two checkpoints dated the same millisecond
                # come back in a stable order rather than the query plan's.
                .order_by(
                    PendingCheckpointModel.trigger_at.asc(),
                    PendingCheckpointModel.id.asc(),
                )
                .limit(limit)
                .all()
            )
            return [_to_domain(row) for row in rows]
        finally:
            session.close()

    def open_for_application(self, application_id: UUID) -> list[PendingCheckpoint]:
        """Every wait still outstanding on one application, soonest first."""
        session = self.session_factory()
        try:
            rows = (
                session.query(PendingCheckpointModel)
                .filter(
                    PendingCheckpointModel.application_id == application_id,
                    PendingCheckpointModel.status == PendingCheckpointStatus.PENDING.value,
                )
                .order_by(PendingCheckpointModel.trigger_at.asc())
                .all()
            )
            return [_to_domain(row) for row in rows]
        finally:
            session.close()

    def history_for_application(self, application_id: UUID) -> list[PendingCheckpoint]:
        """Every wait ever attached to one application, oldest first.

        Closed rows are kept rather than deleted: "why was no follow-up sent?"
        is answerable only if the checkpoint that decided not to send one is
        still there, with its status and its `closed_reason`.
        """
        session = self.session_factory()
        try:
            rows = (
                session.query(PendingCheckpointModel)
                .filter(PendingCheckpointModel.application_id == application_id)
                .order_by(PendingCheckpointModel.created_at.asc())
                .all()
            )
            return [_to_domain(row) for row in rows]
        finally:
            session.close()

    @staticmethod
    def _row_for_key(session: Session, dedupe_key: str) -> PendingCheckpointModel | None:
        return (
            session.query(PendingCheckpointModel)
            .filter(PendingCheckpointModel.dedupe_key == dedupe_key)
            .first()
        )


class StorePendingCheckpointScheduler:
    """Binds the graph's `PendingCheckpointScheduler` port to the store.

    An async shim over a synchronous store, and nothing more: the port is async
    because every port a node calls is, and the store is synchronous because
    SQLAlchemy's session is. Kept here rather than in `bootstrap` for the same
    reason `JournaledActionExecutor` is -- it is an adapter, and the composition
    root's job is to choose it, not to contain it.
    """

    def __init__(self, store: PendingCheckpointStore):
        """Wrap the store the graph's scheduled waits are written to."""
        self.store = store

    async def schedule(self, checkpoint: PendingCheckpoint) -> PendingCheckpoint:
        """Persist the wait, returning whichever row now owns it."""
        return self.store.schedule(checkpoint)

    async def reschedule(self, checkpoint: PendingCheckpoint, *, reason: str) -> PendingCheckpoint:
        """Persist the wait and cancel the open ones of its kind it replaces."""
        return self.store.supersede(checkpoint, reason=reason)


def _to_row(checkpoint: PendingCheckpoint) -> PendingCheckpointModel:
    """Flatten a checkpoint into its row, condition included."""
    return PendingCheckpointModel(
        id=checkpoint.checkpoint_id,
        application_id=checkpoint.application_id,
        kind=checkpoint.kind.value,
        reason=checkpoint.reason,
        thread_id=checkpoint.thread_id,
        workflow_id=checkpoint.workflow_id,
        condition_kind=checkpoint.condition.kind.value,
        condition_subject_id=checkpoint.condition.subject_id,
        condition_since=checkpoint.condition.since,
        condition_params=dict(checkpoint.condition.params),
        trigger_at=checkpoint.trigger_at,
        expires_at=checkpoint.expires_at,
        status=checkpoint.status.value,
        dedupe_key=checkpoint.dedupe_key,
        closed_at=checkpoint.closed_at,
        closed_reason=checkpoint.closed_reason,
        created_at=checkpoint.created_at,
        updated_at=checkpoint.created_at,
    )


def _to_domain(row: PendingCheckpointModel) -> PendingCheckpoint:
    """Rebuild the typed checkpoint from its row."""
    return PendingCheckpoint(
        checkpoint_id=row.id,
        application_id=row.application_id,
        kind=FollowUpKind(row.kind),
        condition=CheckpointCondition(
            kind=ConditionKind(row.condition_kind),
            subject_id=row.condition_subject_id,
            since=row.condition_since,
            params=dict(row.condition_params or {}),
        ),
        reason=row.reason,
        thread_id=row.thread_id,
        workflow_id=row.workflow_id,
        created_at=row.created_at,
        trigger_at=row.trigger_at,
        expires_at=row.expires_at,
        status=PendingCheckpointStatus(row.status),
        dedupe_key=row.dedupe_key,
        closed_at=row.closed_at,
        closed_reason=row.closed_reason,
    )


__all__ = [
    "DEFAULT_SWEEP_LIMIT",
    "PendingCheckpointStore",
    "StorePendingCheckpointScheduler",
]
