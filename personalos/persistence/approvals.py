"""The approval queue: what a human is being asked, and what they answered.

A workflow parked at `approval_checkpoint` holds its `ApprovalRequest`s in the
interrupt it raised, which is part of the checkpoint. That stored request is
the source of truth for everything here -- the list is read out of it, and a
decision is checked against the `action_hash` it carries -- so nothing a
reviewer is shown, and nothing a decision is compared with, is re-derived from
state that may have changed since the run parked.

An approval is addressed by its request's id, and that id is the primary key
of the `approvals` row a decision writes. Two properties follow:

- **A decision is recorded exactly once.** The second insert for the same
  request collides on the primary key, so two reviewers racing each other
  produce one row and one outcome; the loser is handed the winner's.
- **The workflow is resumed exactly once.** The resume is queued under the
  checkpoint it answers (`WorkflowCommandQueue.enqueue(dedupe=...)`), so a
  repeat submission that reaches the queue again is refused there.

One interrupt can carry several requests. The run can only be resumed with an
answer to all of them, so each decision is recorded as it arrives and the
resume is queued by whichever one completes the set.

**Nobody answering is an answer too, eventually.** A request past its expiry
can no longer be approved, so a workflow parked on it would wait forever.
`expire_overdue` -- run on a schedule by the worker, never by a request --
records each such request as `EXPIRED` and resumes the run exactly as a denial
would, with `APPROVAL_EXPIRY_ACTOR` as the decider so the refusal the graph
records says "expired" and not "denied". It is the same guarded insert a
reviewer's decision is, so an expiry racing a last-minute answer leaves one of
them on record, not both.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.errors import Conflict, NotFound
from personalos.domain.job_search import (
    APPROVAL_EXPIRY_ACTOR,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
)
from personalos.domain.models import AuditEventResult
from personalos.domain.workflow import (
    ApprovalOutcome,
    ApprovalState,
    PendingApproval,
    PendingInterrupt,
    RecordedApprovalDecision,
    ThreadSnapshot,
    WorkflowCommand,
    WorkflowCommandKind,
)
from personalos.observability import tracing
from personalos.observability.tracing import Attr
from personalos.persistence.checkpointer import WorkflowThreadRegistry
from personalos.persistence.models import ApprovalModel, AuditEventModel, WorkflowRunModel
from personalos.persistence.repositories import AuditEventRepository
from personalos.persistence.workflow_commands import DuplicateWorkflowCommand, WorkflowCommandQueue
from personalos.persistence.workflow_status import WorkflowStatusReader

logger = logging.getLogger(__name__)

#: `audit_events.action` prefix for a recorded decision; the outcome follows it.
AUDIT_DECISION_PREFIX = "approval."
#: `audit_events.action` for a decision that was refused rather than recorded.
AUDIT_REFUSED_HASH_MISMATCH = "approval.refused.hash_mismatch"
AUDIT_REFUSED_EXPIRED = "approval.refused.expired"

_STATUS_BY_OUTCOME = {
    ApprovalOutcome.APPROVED: ApprovalState.APPROVED,
    ApprovalOutcome.DENIED_BY_USER: ApprovalState.DENIED,
    # Stored as a denial -- the run takes its declined branch either way --
    # and told apart by the outcome written to `approvals.note`.
    ApprovalOutcome.EXPIRED: ApprovalState.DENIED,
}
_VERDICT_BY_STATE = {
    ApprovalState.APPROVED.value: ApprovalVerdict.APPROVED,
    ApprovalState.DENIED.value: ApprovalVerdict.REJECTED,
}


class ApprovalHashMismatch(Conflict):
    """The decision names a hash the approval was not raised with."""

    default_message = "action_hash does not match the action this approval was requested for"


class ApprovalExpired(Conflict):
    """The approval's window closed before it was approved."""

    default_message = "this approval has expired and can no longer be approved"


class ApprovalAlreadyDecided(Conflict):
    """The approval was already decided the other way."""

    default_message = "this approval has already been decided"


@dataclass(frozen=True)
class _OpenRequest:
    """One request inside a pending interrupt, with where it is parked."""

    workflow_id: UUID
    thread: ThreadSnapshot
    interrupt: PendingInterrupt
    request: ApprovalRequest
    #: Every request the same interrupt raised, this one included.
    batch: tuple[ApprovalRequest, ...]


class ApprovalQueue:
    """Lists pending approvals and records the decisions made on them."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        reader: WorkflowStatusReader,
        registry: WorkflowThreadRegistry,
        commands: WorkflowCommandQueue,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Initialize with the database, reader, registry and queue the workflows live in."""
        self.session_factory = session_factory
        self.reader = reader
        self.registry = registry
        self.commands = commands
        self.clock = clock

    def list_pending(self) -> list[PendingApproval]:
        """Every request still owed an answer, oldest first, expired ones marked."""
        now = self.clock()
        open_requests = list(self._open_requests())
        decided = self._decided_rows([item.request.request_id for item in open_requests])
        pending = [
            _pending(item, now) for item in open_requests if item.request.request_id not in decided
        ]
        return sorted(pending, key=lambda approval: approval.requested_at)

    def decide(
        self,
        approval_id: UUID,
        *,
        approved: bool,
        action_hash: str,
        actor_id: str,
        correlation_id: UUID | None = None,
    ) -> RecordedApprovalDecision:
        """Record a reviewer's decision and, once its interrupt is answered, resume the run.

        Raises `ApprovalHashMismatch` when `action_hash` is not the hash the
        request was raised with, `ApprovalExpired` when approving past the
        request's expiry, `ApprovalAlreadyDecided` when it was decided the other
        way, and `NotFound` when no such approval is pending. A repeat of a
        decision already recorded returns that decision and resumes nothing.
        """
        now = self.clock()
        outcome = ApprovalOutcome.APPROVED if approved else ApprovalOutcome.DENIED_BY_USER
        correlation_id = correlation_id or uuid4()
        found = self._find(approval_id)

        with self._decision_span(found, approval_id, actor_id, correlation_id) as span:
            recorded = self._decided_rows([approval_id]).get(approval_id)
            if recorded is None:
                if found is None:
                    raise NotFound(
                        f"no pending approval '{approval_id}'",
                        details={"approval_id": str(approval_id)},
                    )
                self._check(
                    found, approved=approved, action_hash=action_hash, actor_id=actor_id, now=now
                )
                replayed = not self._record(found, outcome=outcome, actor_id=actor_id, now=now)
                recorded = self._decided_rows([approval_id])[approval_id]
            else:
                replayed = True

            if replayed:
                # The original outcome stands. A repeat is only "the same decision"
                # if it names the same action and the same answer.
                if action_hash != recorded.action_hash:
                    raise ApprovalHashMismatch(details={"approval_id": str(approval_id)})
                if recorded.status != _STATUS_BY_OUTCOME[outcome].value:
                    raise ApprovalAlreadyDecided(
                        f"approval '{approval_id}' was already {recorded.status}; that decision stands",
                        details={"approval_id": str(approval_id)},
                    )

            # Also reached on a repeat, so a decision recorded by a request that
            # died before queuing the resume is completed by the retry. The queue's
            # dedupe key is what keeps that from ever resuming twice.
            resumed = found is None or self._resume_if_answered(
                found, actor_id=actor_id, correlation_id=correlation_id
            )
            decision = RecordedApprovalDecision(
                approval_id=approval_id,
                workflow_id=found.workflow_id if found else self._workflow_id(recorded),
                action_hash=recorded.action_hash,
                state=ApprovalState(recorded.status),
                outcome=_outcome(recorded),
                decided_by=self._decided_by([approval_id]).get(approval_id),
                decided_at=recorded.decided_at,
                resumed=resumed,
                replayed=replayed,
            )
            tracing.set_attributes(
                span,
                {
                    Attr.APPROVAL_OUTCOME: decision.outcome,
                    Attr.OUTCOME: "resumed" if decision.resumed else "recorded",
                    "replayed": decision.replayed,
                },
            )
            return decision

    @staticmethod
    def _decision_span(
        found: _OpenRequest | None,
        approval_id: UUID,
        actor_id: str,
        correlation_id: UUID,
    ):
        """The span a decision is recorded in: the workflow's trace, when it is known."""
        attributes = {Attr.APPROVAL_ID: approval_id}
        if found is None:
            return tracing.span("approval.decide", attributes)
        return tracing.workflow_span(
            "approval.decide",
            workflow_id=found.workflow_id,
            correlation_id=correlation_id,
            thread_id=found.thread.thread_id,
            actor_id=actor_id,
            attributes={**attributes, Attr.ACTION_ID: found.request.action_id},
        )

    def expire_overdue(
        self, *, now: datetime | None = None, actor_id: str = APPROVAL_EXPIRY_ACTOR
    ) -> list[UUID]:
        """Close every unanswered request past its expiry, resuming runs that are now answered.

        Returns the ids of the approvals this call expired. One a reviewer
        answered first is skipped, not overwritten; a run whose interrupt
        still has live requests is left parked until they are answered too.
        """
        now = now or self.clock()
        open_requests = list(self._open_requests())
        decided = self._decided_rows([item.request.request_id for item in open_requests])
        expired: list[UUID] = []
        for item in open_requests:
            if item.request.request_id in decided or not item.request.is_expired(now):
                continue
            if not self._record(item, outcome=ApprovalOutcome.EXPIRED, actor_id=actor_id, now=now):
                continue
            expired.append(item.request.request_id)
            self._resume_if_answered(item, actor_id=actor_id, correlation_id=uuid4())
        if expired:
            logger.info("expired %s unanswered approval(s) at %s", len(expired), now.isoformat())
        return expired

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_requests(self) -> Iterator[_OpenRequest]:
        """Every request in an interrupt nobody has queued an answer to."""
        for workflow_id in self.reader.interrupted_workflow_ids():
            snapshot = self.reader.read(workflow_id)
            if snapshot is None:
                continue
            threads = {thread.thread_id: thread for thread in snapshot.threads}
            for interrupt in snapshot.pending_interrupts:
                batch = _requests_in(interrupt)
                for request in batch:
                    yield _OpenRequest(
                        workflow_id=workflow_id,
                        thread=threads[interrupt.thread_id],
                        interrupt=interrupt,
                        request=request,
                        batch=batch,
                    )

    def _find(self, approval_id: UUID) -> _OpenRequest | None:
        return next(
            (item for item in self._open_requests() if item.request.request_id == approval_id),
            None,
        )

    def _check(
        self, found: _OpenRequest, *, approved: bool, action_hash: str, actor_id: str, now: datetime
    ) -> None:
        """Refuse a decision that does not fit the stored request, and audit the refusal."""
        request = found.request
        if action_hash != request.action_hash:
            self._audit_refusal(found, actor_id, AUDIT_REFUSED_HASH_MISMATCH)
            raise ApprovalHashMismatch(
                f"action_hash does not match the action approval '{request.request_id}' was "
                f"requested for; the action may have changed since it was listed",
                details={"approval_id": str(request.request_id)},
            )
        if approved and request.is_expired(now):
            self._audit_refusal(found, actor_id, AUDIT_REFUSED_EXPIRED)
            raise ApprovalExpired(
                f"approval '{request.request_id}' expired at {request.expires_at.isoformat()} "
                f"and can no longer be approved",
                details={"approval_id": str(request.request_id)},
            )

    def _record(
        self, found: _OpenRequest, *, outcome: ApprovalOutcome, actor_id: str, now: datetime
    ) -> bool:
        """Write the `approvals` and `audit_events` rows together. False if already decided."""
        request = found.request
        thread = self.registry.resolve(found.thread.thread_id)
        session = self.session_factory()
        try:
            session.add(
                ApprovalModel(
                    id=request.request_id,
                    workflow_run_id=thread.workflow_run_id if thread else None,
                    action_hash=request.action_hash,
                    status=_STATUS_BY_OUTCOME[outcome].value,
                    note=outcome.value,
                    decided_at=now,
                    created_at=request.requested_at,
                    updated_at=now,
                )
            )
            try:
                AuditEventRepository(session).create(
                    actor=actor_id,
                    action=f"{AUDIT_DECISION_PREFIX}{outcome.value}",
                    target_ref=request.target,
                    result=AuditEventResult.SUCCESS.value,
                    workflow_id=found.workflow_id,
                    approval_ref=str(request.request_id),
                    commit=False,
                )
                session.commit()
            except IntegrityError:
                # Lost the race on `approvals.id`: someone else's decision is
                # the one on record, and neither of this one's rows was kept.
                session.rollback()
                return False
        finally:
            session.close()
        logger.info(
            "approval %s for action %s on workflow %s recorded as %s by %s",
            request.request_id,
            request.action_id,
            found.workflow_id,
            outcome.value,
            actor_id,
        )
        return True

    def _audit_refusal(self, found: _OpenRequest, actor_id: str, action: str) -> None:
        session = self.session_factory()
        try:
            AuditEventRepository(session).create(
                actor=actor_id,
                action=action,
                target_ref=found.request.target,
                result=AuditEventResult.FAILURE.value,
                workflow_id=found.workflow_id,
                approval_ref=str(found.request.request_id),
            )
        finally:
            session.close()

    def _resume_if_answered(
        self, found: _OpenRequest, *, actor_id: str, correlation_id: UUID
    ) -> bool:
        """Queue the resume once every request of the interrupt has a decision."""
        rows = self._decided_rows([request.request_id for request in found.batch])
        if len(rows) < len(found.batch):
            return False
        deciders = self._decided_by(list(rows))
        decisions = [
            ApprovalDecision(
                action_id=request.action_id,
                action_fingerprint=rows[request.request_id].action_hash,
                verdict=_VERDICT_BY_STATE[rows[request.request_id].status],
                decided_by=deciders.get(request.request_id) or actor_id,
                decided_at=rows[request.request_id].decided_at,
                note=rows[request.request_id].note,
                request_id=request.request_id,
            ).model_dump(mode="json")
            for request in found.batch
        ]
        try:
            self.commands.enqueue(
                WorkflowCommand(
                    command_id=uuid4(),
                    kind=WorkflowCommandKind.RESUME,
                    workflow_id=found.workflow_id,
                    thread_id=found.thread.thread_id,
                    actor_id=actor_id,
                    correlation_id=correlation_id,
                    graph_input=decisions,
                    interrupt_ids=(
                        (found.interrupt.interrupt_id,) if found.interrupt.interrupt_id else ()
                    ),
                ),
                dedupe=found.thread.checkpoint_id,
            )
        except DuplicateWorkflowCommand:
            logger.info(
                "workflow %s is already resumed past checkpoint %s; not queuing again",
                found.workflow_id,
                found.thread.checkpoint_id,
            )
        return True

    def _decided_rows(self, approval_ids: list[UUID]) -> dict[UUID, ApprovalModel]:
        if not approval_ids:
            return {}
        session = self.session_factory()
        try:
            rows = (
                session.query(ApprovalModel)
                .filter(
                    ApprovalModel.id.in_(approval_ids),
                    ApprovalModel.status != ApprovalState.PENDING.value,
                )
                .all()
            )
            for row in rows:
                session.expunge(row)
            return {row.id: row for row in rows}
        finally:
            session.close()

    def _decided_by(self, approval_ids: list[UUID]) -> dict[UUID, str]:
        """Who decided each approval, from the audit trail -- the only place it is kept."""
        if not approval_ids:
            return {}
        actions = [f"{AUDIT_DECISION_PREFIX}{outcome.value}" for outcome in ApprovalOutcome]
        session = self.session_factory()
        try:
            rows = (
                session.query(AuditEventModel.approval_ref, AuditEventModel.actor)
                .filter(
                    AuditEventModel.approval_ref.in_([str(i) for i in approval_ids]),
                    AuditEventModel.action.in_(actions),
                )
                .all()
            )
            return {UUID(approval_ref): actor for approval_ref, actor in rows}
        finally:
            session.close()

    def _workflow_id(self, row: ApprovalModel) -> UUID | None:
        if row.workflow_run_id is None:
            return None
        session = self.session_factory()
        try:
            run = (
                session.query(WorkflowRunModel.workflow_id)
                .filter(WorkflowRunModel.id == row.workflow_run_id)
                .first()
            )
            return run[0] if run else None
        finally:
            session.close()


def _requests_in(interrupt: PendingInterrupt) -> tuple[ApprovalRequest, ...]:
    """The approval requests an interrupt carries; none if it is not an approval."""
    value = interrupt.value
    raw = value.get("requests") if isinstance(value, dict) else None
    if not isinstance(raw, list):
        return ()
    requests: list[ApprovalRequest] = []
    for item in raw:
        try:
            requests.append(ApprovalRequest.model_validate(item))
        except ValidationError:
            logger.warning(
                "interrupt %s on thread '%s' carries a request that is not an "
                "ApprovalRequest; it cannot be listed or decided through the approval queue",
                interrupt.interrupt_id,
                interrupt.thread_id,
            )
    return tuple(requests)


def _pending(item: _OpenRequest, now: datetime) -> PendingApproval:
    request = item.request
    return PendingApproval(
        approval_id=request.request_id,
        workflow_id=item.workflow_id,
        thread_id=item.thread.thread_id,
        action_id=request.action_id,
        action_hash=request.action_hash,
        action_type=request.kind.value,
        target=request.target,
        summary=request.summary,
        risk_level=request.risk.value,
        requested_scopes=request.requested_scopes,
        requested_at=request.requested_at,
        expires_at=request.expires_at,
        state=ApprovalState.EXPIRED if request.is_expired(now) else ApprovalState.PENDING,
        preview=request.preview,
    )


def _outcome(row: ApprovalModel) -> ApprovalOutcome:
    if row.status == ApprovalState.APPROVED.value:
        return ApprovalOutcome.APPROVED
    if row.note == ApprovalOutcome.EXPIRED.value:
        return ApprovalOutcome.EXPIRED
    return ApprovalOutcome.DENIED_BY_USER


__all__ = [
    "AUDIT_DECISION_PREFIX",
    "AUDIT_REFUSED_HASH_MISMATCH",
    "AUDIT_REFUSED_EXPIRED",
    "ApprovalHashMismatch",
    "ApprovalExpired",
    "ApprovalAlreadyDecided",
    "ApprovalQueue",
]
