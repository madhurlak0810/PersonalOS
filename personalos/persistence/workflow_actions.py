"""Reads every action a workflow touched, and labels each with what became of it.

The status itself is decided in `personalos.domain.action_status`; this module
only gathers the facts it is decided from, all of them stored rows:

- each thread's newest checkpoint, for the intents, approval requests,
  decisions, refusals and receipts the run holds in state, and its shortlist;
- `approvals`, for a decision recorded through the API that the resumed run
  has not written to state yet;
- `tool_executions` and `audit_events`, for the execution row and the audit
  entries an action's receipt links to -- matched on the idempotency key the
  executor records every call under.

Like `WorkflowStatusReader`, it needs no compiled graph, so any API process --
or the CLI -- can answer from it.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.action_status import (
    ActionLinks,
    RecordedDecision,
    WorkflowAction,
    WorkflowActions,
    derive_shortlist,
    derive_thread_actions,
)
from personalos.domain.workflow import ApprovalOutcome, ApprovalState
from personalos.persistence.approvals import AUDIT_DECISION_PREFIX
from personalos.persistence.checkpointer import SqlAlchemyCheckpointSaver
from personalos.persistence.models import (
    ApprovalModel,
    AuditEventModel,
    ToolExecutionModel,
    WorkflowModel,
    WorkflowRunModel,
)

#: State keys an approval request is held under; the ids in it are what the
#: `approvals` table is keyed by.
_REQUESTS_KEY = "approval_requests"
_INTENTS_KEY = "pending_actions"


class WorkflowActionReader:
    """Builds a workflow's `WorkflowActions` from stored rows."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        checkpointer: SqlAlchemyCheckpointSaver,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Initialize with the database and the saver its checkpoints are read through."""
        self.session_factory = session_factory
        self.checkpointer = checkpointer
        self.clock = clock

    def read(self, workflow_id: UUID) -> WorkflowActions | None:
        """The workflow's actions and shortlist, or `None` if there is no such workflow."""
        session = self.session_factory()
        try:
            if session.get(WorkflowModel, workflow_id) is None:
                return None
            thread_ids = [
                thread_id
                for (thread_id,) in session.query(WorkflowRunModel.thread_id)
                .filter(WorkflowRunModel.workflow_id == workflow_id)
                .order_by(WorkflowRunModel.created_at.asc(), WorkflowRunModel.thread_id.asc())
                .all()
            ]
            states = {thread_id: self._values(thread_id) for thread_id in thread_ids}
            recorded = _recorded_decisions(session, states.values())
            links = _execution_links(session, states.values())
        finally:
            session.close()

        now = self.clock()
        actions: list[WorkflowAction] = []
        shortlist = []
        for thread_id, values in states.items():
            thread_actions = derive_thread_actions(
                values,
                workflow_id=workflow_id,
                thread_id=thread_id,
                recorded=recorded,
                links=links,
                now=now,
            )
            actions.extend(thread_actions)
            shortlist.extend(derive_shortlist(values, thread_id=thread_id, actions=thread_actions))
        return WorkflowActions(
            workflow_id=workflow_id, actions=tuple(actions), shortlist=tuple(shortlist)
        )

    def _values(self, thread_id: str) -> dict[str, Any]:
        stored = self.checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if stored is None:
            return {}
        return dict(stored.checkpoint.get("channel_values") or {})


def _ids_in(states: Any, key: str, field: str) -> set[str]:
    found: set[str] = set()
    for values in states:
        for raw in values.get(key) or ():
            if isinstance(raw, dict) and raw.get(field):
                found.add(str(raw[field]))
    return found


def _recorded_decisions(session: Session, states: Any) -> dict[UUID, RecordedDecision]:
    """Decisions on the `approvals` table for the requests these threads raised."""
    request_ids = []
    for raw in _ids_in(states, _REQUESTS_KEY, "request_id"):
        try:
            request_ids.append(UUID(raw))
        except ValueError:
            continue
    if not request_ids:
        return {}
    rows = session.query(ApprovalModel).filter(ApprovalModel.id.in_(request_ids)).all()
    deciders = _deciders(session, [row.id for row in rows])
    recorded: dict[UUID, RecordedDecision] = {}
    for row in rows:
        try:
            state = ApprovalState(row.status)
        except ValueError:
            continue
        if state in (ApprovalState.APPROVED, ApprovalState.DENIED):
            recorded[row.id] = RecordedDecision(state=state, decided_by=deciders.get(row.id))
    return recorded


def _deciders(session: Session, approval_ids: list[UUID]) -> dict[UUID, str]:
    """Who decided each approval, from the audit event its decision wrote."""
    if not approval_ids:
        return {}
    refs = {str(approval_id): approval_id for approval_id in approval_ids}
    rows = (
        session.query(AuditEventModel.approval_ref, AuditEventModel.actor)
        .filter(
            AuditEventModel.approval_ref.in_(list(refs)),
            AuditEventModel.action.in_(
                [f"{AUDIT_DECISION_PREFIX}{outcome.value}" for outcome in ApprovalOutcome]
            ),
        )
        .order_by(AuditEventModel.timestamp.asc())
        .all()
    )
    return {refs[ref]: actor for ref, actor in rows if ref in refs}


def _execution_links(session: Session, states: Any) -> dict[str, ActionLinks]:
    """The execution row and its audit events, by the idempotency key of each action."""
    keys = _ids_in(states, _INTENTS_KEY, "idempotency_key") | _ids_in(
        states, _REQUESTS_KEY, "idempotency_key"
    )
    if not keys:
        return {}
    executions = (
        session.query(ToolExecutionModel)
        .filter(ToolExecutionModel.idempotency_key.in_(sorted(keys)))
        .all()
    )
    if not executions:
        return {}
    events: dict[UUID, list[UUID]] = {}
    for event_id, operation_id in (
        session.query(AuditEventModel.id, AuditEventModel.operation_id)
        .filter(AuditEventModel.operation_id.in_([row.operation_id for row in executions]))
        .order_by(AuditEventModel.timestamp.asc(), AuditEventModel.id.asc())
        .all()
    ):
        events.setdefault(operation_id, []).append(event_id)
    return {
        row.idempotency_key: ActionLinks(
            operation_id=row.operation_id,
            tool_execution_status=row.status,
            audit_event_ids=tuple(events.get(row.operation_id, ())),
        )
        for row in executions
    }


__all__ = ["WorkflowActionReader"]
