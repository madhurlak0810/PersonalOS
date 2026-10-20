"""The audit trail's reads: audit events joined to what authorized them.

Read-only by construction. `AuditTrailStore` has no method that writes, and
each read is a fresh session that is closed without a commit -- the routes over
it (`GET /v1/audit`, `GET /v1/workflows/{id}/audit`) have no path to change or
remove an `audit_events` row, which is the property the trail's
`AuditEventRepository` already promises on the write side.

**Linking.** Each entry carries the `policy_decisions`, `tool_executions` and
`approvals` rows of the action it belongs to. Where a row cites another
directly it is followed: an audit event's `policy_decision_id`, `operation_id`
and `approval_ref`; an execution's `policy_decision_id` and `approval_ref`.
Where nothing cites it -- the approval step is recorded before the call that
redeems it, and the first policy decision before either -- the rows are matched
on the action's fingerprint, which every one of them stores
(`args_hash`, `action_hash`, `request_fingerprint`) and which is, by design,
the one value that identifies an action across all four tables.

**Redaction.** Every value is passed through the redactor before it leaves
here. Rows are redacted on the way in too, but the redactor learns secrets at
runtime and a row may predate the rule that would have caught it; the trail
cannot be rewritten, so the response is where that is fixed.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.orm import Session

from personalos.domain.audit import (
    STAGE_ORDER,
    AuditLinks,
    AuditPage,
    AuditQuery,
    AuditStage,
    AuditTrailEntry,
    LinkedApproval,
    LinkedPolicyDecision,
    LinkedToolExecution,
    WorkflowAuditTrail,
)
from personalos.domain.models import AuditEventResult
from personalos.domain.redaction import redact
from personalos.domain.workflow import application_transition_thread_id
from personalos.persistence.approvals import AUDIT_DECISION_PREFIX
from personalos.persistence.models import (
    ApplicationModel,
    ApprovalModel,
    AuditEventModel,
    PolicyDecisionModel,
    ToolExecutionModel,
    WorkflowModel,
    WorkflowRunModel,
)


class AuditTrailStore:
    """Reads the audit trail. Writes nothing."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Take the factory each read opens its session from."""
        self.session_factory = session_factory

    def search(self, query: AuditQuery) -> AuditPage:
        """Audit events matching `query`, newest first, a page at a time."""
        session = self.session_factory()
        try:
            q = session.query(AuditEventModel)
            if query.workflow_id is not None:
                q = q.filter(AuditEventModel.workflow_id == query.workflow_id)
            if query.application_id is not None:
                q = q.filter(
                    AuditEventModel.workflow_id.in_(
                        _application_workflow_ids(session, query.application_id)
                    )
                )
            if query.actor is not None:
                q = q.filter(AuditEventModel.actor == query.actor)
            if query.action is not None:
                q = q.filter(AuditEventModel.action == query.action)
            if query.since is not None:
                q = q.filter(AuditEventModel.timestamp >= query.since)
            if query.until is not None:
                q = q.filter(AuditEventModel.timestamp <= query.until)

            total = q.count()
            events = (
                q.order_by(AuditEventModel.timestamp.desc(), AuditEventModel.id.desc())
                .offset(query.offset)
                .limit(query.limit)
                .all()
            )
            chains = _Chains.load(session, events)
            return AuditPage(
                entries=tuple(chains.entry(event) for event in events),
                total=total,
                limit=query.limit,
                offset=query.offset,
            )
        finally:
            session.close()

    def workflow_trail(self, workflow_id: UUID) -> WorkflowAuditTrail | None:
        """Everything recorded for one workflow, oldest first; `None` if no such workflow.

        Each action appears as intent, then its policy decisions, approval and
        execution, interleaved with the workflow's other audit events in the
        order they were recorded.
        """
        session = self.session_factory()
        try:
            if session.get(WorkflowModel, workflow_id) is None:
                return None
            events = (
                session.query(AuditEventModel)
                .filter(AuditEventModel.workflow_id == workflow_id)
                .order_by(AuditEventModel.timestamp, AuditEventModel.id)
                .all()
            )
            decisions = (
                session.query(PolicyDecisionModel)
                .filter(PolicyDecisionModel.workflow_id == workflow_id)
                .order_by(PolicyDecisionModel.decided_at, PolicyDecisionModel.id)
                .all()
            )
            chains = _Chains.load(session, events, workflow_ids={workflow_id}, decisions=decisions)

            entries: list[AuditTrailEntry] = []
            proposed: set[str] = set()
            for decision in decisions:
                if decision.args_hash not in proposed:
                    proposed.add(decision.args_hash)
                    entries.append(chains.intent(decision, events))
                entries.append(chains.decision(decision))
            entries.extend(chains.entry(event) for event in events)
            entries.sort(key=lambda entry: (entry.at, STAGE_ORDER[entry.stage]))
            return WorkflowAuditTrail(workflow_id=workflow_id, entries=tuple(entries))
        finally:
            session.close()


# --- Linking ----------------------------------------------------------------------


class _Chains:
    """The rows behind a set of audit events, indexed to resolve each entry's links."""

    def __init__(
        self,
        decisions: Iterable[PolicyDecisionModel],
        executions: Iterable[ToolExecutionModel],
        approvals: Iterable[ApprovalModel],
        decided_by: dict[UUID, str],
    ):
        self.decisions = {row.id: row for row in decisions}
        self.executions = {row.operation_id: row for row in executions}
        self.approvals = {row.id: row for row in approvals}
        self.decided_by = decided_by

        self.decisions_by_hash: dict[str, list[PolicyDecisionModel]] = defaultdict(list)
        for decision in sorted(self.decisions.values(), key=lambda r: r.decided_at):
            self.decisions_by_hash[decision.args_hash].append(decision)
        # Later rows win: the latest execution and approval of an action.
        self.executions_by_approval: dict[str, ToolExecutionModel] = {}
        self.executions_by_hash: dict[str, ToolExecutionModel] = {}
        for execution in sorted(self.executions.values(), key=lambda r: r.created_at):
            if execution.approval_ref:
                self.executions_by_approval[execution.approval_ref] = execution
            if execution.request_fingerprint:
                self.executions_by_hash[execution.request_fingerprint] = execution
        self.approvals_by_hash: dict[str, ApprovalModel] = {}
        for approval in sorted(self.approvals.values(), key=lambda r: r.created_at):
            self.approvals_by_hash[approval.action_hash] = approval

    @classmethod
    def load(
        cls,
        session: Session,
        events: list[AuditEventModel],
        *,
        workflow_ids: set[UUID] | None = None,
        decisions: Sequence[PolicyDecisionModel] = (),
    ) -> "_Chains":
        """Fetch every row `events` (and `decisions`) cite, or share an action hash with."""
        workflow_ids = set(workflow_ids or ()) | {e.workflow_id for e in events if e.workflow_id}
        operation_ids = {e.operation_id for e in events if e.operation_id}
        approval_refs = {e.approval_ref for e in events if e.approval_ref}
        decision_hashes = {d.args_hash for d in decisions}

        conditions = []
        if operation_ids:
            conditions.append(ToolExecutionModel.operation_id.in_(operation_ids))
        if approval_refs:
            conditions.append(ToolExecutionModel.approval_ref.in_(approval_refs))
        if workflow_ids:
            conditions.append(ToolExecutionModel.workflow_id.in_(workflow_ids))
        if decision_hashes:
            conditions.append(ToolExecutionModel.request_fingerprint.in_(decision_hashes))
        executions = (
            session.query(ToolExecutionModel).filter(or_(*conditions)).all() if conditions else []
        )

        approval_ids = _uuids(
            approval_refs | {x.approval_ref for x in executions if x.approval_ref}
        )
        approval_conditions = []
        if approval_ids:
            approval_conditions.append(ApprovalModel.id.in_(approval_ids))
        if decision_hashes:
            approval_conditions.append(ApprovalModel.action_hash.in_(decision_hashes))
        approvals = (
            session.query(ApprovalModel).filter(or_(*approval_conditions)).all()
            if approval_conditions
            else []
        )

        decision_ids = {e.policy_decision_id for e in events if e.policy_decision_id} | {
            x.policy_decision_id for x in executions if x.policy_decision_id
        }
        hashes = {a.action_hash for a in approvals} | {
            x.request_fingerprint for x in executions if x.request_fingerprint
        }
        decision_conditions = []
        if decision_ids:
            decision_conditions.append(PolicyDecisionModel.id.in_(decision_ids))
        if hashes and workflow_ids:
            decision_conditions.append(
                PolicyDecisionModel.workflow_id.in_(workflow_ids)
                & PolicyDecisionModel.args_hash.in_(hashes)
            )
        linked_decisions = (
            session.query(PolicyDecisionModel).filter(or_(*decision_conditions)).all()
            if decision_conditions
            else []
        )

        return cls(
            [*decisions, *linked_decisions],
            executions,
            approvals,
            _decided_by(session, [a.id for a in approvals]),
        )

    # --- Entries ------------------------------------------------------------------

    def entry(self, event: AuditEventModel) -> AuditTrailEntry:
        """An `audit_events` row as a trail entry, with its chain attached."""
        execution = self.executions.get(event.operation_id) if event.operation_id else None
        approval = self._approval_by_ref(event.approval_ref) or (
            self._approval_by_ref(execution.approval_ref) if execution else None
        )
        if event.action.startswith("approval."):
            stage = AuditStage.APPROVAL
            execution = execution or (
                self.executions_by_approval.get(event.approval_ref) if event.approval_ref else None
            )
        elif event.operation_id is not None:
            stage = AuditStage.EXECUTION
        else:
            stage = AuditStage.EVENT

        action_hash = (
            (execution.request_fingerprint if execution else None)
            or (approval.action_hash if approval else None)
            or self._decision_hash(event.policy_decision_id)
        )
        decision = (
            self.decisions.get(event.policy_decision_id) if event.policy_decision_id else None
        )
        decision = decision or (
            self.decisions.get(execution.policy_decision_id)
            if execution and execution.policy_decision_id
            else None
        )
        decision = decision or self._decision_before(action_hash, event.timestamp)
        return _entry(
            stage=stage,
            at=event.timestamp,
            actor=event.actor,
            action=event.action,
            target_ref=event.target_ref,
            result=event.result,
            policy_decision=event.policy_decision or (decision.decision if decision else None),
            action_hash=action_hash,
            workflow_id=event.workflow_id,
            audit_event_id=event.id,
            links=self._links(decision, execution, approval),
        )

    def intent(
        self, decision: PolicyDecisionModel, events: list[AuditEventModel]
    ) -> AuditTrailEntry:
        """The action as first proposed, read from the first decision on its hash."""
        execution, approval = self._downstream(decision)
        return _entry(
            stage=AuditStage.INTENT,
            at=decision.decided_at,
            actor=decision.principal,
            action=decision.tool,
            target_ref=self._target_of(decision.args_hash, events),
            action_hash=decision.args_hash,
            workflow_id=decision.workflow_id,
            links=self._links(decision, execution, approval),
        )

    def decision(self, decision: PolicyDecisionModel) -> AuditTrailEntry:
        """One verdict of the policy engine."""
        execution, approval = self._downstream(decision)
        return _entry(
            stage=AuditStage.POLICY_DECISION,
            at=decision.decided_at,
            actor=decision.principal,
            action=decision.tool,
            policy_decision=decision.decision,
            action_hash=decision.args_hash,
            workflow_id=decision.workflow_id,
            links=self._links(decision, execution, approval),
        )

    # --- Resolution ---------------------------------------------------------------

    def _downstream(self, decision: PolicyDecisionModel):
        """The execution and approval that followed a decision, if there were any."""
        execution = next(
            (x for x in self.executions.values() if x.policy_decision_id == decision.id), None
        ) or self.executions_by_hash.get(decision.args_hash)
        approval = (
            self._approval_by_ref(execution.approval_ref) if execution else None
        ) or self.approvals_by_hash.get(decision.args_hash)
        return execution, approval

    def _approval_by_ref(self, ref: str | None) -> ApprovalModel | None:
        ids = _uuids([ref]) if ref else set()
        return self.approvals.get(next(iter(ids))) if ids else None

    def _decision_hash(self, decision_id: UUID | None) -> str | None:
        decision = self.decisions.get(decision_id) if decision_id else None
        return decision.args_hash if decision else None

    def _decision_before(self, action_hash: str | None, at) -> PolicyDecisionModel | None:
        """The latest decision on the action at or before `at`, else its first."""
        candidates = self.decisions_by_hash.get(action_hash or "", [])
        before = [d for d in candidates if d.decided_at <= at]
        return before[-1] if before else (candidates[0] if candidates else None)

    def _target_of(self, action_hash: str, events: list[AuditEventModel]) -> str | None:
        """What the action was aimed at, from the first audit event in its chain."""
        for event in events:
            execution = self.executions.get(event.operation_id) if event.operation_id else None
            approval = self._approval_by_ref(event.approval_ref)
            hashes = {
                execution.request_fingerprint if execution else None,
                approval.action_hash if approval else None,
                self._decision_hash(event.policy_decision_id),
            }
            if action_hash in hashes:
                return event.target_ref
        return None

    def _links(self, decision, execution, approval) -> AuditLinks:
        return AuditLinks(
            policy_decision=_linked_decision(decision) if decision else None,
            tool_execution=_linked_execution(execution) if execution else None,
            approval=self._linked_approval(approval) if approval else None,
        )

    def _linked_approval(self, row: ApprovalModel) -> LinkedApproval:
        return LinkedApproval(
            **redact(
                {
                    "id": row.id,
                    "status": row.status,
                    "action_hash": row.action_hash,
                    "decided_by": self.decided_by.get(row.id),
                    "requested_at": row.created_at,
                    "decided_at": row.decided_at,
                }
            )
        )


# --- Row projections ------------------------------------------------------------
#
# Every projection is built from a redacted dict, so no column reaches a
# response without passing the redactor.


def _entry(**fields: Any) -> AuditTrailEntry:
    links = fields.pop("links")
    return AuditTrailEntry(**redact(fields), links=links)


def _linked_decision(row: PolicyDecisionModel) -> LinkedPolicyDecision:
    return LinkedPolicyDecision(
        **redact(
            {
                "id": row.id,
                "principal": row.principal,
                "tool": row.tool,
                "args_hash": row.args_hash,
                "decision": row.decision,
                "requested_scopes": list(row.requested_scopes or []),
                "decided_at": row.decided_at,
            }
        )
    )


def _linked_execution(row: ToolExecutionModel) -> LinkedToolExecution:
    return LinkedToolExecution(
        **redact(
            {
                "operation_id": row.operation_id,
                "tool_name": row.tool_name,
                "idempotency_key": row.idempotency_key,
                "status": row.status,
                "request_fingerprint": row.request_fingerprint,
                "policy_decision_id": row.policy_decision_id,
                "approval_ref": row.approval_ref,
                "approved_by": row.approved_by,
                "attempts": row.attempts,
                "receipt": row.receipt_json,
                "error": row.error,
                "created_at": row.created_at,
                "completed_at": row.completed_at,
            }
        )
    )


# --- Lookups ----------------------------------------------------------------------


def _application_workflow_ids(session: Session, application_id: UUID) -> list[UUID]:
    """The workflows run on an application's own thread (see `application_transition_thread_id`)."""
    application = session.get(ApplicationModel, application_id)
    if application is None:
        return []
    thread_id = application_transition_thread_id(application.user_id, application_id)
    return [
        row.workflow_id
        for row in session.query(WorkflowRunModel.workflow_id)
        .filter(WorkflowRunModel.thread_id == thread_id)
        .all()
    ]


def _decided_by(session: Session, approval_ids: list[UUID]) -> dict[UUID, str]:
    """Who recorded each approval's decision, from its successful `approval.*` event."""
    if not approval_ids:
        return {}
    rows = (
        session.query(AuditEventModel.approval_ref, AuditEventModel.actor)
        .filter(
            AuditEventModel.approval_ref.in_([str(i) for i in approval_ids]),
            AuditEventModel.action.startswith(AUDIT_DECISION_PREFIX),
            AuditEventModel.result == AuditEventResult.SUCCESS.value,
        )
        .all()
    )
    return {UUID(ref): actor for ref, actor in rows}


def _uuids(refs: Iterable[str | None]) -> set[UUID]:
    """The refs that are UUIDs; an approval ref may also be an action id or free text."""
    parsed = set()
    for ref in refs:
        try:
            parsed.add(UUID(str(ref)))
        except ValueError:
            continue
    return parsed


__all__ = ["AuditTrailStore"]
