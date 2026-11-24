"""Writes a cross-domain plan's story into the append-only audit trail.

`SqlPlanAuditRecorder` satisfies the Supervisor's `PlanAuditRecorder` port.
Each `PlanAuditEntry` becomes one `audit_events` row under the plan's
workflow: the plan as proposed, then for every step its intent, the policy
verdict it runs under, its approval and its receipt, then any dangling effect
and how the plan finished (`personalos.domain.task_plan`).

**In order.** The trail is read oldest first, tie-broken by a random row id,
so rows written in one call get strictly increasing timestamps: a step's
approval can never sort before its intent.

**Once.** An entry's key leads its `target_ref`, and a row with the same
workflow, action and `target_ref` is not written again -- a Supervisor step
re-run after a crash re-records what it had recorded, and that is a no-op.

**Linked.** An approval entry carries the step's `ApprovalRequest` id as its
`approval_ref`, which is the `approvals` row's id: the trail links a plan's
approval entry to the decision a person recorded.
"""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.redaction import redact_text
from personalos.domain.task_plan import PlanAuditEntry
from personalos.persistence.models import AuditEventModel, WorkflowModel

_TICK = timedelta(microseconds=1)


class SqlPlanAuditRecorder:
    """Appends plan audit entries to `audit_events`, in order and at most once each."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        self.session_factory = session_factory
        self.clock = clock

    def record(
        self,
        entries: Sequence[PlanAuditEntry],
        *,
        workflow_id: str | None,
        actor_id: str,
    ) -> None:
        if not entries:
            return
        session = self.session_factory()
        try:
            workflow = UUID(str(workflow_id)) if workflow_id else None
            if workflow is not None and session.get(WorkflowModel, workflow) is None:
                workflow = None
            latest = None
            if workflow is not None:
                latest = (
                    session.query(AuditEventModel.timestamp)
                    .filter(AuditEventModel.workflow_id == workflow)
                    .order_by(AuditEventModel.timestamp.desc())
                    .limit(1)
                    .scalar()
                )
            at = self.clock()
            if latest is not None and at <= latest:
                at = latest + _TICK
            for entry in entries:
                target_ref = redact_text(
                    entry.target_ref
                    if entry.target_ref.startswith(entry.key)
                    else f"{entry.key} | {entry.target_ref}"
                )[:500]
                action = redact_text(entry.action)[:255]
                exists = (
                    session.query(AuditEventModel.id)
                    .filter(
                        AuditEventModel.workflow_id == workflow,
                        AuditEventModel.action == action,
                        AuditEventModel.target_ref == target_ref,
                    )
                    .first()
                )
                if exists is not None:
                    continue
                session.add(
                    AuditEventModel(
                        actor=redact_text(actor_id)[:255],
                        workflow_id=workflow,
                        action=action,
                        target_ref=target_ref,
                        policy_decision=entry.policy_decision,
                        approval_ref=entry.approval_ref,
                        result=entry.result,
                        timestamp=at,
                    )
                )
                at += _TICK
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


__all__ = ["SqlPlanAuditRecorder"]
