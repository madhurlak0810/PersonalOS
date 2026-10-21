"""A workflow's failed steps, gathered from the rows that each know part of them.

`GET /v1/workflows/{id}/failures` needs, per failed step:

- the stored error and where it was raised -- the `__error__` checkpoint
  write, via `WorkflowStatusReader`;
- the side effects the step had started -- `tool_executions` rows of the
  workflow claimed since the checkpoint the step began from;
- how many times it has been retried -- retry commands queued for it, counted
  in `outbox_events` under the failure's own dedupe prefix;
- whether it was dismissed -- an `audit_events` row, append-only like every
  other one;
- the health of the integration it broke, when it broke one.

What that adds up to -- class, status, whether a retry is allowed -- is
`personalos.domain.failures.assess_failure`'s call. This module only reads,
and records the two things a user does to a failure in the audit trail.
"""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.errors import FailureClass
from personalos.domain.failures import (
    FailedOperation,
    WorkflowFailure,
    assess_failure,
    classify,
    failure_id,
    operation_recovery,
    parse_stored_error,
    redacted_receipt,
)
from personalos.domain.models import AuditEventResult
from personalos.domain.redaction import redact_text
from personalos.domain.workflow import RecoverableFailure, WorkflowCommandKind, WorkflowSnapshot
from personalos.persistence.integration_health import IntegrationHealthStore
from personalos.persistence.models import AuditEventModel, ToolExecutionModel
from personalos.persistence.repositories import AuditEventRepository
from personalos.persistence.workflow_commands import WorkflowCommandQueue
from personalos.persistence.workflow_status import WorkflowStatusReader

#: `audit_events.action` for a dismissed failure, and for a queued retry.
FAILURE_DISMISSED = "workflow.failure.dismissed"
FAILURE_RETRY_REQUESTED = "workflow.failure.retry_requested"


def failure_target(failure_id: UUID) -> str:
    """`audit_events.target_ref` for an event about one failure."""
    return f"failure:{failure_id}"


def retry_dedupe_prefix(failure_id: UUID) -> str:
    """The dedupe prefix every retry of one failure is queued under."""
    return f"{failure_id}:"


def retry_dedupe(failure_id: UUID, attempt: int) -> str:
    """The dedupe key of one retry: one per attempt, so two racing retries collide."""
    return f"{retry_dedupe_prefix(failure_id)}{attempt}"


class WorkflowFailureStore:
    """Builds `WorkflowFailure`s for a workflow and records what users do to them."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        reader: WorkflowStatusReader,
        commands: WorkflowCommandQueue,
        integrations: IntegrationHealthStore,
    ):
        """Initialize over the stores a failure is assembled from."""
        self.session_factory = session_factory
        self.reader = reader
        self.commands = commands
        self.integrations = integrations

    def snapshot(self, workflow_id: UUID) -> WorkflowSnapshot | None:
        """The workflow's snapshot, or `None` when it does not exist."""
        return self.reader.read(workflow_id)

    def failures(
        self, snapshot: WorkflowSnapshot, *, include_dismissed: bool = False
    ) -> list[WorkflowFailure]:
        """Every failed step of the workflow, in thread order."""
        dismissals = self._dismissals(snapshot.workflow_id)
        found = [
            self._assess(snapshot.workflow_id, stored, dismissals)
            for stored in snapshot.recoverable_failures
        ]
        return [failure for failure in found if include_dismissed or failure.dismissed_by is None]

    def record_retry(self, failure: WorkflowFailure, *, actor: str) -> None:
        """Append the audit row for a retry that was queued."""
        self._audit(failure, actor=actor, action=FAILURE_RETRY_REQUESTED)

    def record_dismissal(self, failure: WorkflowFailure, *, actor: str) -> None:
        """Append the audit row that closes a failure without a retry."""
        self._audit(failure, actor=actor, action=FAILURE_DISMISSED)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assess(
        self,
        workflow_id: UUID,
        stored: RecoverableFailure,
        dismissals: dict[str, tuple[str, datetime]],
    ) -> WorkflowFailure:
        error = parse_stored_error(stored.message)
        integration = None
        if classify(error) == FailureClass.PROVIDER_PERMANENT and error.provider:
            integration = self.integrations.get(error.provider)

        identity = failure_id(stored.thread_id, stored.checkpoint_id, stored.task_id)
        queued, pending = self.commands.count_keyed(
            workflow_id, WorkflowCommandKind.RETRY.value, retry_dedupe_prefix(identity)
        )
        dismissed_by, dismissed_at = dismissals.get(failure_target(identity), (None, None))
        return assess_failure(
            workflow_id=workflow_id,
            thread_id=stored.thread_id,
            step=stored.step,
            checkpoint_id=stored.checkpoint_id,
            task_id=stored.task_id,
            stored_error=stored.message,
            failed_at=stored.checkpoint_at,
            operations=self._operations(workflow_id, stored.checkpoint_at),
            integration=integration,
            retries_queued=queued,
            retry_pending=pending > 0,
            dismissed_by=dismissed_by,
            dismissed_at=dismissed_at,
        )

    def _operations(self, workflow_id: UUID, since: datetime | None) -> list[FailedOperation]:
        """Side effects the workflow claimed after the failed step's checkpoint.

        `tool_executions` does not say which step claimed a row, so time does:
        the failed step is what ran on its thread after that checkpoint was
        written, and a retry re-runs it against the same checkpoint.
        """
        if since is None:
            return []
        session = self.session_factory()
        try:
            rows = (
                session.query(ToolExecutionModel)
                .filter(
                    ToolExecutionModel.workflow_id == workflow_id,
                    ToolExecutionModel.created_at >= since,
                )
                .order_by(ToolExecutionModel.created_at.asc())
                .all()
            )
            return [
                FailedOperation(
                    operation_id=row.operation_id,
                    tool_name=row.tool_name,
                    idempotency_key=row.idempotency_key,
                    status=row.status,
                    attempts=row.attempts,
                    recovery=operation_recovery(row.status),
                    receipt=(
                        redacted_receipt(row.receipt_json) if row.status == "completed" else None
                    ),
                    error=redact_text(row.error) if row.error else None,
                )
                for row in rows
            ]
        finally:
            session.close()

    def _dismissals(self, workflow_id: UUID) -> dict[str, tuple[str, datetime]]:
        """Who dismissed which failure, and when; the first dismissal wins."""
        session = self.session_factory()
        try:
            rows = (
                session.query(AuditEventModel)
                .filter(
                    AuditEventModel.workflow_id == workflow_id,
                    AuditEventModel.action == FAILURE_DISMISSED,
                )
                .order_by(AuditEventModel.timestamp.asc())
                .all()
            )
            found: dict[str, tuple[str, datetime]] = {}
            for row in rows:
                found.setdefault(row.target_ref, (row.actor, row.timestamp))
            return found
        finally:
            session.close()

    def _audit(self, failure: WorkflowFailure, *, actor: str, action: str) -> None:
        session = self.session_factory()
        try:
            AuditEventRepository(session).create(
                actor=actor,
                action=action,
                target_ref=failure_target(failure.failure_id),
                result=AuditEventResult.SUCCESS.value,
                workflow_id=failure.workflow_id,
            )
        finally:
            session.close()


__all__ = [
    "FAILURE_DISMISSED",
    "FAILURE_RETRY_REQUESTED",
    "WorkflowFailureStore",
    "failure_target",
    "retry_dedupe",
    "retry_dedupe_prefix",
]
