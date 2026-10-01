"""The durable record of one mutating action: its execution row and its audit trail.

`ToolExecutionRepository`, `AuditEventRepository` and `OutboxEventRepository`
each commit on their own. That is right for a single write and wrong for an
outcome, which is three writes that have to agree: the receipt on the
`tool_executions` row, the `audit_events` row saying what happened and under
which policy decision, and the outbox row announcing it. Committed separately,
a crash between them leaves a submitted application with no audit entry, or an
audit entry for an action whose receipt was never stored.

So this module is where those writes are grouped into transactions:

- **the claim**, committed alone and *before* the provider is called, carrying
  the policy decision and approval the call is being made under;
- **the outcome**, committed once and *after* it: receipt and status, audit
  row, and domain event together or not at all.

Everything is its own short-lived session from a factory, for the reason
`SqlOperationStore` gives: the claim has to be visible to other processes
before the side effect runs, which it cannot be inside a caller's transaction.

This module stores what it is given and decides nothing. Whether an action may
run, and what to do about one whose outcome is unknown, is
`personalos.executor.tool_executor`'s call.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.models import AuditEventResult
from personalos.persistence.repositories import (
    AuditEventRepository,
    OutboxEventRepository,
    PolicyDecisionRepository,
    ToolExecutionRepository,
    WorkflowRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionAuthorization:
    """What an execution is being made under.

    `policy_decision_id` is the `policy_decisions` row the engine wrote for
    this attempt; `approval_ref` and `approved_by` identify the human approval
    that verdict was redeemed with, and are `None` for an action policy allowed
    outright.
    """

    policy_decision_id: UUID | None = None
    approval_ref: str | None = None
    approved_by: str | None = None


@dataclass(frozen=True)
class AuditEntry:
    """Who did what to what. The decision and the result are added by the ledger."""

    actor: str
    action: str
    target_ref: str


@dataclass(frozen=True)
class DomainEvent:
    """An event to enqueue in the outbox alongside a successful outcome."""

    type: str
    payload: dict[str, Any]


class ExecutionLedger:
    """Writes the `tool_executions` and `audit_events` records for mutating actions."""

    def __init__(self, session_factory: Callable[[], Session], *, workflow_id: UUID | None = None):
        """Take the factory each write opens its session from."""
        self.session_factory = session_factory
        self.workflow_id = workflow_id

    def claim(
        self,
        idempotency_key: str,
        tool_name: str,
        *,
        request_fingerprint: str,
        authorization: ExecutionAuthorization,
    ) -> tuple[dict[str, Any], bool]:
        """Commit the intent to execute. Returns (record, claimed).

        When `claimed` is False a row for this key already existed and is
        returned as it stands, with the authorization it was first made under.
        """
        session = self.session_factory()
        try:
            record, claimed = ToolExecutionRepository(session).claim(
                idempotency_key,
                tool_name,
                self._linked_workflow_id(session),
                request_fingerprint=request_fingerprint,
                policy_decision_id=authorization.policy_decision_id,
                approval_ref=authorization.approval_ref,
                approved_by=authorization.approved_by,
            )
            return record.to_dict(), claimed
        finally:
            session.close()

    def record_outcome(
        self,
        idempotency_key: str,
        receipt: dict[str, Any],
        *,
        succeeded: bool,
        audit: AuditEntry,
        event: DomainEvent | None = None,
    ) -> None:
        """Commit the provider's receipt, its audit row and its event as one write.

        `succeeded` is whether the provider accepted the action. A rejection is
        still a recorded, replayable outcome -- the call was made and answered
        -- so the row completes either way; only the audit result differs, and
        `event` is enqueued for a success only.
        """
        session = self.session_factory()
        try:
            execution = ToolExecutionRepository(session).complete(
                idempotency_key, receipt, commit=False
            )
            self._audit(
                session,
                execution,
                audit,
                AuditEventResult.SUCCESS if succeeded else AuditEventResult.FAILURE,
            )
            if event is not None and succeeded:
                OutboxEventRepository(session).create(
                    type=event.type,
                    payload={**event.payload, "operation_id": str(execution.operation_id)},
                    # One execution row, one event: a second enqueue for the
                    # same execution is a duplicate by construction.
                    dedupe_key=f"{event.type}:{execution.operation_id}",
                    commit=False,
                )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def record_failure(self, idempotency_key: str, error: str, *, audit: AuditEntry) -> None:
        """Commit a call that raised, with the audit row saying so."""
        session = self.session_factory()
        try:
            execution = ToolExecutionRepository(session).fail(idempotency_key, error, commit=False)
            self._audit(session, execution, audit, AuditEventResult.FAILURE)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def mark_unknown(self, idempotency_key: str) -> bool:
        """Flag an execution with no recorded outcome as in doubt."""
        session = self.session_factory()
        try:
            return ToolExecutionRepository(session).mark_unknown(idempotency_key)
        finally:
            session.close()

    def reclaim(self, idempotency_key: str, authorization: ExecutionAuthorization) -> bool:
        """Take an in-doubt execution back for another attempt. True when this caller won it."""
        session = self.session_factory()
        try:
            return ToolExecutionRepository(session).reclaim(
                idempotency_key,
                policy_decision_id=authorization.policy_decision_id,
                approval_ref=authorization.approval_ref,
                approved_by=authorization.approved_by,
            )
        finally:
            session.close()

    def _audit(self, session: Session, execution, audit: AuditEntry, result: AuditEventResult):
        """Stage the audit row for an execution, citing what that row was authorized by.

        The decision is read off the execution row rather than passed in, so
        an audit entry can only ever name the decision the call was actually
        made under -- including when the outcome is being recorded by a later
        attempt that reconciled it, under a different decision of its own.
        """
        decision = (
            PolicyDecisionRepository(session).get_by_id(execution.policy_decision_id)
            if execution.policy_decision_id
            else None
        )
        AuditEventRepository(session).create(
            actor=audit.actor,
            action=audit.action,
            target_ref=audit.target_ref,
            result=result.value,
            workflow_id=execution.workflow_id,
            policy_decision=decision.decision if decision else None,
            policy_decision_id=execution.policy_decision_id,
            operation_id=execution.operation_id,
            approval_ref=execution.approval_ref,
            commit=False,
        )

    def _linked_workflow_id(self, session: Session) -> UUID | None:
        """`workflow_id`, or None when that workflow was never registered.

        `tool_executions.workflow_id` references `workflows`; an action from an
        unregistered run is still recorded, just without the link, as
        `SqlPolicyDecisionLog` does for its own rows.
        """
        if self.workflow_id is None:
            return None
        if WorkflowRepository(session).get_by_id(self.workflow_id) is None:
            logger.debug(
                "workflow %s is not registered; recording execution unlinked", self.workflow_id
            )
            return None
        return self.workflow_id


__all__ = ["AuditEntry", "DomainEvent", "ExecutionAuthorization", "ExecutionLedger"]
