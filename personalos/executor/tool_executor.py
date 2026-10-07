"""The executor every mutating action goes through.

An outward-facing action -- submitting an application, sending a recruiter a
message -- has to hold two properties that neither the graph nor the provider
adapter can give it on their own:

**It happens at most once per idempotency key.** The flow is fixed:

1. policy is asked, and its verdict is committed to `policy_decisions`;
2. the key is claimed in `tool_executions` (status `in_progress`), carrying
   that decision and the approval it was redeemed with, and committed;
3. the provider is called;
4. the receipt, status `completed`, the `audit_events` row and the domain event
   are committed together.

A retry that finds step 4 done returns the stored receipt and calls nothing. A
retry that finds a claim with no outcome does not know whether step 3 took
effect, so the row becomes `unknown` and the *provider* is asked: if the action
landed, its receipt is recorded and returned; if it provably did not, the
action is executed again; if the provider cannot say, nothing is executed and
the action comes back not-ok for a human to settle. With no reconciler wired
the last case is the only one, which is the rule
`personalos.persistence.action_journal` documents: a duplicate application
cannot be taken back, a missed one can be resubmitted deliberately.

An `in_progress` row found on entry is treated as an abandoned attempt, not a
live one. That rests on the workflow lease (`personalos.persistence.leases`),
which keeps two workers off the same run.

**It is auditable back to what authorized it.** Every executed action leaves an
`audit_events` row naming the actor, the workflow, the action, its target, the
result, and the `policy_decisions` row (and approval, where one was needed)
that it ran under. The same references are on the `tool_executions` row.

This satisfies the graph's `ActionExecutor` port, so it is bound in the
composition root in place of the bare provider adapter it wraps.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol
from uuid import UUID

from personalos.domain.errors import IdempotencyConflict
from personalos.domain.job_search import ActionIntent, ActionKind, ActionReceipt, ApprovalDecision
from personalos.domain.models import ToolExecutionStatus
from personalos.domain.redaction import redact
from personalos.persistence.execution_ledger import (
    AuditEntry,
    DomainEvent,
    ExecutionAuthorization,
    ExecutionLedger,
)
from personalos.policy import (
    ApprovalRequired,
    Decision,
    IntentOrigin,
    PolicyDecision,
    PolicyDenied,
    PolicyEngine,
)
from personalos.policy.permissions import Provenance

logger = logging.getLogger(__name__)

#: The classified tool each kind of action is evaluated as. The permission
#: class and scopes policy applies come from this reference, so a kind with no
#: entry falls through to an unclassified name and is denied.
ACTION_TOOLS: Mapping[ActionKind, str] = MappingProxyType(
    {
        ActionKind.SUBMIT_APPLICATION: "jobs.submit_application",
        ActionKind.SEND_RECRUITER_MESSAGE: "google.gmail_send_message",
        ActionKind.OVERWRITE_DOCUMENT: "files.overwrite_document",
        ActionKind.CREATE_CALENDAR_EVENT: "google.calendar_create_event",
        ActionKind.UPDATE_CALENDAR_EVENT: "google.calendar_update_event",
    }
)

#: `outbox_events.type` enqueued when an action is confirmed to have taken effect.
ACTION_SUCCEEDED_EVENT = "action.succeeded"


class ActionExecutorPort(Protocol):
    """The provider adapter this executor wraps."""

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Perform the approved action and return its receipt."""
        ...


class ReconcileOutcome(str, Enum):
    """What the provider says about an action whose outcome was never recorded."""

    #: The action took effect. Its receipt is recorded and nothing is re-run.
    APPLIED = "applied"
    #: The action provably never took effect. It is safe to execute again.
    NOT_APPLIED = "not_applied"
    #: The provider cannot say either way. Nothing is executed.
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class Reconciliation:
    """A reconciler's answer. `receipt` is required when the action was applied."""

    outcome: ReconcileOutcome
    receipt: ActionReceipt | None = None


class ProviderReconciler(Protocol):
    """Asks the provider whether an action already took effect.

    Implemented per provider, typically by looking the action up under the
    idempotency key it was sent with. It must answer `NOT_APPLIED` only when
    the provider positively reports no such action: that answer is what
    permits a second call.
    """

    async def reconcile(self, intent: ActionIntent) -> Reconciliation:
        """Report whether this intent's side effect exists at the provider."""
        ...


class ToolExecutor:
    """Runs a mutating action at most once, under a recorded policy decision."""

    def __init__(
        self,
        inner: ActionExecutorPort,
        policy: PolicyEngine,
        ledger: ExecutionLedger,
        *,
        reconciler: ProviderReconciler | None = None,
        action_tools: Mapping[ActionKind, str] = ACTION_TOOLS,
    ):
        """Wrap `inner`, authorizing through `policy` and recording through `ledger`."""
        if inner is None:
            raise ValueError("ToolExecutor requires an inner ActionExecutor")
        if policy is None:
            raise ValueError(
                "ToolExecutor requires a PolicyEngine; a mutating action cannot run "
                "without a policy decision to record against it"
            )
        self.inner = inner
        self.policy = policy
        self.ledger = ledger
        self.reconciler = reconciler
        self.action_tools = action_tools

    @property
    def workflow_id(self) -> UUID | None:
        """The workflow this executor's actions are recorded against."""
        return self.ledger.workflow_id

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Authorize, then execute the action unless its key already has an outcome.

        Raises `PolicyDenied` or `ApprovalRequired` when policy does not clear
        the action, and `IdempotencyConflict` when the key was already used for
        a different action. Policy is asked on every attempt, replays included,
        so a retry never gets an answer the engine did not see.
        """
        tool = self.action_tools.get(intent.kind, f"job_search.{intent.kind.value}")
        authorization = self._authorize(intent, decision, tool)

        record, claimed = self.ledger.claim(
            intent.idempotency_key,
            tool,
            request_fingerprint=intent.fingerprint(),
            authorization=authorization,
        )
        if claimed:
            return await self._run(intent, decision, tool)

        stored_fingerprint = record.get("request_fingerprint")
        if stored_fingerprint and stored_fingerprint != intent.fingerprint():
            raise IdempotencyConflict(
                f"idempotency_key '{intent.idempotency_key}' was already used for a "
                f"different action; use a new key",
                details={"idempotency_key": intent.idempotency_key},
            )

        if record.get("status") == ToolExecutionStatus.COMPLETED.value and record.get(
            "receipt_json"
        ):
            logger.info(
                "action %s (%s) already completed under idempotency key '%s'; returning "
                "its stored receipt instead of executing again",
                intent.action_id,
                intent.kind.value,
                intent.idempotency_key,
            )
            return self._rebind(intent, record["receipt_json"])

        return await self._resolve_in_doubt(intent, decision, tool, record, authorization)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _authorize(
        self, intent: ActionIntent, decision: ApprovalDecision, tool: str
    ) -> ExecutionAuthorization:
        """Get this attempt's policy verdict and check the approval against it."""
        verdict: PolicyDecision = self.policy.evaluate_action(
            intent.requested_by,
            self.workflow_id,
            tool,
            intent.fingerprint(),
            list(intent.risk_profile().scopes),
            # SYSTEM: the action's shape is fixed by the graph, and whatever a
            # model contributed to its payload is what the reviewer approved.
            Provenance(origin=IntentOrigin.SYSTEM, requested_by=intent.requested_by),
            action_id=intent.action_id,
        )
        if verdict.decision == Decision.DENY:
            raise PolicyDenied(verdict)

        approved = decision is not None and decision.authorizes(intent)
        if verdict.decision == Decision.REQUIRE_APPROVAL and not approved:
            raise ApprovalRequired(verdict)

        return ExecutionAuthorization(
            policy_decision_id=verdict.record_id,
            approval_ref=str(decision.request_id or decision.action_id) if approved else None,
            approved_by=decision.decided_by if approved else None,
        )

    async def _run(self, intent: ActionIntent, decision: ApprovalDecision, tool: str):
        """Call the provider for a claimed key and commit what came back."""
        audit = self._audit_entry(intent, tool)
        try:
            receipt = await self.inner.execute(intent, decision)
        except Exception as exc:
            self.ledger.record_failure(intent.idempotency_key, str(exc), audit=audit)
            raise

        # Redacted once, here, so the stored receipt and the returned one are
        # the same value: a replay must hand back exactly what this attempt did.
        receipt = redact(receipt)
        self._record(intent, receipt, audit)
        return receipt

    async def _resolve_in_doubt(
        self,
        intent: ActionIntent,
        decision: ApprovalDecision,
        tool: str,
        record: dict,
        authorization: ExecutionAuthorization,
    ) -> ActionReceipt:
        """Settle a key that was claimed before and has no receipt."""
        key = intent.idempotency_key
        self.ledger.mark_unknown(key)

        if self.reconciler is None:
            return self._in_doubt(intent, record, "no reconciler is configured for it")

        try:
            reconciliation = await self.reconciler.reconcile(intent)
        except Exception:
            logger.exception(
                "reconciling action %s (%s) against the provider failed",
                intent.action_id,
                intent.kind.value,
            )
            return self._in_doubt(intent, record, "the provider could not be asked")

        if reconciliation.outcome == ReconcileOutcome.APPLIED and reconciliation.receipt:
            receipt = redact(
                reconciliation.receipt.model_copy(update={"action_id": intent.action_id})
            )
            logger.info(
                "action %s (%s) was found at the provider; recording its receipt "
                "without executing again",
                intent.action_id,
                intent.kind.value,
            )
            self._record(intent, receipt, self._audit_entry(intent, tool))
            return receipt

        if reconciliation.outcome == ReconcileOutcome.NOT_APPLIED:
            if self.ledger.reclaim(key, authorization):
                logger.info(
                    "action %s (%s) never reached the provider; executing it again",
                    intent.action_id,
                    intent.kind.value,
                )
                return await self._run(intent, decision, tool)
            return self._in_doubt(intent, record, "another attempt took it over")

        return self._in_doubt(intent, record, "the provider could not confirm either way")

    def _record(self, intent: ActionIntent, receipt: ActionReceipt, audit: AuditEntry) -> None:
        self.ledger.record_outcome(
            intent.idempotency_key,
            receipt.model_dump(mode="json"),
            succeeded=receipt.ok,
            audit=audit,
            event=DomainEvent(
                type=ACTION_SUCCEEDED_EVENT,
                payload={
                    "kind": intent.kind.value,
                    "target": intent.target,
                    "idempotency_key": intent.idempotency_key,
                    "workflow_id": str(self.workflow_id) if self.workflow_id else None,
                    "external_reference": receipt.external_reference,
                },
            ),
        )

    @staticmethod
    def _audit_entry(intent: ActionIntent, tool: str) -> AuditEntry:
        return AuditEntry(actor=intent.requested_by, action=tool, target_ref=intent.target)

    @staticmethod
    def _rebind(intent: ActionIntent, stored: dict) -> ActionReceipt:
        """The stored receipt, addressed to the intent now asking for it.

        `action_id` is minted per intent, so a re-proposed action has a new
        one; the rest of the run matches receipts to intents on it.
        """
        return ActionReceipt.model_validate({**stored, "action_id": str(intent.action_id)})

    @staticmethod
    def _in_doubt(intent: ActionIntent, record: dict, why: str) -> ActionReceipt:
        detail = (
            f"a previous attempt at '{intent.idempotency_key}' did not record an outcome "
            f"(status={record.get('status')}) and {why}; not retried, because this action "
            f"may already have taken effect outside this system"
        )
        logger.warning(
            "action %s (%s) is in doubt: %s", intent.action_id, intent.kind.value, detail
        )
        return ActionReceipt(action_id=intent.action_id, ok=False, detail=detail)


__all__ = [
    "ACTION_SUCCEEDED_EVENT",
    "ACTION_TOOLS",
    "ActionExecutorPort",
    "ProviderReconciler",
    "ReconcileOutcome",
    "Reconciliation",
    "ToolExecutor",
]
