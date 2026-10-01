"""Contract tests for `ToolExecutor`: at-most-once execution, with its paper trail.

Two properties, tested against a real database through fresh sessions:

- a mutating action runs at most once per idempotency key, including across a
  crash that left its outcome unrecorded, where the provider is asked before
  anything is re-executed;
- every executed action leaves `tool_executions` and `audit_events` rows that
  cite the `policy_decisions` row, and the approval, it ran under.

A crash is a `BaseException` raised right after the side effect, for the reason
`test_action_journal.py` gives: it is the only in-process failure that leaves a
claim with neither a receipt nor a recorded error.
"""

import pytest

from personalos.bootstrap import build_tool_executor
from personalos.domain.errors import IdempotencyConflict
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalVerdict,
)
from personalos.domain.models import AuditEventResult, ToolExecutionStatus
from personalos.executor.tool_executor import (
    ACTION_SUCCEEDED_EVENT,
    ReconcileOutcome,
    Reconciliation,
    ToolExecutor,
)
from personalos.persistence.execution_ledger import ExecutionLedger
from personalos.persistence.models import (
    AuditEventModel,
    OutboxEventModel,
    PolicyDecisionModel,
    ToolExecutionModel,
)
from personalos.policy import (
    ApprovalRequired,
    Decision,
    PermissionClass,
    PolicyDenied,
    PolicyEngine,
    default_policy_engine,
)
from tests.fixtures.durable_workflow import session_factory

SUBMIT_TOOL = "jobs.submit_application"


class _Crash(BaseException):
    """A failure that unwinds past `except Exception`, like a process death."""


class SpyProvider:
    """Counts the side effects it performed, and can die or fail around one."""

    def __init__(self, *, ok: bool = True, crash_after: bool = False, error: str | None = None):
        self.ok = ok
        self.crash_after = crash_after
        self.error = error
        self.calls: list[ActionIntent] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        if self.error:
            raise RuntimeError(self.error)
        self.calls.append(intent)
        if self.crash_after:
            raise _Crash("worker died after submitting, before recording the receipt")
        return ActionReceipt(
            action_id=intent.action_id,
            ok=self.ok,
            external_reference=f"ext-{len(self.calls)}" if self.ok else None,
            detail=None if self.ok else "provider rejected it",
        )


class ScriptedReconciler:
    """Answers with a fixed outcome and records what it was asked about."""

    def __init__(self, outcome: ReconcileOutcome, *, external_reference: str | None = None):
        self.outcome = outcome
        self.external_reference = external_reference
        self.asked: list[str] = []

    async def reconcile(self, intent: ActionIntent) -> Reconciliation:
        self.asked.append(intent.idempotency_key)
        receipt = None
        if self.outcome == ReconcileOutcome.APPLIED:
            receipt = ActionReceipt(
                action_id=intent.action_id, ok=True, external_reference=self.external_reference
            )
        return Reconciliation(self.outcome, receipt)


def _intent(key: str = "submit-acme-backend", **payload) -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.SUBMIT_APPLICATION,
        target="https://example.test/acme/backend",
        summary="Submit an application to Acme for Backend Engineer",
        payload={"dedupe_key": "acme:backend", **payload},
        idempotency_key=key,
    )


def _decision(
    intent: ActionIntent, verdict: ApprovalVerdict = ApprovalVerdict.APPROVED
) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=verdict,
        decided_by="reviewer@example.test",
    )


def _rows(factory, model):
    session = factory()
    try:
        return [row.to_dict() for row in session.query(model).all()]
    finally:
        session.close()


def _execution(factory, key: str) -> dict:
    (row,) = [r for r in _rows(factory, ToolExecutionModel) if r["idempotency_key"] == key]
    return row


# --- The executor flow -------------------------------------------------------


async def test_a_first_attempt_leaves_a_complete_execution_and_audit_record(tmp_path):
    """Policy decision, claim, provider call, then receipt + audit + event."""
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider()
    intent = _intent()

    receipt = await build_tool_executor(provider, factory).execute(intent, _decision(intent))

    assert receipt.ok is True
    assert len(provider.calls) == 1

    (decision,) = _rows(factory, PolicyDecisionModel)
    assert decision["tool"] == SUBMIT_TOOL
    assert decision["decision"] == Decision.REQUIRE_APPROVAL.value
    assert decision["args_hash"] == intent.fingerprint()

    execution = _execution(factory, intent.idempotency_key)
    assert execution["status"] == ToolExecutionStatus.COMPLETED.value
    assert execution["tool_name"] == SUBMIT_TOOL
    assert execution["receipt_json"]["external_reference"] == "ext-1"
    assert execution["policy_decision_id"] == decision["id"]
    assert execution["approval_ref"] == str(intent.action_id)
    assert execution["approved_by"] == "reviewer@example.test"

    (audit,) = _rows(factory, AuditEventModel)
    assert audit["actor"] == intent.requested_by
    assert audit["action"] == SUBMIT_TOOL
    assert audit["target_ref"] == intent.target
    assert audit["policy_decision"] == Decision.REQUIRE_APPROVAL.value
    assert audit["policy_decision_id"] == decision["id"]
    assert audit["operation_id"] == execution["operation_id"]
    assert audit["approval_ref"] == str(intent.action_id)
    assert audit["result"] == AuditEventResult.SUCCESS.value
    assert audit["timestamp"]

    (event,) = _rows(factory, OutboxEventModel)
    assert event["type"] == ACTION_SUCCEEDED_EVENT
    assert event["payload_json"]["operation_id"] == execution["operation_id"]
    assert event["payload_json"]["external_reference"] == "ext-1"


async def test_the_claim_and_its_decision_are_committed_before_the_provider_is_called(tmp_path):
    """Observed from a separate session, while the side effect is still running."""
    factory = session_factory(tmp_path / "exec.db")
    observed: list[dict] = []
    intent = _intent()

    class ObservingProvider:
        async def execute(self, action_intent, decision):
            other = session_factory(tmp_path / "exec.db")
            observed.append(_execution(other, action_intent.idempotency_key))
            observed.append({"audits": len(_rows(other, AuditEventModel))})
            return ActionReceipt(action_id=action_intent.action_id, ok=True)

    await build_tool_executor(ObservingProvider(), factory).execute(intent, _decision(intent))

    assert observed[0]["status"] == ToolExecutionStatus.IN_PROGRESS.value
    assert observed[0]["policy_decision_id"] is not None
    assert observed[1] == {"audits": 0}, "the audit row was written before the outcome"


# --- Retries -----------------------------------------------------------------


async def test_retrying_a_succeeded_operation_returns_the_original_receipt(tmp_path):
    """The acceptance criterion: same key, one side effect, same receipt."""
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider()
    intent = _intent()
    first = await build_tool_executor(provider, factory).execute(intent, _decision(intent))

    # A restarted process: new executor, new sessions, and a re-proposed intent
    # with a new action id but the same idempotency key.
    retry = _intent()
    second = await build_tool_executor(
        provider, session_factory(tmp_path / "exec.db")
    ).execute(retry, _decision(retry))

    assert len(provider.calls) == 1, "the application was submitted twice"
    assert second.ok is True
    assert second.external_reference == first.external_reference
    assert second.action_id == retry.action_id
    assert len(_rows(factory, AuditEventModel)) == 1
    assert len(_rows(factory, OutboxEventModel)) == 1
    assert _execution(factory, intent.idempotency_key)["attempts"] == 1


async def test_a_key_reused_for_a_different_action_is_rejected(tmp_path):
    """A stored receipt is only replayed to the action it belongs to."""
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider()
    intent = _intent()
    executor = build_tool_executor(provider, factory)
    await executor.execute(intent, _decision(intent))

    other = _intent(company="Globex")
    with pytest.raises(IdempotencyConflict):
        await executor.execute(other, _decision(other))

    assert len(provider.calls) == 1


async def test_a_rejected_submission_is_recorded_as_a_failure_and_replayed(tmp_path):
    """The provider said no: a real outcome, audited as a failure, never re-sent."""
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider(ok=False)
    intent = _intent()
    executor = build_tool_executor(provider, factory)

    first = await executor.execute(intent, _decision(intent))
    second = await executor.execute(intent, _decision(intent))

    assert len(provider.calls) == 1
    assert first.ok is False and second.detail == "provider rejected it"
    (audit,) = _rows(factory, AuditEventModel)
    assert audit["result"] == AuditEventResult.FAILURE.value
    assert _rows(factory, OutboxEventModel) == []


async def test_a_provider_error_is_recorded_audited_and_re_raised(tmp_path):
    factory = session_factory(tmp_path / "exec.db")
    intent = _intent()

    with pytest.raises(RuntimeError, match="provider unreachable"):
        await build_tool_executor(SpyProvider(error="provider unreachable"), factory).execute(
            intent, _decision(intent)
        )

    execution = _execution(factory, intent.idempotency_key)
    assert execution["status"] == ToolExecutionStatus.FAILED.value
    assert "provider unreachable" in execution["error"]
    (audit,) = _rows(factory, AuditEventModel)
    assert audit["result"] == AuditEventResult.FAILURE.value
    assert audit["policy_decision_id"] == execution["policy_decision_id"]


# --- Unknown outcomes --------------------------------------------------------


async def _crash_mid_submission(tmp_path, intent: ActionIntent):
    """Leave a claim with no outcome behind, as a killed worker would."""
    factory = session_factory(tmp_path / "exec.db")
    crashed = SpyProvider(crash_after=True)
    with pytest.raises(_Crash):
        await build_tool_executor(crashed, factory).execute(intent, _decision(intent))
    assert len(crashed.calls) == 1
    assert _execution(factory, intent.idempotency_key)["status"] == (
        ToolExecutionStatus.IN_PROGRESS.value
    )
    return session_factory(tmp_path / "exec.db")


async def test_an_unknown_outcome_is_not_retried_without_a_reconciler(tmp_path):
    intent = _intent()
    factory = await _crash_mid_submission(tmp_path, intent)

    survivor = SpyProvider()
    receipt = await build_tool_executor(survivor, factory).execute(intent, _decision(intent))

    assert survivor.calls == [], "the resumed run submitted a second time"
    assert receipt.ok is False
    assert "did not record an outcome" in receipt.detail
    assert _execution(factory, intent.idempotency_key)["status"] == (
        ToolExecutionStatus.UNKNOWN.value
    )
    assert _rows(factory, AuditEventModel) == []


async def test_an_unknown_outcome_found_at_the_provider_is_recorded_not_re_executed(tmp_path):
    """The crashed attempt did submit. Its receipt is recovered, and audited."""
    intent = _intent()
    factory = await _crash_mid_submission(tmp_path, intent)
    original_decision = _execution(factory, intent.idempotency_key)["policy_decision_id"]

    survivor = SpyProvider()
    reconciler = ScriptedReconciler(ReconcileOutcome.APPLIED, external_reference="ext-found")
    receipt = await build_tool_executor(survivor, factory, reconciler=reconciler).execute(
        intent, _decision(intent)
    )

    assert survivor.calls == []
    assert reconciler.asked == [intent.idempotency_key]
    assert receipt.ok is True
    assert receipt.external_reference == "ext-found"

    execution = _execution(factory, intent.idempotency_key)
    assert execution["status"] == ToolExecutionStatus.COMPLETED.value
    (audit,) = _rows(factory, AuditEventModel)
    assert audit["result"] == AuditEventResult.SUCCESS.value
    # The decision cited is the one the submission was actually made under.
    assert audit["policy_decision_id"] == original_decision
    assert len(_rows(factory, OutboxEventModel)) == 1


async def test_an_unknown_outcome_the_provider_never_saw_is_executed_again(tmp_path):
    intent = _intent()
    factory = await _crash_mid_submission(tmp_path, intent)

    survivor = SpyProvider()
    receipt = await build_tool_executor(
        survivor, factory, reconciler=ScriptedReconciler(ReconcileOutcome.NOT_APPLIED)
    ).execute(intent, _decision(intent))

    assert len(survivor.calls) == 1
    assert receipt.ok is True
    execution = _execution(factory, intent.idempotency_key)
    assert execution["status"] == ToolExecutionStatus.COMPLETED.value
    assert execution["attempts"] == 2
    (audit,) = _rows(factory, AuditEventModel)
    # Re-executed under this attempt's decision, which is the one on the row.
    assert audit["policy_decision_id"] == execution["policy_decision_id"]


@pytest.mark.parametrize("raises", [False, True])
async def test_an_unknown_outcome_the_provider_cannot_confirm_stays_unknown(tmp_path, raises):
    intent = _intent()
    factory = await _crash_mid_submission(tmp_path, intent)

    class BrokenReconciler:
        async def reconcile(self, action_intent):
            raise RuntimeError("provider lookup timed out")

    reconciler = (
        BrokenReconciler() if raises else ScriptedReconciler(ReconcileOutcome.INDETERMINATE)
    )
    survivor = SpyProvider()
    receipt = await build_tool_executor(survivor, factory, reconciler=reconciler).execute(
        intent, _decision(intent)
    )

    assert survivor.calls == []
    assert receipt.ok is False
    assert _execution(factory, intent.idempotency_key)["status"] == (
        ToolExecutionStatus.UNKNOWN.value
    )


async def test_a_failed_action_is_reconciled_before_it_is_retried(tmp_path):
    """A call that raised may still have landed; the provider decides, not a retry loop."""
    factory = session_factory(tmp_path / "exec.db")
    intent = _intent()
    with pytest.raises(RuntimeError):
        await build_tool_executor(SpyProvider(error="timeout"), factory).execute(
            intent, _decision(intent)
        )

    untouched = SpyProvider()
    receipt = await build_tool_executor(untouched, factory).execute(intent, _decision(intent))
    assert untouched.calls == [] and receipt.ok is False

    retried = SpyProvider()
    receipt = await build_tool_executor(
        retried, factory, reconciler=ScriptedReconciler(ReconcileOutcome.NOT_APPLIED)
    ).execute(intent, _decision(intent))
    assert len(retried.calls) == 1 and receipt.ok is True


# --- Policy ------------------------------------------------------------------


async def test_an_unapproved_action_is_refused_before_anything_is_claimed(tmp_path):
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider()
    intent = _intent()

    with pytest.raises(ApprovalRequired):
        await build_tool_executor(provider, factory).execute(
            intent, _decision(intent, ApprovalVerdict.REJECTED)
        )

    assert provider.calls == []
    assert _rows(factory, ToolExecutionModel) == []
    # The verdict itself is still on record.
    assert len(_rows(factory, PolicyDecisionModel)) == 1


async def test_an_approval_for_a_different_payload_does_not_clear_the_action(tmp_path):
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider()
    intent = _intent()
    stale = _decision(_intent(company="Globex")).model_copy(update={"action_id": intent.action_id})

    with pytest.raises(ApprovalRequired):
        await build_tool_executor(provider, factory).execute(intent, stale)

    assert provider.calls == []


async def test_a_denied_action_is_never_executed(tmp_path):
    factory = session_factory(tmp_path / "exec.db")
    provider = SpyProvider()
    intent = _intent()
    policy = PolicyEngine(class_outcomes={PermissionClass.WRITE_EXTERNAL: Decision.DENY})
    executor = ToolExecutor(provider, policy, ExecutionLedger(factory))

    with pytest.raises(PolicyDenied):
        await executor.execute(intent, _decision(intent))

    assert provider.calls == []
    assert _rows(factory, ToolExecutionModel) == []


def test_the_executor_requires_a_provider_and_a_policy_engine(tmp_path):
    ledger = ExecutionLedger(session_factory(tmp_path / "exec.db"))
    with pytest.raises(ValueError, match="requires an inner ActionExecutor"):
        ToolExecutor(None, default_policy_engine(), ledger)
    with pytest.raises(ValueError, match="requires a PolicyEngine"):
        ToolExecutor(SpyProvider(), None, ledger)
