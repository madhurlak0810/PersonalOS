"""Graph scenarios for idempotent, audited mutations.

The two acceptance criteria this file exists for, run through the real Job
Search graph on a database-backed checkpointer with a `ToolExecutor` bound as
its `action_executor`:

1. **Retrying an operation that already succeeded returns the original receipt
   and causes no second side effect.** Two separate workflows reach the same
   submission -- the same idempotency key -- and the company receives one
   application.
2. **Every successful mutating action has an `audit_events` row referencing a
   `policy_decisions` row.** Checked against every completed execution the run
   left behind, not against a hand-picked one.

The "external write" is a line in a shared log, as in `test_durable_resume.py`,
so the count of submissions is taken from what the provider was actually asked
to do rather than from what the executor says it did.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import build_tool_executor
from personalos.domain.models import (
    ApplicationStatus,
    AuditEventResult,
    PolicyDecisionOutcome,
    ToolExecutionStatus,
)
from personalos.domain.workflow import job_search_thread_id
from personalos.executor.tool_executor import ACTION_SUCCEEDED_EVENT
from personalos.persistence.checkpointer import WorkflowThreadRegistry
from personalos.persistence.leases import WorkflowLeaseStore
from personalos.persistence.models import (
    AuditEventModel,
    OutboxEventModel,
    PolicyDecisionModel,
    ToolExecutionModel,
)
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes


def _run_job_search(db_path: Path, log: durable.EventLog, search_key: str):
    """Run one job search workflow to completion on its own engine and thread.

    Returns `(final_state, thread, session_factory)`. Each call is a separate
    worker: a new engine, registry and graph over the same database file.
    """
    factory = durable.session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)
    thread = registry.register(
        thread_id=job_search_thread_id(fakes.USER_ID, search_key),
        workflow_name=durable.WORKFLOW_NAME,
        user_id=fakes.USER_ID,
    )
    graph, _ports = durable.build_graph(
        factory,
        log,
        registry=registry,
        workflow_id=thread.workflow_id,
        wrap_executor=lambda provider: build_tool_executor(
            provider, factory, workflow_id=thread.workflow_id
        ),
    )
    runner = DurableWorkflowRunner(
        graph, registry=registry, leases=WorkflowLeaseStore(factory), owner=f"worker:{search_key}"
    )
    final = asyncio.run(runner.start(thread, durable.initial_state()))
    return final, thread, factory


def _rows(factory, model) -> list[dict]:
    session = factory()
    try:
        return [row.to_dict() for row in session.query(model).all()]
    finally:
        session.close()


def test_every_successful_mutating_action_has_an_audit_row_citing_a_policy_decision(tmp_path):
    """Each completed execution is audited against the verdict that cleared it."""
    log = durable.EventLog(tmp_path / "events.jsonl")
    started = datetime.utcnow()

    final, thread, factory = _run_job_search(tmp_path / "audit.db", log, "audit")

    assert final["application"]["status"] == ApplicationStatus.APPLIED.value
    succeeded = [receipt for receipt in final["action_receipts"] if receipt["ok"]]
    assert succeeded, "the scenario performed no mutating action; the test would be vacuous"

    executions = [
        row
        for row in _rows(factory, ToolExecutionModel)
        if row["status"] == ToolExecutionStatus.COMPLETED.value
    ]
    assert len(executions) == len(succeeded) == log.count(durable.EVENT_EXTERNAL_SUBMISSION)

    decisions = {row["id"]: row for row in _rows(factory, PolicyDecisionModel)}
    audits = _rows(factory, AuditEventModel)
    events = _rows(factory, OutboxEventModel)

    for execution in executions:
        (audit,) = [row for row in audits if row["operation_id"] == execution["operation_id"]]

        # The audit row references a real policy_decisions row, and it is the
        # same one the execution itself was recorded under.
        decision = decisions[audit["policy_decision_id"]]
        assert audit["policy_decision_id"] == execution["policy_decision_id"]
        assert decision["tool"] == execution["tool_name"] == audit["action"]
        assert decision["decision"] == PolicyDecisionOutcome.REQUIRE_APPROVAL.value
        assert decision["workflow_id"] == str(thread.workflow_id)

        # Every field the audit trail promises.
        assert audit["actor"].startswith("graph:job_search")
        assert audit["workflow_id"] == str(thread.workflow_id)
        assert audit["target_ref"]
        assert audit["policy_decision"] == decision["decision"]
        assert audit["result"] == AuditEventResult.SUCCESS.value
        assert datetime.fromisoformat(audit["timestamp"]) >= started

        # A decision that needed approval names the approval it was redeemed with.
        assert audit["approval_ref"] == execution["approval_ref"]
        assert execution["approval_ref"] and execution["approved_by"]

        # And the domain event went out with the same outcome.
        (event,) = [
            row
            for row in events
            if row["type"] == ACTION_SUCCEEDED_EVENT
            and row["payload_json"]["operation_id"] == execution["operation_id"]
        ]
        assert event["payload_json"]["idempotency_key"] == execution["idempotency_key"]


def test_retrying_a_succeeded_submission_returns_its_receipt_without_submitting_again(tmp_path):
    """Two workflows reach the same submission; the company gets one application."""
    db_path = tmp_path / "retry.db"
    log = durable.EventLog(tmp_path / "events.jsonl")

    first, _thread, factory = _run_job_search(db_path, log, "first")
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1

    # A different workflow, on a different thread, proposing the same
    # submission: same idempotency key, already succeeded.
    second, _other_thread, _factory = _run_job_search(db_path, log, "second")

    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1, "the application was submitted twice"
    assert second["application"]["status"] == ApplicationStatus.APPLIED.value
    assert (
        second["application"]["external_reference"]
        == first["application"]["external_reference"]
    )
    (first_receipt,) = first["action_receipts"]
    (second_receipt,) = second["action_receipts"]
    assert second_receipt["ok"] is True
    assert second_receipt["external_reference"] == first_receipt["external_reference"]

    # One execution, one audit entry, one event -- the retry added none of them.
    (execution,) = _rows(factory, ToolExecutionModel)
    assert execution["status"] == ToolExecutionStatus.COMPLETED.value
    assert execution["attempts"] == 1
    assert len(_rows(factory, AuditEventModel)) == 1
    assert len(_rows(factory, OutboxEventModel)) == 1
    # The retry was still put to policy, and that verdict is on record too.
    assert len(_rows(factory, PolicyDecisionModel)) == 2
