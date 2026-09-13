"""Tests for the tool-execution, policy-decision, and audit-event schema.

Covers the acceptance criteria from the schema migration: the three tables
are created with the documented indexes, a retried tool execution replays
its stored receipt rather than creating a duplicate row, and the audit
trail exposes no update/delete path in application code.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from personalos.domain.models import (
    AuditEventResult,
    PolicyDecisionOutcome,
    ToolExecutionStatus,
)
from personalos.persistence.models import Base, ToolExecutionModel
from personalos.persistence.repositories import (
    AuditEventRepository,
    PolicyDecisionRepository,
    ToolExecutionRepository,
    WorkflowRepository,
)


def _open(db_path):
    """A fresh engine + session over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory(), engine


# ----------------------------------------------------------------------
# Migration shape: tables and the documented indexes exist.
# ----------------------------------------------------------------------


def test_schema_creates_all_three_tables_with_documented_indexes(tmp_path):
    """tool_executions, policy_decisions, and audit_events all exist.

    tool_executions is indexed on idempotency_key (via its unique
    constraint) and workflow_id; policy_decisions and audit_events are each
    indexed on workflow_id.
    """
    _, engine = _open(tmp_path / "audit.db")
    try:
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert {"tool_executions", "policy_decisions", "audit_events"} <= table_names

        tool_execution_indexed_columns = {
            column
            for index in inspector.get_indexes("tool_executions")
            for column in index["column_names"]
        } | {
            column
            for uc in inspector.get_unique_constraints("tool_executions")
            for column in uc["column_names"]
        }
        assert "idempotency_key" in tool_execution_indexed_columns
        assert "workflow_id" in tool_execution_indexed_columns

        for table in ("policy_decisions", "audit_events"):
            indexed_columns = {
                column
                for index in inspector.get_indexes(table)
                for column in index["column_names"]
            }
            assert "workflow_id" in indexed_columns
    finally:
        engine.dispose()


# ----------------------------------------------------------------------
# tool_executions: idempotency / replay semantics.
# ----------------------------------------------------------------------


def test_retried_tool_execution_replays_stored_receipt(tmp_path):
    """A retried tool call gets back the original receipt, not a new row."""
    session, engine = _open(tmp_path / "audit.db")
    try:
        workflow = WorkflowRepository(session).create(name="job_search")
        repo = ToolExecutionRepository(session)
        key = "idem-key-12345"

        first, claimed_first = repo.claim(key, "jobs.apply", workflow_id=workflow.id)
        assert claimed_first is True
        assert first.status == ToolExecutionStatus.IN_PROGRESS.value

        receipt = {"applied": True, "application_id": "app-1"}
        completed = repo.complete(key, receipt)
        assert completed.status == ToolExecutionStatus.COMPLETED.value
        assert completed.receipt_json == receipt

        # Simulate a retry of the exact same tool call.
        second, claimed_second = repo.claim(key, "jobs.apply", workflow_id=workflow.id)
        assert claimed_second is False
        assert second.operation_id == first.operation_id
        assert second.receipt_json == receipt

        rows = (
            session.query(ToolExecutionModel)
            .filter(ToolExecutionModel.idempotency_key == key)
            .count()
        )
        assert rows == 1
    finally:
        session.close()
        engine.dispose()


def test_tool_execution_idempotency_key_is_unique(tmp_path):
    """The schema itself rejects a duplicate idempotency_key insert."""
    session, engine = _open(tmp_path / "audit.db")
    try:
        session.add(
            ToolExecutionModel(tool_name="jobs.apply", idempotency_key="dup-key")
        )
        session.commit()

        session.add(
            ToolExecutionModel(tool_name="jobs.withdraw", idempotency_key="dup-key")
        )
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.close()
        engine.dispose()


def test_failed_tool_execution_is_recorded(tmp_path):
    """A failed attempt is recorded with its error, not silently dropped."""
    session, engine = _open(tmp_path / "audit.db")
    try:
        repo = ToolExecutionRepository(session)
        key = "idem-key-fail"

        repo.claim(key, "jobs.apply")
        failed = repo.fail(key, "upstream timeout")

        assert failed.status == ToolExecutionStatus.FAILED.value
        assert failed.error == "upstream timeout"
        assert repo.get_by_key(key).status == ToolExecutionStatus.FAILED.value
    finally:
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# policy_decisions
# ----------------------------------------------------------------------


def test_policy_decision_is_queryable_by_workflow_id(tmp_path):
    """A recorded decision is findable via its workflow_id."""
    session, engine = _open(tmp_path / "audit.db")
    try:
        workflow = WorkflowRepository(session).create(name="job_search")
        repo = PolicyDecisionRepository(session)

        repo.create(
            principal="user:alice",
            tool="jobs.apply",
            args_hash="a" * 64,
            decision=PolicyDecisionOutcome.REQUIRE_APPROVAL.value,
            workflow_id=workflow.id,
            requested_scopes=["jobs.apply"],
        )

        decisions = repo.get_by_workflow_id(workflow.id)
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.principal == "user:alice"
        assert decision.decision == PolicyDecisionOutcome.REQUIRE_APPROVAL.value
        assert decision.requested_scopes == ["jobs.apply"]
    finally:
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# audit_events: append-only.
# ----------------------------------------------------------------------


def test_audit_event_is_queryable_by_workflow_id(tmp_path):
    """An appended audit event is findable via its workflow_id."""
    session, engine = _open(tmp_path / "audit.db")
    try:
        workflow = WorkflowRepository(session).create(name="job_search")
        repo = AuditEventRepository(session)

        repo.create(
            actor="user:alice",
            action="jobs.apply",
            target_ref="applications/app-1",
            result=AuditEventResult.SUCCESS.value,
            workflow_id=workflow.id,
            policy_decision=PolicyDecisionOutcome.ALLOW.value,
        )

        events = repo.get_by_workflow_id(workflow.id)
        assert len(events) == 1
        event = events[0]
        assert event.actor == "user:alice"
        assert event.result == AuditEventResult.SUCCESS.value
        assert event.policy_decision == PolicyDecisionOutcome.ALLOW.value
        assert isinstance(event.timestamp, datetime)
    finally:
        session.close()
        engine.dispose()


def test_audit_event_repository_exposes_no_update_or_delete(tmp_path):
    """The audit trail cannot be rewritten: only create and reads exist."""
    assert not hasattr(AuditEventRepository, "update")
    assert not hasattr(AuditEventRepository, "delete")
