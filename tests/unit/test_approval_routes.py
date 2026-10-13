"""Tests for the v1 approval queue endpoints.

The acceptance criteria this file exists for:

1. **A stale or mismatched `action_hash` is refused** with a 409, and records no
   decision -- `test_a_mismatched_action_hash_is_refused`.
2. **An expired approval cannot be approved** --
   `test_approving_an_expired_approval_is_refused`.
3. **Double-submitting a decision does not resume the workflow twice** --
   `test_double_submitting_a_decision_resumes_the_workflow_once`.

The API and worker halves are real, as in `test_workflow_routes`: an app built by
`create_app()` over a database file, and `process_next` running a job-search
thread on the durable checkpointer. The graph is a stand-in with the real
shapes at its edges -- it interrupts with real `ApprovalRequest`s and spends the
answer through the real `authorize_execution` -- because what is under test is
the queue around the interrupt, not the job search.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    build_durable_checkpointer,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.domain.job_search import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    authorize_execution,
)
from personalos.domain.workflow import (
    JOB_SEARCH_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
)
from personalos.persistence.models import ApprovalModel, AuditEventModel, Base, OutboxEventModel
from tests.fixtures import job_search_fakes as fakes

USER_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000aa")


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "approvals.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def session_factory(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    finally:
        engine.dispose()


def api_process(db_path: Path) -> TestClient:
    """A fresh API instance over the database: new app, new engine, new services."""
    engine = create_engine(f"sqlite:///{db_path}")
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


class _JobState(TypedDict, total=False):
    pending_actions: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    executed: list[str]
    refusals: list[str]
    #: How many times the node after the approval ran. A workflow resumed twice
    #: would run it twice.
    executions: int


def _approval_checkpoint(state: _JobState) -> dict[str, Any]:
    decisions = interrupt(
        {"workflow": "job_search", "stage": "submission", "requests": state["approval_requests"]}
    )
    return {"approvals": decisions}


def _execute(state: _JobState) -> dict[str, Any]:
    requests = {
        r.action_id: r for r in map(ApprovalRequest.model_validate, state["approval_requests"])
    }
    decisions = {
        d.action_id: d for d in map(ApprovalDecision.model_validate, state.get("approvals") or [])
    }
    executed, refusals = [], []
    for intent in map(ActionIntent.model_validate, state["pending_actions"]):
        refusal = authorize_execution(
            intent=intent,
            request=requests.get(intent.action_id),
            decision=decisions.get(intent.action_id),
            now=datetime.utcnow(),
        )
        if refusal is None:
            executed.append(str(intent.action_id))
        else:
            refusals.append(refusal.reason.value)
    return {
        "executed": executed,
        "refusals": refusals,
        "executions": state.get("executions", 0) + 1,
    }


class Worker:
    """A worker process: its own engine, registry, checkpointer and graph."""

    def __init__(self, db_path: Path):
        engine = create_engine(f"sqlite:///{db_path}")
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.registry = build_workflow_thread_registry(self.factory)
        self.checkpointer = build_durable_checkpointer(self.factory, self.registry)
        self.leases = build_workflow_lease_store(self.factory)
        self.queue = build_workflow_command_queue(self.factory)

        job = StateGraph(_JobState)
        job.add_node("approval_checkpoint", _approval_checkpoint)
        job.add_node("execute_approved_actions", _execute)
        job.add_edge(START, "approval_checkpoint")
        job.add_edge("approval_checkpoint", "execute_approved_actions")
        job.add_edge("execute_approved_actions", END)
        self.graph = job.compile(checkpointer=self.checkpointer)

    def run_next(self) -> bool:
        runner = DurableWorkflowRunner(self.graph, registry=self.registry, leases=self.leases)
        return asyncio.run(process_next(self.queue, {JOB_SEARCH_THREAD_NAMESPACE: runner}))

    def final_state(self, thread_id: str) -> dict[str, Any]:
        state = asyncio.run(self.graph.aget_state({"configurable": {"thread_id": thread_id}}))
        return state.values


def park(
    db_path: Path, *companies: str, requested_at: datetime | None = None
) -> tuple[Worker, str, str, list[ApprovalRequest]]:
    """Run a job-search thread to its approval, with one request per company.

    Returns the worker, the workflow id, the thread id and the requests the
    thread is parked on.
    """
    worker = Worker(db_path)
    intents = [fakes.submit_intent(fakes.posting(company=company)) for company in companies]
    requests = [
        ApprovalRequest.for_intent(intent, now=requested_at or datetime.utcnow())
        for intent in intents
    ]
    workflow_id = worker.registry.create_workflow(JOB_SEARCH_WORKFLOW)
    thread = register_job_search_thread(
        user_id=USER_ID,
        registry=worker.registry,
        workflow_id=workflow_id,
        search_key=str(uuid4()),
    )
    worker.queue.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.START,
            workflow_id=workflow_id,
            thread_id=thread.thread_id,
            actor_id="test",
            correlation_id=uuid4(),
            graph_input={
                "pending_actions": [i.model_dump(mode="json") for i in intents],
                "approval_requests": [r.model_dump(mode="json") for r in requests],
            },
        )
    )
    assert worker.run_next()
    return worker, str(workflow_id), thread.thread_id, requests


def decide(client: TestClient, request: ApprovalRequest, approved: bool, **overrides: Any):
    body = {"approved": approved, "action_hash": request.action_hash, **overrides}
    return client.post(
        f"/v1/approvals/{request.request_id}/decision",
        json=body,
        headers={"X-Actor-Id": "user:reviewer"},
    )


def resume_commands(session_factory) -> int:
    session = session_factory()
    try:
        return (
            session.query(OutboxEventModel)
            .filter(OutboxEventModel.dedupe_key.like("workflow:%:resume:%"))
            .count()
        )
    finally:
        session.close()


def rows(session_factory, model) -> list[Any]:
    session = session_factory()
    try:
        return session.query(model).all()
    finally:
        session.close()


# --- GET /v1/approvals ----------------------------------------------------------


def test_list_is_empty_when_nothing_is_waiting(db_path):
    assert api_process(db_path).get("/v1/approvals").json() == {"approvals": []}


def test_list_reports_what_a_reviewer_needs_to_decide(db_path):
    _, workflow_id, thread_id, [request] = park(db_path, "Acme")

    response = api_process(db_path).get("/v1/approvals")

    assert response.status_code == 200, response.text
    [item] = response.json()["approvals"]
    assert item["id"] == str(request.request_id)
    assert item["workflow_id"] == workflow_id
    assert item["thread_id"] == thread_id
    assert item["status"] == "pending"
    assert item["action_type"] == "submit_application"
    assert item["target"] == request.target
    assert item["summary"] == request.summary
    assert item["risk_level"] == request.risk.value
    assert item["requested_scopes"] == list(request.requested_scopes)
    assert item["action_hash"] == request.action_hash
    assert datetime.fromisoformat(item["expires_at"]) == request.expires_at


def test_list_spans_workflows_and_drops_what_has_been_decided(db_path):
    _, _, _, [first] = park(db_path, "Acme")
    _, _, _, [second] = park(db_path, "Globex")
    client = api_process(db_path)

    listed = {item["id"] for item in client.get("/v1/approvals").json()["approvals"]}
    assert listed == {str(first.request_id), str(second.request_id)}

    assert decide(client, first, True).status_code == 200
    [remaining] = client.get("/v1/approvals").json()["approvals"]
    assert remaining["id"] == str(second.request_id)


def test_an_expired_approval_is_listed_as_expired(db_path):
    park(db_path, "Acme", requested_at=datetime.utcnow() - timedelta(days=365))

    [item] = api_process(db_path).get("/v1/approvals").json()["approvals"]

    assert item["status"] == "expired"


# --- POST /v1/approvals/{id}/decision -------------------------------------------


def test_approving_resumes_the_workflow_and_executes_the_action(db_path, session_factory):
    worker, workflow_id, thread_id, [request] = park(db_path, "Acme")
    client = api_process(db_path)

    response = decide(client, request, True)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(request.request_id)
    assert body["workflow_id"] == workflow_id
    assert body["status"] == "approved"
    assert body["outcome"] == "APPROVED"
    assert body["decided_by"] == "user:reviewer"
    assert body["resumed"] is True
    assert client.get(f"/v1/workflows/{workflow_id}").json()["status"] == "queued"

    assert worker.run_next()
    final = worker.final_state(thread_id)
    assert final["executed"] == [str(request.action_id)]
    assert final["approvals"][0]["decided_by"] == "user:reviewer"

    [approval] = rows(session_factory, ApprovalModel)
    assert approval.id == request.request_id
    assert approval.status == "approved"
    assert approval.action_hash == request.action_hash
    assert approval.decided_at is not None
    [audit] = rows(session_factory, AuditEventModel)
    assert audit.action == "approval.APPROVED"
    assert audit.actor == "user:reviewer"
    assert audit.approval_ref == str(request.request_id)
    assert str(audit.workflow_id) == workflow_id
    assert audit.result == "success"


def test_rejecting_records_denied_by_user_and_takes_the_declined_branch(db_path, session_factory):
    worker, _, thread_id, [request] = park(db_path, "Acme")

    response = decide(api_process(db_path), request, False)

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "denied"
    assert response.json()["outcome"] == "DENIED_BY_USER"
    assert response.json()["resumed"] is True

    assert worker.run_next()
    final = worker.final_state(thread_id)
    assert final["executed"] == []
    assert final["refusals"] == ["denied_by_user"]

    [approval] = rows(session_factory, ApprovalModel)
    assert approval.status == "denied"
    assert approval.note == "DENIED_BY_USER"
    [audit] = rows(session_factory, AuditEventModel)
    assert audit.action == "approval.DENIED_BY_USER"


def test_a_mismatched_action_hash_is_refused(db_path, session_factory):
    """Acceptance: a stale hash is a 409 and decides nothing."""
    _, workflow_id, _, [request] = park(db_path, "Acme")
    client = api_process(db_path)

    response = decide(client, request, True, action_hash="0" * 64)

    assert response.status_code == 409
    assert response.json()["error_code"] == "conflict"
    assert "action_hash" in response.json()["message"]
    assert rows(session_factory, ApprovalModel) == []
    assert resume_commands(session_factory) == 0
    assert client.get(f"/v1/workflows/{workflow_id}").json()["status"] == "waiting"
    [audit] = rows(session_factory, AuditEventModel)
    assert audit.action == "approval.refused.hash_mismatch"
    assert audit.result == "failure"

    # The approval is untouched: the right hash still decides it.
    assert decide(client, request, True).status_code == 200


def test_approving_an_expired_approval_is_refused(db_path, session_factory):
    """Acceptance: past its expiry an approval can be denied, never approved."""
    worker, workflow_id, thread_id, [request] = park(
        db_path, "Acme", requested_at=datetime.utcnow() - timedelta(days=365)
    )
    client = api_process(db_path)

    response = decide(client, request, True)

    assert response.status_code == 409
    assert response.json()["error_code"] == "conflict"
    assert "expired" in response.json()["message"]
    assert rows(session_factory, ApprovalModel) == []
    assert resume_commands(session_factory) == 0
    assert client.get(f"/v1/workflows/{workflow_id}").json()["status"] == "waiting"

    # Denying it is how a reviewer clears it, and the run moves on.
    assert decide(client, request, False).status_code == 200
    assert worker.run_next()
    assert worker.final_state(thread_id)["executed"] == []


def test_double_submitting_a_decision_resumes_the_workflow_once(db_path, session_factory):
    """Acceptance: the repeat gets the original outcome and queues nothing."""
    worker, _, thread_id, [request] = park(db_path, "Acme")

    first = decide(api_process(db_path), request, True)
    second = decide(api_process(db_path), request, True)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert resume_commands(session_factory) == 1
    assert len(rows(session_factory, ApprovalModel)) == 1
    assert len(rows(session_factory, AuditEventModel)) == 1

    assert worker.run_next()
    assert worker.run_next() is False
    assert worker.final_state(thread_id)["executions"] == 1

    # Still idempotent once the run has moved past the approval.
    third = decide(api_process(db_path), request, True)
    assert third.status_code == 200
    assert third.json() == first.json()
    assert resume_commands(session_factory) == 1
    assert worker.run_next() is False


def test_a_recorded_decision_cannot_be_reversed(db_path, session_factory):
    _, _, _, [request] = park(db_path, "Acme")
    client = api_process(db_path)
    assert decide(client, request, False).status_code == 200

    response = decide(client, request, True)

    assert response.status_code == 409
    assert "already denied" in response.json()["message"]
    [approval] = rows(session_factory, ApprovalModel)
    assert approval.status == "denied"


def test_a_repeat_with_the_wrong_hash_is_still_refused(db_path):
    _, _, _, [request] = park(db_path, "Acme")
    client = api_process(db_path)
    assert decide(client, request, True).status_code == 200

    assert decide(client, request, True, action_hash="0" * 64).status_code == 409


def test_the_workflow_resumes_only_once_every_request_is_answered(db_path, session_factory):
    worker, workflow_id, thread_id, [acme, globex] = park(db_path, "Acme", "Globex")
    client = api_process(db_path)

    first = decide(client, acme, True)

    assert first.status_code == 200
    assert first.json()["resumed"] is False
    assert resume_commands(session_factory) == 0
    assert client.get(f"/v1/workflows/{workflow_id}").json()["status"] == "waiting"
    [left] = client.get("/v1/approvals").json()["approvals"]
    assert left["id"] == str(globex.request_id)

    second = decide(client, globex, False)

    assert second.json()["resumed"] is True
    assert resume_commands(session_factory) == 1
    assert worker.run_next()
    final = worker.final_state(thread_id)
    assert final["executed"] == [str(acme.action_id)]
    assert final["refusals"] == ["denied_by_user"]


def test_an_unknown_approval_is_a_404(db_path):
    response = api_process(db_path).post(
        f"/v1/approvals/{uuid4()}/decision", json={"approved": True, "action_hash": "f" * 64}
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"


def test_a_decision_must_carry_the_action_hash(db_path):
    _, _, _, [request] = park(db_path, "Acme")
    response = api_process(db_path).post(
        f"/v1/approvals/{request.request_id}/decision", json={"approved": True}
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation"


def test_openapi_lists_the_v1_approval_routes():
    paths = set(create_app().openapi()["paths"])
    assert {"/v1/approvals", "/v1/approvals/{approval_id}/decision"} <= paths
