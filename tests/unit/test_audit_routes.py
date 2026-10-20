"""Tests for the v1 audit trail endpoints.

The acceptance criteria this file exists for:

1. **A completed workflow with one approved send reads as intent -> policy
   decision REQUIRE_APPROVAL -> approval -> execution receipt, in order** --
   `test_an_approved_send_reads_as_intent_decision_approval_execution`.
2. **A seeded fake secret in the source rows comes back redacted** --
   `test_a_secret_in_the_source_rows_is_redacted_in_every_response`.

The send in (1) goes through the real write side: the board's transition
endpoint records the request-time policy decision, the approval endpoint
records the decision, and the real `ToolExecutor` runs the send inside a
worker-driven graph, so the rows the trail reads are the rows production
writes. The graph is a stand-in with real shapes at its edges, as in
`test_approval_routes`, because what is under test is the trail, not the job
search.
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
from apps.api.routes.applications import build_application_services, get_application_services
from apps.api.routes.audit import build_audit_store, get_audit_store
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    build_durable_checkpointer,
    build_tool_executor,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
)
from personalos.domain.applications import RequestedTransition, transition_action_intent
from personalos.domain.job_search import (
    ActionIntent,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRequest,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.redaction import REDACTED, default_redactor
from personalos.domain.workflow import JOB_SEARCH_THREAD_NAMESPACE
from personalos.persistence.application_lifecycle import ApplicationLifecycleStore
from personalos.persistence.models import (
    ApprovalModel,
    AuditEventModel,
    Base,
    PolicyDecisionModel,
    ToolExecutionModel,
    UserModel,
)
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository

S = ApplicationStatus
ACTOR = "maddy"
T0 = datetime(2026, 10, 1, 9, 0, 0)


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "audit.db"
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
    app.dependency_overrides[get_audit_store] = lambda: build_audit_store(factory)
    app.dependency_overrides[get_application_services] = lambda: build_application_services(factory)
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


def applied_application(factory) -> UUID:
    """An APPLIED application, so withdrawing it can ask for the recruiter to be told."""
    session = factory()
    try:
        user = UserModel(email=f"candidate-{uuid4()}@example.com")
        session.add(user)
        session.commit()
        posting = JobPostingRepository(session).create(
            source="greenhouse",
            title="Staff Engineer",
            company="Acme",
            description_hash="a" * 64,
            dedupe_key=f"acme:staff:{uuid4()}",
            url=f"https://jobs.example.com/{uuid4()}",
        )
        application_id = (
            ApplicationRepository(session).create(job_posting_id=posting.id, user_id=user.id).id
        )
    finally:
        session.close()
    lifecycle = ApplicationLifecycleStore(factory)
    for status in (S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED):
        lifecycle.transition(application_id, status, actor="setup")
    return application_id


class RecruiterMailer:
    """The provider: accepts the send and hands back a receipt."""

    def __init__(self):
        self.sent: list[ActionIntent] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.sent.append(intent)
        return ActionReceipt(
            action_id=intent.action_id, ok=True, external_reference=f"gmail-{len(self.sent)}"
        )


class _SendState(TypedDict, total=False):
    user_id: str
    requested_transitions: list[dict[str, Any]]
    pending_actions: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    receipts: list[dict[str, Any]]


class Worker:
    """A worker process whose graph proposes the send, parks for approval, then runs it."""

    def __init__(self, db_path: Path):
        engine = create_engine(f"sqlite:///{db_path}")
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.registry = build_workflow_thread_registry(self.factory)
        self.checkpointer = build_durable_checkpointer(self.factory, self.registry)
        self.leases = build_workflow_lease_store(self.factory)
        self.queue = build_workflow_command_queue(self.factory)
        self.mailer = RecruiterMailer()

        graph = StateGraph(_SendState)
        graph.add_node("propose", self._propose)
        graph.add_node("approval_checkpoint", self._approval_checkpoint)
        graph.add_node("execute_approved_actions", self._execute)
        graph.add_edge(START, "propose")
        graph.add_edge("propose", "approval_checkpoint")
        graph.add_edge("approval_checkpoint", "execute_approved_actions")
        graph.add_edge("execute_approved_actions", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _propose(state: _SendState) -> dict[str, Any]:
        intents = [
            transition_action_intent(RequestedTransition.model_validate(t))
            for t in state["requested_transitions"]
        ]
        requests = [ApprovalRequest.for_intent(i, now=datetime.utcnow()) for i in intents]
        return {
            "pending_actions": [i.model_dump(mode="json") for i in intents],
            "approval_requests": [r.model_dump(mode="json") for r in requests],
        }

    @staticmethod
    def _approval_checkpoint(state: _SendState) -> dict[str, Any]:
        decisions = interrupt(
            {
                "workflow": "job_search",
                "stage": "transition",
                "requests": state["approval_requests"],
            }
        )
        return {"approvals": decisions}

    async def _execute(self, state: _SendState, config) -> dict[str, Any]:
        thread = self.registry.resolve(config["configurable"]["thread_id"])
        executor = build_tool_executor(self.mailer, self.factory, workflow_id=thread.workflow_id)
        decisions = {
            d.action_id: d for d in map(ApprovalDecision.model_validate, state["approvals"])
        }
        receipts = []
        for intent in map(ActionIntent.model_validate, state["pending_actions"]):
            receipt = await executor.execute(intent, decisions[intent.action_id])
            receipts.append(receipt.model_dump(mode="json"))
        return {"receipts": receipts}

    def run_next(self) -> bool:
        runner = DurableWorkflowRunner(self.graph, registry=self.registry, leases=self.leases)
        return asyncio.run(process_next(self.queue, {JOB_SEARCH_THREAD_NAMESPACE: runner}))


def approved_send(db_path: Path, factory) -> tuple[UUID, UUID, Worker]:
    """Withdraw an application, telling the recruiter, and approve the send.

    Returns the application id, the workflow id, and the worker that ran it.
    """
    application_id = applied_application(factory)
    queued = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={
            "to": "withdrawn",
            "effect": "notify_recruiter",
            "recipient": "recruiter@acme.example",
            "message": "I'm withdrawing my application.",
        },
        headers={"X-Actor-Id": ACTOR},
    )
    assert queued.status_code == 202, queued.text
    workflow_id = UUID(queued.json()["workflow_id"])

    worker = Worker(db_path)
    assert worker.run_next()  # proposes the send and parks at the approval

    (pending,) = api_process(db_path).get("/v1/approvals").json()["approvals"]
    decided = api_process(db_path).post(
        f"/v1/approvals/{pending['id']}/decision",
        json={"approved": True, "action_hash": pending["action_hash"]},
        headers={"X-Actor-Id": ACTOR},
    )
    assert decided.status_code == 200, decided.text
    assert decided.json()["resumed"] is True

    assert worker.run_next()  # resumes and sends
    assert len(worker.mailer.sent) == 1
    return application_id, workflow_id, worker


def stages(entries: list[dict]) -> list[str]:
    return [entry["stage"] for entry in entries]


# --- The workflow trail -----------------------------------------------------------


def test_an_approved_send_reads_as_intent_decision_approval_execution(db_path, session_factory):
    _, workflow_id, _ = approved_send(db_path, session_factory)

    response = api_process(db_path).get(f"/v1/workflows/{workflow_id}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"] == str(workflow_id)
    entries = body["entries"]

    # The four steps, in that order (the executor re-checks policy before the
    # call, which is a second decision between the approval and the execution).
    intent = next(e for e in entries if e["stage"] == "intent")
    decision = next(e for e in entries if e["stage"] == "policy_decision")
    approval = next(e for e in entries if e["stage"] == "approval")
    execution = next(e for e in entries if e["stage"] == "execution")
    assert (
        entries.index(intent)
        < entries.index(decision)
        < entries.index(approval)
        < entries.index(execution)
    )
    assert stages(entries) == [
        "intent",
        "policy_decision",
        "approval",
        "policy_decision",
        "execution",
    ]
    timestamps = [datetime.fromisoformat(e["at"]) for e in entries]
    assert timestamps == sorted(timestamps)

    assert intent["actor"] == ACTOR
    assert intent["action"] == "google.gmail_send_message"
    assert decision["policy_decision"] == "require_approval"
    assert approval["action"] == "approval.APPROVED"
    assert approval["actor"] == ACTOR
    assert approval["result"] == "success"
    assert execution["action"] == "google.gmail_send_message"
    assert execution["result"] == "success"
    assert execution["policy_decision"] == "require_approval"

    # The receipt is on the execution.
    receipt = execution["links"]["tool_execution"]["receipt"]
    assert receipt["ok"] is True
    assert receipt["external_reference"] == "gmail-1"
    assert execution["links"]["tool_execution"]["status"] == "completed"

    # Every step is one action: same hash, and each links the whole chain.
    (action_hash,) = {e["action_hash"] for e in entries}
    approval_id = approval["links"]["approval"]["id"]
    operation_id = execution["links"]["tool_execution"]["operation_id"]
    for entry in entries:
        links = entry["links"]
        assert links["policy_decision"]["args_hash"] == action_hash
        assert links["approval"]["id"] == approval_id
        assert links["approval"]["status"] == "approved"
        assert links["approval"]["decided_by"] == ACTOR
        assert links["tool_execution"]["operation_id"] == operation_id
        assert links["tool_execution"]["approval_ref"] == approval_id
    # The execution cites the decision it actually ran under, the second one.
    decisions = [e for e in entries if e["stage"] == "policy_decision"]
    assert (
        execution["links"]["policy_decision"]["id"]
        == decisions[-1]["links"]["policy_decision"]["id"]
        == execution["links"]["tool_execution"]["policy_decision_id"]
    )
    assert (
        intent["links"]["policy_decision"]["id"] == decisions[0]["links"]["policy_decision"]["id"]
    )
    # The intent says what the action was aimed at.
    assert intent["target_ref"] == approval["target_ref"]


def test_an_unknown_workflow_trail_is_a_404(db_path):
    response = api_process(db_path).get(f"/v1/workflows/{uuid4()}/audit")

    assert response.status_code == 404


def test_a_workflow_with_nothing_recorded_has_an_empty_trail(db_path, session_factory):
    workflow_id = build_workflow_thread_registry(session_factory).create_workflow(
        JOB_SEARCH_WORKFLOW
    )

    response = api_process(db_path).get(f"/v1/workflows/{workflow_id}/audit")

    assert response.status_code == 200
    assert response.json()["entries"] == []


# --- Search -------------------------------------------------------------------------


def test_search_is_newest_first_and_links_each_event(db_path, session_factory):
    _, workflow_id, _ = approved_send(db_path, session_factory)

    body = api_process(db_path).get("/v1/audit", params={"workflow_id": str(workflow_id)}).json()

    assert body["total"] == 2
    assert [e["stage"] for e in body["entries"]] == ["execution", "approval"]
    timestamps = [datetime.fromisoformat(e["at"]) for e in body["entries"]]
    assert timestamps == sorted(timestamps, reverse=True)
    execution, approval = body["entries"]
    assert execution["links"]["tool_execution"]["receipt"]["external_reference"] == "gmail-1"
    assert execution["links"]["approval"]["id"] == approval["links"]["approval"]["id"]
    # The approval is recorded before the call, yet still reaches it.
    assert (
        approval["links"]["tool_execution"]["operation_id"]
        == execution["links"]["tool_execution"]["operation_id"]
    )
    assert approval["links"]["policy_decision"]["decision"] == "require_approval"


def test_search_filters_by_application_actor_action_and_time(db_path, session_factory):
    application_id, workflow_id, _ = approved_send(db_path, session_factory)
    other = seed_event(session_factory, actor="someone-else", action="files.overwrite_document")
    client = api_process(db_path)

    def ids(**params) -> list[str]:
        return [
            e["audit_event_id"] for e in client.get("/v1/audit", params=params).json()["entries"]
        ]

    every = ids()
    assert len(every) == 3 and str(other) in every

    by_application = client.get("/v1/audit", params={"application_id": str(application_id)})
    assert {e["workflow_id"] for e in by_application.json()["entries"]} == {str(workflow_id)}
    assert by_application.json()["total"] == 2
    assert ids(application_id=str(uuid4())) == []

    assert ids(actor="someone-else") == [str(other)]
    (approved,) = client.get("/v1/audit", params={"action": "approval.APPROVED"}).json()["entries"]
    assert approved["stage"] == "approval"

    assert ids(until=(T0 - timedelta(days=1)).isoformat()) == []
    assert ids(since=T0.isoformat(), until=T0.isoformat()) == [str(other)]
    assert len(ids(since=(T0 + timedelta(seconds=1)).isoformat())) == 2


def test_search_paginates(db_path, session_factory):
    seeded = [seed_event(session_factory, at=T0 + timedelta(minutes=minute)) for minute in range(5)]
    client = api_process(db_path)

    first = client.get("/v1/audit", params={"limit": 2}).json()
    second = client.get("/v1/audit", params={"limit": 2, "offset": first["next_offset"]}).json()
    last = client.get("/v1/audit", params={"limit": 2, "offset": second["next_offset"]}).json()

    assert first["total"] == 5
    assert first["next_offset"] == 2 and second["next_offset"] == 4
    assert last["next_offset"] is None
    pages = [e["audit_event_id"] for page in (first, second, last) for e in page["entries"]]
    assert pages == [str(i) for i in reversed(seeded)]


def test_an_inverted_time_range_and_an_oversized_page_are_refused(db_path):
    client = api_process(db_path)

    inverted = client.get(
        "/v1/audit",
        params={"since": T0.isoformat(), "until": (T0 - timedelta(hours=1)).isoformat()},
    )
    assert inverted.status_code == 422
    assert client.get("/v1/audit", params={"limit": 1000}).status_code == 422


# --- Read-only, redacted ------------------------------------------------------------


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_no_endpoint_modifies_the_trail(db_path, session_factory, method):
    event_id = seed_event(session_factory)
    workflow_id = build_workflow_thread_registry(session_factory).create_workflow(
        JOB_SEARCH_WORKFLOW
    )
    client = api_process(db_path)

    for path in ("/v1/audit", f"/v1/audit/{event_id}", f"/v1/workflows/{workflow_id}/audit"):
        assert getattr(client, method)(path).status_code in (404, 405)

    session = session_factory()
    try:
        (row,) = session.query(AuditEventModel).all()
        assert row.id == event_id and row.actor == "seed"
    finally:
        session.close()


FAKE_SECRET = "seeded-fake-secret-0123456789"
BEARER = "Bearer ya29.a0AfH6SMBfakefakefakefakefake"


def test_a_secret_in_the_source_rows_is_redacted_in_every_response(db_path, session_factory):
    """Rows written around the repositories, as a legacy or corrupted row would be."""
    workflow_id = build_workflow_thread_registry(session_factory).create_workflow(
        JOB_SEARCH_WORKFLOW
    )
    approval_id = uuid4()
    session = session_factory()
    try:
        decision = PolicyDecisionModel(
            principal=f"api_key={FAKE_SECRET}",
            workflow_id=workflow_id,
            tool="google.gmail_send_message",
            args_hash="f" * 64,
            decision="require_approval",
            requested_scopes=["gmail.send"],
            decided_at=T0,
        )
        session.add(decision)
        session.flush()
        session.add(
            ApprovalModel(
                id=approval_id,
                action_hash="f" * 64,
                status="approved",
                note=f"client_secret={FAKE_SECRET}",
                decided_at=T0 + timedelta(minutes=1),
                created_at=T0,
                updated_at=T0,
            )
        )
        execution = ToolExecutionModel(
            workflow_id=workflow_id,
            tool_name="google.gmail_send_message",
            idempotency_key="send-1",
            status="completed",
            request_fingerprint="f" * 64,
            policy_decision_id=decision.id,
            approval_ref=str(approval_id),
            approved_by=ACTOR,
            receipt_json={
                "ok": True,
                "external_reference": "gmail-1",
                "access_token": FAKE_SECRET,
                "detail": f"sent with Authorization: {BEARER}",
            },
            error=f"retried after refresh_token={FAKE_SECRET} was rotated",
            created_at=T0 + timedelta(minutes=2),
            updated_at=T0 + timedelta(minutes=2),
        )
        session.add(execution)
        session.flush()
        session.add_all(
            [
                AuditEventModel(
                    actor=ACTOR,
                    workflow_id=workflow_id,
                    action="approval.APPROVED",
                    target_ref=f"recruiter@acme.example?token={FAKE_SECRET}",
                    approval_ref=str(approval_id),
                    result="success",
                    timestamp=T0 + timedelta(minutes=1),
                ),
                AuditEventModel(
                    actor=ACTOR,
                    workflow_id=workflow_id,
                    action="google.gmail_send_message",
                    target_ref=f"recruiter@acme.example {BEARER}",
                    policy_decision="require_approval",
                    policy_decision_id=decision.id,
                    operation_id=execution.operation_id,
                    approval_ref=str(approval_id),
                    result="success",
                    timestamp=T0 + timedelta(minutes=2),
                ),
            ]
        )
        session.commit()
    finally:
        session.close()
    client = api_process(db_path)

    trail = client.get(f"/v1/workflows/{workflow_id}/audit")
    search = client.get("/v1/audit")

    for response in (trail, search):
        assert response.status_code == 200
        assert FAKE_SECRET not in response.text
        assert "ya29." not in response.text
        assert REDACTED in response.text
    execution_entry = next(e for e in trail.json()["entries"] if e["stage"] == "execution")
    receipt = execution_entry["links"]["tool_execution"]["receipt"]
    assert receipt["access_token"] == REDACTED
    assert receipt["external_reference"] == "gmail-1"  # the rest is left alone
    assert stages(trail.json()["entries"]) == [
        "intent",
        "policy_decision",
        "approval",
        "execution",
    ]


def test_a_secret_learned_after_the_row_was_written_is_redacted_on_read(db_path, session_factory):
    """The redactor learns secrets at runtime; a row written before it knew one
    is still scrubbed on the way out."""
    known = "plain-looking-value-the-broker-minted"
    seed_event(session_factory, actor=ACTOR, action=f"send with {known}")
    default_redactor().register_secret(known)
    try:
        response = api_process(db_path).get("/v1/audit")
    finally:
        default_redactor().forget_secrets()

    assert known not in response.text
    assert response.json()["entries"][0]["action"] == f"send with {REDACTED}"


# --- Helpers ----------------------------------------------------------------------


def seed_event(
    factory, *, actor: str = "seed", action: str = "files.overwrite_document", at: datetime = T0
) -> UUID:
    """An audit row with no chain, written directly."""
    session = factory()
    try:
        event = AuditEventModel(
            actor=actor, action=action, target_ref="notes.md", result="success", timestamp=at
        )
        session.add(event)
        session.commit()
        return event.id
    finally:
        session.close()
