"""One workflow, one trace: from the request that starts it to the receipt of what it did.

The acceptance criteria this file exists for:

1. **A workflow that goes API -> interrupt -> approval -> resume -> executor
   produces a single connected trace** --
   `test_api_interrupt_approval_resume_and_execution_are_one_connected_trace`.
   Two API requests and two worker runs, each building its own app, engine and
   graphs, so nothing connects their spans except what crossed the outbox.
2. **A span attribute dump from a workflow containing a seeded fake secret
   contains no secret value** --
   `test_a_seeded_secret_appears_in_no_exported_span`.

The pieces are the ones production runs: `create_app()` serving `POST /v1/chat`
and `POST /v1/workflows/{id}/resume`, `process_next` claiming the commands, a
real `SupervisorGraph` handing off to a job-search thread, and a real
`ToolExecutor` (policy engine, ledger, retries) executing the approved action.
The job-search graph is a stand-in with the real interrupt shape, as in
`tests/unit/test_workflow_routes.py`, because what is under test is the trace
around the run, not the job search.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    build_durable_checkpointer,
    build_job_search_subgraph_runner,
    build_policy_engine,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.domain.errors import RetryableFailure
from personalos.domain.job_search import ActionIntent, ActionReceipt, ApprovalDecision
from personalos.domain.redaction import REDACTED, default_redactor
from personalos.domain.retry_policy import DRAFT
from personalos.domain.workflow import JOB_SEARCH_THREAD_NAMESPACE, SUPERVISOR_THREAD_NAMESPACE
from personalos.executor.tool_executor import ToolExecutor
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.routing import KeywordIntentClassifier
from personalos.observability.tracing import Attr, traced_node, workflow_trace_id
from personalos.observability.tracing_setup import configure_tracing, reset_tracing
from personalos.persistence.execution_ledger import ExecutionLedger
from personalos.persistence.models import Base
from tests.fixtures import job_search_fakes as fakes

USER_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000aa")

#: Matches the token-shaped redaction rule.
PATTERN_SECRET = "sk-" + "f4keS3cretValue0123456789abcdefXYZ"
#: Matches no pattern: only the exact-value rule for registered secrets catches it.
REGISTERED_SECRET = "fake-refresh-token-q7Zp2LmN9xVb4WcR"

INTENT: ActionIntent = fakes.submit_intent()


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """Tracing on, exporting (through the redacting exporter) to memory."""
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, service_name="test", install_global=False)
    try:
        yield exporter
    finally:
        reset_tracing()


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "tracing.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def registered_secret():
    redactor = default_redactor()
    redactor.register_secret(REGISTERED_SECRET)
    try:
        yield REGISTERED_SECRET
    finally:
        redactor.forget_secrets()


def api_process(db_path: Path) -> TestClient:
    """A fresh API instance over the database: new app, new engine, new services."""
    engine = create_engine(f"sqlite:///{db_path}")
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


class FlakyProvider:
    """Fails its first submission with a leaky transient error, then succeeds."""

    def __init__(self):
        self.calls = 0

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.calls += 1
        if self.calls == 1:
            raise RetryableFailure(
                "board returned 503 for POST /applications "
                f"(Authorization: Bearer {PATTERN_SECRET}; refresh_token={REGISTERED_SECRET})"
            )
        return ActionReceipt(action_id=intent.action_id, ok=True, external_reference="app-42")


class _JobState(TypedDict, total=False):
    user_id: str
    query: str
    prepare_application: bool
    approvals: list[dict[str, Any]]
    receipts: list[dict[str, Any]]


def _approval_checkpoint(state: _JobState) -> dict[str, Any]:
    decisions = interrupt(
        {
            "workflow": "job_search",
            "stage": "submission",
            "requests": [
                {"action_id": str(INTENT.action_id), "action_fingerprint": INTENT.fingerprint()}
            ],
        }
    )
    return {"approvals": decisions}


async def _noop_sleep(_: float) -> None:
    return None


class Worker:
    """A worker process: its own engine, registry, checkpointer, graphs and executor."""

    def __init__(self, db_path: Path, provider: FlakyProvider):
        engine = create_engine(f"sqlite:///{db_path}")
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.registry = build_workflow_thread_registry(self.factory)
        self.checkpointer = build_durable_checkpointer(self.factory, self.registry)
        self.leases = build_workflow_lease_store(self.factory)
        self.queue = build_workflow_command_queue(self.factory)
        self.provider = provider

        job = StateGraph(_JobState)
        job.add_node(
            "approval_checkpoint", traced_node("approval_checkpoint", _approval_checkpoint)
        )
        job.add_node(
            "execute_approved_actions", traced_node("execute_approved_actions", self._execute)
        )
        job.add_edge(START, "approval_checkpoint")
        job.add_edge("approval_checkpoint", "execute_approved_actions")
        job.add_edge("execute_approved_actions", END)
        self.job_graph = job.compile(checkpointer=self.checkpointer)

    async def _execute(self, state: _JobState, config: RunnableConfig) -> dict[str, Any]:
        workflow_id = UUID(config["configurable"]["workflow_id"])
        executor = ToolExecutor(
            self.provider,
            build_policy_engine(self.factory),
            ExecutionLedger(self.factory, workflow_id=workflow_id),
            retry_policies=lambda tool: DRAFT,
            sleep=_noop_sleep,
        )
        receipts = []
        for raw in state.get("approvals") or []:
            receipt = await executor.execute(INTENT, ApprovalDecision.model_validate(raw))
            receipts.append(receipt.model_dump(mode="json"))
        return {"receipts": receipts}

    def run_next(self, workflow_id: UUID) -> bool:
        job_thread = register_job_search_thread(
            user_id=USER_ID, registry=self.registry, workflow_id=workflow_id
        )
        supervisor = SupervisorGraph(
            KeywordIntentClassifier(),
            build_job_search_subgraph_runner(self.job_graph, user_id=USER_ID, thread=job_thread),
            checkpointer=self.checkpointer,
        ).build()
        runners = {
            SUPERVISOR_THREAD_NAMESPACE: DurableWorkflowRunner(
                supervisor, registry=self.registry, leases=self.leases
            ),
            JOB_SEARCH_THREAD_NAMESPACE: DurableWorkflowRunner(
                self.job_graph, registry=self.registry, leases=self.leases
            ),
        }
        return asyncio.run(process_next(self.queue, runners))


def run_demo_workflow(db_path: Path, *, message: str) -> tuple[UUID, FlakyProvider]:
    """API -> worker (to the approval) -> API resume -> worker (executes). Returns the workflow."""
    provider = FlakyProvider()

    started = api_process(db_path).post(
        "/v1/chat", json={"message": message}, headers={"X-Actor-Id": "user:candidate"}
    )
    assert started.status_code == 202, started.text
    workflow_id = UUID(started.json()["workflow_id"])

    assert Worker(db_path, provider).run_next(workflow_id)
    parked = api_process(db_path).get(f"/v1/workflows/{workflow_id}").json()
    assert parked["status"] == "waiting", parked

    resumed = api_process(db_path).post(
        f"/v1/workflows/{workflow_id}/resume",
        json={
            "decisions": [
                {
                    "action_id": str(INTENT.action_id),
                    "action_fingerprint": INTENT.fingerprint(),
                    "verdict": "approved",
                }
            ]
        },
        headers={"X-Actor-Id": "user:reviewer"},
    )
    assert resumed.status_code == 202, resumed.text

    assert Worker(db_path, provider).run_next(workflow_id)
    done = api_process(db_path).get(f"/v1/workflows/{workflow_id}").json()
    assert done["status"] == "completed", done
    return workflow_id, provider


def _by_name(finished) -> dict[str, list]:
    named: dict[str, list] = {}
    for span in finished:
        named.setdefault(span.name, []).append(span)
    return named


# --- Tests ----------------------------------------------------------------------


def test_api_interrupt_approval_resume_and_execution_are_one_connected_trace(spans, db_path):
    workflow_id, provider = run_demo_workflow(db_path, message="find me backend jobs")
    assert provider.calls == 2

    finished = spans.get_finished_spans()
    trace_id = workflow_trace_id(workflow_id)

    # Every span the run produced is in the workflow's trace, identified by
    # the workflow id itself.
    assert finished
    assert {span.context.trace_id for span in finished} == {trace_id}

    # ...and the trace is a single tree: one root, every other span's parent
    # is a span of the same trace.
    ids = {span.context.span_id for span in finished}
    roots = [span for span in finished if span.parent is None]
    assert [root.name for root in roots] == ["api.chat"]
    orphans = [span.name for span in finished if span.parent and span.parent.span_id not in ids]
    assert orphans == []

    named = _by_name(finished)
    by_id = {span.context.span_id: span for span in finished}

    def parent(span):
        return by_id[span.parent.span_id]

    # The trace context crossed the outbox: each worker run is a child of the
    # request that queued it, not merely of the workflow.
    (start,) = named["worker.command.start"]
    (resume,) = named["worker.command.resume"]
    assert parent(start).name == "api.chat"
    assert parent(resume).name == "api.workflow.resume"

    # Every step is there: nodes, the model call, the interrupt, the policy
    # decision, each attempt at the action, and its receipt.
    expected = {
        "api.chat",
        "worker.command.start",
        "graph.run",
        "node.load_context",
        "node.classify_intent",
        "model.classify_intent",
        "node.run_job_subgraph",
        "node.approval_checkpoint",
        "api.workflow.resume",
        "worker.command.resume",
        "node.execute_approved_actions",
        "tool.execute",
        "policy.evaluate",
        "tool.attempt",
        "tool.receipt",
    }
    assert expected <= set(named), expected - set(named)

    # The interrupt is a pause, not a failure.
    parked_node = named["node.approval_checkpoint"][0]
    assert parked_node.attributes[Attr.INTERRUPTED] is True
    assert parked_node.status.is_ok

    # A retry is a span of its own, numbered.
    attempts = sorted(named["tool.attempt"], key=lambda span: span.attributes[Attr.ATTEMPT])
    assert [span.attributes[Attr.ATTEMPT] for span in attempts] == [1, 2]
    assert not attempts[0].status.is_ok and attempts[1].status.is_ok
    assert all(parent(span).name == "tool.execute" for span in attempts)

    # Consistent attributes: every span carries the workflow id, and the ones
    # that describe a step say which.
    assert {span.attributes[Attr.WORKFLOW_ID] for span in finished} == {str(workflow_id)}
    (policy,) = named["policy.evaluate"]
    assert policy.attributes[Attr.POLICY_DECISION] == "require_approval"
    assert policy.attributes[Attr.TOOL_NAME] == "jobs.submit_application"
    assert policy.attributes[Attr.ACTION_ID] == str(INTENT.action_id)
    (receipt,) = named["tool.receipt"]
    assert receipt.attributes[Attr.RECEIPT_OK] is True
    assert receipt.attributes[Attr.EXTERNAL_REFERENCE] == "app-42"
    assert receipt.attributes[Attr.THREAD_ID].startswith("job_search:")
    assert named["node.load_context"][0].attributes[Attr.THREAD_ID].startswith("supervisor:")
    assert resume.attributes[Attr.ACTOR_ID] == "user:reviewer"
    assert start.attributes[Attr.ACTOR_ID] == "user:candidate"
    assert {Attr.CORRELATION_ID in span.attributes for span in finished} == {True}


def test_a_seeded_secret_appears_in_no_exported_span(spans, db_path, registered_secret):
    workflow_id, _ = run_demo_workflow(
        db_path,
        message=f"find me backend jobs; my token is {PATTERN_SECRET} and {registered_secret}",
    )

    finished = spans.get_finished_spans()
    dump = "\n".join(span.to_json() for span in finished)
    attributes = json.dumps(
        [
            {
                "attributes": dict(span.attributes),
                "events": [dict(event.attributes or {}) for event in span.events],
                "status": span.status.description,
            }
            for span in finished
        ]
    )

    # The leaky error did reach the trace -- as a recorded exception on the
    # failed attempt -- so this is not passing for want of anything to redact.
    (failed,) = [span for span in finished if span.name == "tool.attempt" and not span.status.is_ok]
    (exception,) = [event for event in failed.events if event.name == "exception"]
    assert "board returned 503" in exception.attributes["exception.message"]
    assert REDACTED in exception.attributes["exception.message"]

    for secret in (PATTERN_SECRET, registered_secret):
        assert secret not in dump
        assert secret not in attributes
    # The chat message is a payload: no span carries it, redacted or not.
    assert "find me backend jobs" not in dump
    assert str(workflow_id) in attributes


def test_tracing_disabled_still_runs_the_workflow_and_carries_no_context(db_path):
    """Off is the default: the same workflow runs, and no command carries a trace."""
    workflow_id, provider = run_demo_workflow(db_path, message="find me backend jobs")
    assert provider.calls == 2

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as connection:
            rows = connection.exec_driver_sql(
                "SELECT payload_json FROM outbox_events WHERE type = 'workflow.command'"
            ).fetchall()
    finally:
        engine.dispose()
    payloads = [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
    assert payloads and all(payload["trace_context"] == {} for payload in payloads)
