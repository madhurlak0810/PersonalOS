"""Tests for the v1 workflow lifecycle endpoints.

The acceptance criteria this file exists for:

1. **Status survives an API restart.** `test_status_survives_an_api_process_restart`
   starts a workflow from a child process that then exits, and reads its status
   from this one. A second variant drives the workflow to an approval with a
   worker and reads it back through a freshly built app.
2. **Resuming a workflow that is not waiting is a 409**, not a silent no-op --
   whether it is still queued, already answered, or finished.
3. **An unknown workflow is a 404.**

"API process" here is an app built by `create_app()` whose services are
constructed per request from a session factory. Every "restart" builds a new
app and new services over the same database file, so nothing reaches the
second instance except what the first wrote to disk.

The worker side is real: `apps.worker.workflow_commands.process_next` claims
the queued command and runs a real `SupervisorGraph` on the durable
checkpointer. Only the job-search domain graph is a stand-in -- a three-node
graph with the same interrupt shape as `approval_checkpoint` -- because what is
under test is the lifecycle around an interrupt, not the job search.
"""

import asyncio
import json
import os
import subprocess
import sys
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
    build_durable_checkpointer,
    build_job_search_subgraph_runner,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.domain.workflow import JOB_SEARCH_THREAD_NAMESPACE, SUPERVISOR_THREAD_NAMESPACE
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.routing import KeywordIntentClassifier
from personalos.persistence.models import Base, CheckpointModel

REPO_ROOT = Path(__file__).resolve().parents[2]

USER_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000aa")
ACTION_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000bb")
FINGERPRINT = "f" * 64
JOB_MESSAGE = "find me python backend jobs"


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "workflows.db"
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
    """A fresh API instance over the database: new app, new engine, new services.

    Its own engine as well as its own app, so not even a connection pool is
    shared with whatever instance served the previous request.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


class _JobState(TypedDict, total=False):
    user_id: str
    query: str
    prepare_application: bool
    shortlist: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    application: dict[str, Any] | None


def _shortlist(state: _JobState) -> dict[str, Any]:
    return {"shortlist": [{"title": "Backend Engineer", "company": "Acme"}]}


def _approval_checkpoint(state: _JobState) -> dict[str, Any]:
    decisions = interrupt(
        {
            "workflow": "job_search",
            "stage": "submission",
            "requests": [{"action_id": str(ACTION_ID), "action_fingerprint": FINGERPRINT}],
        }
    )
    return {"approvals": decisions}


def _submit(state: _JobState) -> dict[str, Any]:
    approved = [d for d in state.get("approvals") or [] if d["verdict"] == "approved"]
    return {"application": {"status": "submitted" if approved else "withheld"}}


class Worker:
    """A worker process: its own engine, registry, checkpointer and graphs.

    Wires a Supervisor whose job subgraph runs on a job-search thread of the
    same workflow -- the composition `personalos.bootstrap` describes -- and
    processes queued commands with `process_next`.
    """

    def __init__(self, db_path: Path, *, job_subgraph_fails: bool = False):
        engine = create_engine(f"sqlite:///{db_path}")
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.registry = build_workflow_thread_registry(self.factory)
        self.checkpointer = build_durable_checkpointer(self.factory, self.registry)
        self.leases = build_workflow_lease_store(self.factory)
        self.queue = build_workflow_command_queue(self.factory)
        self.job_subgraph_fails = job_subgraph_fails

        job = StateGraph(_JobState)
        job.add_node("shortlist", _shortlist)
        job.add_node("approval_checkpoint", _approval_checkpoint)
        job.add_node("submit", _submit)
        job.add_edge(START, "shortlist")
        job.add_edge("shortlist", "approval_checkpoint")
        job.add_edge("approval_checkpoint", "submit")
        job.add_edge("submit", END)
        self.job_graph = job.compile(checkpointer=self.checkpointer)

    def _supervisor_runner(self, workflow_id: UUID) -> DurableWorkflowRunner:
        job_thread = register_job_search_thread(
            user_id=USER_ID, registry=self.registry, workflow_id=workflow_id
        )
        job_subgraph = build_job_search_subgraph_runner(
            self.job_graph, user_id=USER_ID, thread=job_thread
        )
        if self.job_subgraph_fails:

            async def job_subgraph(task_dag, state):  # noqa: ARG001 - port signature
                raise RuntimeError("job board provider timed out")

        graph = SupervisorGraph(
            KeywordIntentClassifier(), job_subgraph, checkpointer=self.checkpointer
        ).build()
        return DurableWorkflowRunner(graph, registry=self.registry, leases=self.leases)

    def run_next(self, workflow_id: UUID) -> bool:
        runners = {
            SUPERVISOR_THREAD_NAMESPACE: self._supervisor_runner(workflow_id),
            JOB_SEARCH_THREAD_NAMESPACE: DurableWorkflowRunner(
                self.job_graph, registry=self.registry, leases=self.leases
            ),
        }
        return asyncio.run(process_next(self.queue, runners))


def start(client: TestClient, **headers: str) -> dict[str, Any]:
    response = client.post("/v1/chat", json={"message": JOB_MESSAGE}, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


def approve(client: TestClient, workflow_id: str, **headers: str):
    return client.post(
        f"/v1/workflows/{workflow_id}/resume",
        json={
            "decisions": [
                {
                    "action_id": str(ACTION_ID),
                    "action_fingerprint": FINGERPRINT,
                    "verdict": "approved",
                }
            ]
        },
        headers=headers,
    )


def parked_at_approval(db_path: Path, **headers: str) -> tuple[str, Worker]:
    """Start a workflow through the API and let a worker run it to the approval."""
    started = start(api_process(db_path), **headers)
    worker = Worker(db_path)
    assert worker.run_next(UUID(started["workflow_id"]))
    return started["workflow_id"], worker


# --- POST /v1/chat --------------------------------------------------------------


def test_chat_starts_a_workflow_and_hands_it_to_the_worker(db_path):
    """The request returns at once with ids; the run itself is only queued."""
    client = api_process(db_path)
    response = client.post(
        "/v1/chat", json={"message": JOB_MESSAGE}, headers={"X-Actor-Id": "user:alice"}
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert UUID(body["workflow_id"])
    assert body["thread_id"].startswith("supervisor:")
    assert body["status"] == "queued"
    assert body["actor_id"] == "user:alice"
    assert response.headers["Location"] == f"/v1/workflows/{body['workflow_id']}"

    status = client.get(f"/v1/workflows/{body['workflow_id']}").json()
    assert status["status"] == "queued"
    assert status["queued_commands"] == 1
    assert status["completed_steps"] == []
    assert [t["thread_id"] for t in status["threads"]] == [body["thread_id"]]


def test_each_new_conversation_is_its_own_workflow(db_path):
    """Two conversations never share a workflow_id, so neither's status is the other's."""
    client = api_process(db_path)
    first, second = start(client), start(client)
    assert first["workflow_id"] != second["workflow_id"]
    assert first["thread_id"] != second["thread_id"]


def test_chat_rejects_an_empty_message(db_path):
    response = api_process(db_path).post("/v1/chat", json={"message": ""})
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation"


def test_chat_on_an_unknown_thread_is_a_404(db_path):
    response = api_process(db_path).post(
        "/v1/chat", json={"message": JOB_MESSAGE, "thread_id": "supervisor:nope"}
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"


def test_chat_continues_a_finished_conversation_on_the_same_thread(db_path):
    """A follow-up message queues a new run on the existing thread and workflow."""
    workflow_id, worker = parked_at_approval(db_path)
    client = api_process(db_path)
    assert approve(client, workflow_id).status_code == 202
    assert worker.run_next(UUID(workflow_id))

    status = client.get(f"/v1/workflows/{workflow_id}").json()
    supervisor_thread = status["threads"][0]["thread_id"]
    response = client.post(
        "/v1/chat", json={"message": "and remote ones?", "thread_id": supervisor_thread}
    )

    assert response.status_code == 202, response.text
    assert response.json()["workflow_id"] == workflow_id
    assert response.json()["thread_id"] == supervisor_thread


def test_chat_continuing_a_conversation_parked_on_approval_is_a_409(db_path):
    """A new message would restart the thread and strand the open approval."""
    workflow_id, _ = parked_at_approval(db_path)
    client = api_process(db_path)
    thread_id = client.get(f"/v1/workflows/{workflow_id}").json()["threads"][0]["thread_id"]

    response = client.post("/v1/chat", json={"message": "hello?", "thread_id": thread_id})

    assert response.status_code == 409
    body = response.json()
    assert body["error_code"] == "conflict"
    assert "resume" in body["message"]


# --- GET /v1/workflows/{id} -----------------------------------------------------


def test_status_survives_an_api_process_restart(db_path):
    """Acceptance: start in one process, let it exit, read the status from another.

    The starting process runs the real wiring (`DATABASE_URL`, default
    dependency); it is gone by the time this process asks, so the answer can
    only have come from the database.
    """
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.fixtures.api_process",
            "POST",
            "/v1/chat",
            json.dumps({"message": JOB_MESSAGE}),
            "X-Actor-Id=user:alice",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    status_code, body = result.stdout.strip().splitlines()[-2:]
    assert status_code == "202", body
    started = json.loads(body)

    response = api_process(db_path).get(f"/v1/workflows/{started['workflow_id']}")

    assert response.status_code == 200, response.text
    status = response.json()
    assert status["workflow_id"] == started["workflow_id"]
    assert status["status"] == "queued"
    assert status["actor_id"] == "user:alice"
    assert status["correlation_id"] == started["correlation_id"]
    assert [t["thread_id"] for t in status["threads"]] == [started["thread_id"]]


def test_status_after_restart_reports_the_pending_approval(db_path):
    """Acceptance, past the queue: a worker parks the run, a new API instance reports it.

    Current step, pending approval and completed steps all come from the
    checkpoints the worker wrote; the API instance asking never compiled a
    graph.
    """
    workflow_id, _ = parked_at_approval(db_path)

    status = api_process(db_path).get(f"/v1/workflows/{workflow_id}").json()

    assert status["status"] == "waiting"
    job_thread = next(t for t in status["threads"] if t["kind"] == JOB_SEARCH_THREAD_NAMESPACE)
    assert job_thread["waiting"] is True
    assert status["current_step"] == {
        "thread_id": job_thread["thread_id"],
        "step": "approval_checkpoint",
    }

    approval = status["pending_approval"]
    assert approval["thread_id"] == job_thread["thread_id"]
    assert approval["step"] == "approval_checkpoint"
    assert approval["interrupt_id"]
    assert approval["payload"]["requests"][0]["action_id"] == str(ACTION_ID)

    completed = [step["step"] for step in status["completed_steps"]]
    assert completed[:4] == ["load_context", "classify_intent", "plan_work", "run_job_subgraph"]
    assert "shortlist" in completed
    assert "approval_checkpoint" not in completed
    assert status["recoverable_failures"] == []
    assert status["queued_commands"] == 0


def test_a_failed_node_is_reported_as_a_recoverable_failure(db_path):
    """A node that raised leaves its checkpoint intact, and says where it stopped."""
    started = start(api_process(db_path))
    worker = Worker(db_path, job_subgraph_fails=True)
    with pytest.raises(RuntimeError, match="timed out"):
        worker.run_next(UUID(started["workflow_id"]))

    status = api_process(db_path).get(f"/v1/workflows/{started['workflow_id']}").json()

    assert status["status"] == "failed"
    assert status["current_step"] == {
        "thread_id": started["thread_id"],
        "step": "run_job_subgraph",
    }
    [failure] = status["recoverable_failures"]
    assert failure["step"] == "run_job_subgraph"
    assert "timed out" in failure["message"]
    assert status["pending_approval"] is None


def test_unknown_workflow_is_a_404(db_path):
    """Acceptance: an id nothing was ever started under is not found, not empty."""
    response = api_process(db_path).get(f"/v1/workflows/{uuid4()}")

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "not_found"
    assert set(body) == {"error_code", "message", "context_id"}


def test_a_malformed_workflow_id_is_a_validation_error(db_path):
    response = api_process(db_path).get("/v1/workflows/not-a-uuid")
    assert response.status_code == 422


# --- POST /v1/workflows/{id}/resume ---------------------------------------------


def test_resume_queues_the_decision_and_the_worker_finishes_the_run(db_path):
    """The answer goes to the worker, which resumes at the approval and completes."""
    workflow_id, worker = parked_at_approval(db_path)
    client = api_process(db_path)

    response = approve(client, workflow_id, **{"X-Actor-Id": "user:reviewer"})

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["thread_id"].startswith("job_search:")

    # Answered but not yet run: no longer pending, not yet done.
    queued = client.get(f"/v1/workflows/{workflow_id}").json()
    assert queued["status"] == "queued"
    assert queued["pending_approval"] is None

    assert worker.run_next(UUID(workflow_id))
    done = api_process(db_path).get(f"/v1/workflows/{workflow_id}").json()
    assert done["status"] == "completed"
    assert done["current_step"] is None
    assert {"approval_checkpoint", "submit"} <= {s["step"] for s in done["completed_steps"]}

    final = asyncio.run(
        worker.job_graph.aget_state({"configurable": {"thread_id": body["thread_id"]}})
    )
    assert final.values["application"] == {"status": "submitted"}
    [decision] = final.values["approvals"]
    assert decision["decided_by"] == "user:reviewer"


def test_resuming_a_workflow_that_is_not_waiting_is_a_409(db_path):
    """Acceptance: queued, already answered, or finished -- each is a clear 409.

    The finished case runs first: the worker claims the oldest queued command,
    and a workflow left queued ahead of it would be the one it ran.
    """
    client = api_process(db_path)

    # Finished: the approval was answered and the run completed.
    workflow_id, worker = parked_at_approval(db_path)
    assert approve(client, workflow_id).status_code == 202
    assert worker.run_next(UUID(workflow_id))

    response = approve(client, workflow_id)
    assert response.status_code == 409
    body = response.json()
    assert body["error_code"] == "conflict"
    assert "not waiting" in body["message"]
    assert "completed" in body["message"]

    # Still queued: nothing has run, so nothing is waiting.
    queued = start(client)
    response = approve(client, queued["workflow_id"])
    assert response.status_code == 409
    assert "queued" in response.json()["message"]


def test_a_second_answer_to_the_same_approval_is_a_409(db_path):
    """Two reviewers answering the same question: the second is refused, not queued."""
    workflow_id, _ = parked_at_approval(db_path)

    assert approve(api_process(db_path), workflow_id).status_code == 202
    second = approve(api_process(db_path), workflow_id)

    assert second.status_code == 409
    assert second.json()["error_code"] == "conflict"
    status = api_process(db_path).get(f"/v1/workflows/{workflow_id}").json()
    assert status["queued_commands"] == 1


def test_resuming_an_unknown_workflow_is_a_404(db_path):
    response = approve(api_process(db_path), str(uuid4()))
    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"


def test_resume_requires_exactly_one_kind_of_answer(db_path):
    """`decisions` and `event` are alternatives; both or neither is malformed."""
    client = api_process(db_path)
    workflow_id = str(uuid4())

    neither = client.post(f"/v1/workflows/{workflow_id}/resume", json={})
    both = client.post(
        f"/v1/workflows/{workflow_id}/resume",
        json={
            "event": {"kind": "reply"},
            "decisions": [
                {"action_id": str(ACTION_ID), "action_fingerprint": "x", "verdict": "approved"}
            ],
        },
    )

    assert neither.status_code == 422
    assert both.status_code == 422


def test_resume_refuses_a_pending_verdict(db_path):
    """`pending` is not an answer; accepting it would resume the run into another wait."""
    response = api_process(db_path).post(
        f"/v1/workflows/{uuid4()}/resume",
        json={
            "decisions": [
                {"action_id": str(ACTION_ID), "action_fingerprint": "x", "verdict": "pending"}
            ]
        },
    )
    assert response.status_code == 422


def test_a_client_cannot_choose_who_decided(db_path):
    """`decided_by` comes from the actor header; a body field claiming it is rejected."""
    workflow_id, _ = parked_at_approval(db_path)
    response = api_process(db_path).post(
        f"/v1/workflows/{workflow_id}/resume",
        json={
            "decisions": [
                {
                    "action_id": str(ACTION_ID),
                    "action_fingerprint": FINGERPRINT,
                    "verdict": "approved",
                    "decided_by": "user:ceo",
                }
            ]
        },
    )
    assert response.status_code == 422


# --- Identity propagation -------------------------------------------------------


def test_actor_and_correlation_id_reach_the_workers_checkpoints(db_path, session_factory):
    """Phase A contract: the request's identity is what the worker's run is stamped with."""
    correlation_id = "6f1e7b3a-0000-4000-8000-0000000000cc"
    started = start(
        api_process(db_path),
        **{"X-Actor-Id": "user:alice", "X-Correlation-Id": correlation_id},
    )
    assert started["correlation_id"] == correlation_id
    assert Worker(db_path).run_next(UUID(started["workflow_id"]))

    session = session_factory()
    try:
        rows = (
            session.query(CheckpointModel)
            .filter(CheckpointModel.thread_id == started["thread_id"])
            .all()
        )
    finally:
        session.close()
    assert rows
    for row in rows:
        assert row.checkpoint_metadata["actor_id"] == "user:alice"
        assert row.checkpoint_metadata["correlation_id"] == correlation_id

    status = api_process(db_path).get(f"/v1/workflows/{started['workflow_id']}").json()
    assert status["actor_id"] == "user:alice"
    assert status["correlation_id"] == correlation_id


def test_openapi_lists_the_v1_workflow_routes():
    paths = set(create_app().openapi()["paths"])
    assert {
        "/v1/chat",
        "/v1/workflows/{workflow_id}",
        "/v1/workflows/{workflow_id}/resume",
    } <= paths
