"""Tests for `GET /v1/workflows/{id}/events`: progress streamed from the durable log.

The acceptance criteria this file exists for:

1. **A client that disconnects mid-workflow and reconnects with Last-Event-ID
   receives exactly the missed events.**
   `test_a_client_that_reconnects_with_last_event_id_gets_exactly_what_it_missed`
   (over HTTP) and `test_a_live_client_dropped_mid_run_resumes_without_gap_or_repeat`
   (a following stream, cut off while the worker is still running).
2. **An approval_requested event is emitted when a workflow interrupts**, and
   approval_resolved when its answer is picked up.
   `test_an_interrupt_emits_approval_requested_and_its_answer_approval_resolved`.

Everything below the HTTP layer is real: a three-step graph
(`prepare -> review -> finish`, where `review` interrupts) on the durable
checkpointer, run by `process_next` the way the worker runs it, with the API
reading the same SQLite file from a separate engine.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.events import (
    EventStreamSettings,
    build_progress_log,
    get_progress_log,
    progress_stream,
)
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    build_durable_checkpointer,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_progress_log,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.domain.errors import ProviderPermanentFailure
from personalos.domain.progress import WORKFLOW_AGGREGATE, ProgressEventType
from personalos.domain.workflow import (
    JOB_SEARCH_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
)
from personalos.persistence.models import Base, EventLogModel
from personalos.persistence.workflow_progress import WorkflowProgressLog

USER_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000dd")
SECRET = "sk-live-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"
ACTOR = "user:alice"

#: The whole run, as the log records it.
FULL_RUN = [
    "step_started",  # prepare
    "step_completed",
    "step_started",  # review
    "approval_requested",
    "approval_resolved",
    "step_started",  # review, re-entered with the answer
    "step_completed",
    "step_started",  # finish
    "step_completed",
    "workflow_completed",
]


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "events.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


def session_factory(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def api_process(db_path: Path) -> TestClient:
    """A fresh API instance over the database, sharing nothing but the file."""
    factory = session_factory(db_path)
    app = create_app()
    app.dependency_overrides[get_progress_log] = lambda: build_progress_log(factory)
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    app.dependency_overrides[get_failure_services] = lambda: build_failure_services(factory)
    return TestClient(app)


class _State(TypedDict, total=False):
    prepared: bool
    decisions: Any
    finished: bool


def request_payload(action_id: UUID) -> dict[str, Any]:
    """What `review` interrupts with: the job search approval shape, secret included."""
    return {
        "workflow": "job_search",
        "stage": "application",
        "requests": [
            {
                "request_id": str(uuid4()),
                "action_id": str(action_id),
                "action_hash": "a" * 64,
                "kind": "submit_application",
                "target": f"https://example.test/apply?token={SECRET}",
                "summary": f"Submit to Acme with key {SECRET}",
                "risk": "high",
                "idempotency_key": "submit-acme",
                "expires_at": "2026-12-01T00:00:00",
            }
        ],
    }


def builder(harness: "Harness") -> StateGraph:
    graph = StateGraph(_State)
    graph.add_node("prepare", harness._prepare)
    graph.add_node("review", harness._review)
    graph.add_node("finish", harness._finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "review")
    graph.add_edge("review", "finish")
    graph.add_edge("finish", END)
    return graph


class Harness:
    """One workflow on one thread, its graph, and a worker to run its commands."""

    def __init__(
        self,
        db_path: Path,
        *,
        finish_errors: list[Exception] | None = None,
        progress: Any = None,
    ):
        self.factory = session_factory(db_path)
        self.registry = build_workflow_thread_registry(self.factory)
        self.queue = build_workflow_command_queue(self.factory)
        self.finish_errors = list(finish_errors or [])
        self.action_id = uuid4()
        self.correlation_id = uuid4()

        self.workflow_id = self.registry.create_workflow(JOB_SEARCH_WORKFLOW)
        self.thread = register_job_search_thread(
            user_id=USER_ID, registry=self.registry, workflow_id=self.workflow_id
        )
        self.graph = builder(self).compile(
            checkpointer=build_durable_checkpointer(self.factory, self.registry)
        )
        self.runner = DurableWorkflowRunner(
            self.graph,
            registry=self.registry,
            leases=build_workflow_lease_store(self.factory),
            progress=progress,
        )

    def _prepare(self, state: _State) -> dict[str, Any]:
        return {"prepared": True}

    def _review(self, state: _State) -> dict[str, Any]:
        return {"decisions": interrupt(request_payload(self.action_id))}

    def _finish(self, state: _State) -> dict[str, Any]:
        if self.finish_errors:
            raise self.finish_errors.pop(0)
        return {"finished": True}

    def start(self) -> None:
        self.queue.enqueue(
            WorkflowCommand(
                command_id=uuid4(),
                kind=WorkflowCommandKind.START,
                workflow_id=self.workflow_id,
                thread_id=self.thread.thread_id,
                actor_id=ACTOR,
                correlation_id=self.correlation_id,
                graph_input={},
            )
        )

    def run_next(self) -> bool:
        return asyncio.run(process_next(self.queue, {JOB_SEARCH_THREAD_NAMESPACE: self.runner}))

    def approve(self, client: TestClient) -> None:
        response = client.post(
            f"/v1/workflows/{self.workflow_id}/resume",
            headers={"X-Actor-Id": ACTOR},
            json={
                "decisions": [
                    {
                        "action_id": str(self.action_id),
                        "action_fingerprint": "a" * 64,
                        "verdict": "approved",
                    }
                ]
            },
        )
        assert response.status_code == 202, response.text

    def run_to_completion(self, client: TestClient) -> None:
        self.start()
        assert self.run_next()
        self.approve(client)
        assert self.run_next()


def parse_frames(text: str) -> list[dict[str, Any]]:
    """The event frames of an SSE body; `retry:` and comment frames are skipped."""
    frames = []
    for block in text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(": ")
            fields[name] = value
        if "event" in fields:
            frames.append(
                {"id": fields["id"], "event": fields["event"], "data": json.loads(fields["data"])}
            )
    return frames


def read_events(client: TestClient, workflow_id: UUID, **headers: str) -> list[dict[str, Any]]:
    response = client.get(
        f"/v1/workflows/{workflow_id}/events", params={"follow": "false"}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_frames(response.text)


def kinds(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["event"] for frame in frames]


# --- Acceptance -----------------------------------------------------------------


def test_a_client_that_reconnects_with_last_event_id_gets_exactly_what_it_missed(db_path):
    """Acceptance 1: everything after the last id it saw, and nothing it already had."""
    harness = Harness(db_path)
    client = api_process(db_path)
    harness.start()
    assert harness.run_next()  # parks at the approval

    # The client watches the first part of the run, then drops after two events.
    before = read_events(client, harness.workflow_id)
    assert kinds(before) == FULL_RUN[:4]
    seen = before[:2]

    # The run carries on with nobody connected.
    harness.approve(client)
    assert harness.run_next()

    resumed = read_events(client, harness.workflow_id, **{"Last-Event-ID": seen[-1]["id"]})
    everything = read_events(client, harness.workflow_id)

    assert kinds(everything) == FULL_RUN
    assert resumed == everything[2:]
    assert {f["id"] for f in seen}.isdisjoint(f["id"] for f in resumed)
    assert seen + resumed == everything


def test_a_live_client_dropped_mid_run_resumes_without_gap_or_repeat(db_path):
    """Acceptance 1, live: a following stream cut off while the worker is running."""
    harness = Harness(db_path)
    client = api_process(db_path)
    log = build_workflow_progress_log(session_factory(db_path))
    settings = EventStreamSettings(poll_seconds=0.01, heartbeat_seconds=60)

    async def never_disconnected() -> bool:
        return False

    async def scenario() -> tuple[list[dict], list[dict]]:
        harness.start()
        received: list[dict[str, Any]] = []

        async def watch() -> None:
            stream = progress_stream(
                log,
                harness.workflow_id,
                after=None,
                follow=True,
                settings=settings,
                is_disconnected=never_disconnected,
            )
            async for frame in stream:
                received.extend(parse_frames(frame))
                if len(received) == 3:
                    # The connection drops: whatever the stream would send
                    # next never reaches this client.
                    await stream.aclose()
                    return

        watcher = asyncio.create_task(watch())
        # The worker runs in its own thread, as it would in its own process.
        await asyncio.to_thread(harness.run_next)
        await asyncio.wait_for(watcher, timeout=10)
        await asyncio.to_thread(harness.approve, client)
        await asyncio.to_thread(harness.run_next)

        reconnected: list[dict[str, Any]] = []
        async for frame in progress_stream(
            log,
            harness.workflow_id,
            after=UUID(received[-1]["id"]),
            follow=False,
            settings=settings,
            is_disconnected=never_disconnected,
        ):
            reconnected.extend(parse_frames(frame))
        return received, reconnected

    received, reconnected = asyncio.run(scenario())

    assert kinds(received) == FULL_RUN[:3]
    assert kinds(reconnected) == FULL_RUN[3:]
    assert received + reconnected == read_events(client, harness.workflow_id)


def test_an_interrupt_emits_approval_requested_and_its_answer_approval_resolved(db_path):
    """Acceptance 2: the interrupt is announced, minimal and secret-free."""
    harness = Harness(db_path)
    client = api_process(db_path)
    harness.start()
    assert harness.run_next()

    response = client.get(f"/v1/workflows/{harness.workflow_id}/events", params={"follow": "false"})
    assert SECRET not in response.text
    [requested] = [f for f in parse_frames(response.text) if f["event"] == "approval_requested"]
    pending = client.get(f"/v1/workflows/{harness.workflow_id}").json()["pending_approval"]

    event = requested["data"]
    assert event["event_id"] == requested["id"]
    assert event["type"] == "approval_requested"
    assert event["workflow_id"] == str(harness.workflow_id)
    assert event["correlation_id"] == str(harness.correlation_id)
    assert event["thread_id"] == harness.thread.thread_id
    assert event["data"]["step"] == "review"
    assert event["data"]["interrupt_id"] == pending["interrupt_id"]
    assert event["data"]["stage"] == "application"
    [request] = event["data"]["requests"]
    assert set(request) == {"request_id", "action_id", "kind", "risk", "expires_at"}
    assert request["action_id"] == str(harness.action_id)

    harness.approve(client)
    assert harness.run_next()

    [resolved] = [
        f for f in read_events(client, harness.workflow_id) if f["event"] == "approval_resolved"
    ]
    assert resolved["data"]["data"] == {
        "steps": ["review"],
        "interrupt_ids": [pending["interrupt_id"]],
        "resolved_by": ACTOR,
        "decisions": [
            {"request_id": None, "action_id": str(harness.action_id), "verdict": "approved"}
        ],
    }


# --- Failures -------------------------------------------------------------------


def test_a_failed_step_emits_failure_under_the_id_the_failures_endpoint_uses(db_path):
    refusal = ProviderPermanentFailure(
        f"google refused the token {SECRET}",
        provider="google",
        reason=ProviderPermanentFailure.MISSING_SCOPE,
    )
    harness = Harness(db_path, finish_errors=[refusal])
    client = api_process(db_path)
    harness.start()
    assert harness.run_next()
    harness.approve(client)
    with pytest.raises(ProviderPermanentFailure):
        harness.run_next()

    response = client.get(f"/v1/workflows/{harness.workflow_id}/events", params={"follow": "false"})
    assert SECRET not in response.text
    frames = parse_frames(response.text)
    assert kinds(frames)[-2:] == ["step_started", "failure"]
    failure = frames[-1]["data"]["data"]
    [listed] = client.get(f"/v1/workflows/{harness.workflow_id}/failures").json()["failures"]
    assert failure["step"] == "finish"
    assert failure["failure_id"] == listed["failure_id"]
    assert failure["failure_class"] == "provider_permanent"
    assert "google refused the token" in failure["message"]


# --- Streaming never changes the run --------------------------------------------


class BrokenProgressLog:
    """A progress log that refuses every write."""

    def __init__(self):
        self.attempts = 0

    def record(self, *args: Any, **kwargs: Any) -> None:
        self.attempts += 1
        raise RuntimeError("event_log is unavailable")


def test_a_progress_log_that_cannot_be_written_does_not_change_the_run(db_path):
    broken = BrokenProgressLog()
    harness = Harness(db_path, progress=broken)
    client = api_process(db_path)
    harness.run_to_completion(client)

    assert broken.attempts == len(FULL_RUN)
    state = asyncio.run(harness.graph.aget_state(harness.thread.config())).values
    assert state["finished"] is True
    assert client.get(f"/v1/workflows/{harness.workflow_id}").json()["status"] == "completed"


def test_the_runner_returns_what_ainvoke_would(db_path):
    """Streaming to watch the tasks leaves the result -- interrupts included -- alone."""
    harness = Harness(db_path)
    thread = harness.thread
    streamed = asyncio.run(harness.runner.start(thread, {}))

    plain = builder(harness).compile(checkpointer=InMemorySaver())
    expected = asyncio.run(plain.ainvoke({}, config={"configurable": {"thread_id": "plain"}}))

    assert streamed.keys() == expected.keys()
    assert streamed["prepared"] is True
    assert [i.value for i in streamed["__interrupt__"]][0]["stage"] == "application"


def test_events_recorded_with_no_client_connected_are_all_there_later(db_path):
    harness = Harness(db_path)
    client = api_process(db_path)
    harness.run_to_completion(client)
    assert kinds(read_events(client, harness.workflow_id)) == FULL_RUN


# --- Cursor and request validation ----------------------------------------------


def test_the_query_parameter_resumes_like_the_header_and_the_header_wins(db_path):
    harness = Harness(db_path)
    client = api_process(db_path)
    harness.run_to_completion(client)
    everything = read_events(client, harness.workflow_id)

    response = client.get(
        f"/v1/workflows/{harness.workflow_id}/events",
        params={"follow": "false", "last_event_id": everything[4]["id"]},
    )
    assert parse_frames(response.text) == everything[5:]

    response = client.get(
        f"/v1/workflows/{harness.workflow_id}/events",
        params={"follow": "false", "last_event_id": everything[4]["id"]},
        headers={"Last-Event-ID": everything[7]["id"]},
    )
    assert parse_frames(response.text) == everything[8:]


def test_resuming_from_the_last_event_returns_nothing(db_path):
    harness = Harness(db_path)
    client = api_process(db_path)
    harness.run_to_completion(client)
    last = read_events(client, harness.workflow_id)[-1]
    assert read_events(client, harness.workflow_id, **{"Last-Event-ID": last["id"]}) == []


def test_a_cursor_from_another_workflow_is_refused_not_replayed(db_path):
    first, second = Harness(db_path), Harness(db_path)
    client = api_process(db_path)
    first.run_to_completion(client)
    foreign = read_events(client, first.workflow_id)[0]["id"]

    response = client.get(
        f"/v1/workflows/{second.workflow_id}/events",
        params={"follow": "false"},
        headers={"Last-Event-ID": foreign},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation"


def test_a_malformed_cursor_is_refused(db_path):
    harness = Harness(db_path)
    client = api_process(db_path)
    response = client.get(
        f"/v1/workflows/{harness.workflow_id}/events",
        params={"follow": "false"},
        headers={"Last-Event-ID": "not-an-id"},
    )
    assert response.status_code == 422


def test_an_unknown_workflow_is_404(db_path):
    response = api_process(db_path).get(
        f"/v1/workflows/{uuid4()}/events", params={"follow": "false"}
    )
    assert response.status_code == 404


def test_the_stream_suggests_a_reconnect_delay_and_frames_each_event(db_path):
    harness = Harness(db_path)
    client = api_process(db_path)
    harness.start()
    assert harness.run_next()
    response = client.get(f"/v1/workflows/{harness.workflow_id}/events", params={"follow": "false"})
    assert response.headers["cache-control"] == "no-cache"
    assert response.text.startswith("retry: 3000\n\n")
    first = response.text.split("\n\n")[1].splitlines()
    assert first[0].startswith("id: ")
    assert first[1] == "event: step_started"
    assert first[2].startswith("data: {")


# --- Ordering -------------------------------------------------------------------


def test_an_event_is_never_stamped_behind_one_already_in_the_log(db_path):
    """A worker whose clock runs behind the last one still appends after it."""
    factory = session_factory(db_path)
    harness = Harness(db_path)
    log = WorkflowProgressLog(factory)
    ahead = datetime.utcnow() + timedelta(hours=1)
    session = factory()
    try:
        row = EventLogModel(
            aggregate_type=WORKFLOW_AGGREGATE,
            aggregate_id=harness.workflow_id,
            event_type=ProgressEventType.STEP_STARTED.stored,
            payload_json={"thread_id": harness.thread.thread_id, "data": {"step": "prepare"}},
            occurred_at=ahead,
        )
        session.add(row)
        session.commit()
        earlier_id = row.id
    finally:
        session.close()

    later_id = log.record(
        harness.workflow_id,
        ProgressEventType.STEP_COMPLETED,
        thread_id=harness.thread.thread_id,
        data={"step": "prepare"},
    )

    [after] = log.events_after(harness.workflow_id, earlier_id)
    assert after.event_id == later_id
    assert after.at > ahead
    assert [e.event_id for e in log.events_after(harness.workflow_id)] == [earlier_id, later_id]


def test_a_long_backlog_is_read_in_batches_without_loss(db_path):
    harness = Harness(db_path)
    log = build_workflow_progress_log(session_factory(db_path))
    for index in range(7):
        log.record(harness.workflow_id, ProgressEventType.STEP_STARTED, data={"step": str(index)})

    async def drain() -> list[dict[str, Any]]:
        async def connected() -> bool:
            return False

        frames: list[dict[str, Any]] = []
        async for frame in progress_stream(
            log,
            harness.workflow_id,
            after=None,
            follow=False,
            settings=EventStreamSettings(batch_size=3),
            is_disconnected=connected,
        ):
            frames.extend(parse_frames(frame))
        return frames

    frames = asyncio.run(drain())
    assert [f["data"]["data"]["step"] for f in frames] == [str(i) for i in range(7)]
