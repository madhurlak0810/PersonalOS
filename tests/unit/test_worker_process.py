"""Long work runs in the worker, and survives the worker.

The acceptance criteria this file exists for:

1. **A discovery request returns before discovery completes, and the workflow
   later reaches completion through the worker.**
   `test_a_chat_request_is_accepted_before_discovery_runs_and_the_worker_completes_it`
   sends `POST /v1/chat`, finds the job boards untouched when the `202` comes
   back, then lets a worker drain the queue and reads the workflow back as
   completed -- with its progress in the log the event stream serves.
2. **A worker restarted mid-workflow continues from its checkpoint.**
   `test_a_worker_killed_mid_workflow_is_replaced_by_one_that_continues_from_the_checkpoint`
   kills a real worker process with SIGKILL after discovery and scoring are
   checkpointed. A second process reclaims the stranded command and finishes
   the workflow without searching the job boards again.
3. **A follow-up checkpoint that came due while the worker was down fires
   once when it returns.**
   `test_a_follow_up_due_while_the_worker_was_down_fires_once_when_it_returns`
   schedules a wait, runs no worker past its trigger, starts one, and checks the
   follow-up fires -- and that a second worker starting later does not fire it
   again.

Around those: approval expiry is swept by the worker rather than left parked
for ever, and a scheduled sweep that raises does not stop the others.
"""

import asyncio
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.main import WorkerGraphs, build_worker
from apps.worker.scheduler import ScheduledSweep, Scheduler
from personalos.bootstrap import build_workflow_progress_log
from personalos.domain.checkpoints import ConditionKind, PendingCheckpointStatus
from personalos.domain.job_search import (
    APPROVAL_EXPIRY_ACTOR,
    ActionKind,
    ApprovalRequest,
    JobSearchEventType,
    RefusalReason,
)
from personalos.domain.models import OutboxEventStatus
from personalos.domain.progress import ProgressEventType
from personalos.domain.workflow import ApprovalOutcome, WorkflowStatus
from personalos.graphs.job_search import APPROVAL_CHECKPOINT
from personalos.persistence.models import ApprovalModel, Base, OutboxEventModel
from personalos.persistence.workflow_commands import WORKFLOW_COMMAND_EVENT
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures import worker_process
from tests.graph_scenarios.test_pending_checkpoints import AT_TRIGGER, NOW, Deployment

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Fixtures -------------------------------------------------------------------


class Clock:
    """One clock for the graph, the stores and the worker, moved by hand."""

    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def _database(tmp_path) -> tuple[Path, sessionmaker]:
    path = tmp_path / "worker.db"
    engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    return path, sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _client(factory) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


def _chat(client: TestClient, message: str = "find me a python job") -> dict:
    response = client.post("/v1/chat", json={"message": message, "user_id": str(fakes.USER_ID)})
    assert response.status_code == 202, response.text
    return response.json()


def _status(client: TestClient, workflow_id: str) -> dict:
    response = client.get(f"/v1/workflows/{workflow_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _command_rows(factory) -> list[OutboxEventModel]:
    session = factory()
    try:
        return (
            session.query(OutboxEventModel)
            .filter(OutboxEventModel.type == WORKFLOW_COMMAND_EVENT)
            .all()
        )
    finally:
        session.close()


def _run_worker(db_path: Path, log_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tests.fixtures.worker_process", str(db_path), str(log_path), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


# --- Acceptance 1: the request returns, the worker does the work ---------------


async def test_a_chat_request_is_accepted_before_discovery_runs_and_the_worker_completes_it(
    tmp_path,
):
    """`POST /v1/chat` queues the search and returns; a worker runs it to the end.

    "Returns before discovery completes" is checked as strongly as it can be:
    by the time the response is in hand, discovery has not *started* -- no job
    board has been searched, and the only thing in the database is a queued
    command. Then a worker, sharing nothing with the API but the database,
    drains the queue, and the API reports the workflow completed.
    """
    db_path, factory = _database(tmp_path)
    log = durable.EventLog(tmp_path / "events.jsonl")
    client = _client(factory)

    accepted = _chat(client)

    assert accepted["status"] == WorkflowStatus.QUEUED.value
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 0
    queued = _status(client, accepted["workflow_id"])
    assert queued["status"] == WorkflowStatus.QUEUED.value
    assert queued["queued_commands"] == 1

    graphs = worker_process.build_graphs(factory, log, lease_ttl_seconds=60)
    worker = build_worker(factory, graphs, idle_seconds=0.01)
    assert await worker.drain() == 1

    done = _status(client, accepted["workflow_id"])
    assert done["status"] == WorkflowStatus.COMPLETED.value
    assert done["queued_commands"] == 0
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1

    # The worker reported as it went, where `GET .../events` streams from.
    events = build_workflow_progress_log(factory).events_after(UUID(accepted["workflow_id"]))
    kinds = {event.type for event in events}
    assert ProgressEventType.STEP_STARTED in kinds
    assert ProgressEventType.WORKFLOW_COMPLETED in kinds


# --- Acceptance 2: kill the worker, start another ------------------------------


def test_a_worker_killed_mid_workflow_is_replaced_by_one_that_continues_from_the_checkpoint(
    tmp_path,
):
    """A SIGKILLed worker's command is reclaimed, and the run picks up where it died.

    The first worker searches, scores and shortlists, then dies at the approval
    review: no `finally`, no settle, its command left claimed and its lease
    left held. Once the lease TTL has passed, a second worker reclaims the
    command, recognises the checkpoint as that command's own, and resumes it.
    The witness is the shared log: one search across both processes, two
    approval reviews (the one that died and the one that finished), and one
    submission.
    """
    db_path, factory = _database(tmp_path)
    log_path = tmp_path / "events.jsonl"
    log = durable.EventLog(log_path)
    client = _client(factory)
    ttl = "1"

    accepted = _chat(client)

    killed = _run_worker(db_path, log_path, ttl, durable.KILL_AT_APPROVAL)
    assert killed.returncode == -9, killed.stderr
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    assert log.count(durable.EVENT_APPROVAL_REVIEW) == 1
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 0
    # Stranded: claimed by a process that no longer exists.
    [row] = _command_rows(factory)
    assert row.status == OutboxEventStatus.IN_PROGRESS.value

    # Long enough for both the claim and the dead worker's lease to go stale.
    time.sleep(1.5)

    resumed = _run_worker(db_path, log_path, ttl)
    assert resumed.returncode == 0, resumed.stderr
    assert log.count(worker_process.EVENT_WORKER_DRAINED) == 1

    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1, "the restarted worker searched again"
    assert log.count(durable.EVENT_APPROVAL_REVIEW) == 2
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1
    [row] = _command_rows(factory)
    assert row.status == OutboxEventStatus.DISPATCHED.value
    assert row.attempts == 2
    assert _status(client, accepted["workflow_id"])["status"] == WorkflowStatus.COMPLETED.value


def test_a_worker_without_graphs_leaves_commands_queued_for_one_that_has_them(tmp_path):
    """A worker that cannot run a command does not claim it and strand it."""
    db_path, factory = _database(tmp_path)
    client = _client(factory)
    accepted = _chat(client)

    worker = build_worker(factory, WorkerGraphs(), idle_seconds=0.01)

    async def briefly() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        await asyncio.sleep(0.2)
        stop.set()
        await task

    asyncio.run(briefly())

    [row] = _command_rows(factory)
    assert row.status == OutboxEventStatus.PENDING.value
    assert _status(client, accepted["workflow_id"])["status"] == WorkflowStatus.QUEUED.value


# --- Acceptance 3: a follow-up due while the worker was down -------------------


async def _run_until(worker, done, *, timeout: float = 10.0) -> None:
    """Run the worker until `done()` holds, then stop it the way SIGTERM does."""
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    deadline = time.monotonic() + timeout
    try:
        while not done():
            assert time.monotonic() < deadline, "the worker never got there"
            await asyncio.sleep(0.02)
    finally:
        stop.set()
        await task


async def test_a_follow_up_due_while_the_worker_was_down_fires_once_when_it_returns(tmp_path):
    """No worker ran past the trigger; the first one to start fires the follow-up, once.

    The wait was scheduled by a run that finished; then nothing ran for eight
    days. Nothing in any process was counting down -- the only record of the
    wait is its `trigger_at`. A worker starting on day eight sweeps at startup,
    finds it overdue, and fires it. A second worker starting a minute later
    (a restart, a second replica) finds it already fired and does nothing.
    """
    evaluator = fakes.ScriptedConditionEvaluator({ConditionKind.RECRUITER_RESPONSE_RECEIVED: False})
    clock = Clock()
    deployment = Deployment(tmp_path, evaluator=evaluator, clock=clock)
    await deployment.apply()
    [wait] = deployment.store.open_for_application(fakes.APPLICATION_ID)
    assert wait.trigger_at == AT_TRIGGER
    searches = deployment.searches

    # Eight days pass with no worker running at all.
    clock.now = AT_TRIGGER + timedelta(days=1)
    graphs = WorkerGraphs(runners={"job_search": deployment.runner}, evaluator=evaluator)

    first = build_worker(deployment.factory, graphs, idle_seconds=0.01, clock=clock)
    await _run_until(
        first,
        lambda: deployment.store.get(wait.checkpoint_id).status != PendingCheckpointStatus.PENDING,
    )

    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.FIRED
    assert deployment.events.count(JobSearchEventType.FOLLOW_UP_TRIGGERED.value) == 1
    state = await deployment.runner.inspect(deployment.thread)
    assert state.next == (APPROVAL_CHECKPOINT,)
    requests = [ApprovalRequest.model_validate(raw) for raw in state.values["approval_requests"]]
    assert [r.kind for r in requests].count(ActionKind.SEND_RECRUITER_MESSAGE) == 1
    # Continued on the thread's own state: the boards were not searched again.
    assert deployment.searches == searches

    clock.now += timedelta(minutes=1)
    second = build_worker(deployment.factory, graphs, idle_seconds=0.01, clock=clock)
    results = await second.scheduler.run_once()

    assert results["follow_up_checkpoints"].considered == 0
    assert deployment.events.count(JobSearchEventType.FOLLOW_UP_TRIGGERED.value) == 1


# --- Approval expiry ------------------------------------------------------------


async def test_an_approval_nobody_answered_is_expired_by_the_worker_and_the_run_moves_on(
    tmp_path,
):
    """Past its expiry, an unanswered approval is closed by the sweep, not left parked.

    The follow-up from the previous test parks at an approval. Nobody answers;
    the worker comes back after the request's window has closed, records it as
    expired, and resumes the run down its declined branch. The refusal the run
    records says `approval_expired` -- it was nobody's "no" -- and nothing is
    sent.
    """
    evaluator = fakes.ScriptedConditionEvaluator({ConditionKind.RECRUITER_RESPONSE_RECEIVED: False})
    clock = Clock()
    deployment = Deployment(tmp_path, evaluator=evaluator, clock=clock)
    await deployment.apply()
    graphs = WorkerGraphs(runners={"job_search": deployment.runner}, evaluator=evaluator)
    clock.now = AT_TRIGGER + timedelta(days=1)
    worker = build_worker(deployment.factory, graphs, idle_seconds=0.01, clock=clock)
    results = await worker.scheduler.run_once()
    assert len(results["follow_up_checkpoints"].fired) == 1
    # Still inside its window: parked, not expired.
    assert results["approval_expiry"] == []
    state = await deployment.runner.inspect(deployment.thread)
    assert state.next == (APPROVAL_CHECKPOINT,)
    [request] = [
        ApprovalRequest.model_validate(raw)
        for raw in state.values["approval_requests"]
        if raw["kind"] == ActionKind.SEND_RECRUITER_MESSAGE.value
    ]

    clock.now = request.expires_at + timedelta(minutes=1)
    results = await worker.scheduler.run_once()
    assert results["approval_expiry"] == [request.request_id]
    assert await worker.drain() == 1

    session = deployment.factory()
    try:
        row = session.get(ApprovalModel, request.request_id)
        assert row.note == ApprovalOutcome.EXPIRED.value
    finally:
        session.close()
    state = await deployment.runner.inspect(deployment.thread)
    assert state.next == ()
    decisions = [d for d in state.values["approvals"] if d["action_id"] == str(request.action_id)]
    assert [d["decided_by"] for d in decisions] == [APPROVAL_EXPIRY_ACTOR]
    refusals = [
        r for r in state.values["approval_refusals"] if r["action_id"] == str(request.action_id)
    ]
    assert [r["reason"] for r in refusals] == [RefusalReason.EXPIRED.value]
    assert [intent.kind for intent, _ in deployment.ports["action_executor"].executed] == [
        ActionKind.SUBMIT_APPLICATION
    ]

    # A second sweep has nothing left to expire.
    assert (await worker.scheduler.run_once())["approval_expiry"] == []


# --- Scheduler ------------------------------------------------------------------


async def test_a_failing_sweep_does_not_stop_the_others():
    """One sweep raising is logged and returned; the rest still run."""
    ran: list[str] = []

    def broken() -> None:
        raise RuntimeError("database unavailable")

    async def healthy() -> str:
        ran.append("healthy")
        return "ok"

    scheduler = Scheduler(
        [ScheduledSweep("broken", 60, broken), ScheduledSweep("healthy", 60, healthy)]
    )
    results = await scheduler.run_once()

    assert isinstance(results["broken"], RuntimeError)
    assert results["healthy"] == "ok"
    assert ran == ["healthy"]
