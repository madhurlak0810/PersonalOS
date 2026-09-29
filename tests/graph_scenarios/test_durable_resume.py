"""Integration tests for durable checkpointing and resume by workflow/thread id.

The two acceptance criteria this file exists for:

1. **A killed worker resumes from the correct step.** `test_a_killed_worker_...`
   spawns a real subprocess, lets it run a real Job Search workflow against a
   real database-backed checkpointer, and `SIGKILL`s it at the approval step. A
   restarted worker then resumes, and the test asserts both that it picked up
   *at* that step and that it did not redo the work in front of it.
2. **Two concurrent resumes of the same workflow cannot both proceed.**
   `test_two_concurrent_resumes_...` runs two resumes at once, in two threads
   with separate engines, sessions and compiled graphs, and asserts exactly one
   ran while the other was refused the lease.

Two properties are tested with a subprocess rather than simulated in-process,
because simulating them would test something else:

- A `SIGKILL`ed process does not unwind, does not run `finally` blocks and does
  not flush buffers. An exception raised inside a node does all three. Only the
  first is a crash.
- A restart has to see the state through a *new* engine, session and identity
  map. Every restart here builds its own, so nothing survives in memory that the
  database did not actually store.

Everything is on file-backed SQLite. That is not the production dialect, and the
lease deliberately does not depend on one: see `personalos.persistence.leases`
for why its guarantee rests on a unique constraint and a guarded `UPDATE` rather
than on `SELECT ... FOR UPDATE`, which SQLite ignores.
"""

import asyncio
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest

from apps.worker.workflow_runner import (
    AmbiguousWorkflowResume,
    DurableWorkflowRunner,
    NothingToResume,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.workflow import job_search_thread_id
from personalos.persistence.checkpointer import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import (
    LEASE_EXCLUSION_REASONS,
    LEASE_REASON_HELD,
    WorkflowLeaseStore,
    WorkflowLeaseUnavailable,
)
from personalos.persistence.models import CheckpointModel, WorkflowRunModel
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A clock far enough ahead that any lease taken with the default TTL has
#: expired. Stands in for the wall-clock gap between a worker being killed and
#: an operator noticing, without a test that sleeps for it.
_AFTER_LEASE_EXPIRY = timedelta(hours=1)


def _clock_after_expiry():
    return datetime.utcnow() + _AFTER_LEASE_EXPIRY


class Worker:
    """One simulated worker process: its own engine, registry, graph and runner.

    A class rather than a fixture because a restart test needs several of them
    over the same database, and the whole point is that each is built from
    scratch: sharing an engine between "the worker that died" and "the worker
    that took over" would let SQLAlchemy's identity map stand in for the
    durability being tested.
    """

    def __init__(
        self,
        db_path: Path,
        log: durable.EventLog,
        *,
        kill_at: str | None = None,
        journal: bool = True,
        lease_clock=datetime.utcnow,
        owner: str = "worker",
        approval_delay: float = 0.0,
    ):
        self.factory = durable.session_factory(db_path)
        self.registry = WorkflowThreadRegistry(self.factory)
        self.leases = WorkflowLeaseStore(self.factory, clock=lease_clock)
        self.graph, self.ports = durable.build_graph(
            self.factory,
            log,
            kill_at=kill_at,
            journal=journal,
            registry=self.registry,
            approval_delay=approval_delay,
        )
        self.runner = DurableWorkflowRunner(
            self.graph, registry=self.registry, leases=self.leases, owner=owner
        )

    def register(self, thread_id: str, *, workflow_id=None):
        """Bind a thread to the job search workflow, as the composition root does."""
        return self.registry.register(
            thread_id=thread_id,
            workflow_name=durable.WORKFLOW_NAME,
            workflow_id=workflow_id,
            user_id=fakes.USER_ID,
        )


def _run_worker_subprocess(db_path: Path, log_path: Path, thread_id: str, kill_at: str):
    """Run the workflow in a child process that kills itself at `kill_at`."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.fixtures.durable_workflow",
            str(db_path),
            str(log_path),
            thread_id,
            kill_at,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


# --- Acceptance: kill and restart --------------------------------------------


def test_a_killed_worker_resumes_at_the_step_it_died_on_not_from_scratch(tmp_path):
    """A `SIGKILL`ed worker's workflow resumes at the approval step.

    The acceptance criterion, end to end: a real child process runs the pipeline
    as far as the approval checkpoint -- past discovery, scoring and the
    shortlist -- and is killed there without unwinding. A fresh worker then
    resumes the same `workflow_id` and the test asserts three things:

    - the stored checkpoint says the next step is the approval checkpoint, so the
      resume starts *there* rather than at the beginning;
    - the job boards are searched exactly once across both processes, so the work
      in front of the crash was restored rather than recomputed;
    - the run finishes as an `APPLIED` application, so nothing was lost in the
      middle.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "kill-restart")

    # A worker registers the thread before the child runs it, exactly as the
    # composition root does: the durable checkpointer refuses to checkpoint a
    # thread that is not bound to a workflow.
    operator = Worker(db_path, log, lease_clock=_clock_after_expiry)
    thread = operator.register(thread_id)

    killed = _run_worker_subprocess(db_path, log.path, thread_id, durable.KILL_AT_APPROVAL)

    # -9 is SIGKILL: the child really died rather than exiting or raising.
    assert killed.returncode == -9, (
        f"expected the worker to be killed, got returncode {killed.returncode}\n"
        f"stdout: {killed.stdout}\nstderr: {killed.stderr}"
    )
    assert durable.EVENT_RUN_FINISHED not in log.names()
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    assert log.count(durable.EVENT_APPROVAL_REVIEW) == 1

    # What the dead worker left behind, read by a worker that never shared a
    # connection with it.
    restarted = Worker(db_path, log, lease_clock=_clock_after_expiry, owner="worker-2")
    state = asyncio.run(restarted.runner.inspect(thread))

    assert state.is_resumable
    assert state.next == ("approval_checkpoint_for_external_submission",)
    # The work in front of the crash is in the checkpoint, not lost.
    assert len(state.values["shortlist"]) == 1
    assert state.values["application_packet"] is not None

    final = asyncio.run(restarted.runner.resume(workflow_id=thread.workflow_id))

    assert final["application"]["status"] == ApplicationStatus.APPLIED.value
    # The whole point: discovery ran once, in the process that died.
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    assert log.count(durable.EVENT_APPROVAL_REVIEW) == 2
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1

    # And the run is recorded as finished, so it is not offered for resume again.
    session = restarted.factory()
    try:
        run = session.query(WorkflowRunModel).filter(WorkflowRunModel.thread_id == thread_id).one()
        assert run.status == RUN_STATUS_COMPLETED
        assert run.workflow_id == thread.workflow_id
        checkpoints = (
            session.query(CheckpointModel)
            .filter(CheckpointModel.workflow_id == thread.workflow_id)
            .count()
        )
        assert checkpoints > 1, "a resumable run should have left a checkpoint per step"
    finally:
        session.close()


def test_a_crash_after_the_external_write_does_not_submit_a_second_time(tmp_path):
    """A worker killed between the submission and its receipt does not resubmit.

    The window the action journal exists for. The child submits -- the log line
    stands in for the application the company has actually received -- and dies
    before the receipt is recorded, so the checkpoint says the submission has not
    happened. The resumed run re-reaches the same action and must *not* act on
    that, because the journal's claim outlived the process that made it.

    The application is then persisted as `READY_TO_APPLY` with a receipt that
    says why: the outcome of the first attempt is genuinely unknown, and treating
    unknown as success would be as wrong as resubmitting.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "crash-after-submit")

    operator = Worker(db_path, log, lease_clock=_clock_after_expiry)
    thread = operator.register(thread_id)

    killed = _run_worker_subprocess(db_path, log.path, thread_id, durable.KILL_AT_AFTER_SUBMISSION)

    assert killed.returncode == -9, killed.stderr
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1
    assert durable.EVENT_APPLICATION_PERSISTED not in log.names()

    restarted = Worker(db_path, log, lease_clock=_clock_after_expiry, owner="worker-2")
    final = asyncio.run(restarted.runner.resume(workflow_id=thread.workflow_id))

    # Exactly one submission, across both processes. This is the requirement.
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1

    receipts = final["action_receipts"]
    assert [receipt["ok"] for receipt in receipts] == [False]
    assert "did not record an outcome" in receipts[0]["detail"]
    assert final["application"]["status"] == ApplicationStatus.READY_TO_APPLY.value


def test_the_lease_a_killed_worker_never_released_blocks_a_resume_until_it_expires(tmp_path):
    """A hard-killed worker's lease still holds, and then expires.

    Both halves matter. While the lease is held, a resume is refused -- which is
    the same protection as the concurrent-resume case, arriving via the most
    realistic route to it. Once it expires, the workflow becomes resumable again,
    because a lease that only its (dead) holder could release would strand the
    workflow forever.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "orphaned-lease")

    operator = Worker(db_path, log, lease_clock=_clock_after_expiry)
    thread = operator.register(thread_id)
    killed = _run_worker_subprocess(db_path, log.path, thread_id, durable.KILL_AT_APPROVAL)
    assert killed.returncode == -9, killed.stderr

    # A worker on the real clock sees the dead worker's lease as still held.
    impatient = Worker(db_path, log, owner="worker-too-soon")
    with pytest.raises(WorkflowLeaseUnavailable, match="is leased by"):
        asyncio.run(impatient.runner.resume(workflow_id=thread.workflow_id))

    # A worker arriving after the TTL takes it over and proceeds.
    patient = Worker(db_path, log, lease_clock=_clock_after_expiry, owner="worker-later")
    final = asyncio.run(patient.runner.resume(workflow_id=thread.workflow_id))
    assert final["application"]["status"] == ApplicationStatus.APPLIED.value


# --- Acceptance: concurrent resumes ------------------------------------------


def test_two_concurrent_resumes_of_one_workflow_cannot_both_proceed(tmp_path):
    """Exactly one of two simultaneous resumes runs; the other is refused.

    The second acceptance criterion. Two workers, each with its own engine,
    session, compiled graph and lease store, resume the same `workflow_id` at the
    same moment -- released together by a `Barrier` so neither can be first by
    accident. One acquires the lease and completes the workflow; the other raises
    `WorkflowLeaseUnavailable` and does nothing.

    The assertion is not only on the exceptions. The approval gate is reviewed
    exactly once more than the dead child reviewed it, which is what rules out
    the failure this prevents: two resumes that *both* proceeded would each run
    the approval step, and the second one's submission would be a duplicate no
    error message could undo.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "concurrent")

    operator = Worker(db_path, log, lease_clock=_clock_after_expiry)
    thread = operator.register(thread_id)
    killed = _run_worker_subprocess(db_path, log.path, thread_id, durable.KILL_AT_APPROVAL)
    assert killed.returncode == -9, killed.stderr

    reviews_before = log.count(durable.EVENT_APPROVAL_REVIEW)
    barrier = Barrier(2)
    results: dict[str, object] = {}

    def resume(name: str) -> None:
        # The winner dawdles inside the approval step, so the loser's request for
        # the lease definitely lands while the lease is genuinely held rather
        # than after it was released.
        worker = Worker(
            db_path,
            log,
            lease_clock=_clock_after_expiry,
            owner=name,
            approval_delay=0.25,
        )
        barrier.wait(timeout=30)
        try:
            results[name] = asyncio.run(worker.runner.resume(workflow_id=thread.workflow_id))
        except Exception as exc:  # recorded, then asserted on below
            results[name] = exc

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(resume, ["worker-a", "worker-b"]))

    refused = [value for value in results.values() if isinstance(value, Exception)]
    proceeded = [value for value in results.values() if not isinstance(value, Exception)]

    assert len(proceeded) == 1, f"both resumes proceeded: {results}"
    assert len(refused) == 1
    assert isinstance(refused[0], WorkflowLeaseUnavailable), refused[0]
    # Refused *by the lease*, not by the database happening to be busy: SQLite
    # serializes writers, so a test that accepted any refusal here would pass
    # even with the exclusion rule removed.
    assert refused[0].details["reason"] in LEASE_EXCLUSION_REASONS, refused[0].details

    assert proceeded[0]["application"]["status"] == ApplicationStatus.APPLIED.value
    # The decisive assertion: the step was run once, not twice.
    assert log.count(durable.EVENT_APPROVAL_REVIEW) == reviews_before + 1
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1


def test_a_resume_is_refused_outright_while_another_worker_holds_the_lease(tmp_path):
    """The exclusion rule itself, with no race to win.

    The threaded test above proves two *simultaneous* resumes cannot both
    proceed; this proves why, without depending on thread scheduling for it. The
    lease is taken by hand -- standing in for a worker that is part way through a
    resume -- and a second resume is refused for exactly that reason, then
    succeeds once the lease is given up.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "held-lease")

    operator = Worker(db_path, log, lease_clock=_clock_after_expiry)
    thread = operator.register(thread_id)
    killed = _run_worker_subprocess(db_path, log.path, thread_id, durable.KILL_AT_APPROVAL)
    assert killed.returncode == -9, killed.stderr

    other = Worker(db_path, log, lease_clock=_clock_after_expiry, owner="worker-holding")
    lease = other.leases.acquire(thread.workflow_id, owner="worker-holding")

    second = Worker(db_path, log, lease_clock=_clock_after_expiry, owner="worker-waiting")
    with pytest.raises(WorkflowLeaseUnavailable) as refusal:
        asyncio.run(second.runner.resume(workflow_id=thread.workflow_id))
    assert refusal.value.details == {
        "reason": LEASE_REASON_HELD,
        "holder": "worker-holding",
    }
    # Refused before running anything: the approval step was not re-entered.
    assert log.count(durable.EVENT_APPROVAL_REVIEW) == 1

    assert other.leases.release(lease) is True
    final = asyncio.run(second.runner.resume(workflow_id=thread.workflow_id))
    assert final["application"]["status"] == ApplicationStatus.APPLIED.value


def test_a_second_resume_after_the_first_finished_is_a_no_op(tmp_path):
    """Resuming a completed workflow returns its state instead of re-running it.

    The sequential counterpart to the concurrency test: once the workflow has
    run out of steps, a resume has nothing to advance, and inventing something
    for it to do would re-submit an application that is already in.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "already-done")

    worker = Worker(db_path, log)
    thread = worker.register(thread_id)
    asyncio.run(worker.runner.start(thread, durable.initial_state()))
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1

    second = Worker(db_path, log, owner="worker-2")
    values = asyncio.run(second.runner.resume(thread_id=thread_id))

    assert values["application"]["status"] == ApplicationStatus.APPLIED.value
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1


# --- Identity ----------------------------------------------------------------


def test_the_same_stable_thread_id_rejoins_its_run_across_processes(tmp_path):
    """Re-registering a derived thread id continues its run rather than forking one.

    What makes a thread id worth deriving: the worker that restarts has only the
    facts (this candidate, this search), recomputes the same id, and lands on the
    same `workflow_runs` row -- so its checkpoints join the ones already there.
    A registration that minted a second run would leave the first one's state
    stranded under a workflow nothing points at any more.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "rejoin")

    first = Worker(db_path, log)
    thread_a = first.register(thread_id)

    second = Worker(db_path, log, owner="worker-2")
    thread_b = second.register(thread_id)

    assert thread_b.workflow_run_id == thread_a.workflow_run_id
    assert thread_b.workflow_id == thread_a.workflow_id

    session = second.factory()
    try:
        assert (
            session.query(WorkflowRunModel).filter(WorkflowRunModel.thread_id == thread_id).count()
            == 1
        )
    finally:
        session.close()


def test_a_conversation_and_its_domain_run_are_separate_threads_of_one_workflow(tmp_path):
    """The Supervisor and the Job Search subgraph checkpoint separately, under one workflow.

    Both graphs run on the same durable checkpointer, each on its own
    `thread_id`, both registered against the same `workflow_id`. That is the
    shape the requirement asks for -- a thread per conversation, a workflow per
    business process -- and it is checked by reading the stored checkpoints back
    and finding both threads under the one workflow.
    """
    from personalos.bootstrap import (
        build_job_search_subgraph_runner,
        register_job_search_thread,
        register_supervisor_thread,
    )
    from personalos.domain.routing import RouteDecision, RouteDomain
    from personalos.graphs.supervisor import SupervisorGraph
    from personalos.persistence.checkpointer import SqlAlchemyCheckpointSaver

    class StubClassifier:
        def classify(self, message):
            return RouteDecision(domain=RouteDomain.JOB, confidence=0.95)

    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    factory = durable.session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)

    job_thread = register_job_search_thread(
        user_id=fakes.USER_ID, registry=registry, search_key="supervised"
    )
    supervisor_thread = register_supervisor_thread(
        conversation_key="conversation-1",
        registry=registry,
        workflow_id=job_thread.workflow_id,
        user_id=fakes.USER_ID,
    )

    subgraph, _ports = durable.build_graph(factory, log, registry=registry)
    supervisor = SupervisorGraph(
        StubClassifier(),
        build_job_search_subgraph_runner(
            subgraph,
            user_id=fakes.USER_ID,
            thread=job_thread,
            prepare_application=True,
        ),
        checkpointer=SqlAlchemyCheckpointSaver(factory, registry),
    ).build()

    final = asyncio.run(
        supervisor.ainvoke({"message": "find me a python job"}, config=supervisor_thread.config())
    )

    assert final["result"]["application"]["status"] == ApplicationStatus.APPLIED.value

    session = factory()
    try:
        threads = {
            row.thread_id
            for row in session.query(CheckpointModel)
            .filter(CheckpointModel.workflow_id == job_thread.workflow_id)
            .all()
        }
        assert threads == {supervisor_thread.thread_id, job_thread.thread_id}
    finally:
        session.close()

    # Resuming by workflow_id is refused while it is ambiguous, rather than
    # advancing whichever of the two threads happened to be found first.
    runner = DurableWorkflowRunner(
        supervisor,
        registry=registry,
        leases=WorkflowLeaseStore(factory),
        owner="operator",
    )
    with pytest.raises(AmbiguousWorkflowResume, match="has 2 threads"):
        asyncio.run(runner.resume(workflow_id=job_thread.workflow_id))


def test_resuming_a_thread_that_was_never_run_is_refused(tmp_path):
    """A registered-but-never-started thread has to be started, not resumed."""
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    worker = Worker(db_path, log)
    thread_id = job_search_thread_id(fakes.USER_ID, "never-run")
    worker.register(thread_id)

    with pytest.raises(NothingToResume, match="has no checkpoint"):
        asyncio.run(worker.runner.resume(thread_id=thread_id))


def test_resuming_an_unregistered_thread_is_refused(tmp_path):
    """A thread nothing bound to a workflow cannot be resumed by id either."""
    from personalos.persistence.checkpointer import UnregisteredWorkflowThread

    worker = Worker(tmp_path / "durable.db", durable.EventLog(tmp_path / "events.jsonl"))

    with pytest.raises(UnregisteredWorkflowThread, match="not bound to a workflow"):
        asyncio.run(worker.runner.resume(thread_id=f"never-registered-{uuid4()}"))


def test_a_run_in_flight_is_visible_as_running_without_reading_a_checkpoint(tmp_path):
    """The killed worker's run is left marked `running`, which is how it is found.

    An operator looking for workflows to resume should not have to deserialize
    every checkpoint in the database to find the stalled ones. The status column
    answers it, and a worker killed mid-run leaves it at `running` -- there is no
    `finally` on a `SIGKILL` to set it to anything else.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "status")

    operator = Worker(db_path, log)
    operator.register(thread_id)
    killed = _run_worker_subprocess(db_path, log.path, thread_id, durable.KILL_AT_APPROVAL)
    assert killed.returncode == -9, killed.stderr

    session = Worker(db_path, log).factory()
    try:
        run = session.query(WorkflowRunModel).filter(WorkflowRunModel.thread_id == thread_id).one()
        assert run.status == RUN_STATUS_RUNNING
        assert run.started_at is not None
        assert run.completed_at is None
    finally:
        session.close()
