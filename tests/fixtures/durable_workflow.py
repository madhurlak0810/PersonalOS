"""Durable-workflow test fixtures, plus a worker process that can be killed.

Everything here exists to support one kind of test: run a real workflow on a
real database-backed checkpointer, kill the process that is running it, and see
what a restarted process does with what is left behind.

That needs two things the ordinary fakes in `job_search_fakes` cannot give:

**Ports whose calls outlive the process.** A fake that records calls on itself
records nothing a killed process can be asked about. Every port here appends a
line to a shared log file instead, so the parent test can read what the dead
child actually did -- and, because the resumed run logs to the same file, can
tell "searched the job boards once" from "searched them again on resume".

**A real process to kill.** `main` runs one workflow to completion, or until it
reaches the step it was told to die at, at which point it `SIGKILL`s itself. A
hard kill and not an exception, deliberately: an exception unwinds, runs
`finally` blocks and flushes buffers, which is exactly the graceful path that a
crash does *not* take, and testing against it would prove nothing about
durability.

Run as `python -m tests.fixtures.durable_workflow <db> <log> <thread-id> [mode]
[decision-file]`.
"""

import json
import os
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.domain.job_search import (
    ActionIntent,
    ActionReceipt,
    ApprovalDecision,
    RawPosting,
    SearchProfile,
)
from personalos.persistence.action_journal import JournaledActionExecutor
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.models import Base
from tests.fixtures import job_search_fakes as fakes

#: The workflow definition every fixture here registers threads under, matching
#: `personalos.bootstrap.JOB_SEARCH_WORKFLOW`.
WORKFLOW_NAME = "job_search"

#: Log event names. Constants because the assertions and the ports both spell
#: them, and a typo in either would silently make a test vacuous.
EVENT_PROVIDER_SEARCH = "provider.search"
EVENT_APPROVAL_REVIEW = "approval.review"
EVENT_EXTERNAL_SUBMISSION = "external.submission"
EVENT_APPLICATION_PERSISTED = "application.persisted"
EVENT_RUN_FINISHED = "run.finished"

#: Values `kill_at` may take: the point in the pipeline at which the worker dies.
#: `approval` dies before any side effect (the "between shortlist and approval"
#: case); `after_submission` dies *after* the external write but before the
#: receipt is recorded, which is the window a resume must not reopen.
KILL_AT_APPROVAL = "approval"
KILL_AT_AFTER_SUBMISSION = "after_submission"

#: Modes that exercise the approval *interrupt* rather than a crash. The worker
#: runs with a gate that has nothing on file, so the graph parks at the
#: interrupt and the process exits normally -- which is the situation a real
#: deployment is in for most of an approval's life, and the one a restart has to
#: survive.
MODE_AWAIT_APPROVAL = "await_approval"
#: Resume a parked workflow with a decision supplied on disk by another process.
MODE_RESUME_APPROVAL = "resume_approval"

#: Modes in which the worker never takes a side effect on its own initiative.
_INTERRUPT_MODES = (MODE_AWAIT_APPROVAL, MODE_RESUME_APPROVAL)

#: Recorded when a run ends parked at the approval interrupt rather than
#: finishing, so a test spanning two processes can tell the two apart.
EVENT_RUN_PAUSED = "run.paused"


# --- Recording ---------------------------------------------------------------


class EventLog:
    """An append-only JSON-lines log, shared across processes.

    Opened, written, flushed and `fsync`ed per event rather than kept open: the
    writer is a process that will be `SIGKILL`ed without warning, and a buffered
    line is a line the test never sees.
    """

    def __init__(self, path: str | Path):
        """Initialize against a log file path, which need not exist yet."""
        self.path = Path(path)

    def append(self, event: str, **fields: Any) -> None:
        """Record one event, durably enough to survive a hard kill."""
        line = json.dumps({"event": event, "pid": os.getpid(), **fields}, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def events(self) -> list[dict[str, Any]]:
        """Every recorded event, in order."""
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def names(self) -> list[str]:
        """Just the event names, in order."""
        return [event["event"] for event in self.events()]

    def count(self, event: str) -> int:
        """How many times an event was recorded, across every process."""
        return self.names().count(event)


def _die() -> None:
    """End this process the way a crash does: no unwinding, no flushing.

    `SIGKILL` to self rather than `sys.exit` or `os._exit`: it is the signal a
    container runtime, an OOM killer or an impatient operator actually sends, and
    it is the only one Python cannot intercept.
    """
    os.kill(os.getpid(), signal.SIGKILL)


class RecordingProvider(fakes.FakeProvider):
    """A job board that records every search in the shared log.

    The witness for "did the resume restart from scratch?": a resumed run that
    searched the boards again has thrown away the work it was supposed to have
    kept.
    """

    def __init__(self, log: EventLog, **kwargs: Any):
        super().__init__(**kwargs)
        self.log = log

    async def search(self, search_profile: SearchProfile) -> Sequence[RawPosting]:
        self.log.append(EVENT_PROVIDER_SEARCH, provider=self.name)
        return await super().search(search_profile)


class RecordingApprovalGate(fakes.FakeApprovalGate):
    """An approval gate that records each review, and can die at or dawdle over one.

    `delay` is what makes the concurrent-resume test deterministic rather than
    lucky: the worker that wins the lease stays inside the graph for measurably
    longer than the loser takes to ask for it, so the two resumes genuinely
    overlap instead of accidentally queueing.
    """

    def __init__(self, log: EventLog, *, die: bool = False, delay: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.log = log
        self.die = die
        self.delay = delay

    async def review(self, intent: ActionIntent) -> ApprovalDecision:
        self.log.append(EVENT_APPROVAL_REVIEW, kind=intent.kind.value)
        if self.die:
            _die()
        if self.delay:
            import asyncio

            await asyncio.sleep(self.delay)
        return await super().review(intent)


class RecordingInterruptOnlyGate(fakes.NoStandingApprovalGate):
    """A gate with no decision on file, recording each action it was asked about.

    Every review it logs is an action that then parked the run: this is the
    witness for "the worker reached the approval point and stopped there
    without acting".
    """

    def __init__(self, log: EventLog):
        super().__init__()
        self.log = log

    async def review(self, intent: ActionIntent) -> ApprovalDecision:
        self.log.append(EVENT_APPROVAL_REVIEW, kind=intent.kind.value)
        return await super().review(intent)


class RecordingActionExecutor(fakes.FakeActionExecutor):
    """An action executor whose 'external write' is a log line, and can die after it.

    The log line stands in for the side effect that cannot be taken back -- the
    application actually submitted to the company. Dying immediately after it,
    before the caller can record a receipt, is the crash the action journal
    exists for, and counting these lines across both processes is how a test
    proves the submission happened exactly once.
    """

    def __init__(self, log: EventLog, *, die_after: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.log = log
        self.die_after = die_after

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        receipt = await super().execute(intent, decision)
        self.log.append(
            EVENT_EXTERNAL_SUBMISSION,
            kind=intent.kind.value,
            idempotency_key=intent.idempotency_key,
        )
        if self.die_after:
            _die()
        return receipt


class RecordingApplicationStore(fakes.FakeApplicationStore):
    """An application store that records each persisted application."""

    def __init__(self, log: EventLog, **kwargs: Any):
        super().__init__(**kwargs)
        self.log = log

    async def create(self, **kwargs: Any):
        application = await super().create(**kwargs)
        self.log.append(
            EVENT_APPLICATION_PERSISTED,
            status=application.status.value,
            submitted=application.submitted,
        )
        return application


# --- Wiring ------------------------------------------------------------------


def session_factory(db_path: str | Path, *, create: bool = True) -> Callable[[], Any]:
    """A session factory over a file-backed SQLite database.

    A *new* engine each call, which is the point: a test that reuses the engine
    that wrote the data is not testing a restart, it is testing an identity map.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    if create:
        Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def build_graph(
    factory: Callable[[], Any],
    log: EventLog,
    *,
    kill_at: str | None = None,
    journal: bool = True,
    registry: WorkflowThreadRegistry | None = None,
    workflow_id: UUID | None = None,
    approval_delay: float = 0.0,
    mode: str | None = None,
    checkpoint_scheduler: Any = None,
    recruiter_inbox: Any = None,
    recruiter_classifier: Any = None,
):
    """Compile the Job Search subgraph on a durable checkpointer, with logging ports.

    Returns `(compiled_graph, ports)`. `ports` carries the concrete fakes so a
    single-process test can assert against them directly; a test spanning a kill
    reads `log` instead, because the ports died with their process.

    `mode` decides which approval gate is wired. The crash tests want a gate
    with a standing `APPROVED` so the run reaches the side effect they are about
    to interrupt; the approval-interrupt tests want one with nothing on file, so
    the run parks at `interrupt()` instead.

    `checkpoint_scheduler` and the recruiter pair are left unwired unless a test
    passes them: the pending-checkpoint tests need the recruiter branch to run
    (it is what schedules a durable wait), and the crash tests need it not to.
    """
    from personalos.graphs.job_search import JobSearchGraph

    registry = registry or WorkflowThreadRegistry(factory)
    executor: Any = RecordingActionExecutor(log, die_after=(kill_at == KILL_AT_AFTER_SUBMISSION))
    if journal:
        executor = JournaledActionExecutor(executor, factory, workflow_id=workflow_id)

    gate: Any = (
        RecordingInterruptOnlyGate(log)
        if mode in _INTERRUPT_MODES
        else RecordingApprovalGate(log, die=(kill_at == KILL_AT_APPROVAL), delay=approval_delay)
    )

    ports: dict[str, Any] = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [RecordingProvider(log)],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": gate,
        "action_executor": executor,
        "application_store": RecordingApplicationStore(log),
        "event_emitter": fakes.FakeEventEmitter(),
        "checkpointer": SqlAlchemyCheckpointSaver(factory, registry),
    }
    # Off unless a test asks for them, so the pipeline tests keep the shape
    # they were written against: no recruiter branch, and no durable waits
    # scheduled behind their backs.
    if checkpoint_scheduler is not None:
        ports["checkpoint_scheduler"] = checkpoint_scheduler
    if recruiter_inbox is not None:
        ports["recruiter_inbox"] = recruiter_inbox
        ports["recruiter_classifier"] = recruiter_classifier
    return JobSearchGraph(**ports).build(), ports


def initial_state() -> dict[str, Any]:
    """The input a job search run starts from, with the application step on."""
    return {"user_id": str(fakes.USER_ID), "prepare_application": True}


# --- Worker entry point ------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    """Run one workflow on a durable checkpointer, optionally dying or pausing part way.

    `mode` is either a `KILL_AT_*` value (die at that step) or a `MODE_*` value
    (park at the approval interrupt, or resume a parked run with a decision read
    from `decision-file`).

    Deliberately does not catch the graph's exceptions: a worker that swallowed
    them would leave a database state no real crash produces.
    """
    import asyncio

    if len(argv) < 3:
        raise SystemExit(
            "usage: python -m tests.fixtures.durable_workflow <db> <log> <thread-id> "
            "[mode] [decision-file]"
        )
    db_path, log_path, thread_id = argv[0], argv[1], argv[2]
    mode = argv[3] if len(argv) > 3 else None
    decision_path = argv[4] if len(argv) > 4 else None

    log = EventLog(log_path)
    factory = session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)
    # Idempotent: the parent already registered this thread, and re-registering
    # it from a second process must rejoin that run rather than fork a new one.
    thread = registry.register(
        thread_id=thread_id, workflow_name=WORKFLOW_NAME, user_id=fakes.USER_ID
    )
    graph, _ports = build_graph(
        factory,
        log,
        kill_at=None if mode in _INTERRUPT_MODES else mode,
        registry=registry,
        workflow_id=thread.workflow_id,
        mode=mode,
    )

    async def run() -> None:
        from apps.worker.workflow_runner import DurableWorkflowRunner
        from personalos.persistence.leases import WorkflowLeaseStore

        runner = DurableWorkflowRunner(
            graph,
            registry=registry,
            leases=WorkflowLeaseStore(factory),
            owner=f"worker:{os.getpid()}",
        )

        if mode == MODE_RESUME_APPROVAL:
            # The decision was minted by another process entirely, from the
            # request this one reads back out of the checkpoint. That is the
            # whole shape of a days-later approval: the answer arrives from
            # somewhere the original run no longer exists.
            from langgraph.types import Command

            decisions = json.loads(Path(decision_path).read_text(encoding="utf-8"))
            await runner.resume(
                thread_id=thread_id, resume_input=Command(resume=decisions)
            )
        else:
            await runner.start(thread, initial_state())

        # Parked and finished are different outcomes, and a test spanning two
        # processes can only tell them apart if the worker says which happened.
        state = await runner.inspect(thread)
        log.append(EVENT_RUN_PAUSED if state.next else EVENT_RUN_FINISHED)

    asyncio.run(run())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main(sys.argv[1:]))
