"""The worker process: where every long-running step actually runs.

The API answers in milliseconds because it does nothing slow. `POST /v1/chat`,
`POST /v1/workflows/{id}/resume`, an approval decision and a retry each write a
row and return `202 Accepted`; job discovery, scoring, artifact drafting and
mailbox reads all happen inside a graph run, and graph runs happen here.
Progress reaches the caller through the workflow's progress log, which
`GET /v1/workflows/{id}/events` streams from any API process.

One worker runs three loops side by side, all against the same database:

- **Workflow commands** (`apps.worker.workflow_commands.run_commands`): claim a
  queued start, resume or retry and run it under the workflow's lease.
- **Outbox events** (`apps.worker.outbox_dispatcher.OutboxDispatcher.run`):
  deliver each event to its consumer, which may queue a workflow command.
- **Scheduled sweeps** (`apps.worker.scheduler.Scheduler`): fire follow-up
  checkpoints that came due, stall quiet applications, expire unanswered
  approvals.

**This process remembers nothing a restart would lose.** A claimed command is a
row; a run's progress is its checkpoints; a wait is a `pending_checkpoints` row
with a `trigger_at`. Kill the worker anywhere and start another: the claim goes
stale and is reclaimed, the reclaiming worker sees the run's own checkpoint and
continues from it, and the first sweep finds every wait that came due in the
meantime. Any number of workers may run at once; claims, leases and guarded
writes keep each piece of work to one of them.

Which graphs a worker runs is a deployment decision, made by a factory named in
`settings.worker_graphs` (`module:callable`, taking a session factory and
returning `WorkerGraphs`). A worker started without one still delivers events
and runs the approval and stall sweeps, and leaves workflow commands queued --
where `GET /v1/workflows/{id}` reports them -- for a worker that can run them.
"""

import asyncio
import importlib
import logging
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from langgraph.config import get_config

from apps.worker.checkpoint_monitor import CheckpointConditionEvaluator, PendingCheckpointMonitor
from apps.worker.outbox_dispatcher import OutboxDispatcher, build_outbox_dispatcher
from apps.worker.scheduler import ScheduledSweep, Scheduler
from apps.worker.stall_monitor import StalledApplicationMonitor
from apps.worker.workflow_commands import process_next, run_commands
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    build_application_lifecycle_store,
    build_approval_queue,
    build_checkpoint_condition_evaluator,
    build_job_search_subgraph_runner,
    build_pending_checkpoint_store,
    build_worker_command_queue,
    build_workflow_thread_registry,
    register_job_search_thread,
    register_mcp_servers,
)
from personalos.config import settings
from personalos.domain.errors import ValidationFailed
from personalos.domain.workflow import JOB_SEARCH_THREAD_NAMESPACE
from personalos.persistence.checkpointer import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    WorkflowThreadRegistry,
)
from personalos.persistence.database import SessionLocal
from personalos.persistence.leases import DEFAULT_LEASE_TTL_SECONDS
from personalos.persistence.workflow_commands import WorkflowCommandQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerGraphs:
    """The graphs one worker can run, as runners keyed by thread kind.

    `runners` maps a thread id's namespace -- `supervisor`, `job_search`, ... --
    to the runner for the graph threads of that kind run on, exactly as
    `apps.worker.workflow_commands.process_next` looks them up. The job search
    runner is also what a fired follow-up checkpoint starts its thread on.
    `evaluator` answers the conditions those checkpoints wait on; by default,
    from the database.
    """

    runners: Mapping[str, DurableWorkflowRunner] = field(default_factory=dict)
    evaluator: CheckpointConditionEvaluator | None = None

    @property
    def follow_ups(self) -> DurableWorkflowRunner | None:
        """The runner a fired follow-up checkpoint resumes its thread on."""
        return self.runners.get(JOB_SEARCH_THREAD_NAMESPACE)


#: Composes a worker's graphs over one database.
GraphFactory = Callable[[Callable[[], Any]], WorkerGraphs]


class WorkflowScopedJobSubgraph:
    """The Supervisor's job-search port, bound per run to that run's own workflow.

    `build_job_search_subgraph_runner` binds a runner to one thread, which fits
    a process that runs one conversation. A worker runs every conversation, so
    this resolves the thread at call time instead: the job search for a
    Supervisor run goes on a job-search thread of the *same* workflow, keyed by
    the workflow id, registered on first use and found again on every later
    call -- including the one a restarted worker makes when it re-runs the
    Supervisor step it died in, which is what lets that call continue the
    search rather than begin it again.

    The candidate is the one the Supervisor thread was registered for; a
    conversation started without one falls back to `default_user_id`, and with
    neither the request is refused rather than searched for nobody.
    """

    def __init__(
        self,
        compiled_subgraph: Any,
        registry: WorkflowThreadRegistry,
        *,
        default_user_id: UUID | None = None,
        prepare_application: bool = False,
    ):
        """Take the compiled job-search graph and the registry its threads are bound in."""
        self.subgraph = compiled_subgraph
        self.registry = registry
        self.default_user_id = default_user_id
        self.prepare_application = prepare_application

    async def __call__(self, task_dag: dict[str, Any], state: Any) -> dict[str, Any]:
        """Run the job search for the Supervisor run currently executing."""
        configurable = get_config().get("configurable") or {}
        supervisor_thread_id = configurable.get("thread_id")
        workflow_id = configurable.get("workflow_id")
        if not supervisor_thread_id or not workflow_id:
            raise ValidationFailed(
                "the job search port was called outside a registered Supervisor run"
            )
        workflow_id = UUID(str(workflow_id))
        user_id = self.registry.user_id_for(supervisor_thread_id) or self.default_user_id
        if user_id is None:
            raise ValidationFailed(
                f"workflow {workflow_id} names no candidate to search for; start the "
                f"conversation with a user_id"
            )
        correlation_id = configurable.get("correlation_id")
        thread = register_job_search_thread(
            user_id=user_id,
            registry=self.registry,
            workflow_id=workflow_id,
            search_key=str(workflow_id),
            actor_id=configurable.get("actor_id") or "system",
            correlation_id=UUID(str(correlation_id)) if correlation_id else None,
        )
        runner = build_job_search_subgraph_runner(
            self.subgraph,
            user_id=user_id,
            thread=thread,
            prepare_application=self.prepare_application,
        )
        # Marked as `DurableWorkflowRunner` marks the threads it runs, since
        # this one is run from inside the Supervisor's step rather than by a
        # runner: the workflow's status is read from every thread's row.
        self.registry.mark_status(thread.thread_id, RUN_STATUS_RUNNING, started=True)
        try:
            result = await runner(task_dag, state)
        except Exception:
            self.registry.mark_status(thread.thread_id, RUN_STATUS_FAILED, finished=True)
            raise
        stored = await self.subgraph.aget_state(thread.config())
        if not stored.next:
            self.registry.mark_status(thread.thread_id, RUN_STATUS_COMPLETED, finished=True)
        return result


class Worker:
    """Runs workflow commands, outbox deliveries and scheduled sweeps until stopped."""

    def __init__(
        self,
        *,
        commands: WorkflowCommandQueue,
        runners: Mapping[str, DurableWorkflowRunner],
        dispatcher: OutboxDispatcher | None = None,
        scheduler: Scheduler | None = None,
        idle_seconds: float = 1.0,
    ):
        """Wire the queue and runners, and optionally the dispatcher and scheduler."""
        self.commands = commands
        self.runners = dict(runners)
        self.dispatcher = dispatcher
        self.scheduler = scheduler
        self.idle_seconds = idle_seconds

    async def run(self, stop: asyncio.Event) -> None:
        """Run every loop this worker has until `stop` is set."""
        loops = []
        if self.runners:
            loops.append(
                run_commands(self.commands, self.runners, stop, idle_seconds=self.idle_seconds)
            )
        else:
            logger.warning(
                "this worker has no graphs wired (settings.worker_graphs is empty); "
                "workflow commands stay queued for a worker that can run them"
            )
        if self.dispatcher is not None:
            loops.append(self.dispatcher.run(stop, idle_seconds=self.idle_seconds))
        if self.scheduler is not None:
            loops.append(self.scheduler.run(stop))
        logger.info("worker started with %s loop(s)", len(loops))
        await asyncio.gather(*loops)
        logger.info("worker stopped")

    async def drain(self, limit: int | None = None) -> int:
        """Run queued commands until none is claimable (or `limit` ran). Returns how many."""
        ran = 0
        while limit is None or ran < limit:
            if not await process_next(self.commands, self.runners):
                break
            ran += 1
        return ran


def build_scheduler(
    session_factory: Callable[[], Any],
    graphs: WorkerGraphs,
    *,
    clock: Callable[[], datetime] | None = None,
) -> Scheduler:
    """The sweeps a worker runs: follow-up checkpoints, stalled applications, approval expiry.

    The follow-up sweep is left out when no job search graph is wired, since a
    checkpoint it fired would have nothing to run on; the checkpoint stays
    pending, and due, for a worker that has one.
    """
    clock_kwargs = {"clock": clock} if clock is not None else {}
    registry = build_workflow_thread_registry(session_factory)
    approvals = build_approval_queue(session_factory, registry)
    if clock is not None:
        approvals.clock = clock
    stalls = StalledApplicationMonitor(
        store=build_application_lifecycle_store(session_factory), **clock_kwargs
    )
    sweeps = [
        ScheduledSweep(
            "approval_expiry",
            settings.worker_approval_expiry_sweep_seconds,
            approvals.expire_overdue,
        ),
        ScheduledSweep(
            "stalled_applications",
            settings.application_stall_check_interval_seconds,
            stalls.sweep,
        ),
    ]
    if graphs.follow_ups is not None:
        store = build_pending_checkpoint_store(session_factory)
        if clock is not None:
            store.clock = clock
        monitor = PendingCheckpointMonitor(
            store=store,
            evaluator=graphs.evaluator or build_checkpoint_condition_evaluator(session_factory),
            runner=graphs.follow_ups,
            registry=registry,
            **clock_kwargs,
        )
        sweeps.insert(
            0,
            ScheduledSweep(
                "follow_up_checkpoints", settings.worker_checkpoint_sweep_seconds, monitor.sweep
            ),
        )
    return Scheduler(sweeps)


def build_worker(
    session_factory: Callable[[], Any] = SessionLocal,
    graphs: WorkerGraphs | None = None,
    *,
    lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    idle_seconds: float | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Worker:
    """A worker over one database, running `graphs`.

    `lease_ttl_seconds` must match the TTL the runners' lease stores were built
    with: it is how old a claim has to be before this worker treats the worker
    that made it as gone.
    """
    graphs = graphs or WorkerGraphs()
    return Worker(
        commands=build_worker_command_queue(session_factory, lease_ttl_seconds=lease_ttl_seconds),
        runners=graphs.runners,
        dispatcher=build_outbox_dispatcher(session_factory),
        scheduler=build_scheduler(session_factory, graphs, clock=clock),
        idle_seconds=settings.worker_idle_seconds if idle_seconds is None else idle_seconds,
    )


def load_graph_factory(spec: str) -> GraphFactory:
    """Import the `module:callable` a deployment named as its graph factory."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"worker graph factory must be 'module:callable', not '{spec}'")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError(f"worker graph factory '{spec}' is not callable")
    return factory


def main(graph_factory: str | None = None) -> None:
    """Run a worker until SIGINT or SIGTERM, finishing the step in hand first.

    A stop lets each loop finish what it is doing and then exit; a run in the
    middle of a step completes that step's checkpoint. A worker killed outright
    loses nothing either -- see the module docstring -- it just takes a lease
    TTL longer for another worker to pick its command up.
    """
    spec = graph_factory if graph_factory is not None else settings.worker_graphs
    register_mcp_servers()
    graphs = load_graph_factory(spec)(SessionLocal) if spec else WorkerGraphs()
    worker = build_worker(SessionLocal, graphs)

    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await worker.run(stop)

    asyncio.run(run())


__all__ = [
    "GraphFactory",
    "Worker",
    "WorkerGraphs",
    "WorkflowScopedJobSubgraph",
    "build_scheduler",
    "build_worker",
    "load_graph_factory",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - the process entry point
    logging.basicConfig(level=settings.log_level)
    main()
