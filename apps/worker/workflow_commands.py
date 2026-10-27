"""Runs the workflow commands the API queued: the worker half of the hand-off.

The API never runs a graph. `POST /v1/chat`, `POST /v1/workflows/{id}/resume`
and `POST /v1/workflows/{id}/retry` write a `WorkflowCommand` to `outbox_events` and return; this module is what
turns one of those rows into a `DurableWorkflowRunner` call, under the
workflow's lease. The polling loop that calls `process_next` belongs to the
worker process (Phase G); everything a single command needs is here so that
loop has nothing to decide but *when*.

The request's identity crosses the hand-off in the command, not in any process:
`actor_id` and `correlation_id` are written into the run's `configurable`, which
is where `SupervisorGraph._load_context` reads the actor from and what every
checkpoint written during the run is stamped with.

**A redelivered command continues; it does not start over.** The command's id
goes into the `configurable` too, so every checkpoint its run writes says which
command wrote it. A worker killed mid-run leaves its command unsettled, and the
queue hands it to the next worker once the claim is stale; that worker finds
the thread's latest checkpoint stamped with this very command and resumes from
it with no input -- the step that was running when the first worker died runs
again, and nothing before it does. Had the first worker died before writing
anything, there is no such checkpoint and the command simply runs as queued.
"""

import asyncio
import logging
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from langgraph.types import Command
from opentelemetry.trace import SpanKind

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.domain.workflow import WorkflowCommand, WorkflowCommandKind, thread_namespace
from personalos.observability import tracing
from personalos.observability.tracing import Attr
from personalos.persistence.leases import WorkflowLeaseUnavailable
from personalos.persistence.workflow_commands import WorkflowCommandQueue

logger = logging.getLogger(__name__)

#: The `configurable` key, and so the checkpoint metadata key, naming the
#: command a run is carrying out.
COMMAND_ID_KEY = "command_id"

#: How long a command whose workflow is leased elsewhere waits before it is
#: claimable again. Without it the loop that released it claims it straight back.
DEFAULT_LEASE_BUSY_DELAY = timedelta(seconds=5)


def command_configurable(command: WorkflowCommand) -> dict[str, Any]:
    """The `configurable` entries that carry a command's identity into its run."""
    return {
        "actor_id": command.actor_id,
        "correlation_id": str(command.correlation_id),
        COMMAND_ID_KEY: str(command.command_id),
    }


async def _already_under_way(command: WorkflowCommand, runner: DurableWorkflowRunner) -> bool:
    """True when the thread's latest checkpoint was written by this command's own run."""
    thread = runner.registry.resolve(command.thread_id)
    if thread is None:
        return False
    metadata = await runner.checkpoint_metadata(thread)
    return metadata.get(COMMAND_ID_KEY) == str(command.command_id)


async def run_workflow_command(
    command: WorkflowCommand, runner: DurableWorkflowRunner, *, redelivered: bool = False
) -> dict[str, Any]:
    """Carry out one command on the runner for the graph its thread runs.

    `redelivered` says an earlier claim of this command was never settled. If
    that earlier run got as far as a checkpoint, the run is continued from it
    rather than carried out again from the top.
    """
    configurable = command_configurable(command)
    logger.info(
        "running %s command %s on thread '%s' of workflow %s (actor_id=%s, correlation_id=%s)",
        command.kind.value,
        command.command_id,
        command.thread_id,
        command.workflow_id,
        command.actor_id,
        command.correlation_id,
    )
    if redelivered and await _already_under_way(command, runner):
        logger.info(
            "command %s was already under way on thread '%s' when its worker stopped; "
            "continuing from its last checkpoint",
            command.command_id,
            command.thread_id,
        )
        return await runner.resume(
            thread_id=command.thread_id, resume_input=None, configurable=configurable
        )
    if command.kind == WorkflowCommandKind.START:
        thread = runner.registry.require(command.thread_id)
        return await runner.start(thread, command.graph_input or {}, configurable=configurable)
    if command.kind == WorkflowCommandKind.RETRY:
        # No input at all: LangGraph loads the checkpoint the failed step
        # started from and re-runs only that step. Anything it had already
        # sent is settled by `ToolExecutor` from the operation's idempotency
        # key -- the stored receipt if it landed, reconciliation if unknown.
        return await runner.resume(
            thread_id=command.thread_id, resume_input=None, configurable=configurable
        )
    # An interrupt is answered with `Command(resume=...)`; passing the value
    # bare would be read as fresh input and restart the thread instead.
    return await runner.resume(
        thread_id=command.thread_id,
        resume_input=Command(resume=command.graph_input),
        configurable=configurable,
    )


async def process_next(
    queue: WorkflowCommandQueue,
    runners: Mapping[str, DurableWorkflowRunner],
    *,
    lease_busy_delay: timedelta = DEFAULT_LEASE_BUSY_DELAY,
) -> bool:
    """Claim and run the oldest queued command. False when there was none.

    `runners` maps a thread kind -- the namespace of its derived id,
    `supervisor` or `job_search` -- to the runner for the graph it runs on.
    Keyed by thread rather than by workflow because one workflow's Supervisor
    thread and Job Search thread are resumed on different graphs.

    A command whose workflow is leased by another worker is put back rather
    than failed: the other worker is mid-run on the same process, and this
    command is next in line, not wrong. It is held back `lease_busy_delay`
    so this worker does not spin on it. Any other failure settles the row as
    failed; the runner has already marked the thread failed, and its last
    checkpoint is what a later resume starts from.
    """
    claimed = queue.claim_next()
    if claimed is None:
        return False
    command = claimed.command
    try:
        # A child of the span that queued the command -- the request, or the
        # event delivery -- carried across in the command itself.
        with tracing.workflow_span(
            f"worker.command.{command.kind.value}",
            workflow_id=command.workflow_id,
            correlation_id=command.correlation_id,
            thread_id=command.thread_id,
            actor_id=command.actor_id,
            carrier=command.trace_context,
            kind=SpanKind.CONSUMER,
            attributes={
                Attr.COMMAND_ID: command.command_id,
                Attr.COMMAND_KIND: command.kind,
                Attr.ATTEMPT: claimed.attempt,
            },
        ):
            kind = thread_namespace(command.thread_id)
            runner = runners.get(kind or "")
            if runner is None:
                raise LookupError(
                    f"no runner is wired for '{kind}' threads (thread '{command.thread_id}')"
                )
            await run_workflow_command(command, runner, redelivered=claimed.redelivered)
    except WorkflowLeaseUnavailable:
        logger.info(
            "workflow %s is leased elsewhere; returning command %s to the queue",
            claimed.command.workflow_id,
            claimed.command.command_id,
        )
        queue.release(claimed.outbox_id, delay=lease_busy_delay)
        return True
    except Exception:
        logger.exception("workflow command %s failed", claimed.command.command_id)
        queue.mark_failed(claimed.outbox_id)
        raise
    queue.mark_done(claimed.outbox_id)
    return True


async def run_commands(
    queue: WorkflowCommandQueue,
    runners: Mapping[str, DurableWorkflowRunner],
    stop: asyncio.Event,
    *,
    idle_seconds: float = 1.0,
    lease_busy_delay: timedelta = DEFAULT_LEASE_BUSY_DELAY,
) -> None:
    """Run queued commands until `stop` is set, pausing `idle_seconds` when there are none.

    A command that raised has already been settled as failed by `process_next`;
    the loop logs nothing more about it and moves on, because one bad command
    must not stop every other workflow behind it.
    """
    while not stop.is_set():
        try:
            ran = await process_next(queue, runners, lease_busy_delay=lease_busy_delay)
        except Exception:
            # Either the command failed (already logged and settled) or the
            # claim itself could not reach the database; in the second case
            # back off as if idle, so a database blip is not a busy loop.
            ran = False
        if not ran:
            try:
                await asyncio.wait_for(stop.wait(), timeout=idle_seconds)
            except asyncio.TimeoutError:
                pass


__all__ = [
    "COMMAND_ID_KEY",
    "DEFAULT_LEASE_BUSY_DELAY",
    "command_configurable",
    "run_workflow_command",
    "process_next",
    "run_commands",
]
