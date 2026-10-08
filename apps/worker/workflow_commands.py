"""Runs the workflow commands the API queued: the worker half of the hand-off.

The API never runs a graph. `POST /v1/chat` and `POST /v1/workflows/{id}/resume`
write a `WorkflowCommand` to `outbox_events` and return; this module is what
turns one of those rows into a `DurableWorkflowRunner` call, under the
workflow's lease. The polling loop that calls `process_next` belongs to the
worker process (Phase G); everything a single command needs is here so that
loop has nothing to decide but *when*.

The request's identity crosses the hand-off in the command, not in any process:
`actor_id` and `correlation_id` are written into the run's `configurable`, which
is where `SupervisorGraph._load_context` reads the actor from and what every
checkpoint written during the run is stamped with.
"""

import logging
from collections.abc import Mapping
from typing import Any

from langgraph.types import Command

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.domain.workflow import WorkflowCommand, WorkflowCommandKind, thread_namespace
from personalos.persistence.leases import WorkflowLeaseUnavailable
from personalos.persistence.workflow_commands import WorkflowCommandQueue

logger = logging.getLogger(__name__)


def command_configurable(command: WorkflowCommand) -> dict[str, Any]:
    """The `configurable` entries that carry a command's identity into its run."""
    return {
        "actor_id": command.actor_id,
        "correlation_id": str(command.correlation_id),
    }


async def run_workflow_command(
    command: WorkflowCommand, runner: DurableWorkflowRunner
) -> dict[str, Any]:
    """Carry out one command on the runner for the graph its thread runs."""
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
    if command.kind == WorkflowCommandKind.START:
        thread = runner.registry.require(command.thread_id)
        return await runner.start(thread, command.graph_input or {}, configurable=configurable)
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
) -> bool:
    """Claim and run the oldest queued command. False when there was none.

    `runners` maps a thread kind -- the namespace of its derived id,
    `supervisor` or `job_search` -- to the runner for the graph it runs on.
    Keyed by thread rather than by workflow because one workflow's Supervisor
    thread and Job Search thread are resumed on different graphs.

    A command whose workflow is leased by another worker is put back rather
    than failed: the other worker is mid-run on the same process, and this
    command is next in line, not wrong. Any other failure settles the row as
    failed; the runner has already marked the thread failed, and its last
    checkpoint is what a later resume starts from.
    """
    claimed = queue.claim_next()
    if claimed is None:
        return False
    try:
        kind = thread_namespace(claimed.command.thread_id)
        runner = runners.get(kind or "")
        if runner is None:
            raise LookupError(
                f"no runner is wired for '{kind}' threads "
                f"(thread '{claimed.command.thread_id}')"
            )
        await run_workflow_command(claimed.command, runner)
    except WorkflowLeaseUnavailable:
        logger.info(
            "workflow %s is leased elsewhere; returning command %s to the queue",
            claimed.command.workflow_id,
            claimed.command.command_id,
        )
        queue.release(claimed.outbox_id)
        return True
    except Exception:
        logger.exception("workflow command %s failed", claimed.command.command_id)
        queue.mark_failed(claimed.outbox_id)
        raise
    queue.mark_done(claimed.outbox_id)
    return True


__all__ = ["command_configurable", "run_workflow_command", "process_next"]
