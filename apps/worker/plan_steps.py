"""The worker half of cross-domain plans: step threads, and waking the plan back up.

A Supervisor plan runs each step on a thread of its own (`personalos.graphs.
plan_steps`). Two things about that need the worker:

- **Whose thread.** `WorkflowScopedStepThreads` registers each step's thread
  under the Supervisor run's *own* workflow, owned by the conversation's
  candidate -- so a plan is one workflow however many steps it has (the
  Phase L active-workflow cap counts it once), every step's approval is that
  actor's to answer, and another actor's scoped reads 404 on all of them.
- **Who resumes the plan.** A step parked at its approval is answered on its
  own thread: the approval queue resumes that thread, not the Supervisor's.
  `PlanStepResumer` runs after every command the worker carries out; when the
  command moved a step thread that a parked plan is waiting on, it queues one
  resume of that Supervisor thread, which then reads where every waiting step
  stands. The resume is deduplicated on the Supervisor's checkpoint, and the
  workflow lease keeps it from running alongside the step's own run, so two
  steps finishing back to back wake the plan once and it sees both.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID, uuid4

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    register_calendar_thread,
    register_communications_thread,
    register_file_task_thread,
    register_job_search_thread,
)
from personalos.domain.errors import ValidationFailed
from personalos.domain.routing import RouteDomain
from personalos.domain.task_plan import PlanStep
from personalos.domain.workflow import (
    SUPERVISOR_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
    WorkflowThread,
    thread_namespace,
)
from personalos.graphs.plan_steps import BoundThread, StepContext, step_task_key
from personalos.graphs.supervisor import STAGE_PLAN_STEPS
from personalos.persistence.checkpointer import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    WorkflowThreadRegistry,
)
from personalos.persistence.workflow_commands import DuplicateWorkflowCommand, WorkflowCommandQueue

logger = logging.getLogger(__name__)


def _register_job(**kwargs: Any) -> WorkflowThread:
    task_key = kwargs.pop("task_key")
    return register_job_search_thread(search_key=task_key, **kwargs)


_REGISTER: Mapping[RouteDomain, Callable[..., WorkflowThread]] = {
    RouteDomain.CALENDAR: register_calendar_thread,
    RouteDomain.COMMUNICATIONS: register_communications_thread,
    RouteDomain.FILE: register_file_task_thread,
    RouteDomain.JOB: _register_job,
}


class WorkflowScopedStepThreads:
    """Binds each plan step to a registered thread of the Supervisor run's workflow."""

    def __init__(self, registry: WorkflowThreadRegistry, *, default_user_id: UUID | None = None):
        self.registry = registry
        self.default_user_id = default_user_id

    def bind(self, step: PlanStep, context: StepContext) -> BoundThread:
        if not context.workflow_id:
            raise ValidationFailed("a plan step was run outside a registered Supervisor run")
        workflow_id = UUID(context.workflow_id)
        user_id = self.registry.user_id_for(context.parent_thread_id) or self.default_user_id
        if user_id is None:
            raise ValidationFailed(
                f"workflow {workflow_id} names no candidate to run plan step '{step.id}' for; "
                f"start the conversation with a user_id"
            )
        thread = _REGISTER[step.domain](
            user_id=user_id,
            registry=self.registry,
            workflow_id=workflow_id,
            task_key=step_task_key(context, step),
            actor_id=context.actor_id,
            correlation_id=UUID(context.correlation_id) if context.correlation_id else None,
        )
        configurable = dict(thread.config().get("configurable") or {})
        configurable.pop("thread_id", None)
        configurable.setdefault("actor_id", context.actor_id)
        if context.correlation_id:
            configurable.setdefault("correlation_id", context.correlation_id)
        return BoundThread(thread_id=thread.thread_id, user_id=user_id, configurable=configurable)

    def started(self, thread: BoundThread) -> None:
        self.registry.mark_status(thread.thread_id, RUN_STATUS_RUNNING, started=True)

    def settled(self, thread: BoundThread, *, finished: bool, failed: bool = False) -> None:
        if failed:
            self.registry.mark_status(thread.thread_id, RUN_STATUS_FAILED, finished=True)
        elif finished:
            self.registry.mark_status(thread.thread_id, RUN_STATUS_COMPLETED, finished=True)


#: The `graph_input` key a plan resume is queued with. A non-empty value: an
#: empty mapping would be read as "answer no interrupt" and resume nothing.
RESUMED_BY_STEP = "plan_step_thread"


class PlanStepResumer:
    """After a command moves a plan step's thread, wake the plan that waits on it."""

    def __init__(self, queue: WorkflowCommandQueue, runners: Mapping[str, DurableWorkflowRunner]):
        self.queue = queue
        self.runners = runners

    async def __call__(self, command: WorkflowCommand) -> None:
        kind = thread_namespace(command.thread_id)
        supervisor = self.runners.get(SUPERVISOR_THREAD_NAMESPACE)
        step_runner = self.runners.get(kind or "")
        if kind == SUPERVISOR_THREAD_NAMESPACE or supervisor is None or step_runner is None:
            return
        step_thread = supervisor.registry.resolve(command.thread_id)
        if step_thread is None:
            return
        step_state = await step_runner.graph.aget_state(step_thread.config())
        if step_state.next and _has_interrupt(step_state):
            # Still parked -- answered in part, or asked again. Nothing to wake.
            return
        for parent in supervisor.registry.threads_for_workflow(command.workflow_id):
            if thread_namespace(parent.thread_id) != SUPERVISOR_THREAD_NAMESPACE:
                continue
            snapshot = await supervisor.graph.aget_state(parent.config())
            interrupt_id = _waiting_on(snapshot, command.thread_id)
            if interrupt_id is None:
                continue
            checkpoint_id = ((snapshot.config or {}).get("configurable") or {}).get("checkpoint_id")
            try:
                self.queue.enqueue(
                    WorkflowCommand(
                        command_id=uuid4(),
                        kind=WorkflowCommandKind.RESUME,
                        workflow_id=command.workflow_id,
                        thread_id=parent.thread_id,
                        actor_id=command.actor_id,
                        correlation_id=command.correlation_id,
                        graph_input={RESUMED_BY_STEP: command.thread_id},
                        interrupt_ids=(interrupt_id,) if interrupt_id else (),
                    ),
                    dedupe=f"plan:{checkpoint_id or parent.thread_id}",
                )
                logger.info(
                    "plan step thread '%s' moved; resuming plan thread '%s'",
                    command.thread_id,
                    parent.thread_id,
                )
            except DuplicateWorkflowCommand:
                logger.info(
                    "plan thread '%s' is already queued to resume past checkpoint %s",
                    parent.thread_id,
                    checkpoint_id,
                )


def _has_interrupt(snapshot: Any) -> bool:
    return any(task.interrupts for task in snapshot.tasks or ())


def _waiting_on(snapshot: Any, thread_id: str) -> str | None:
    """The id of the plan interrupt that waits on `thread_id`, or `None`."""
    for task in snapshot.tasks or ():
        for pending in task.interrupts or ():
            value = pending.value if isinstance(pending.value, dict) else {}
            if value.get("stage") != STAGE_PLAN_STEPS:
                continue
            waiting = {item.get("thread_id") for item in value.get("waiting_on") or ()}
            if thread_id in waiting:
                return getattr(pending, "id", None) or ""
    return None


__all__ = ["RESUMED_BY_STEP", "PlanStepResumer", "WorkflowScopedStepThreads"]
