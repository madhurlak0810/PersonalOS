"""Workflow lifecycle over HTTP: start, inspect, resume. Version 1.

Three endpoints, none of which runs a graph:

- `POST /v1/chat` registers (or finds) a Supervisor thread and queues a
  `start` command for it.
- `GET /v1/workflows/{id}` reports where the workflow stands, read entirely
  from the database.
- `POST /v1/workflows/{id}/resume` queues the answer to the interrupt the
  workflow is parked on -- and refuses, with a 409, when it is not parked on
  one.

**No workflow state lives in this process.** Each request builds its registry,
reader and queue fresh from a session factory (`get_workflow_services`), so a
request served by an API instance that booted a second ago gets the same answer
as one served by the instance that started the workflow. The only in-memory
cache below this layer, `WorkflowThreadRegistry`'s memo, dies with the request.

**Long work goes to the worker.** Starting and resuming both end at
`WorkflowCommandQueue.enqueue` and return `202 Accepted`; the worker claims the
command and runs it under the workflow's lease (`apps.worker.workflow_commands`).

**Identity crosses the hand-off in the command.** The caller's `X-Actor-Id` and
the request's correlation id (`CorrelationIdMiddleware`) are written into the
command and onto the thread's `workflow_runs` row, and the worker puts both into
the run's config -- the Phase A `ExecutionContext` contract, carried into a
process that never saw the request. An approval decision's `decided_by` is the
actor too, never a field the caller fills in.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from personalos.bootstrap import (
    SUPERVISOR_WORKFLOW,
    build_workflow_command_queue,
    build_workflow_status_reader,
    build_workflow_thread_registry,
    register_supervisor_thread,
)
from personalos.domain.errors import Conflict, NotFound, ValidationFailed
from personalos.domain.job_search import ApprovalDecision, ApprovalVerdict
from personalos.domain.workflow import (
    SUPERVISOR_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
    WorkflowSnapshot,
    WorkflowStatus,
    thread_namespace,
)
from personalos.persistence.checkpointer import WorkflowThreadRegistry
from personalos.persistence.database import SessionLocal
from personalos.persistence.workflow_commands import WorkflowCommandQueue
from personalos.persistence.workflow_status import WorkflowStatusReader

logger = logging.getLogger(__name__)

router = APIRouter()

#: Actor recorded when a caller does not declare one, matching the jobs routes.
_DEFAULT_ACTOR = "api"

#: Upper bound on one chat message. Generous for a request, and well short of
#: anything that would make a checkpoint row unreasonable.
_MAX_MESSAGE_LENGTH = 10_000


# --- Dependencies ---------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowServices:
    """What a workflow request needs, built per request from a session factory."""

    registry: WorkflowThreadRegistry
    reader: WorkflowStatusReader
    commands: WorkflowCommandQueue


def build_workflow_services(session_factory: Callable[[], Any]) -> WorkflowServices:
    """Wire the registry, reader and queue against one database."""
    registry = build_workflow_thread_registry(session_factory)
    return WorkflowServices(
        registry=registry,
        reader=build_workflow_status_reader(session_factory, registry),
        commands=build_workflow_command_queue(session_factory),
    )


def get_workflow_services() -> WorkflowServices:
    """FastAPI dependency: fresh services over the configured database."""
    return build_workflow_services(SessionLocal)


def _correlation_id(request: Request) -> UUID:
    """The request's correlation id, or a fresh one on an app without the middleware."""
    return getattr(request.state, "correlation_id", None) or uuid4()


# --- Request / response models --------------------------------------------------


class ChatRequest(BaseModel):
    """Start a conversation, or continue one by naming its thread."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_LENGTH)
    #: The thread a previous `POST /v1/chat` returned. Omitted, a new
    #: conversation -- and a new workflow -- is started.
    thread_id: str | None = None
    #: The candidate the conversation is about, when the caller knows it.
    user_id: UUID | None = None


class ChatResponse(BaseModel):
    """What was queued: where to poll, and which thread to continue on."""

    workflow_id: UUID
    thread_id: str
    command_id: UUID
    status: WorkflowStatus
    correlation_id: UUID
    actor_id: str


class ApprovalDecisionInput(BaseModel):
    """A reviewer's answer to one `ApprovalRequest` from a pending approval.

    `decided_by` is not here on purpose: it is the request's actor.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: UUID
    action_fingerprint: str = Field(min_length=1)
    verdict: Literal[ApprovalVerdict.APPROVED, ApprovalVerdict.REJECTED]
    request_id: UUID | None = None
    note: str | None = None


class ResumeRequest(BaseModel):
    """The answer to whatever the workflow is waiting on.

    Exactly one of `decisions` (for an approval) or `event` (for anything else
    an interrupt waits on) must be given.
    """

    model_config = ConfigDict(extra="forbid")

    decisions: list[ApprovalDecisionInput] | None = Field(default=None, min_length=1)
    event: dict[str, Any] | None = None
    #: Which waiting thread to resume, when more than one is waiting.
    thread_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_answer(self) -> "ResumeRequest":
        if (self.decisions is None) == (self.event is None):
            raise ValueError("give exactly one of 'decisions' or 'event'")
        return self


class ResumeResponse(BaseModel):
    """What was queued in answer to the interrupt."""

    workflow_id: UUID
    thread_id: str
    command_id: UUID
    status: WorkflowStatus
    correlation_id: UUID
    actor_id: str


class StepView(BaseModel):
    """One graph node, on the thread it belongs to."""

    thread_id: str
    step: str


class PendingApprovalView(BaseModel):
    """The question a workflow is parked on."""

    thread_id: str
    step: str | None = None
    interrupt_id: str | None = None
    #: The interrupt's payload as the graph raised it; for the job search
    #: approval checkpoint, `{"workflow", "stage", "requests": [...]}`.
    payload: Any = None


class RecoverableFailureView(BaseModel):
    """A node failure whose checkpoint is intact, so a resume can retry it."""

    thread_id: str
    step: str | None = None
    message: str


class ThreadView(BaseModel):
    """One thread of the workflow."""

    thread_id: str
    #: Which graph the thread runs on: `supervisor`, `job_search`, ...
    kind: str | None = None
    run_status: str
    checkpoint_id: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    waiting: bool = False


class WorkflowStatusResponse(BaseModel):
    """Where a workflow stands, as of the last row any process wrote."""

    workflow_id: UUID
    name: str | None = None
    status: WorkflowStatus
    current_step: StepView | None = None
    pending_approval: PendingApprovalView | None = None
    completed_steps: list[StepView] = Field(default_factory=list)
    recoverable_failures: list[RecoverableFailureView] = Field(default_factory=list)
    #: Commands handed to the worker that it has not finished.
    queued_commands: int = 0
    threads: list[ThreadView] = Field(default_factory=list)
    actor_id: str | None = None
    correlation_id: UUID | None = None
    created_at: datetime | None = None

    @classmethod
    def from_snapshot(cls, snapshot: WorkflowSnapshot) -> "WorkflowStatusResponse":
        """Project the domain snapshot onto the v1 wire shape."""
        pending = snapshot.pending_interrupts
        waiting = {interrupt.thread_id for interrupt in pending}
        current = snapshot.current_step
        origin = snapshot.threads[0] if snapshot.threads else None
        return cls(
            workflow_id=snapshot.workflow_id,
            name=snapshot.name,
            status=snapshot.status,
            current_step=StepView(**current.model_dump()) if current else None,
            pending_approval=(
                PendingApprovalView(
                    thread_id=pending[0].thread_id,
                    step=pending[0].step,
                    interrupt_id=pending[0].interrupt_id,
                    payload=pending[0].value,
                )
                if pending
                else None
            ),
            completed_steps=[StepView(**step.model_dump()) for step in snapshot.completed_steps],
            recoverable_failures=[
                RecoverableFailureView(**failure.model_dump())
                for failure in snapshot.recoverable_failures
            ],
            queued_commands=len(snapshot.queued),
            threads=[
                ThreadView(
                    thread_id=thread.thread_id,
                    kind=thread.kind,
                    run_status=thread.run_status,
                    checkpoint_id=thread.checkpoint_id,
                    next_steps=list(thread.next_steps),
                    completed_steps=list(thread.completed_steps),
                    waiting=thread.thread_id in waiting,
                )
                for thread in snapshot.threads
            ],
            actor_id=origin.actor_id if origin else None,
            correlation_id=origin.correlation_id if origin else None,
            created_at=snapshot.created_at,
        )


# --- Routes ---------------------------------------------------------------------


def _read_or_404(services: WorkflowServices, workflow_id: UUID) -> WorkflowSnapshot:
    snapshot = services.reader.read(workflow_id)
    if snapshot is None:
        raise NotFound(
            f"workflow '{workflow_id}' not found", details={"workflow_id": str(workflow_id)}
        )
    return snapshot


@router.post("/chat", response_model=ChatResponse, status_code=202)
async def chat(
    http_request: Request,
    http_response: Response,
    request: ChatRequest,
    services: WorkflowServices = Depends(get_workflow_services),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
):
    """Start a Supervisor conversation, or continue one, by queuing a run of it.

    Continuing a conversation whose workflow is parked on an approval is a
    409: a new message would start the thread over and strand the question it
    is waiting on. Answer it through the resume endpoint first.
    """
    actor_id = x_actor_id or _DEFAULT_ACTOR
    correlation_id = _correlation_id(http_request)

    if request.thread_id is None:
        workflow_id = services.registry.create_workflow(SUPERVISOR_WORKFLOW)
        thread = register_supervisor_thread(
            conversation_key=str(uuid4()),
            registry=services.registry,
            workflow_id=workflow_id,
            user_id=request.user_id,
            actor_id=actor_id,
            correlation_id=correlation_id,
        )
    else:
        thread = services.registry.resolve(request.thread_id)
        if thread is None:
            raise NotFound(
                f"thread '{request.thread_id}' not found",
                details={"thread_id": request.thread_id},
            )
        if thread_namespace(thread.thread_id) != SUPERVISOR_THREAD_NAMESPACE:
            raise ValidationFailed(
                f"thread '{request.thread_id}' is not a conversation thread; only "
                f"'{SUPERVISOR_THREAD_NAMESPACE}' threads can be continued through /v1/chat"
            )
        snapshot = _read_or_404(services, thread.workflow_id)
        if snapshot.pending_interrupts:
            raise Conflict(
                f"workflow {thread.workflow_id} is waiting on an approval; answer it with "
                f"POST /v1/workflows/{thread.workflow_id}/resume before sending another message"
            )

    command = services.commands.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.START,
            workflow_id=thread.workflow_id,
            thread_id=thread.thread_id,
            actor_id=actor_id,
            correlation_id=correlation_id,
            graph_input={"message": request.message},
        )
    )
    logger.info(
        "queued start command %s for thread '%s' of workflow %s "
        "(actor_id=%s, correlation_id=%s)",
        command.command_id,
        thread.thread_id,
        thread.workflow_id,
        actor_id,
        correlation_id,
    )
    http_response.headers["Location"] = f"/v1/workflows/{thread.workflow_id}"
    return ChatResponse(
        workflow_id=thread.workflow_id,
        thread_id=thread.thread_id,
        command_id=command.command_id,
        status=WorkflowStatus.QUEUED,
        correlation_id=correlation_id,
        actor_id=actor_id,
    )


@router.get("/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow(
    workflow_id: UUID,
    services: WorkflowServices = Depends(get_workflow_services),
):
    """Current step, status, pending approval, completed steps and recoverable failures."""
    return WorkflowStatusResponse.from_snapshot(_read_or_404(services, workflow_id))


@router.post("/workflows/{workflow_id}/resume", response_model=ResumeResponse, status_code=202)
async def resume_workflow(
    workflow_id: UUID,
    http_request: Request,
    request: ResumeRequest,
    services: WorkflowServices = Depends(get_workflow_services),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
):
    """Queue the answer to the interrupt a workflow is parked on.

    A workflow that is not parked on anything -- still queued, running,
    finished, or already answered and waiting for the worker -- is a 409
    naming its actual status, never a silently accepted no-op. So is an answer
    racing another answer to the same interrupt: the queue refuses the second.
    """
    actor_id = x_actor_id or _DEFAULT_ACTOR
    correlation_id = _correlation_id(http_request)
    snapshot = _read_or_404(services, workflow_id)

    pending = [
        interrupt
        for interrupt in snapshot.pending_interrupts
        if request.thread_id is None or interrupt.thread_id == request.thread_id
    ]
    if not pending:
        where = f" on thread '{request.thread_id}'" if request.thread_id else ""
        raise Conflict(
            f"workflow {workflow_id} is not waiting on anything{where} "
            f"(status: {snapshot.status.value}); there is nothing to resume"
        )
    threads = sorted({interrupt.thread_id for interrupt in pending})
    if len(threads) > 1:
        raise Conflict(
            f"workflow {workflow_id} has {len(threads)} threads waiting "
            f"({', '.join(threads)}); pass thread_id to say which one to resume"
        )
    thread = next(t for t in snapshot.threads if t.thread_id == threads[0])

    if request.decisions is not None:
        decided_at = datetime.utcnow()
        resume_value: Any = [
            ApprovalDecision(
                **decision.model_dump(),
                decided_by=actor_id,
                decided_at=decided_at,
            ).model_dump(mode="json")
            for decision in request.decisions
        ]
    else:
        resume_value = request.event

    command = services.commands.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.RESUME,
            workflow_id=workflow_id,
            thread_id=thread.thread_id,
            actor_id=actor_id,
            correlation_id=correlation_id,
            graph_input=resume_value,
            interrupt_ids=tuple(i.interrupt_id for i in pending if i.interrupt_id),
        ),
        # One answer per parked checkpoint: a second resume of the same stop
        # collides here even if it raced past the pending check above.
        dedupe=thread.checkpoint_id,
    )
    logger.info(
        "queued resume command %s for thread '%s' of workflow %s "
        "(actor_id=%s, correlation_id=%s)",
        command.command_id,
        thread.thread_id,
        workflow_id,
        actor_id,
        correlation_id,
    )
    return ResumeResponse(
        workflow_id=workflow_id,
        thread_id=thread.thread_id,
        command_id=command.command_id,
        status=WorkflowStatus.QUEUED,
        correlation_id=correlation_id,
        actor_id=actor_id,
    )


__all__ = [
    "router",
    "WorkflowServices",
    "build_workflow_services",
    "get_workflow_services",
    "ChatRequest",
    "ChatResponse",
    "ApprovalDecisionInput",
    "ResumeRequest",
    "ResumeResponse",
    "WorkflowStatusResponse",
]
