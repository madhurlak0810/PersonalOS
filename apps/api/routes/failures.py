"""Failed workflow steps over HTTP: see them, retry them, or close them. Version 1.

Three endpoints:

- `GET /v1/workflows/{id}/failures` lists each failed step with its failure
  class, the step that raised, whether a retry is allowed (and why not), the
  last error (redacted), attempts so far, and the side effects it had started.
- `POST /v1/workflows/{id}/retry` queues a re-run of one failed step.
- `POST /v1/workflows/{id}/dismiss` acknowledges a failure and closes it
  without a retry.

**A retry resumes, it does not restart.** It queues a `retry` command, and the
worker re-runs only the failed step, from the checkpoint written just before
it (`apps.worker.workflow_commands`).

**A retry never sends blindly.** Every send in the re-run step goes back
through `ToolExecutor` under its original idempotency key: one that already
succeeded returns its stored receipt and calls nothing, and one whose outcome
is unknown is reconciled against the provider before anything is sent. The
response lists each such operation and which of the two will happen to it.

**Some failures are not offered a retry.** A policy denial is shown as
`denied`, and can only be dismissed. A provider-permanent failure (revoked
OAuth, a missing scope) marks the integration degraded and asks for a
reconnect. A step that has failed `MAX_FAILURE_ATTEMPTS` times can only be
dismissed. Each of these is a 409 from the retry endpoint, never a no-op.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from personalos.bootstrap import (
    build_workflow_command_queue,
    build_workflow_failure_store,
    build_workflow_thread_registry,
)
from personalos.domain.errors import Conflict, NotFound
from personalos.domain.failures import (
    FailedOperation,
    FailureStatus,
    WorkflowFailure,
)
from personalos.domain.workflow import (
    WorkflowCommand,
    WorkflowCommandKind,
    WorkflowSnapshot,
    WorkflowStatus,
)
from personalos.persistence.database import SessionLocal
from personalos.persistence.workflow_commands import WorkflowCommandQueue
from personalos.persistence.workflow_failures import WorkflowFailureStore, retry_dedupe

logger = logging.getLogger(__name__)

router = APIRouter()

#: Actor recorded when a caller does not declare one, matching the other routes.
_DEFAULT_ACTOR = "api"


# --- Dependencies ---------------------------------------------------------------


@dataclass(frozen=True)
class FailureServices:
    """What a failure request needs, built per request from a session factory."""

    store: WorkflowFailureStore
    commands: WorkflowCommandQueue


def build_failure_services(session_factory: Callable[[], Any]) -> FailureServices:
    """Wire the failure store and command queue against one database."""
    registry = build_workflow_thread_registry(session_factory)
    return FailureServices(
        store=build_workflow_failure_store(session_factory, registry),
        commands=build_workflow_command_queue(session_factory),
    )


def get_failure_services() -> FailureServices:
    """FastAPI dependency: fresh services over the configured database."""
    return build_failure_services(SessionLocal)


def _correlation_id(request: Request) -> UUID:
    return getattr(request.state, "correlation_id", None) or uuid4()


# --- Request / response models --------------------------------------------------


class WorkflowFailuresResponse(BaseModel):
    """A workflow's failed steps."""

    workflow_id: UUID
    status: WorkflowStatus
    failures: list[WorkflowFailure] = Field(default_factory=list)


class FailureSelection(BaseModel):
    """Which failure to act on. Optional while the workflow has only one open."""

    model_config = ConfigDict(extra="forbid")

    failure_id: UUID | None = None


class RetryResponse(BaseModel):
    """What was queued, and what the re-run will do about each side effect."""

    workflow_id: UUID
    failure_id: UUID
    thread_id: str
    #: The step being re-run; the steps before it are not.
    step: str | None = None
    command_id: UUID
    #: Which attempt at the step this is, the first one included.
    attempt: int
    status: FailureStatus
    #: Operations already succeeded carry their stored receipt, which the
    #: re-run returns instead of sending again.
    operations: list[FailedOperation] = Field(default_factory=list)
    correlation_id: UUID
    actor_id: str


class DismissResponse(BaseModel):
    """The failure, closed."""

    workflow_id: UUID
    failure_id: UUID
    status: FailureStatus
    dismissed_by: str
    dismissed_at: datetime


# --- Routes ---------------------------------------------------------------------


def _snapshot_or_404(services: FailureServices, workflow_id: UUID) -> WorkflowSnapshot:
    snapshot = services.store.snapshot(workflow_id)
    if snapshot is None:
        raise NotFound(
            f"workflow '{workflow_id}' not found", details={"workflow_id": str(workflow_id)}
        )
    return snapshot


def _select(
    services: FailureServices, snapshot: WorkflowSnapshot, failure_id: UUID | None
) -> WorkflowFailure:
    """The failure a request means: the one named, or the only open one."""
    failures = services.store.failures(snapshot, include_dismissed=True)
    if failure_id is not None:
        for failure in failures:
            if failure.failure_id == failure_id:
                return failure
        raise NotFound(
            f"workflow {snapshot.workflow_id} has no failure '{failure_id}'",
            details={"workflow_id": str(snapshot.workflow_id), "failure_id": str(failure_id)},
        )
    open_failures = [f for f in failures if f.status != FailureStatus.DISMISSED]
    if not open_failures:
        raise Conflict(
            f"workflow {snapshot.workflow_id} has no open failures "
            f"(status: {snapshot.status.value})"
        )
    if len(open_failures) > 1:
        raise Conflict(
            f"workflow {snapshot.workflow_id} has {len(open_failures)} open failures "
            f"({', '.join(str(f.failure_id) for f in open_failures)}); pass failure_id "
            f"to say which one"
        )
    return open_failures[0]


@router.get("/workflows/{workflow_id}/failures", response_model=WorkflowFailuresResponse)
async def list_failures(
    workflow_id: UUID,
    include_dismissed: bool = Query(default=False),
    services: FailureServices = Depends(get_failure_services),
):
    """Every failed step, with its class, whether it can be retried, and why not."""
    snapshot = _snapshot_or_404(services, workflow_id)
    return WorkflowFailuresResponse(
        workflow_id=workflow_id,
        status=snapshot.status,
        failures=services.store.failures(snapshot, include_dismissed=include_dismissed),
    )


@router.post("/workflows/{workflow_id}/retry", response_model=RetryResponse, status_code=202)
async def retry_failure(
    workflow_id: UUID,
    http_request: Request,
    request: FailureSelection | None = None,
    services: FailureServices = Depends(get_failure_services),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
):
    """Queue a re-run of a failed step, from the checkpoint before it.

    A 409 when the failure is not retryable -- a policy denial, a provider
    that needs reconnecting, a step out of attempts, one already dismissed or
    already being retried -- naming the reason.
    """
    actor_id = x_actor_id or _DEFAULT_ACTOR
    correlation_id = _correlation_id(http_request)
    snapshot = _snapshot_or_404(services, workflow_id)
    failure = _select(services, snapshot, (request or FailureSelection()).failure_id)

    if not failure.retryable:
        raise Conflict(
            f"failure {failure.failure_id} of workflow {workflow_id} cannot be retried: "
            f"{failure.not_retryable_reason}",
            details={"failure_id": str(failure.failure_id), "status": failure.status.value},
        )

    attempt = failure.attempts + 1
    command = services.commands.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.RETRY,
            workflow_id=workflow_id,
            thread_id=failure.thread_id,
            actor_id=actor_id,
            correlation_id=correlation_id,
        ),
        # One retry per attempt: a second request racing this one for the
        # same attempt collides at the database instead of queuing a duplicate.
        dedupe=retry_dedupe(failure.failure_id, attempt),
    )
    services.store.record_retry(failure, actor=actor_id)
    logger.info(
        "queued retry %s (attempt %s) of step '%s' on thread '%s' of workflow %s "
        "(failure %s, class %s, actor_id=%s, correlation_id=%s)",
        command.command_id,
        attempt,
        failure.step,
        failure.thread_id,
        workflow_id,
        failure.failure_id,
        failure.failure_class.value,
        actor_id,
        correlation_id,
    )
    return RetryResponse(
        workflow_id=workflow_id,
        failure_id=failure.failure_id,
        thread_id=failure.thread_id,
        step=failure.step,
        command_id=command.command_id,
        attempt=attempt,
        status=FailureStatus.RETRY_QUEUED,
        operations=failure.operations,
        correlation_id=correlation_id,
        actor_id=actor_id,
    )


@router.post("/workflows/{workflow_id}/dismiss", response_model=DismissResponse)
async def dismiss_failure(
    workflow_id: UUID,
    request: FailureSelection | None = None,
    services: FailureServices = Depends(get_failure_services),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
):
    """Acknowledge a failure and close it, without retrying it.

    Nothing is re-run and nothing is undone: the step stays failed, and the
    dismissal is an audit event naming who closed it. A 409 for a failure
    already dismissed, or one with a retry queued -- let the retry finish.
    """
    actor_id = x_actor_id or _DEFAULT_ACTOR
    snapshot = _snapshot_or_404(services, workflow_id)
    failure = _select(services, snapshot, (request or FailureSelection()).failure_id)

    if failure.status == FailureStatus.DISMISSED:
        raise Conflict(
            f"failure {failure.failure_id} of workflow {workflow_id} was already dismissed "
            f"by {failure.dismissed_by}"
        )
    if failure.status == FailureStatus.RETRY_QUEUED:
        raise Conflict(
            f"failure {failure.failure_id} of workflow {workflow_id} has a retry queued; "
            f"it cannot be dismissed until that retry has run"
        )

    services.store.record_dismissal(failure, actor=actor_id)
    [dismissed] = [
        f
        for f in services.store.failures(snapshot, include_dismissed=True)
        if f.failure_id == failure.failure_id
    ]
    logger.info(
        "failure %s (%s) of workflow %s dismissed by %s",
        failure.failure_id,
        failure.failure_class.value,
        workflow_id,
        actor_id,
    )
    return DismissResponse(
        workflow_id=workflow_id,
        failure_id=failure.failure_id,
        status=dismissed.status,
        dismissed_by=dismissed.dismissed_by or actor_id,
        dismissed_at=dismissed.dismissed_at or datetime.utcnow(),
    )


__all__ = [
    "router",
    "FailureServices",
    "build_failure_services",
    "get_failure_services",
    "FailureSelection",
    "WorkflowFailuresResponse",
    "RetryResponse",
    "DismissResponse",
]
