"""The application pipeline over HTTP: the board, one application, a move. Version 1.

Three endpoints:

- `GET /v1/applications` is the board: every application grouped by lifecycle
  state, filtered by state, company and creation date, a page at a time.
- `GET /v1/applications/{id}` is one application in full: its fit score and the
  evidence-backed match behind it, its tailored documents, what recruiters have
  said, the waits still open on it, and any approval its board moves are
  parked on.
- `POST /v1/applications/{id}/transition` asks for a status change.

**Reads come from the status projection.** `application_status_view` is what
every transition moves in the same transaction as its event, so the board
shows a move the moment the request that made it returns.

**A move is never a raw UPDATE.** It is checked against the lifecycle first
(`check_manual_transition`); an illegal one is a 422 that lists the states the
application *can* move to. A legal one is written through
`ApplicationLifecycleStore.transition`, the same guarded write -- and the same
`application.status_changed` event -- as every other caller.

**A move that implies an external effect is not applied here.** Asking to
submit the application, or to tell the recruiter, is put to the policy engine
(and recorded there); a denial is a 403. Anything else is recorded as an
`application.transition_requested` event and handed to the worker as a
job-search run on the application's own thread, which proposes the effect,
parks at the approval checkpoint -- where it shows up in `GET /v1/approvals` --
and writes the status change only once the effect has run. That answer is a
202, and the board keeps showing the old status until then.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from apps.api.routes.approvals import PendingApprovalItem
from apps.api.routes.workflows import WorkflowServices, build_workflow_services
from personalos.bootstrap import (
    build_application_board_store,
    build_application_lifecycle_store,
    build_policy_engine,
    register_application_transition_thread,
)
from personalos.domain.action_status import WorkflowAction, recommended_action
from personalos.domain.applications import (
    MAX_BOARD_PAGE_SIZE,
    ApplicationCard,
    ArtifactSummary,
    BoardQuery,
    RecruiterEventSummary,
    RequestedTransition,
    StatusChange,
    TransitionEffect,
    check_manual_transition,
    transition_action_intent,
)
from personalos.domain.checkpoints import PendingCheckpoint
from personalos.domain.errors import Conflict, NotFound, PolicyDeniedError
from personalos.domain.job_search import JobMatch
from personalos.domain.models import ApplicationStatus
from personalos.domain.workflow import (
    WorkflowCommand,
    WorkflowCommandKind,
    WorkflowStatus,
    application_transition_thread_id,
)
from personalos.executor.tool_executor import ACTION_TOOLS
from personalos.persistence.application_board import ApplicationBoardStore
from personalos.persistence.application_lifecycle import ApplicationLifecycleStore
from personalos.persistence.database import SessionLocal
from personalos.policy.engine import PolicyEngine
from personalos.policy.intents import Decision, IntentOrigin
from personalos.policy.permissions import Provenance

logger = logging.getLogger(__name__)

router = APIRouter()

#: Actor recorded when a caller does not declare one, matching the other routes.
_DEFAULT_ACTOR = "api"

#: Upper bounds on the free text a move carries.
_MAX_REASON_LENGTH = 2_000
_MAX_MESSAGE_LENGTH = 10_000

#: Workflow statuses in which a board move on the application is still in flight.
_IN_FLIGHT = frozenset({WorkflowStatus.QUEUED, WorkflowStatus.RUNNING, WorkflowStatus.WAITING})


# --- Dependencies ---------------------------------------------------------------


@dataclass(frozen=True)
class ApplicationServices:
    """What an application request needs, built per request from a session factory."""

    board: ApplicationBoardStore
    lifecycle: ApplicationLifecycleStore
    policy: PolicyEngine
    workflows: WorkflowServices


def build_application_services(session_factory: Callable[[], Any]) -> ApplicationServices:
    """Wire the board, lifecycle, policy engine and workflow services against one database."""
    return ApplicationServices(
        board=build_application_board_store(session_factory),
        lifecycle=build_application_lifecycle_store(session_factory),
        policy=build_policy_engine(session_factory),
        workflows=build_workflow_services(session_factory),
    )


def get_application_services() -> ApplicationServices:
    """FastAPI dependency: fresh services over the configured database."""
    return build_application_services(SessionLocal)


# --- Request / response models --------------------------------------------------


class BoardColumnView(BaseModel):
    """One lifecycle state: this page's cards in it, and how many match in all."""

    status: ApplicationStatus
    total: int
    applications: list[ApplicationCard] = Field(default_factory=list)


class BoardResponse(BaseModel):
    """The board, one column per state, in lifecycle order."""

    columns: list[BoardColumnView]
    total: int
    limit: int
    offset: int
    #: Pass as `offset` for the next page; absent on the last one.
    next_offset: int | None = None


class ApplicationDetailResponse(BaseModel):
    """One application in full."""

    application: ApplicationCard
    #: The recorded match's score, 0..1; absent until a match is recorded.
    fit_score: float | None = None
    #: The evidence-backed match: matched and missing requirements, each match
    #: citing the evidence it rests on.
    match: JobMatch | None = None
    match_recorded_at: datetime | None = None
    #: The states `POST .../transition` will accept from here.
    allowed_next_states: list[ApplicationStatus]
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    recruiter_events: list[RecruiterEventSummary] = Field(default_factory=list)
    pending_checkpoints: list[PendingCheckpoint] = Field(default_factory=list)
    #: Approvals a board move on this application is parked on.
    pending_approvals: list[PendingApprovalItem] = Field(default_factory=list)
    history: list[StatusChange] = Field(default_factory=list)


class TransitionRequest(BaseModel):
    """A status change, and optionally something to do outside the system with it."""

    model_config = ConfigDict(extra="forbid")

    to: ApplicationStatus
    reason: str | None = Field(default=None, max_length=_MAX_REASON_LENGTH)
    #: Omitted, the move records something that already happened and is applied
    #: at once. Given, the effect goes through policy and approval first.
    effect: TransitionEffect | None = None
    #: For `notify_recruiter`: who to tell.
    recipient: str | None = Field(default=None, max_length=320)
    #: For `notify_recruiter`, the message; for `submit_application`, a cover
    #: note in place of the latest cover letter.
    message: str | None = Field(default=None, max_length=_MAX_MESSAGE_LENGTH)


class TransitionOutcomeKind(str, Enum):
    """What the request did."""

    #: The status was changed, and its event written.
    APPLIED = "applied"
    #: The effect was queued for approval; the status has not changed yet.
    QUEUED = "queued"


class TransitionResponse(BaseModel):
    """The answer to a move: the card as it now stands, and what happened."""

    outcome: TransitionOutcomeKind
    from_status: ApplicationStatus
    to_status: ApplicationStatus
    effect: TransitionEffect | None = None
    application: ApplicationCard
    #: The `event_log` row this request wrote: the status change when applied,
    #: the transition request when queued.
    event_id: UUID
    #: When queued: where to follow the run, and the policy verdict it carries.
    workflow_id: UUID | None = None
    thread_id: str | None = None
    command_id: UUID | None = None
    policy_decision: Decision | None = None
    #: When queued: the effect, as proposed. `RECOMMENDED` -- nothing has been
    #: sent or submitted; follow `workflow_id` for its approval and receipt.
    action: WorkflowAction | None = None


# --- Routes ---------------------------------------------------------------------


def _card_or_404(services: ApplicationServices, application_id: UUID) -> ApplicationCard:
    card = services.board.card(application_id)
    if card is None:
        raise NotFound(
            f"application '{application_id}' not found",
            details={"application_id": str(application_id)},
        )
    return card


@router.get("/applications", response_model=BoardResponse)
async def list_applications(
    state: list[ApplicationStatus] | None = Query(default=None),
    company: str | None = Query(default=None, max_length=255),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    user_id: UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=MAX_BOARD_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    services: ApplicationServices = Depends(get_application_services),
):
    """The board: applications grouped by lifecycle state, a page at a time.

    `state` may be repeated to show several columns. Pages run through the
    board in column order, most recently active first within a column.
    """
    board = services.board.board(
        BoardQuery(
            states=tuple(state or ()),
            company=company or None,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
    )
    return BoardResponse(
        columns=[
            BoardColumnView(
                status=column.status,
                total=column.total,
                applications=list(column.applications),
            )
            for column in board.columns
        ],
        total=board.total,
        limit=board.limit,
        offset=board.offset,
        next_offset=board.next_offset,
    )


@router.get("/applications/{application_id}", response_model=ApplicationDetailResponse)
async def get_application(
    application_id: UUID,
    services: ApplicationServices = Depends(get_application_services),
):
    """One application: fit, evidence, documents, recruiter events and open waits."""
    detail = services.board.detail(application_id)
    if detail is None:
        raise NotFound(
            f"application '{application_id}' not found",
            details={"application_id": str(application_id)},
        )
    thread_id = application_transition_thread_id(detail.card.user_id, application_id)
    return ApplicationDetailResponse(
        application=detail.card,
        fit_score=detail.fit_score,
        match=detail.match.match if detail.match else None,
        match_recorded_at=detail.match.recorded_at if detail.match else None,
        allowed_next_states=list(detail.allowed_next_states),
        artifacts=list(detail.artifacts),
        recruiter_events=list(detail.recruiter_events),
        pending_checkpoints=list(detail.pending_checkpoints),
        pending_approvals=[
            PendingApprovalItem.from_pending(approval)
            for approval in services.workflows.approvals.list_pending()
            if approval.thread_id == thread_id
        ],
        history=list(detail.history),
    )


@router.post("/applications/{application_id}/transition", response_model=TransitionResponse)
async def transition_application(
    application_id: UUID,
    http_request: Request,
    http_response: Response,
    request: TransitionRequest,
    services: ApplicationServices = Depends(get_application_services),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
):
    """Move an application to another lifecycle state.

    An illegal move is a 422 whose body lists `allowed_next_states`. A move
    with no `effect` is applied at once (200). A move with an `effect` is
    put to the policy engine -- a denial is a 403 -- and otherwise queued for
    approval (202); a second one while the first is still in flight is a 409.
    """
    actor_id = x_actor_id or _DEFAULT_ACTOR
    card = _card_or_404(services, application_id)
    check_manual_transition(
        card.status, request.to, resume_status=card.resume_status, effect=request.effect
    )

    if request.effect is None:
        state = services.lifecycle.transition(
            application_id, request.to, reason=request.reason, actor=actor_id
        )
        logger.info(
            "application %s moved %s -> %s by %s",
            application_id,
            card.status.value,
            request.to.value,
            actor_id,
        )
        return TransitionResponse(
            outcome=TransitionOutcomeKind.APPLIED,
            from_status=card.status,
            to_status=request.to,
            application=_card_or_404(services, application_id),
            event_id=state.last_event_id,
        )

    http_response.status_code = 202
    return _queue_effectful_transition(
        services,
        card,
        request,
        actor_id=actor_id,
        correlation_id=getattr(http_request.state, "correlation_id", None) or uuid4(),
    )


def _queue_effectful_transition(
    services: ApplicationServices,
    card: ApplicationCard,
    request: TransitionRequest,
    *,
    actor_id: str,
    correlation_id: UUID,
) -> TransitionResponse:
    """Policy-check a move's effect and hand it to the worker; see the module docstring."""
    application_id = card.application_id
    workflows = services.workflows
    thread = register_application_transition_thread(
        user_id=card.user_id,
        application_id=application_id,
        registry=workflows.registry,
        actor_id=actor_id,
        correlation_id=correlation_id,
    )
    snapshot = workflows.reader.read(thread.workflow_id)
    if snapshot is not None and snapshot.status in _IN_FLIGHT:
        raise Conflict(
            f"a transition on application {application_id} is already in flight "
            f"(workflow {thread.workflow_id}: {snapshot.status.value}); wait for it, or "
            f"answer its approval, before requesting another"
        )

    is_submission = request.effect is TransitionEffect.SUBMIT_APPLICATION
    transition = RequestedTransition(
        application_id=application_id,
        from_status=card.status,
        to_status=request.to,
        effect=request.effect,
        reason=request.reason,
        requested_by=actor_id,
        company=card.company,
        title=card.title,
        posting_url=card.url,
        recipient=request.recipient,
        message=(
            request.message or services.board.cover_note(application_id)
            if is_submission
            else request.message
        ),
        attachments=(
            tuple(services.board.submission_attachments(application_id)) if is_submission else ()
        ),
    )
    intent = transition_action_intent(transition)
    verdict = services.policy.evaluate_action(
        actor_id,
        thread.workflow_id,
        ACTION_TOOLS[intent.kind],
        intent.fingerprint(),
        list(intent.risk_profile().scopes),
        Provenance(origin=IntentOrigin.USER, requested_by=actor_id),
        action_id=intent.action_id,
    )
    if verdict.decision == Decision.DENY:
        raise PolicyDeniedError(
            f"policy denied '{request.effect.value}' on application {application_id}: "
            f"{verdict.reason}"
        )

    event_id = services.board.record_transition_request(transition)
    command = workflows.commands.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.START,
            workflow_id=thread.workflow_id,
            thread_id=thread.thread_id,
            actor_id=actor_id,
            correlation_id=correlation_id,
            graph_input={
                "user_id": str(card.user_id),
                "requested_transitions": [transition.model_dump(mode="json")],
            },
        )
    )
    logger.info(
        "queued %s for application %s (%s -> %s) by %s on thread '%s' (command_id=%s, policy=%s)",
        request.effect.value,
        application_id,
        card.status.value,
        request.to.value,
        actor_id,
        thread.thread_id,
        command.command_id,
        verdict.decision.value,
    )
    return TransitionResponse(
        outcome=TransitionOutcomeKind.QUEUED,
        from_status=card.status,
        to_status=request.to,
        effect=request.effect,
        application=card,
        event_id=event_id,
        workflow_id=thread.workflow_id,
        thread_id=thread.thread_id,
        command_id=command.command_id,
        policy_decision=verdict.decision,
        action=recommended_action(
            intent, workflow_id=thread.workflow_id, thread_id=thread.thread_id
        ),
    )


__all__ = [
    "router",
    "ApplicationServices",
    "build_application_services",
    "get_application_services",
    "BoardColumnView",
    "BoardResponse",
    "ApplicationDetailResponse",
    "TransitionRequest",
    "TransitionOutcomeKind",
    "TransitionResponse",
]
