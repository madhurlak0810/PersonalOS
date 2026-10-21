"""The approval queue over HTTP: list what is waiting, decide it. Version 1.

Two endpoints, for a human acting on workflows parked at an approval:

- `GET /v1/approvals` lists every request still owed an answer, across all
  workflows, with what a reviewer needs to decide it.
- `POST /v1/approvals/{id}/decision` records the answer to one of them and,
  once its interrupt is fully answered, queues the workflow's resume.

Like the workflow routes, neither runs a graph and neither keeps state in this
process: the list is read from the checkpoints, and a decision ends at the same
`WorkflowCommandQueue.enqueue` a `POST /v1/workflows/{id}/resume` does.

**A reviewer sees the action, not a summary of it.** Each item is labelled
`action_status: AWAITING_APPROVAL` -- nothing has happened yet -- and carries
the target, the preview (recipient, subject, body diff, attachments, whether it
is a reply or a new thread) and the consequences of saying yes.

**A decision is bound to the action it was shown.** The body must carry the
`action_hash` the list returned; one that does not match the hash stored with
the request is a 409, so an approval of an action that has since changed can
never be spent on what replaced it. The decider is the request's actor
(`X-Actor-Id`), never a field of the body.
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from apps.api.routes.workflows import WorkflowServices, get_workflow_services
from personalos.domain.action_status import ActionStatus, consequences
from personalos.domain.job_search import ActionKind, ActionPreview
from personalos.domain.workflow import (
    ApprovalOutcome,
    ApprovalState,
    PendingApproval,
    RecordedApprovalDecision,
)

logger = logging.getLogger(__name__)

router = APIRouter()

#: Actor recorded when a caller does not declare one, matching the other routes.
_DEFAULT_ACTOR = "api"


# --- Request / response models --------------------------------------------------


class PendingApprovalItem(BaseModel):
    """One request a reviewer is being asked to decide."""

    id: UUID
    workflow_id: UUID
    thread_id: str
    #: `pending`, or `expired` once it can no longer be approved.
    status: ApprovalState
    action_id: UUID
    #: Always `AWAITING_APPROVAL`: an action in this list has not happened.
    action_status: ActionStatus
    action_type: str
    target: str
    summary: str
    #: Recipient, subject, body diff, attachments, reply or new thread.
    preview: ActionPreview | None = None
    #: What approving it will do, one sentence each.
    consequences: list[str] = Field(default_factory=list)
    risk_level: str
    requested_scopes: list[str] = Field(default_factory=list)
    #: Echo this back in the decision; it pins the decision to this action.
    action_hash: str
    requested_at: datetime
    expires_at: datetime

    @classmethod
    def from_pending(cls, approval: PendingApproval) -> "PendingApprovalItem":
        """Project the domain value onto the v1 wire shape."""
        return cls(
            id=approval.approval_id,
            workflow_id=approval.workflow_id,
            thread_id=approval.thread_id,
            status=approval.state,
            action_id=approval.action_id,
            action_status=ActionStatus.AWAITING_APPROVAL,
            action_type=approval.action_type,
            target=approval.target,
            summary=approval.summary,
            preview=approval.preview,
            consequences=list(
                consequences(ActionKind(approval.action_type), approval.target, approval.preview)
            ),
            risk_level=approval.risk_level,
            requested_scopes=list(approval.requested_scopes),
            action_hash=approval.action_hash,
            requested_at=approval.requested_at,
            expires_at=approval.expires_at,
        )


class PendingApprovalsResponse(BaseModel):
    """The approval queue as of the last checkpoint any process wrote."""

    approvals: list[PendingApprovalItem] = Field(default_factory=list)


class DecisionRequest(BaseModel):
    """A reviewer's answer to one approval."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    #: The `action_hash` the approval was listed with.
    action_hash: str = Field(min_length=1)


class DecisionResponse(BaseModel):
    """The decision on record -- the same body for the first submission and any repeat."""

    id: UUID
    workflow_id: UUID | None = None
    status: ApprovalState
    outcome: ApprovalOutcome
    action_hash: str
    decided_by: str | None = None
    decided_at: datetime
    #: True once the workflow's resume is queued; false while other requests
    #: raised with this one are still unanswered.
    resumed: bool

    @classmethod
    def from_recorded(cls, decision: RecordedApprovalDecision) -> "DecisionResponse":
        """Project the domain value onto the v1 wire shape."""
        return cls(
            id=decision.approval_id,
            workflow_id=decision.workflow_id,
            status=decision.state,
            outcome=decision.outcome,
            action_hash=decision.action_hash,
            decided_by=decision.decided_by,
            decided_at=decision.decided_at,
            resumed=decision.resumed,
        )


# --- Routes ---------------------------------------------------------------------


@router.get("/approvals", response_model=PendingApprovalsResponse)
async def list_approvals(
    services: WorkflowServices = Depends(get_workflow_services),
):
    """Every approval still owed an answer, oldest first; expired ones are marked."""
    return PendingApprovalsResponse(
        approvals=[
            PendingApprovalItem.from_pending(approval)
            for approval in services.approvals.list_pending()
        ]
    )


@router.post("/approvals/{approval_id}/decision", response_model=DecisionResponse)
async def decide_approval(
    approval_id: UUID,
    http_request: Request,
    request: DecisionRequest,
    services: WorkflowServices = Depends(get_workflow_services),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
):
    """Approve or reject one pending approval.

    Approving resumes the workflow through the worker; rejecting records
    `DENIED_BY_USER` and resumes it down its declined branch. A mismatched
    `action_hash`, approving an expired approval, and reversing a recorded
    decision are each a 409. Repeating a recorded decision returns it
    unchanged and resumes nothing a second time.
    """
    actor_id = x_actor_id or _DEFAULT_ACTOR
    decision = services.approvals.decide(
        approval_id,
        approved=request.approved,
        action_hash=request.action_hash,
        actor_id=actor_id,
        correlation_id=getattr(http_request.state, "correlation_id", None),
    )
    logger.info(
        "approval %s decided %s by %s (workflow_id=%s, resumed=%s, repeat=%s)",
        approval_id,
        decision.outcome.value,
        actor_id,
        decision.workflow_id,
        decision.resumed,
        decision.replayed,
    )
    return DecisionResponse.from_recorded(decision)


__all__ = [
    "router",
    "PendingApprovalItem",
    "PendingApprovalsResponse",
    "DecisionRequest",
    "DecisionResponse",
]
