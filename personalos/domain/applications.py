"""The application board, and the status changes a person asks for directly.

Two things live here, both pure:

**The board's read model.** `ApplicationCard`, `ApplicationBoard` and
`ApplicationDetail` are what `GET /v1/applications` and
`GET /v1/applications/{id}` answer with, built by
`personalos.persistence.application_board` from the status projection
(`application_status_view`) and the rows that hang off an application. Nothing
here knows where those rows came from.

**Manual transitions.** A person moving a card on the board is a caller like
any other: `check_manual_transition` runs the move through the same lifecycle
table `ApplicationRepository.update_status` enforces, and refuses STALLED for
the reason `parse_recommended_status` does -- it is a fact about the clock, set
by the stall check, not something to be asserted.

Most manual moves only record something that already happened ("I applied on
their site", "I'm skipping this one"); those are applied at once. A move that
also asks this system to *do* something outside it -- submit the application,
tell the recruiter -- carries a `TransitionEffect`. That one is not applied by
the API at all: it becomes a `RequestedTransition`, the job-search graph turns
it into an `ActionIntent` with `transition_action_intent`, and it leaves through
the approval triple like every other outward-facing write. The status changes
only once the action has run.
"""

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personalos.domain.checkpoints import PendingCheckpoint
from personalos.domain.errors import ValidationFailed
from personalos.domain.job_search import (
    PAYLOAD_ATTACHMENTS,
    PAYLOAD_BODY,
    PAYLOAD_RECIPIENT,
    PAYLOAD_SUBJECT,
    ActionIntent,
    ActionKind,
    JobMatch,
)
from personalos.domain.models import (
    ApplicationStatus,
    InvalidApplicationTransition,
    allowed_application_transitions,
)
from personalos.domain.recruiter_events import Commitment

#: `event_log.event_type` recording the scored match an application was found with.
APPLICATION_MATCH_RECORDED_EVENT = "application.match_recorded"
#: `event_log.event_type` recording a transition handed to the graph because
#: it implies an external effect. The status change itself is a separate
#: `application.status_changed`, written only once the effect has run.
APPLICATION_TRANSITION_REQUESTED_EVENT = "application.transition_requested"

#: Board columns, in lifecycle order: `ApplicationStatus` is declared in the
#: order an application moves through it, holding states last.
BOARD_COLUMN_ORDER: tuple[ApplicationStatus, ...] = tuple(ApplicationStatus)

#: Page size when the caller names none, and the most one page may hold.
DEFAULT_BOARD_PAGE_SIZE = 50
MAX_BOARD_PAGE_SIZE = 200


# --- Manual transitions ---------------------------------------------------------


class TransitionEffect(str, Enum):
    """Something outside this system a manual transition asks to have done.

    A closed set, like `ActionKind`, and each member maps onto exactly one of
    those kinds: a transition cannot invent a side effect the graph has no
    reviewed handling for.
    """

    #: Submit the prepared application (to APPLIED).
    SUBMIT_APPLICATION = "submit_application"
    #: Tell the recruiter about the decision (to WITHDRAWN, DECLINED, ACCEPTED).
    NOTIFY_RECRUITER = "notify_recruiter"


#: The statuses each effect can accompany. Submitting is how an application
#: becomes APPLIED; a message to the recruiter only makes sense for a decision
#: the candidate made.
EFFECT_TARGETS: Mapping[TransitionEffect, frozenset[ApplicationStatus]] = MappingProxyType(
    {
        TransitionEffect.SUBMIT_APPLICATION: frozenset({ApplicationStatus.APPLIED}),
        TransitionEffect.NOTIFY_RECRUITER: frozenset(
            {
                ApplicationStatus.WITHDRAWN,
                ApplicationStatus.DECLINED,
                ApplicationStatus.ACCEPTED,
            }
        ),
    }
)

#: The action kind each effect is proposed as.
EFFECT_ACTION_KINDS: Mapping[TransitionEffect, ActionKind] = MappingProxyType(
    {
        TransitionEffect.SUBMIT_APPLICATION: ActionKind.SUBMIT_APPLICATION,
        TransitionEffect.NOTIFY_RECRUITER: ActionKind.SEND_RECRUITER_MESSAGE,
    }
)


class TransitionEffectMismatch(ValidationFailed):
    """An effect was asked for alongside a status it does not belong to."""


def manual_transition_targets(
    current: ApplicationStatus, *, resume_status: ApplicationStatus | None = None
) -> frozenset[ApplicationStatus]:
    """Every status a person may move an application to from `current`.

    The lifecycle's own answer, less STALLED: see the module docstring.
    """
    return allowed_application_transitions(current, resume_status=resume_status) - {
        ApplicationStatus.STALLED
    }


def check_manual_transition(
    current: ApplicationStatus,
    new: ApplicationStatus,
    *,
    resume_status: ApplicationStatus | None = None,
    effect: TransitionEffect | None = None,
) -> ApplicationStatus:
    """Validate a move a person asked for, before anything is written or queued.

    Raises `InvalidApplicationTransition` -- carrying the statuses that *are*
    allowed -- for a move off the lifecycle, and `TransitionEffectMismatch` for
    an effect that does not go with `new`. The store re-validates against the
    row when it writes; this is the early answer, not the only check.
    """
    allowed = manual_transition_targets(current, resume_status=resume_status)
    if new is ApplicationStatus.STALLED:
        raise InvalidApplicationTransition(
            "'stalled' is set by the scheduled stall check and cannot be requested",
            allowed_next_states=allowed,
        )
    if new not in allowed:
        raise InvalidApplicationTransition(
            f"cannot transition application from '{current.value}' to '{new.value}'",
            allowed_next_states=allowed,
        )
    if effect is not None and new not in EFFECT_TARGETS[effect]:
        targets = ", ".join(sorted(status.value for status in EFFECT_TARGETS[effect]))
        raise TransitionEffectMismatch(
            f"effect '{effect.value}' only accompanies a move to: {targets}"
        )
    return new


class RequestedTransition(BaseModel):
    """A manual transition with an external effect, on its way through approval.

    Everything the effect needs is read from the application's own rows when
    the request is made and carried here, so the graph node that proposes the
    action builds it from this value alone. `request_id` doubles as the
    proposed action's `action_id`, which is how the node that applies the
    status change afterwards finds the receipt that says the effect happened.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: UUID = Field(default_factory=uuid4)
    application_id: UUID
    #: The status the request was validated against.
    from_status: ApplicationStatus
    to_status: ApplicationStatus
    effect: TransitionEffect
    reason: str | None = None
    #: The actor who asked: `X-Actor-Id` on the request, never a body field.
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    company: str
    title: str
    posting_url: str | None = None
    #: Who a recruiter notification goes to.
    recipient: str | None = None
    #: The message body: the note to the recruiter, or the submission's cover note.
    message: str | None = None
    #: What a submission attaches, as `ActionIntent` attachments.
    attachments: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def _effect_is_complete(self) -> "RequestedTransition":
        if self.to_status not in EFFECT_TARGETS[self.effect]:
            raise TransitionEffectMismatch(
                f"effect '{self.effect.value}' does not accompany a move to "
                f"'{self.to_status.value}'"
            )
        if self.effect is TransitionEffect.NOTIFY_RECRUITER and not (
            (self.recipient or "").strip() and (self.message or "").strip()
        ):
            raise ValidationFailed("notifying the recruiter needs a recipient and a message")
        if self.effect is TransitionEffect.SUBMIT_APPLICATION and not self.attachments:
            raise ValidationFailed(
                "there is nothing to submit: the application has no tailored resume or "
                "cover letter on record"
            )
        return self


def transition_action_intent(request: RequestedTransition) -> ActionIntent:
    """The `ActionIntent` a requested transition's effect is proposed as.

    Pure, so the hash a reviewer approves is a function of the request alone:
    the same request proposes the same action, on any process, after any
    restart.
    """
    kind = EFFECT_ACTION_KINDS[request.effect]
    role = f"{request.title} at {request.company}"
    if request.effect is TransitionEffect.SUBMIT_APPLICATION:
        target = request.posting_url or f"{request.company} / {request.title}"
        summary = f"Submit an application to {request.company} for {request.title}"
        payload: dict[str, Any] = {
            "application_id": str(request.application_id),
            "company": request.company,
            "title": request.title,
            "url": request.posting_url,
            "artifact_types": [item.get("artifact_type") for item in request.attachments],
            PAYLOAD_RECIPIENT: target,
            PAYLOAD_SUBJECT: f"Application for {role}",
            PAYLOAD_BODY: request.message or "",
            PAYLOAD_ATTACHMENTS: [dict(item) for item in request.attachments],
        }
    else:
        target = request.recipient or ""
        summary = (
            f"Tell {request.recipient} the application for {role} is now "
            f"{request.to_status.value}"
        )
        payload = {
            "application_id": str(request.application_id),
            "to_status": request.to_status.value,
            PAYLOAD_RECIPIENT: request.recipient,
            PAYLOAD_SUBJECT: role,
            PAYLOAD_BODY: request.message,
        }
    return ActionIntent(
        action_id=request.request_id,
        kind=kind,
        target=target,
        summary=summary,
        payload=payload,
        idempotency_key=f"transition-{request.request_id}",
        requested_by=request.requested_by,
        created_at=request.requested_at,
    )


class TransitionOutcome(BaseModel):
    """What became of one requested transition once its action was answered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: UUID
    application_id: UUID
    to_status: ApplicationStatus
    #: True only when the effect ran and the status change was then written.
    applied: bool
    detail: str | None = None


# --- Board read model -----------------------------------------------------------


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BoardQuery(_Value):
    """Which applications the board shows, and which page of them."""

    states: tuple[ApplicationStatus, ...] = ()
    #: Case-insensitive substring of the company name.
    company: str | None = None
    #: Inclusive bounds on when the application was created.
    created_from: datetime | None = None
    created_to: datetime | None = None
    user_id: UUID | None = None
    limit: int = Field(default=DEFAULT_BOARD_PAGE_SIZE, ge=1, le=MAX_BOARD_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "BoardQuery":
        if self.created_from and self.created_to and self.created_from > self.created_to:
            raise ValidationFailed("the date range ends before it starts")
        return self


class ApplicationCard(_Value):
    """One application as the board shows it."""

    application_id: UUID
    user_id: UUID
    job_posting_id: UUID
    company: str
    title: str
    location: str | None = None
    url: str | None = None
    #: From the status projection.
    status: ApplicationStatus
    resume_status: ApplicationStatus | None = None
    applied_at: datetime | None = None
    last_activity_at: datetime | None = None
    created_at: datetime
    #: When the projection last moved; the creation time if it never has.
    status_updated_at: datetime
    #: The `event_log` row the projection reflects; `None` if never transitioned.
    last_event_id: UUID | None = None


class BoardColumn(_Value):
    """Every card in one lifecycle state on this page, and how many match in all."""

    status: ApplicationStatus
    total: int
    applications: tuple[ApplicationCard, ...] = ()


class ApplicationBoard(_Value):
    """The board: one column per state, a page of cards across them."""

    columns: tuple[BoardColumn, ...]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class RecordedMatch(_Value):
    """The scored match an application was found with, as recorded."""

    match: JobMatch
    recorded_at: datetime
    recorded_by: str | None = None


class ArtifactSummary(_Value):
    """One tailored document linked to the application. The text itself is not here."""

    artifact_version_id: UUID
    artifact_type: str
    version: int
    doc_id: str | None = None
    generated_by: str | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    created_at: datetime


class RecruiterEventSummary(_Value):
    """One recruiter-side signal recorded against the application."""

    communication_event_id: UUID
    classification: str
    provider_message_id: str | None = None
    occurred_at: datetime
    commitments: tuple[Commitment, ...] = ()


class StatusChange(_Value):
    """One `application.status_changed` event, as history."""

    event_id: UUID
    from_status: ApplicationStatus | None = None
    to_status: ApplicationStatus
    actor: str | None = None
    reason: str | None = None
    occurred_at: datetime


class ApplicationDetail(_Value):
    """Everything about one application a person deciding what to do next needs."""

    card: ApplicationCard
    #: Where a person may move it from here.
    allowed_next_states: tuple[ApplicationStatus, ...]
    match: RecordedMatch | None = None
    artifacts: tuple[ArtifactSummary, ...] = ()
    recruiter_events: tuple[RecruiterEventSummary, ...] = ()
    #: Durable waits still open on this application.
    pending_checkpoints: tuple[PendingCheckpoint, ...] = ()
    history: tuple[StatusChange, ...] = ()

    @property
    def fit_score(self) -> float | None:
        """The recorded match's score, if a match was recorded."""
        return self.match.match.score if self.match else None


__all__ = [
    "APPLICATION_MATCH_RECORDED_EVENT",
    "APPLICATION_TRANSITION_REQUESTED_EVENT",
    "BOARD_COLUMN_ORDER",
    "DEFAULT_BOARD_PAGE_SIZE",
    "MAX_BOARD_PAGE_SIZE",
    "EFFECT_ACTION_KINDS",
    "EFFECT_TARGETS",
    "TransitionEffect",
    "TransitionEffectMismatch",
    "manual_transition_targets",
    "check_manual_transition",
    "RequestedTransition",
    "transition_action_intent",
    "TransitionOutcome",
    "BoardQuery",
    "ApplicationCard",
    "BoardColumn",
    "ApplicationBoard",
    "RecordedMatch",
    "ArtifactSummary",
    "RecruiterEventSummary",
    "StatusChange",
    "ApplicationDetail",
]
