"""Core domain models for PersonalOS."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.context import ExecutionContext
from personalos.domain.errors import RetryableFailure, ValidationFailed

# Minimum length for an idempotency key. Keys are supplied by callers and must
# carry enough entropy that two unrelated operations cannot collide by accident;
# a UUID4 string is the expected shape.
IDEMPOTENCY_KEY_MIN_LENGTH = 8
IDEMPOTENCY_KEY_MAX_LENGTH = 255


class InvalidIdempotencyKey(ValidationFailed, ValueError):
    """Raised when an idempotency key is missing or malformed.

    Subclasses both `ValidationFailed` (so it reports through the shared
    taxonomy like every other error) and `ValueError` (its original base,
    kept for any caller still catching that broader type).
    """


def validate_idempotency_key(key: Any) -> str:
    """Normalize and validate an idempotency key.

    Returns the stripped key. Raises InvalidIdempotencyKey if it is absent,
    not a string, or too short/long to be a usable dedup key.
    """
    if key is None:
        raise InvalidIdempotencyKey("idempotency_key is required for mutating operations")
    if not isinstance(key, str):
        raise InvalidIdempotencyKey(
            f"idempotency_key must be a string, got {type(key).__name__}"
        )

    normalized = key.strip()
    if len(normalized) < IDEMPOTENCY_KEY_MIN_LENGTH:
        raise InvalidIdempotencyKey(
            f"idempotency_key must be at least {IDEMPOTENCY_KEY_MIN_LENGTH} characters"
        )
    if len(normalized) > IDEMPOTENCY_KEY_MAX_LENGTH:
        raise InvalidIdempotencyKey(
            f"idempotency_key must be at most {IDEMPOTENCY_KEY_MAX_LENGTH} characters"
        )
    return normalized


class JobStatus(str, Enum):
    """Status of a job search task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Job search task."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str | None = None
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Job search specific
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    salary_min: int | None = None
    salary_max: int | None = None
    job_type: str | None = None  # full-time, part-time, contract, etc.

    # Results
    results_count: int = 0
    results: dict[str, Any] = Field(default_factory=dict)

    # Set when status is FAILED. `error_code` is one of the stable
    # `personalos.domain.errors.ErrorCode` values; `error_message` is the
    # already-sanitized message from the error that failed the job.
    error_code: str | None = None
    error_message: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Correlation identity for this run, carried through every intent and
    # event the executor produces while working this job.
    context: ExecutionContext = Field(default_factory=ExecutionContext.new)

    class Config:
        use_enum_values = True


class AgentState(BaseModel):
    """State of an agent during execution."""

    agent_id: UUID
    job_id: UUID
    current_step: str
    step_data: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class EventType(str, Enum):
    """Types of events in the system."""

    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    AGENT_STEP = "agent.step"
    AGENT_ERROR = "agent.error"
    RESULT_FOUND = "result.found"


class Event(BaseModel):
    """Domain event."""

    id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    job_id: UUID
    agent_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)
    context: ExecutionContext = Field(default_factory=ExecutionContext.new)

    class Config:
        use_enum_values = True


class Tool(BaseModel):
    """A tool that an agent can use."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    mcp_server: str | None = None  # Which MCP server provides this tool

    class Config:
        use_enum_values = True


class OperationStatus(str, Enum):
    """Lifecycle of a recorded mutating operation."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Intent(BaseModel):
    """Base contract for a tool's typed parameters.

    Every tool registers the Intent subclass describing the parameters it
    accepts. A caller's raw params are validated into that subclass before
    the handler ever runs, so a malformed call fails as a typed validation
    error instead of reaching tool code. `extra="forbid"` rejects unknown
    fields rather than silently ignoring them.
    """

    model_config = ConfigDict(extra="forbid")


class ActionTarget(BaseModel):
    """Identifies which server and tool a call is directed at."""

    server: str
    tool: str

    @field_validator("server", "tool")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value


class MutatingIntent(Intent):
    """Base contract for any intent that produces a side effect.

    Every mutating intent must carry an idempotency key so the operation can be
    retried safely: a replay of the same key returns the original result instead
    of executing the side effect a second time.
    """

    idempotency_key: str = Field(
        ...,
        description="Caller-supplied key that uniquely identifies this operation attempt",
    )

    @field_validator("idempotency_key")
    @classmethod
    def _check_idempotency_key(cls, value: str) -> str:
        return validate_idempotency_key(value)

    def side_effect_params(self) -> dict[str, Any]:
        """Parameters that define the side effect, excluding the idempotency key.

        Used to fingerprint the request so a key replayed with different
        parameters is rejected rather than silently returning the wrong result.
        """
        return self.model_dump(exclude={"idempotency_key"}, mode="json")


class OperationRecord(BaseModel):
    """Durable record of a mutating operation, keyed by idempotency key."""

    id: UUID = Field(default_factory=uuid4)
    idempotency_key: str
    operation: str
    request_fingerprint: str
    status: OperationStatus = OperationStatus.IN_PROGRESS
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    @property
    def is_replayable(self) -> bool:
        """True when a prior result exists and can be returned as-is."""
        return self.status == OperationStatus.COMPLETED

    class Config:
        use_enum_values = False


class ToolCallRequest(BaseModel):
    """Typed envelope for invoking a tool.

    `params` are raw, caller-supplied values for the target tool's fields;
    the server validates them into that tool's registered Intent subclass
    before execution, so this is the one place a dict is still allowed to
    cross the boundary — everything past it is typed.
    """

    target: ActionTarget
    params: dict[str, Any] = Field(default_factory=dict)
    context: ExecutionContext = Field(default_factory=ExecutionContext.new)


class ToolCallErrorCode(str, Enum):
    """Typed failure categories for a tool call."""

    SERVER_NOT_FOUND = "server_not_found"
    TOOL_NOT_FOUND = "tool_not_found"
    VALIDATION_ERROR = "validation_error"
    MISSING_OPERATION_STORE = "missing_operation_store"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    EXECUTION_ERROR = "execution_error"


class ToolCallError(BaseModel):
    """Typed failure detail for a tool call that did not succeed."""

    code: ToolCallErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Typed outcome of a tool call.

    Exactly one of `result` / `error` is populated, selected by `ok`. Mutating
    calls additionally carry the idempotency key and whether the result was
    replayed from a prior attempt rather than freshly executed.
    """

    target: ActionTarget
    ok: bool
    result: dict[str, Any] | None = None
    error: ToolCallError | None = None
    idempotency_key: str | None = None
    replayed: bool | None = None

    @classmethod
    def succeeded(
        cls,
        target: ActionTarget,
        result: dict[str, Any],
        idempotency_key: str | None = None,
        replayed: bool | None = None,
    ) -> "ToolCallResult":
        """Build a successful result."""
        return cls(
            target=target,
            ok=True,
            result=result,
            idempotency_key=idempotency_key,
            replayed=replayed,
        )

    @classmethod
    def failed(
        cls,
        target: ActionTarget,
        code: ToolCallErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "ToolCallResult":
        """Build a failed result with a typed error."""
        return cls(
            target=target,
            ok=False,
            error=ToolCallError(code=code, message=message, details=details or {}),
        )


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    available_tools: list[str] = Field(default_factory=list)  # Tool names
    model: str = "gpt-4"
    temperature: float = 0.7
    max_iterations: int = 10
    timeout_seconds: int = 300

    class Config:
        use_enum_values = True


class ApplicationStatus(str, Enum):
    """Lifecycle of a tracked job application.

    Only the states an application is allowed to occupy; which moves between
    them are legal is `ALLOWED_APPLICATION_TRANSITIONS` below. Nothing writes
    to `applications.status` directly — every change goes through
    `validate_application_status_transition`, so an LLM-driven caller can
    request a move but never set the column outright.
    """

    DISCOVERED = "discovered"
    SAVED = "saved"
    PREPARING = "preparing"
    READY_TO_APPLY = "ready_to_apply"
    APPLIED = "applied"
    RESPONSE = "response"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    SKIPPED = "skipped"
    FOLLOW_UP_PENDING = "follow_up_pending"
    STALLED = "stalled"


#: Documented lifecycle: DISCOVERED -> SAVED -> PREPARING -> READY_TO_APPLY ->
#: APPLIED -> (RESPONSE) -> INTERVIEWING -> OFFER -> ACCEPTED | DECLINED.
#: RESPONSE is optional: an interview invite or a rejection can be the first
#: thing that comes back. SKIPPED is reachable from any pre-APPLIED state (the
#: candidate drops the lead before applying); REJECTED/WITHDRAWN from any
#: post-APPLIED one. FOLLOW_UP_PENDING is entered while the other side is
#: owed or owes a nudge, and STALLED from any state in
#: `STALLABLE_APPLICATION_STATUSES`.
#:
#: The two holding states have no row of their own: where an application may
#: go from one depends on where it was held *from*, which this table cannot
#: know. See `allowed_application_transitions`.
ALLOWED_APPLICATION_TRANSITIONS: dict[ApplicationStatus, frozenset[ApplicationStatus]] = {
    ApplicationStatus.DISCOVERED: frozenset(
        {ApplicationStatus.SAVED, ApplicationStatus.SKIPPED}
    ),
    ApplicationStatus.SAVED: frozenset(
        {
            ApplicationStatus.PREPARING,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.PREPARING: frozenset(
        {
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.READY_TO_APPLY: frozenset(
        {
            ApplicationStatus.APPLIED,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.APPLIED: frozenset(
        {
            ApplicationStatus.RESPONSE,
            ApplicationStatus.INTERVIEWING,
            ApplicationStatus.REJECTED,
            ApplicationStatus.WITHDRAWN,
            ApplicationStatus.FOLLOW_UP_PENDING,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.RESPONSE: frozenset(
        {
            ApplicationStatus.INTERVIEWING,
            ApplicationStatus.REJECTED,
            ApplicationStatus.WITHDRAWN,
            ApplicationStatus.FOLLOW_UP_PENDING,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.INTERVIEWING: frozenset(
        {
            ApplicationStatus.OFFER,
            ApplicationStatus.REJECTED,
            ApplicationStatus.WITHDRAWN,
            ApplicationStatus.FOLLOW_UP_PENDING,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.OFFER: frozenset(
        {
            ApplicationStatus.ACCEPTED,
            ApplicationStatus.DECLINED,
            ApplicationStatus.WITHDRAWN,
            ApplicationStatus.STALLED,
        }
    ),
    ApplicationStatus.ACCEPTED: frozenset(),
    ApplicationStatus.DECLINED: frozenset(),
    ApplicationStatus.REJECTED: frozenset(),
    ApplicationStatus.WITHDRAWN: frozenset(),
    ApplicationStatus.SKIPPED: frozenset(),
}

#: States an application is parked in rather than progressing through. Leaving
#: one is governed by the status it was entered from (`resume_status`), not by
#: a fixed set of successors.
HOLDING_APPLICATION_STATUSES: frozenset[ApplicationStatus] = frozenset(
    {ApplicationStatus.FOLLOW_UP_PENDING, ApplicationStatus.STALLED}
)

#: States with nothing after them.
TERMINAL_APPLICATION_STATUSES: frozenset[ApplicationStatus] = frozenset(
    status for status, successors in ALLOWED_APPLICATION_TRANSITIONS.items() if not successors
)

#: States the scheduled stall check may move to STALLED. DISCOVERED is left
#: out on purpose: a posting nobody has acted on yet is not a pursuit that
#: went quiet, and stalling it would flag every unread lead after one window.
STALLABLE_APPLICATION_STATUSES: frozenset[ApplicationStatus] = frozenset(
    {
        status
        for status, successors in ALLOWED_APPLICATION_TRANSITIONS.items()
        if ApplicationStatus.STALLED in successors
    }
    | {ApplicationStatus.FOLLOW_UP_PENDING}
)


class InvalidApplicationTransition(ValidationFailed, ValueError):
    """Raised when an application status change is not on the documented lifecycle path.

    Subclasses both `ValidationFailed` (reports through the shared error
    taxonomy) and `ValueError` (its original base), matching
    `InvalidIdempotencyKey` above.
    """


class ApplicationTransitionConflict(RetryableFailure):
    """Raised when an application moved between being read and being written.

    The transition was valid against the status that was read, but another
    writer got there first. Nothing was changed and no event was emitted;
    re-reading and deciding again is safe.
    """


#: `event_log.aggregate_type` for an application's lifecycle events.
APPLICATION_AGGREGATE_TYPE = "application"
#: `event_log.event_type` emitted by every status transition.
APPLICATION_STATUS_CHANGED_EVENT = "application.status_changed"
#: `event_log.payload_json["actor"]` for transitions made by the stall check.
STALL_MONITOR_ACTOR = "stall_monitor"


def allowed_application_transitions(
    current: ApplicationStatus, *, resume_status: ApplicationStatus | None = None
) -> frozenset[ApplicationStatus]:
    """Every status reachable from `current` in one step.

    For a holding state that is the status it was entered from plus whatever
    that status could have moved to -- so a held application resumes or moves
    on exactly as if it had never been held, and DISCOVERED -> ... -> STALLED
    -> OFFER is no more possible than DISCOVERED -> OFFER. A holding state
    with no recorded `resume_status` has nowhere to go: guessing one would be
    setting a status nobody validated.
    """
    if current not in HOLDING_APPLICATION_STATUSES:
        return ALLOWED_APPLICATION_TRANSITIONS.get(current, frozenset())
    if resume_status is None or resume_status in HOLDING_APPLICATION_STATUSES:
        return frozenset()
    return (ALLOWED_APPLICATION_TRANSITIONS[resume_status] | {resume_status}) - {current}


def validate_application_status_transition(
    current: ApplicationStatus,
    new: ApplicationStatus,
    *,
    resume_status: ApplicationStatus | None = None,
) -> ApplicationStatus:
    """Check that `current -> new` is an allowed step on the application lifecycle.

    Returns `new` on success so callers can assign the result inline. Raises
    `InvalidApplicationTransition` for anything else, including re-asserting
    the current status — every transition must be an explicit, documented
    move, not a status set directly by a caller (e.g. an LLM) without going
    through this check.

    `resume_status` is the status a held application was held from; it is
    ignored unless `current` is a holding state.
    """
    if new not in allowed_application_transitions(current, resume_status=resume_status):
        raise InvalidApplicationTransition(
            f"cannot transition application from '{current.value}' to '{new.value}'"
        )
    return new


def resume_status_after(
    current: ApplicationStatus,
    new: ApplicationStatus,
    *,
    resume_status: ApplicationStatus | None = None,
) -> ApplicationStatus | None:
    """The `resume_status` an application carries once `current -> new` is applied.

    Entering a holding state remembers where from; moving between the two
    holding states keeps the original, since FOLLOW_UP_PENDING is not
    somewhere a stalled application should resume *to*; leaving clears it.
    """
    if new not in HOLDING_APPLICATION_STATUSES:
        return None
    return resume_status if current in HOLDING_APPLICATION_STATUSES else current


class ApplicationTransitionRecommendation(BaseModel):
    """A status move proposed by something that is not allowed to make it.

    What an LLM-driven caller hands the lifecycle instead of a status: `to` is
    free text until `parse_recommended_status` has checked it names a real
    state, and the move itself is still subject to
    `validate_application_status_transition` against the application's
    stored status. Nothing on this model is ever written to a row as-is.
    """

    application_id: UUID
    to: str
    reason: str | None = None


def parse_recommended_status(
    recommendation: ApplicationTransitionRecommendation,
) -> ApplicationStatus:
    """The status a recommendation names, if it is one a caller may ask for.

    STALLED is refused here even where the lifecycle allows the edge: it is a
    fact about the clock, established by the scheduled stall check, not
    something to be argued into from a conversation.
    """
    try:
        status = ApplicationStatus(recommendation.to.strip().lower())
    except ValueError as exc:
        raise InvalidApplicationTransition(
            f"'{recommendation.to}' is not an application status"
        ) from exc
    if status is ApplicationStatus.STALLED:
        raise InvalidApplicationTransition(
            "'stalled' is set by the scheduled stall check and cannot be recommended"
        )
    return status


class ApplicationLifecycleState(BaseModel):
    """Where an application is on its lifecycle, read from the database alone."""

    model_config = ConfigDict(frozen=True)

    application_id: UUID
    status: ApplicationStatus
    #: Set only while `status` is a holding state.
    resume_status: ApplicationStatus | None = None
    last_activity_at: datetime
    #: The `event_log` row the status projection currently reflects; `None`
    #: for an application that has never transitioned.
    last_event_id: UUID | None = None


def is_application_stalled(
    status: ApplicationStatus,
    last_activity_at: datetime,
    *,
    now: datetime,
    window: timedelta,
) -> bool:
    """Whether an application has gone a full `window` with no activity."""
    return status in STALLABLE_APPLICATION_STATUSES and last_activity_at <= now - window


class ArtifactType(str, Enum):
    """Kind of tailored document an `artifact_versions` row holds."""

    RESUME = "resume"
    COVER_LETTER = "cover_letter"


class InvalidEvidenceLinkage(ValidationFailed, ValueError):
    """Raised when a generated artifact does not cite the evidence it was built from."""


def validate_evidence_links(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require an artifact version to cite at least one concrete evidence source.

    Each entry must name a `type` (e.g. "resume_section", "project") and a
    `ref` identifying which one, so a generated resume/cover-letter draft can
    never carry a claim that doesn't trace back to something in the
    candidate's own record.
    """
    if not evidence:
        raise InvalidEvidenceLinkage(
            "artifact version must cite at least one evidence source"
        )
    for item in evidence:
        if not isinstance(item, dict) or not item.get("type") or not item.get("ref"):
            raise InvalidEvidenceLinkage(
                "each evidence entry must have a non-blank 'type' and 'ref'"
            )
    return evidence


class CommunicationEventClassification(str, Enum):
    """How a recruiter-side message tied to an application was classified.

    Recorded for every inbound communication so an application's history can
    be read without re-parsing message content, even though the standalone
    Communications Agent that will produce these classifications is out of
    scope for this build — this table only captures its signal.
    """

    RECRUITER_RESPONSE = "recruiter_response"
    INTERVIEW_INVITE = "interview_invite"
    REJECTION = "rejection"
    OFFER = "offer"
    ACTION_REQUIRED = "action_required"
    GENERAL_UPDATE = "general_update"


class ToolExecutionStatus(str, Enum):
    """Lifecycle of one recorded tool-call attempt, keyed by idempotency key.

    Mirrors `OperationStatus`'s values under a table-scoped name, matching how
    `JobStatus` and `WorkflowRunStatus`-shaped columns elsewhere each get
    their own enum rather than sharing one across tables.
    """

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    #: A previous attempt claimed the key and left no outcome, so the action
    #: may or may not have taken effect. Not retryable on its own: it has to
    #: be reconciled against the provider first.
    UNKNOWN = "unknown"


class PolicyDecisionOutcome(str, Enum):
    """Verdict recorded for one proposed tool call in `policy_decisions`.

    Mirrors `personalos.policy.intents.Decision`'s values; duplicated rather
    than imported so the persistence layer does not depend on the policy
    layer (policy depends on domain and persistence depends on domain, never
    the reverse).
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class AuditEventResult(str, Enum):
    """Outcome of the action an `audit_events` row records."""

    SUCCESS = "success"
    FAILURE = "failure"


class OutboxEventStatus(str, Enum):
    """Lifecycle of one row in the transactional outbox.

    PENDING rows are written in the same transaction as the domain mutation
    that produced them. A worker atomically claims a PENDING row (moving it
    to IN_PROGRESS) before dispatching it elsewhere, mirroring
    `ToolExecutionStatus`'s claim/complete/fail shape so exactly one worker
    ever owns a given row at a time.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DISPATCHED = "dispatched"
    FAILED = "failed"


class EvidenceSourceType(str, Enum):
    """What kind of candidate material an `evidence_chunks` row was cut from.

    Distinguishes resume content from project write-ups so a retrieval query
    can restrict to one kind of evidence when grounding a generated claim
    (see `personalos.persistence.models.EvidenceChunkModel`).
    """

    RESUME = "resume"
    PROJECT = "project"
