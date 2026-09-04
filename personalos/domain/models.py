"""Core domain models for PersonalOS."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Minimum length for an idempotency key. Keys are supplied by callers and must
# carry enough entropy that two unrelated operations cannot collide by accident;
# a UUID4 string is the expected shape.
IDEMPOTENCY_KEY_MIN_LENGTH = 8
IDEMPOTENCY_KEY_MAX_LENGTH = 255


class InvalidIdempotencyKey(ValueError):
    """Raised when an idempotency key is missing or malformed."""


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

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

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
