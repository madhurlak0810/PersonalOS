"""Core domain models for PersonalOS."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

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
    description: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Job search specific
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    job_type: Optional[str] = None  # full-time, part-time, contract, etc.
    
    # Results
    results_count: int = 0
    results: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AgentState(BaseModel):
    """State of an agent during execution."""

    agent_id: UUID
    job_id: UUID
    current_step: str
    step_data: Dict[str, Any] = Field(default_factory=dict)
    history: list[Dict[str, Any]] = Field(default_factory=list)
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
    agent_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class Tool(BaseModel):
    """A tool that an agent can use."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    mcp_server: Optional[str] = None  # Which MCP server provides this tool
    
    class Config:
        use_enum_values = True


class OperationStatus(str, Enum):
    """Lifecycle of a recorded mutating operation."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MutatingIntent(BaseModel):
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

    def side_effect_params(self) -> Dict[str, Any]:
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
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    attempts: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def is_replayable(self) -> bool:
        """True when a prior result exists and can be returned as-is."""
        return self.status == OperationStatus.COMPLETED

    class Config:
        use_enum_values = False


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    available_tools: list[str] = Field(default_factory=list)  # Tool names
    model: str = "gpt-4"
    temperature: float = 0.7
    max_iterations: int = 10
    timeout_seconds: int = 300
    
    class Config:
        use_enum_values = True
