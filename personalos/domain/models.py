"""Core domain models for PersonalOS."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
