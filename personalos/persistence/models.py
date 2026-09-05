"""Database models using SQLAlchemy ORM."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class GUID(TypeDecorator):
    """Platform-independent UUID column.

    Uses PostgreSQL's native UUID type where available and falls back to a
    36-character string elsewhere, so the operation log can be exercised against
    SQLite in tests without a Postgres instance.
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, UUID) else UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, UUID) else UUID(str(value))


class JobModel(Base):
    """ORM model for Job."""

    __tablename__ = "jobs"

    id = Column(GUID(), primary_key=True, default=uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum("pending", "running", "completed", "failed", "cancelled", name="job_status"))
    keywords = Column(JSON, nullable=False, default=[])
    locations = Column(JSON, nullable=False, default=[])
    salary_min = Column(String, nullable=True)
    salary_max = Column(String, nullable=True)
    job_type = Column(String(50), nullable=True)
    results_count = Column(String, nullable=False, default="0")
    results = Column(JSON, nullable=False, default={})
    error_code = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    job_metadata = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "keywords": self.keywords,
            "locations": self.locations,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "job_type": self.job_type,
            "results_count": self.results_count,
            "results": self.results,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "job_metadata": self.job_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class EventModel(Base):
    """ORM model for Event."""

    __tablename__ = "events"

    id = Column(GUID(), primary_key=True, default=uuid4)
    event_type = Column(String(50), nullable=False)
    job_id = Column(GUID(), nullable=False)
    agent_id = Column(GUID(), nullable=True)
    data = Column(JSON, nullable=False, default={})
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "job_id": str(self.job_id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class OperationModel(Base):
    """ORM model for the mutating-operation log.

    One row per idempotency key. The unique constraint on `idempotency_key` is
    the dedup primitive: concurrent retries race to insert, the loser reads the
    winner's row instead of repeating the side effect.
    """

    __tablename__ = "operations"

    id = Column(GUID(), primary_key=True, default=uuid4)
    idempotency_key = Column(String(255), nullable=False, unique=True)
    operation = Column(String(255), nullable=False)
    request_fingerprint = Column(String(64), nullable=False)
    status = Column(
        Enum("in_progress", "completed", "failed", name="operation_status"),
        nullable=False,
        default="in_progress",
    )
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    attempts = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("attempts >= 1", name="ck_operations_attempts_positive"),
        Index("ix_operations_operation", "operation"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "idempotency_key": self.idempotency_key,
            "operation": self.operation,
            "request_fingerprint": self.request_fingerprint,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AgentStateModel(Base):
    """ORM model for AgentState."""

    __tablename__ = "agent_states"

    id = Column(GUID(), primary_key=True, default=uuid4)
    agent_id = Column(GUID(), nullable=False)
    job_id = Column(GUID(), nullable=False)
    current_step = Column(String(255), nullable=False)
    step_data = Column(JSON, nullable=False, default={})
    history = Column(JSON, nullable=False, default=[])
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "job_id": str(self.job_id),
            "current_step": self.current_step,
            "step_data": self.step_data,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
