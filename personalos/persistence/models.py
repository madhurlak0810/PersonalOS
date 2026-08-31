"""Database models using SQLAlchemy ORM."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class JobModel(Base):
    """ORM model for Job."""

    __tablename__ = "jobs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
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
            "job_metadata": self.job_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class EventModel(Base):
    """ORM model for Event."""

    __tablename__ = "events"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type = Column(String(50), nullable=False)
    job_id = Column(PG_UUID(as_uuid=True), nullable=False)
    agent_id = Column(PG_UUID(as_uuid=True), nullable=True)
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


class AgentStateModel(Base):
    """ORM model for AgentState."""

    __tablename__ = "agent_states"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PG_UUID(as_uuid=True), nullable=False)
    job_id = Column(PG_UUID(as_uuid=True), nullable=False)
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
