"""Database models using SQLAlchemy ORM."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
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

    # Correlation identity (see `personalos.domain.context.ExecutionContext`),
    # persisted so a worker loading this job in a separate process from the one
    # that created it still runs with the same workflow/correlation identity.
    workflow_id = Column(GUID(), nullable=False, default=uuid4)
    run_id = Column(GUID(), nullable=False, default=uuid4)
    correlation_id = Column(GUID(), nullable=False, default=uuid4)
    actor_id = Column(String(255), nullable=False, default="system")

    __table_args__ = (Index("ix_jobs_correlation_id", "correlation_id"),)

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
            "workflow_id": str(self.workflow_id),
            "run_id": str(self.run_id),
            "correlation_id": str(self.correlation_id),
            "actor_id": self.actor_id,
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


# --- Workflow orchestration schema -----------------------------------------
#
# The tables below back any LangGraph-based workflow, independent of domain:
# who ran it (users), what was run (workflows), each execution attempt
# (workflow_runs) and its steps (workflow_steps), the durable checkpointer
# state LangGraph needs to resume a run (checkpoints), and the human sign-off
# a mutating action needed before it executed (approvals).


class UserModel(Base):
    """ORM model for a person or service identity that can initiate or approve work."""

    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid4)
    email = Column(String(255), nullable=True, unique=True)
    display_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "display_name": self.display_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class WorkflowModel(Base):
    """ORM model for a workflow definition, e.g. 'job_search'."""

    __tablename__ = "workflows"

    id = Column(GUID(), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_workflows_name", "name"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class WorkflowRunModel(Base):
    """ORM model for one execution attempt of a workflow.

    `thread_id` identifies the LangGraph thread this run drives and is the key
    the durable checkpointer resumes by.
    """

    __tablename__ = "workflow_runs"

    id = Column(GUID(), primary_key=True, default=uuid4)
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=False)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    thread_id = Column(String(255), nullable=False, unique=True, default=lambda: str(uuid4()))
    status = Column(
        Enum("pending", "running", "completed", "failed", "cancelled", name="workflow_run_status"),
        nullable=False,
        default="pending",
    )
    # Correlation identity (see `personalos.domain.context.ExecutionContext`)
    # for the run, so every log line and event it produces traces back here.
    correlation_id = Column(GUID(), nullable=True)
    actor_id = Column(String(255), nullable=False, default="system")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_workflow_runs_workflow_id", "workflow_id"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "workflow_id": str(self.workflow_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "thread_id": self.thread_id,
            "status": self.status,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "actor_id": self.actor_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class WorkflowStepModel(Base):
    """ORM model for one step within a workflow run."""

    __tablename__ = "workflow_steps"

    id = Column(GUID(), primary_key=True, default=uuid4)
    workflow_run_id = Column(GUID(), ForeignKey("workflow_runs.id"), nullable=False)
    step_name = Column(String(255), nullable=False)
    sequence = Column(Integer, nullable=False, default=0)
    status = Column(
        Enum("pending", "running", "completed", "failed", "skipped", name="workflow_step_status"),
        nullable=False,
        default="pending",
    )
    input_data = Column(JSON, nullable=False, default={})
    output_data = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_workflow_steps_workflow_run_id", "workflow_run_id"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "workflow_run_id": str(self.workflow_run_id),
            "step_name": self.step_name,
            "sequence": self.sequence,
            "status": self.status,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CheckpointModel(Base):
    """ORM model backing the LangGraph durable checkpointer.

    Keyed by `thread_id` / `checkpoint_ns` / `checkpoint_id`, mirroring
    LangGraph's own checkpoint tuple, and cross-indexed by `workflow_id` so a
    workflow's checkpoints can be found without going through a run.
    """

    __tablename__ = "checkpoints"

    id = Column(GUID(), primary_key=True, default=uuid4)
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=False)
    workflow_run_id = Column(GUID(), ForeignKey("workflow_runs.id"), nullable=True)
    thread_id = Column(String(255), nullable=False)
    checkpoint_ns = Column(String(255), nullable=False, default="")
    checkpoint_id = Column(String(255), nullable=False, default=lambda: str(uuid4()))
    parent_checkpoint_id = Column(String(255), nullable=True)
    checkpoint = Column(JSON, nullable=False, default={})
    checkpoint_metadata = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_checkpoints_workflow_id", "workflow_id"),
        Index("ix_checkpoints_thread_id", "thread_id"),
        UniqueConstraint(
            "thread_id", "checkpoint_ns", "checkpoint_id", name="uq_checkpoints_thread_ns_checkpoint"
        ),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "workflow_id": str(self.workflow_id),
            "workflow_run_id": str(self.workflow_run_id) if self.workflow_run_id else None,
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "checkpoint": self.checkpoint,
            "checkpoint_metadata": self.checkpoint_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ApprovalModel(Base):
    """ORM model for a human sign-off on one proposed action.

    `action_hash` is the fingerprint of the exact proposed action (see
    `personalos.policy.intents.fingerprint_intent`); it is stored verbatim so
    an approval can only ever be matched against the action it was granted
    for; a changed action fingerprints differently and cannot reuse the row.
    """

    __tablename__ = "approvals"

    id = Column(GUID(), primary_key=True, default=uuid4)
    workflow_run_id = Column(GUID(), ForeignKey("workflow_runs.id"), nullable=True)
    workflow_step_id = Column(GUID(), ForeignKey("workflow_steps.id"), nullable=True)
    action_hash = Column(String(64), nullable=False)
    status = Column(
        Enum("pending", "approved", "denied", name="approval_status"),
        nullable=False,
        default="pending",
    )
    requested_by_user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    approved_by_user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    note = Column(Text, nullable=True)
    decided_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_approvals_action_hash", "action_hash"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "workflow_run_id": str(self.workflow_run_id) if self.workflow_run_id else None,
            "workflow_step_id": str(self.workflow_step_id) if self.workflow_step_id else None,
            "action_hash": self.action_hash,
            "status": self.status,
            "requested_by_user_id": str(self.requested_by_user_id)
            if self.requested_by_user_id
            else None,
            "approved_by_user_id": str(self.approved_by_user_id)
            if self.approved_by_user_id
            else None,
            "note": self.note,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# --- Job-search domain schema ------------------------------------------------
#
# The canonical job-search tables: postings discovered from any source
# (job_postings), a user's versioned targeting/preferences (candidate_profiles),
# one user's tracked pursuit of one posting (applications), and the tailored
# resume/cover-letter drafts generated for an application (artifact_versions).
#
# `applications.status` is constrained by its Enum to the values in
# `personalos.domain.models.ApplicationStatus`, but the *transition* between
# them -- whether DISCOVERED may become SAVED, never OFFER directly -- is
# enforced by `ApplicationRepository.update_status` calling
# `personalos.domain.models.validate_application_status_transition`. The
# column alone only rejects an unknown status string, not an illegal move.


class JobPostingModel(Base):
    """ORM model for one job posting discovered from a source (board, referral, etc.).

    `description_hash` and `normalized_json` are the dedupe inputs;
    `dedupe_key` is derived from the posting's normalized content (not
    `source` + `source_job_id`), so the same role scraped from two different
    boards still collapses onto one row rather than creating a duplicate.
    """

    __tablename__ = "job_postings"

    id = Column(GUID(), primary_key=True, default=uuid4)
    source = Column(String(100), nullable=False)
    source_job_id = Column(String(255), nullable=True)
    title = Column(String(500), nullable=False)
    company = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    url = Column(Text, nullable=True)
    raw_json = Column(JSON, nullable=False, default={})
    normalized_json = Column(JSON, nullable=False, default={})
    description_hash = Column(String(64), nullable=False)
    dedupe_key = Column(String(150), nullable=False)
    posted_at = Column(DateTime, nullable=True)
    discovered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("dedupe_key", name="uq_job_postings_dedupe_key"),
        Index("ix_job_postings_source", "source"),
        Index("ix_job_postings_company", "company"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "source": self.source,
            "source_job_id": self.source_job_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "raw_json": self.raw_json,
            "normalized_json": self.normalized_json,
            "description_hash": self.description_hash,
            "dedupe_key": self.dedupe_key,
            "posted_at": self.posted_at.isoformat() if self.posted_at else None,
            "discovered_at": self.discovered_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CandidateProfileModel(Base):
    """ORM model for one version of a user's job-search targeting profile.

    Profiles are versioned rather than updated in place: `profile_version`
    plus the unique constraint below let an application record exactly which
    snapshot of a user's roles/locations/preferences it was prepared against.
    """

    __tablename__ = "candidate_profiles"

    id = Column(GUID(), primary_key=True, default=uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    profile_version = Column(Integer, nullable=False, default=1)
    target_roles = Column(JSON, nullable=False, default=[])
    target_locations = Column(JSON, nullable=False, default=[])
    preferences = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "user_id", "profile_version", name="uq_candidate_profiles_user_version"
        ),
        Index("ix_candidate_profiles_user_id", "user_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "profile_version": self.profile_version,
            "target_roles": self.target_roles,
            "target_locations": self.target_locations,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ApplicationModel(Base):
    """ORM model for one user's tracked pursuit of one job posting.

    See the module docstring above: `status` only constrains the value to a
    known lifecycle state, not the transition between states.
    """

    __tablename__ = "applications"

    id = Column(GUID(), primary_key=True, default=uuid4)
    job_posting_id = Column(GUID(), ForeignKey("job_postings.id"), nullable=False)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    candidate_profile_id = Column(GUID(), ForeignKey("candidate_profiles.id"), nullable=True)
    status = Column(
        Enum(
            "discovered",
            "saved",
            "preparing",
            "ready_to_apply",
            "applied",
            "interviewing",
            "offer",
            "rejected",
            "withdrawn",
            "skipped",
            name="application_status",
        ),
        nullable=False,
        default="discovered",
    )
    resume_doc_id = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    applied_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "job_posting_id", "user_id", name="uq_applications_job_posting_user"
        ),
        Index("ix_applications_user_id", "user_id"),
        Index("ix_applications_status", "status"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "job_posting_id": str(self.job_posting_id),
            "user_id": str(self.user_id),
            "candidate_profile_id": str(self.candidate_profile_id)
            if self.candidate_profile_id
            else None,
            "status": self.status,
            "resume_doc_id": self.resume_doc_id,
            "notes": self.notes,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ArtifactVersionModel(Base):
    """ORM model for one generated resume/cover-letter draft for an application.

    `evidence` must cite the resume section(s) or project(s) the draft was
    generated from (see `personalos.domain.models.validate_evidence_links`):
    `ArtifactVersionRepository.create` rejects an empty list before the row
    is ever written, so no draft can carry an unlinked claim.
    """

    __tablename__ = "artifact_versions"

    id = Column(GUID(), primary_key=True, default=uuid4)
    application_id = Column(GUID(), ForeignKey("applications.id"), nullable=False)
    artifact_type = Column(
        Enum("resume", "cover_letter", name="artifact_type"), nullable=False
    )
    version = Column(Integer, nullable=False, default=1)
    doc_id = Column(String(255), nullable=True)
    content = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=False)
    generated_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "application_id",
            "artifact_type",
            "version",
            name="uq_artifact_versions_app_type_version",
        ),
        Index("ix_artifact_versions_application_id", "application_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "application_id": str(self.application_id),
            "artifact_type": self.artifact_type,
            "version": self.version,
            "doc_id": self.doc_id,
            "content": self.content,
            "evidence": self.evidence,
            "generated_by": self.generated_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CommunicationEventModel(Base):
    """ORM model for one recruiter-side signal tied to an application.

    Captures the classification of an inbound message (interview invite,
    rejection, etc.) so an application's communication history is queryable
    without a standalone Communications Agent, which is out of scope for this
    build. `provider_message_id` is unique per application so the same
    message ingested twice collapses onto one row instead of duplicating.
    """

    __tablename__ = "communication_events"

    id = Column(GUID(), primary_key=True, default=uuid4)
    application_id = Column(GUID(), ForeignKey("applications.id"), nullable=False)
    classification = Column(
        Enum(
            "recruiter_response",
            "interview_invite",
            "rejection",
            "offer",
            "action_required",
            "general_update",
            name="communication_event_classification",
        ),
        nullable=False,
    )
    provider_message_id = Column(String(255), nullable=True)
    occurred_at = Column(DateTime, nullable=False)
    metadata_json = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "application_id",
            "provider_message_id",
            name="uq_communication_events_app_provider_message",
        ),
        Index("ix_communication_events_application_id", "application_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "application_id": str(self.application_id),
            "classification": self.classification,
            "provider_message_id": self.provider_message_id,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
