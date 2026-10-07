"""Database models using SQLAlchemy ORM."""

import json
from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector as PGVector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
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

# Default embedding width, matching OpenAI's text-embedding-3-small/ada-002.
# A model that produces a different width needs its own migration to widen
# the column; `embedding_model`/`embedding_version` on each table only make
# re-embedding *within* this width traceable, not a dimension change.
EMBEDDING_DIMENSION = 1536


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


class Vector(TypeDecorator):
    """Platform-independent embedding-vector column.

    Uses pgvector's native `vector(dim)` type on PostgreSQL -- searchable via
    an ANN index (see the `add_pgvector_embedding_tables` migration) and the
    `cosine_distance`/`l2_distance`/etc. comparators it adds to the mapped
    column -- and falls back to a JSON-encoded list of floats on other
    dialects, so embedding tables can be exercised against SQLite in tests
    without a Postgres+pgvector instance. On that fallback, nearest-neighbor
    search is done in Python (see `personalos.persistence.repositories`),
    not by the database, mirroring `GUID`'s dialect split above.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGVector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps([float(v) for v in value])

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return json.loads(value)


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

    `checkpoint` holds the snapshot as the
    `{"type": ..., "data": <base64>}` envelope
    `personalos.persistence.checkpointer.SqlAlchemyCheckpointSaver` writes, not
    queryable JSON: the value is whatever LangGraph's serializer produced for
    that state, and re-encoding it as plain JSON would round-trip some channel
    values into something the graph cannot rebuild. `checkpoint_metadata` is
    the opposite trade -- JSON-coerced, and therefore inspectable and
    filterable, because that is all it is for.

    The writes belonging to each checkpoint live in `checkpoint_writes`; see
    `CheckpointWriteModel` for why they are a separate table.
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


class CheckpointWriteModel(Base):
    """ORM model for the writes a task produced against one checkpoint.

    The other half of the LangGraph checkpointer's storage, and not an
    optimization: when a worker dies part way through a super-step, the tasks
    that *had* finished are recorded here rather than in the checkpoint, so a
    resume replays only the work that never landed. Without this table a crash
    mid-step would re-run every task in that step, which is exactly the
    duplicate-side-effect case the durable checkpointer exists to prevent.

    Keyed by `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`, mirroring
    LangGraph's own write tuple. `idx` is negative for the reserved channels in
    `langgraph.checkpoint.base.WRITES_IDX_MAP` (`__error__`, `__interrupt__`,
    ...), which are overwritten on re-write, and non-negative for ordinary
    channel writes, which are not -- see
    `personalos.persistence.checkpointer.SqlAlchemyCheckpointSaver.put_writes`.
    """

    __tablename__ = "checkpoint_writes"

    id = Column(GUID(), primary_key=True, default=uuid4)
    thread_id = Column(String(255), nullable=False)
    checkpoint_ns = Column(String(255), nullable=False, default="")
    checkpoint_id = Column(String(255), nullable=False)
    task_id = Column(String(255), nullable=False)
    idx = Column(Integer, nullable=False, default=0)
    channel = Column(String(255), nullable=False)
    #: Serialized write value, as the `{"type": ..., "data": <base64>}` envelope
    #: `personalos.persistence.checkpointer` writes. Opaque to SQL on purpose:
    #: a channel value is whatever the graph put on that channel, and coercing
    #: it to queryable JSON would change what comes back out.
    value = Column(JSON, nullable=False, default={})
    task_path = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index(
            "ix_checkpoint_writes_thread_ns_checkpoint",
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
        ),
        UniqueConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            name="uq_checkpoint_writes_thread_ns_checkpoint_task_idx",
        ),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "idx": self.idx,
            "channel": self.channel,
            "value": self.value,
            "task_path": self.task_path,
            "created_at": self.created_at.isoformat(),
        }


class WorkflowLeaseModel(Base):
    """ORM model for the exclusive, expiring claim one worker holds on a workflow.

    One row per workflow, enforced by the unique constraint on `workflow_id`:
    that constraint is the primitive, not a hint. Two workers racing to resume
    the same workflow from empty both try to insert, and the loser gets an
    `IntegrityError` rather than a second lease -- which is what makes the
    guarantee hold on SQLite too, where `SELECT ... FOR UPDATE` is silently a
    no-op.

    Held-ness is a function of the row, not its existence: a lease is held
    while `released_at IS NULL` and `expires_at` is in the future. An expired
    row is takeable, so a worker that was killed without releasing does not
    strand its workflow forever, and `lease_token` changes on every takeover so
    the previous holder cannot release or renew a lease it has lost. See
    `personalos.persistence.leases`.
    """

    __tablename__ = "workflow_leases"

    id = Column(GUID(), primary_key=True, default=uuid4)
    # Uniqueness is declared once, as the named constraint in `__table_args__`:
    # `unique=True` here as well would emit a second, anonymous UNIQUE clause in
    # `create_all` that the migration does not create, so the schema the tests
    # build and the schema production runs would quietly differ.
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=False)
    thread_id = Column(String(255), nullable=True)
    owner = Column(String(255), nullable=False)
    lease_token = Column(GUID(), nullable=False, default=uuid4)
    acquired_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    released_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_workflow_leases_expires_at", "expires_at"),
        UniqueConstraint("workflow_id", name="uq_workflow_leases_workflow_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "workflow_id": str(self.workflow_id),
            "thread_id": self.thread_id,
            "owner": self.owner,
            "lease_token": str(self.lease_token),
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class PendingCheckpointModel(Base):
    """ORM model for a durable, conditional wait -- one that no process holds open.

    The row *is* the wait. Nothing is sleeping, no task is pending, no
    connection is parked: a checkpoint is actionable because
    `apps.worker.checkpoint_monitor` finds `status = 'pending'` and
    `trigger_at <= now()` here, which is what makes "seven days after applying"
    survive a deploy, a crash and a weekend. See
    `personalos.domain.checkpoints` for why the condition is stored
    declaratively (`condition_kind` / `condition_subject_id` /
    `condition_since` / `condition_params`) rather than pre-evaluated: a
    condition resolved at creation time answers a question about the wrong
    moment.

    `trigger_at` and `expires_at` are separate columns, not a duration and an
    offset, because they answer different questions -- when may this act, and
    after when must it never act. The second is what stops a monitor that was
    down for a week from sending a week-late follow-up, and what stops a
    checkpoint nobody ever swept from sitting `pending` forever.

    `dedupe_key` is unique: re-running the branch that schedules a follow-up
    must rejoin the existing wait rather than stack a second reminder on the
    same application, exactly as `outbox_events.dedupe_key` stops a retried
    enqueue becoming a second message.

    `application_id` deliberately carries no foreign key. A checkpoint is
    scheduled from graph state, which holds ids rather than rows, and the sweep
    that reads it never joins to the application -- it hands the id to a
    condition evaluator. A constraint here would only decide the order in which
    two independently-written tables have to be populated.
    """

    __tablename__ = "pending_checkpoints"

    id = Column(GUID(), primary_key=True, default=uuid4)
    application_id = Column(GUID(), nullable=False)
    #: `personalos.domain.job_search.FollowUpKind`.
    kind = Column(String(50), nullable=False)
    reason = Column(Text, nullable=False)
    #: The thread whose graph path a fired checkpoint starts, and the workflow
    #: whose lease that start is taken under. Stored, not held.
    thread_id = Column(String(255), nullable=False)
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=True)
    #: `personalos.domain.checkpoints.CheckpointCondition`, flattened. Flat
    #: rather than a single JSON blob because `condition_kind` and
    #: `condition_subject_id` are what an operator filters by when asking "what
    #: is still waiting on this application?".
    condition_kind = Column(String(50), nullable=False)
    condition_subject_id = Column(GUID(), nullable=False)
    condition_since = Column(DateTime, nullable=True)
    condition_params = Column(JSON, nullable=False, default={})
    trigger_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    status = Column(
        Enum(
            "pending",
            "resolved",
            "fired",
            "expired",
            "cancelled",
            name="pending_checkpoint_status",
        ),
        nullable=False,
        default="pending",
    )
    dedupe_key = Column(String(255), nullable=False)
    closed_at = Column(DateTime, nullable=True)
    closed_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        # The sweep's index: "everything still waiting, soonest first". Leading
        # with `status` keeps the scan off the closed rows, which are the ones
        # that accumulate.
        Index("ix_pending_checkpoints_status_trigger_at", "status", "trigger_at"),
        Index("ix_pending_checkpoints_application_id", "application_id"),
        UniqueConstraint("dedupe_key", name="uq_pending_checkpoints_dedupe_key"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "application_id": str(self.application_id),
            "kind": self.kind,
            "reason": self.reason,
            "thread_id": self.thread_id,
            "workflow_id": str(self.workflow_id) if self.workflow_id else None,
            "condition_kind": self.condition_kind,
            "condition_subject_id": str(self.condition_subject_id),
            "condition_since": (
                self.condition_since.isoformat() if self.condition_since else None
            ),
            "condition_params": self.condition_params,
            "trigger_at": self.trigger_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "dedupe_key": self.dedupe_key,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "closed_reason": self.closed_reason,
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
# enforced by `personalos.persistence.application_lifecycle.apply_transition`
# (which `ApplicationRepository.update_status` delegates to) calling
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
            "response",
            "interviewing",
            "offer",
            "accepted",
            "declined",
            "rejected",
            "withdrawn",
            "skipped",
            "follow_up_pending",
            "stalled",
            name="application_status",
        ),
        nullable=False,
        default="discovered",
    )
    #: The status a held application (FOLLOW_UP_PENDING, STALLED) was held
    #: from, and so what it may resume to. NULL in every other status.
    resume_status = Column(String(32), nullable=True)
    resume_doc_id = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    applied_at = Column(DateTime, nullable=True)
    #: When something last happened on this application: a transition, or
    #: anything reported through `ApplicationLifecycleStore.record_activity`.
    #: The stall check reads this and nothing else -- never a chat transcript.
    last_activity_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "job_posting_id", "user_id", name="uq_applications_job_posting_user"
        ),
        Index("ix_applications_user_id", "user_id"),
        Index("ix_applications_status", "status"),
        # The stall check's index: "still in play, quiet the longest first".
        Index("ix_applications_status_last_activity_at", "status", "last_activity_at"),
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
            "resume_status": self.resume_status,
            "resume_doc_id": self.resume_doc_id,
            "notes": self.notes,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "last_activity_at": (
                self.last_activity_at.isoformat() if self.last_activity_at else None
            ),
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
    build. `dedupe_key` is unique across the table, so the same message
    delivered twice collapses onto one row whichever application it was
    correlated to; see `personalos.domain.recruiter_events.communication_dedupe_key`.
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
            "unrelated",
            name="communication_event_classification",
        ),
        nullable=False,
    )
    provider_message_id = Column(String(255), nullable=True)
    #: NULL only on rows written before the column existed.
    dedupe_key = Column(String(300), nullable=True)
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
        # A unique index rather than a constraint: SQLite cannot add a
        # constraint to an existing table, and the migration runs on both.
        Index("uq_communication_events_dedupe_key", "dedupe_key", unique=True),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "application_id": str(self.application_id),
            "classification": self.classification,
            "provider_message_id": self.provider_message_id,
            "dedupe_key": self.dedupe_key,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CommitmentModel(Base):
    """ORM model for one commitment read out of a recruiter-side message.

    Who owes what, by when, on what condition. Always tied to the
    `communication_events` row it was extracted from, so a deadline can be
    traced to the message that set it.
    """

    __tablename__ = "commitments"

    id = Column(GUID(), primary_key=True, default=uuid4)
    communication_event_id = Column(
        GUID(), ForeignKey("communication_events.id"), nullable=False
    )
    application_id = Column(GUID(), ForeignKey("applications.id"), nullable=False)
    actor = Column(Enum("user", "external_person", name="commitment_actor"), nullable=False)
    action = Column(Text, nullable=False)
    due_at = Column(DateTime, nullable=True)
    condition = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False)
    source_message_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_commitments_application_id", "application_id"),
        Index("ix_commitments_communication_event_id", "communication_event_id"),
        Index("ix_commitments_due_at", "due_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "communication_event_id": str(self.communication_event_id),
            "application_id": str(self.application_id),
            "actor": self.actor,
            "action": self.action,
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "condition": self.condition,
            "confidence": self.confidence,
            "source_message_id": self.source_message_id,
            "created_at": self.created_at.isoformat(),
        }


# --- Tool execution, policy decision, and audit trail ------------------------
#
# The tables that make every mutating action inspectable and auditable: one
# row per tool-call attempt (tool_executions), the verdict reached on a
# proposed action (policy_decisions), and the append-only record of what
# actually happened (audit_events). `AuditEventRepository` exposes only
# `create` and reads -- there is no update or delete path in application code,
# so the audit trail can only be added to, never rewritten.


class ToolExecutionModel(Base):
    """ORM model for one tool-call attempt, keyed by idempotency key.

    Mirrors `OperationModel`'s idempotency semantics -- the unique constraint
    on `idempotency_key` is the dedup primitive a retried call relies on to
    get back the stored `receipt_json` instead of re-executing -- but scoped
    to a specific tool and workflow run rather than the generic operation log.

    `policy_decision_id`, `approval_ref` and `approved_by` tie the execution
    back to what authorized it: the verdict row the policy engine wrote before
    the call, and the human approval that verdict was redeemed with, if any.
    `request_fingerprint` is the hash of the side effect the key was claimed
    for, so a key reused for a different action is rejected, not replayed.
    """

    __tablename__ = "tool_executions"

    operation_id = Column(GUID(), primary_key=True, default=uuid4)
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=True)
    tool_name = Column(String(255), nullable=False)
    idempotency_key = Column(String(255), nullable=False, unique=True)
    status = Column(
        Enum("in_progress", "completed", "failed", "unknown", name="tool_execution_status"),
        nullable=False,
        default="in_progress",
    )
    request_fingerprint = Column(String(64), nullable=True)
    policy_decision_id = Column(GUID(), ForeignKey("policy_decisions.id"), nullable=True)
    approval_ref = Column(String(255), nullable=True)
    approved_by = Column(String(255), nullable=True)
    attempts = Column(Integer, nullable=False, default=1, server_default="1")
    receipt_json = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (Index("ix_tool_executions_workflow_id", "workflow_id"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "operation_id": str(self.operation_id),
            "workflow_id": str(self.workflow_id) if self.workflow_id else None,
            "tool_name": self.tool_name,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "request_fingerprint": self.request_fingerprint,
            "policy_decision_id": str(self.policy_decision_id)
            if self.policy_decision_id
            else None,
            "approval_ref": self.approval_ref,
            "approved_by": self.approved_by,
            "attempts": self.attempts,
            "receipt_json": self.receipt_json,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class PolicyDecisionModel(Base):
    """ORM model for the verdict reached on one proposed tool call.

    `args_hash` is the fingerprint of the call's arguments (see
    `personalos.policy.intents.fingerprint_intent`), stored rather than the
    raw arguments so the decision log doesn't duplicate -- or leak -- whatever
    the arguments themselves contained.
    """

    __tablename__ = "policy_decisions"

    id = Column(GUID(), primary_key=True, default=uuid4)
    principal = Column(String(255), nullable=False)
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=True)
    tool = Column(String(255), nullable=False)
    args_hash = Column(String(64), nullable=False)
    decision = Column(
        Enum("allow", "deny", "require_approval", name="policy_decision_outcome"),
        nullable=False,
    )
    requested_scopes = Column(JSON, nullable=False, default=[])
    decided_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_policy_decisions_workflow_id", "workflow_id"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "principal": self.principal,
            "workflow_id": str(self.workflow_id) if self.workflow_id else None,
            "tool": self.tool,
            "args_hash": self.args_hash,
            "decision": self.decision,
            "requested_scopes": self.requested_scopes,
            "decided_at": self.decided_at.isoformat(),
        }


class AuditEventModel(Base):
    """ORM model for one append-only audit trail entry.

    No repository method updates or deletes a row here: `AuditEventRepository`
    only ever inserts and reads, so the only way to change the audit trail is
    to add to it.
    """

    __tablename__ = "audit_events"

    id = Column(GUID(), primary_key=True, default=uuid4)
    actor = Column(String(255), nullable=False)
    workflow_id = Column(GUID(), ForeignKey("workflows.id"), nullable=True)
    action = Column(String(255), nullable=False)
    target_ref = Column(String(500), nullable=False)
    # Snapshot of the policy verdict (allow/deny/require_approval) in force
    # for this action, stored as plain text rather than a foreign key or
    # shared enum type, so this row stays a stable historical record even if
    # `policy_decisions` or its vocabulary changes later.
    policy_decision = Column(String(32), nullable=True)
    # The verdict row itself, and the execution it authorized. The snapshot
    # above says what was decided; these say which decision and which call.
    policy_decision_id = Column(GUID(), ForeignKey("policy_decisions.id"), nullable=True)
    operation_id = Column(GUID(), ForeignKey("tool_executions.operation_id"), nullable=True)
    approval_ref = Column(String(255), nullable=True)
    result = Column(Enum("success", "failure", name="audit_event_result"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_audit_events_workflow_id", "workflow_id"),
        Index("ix_audit_events_operation_id", "operation_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "actor": self.actor,
            "workflow_id": str(self.workflow_id) if self.workflow_id else None,
            "action": self.action,
            "target_ref": self.target_ref,
            "policy_decision": self.policy_decision,
            "policy_decision_id": str(self.policy_decision_id)
            if self.policy_decision_id
            else None,
            "operation_id": str(self.operation_id) if self.operation_id else None,
            "approval_ref": self.approval_ref,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
        }


# --- Credential references ---------------------------------------------------
#
# What the database knows about a credential: that it exists, whose it is,
# which provider it is for and what it may be used for. It does not know the
# credential. The refresh token or API key is in the OS secret store, filed
# under `credential_ref`, and only the executor's credential broker reads it.


class CredentialModel(Base):
    """ORM model for a reference to a credential held in the OS secret store.

    There is deliberately no column here that could hold a token. The
    `credential_ref` check constraint is a tripwire rather than a guarantee --
    the guarantee is `CredentialRepository.create` accepting only a
    `CredentialRef` -- but it means a row written around the repository still
    cannot put an arbitrary string where the reference goes.
    """

    __tablename__ = "credentials"

    id = Column(GUID(), primary_key=True, default=uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    provider = Column(String(63), nullable=False)
    kind = Column(
        Enum("oauth_refresh_token", "api_key", "oauth_client_secret", name="credential_kind"),
        nullable=False,
    )
    credential_ref = Column(String(300), nullable=False)
    scopes = Column(JSON, nullable=False, default=list)
    status = Column(
        Enum("active", "revoked", name="credential_status"),
        nullable=False,
        default="active",
    )
    last_exchanged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("credential_ref", name="uq_credentials_credential_ref"),
        CheckConstraint("substr(credential_ref, 1, 7) = 'cred://'", name="ck_credentials_ref_is_reference"),
        Index("ix_credentials_user_id", "user_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "provider": self.provider,
            "kind": self.kind,
            "credential_ref": self.credential_ref,
            "scopes": self.scopes,
            "status": self.status,
            "last_exchanged_at": (
                self.last_exchanged_at.isoformat() if self.last_exchanged_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# --- Transactional outbox, event log, and projections ------------------------
#
# `outbox_events` backs the transactional-outbox pattern: a row is written in
# the same DB transaction as the domain mutation that produced it (see
# `OutboxEventRepository`/`ApplicationRepository.update_status`'s `commit`
# parameter), so a committed domain change can never lose its corresponding
# outbound message to a crash between the two writes. A worker claims a row
# with `OutboxEventRepository.claim_next` -- `SELECT ... FOR UPDATE SKIP
# LOCKED` on Postgres, an atomic conditional UPDATE everywhere else -- before
# dispatching it, so exactly one worker ever owns a given row.
#
# `event_log` is the durable, append-only history of domain events, distinct
# from `outbox_events`: outbox rows are deleted/dispatched-and-forgotten once
# relayed, while event_log rows are never updated or deleted, only ever
# inserted. `EventLogRepository` exposes no update or delete method.
#
# `application_status_view` is a mutable projection recomputed from
# `event_log` -- the current-status read surface for an application, rebuilt
# by `ApplicationStatusViewRepository.recompute` rather than written directly.
# `personalos.persistence.application_lifecycle.apply_transition` is what
# keeps the two in step: one transaction appends the `application.status_changed`
# event and moves the projection to it.


class OutboxEventModel(Base):
    """ORM model for one row in the transactional outbox.

    `dedupe_key` is optional and unique when present, letting a producer
    retry the enqueue itself (e.g. after a crash before its own commit is
    confirmed) without risking a duplicate outbound message.
    """

    __tablename__ = "outbox_events"

    id = Column(GUID(), primary_key=True, default=uuid4)
    type = Column(String(255), nullable=False)
    payload_json = Column(JSON, nullable=False, default={})
    dedupe_key = Column(String(255), nullable=True, unique=True)
    status = Column(
        Enum("pending", "in_progress", "dispatched", "failed", name="outbox_event_status"),
        nullable=False,
        default="pending",
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    dispatched_at = Column(DateTime, nullable=True)

    __table_args__ = (Index("ix_outbox_events_status", "status"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "type": self.type,
            "payload_json": self.payload_json,
            "dedupe_key": self.dedupe_key,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
        }


class EventLogModel(Base):
    """ORM model for one immutable entry in the durable domain event history.

    No repository method updates or deletes a row here -- `EventLogRepository`
    only ever inserts and reads, matching `AuditEventModel`'s append-only
    contract but scoped to domain events feeding projections rather than the
    security/compliance audit trail.
    """

    __tablename__ = "event_log"

    id = Column(GUID(), primary_key=True, default=uuid4)
    aggregate_type = Column(String(100), nullable=False)
    aggregate_id = Column(GUID(), nullable=False)
    event_type = Column(String(255), nullable=False)
    payload_json = Column(JSON, nullable=False, default={})
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_event_log_aggregate_id", "aggregate_id"),
        Index("ix_event_log_aggregate_type_aggregate_id", "aggregate_type", "aggregate_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "aggregate_type": self.aggregate_type,
            "aggregate_id": str(self.aggregate_id),
            "event_type": self.event_type,
            "payload_json": self.payload_json,
            "occurred_at": self.occurred_at.isoformat(),
            "created_at": self.created_at.isoformat(),
        }


class ApplicationStatusViewModel(Base):
    """ORM model for the mutable current-status projection of an application.

    One row per application, overwritten in place by
    `ApplicationStatusViewRepository.recompute` from `event_log` -- this table
    holds no history of its own, only the latest recomputed state.
    `last_event_id` records which event_log row the current snapshot reflects.
    """

    __tablename__ = "application_status_view"

    application_id = Column(GUID(), ForeignKey("applications.id"), primary_key=True)
    status = Column(String(32), nullable=False)
    last_event_id = Column(GUID(), ForeignKey("event_log.id"), nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "application_id": str(self.application_id),
            "status": self.status,
            "last_event_id": str(self.last_event_id) if self.last_event_id else None,
            "updated_at": self.updated_at.isoformat(),
        }


# --- Semantic retrieval (pgvector) -------------------------------------------
#
# Embedding tables backing evidence-grounded job matching and semantic
# scoring/dedup: evidence_chunks (resume/project text cut into retrievable
# chunks), job_posting_embeddings (one embedding per posting per embedding
# model, for similarity scoring and near-duplicate detection), and
# message_embeddings (recruiter message text, for semantic search over
# communication history).
#
# Every row records `embedding_model` (and `embedding_version`) alongside its
# vector so a model upgrade can be rolled out as a new set of rows rather
# than an in-place overwrite: old and new embeddings coexist, queries pin a
# model, and the old rows are cleaned up once callers have moved on. See
# `Vector` above for how the column itself degrades from pgvector to a JSON
# fallback on non-Postgres dialects, and
# `personalos.persistence.repositories` for how nearest-neighbor search
# follows that same split.


class EvidenceChunkModel(Base):
    """ORM model for one retrievable chunk of a candidate's resume or project write-up.

    `chunk_index` orders chunks cut from the same `source_ref` back into
    their original sequence; the uniqueness constraint below stops the same
    chunk from being embedded twice if ingestion is retried.
    """

    __tablename__ = "evidence_chunks"

    id = Column(GUID(), primary_key=True, default=uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    source_type = Column(Enum("resume", "project", name="evidence_source_type"), nullable=False)
    source_ref = Column(String(255), nullable=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    embedding_version = Column(String(50), nullable=False, default="1")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source_type",
            "source_ref",
            "chunk_index",
            name="uq_evidence_chunks_user_source_chunk",
        ),
        Index("ix_evidence_chunks_user_id", "user_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "chunk_text": self.chunk_text,
            "chunk_index": self.chunk_index,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class JobPostingEmbeddingModel(Base):
    """ORM model for one embedding of a job posting's description.

    Scoped per `embedding_model`/`embedding_version` (see the uniqueness
    constraint below) rather than one row per posting, so re-embedding with a
    new model doesn't discard the old vector before every caller has moved
    off it.
    """

    __tablename__ = "job_posting_embeddings"

    id = Column(GUID(), primary_key=True, default=uuid4)
    job_posting_id = Column(GUID(), ForeignKey("job_postings.id"), nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    embedding_version = Column(String(50), nullable=False, default="1")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "job_posting_id",
            "embedding_model",
            "embedding_version",
            name="uq_job_posting_embeddings_posting_model_version",
        ),
        Index("ix_job_posting_embeddings_job_posting_id", "job_posting_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "job_posting_id": str(self.job_posting_id),
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "created_at": self.created_at.isoformat(),
        }


class MessageEmbeddingModel(Base):
    """ORM model for one embedding of a recruiter communication's text.

    Mirrors `JobPostingEmbeddingModel`'s per-model versioning, scoped to
    `communication_events` instead of `job_postings`, to support semantic
    search over recruiter messages.
    """

    __tablename__ = "message_embeddings"

    id = Column(GUID(), primary_key=True, default=uuid4)
    communication_event_id = Column(GUID(), ForeignKey("communication_events.id"), nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    embedding_version = Column(String(50), nullable=False, default="1")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "communication_event_id",
            "embedding_model",
            "embedding_version",
            name="uq_message_embeddings_event_model_version",
        ),
        Index("ix_message_embeddings_communication_event_id", "communication_event_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "communication_event_id": str(self.communication_event_id),
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "created_at": self.created_at.isoformat(),
        }
