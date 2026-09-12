"""Repository pattern for data access."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.context import ExecutionContext
from personalos.domain.models import (
    ApplicationStatus,
    Job,
    JobStatus,
    OperationRecord,
    OperationStatus,
    ToolExecutionStatus,
    validate_application_status_transition,
    validate_evidence_links,
)
from personalos.persistence.models import (
    ApplicationModel,
    ArtifactVersionModel,
    AuditEventModel,
    CandidateProfileModel,
    CheckpointModel,
    CommunicationEventModel,
    JobModel,
    JobPostingModel,
    OperationModel,
    PolicyDecisionModel,
    ToolExecutionModel,
    WorkflowModel,
    WorkflowRunModel,
)


class JobRepository:
    """Repository for Job persistence."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(self, job: Job) -> Job:
        """Create a new job."""
        db_job = JobModel(
            id=job.id,
            title=job.title,
            description=job.description,
            status=job.status.value,
            keywords=job.keywords,
            locations=job.locations,
            salary_min=job.salary_min,
            salary_max=job.salary_max,
            job_type=job.job_type,
            # The column is `job_metadata`: `metadata` is reserved by
            # SQLAlchemy's declarative base and never reaches the database.
            job_metadata=job.metadata,
            workflow_id=job.context.workflow_id,
            run_id=job.context.run_id,
            correlation_id=job.context.correlation_id,
            actor_id=job.context.actor_id,
        )
        self.session.add(db_job)
        self.session.commit()
        return self._to_domain(db_job)

    def get_by_id(self, job_id: UUID) -> Job | None:
        """Get job by ID."""
        db_job = self.session.query(JobModel).filter(JobModel.id == job_id).first()
        return self._to_domain(db_job) if db_job else None

    def get_all(self) -> list[Job]:
        """Get all jobs."""
        db_jobs = self.session.query(JobModel).all()
        return [self._to_domain(db_job) for db_job in db_jobs]

    def update(self, job: Job) -> Job:
        """Update an existing job."""
        db_job = self.session.query(JobModel).filter(JobModel.id == job.id).first()
        if not db_job:
            raise ValueError(f"Job {job.id} not found")

        db_job.title = job.title
        db_job.description = job.description
        db_job.status = job.status.value
        db_job.keywords = job.keywords
        db_job.locations = job.locations
        db_job.salary_min = job.salary_min
        db_job.salary_max = job.salary_max
        db_job.job_type = job.job_type
        db_job.results_count = job.results_count
        db_job.results = job.results
        db_job.error_code = job.error_code
        db_job.error_message = job.error_message
        db_job.job_metadata = job.metadata
        db_job.started_at = job.started_at
        db_job.completed_at = job.completed_at
        db_job.updated_at = job.updated_at

        self.session.commit()
        return self._to_domain(db_job)

    def delete(self, job_id: UUID) -> bool:
        """Delete a job."""
        db_job = self.session.query(JobModel).filter(JobModel.id == job_id).first()
        if not db_job:
            return False
        self.session.delete(db_job)
        self.session.commit()
        return True

    @staticmethod
    def _to_domain(db_job: JobModel) -> Job:
        """Convert ORM model to domain model."""
        return Job(
            id=db_job.id,
            title=db_job.title,
            description=db_job.description,
            status=JobStatus(db_job.status),
            keywords=db_job.keywords or [],
            locations=db_job.locations or [],
            salary_min=db_job.salary_min,
            salary_max=db_job.salary_max,
            job_type=db_job.job_type,
            results_count=int(db_job.results_count) if db_job.results_count else 0,
            results=db_job.results or {},
            error_code=db_job.error_code,
            error_message=db_job.error_message,
            metadata=db_job.job_metadata or {},
            created_at=db_job.created_at,
            updated_at=db_job.updated_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
            context=ExecutionContext(
                workflow_id=db_job.workflow_id,
                run_id=db_job.run_id,
                correlation_id=db_job.correlation_id,
                actor_id=db_job.actor_id,
            ),
        )


class OperationRepository:
    """Repository for the mutating-operation log.

    Backs idempotency: every mutating action claims its idempotency key here
    before running, and records the outcome afterwards.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def get_by_key(self, idempotency_key: str) -> OperationRecord | None:
        """Get the operation recorded under an idempotency key, if any."""
        db_op = self._row_for_key(idempotency_key)
        return self._to_domain(db_op) if db_op else None

    def claim(
        self,
        idempotency_key: str,
        operation: str,
        request_fingerprint: str,
    ) -> tuple[OperationRecord, bool]:
        """Claim an idempotency key for execution.

        Returns (record, claimed). When `claimed` is True the caller owns the
        operation and must run the side effect, then call `complete` or `fail`.
        When False, the record already exists — it is either completed (replay
        its result), still in progress (a concurrent attempt owns it), or the
        fingerprint differs (the caller must reject the request).

        A previously failed operation is re-claimed so retries can proceed.
        """
        existing = self._row_for_key(idempotency_key)

        if existing is None:
            db_op = OperationModel(
                idempotency_key=idempotency_key,
                operation=operation,
                request_fingerprint=request_fingerprint,
                status=OperationStatus.IN_PROGRESS.value,
                attempts=1,
            )
            self.session.add(db_op)
            try:
                self.session.commit()
            except IntegrityError:
                # Lost the insert race: another attempt claimed this key first.
                self.session.rollback()
                winner = self._row_for_key(idempotency_key)
                if winner is None:  # pragma: no cover - unique violation implies a row
                    raise
                return self._to_domain(winner), False
            return self._to_domain(db_op), True

        # A recorded failure is retryable; flip it back to in_progress only if
        # it is still failed and the request is unchanged, so a concurrent retry
        # cannot double-claim it and a key reused for a different request is
        # left untouched for the caller to reject.
        if (
            existing.status == OperationStatus.FAILED.value
            and existing.request_fingerprint == request_fingerprint
        ):
            updated = (
                self.session.query(OperationModel)
                .filter(
                    OperationModel.idempotency_key == idempotency_key,
                    OperationModel.status == OperationStatus.FAILED.value,
                    OperationModel.request_fingerprint == request_fingerprint,
                )
                .update(
                    {
                        OperationModel.status: OperationStatus.IN_PROGRESS.value,
                        OperationModel.attempts: OperationModel.attempts + 1,
                        OperationModel.error: None,
                        OperationModel.updated_at: datetime.utcnow(),
                    },
                    synchronize_session=False,
                )
            )
            self.session.commit()
            self.session.refresh(existing)
            return self._to_domain(existing), updated == 1

        return self._to_domain(existing), False

    def complete(self, idempotency_key: str, result: Any) -> OperationRecord:
        """Record a successful outcome so future retries replay it."""
        db_op = self._require_row(idempotency_key)
        now = datetime.utcnow()
        db_op.status = OperationStatus.COMPLETED.value
        db_op.result = result
        db_op.error = None
        db_op.updated_at = now
        db_op.completed_at = now
        self.session.commit()
        return self._to_domain(db_op)

    def fail(self, idempotency_key: str, error: str) -> OperationRecord:
        """Record a failed outcome, leaving the key available for retry."""
        db_op = self._require_row(idempotency_key)
        db_op.status = OperationStatus.FAILED.value
        db_op.error = error
        db_op.updated_at = datetime.utcnow()
        self.session.commit()
        return self._to_domain(db_op)

    def _row_for_key(self, idempotency_key: str) -> OperationModel | None:
        return (
            self.session.query(OperationModel)
            .filter(OperationModel.idempotency_key == idempotency_key)
            .first()
        )

    def _require_row(self, idempotency_key: str) -> OperationModel:
        db_op = self._row_for_key(idempotency_key)
        if not db_op:
            raise ValueError(f"Operation '{idempotency_key}' not found")
        return db_op

    @staticmethod
    def _to_domain(db_op: OperationModel) -> OperationRecord:
        """Convert ORM model to domain model."""
        return OperationRecord(
            id=db_op.id,
            idempotency_key=db_op.idempotency_key,
            operation=db_op.operation,
            request_fingerprint=db_op.request_fingerprint,
            status=OperationStatus(db_op.status),
            result=db_op.result,
            error=db_op.error,
            attempts=db_op.attempts,
            created_at=db_op.created_at,
            updated_at=db_op.updated_at,
            completed_at=db_op.completed_at,
        )


class WorkflowRepository:
    """Repository for workflow definitions."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(self, *, name: str, description: str | None = None) -> WorkflowModel:
        """Create a new workflow definition."""
        db_workflow = WorkflowModel(name=name, description=description)
        self.session.add(db_workflow)
        self.session.commit()
        return db_workflow

    def get_by_id(self, workflow_id: UUID) -> WorkflowModel | None:
        """Get a workflow definition by ID."""
        return self.session.query(WorkflowModel).filter(WorkflowModel.id == workflow_id).first()


class WorkflowRunRepository:
    """Repository for workflow run persistence.

    A run is one execution attempt of a workflow, identified by the
    `thread_id` its checkpoints are keyed by.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        workflow_id: UUID,
        thread_id: str | None = None,
        user_id: UUID | None = None,
        actor_id: str = "system",
        correlation_id: UUID | None = None,
    ) -> WorkflowRunModel:
        """Create a new workflow run."""
        db_run = WorkflowRunModel(
            workflow_id=workflow_id,
            user_id=user_id,
            thread_id=thread_id or str(uuid4()),
            actor_id=actor_id,
            correlation_id=correlation_id,
        )
        self.session.add(db_run)
        self.session.commit()
        return db_run

    def get_by_id(self, run_id: UUID) -> WorkflowRunModel | None:
        """Get a workflow run by ID."""
        return self.session.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()

    def get_by_workflow_id(self, workflow_id: UUID) -> list[WorkflowRunModel]:
        """Get every run of a workflow, so a restarted process can resume any of them."""
        return (
            self.session.query(WorkflowRunModel)
            .filter(WorkflowRunModel.workflow_id == workflow_id)
            .all()
        )


class CheckpointRepository:
    """Repository for the LangGraph durable checkpointer's storage.

    Checkpoints are keyed by `thread_id` / `checkpoint_ns` / `checkpoint_id`
    and cross-indexed by `workflow_id` so a run's latest state can be found
    without needing the checkpointer's own thread bookkeeping.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def save(
        self,
        *,
        workflow_id: UUID,
        thread_id: str,
        checkpoint: dict[str, Any],
        workflow_run_id: UUID | None = None,
        checkpoint_ns: str = "",
        parent_checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointModel:
        """Persist a new checkpoint."""
        db_checkpoint = CheckpointModel(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            parent_checkpoint_id=parent_checkpoint_id,
            checkpoint=checkpoint,
            checkpoint_metadata=metadata or {},
        )
        self.session.add(db_checkpoint)
        self.session.commit()
        return db_checkpoint

    def get_latest_by_workflow_id(self, workflow_id: UUID) -> CheckpointModel | None:
        """Get the most recently written checkpoint for a workflow, if any."""
        return (
            self.session.query(CheckpointModel)
            .filter(CheckpointModel.workflow_id == workflow_id)
            .order_by(CheckpointModel.created_at.desc())
            .first()
        )


class JobPostingRepository:
    """Repository for discovered job postings.

    `create` lets the dedupe unique constraint on `dedupe_key` do the work:
    inserting a posting whose content already exists raises `IntegrityError`
    rather than silently creating a duplicate row.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        source: str,
        title: str,
        company: str,
        description_hash: str,
        dedupe_key: str,
        normalized_json: dict[str, Any] | None = None,
        raw_json: dict[str, Any] | None = None,
        source_job_id: str | None = None,
        location: str | None = None,
        url: str | None = None,
        posted_at: datetime | None = None,
    ) -> JobPostingModel:
        """Insert a job posting. Raises `IntegrityError` on a duplicate `dedupe_key`."""
        db_posting = JobPostingModel(
            source=source,
            source_job_id=source_job_id,
            title=title,
            company=company,
            location=location,
            url=url,
            raw_json=raw_json or {},
            normalized_json=normalized_json or {},
            description_hash=description_hash,
            dedupe_key=dedupe_key,
            posted_at=posted_at,
        )
        self.session.add(db_posting)
        self.session.commit()
        return db_posting

    def get_by_id(self, job_posting_id: UUID) -> JobPostingModel | None:
        """Get a job posting by ID."""
        return (
            self.session.query(JobPostingModel)
            .filter(JobPostingModel.id == job_posting_id)
            .first()
        )

    def get_by_dedupe_key(self, dedupe_key: str) -> JobPostingModel | None:
        """Get the job posting already stored for this dedupe key, if any."""
        return (
            self.session.query(JobPostingModel)
            .filter(JobPostingModel.dedupe_key == dedupe_key)
            .first()
        )


class CandidateProfileRepository:
    """Repository for a user's versioned job-search targeting profile."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        user_id: UUID,
        profile_version: int = 1,
        target_roles: list[str] | None = None,
        target_locations: list[str] | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> CandidateProfileModel:
        """Insert a new profile version for a user."""
        db_profile = CandidateProfileModel(
            user_id=user_id,
            profile_version=profile_version,
            target_roles=target_roles or [],
            target_locations=target_locations or [],
            preferences=preferences or {},
        )
        self.session.add(db_profile)
        self.session.commit()
        return db_profile

    def get_latest_by_user_id(self, user_id: UUID) -> CandidateProfileModel | None:
        """Get a user's highest-numbered profile version, if any."""
        return (
            self.session.query(CandidateProfileModel)
            .filter(CandidateProfileModel.user_id == user_id)
            .order_by(CandidateProfileModel.profile_version.desc())
            .first()
        )


class ApplicationRepository:
    """Repository for a user's tracked pursuit of a job posting.

    `status` is never written anywhere but `update_status`, and that method
    always runs the move through `validate_application_status_transition`
    first — the one path by which an application's lifecycle state can
    change, so a caller (including an LLM-driven one) can request a
    transition but never set the column outright.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        job_posting_id: UUID,
        user_id: UUID,
        candidate_profile_id: UUID | None = None,
        resume_doc_id: str | None = None,
    ) -> ApplicationModel:
        """Start tracking a job posting for a user, in the initial DISCOVERED status."""
        db_application = ApplicationModel(
            job_posting_id=job_posting_id,
            user_id=user_id,
            candidate_profile_id=candidate_profile_id,
            resume_doc_id=resume_doc_id,
        )
        self.session.add(db_application)
        self.session.commit()
        return db_application

    def get_by_id(self, application_id: UUID) -> ApplicationModel | None:
        """Get an application by ID."""
        return (
            self.session.query(ApplicationModel)
            .filter(ApplicationModel.id == application_id)
            .first()
        )

    def update_status(
        self, application_id: UUID, new_status: ApplicationStatus
    ) -> ApplicationModel:
        """Move an application to `new_status`.

        Raises `ValueError` if the application does not exist, or
        `InvalidApplicationTransition` if `new_status` is not reachable from
        the application's current status (e.g. DISCOVERED -> OFFER).
        """
        db_application = self.get_by_id(application_id)
        if not db_application:
            raise ValueError(f"Application {application_id} not found")

        current = ApplicationStatus(db_application.status)
        validate_application_status_transition(current, new_status)

        db_application.status = new_status.value
        db_application.updated_at = datetime.utcnow()
        if new_status == ApplicationStatus.APPLIED:
            db_application.applied_at = datetime.utcnow()
        self.session.commit()
        return db_application


class ArtifactVersionRepository:
    """Repository for tailored resume/cover-letter drafts generated for an application.

    `create` runs `evidence` through `validate_evidence_links` before
    constructing the row, so a draft can never be persisted without citing
    the resume section(s) or project(s) it was generated from.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        application_id: UUID,
        artifact_type: str,
        evidence: list[dict[str, Any]],
        version: int = 1,
        doc_id: str | None = None,
        content: str | None = None,
        generated_by: str | None = None,
    ) -> ArtifactVersionModel:
        """Insert a new artifact version. Raises `InvalidEvidenceLinkage` if `evidence` is empty."""
        validate_evidence_links(evidence)
        db_artifact = ArtifactVersionModel(
            application_id=application_id,
            artifact_type=artifact_type,
            version=version,
            doc_id=doc_id,
            content=content,
            evidence=evidence,
            generated_by=generated_by,
        )
        self.session.add(db_artifact)
        self.session.commit()
        return db_artifact

    def get_by_application_id(self, application_id: UUID) -> list[ArtifactVersionModel]:
        """Get every artifact version generated for an application."""
        return (
            self.session.query(ArtifactVersionModel)
            .filter(ArtifactVersionModel.application_id == application_id)
            .all()
        )


class CommunicationEventRepository:
    """Repository for recruiter-side signals tied to an application.

    `create` lets the unique constraint on (application_id,
    provider_message_id) do the work: ingesting the same message twice
    raises `IntegrityError` rather than creating a duplicate row.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        application_id: UUID,
        classification: str,
        occurred_at: datetime,
        provider_message_id: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> CommunicationEventModel:
        """Insert a communication event. Raises `IntegrityError` on a duplicate provider message for the application."""
        db_event = CommunicationEventModel(
            application_id=application_id,
            classification=classification,
            provider_message_id=provider_message_id,
            occurred_at=occurred_at,
            metadata_json=metadata_json or {},
        )
        self.session.add(db_event)
        self.session.commit()
        return db_event

    def get_by_application_id(self, application_id: UUID) -> list[CommunicationEventModel]:
        """Get every communication event recorded for an application."""
        return (
            self.session.query(CommunicationEventModel)
            .filter(CommunicationEventModel.application_id == application_id)
            .all()
        )


class ToolExecutionRepository:
    """Repository for the tool-execution ledger.

    Backs idempotency at the tool-call level: `claim` lets the unique
    constraint on `idempotency_key` decide who owns execution, mirroring
    `OperationRepository` but scoped to a specific tool and workflow run, with
    `receipt_json` as the value a retried call replays.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def claim(
        self,
        idempotency_key: str,
        tool_name: str,
        workflow_id: UUID | None = None,
    ) -> tuple[ToolExecutionModel, bool]:
        """Claim an idempotency key for a tool call.

        Returns (record, claimed). When `claimed` is True the caller owns the
        execution and must call `complete` or `fail`. When False, a record for
        this key already exists -- if completed, its `receipt_json` should be
        replayed instead of running the tool again.
        """
        existing = self._row_for_key(idempotency_key)
        if existing is not None:
            return existing, False

        db_execution = ToolExecutionModel(
            workflow_id=workflow_id,
            tool_name=tool_name,
            idempotency_key=idempotency_key,
            status=ToolExecutionStatus.IN_PROGRESS.value,
        )
        self.session.add(db_execution)
        try:
            self.session.commit()
        except IntegrityError:
            # Lost the insert race: another attempt claimed this key first.
            self.session.rollback()
            winner = self._row_for_key(idempotency_key)
            if winner is None:  # pragma: no cover - unique violation implies a row
                raise
            return winner, False
        return db_execution, True

    def complete(self, idempotency_key: str, receipt: dict[str, Any]) -> ToolExecutionModel:
        """Record a successful outcome so a retried call replays `receipt`."""
        db_execution = self._require_row(idempotency_key)
        now = datetime.utcnow()
        db_execution.status = ToolExecutionStatus.COMPLETED.value
        db_execution.receipt_json = receipt
        db_execution.error = None
        db_execution.updated_at = now
        db_execution.completed_at = now
        self.session.commit()
        return db_execution

    def fail(self, idempotency_key: str, error: str) -> ToolExecutionModel:
        """Record a failed outcome."""
        db_execution = self._require_row(idempotency_key)
        db_execution.status = ToolExecutionStatus.FAILED.value
        db_execution.error = error
        db_execution.updated_at = datetime.utcnow()
        self.session.commit()
        return db_execution

    def get_by_key(self, idempotency_key: str) -> ToolExecutionModel | None:
        """Get the tool execution recorded under an idempotency key, if any."""
        return self._row_for_key(idempotency_key)

    def _row_for_key(self, idempotency_key: str) -> ToolExecutionModel | None:
        return (
            self.session.query(ToolExecutionModel)
            .filter(ToolExecutionModel.idempotency_key == idempotency_key)
            .first()
        )

    def _require_row(self, idempotency_key: str) -> ToolExecutionModel:
        db_execution = self._row_for_key(idempotency_key)
        if not db_execution:
            raise ValueError(f"Tool execution '{idempotency_key}' not found")
        return db_execution


class PolicyDecisionRepository:
    """Repository for recorded policy verdicts.

    Every decision the policy engine reaches is written here for audit,
    independent of whether the underlying tool call ever ran.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        principal: str,
        tool: str,
        args_hash: str,
        decision: str,
        workflow_id: UUID | None = None,
        requested_scopes: list[str] | None = None,
    ) -> PolicyDecisionModel:
        """Insert a policy decision record."""
        db_decision = PolicyDecisionModel(
            principal=principal,
            workflow_id=workflow_id,
            tool=tool,
            args_hash=args_hash,
            decision=decision,
            requested_scopes=requested_scopes or [],
        )
        self.session.add(db_decision)
        self.session.commit()
        return db_decision

    def get_by_workflow_id(self, workflow_id: UUID) -> list[PolicyDecisionModel]:
        """Get every policy decision recorded for a workflow run."""
        return (
            self.session.query(PolicyDecisionModel)
            .filter(PolicyDecisionModel.workflow_id == workflow_id)
            .all()
        )


class AuditEventRepository:
    """Repository for the append-only audit trail.

    Only `create` and reads are exposed here -- there is no `update` or
    `delete`, so application code has no path to rewrite or remove an event
    once it has been recorded.
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        *,
        actor: str,
        action: str,
        target_ref: str,
        result: str,
        workflow_id: UUID | None = None,
        policy_decision: str | None = None,
    ) -> AuditEventModel:
        """Append an audit event."""
        db_event = AuditEventModel(
            actor=actor,
            workflow_id=workflow_id,
            action=action,
            target_ref=target_ref,
            policy_decision=policy_decision,
            result=result,
        )
        self.session.add(db_event)
        self.session.commit()
        return db_event

    def get_by_workflow_id(self, workflow_id: UUID) -> list[AuditEventModel]:
        """Get every audit event recorded for a workflow run, oldest first."""
        return (
            self.session.query(AuditEventModel)
            .filter(AuditEventModel.workflow_id == workflow_id)
            .order_by(AuditEventModel.timestamp)
            .all()
        )
