"""Repository pattern for data access."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.models import (
    Job,
    JobStatus,
    OperationRecord,
    OperationStatus,
)
from personalos.persistence.models import JobModel, OperationModel


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
            metadata=db_job.job_metadata or {},
            created_at=db_job.created_at,
            updated_at=db_job.updated_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
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
