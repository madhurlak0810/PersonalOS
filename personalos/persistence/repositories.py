"""Repository pattern for data access."""

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.models import Job, JobStatus
from personalos.persistence.models import JobModel


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
            metadata=job.metadata,
        )
        self.session.add(db_job)
        self.session.commit()
        return self._to_domain(db_job)

    def get_by_id(self, job_id: UUID) -> Optional[Job]:
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
        db_job.metadata = job.metadata
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
            metadata=db_job.metadata or {},
            created_at=db_job.created_at,
            updated_at=db_job.updated_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
        )
