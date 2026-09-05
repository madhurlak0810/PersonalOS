"""Job search API routes."""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from personalos.domain.errors import NotFound
from personalos.domain.models import Job, JobStatus
from personalos.persistence import get_session
from personalos.persistence.repositories import JobRepository

logger = logging.getLogger(__name__)

router = APIRouter()


class JobCreateRequest(BaseModel):
    """Request to create a job search.

    A Pydantic model rather than a plain class: FastAPI derives request
    validation and the OpenAPI schema from the annotation, and rejects the
    route outright if the type is not a valid field type.
    """

    title: str
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    salary_min: int | None = None
    salary_max: int | None = None
    job_type: str | None = None
    description: str | None = None


class JobSummaryResponse(BaseModel):
    """Minimal view returned when a job search is accepted."""

    id: UUID
    title: str
    status: JobStatus
    created_at: str

    @classmethod
    def from_domain(cls, job: Job) -> "JobSummaryResponse":
        """Project a domain job onto the summary view."""
        return cls(
            id=job.id,
            title=job.title,
            status=job.status,
            created_at=job.created_at.isoformat(),
        )


class JobResponse(BaseModel):
    """Full view of a job search and its results."""

    id: UUID
    title: str
    description: str | None = None
    status: JobStatus
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    salary_min: int | None = None
    salary_max: int | None = None
    job_type: str | None = None
    results_count: int = 0
    results: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_domain(cls, job: Job) -> "JobResponse":
        """Project a domain job onto the API view."""
        return cls(
            id=job.id,
            title=job.title,
            description=job.description,
            status=job.status,
            keywords=job.keywords,
            locations=job.locations,
            salary_min=job.salary_min,
            salary_max=job.salary_max,
            job_type=job.job_type,
            results_count=job.results_count,
            results=job.results,
            error_code=job.error_code,
            error_message=job.error_message,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
        )


class JobListResponse(BaseModel):
    """A page of job searches."""

    total: int
    jobs: list[JobResponse] = Field(default_factory=list)


@router.post("/", response_model=JobSummaryResponse, status_code=201)
async def create_job_search(
    request: JobCreateRequest,
    session: Session = Depends(get_session),
):
    """Create a new job search task.

    The search itself is not run inline. Execution belongs to the worker, which
    builds its own executor through ``personalos.bootstrap`` with its own
    session: a request-scoped session is closed before background work would
    finish, so it cannot be reused there.

    Any failure here (validation aside, which FastAPI handles before this body
    runs) propagates to the app-level handlers registered in
    ``apps.api.errors``, which log it with a stack trace and return the
    sanitized error envelope.
    """
    repo = JobRepository(session)

    job = repo.create(
        Job(
            title=request.title,
            description=request.description,
            keywords=request.keywords,
            locations=request.locations,
            salary_min=request.salary_min,
            salary_max=request.salary_max,
            job_type=request.job_type,
        )
    )
    logger.info(f"Created job search: {job.id}")

    return JobSummaryResponse.from_domain(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_search(
    job_id: UUID,
    session: Session = Depends(get_session),
):
    """Get job search details."""
    job = JobRepository(session).get_by_id(job_id)

    if not job:
        raise NotFound(f"job '{job_id}' not found", details={"job_id": str(job_id)})

    return JobResponse.from_domain(job)


@router.get("/", response_model=JobListResponse)
async def list_job_searches(session: Session = Depends(get_session)):
    """List all job searches."""
    jobs = JobRepository(session).get_all()

    return JobListResponse(
        total=len(jobs),
        jobs=[JobResponse.from_domain(job) for job in jobs],
    )
