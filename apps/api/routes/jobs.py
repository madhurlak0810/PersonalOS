"""Job search API routes."""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

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
    """
    try:
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
    except Exception as e:
        logger.exception("Error creating job search")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_search(
    job_id: UUID,
    session: Session = Depends(get_session),
):
    """Get job search details."""
    try:
        job = JobRepository(session).get_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobResponse.from_domain(job)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting job")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/", response_model=JobListResponse)
async def list_job_searches(session: Session = Depends(get_session)):
    """List all job searches."""
    try:
        jobs = JobRepository(session).get_all()

        return JobListResponse(
            total=len(jobs),
            jobs=[JobResponse.from_domain(job) for job in jobs],
        )
    except Exception as e:
        logger.exception("Error listing jobs")
        raise HTTPException(status_code=500, detail=str(e)) from e
