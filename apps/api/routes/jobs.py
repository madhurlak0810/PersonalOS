"""Job search API routes."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from personalos.domain.models import Job, JobStatus
from personalos.executor import JobSearchExecutor
from personalos.persistence import get_session
from personalos.persistence.repositories import JobRepository

logger = logging.getLogger(__name__)

router = APIRouter()


class JobCreateRequest:
    """Request to create a job search."""

    def __init__(
        self,
        title: str,
        keywords: list[str],
        locations: list[str],
        salary_min: int = None,
        salary_max: int = None,
        job_type: str = None,
        description: str = None,
    ):
        self.title = title
        self.keywords = keywords
        self.locations = locations
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.job_type = job_type
        self.description = description


class JobResponse:
    """Response with job data."""

    def __init__(self, job: Job):
        self.id = str(job.id)
        self.title = job.title
        self.description = job.description
        self.status = job.status
        self.keywords = job.keywords
        self.locations = job.locations
        self.salary_min = job.salary_min
        self.salary_max = job.salary_max
        self.job_type = job.job_type
        self.results_count = job.results_count
        self.results = job.results
        self.created_at = job.created_at.isoformat()
        self.updated_at = job.updated_at.isoformat()
        self.started_at = job.started_at.isoformat() if job.started_at else None
        self.completed_at = job.completed_at.isoformat() if job.completed_at else None


@router.post("/", response_model=dict)
async def create_job_search(
    request: JobCreateRequest,
    session: Session = Depends(get_session),
):
    """Create a new job search task."""
    try:
        repo = JobRepository(session)

        # Create job
        job = Job(
            title=request.title,
            description=request.description,
            keywords=request.keywords,
            locations=request.locations,
            salary_min=request.salary_min,
            salary_max=request.salary_max,
            job_type=request.job_type,
        )

        job = repo.create(job)
        logger.info(f"Created job search: {job.id}")

        # Start executor in background
        executor = JobSearchExecutor(repo)
        # Note: In production, this would be queued to a background worker

        return {
            "id": str(job.id),
            "title": job.title,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error creating job search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=dict)
async def get_job_search(
    job_id: UUID,
    session: Session = Depends(get_session),
):
    """Get job search details."""
    try:
        repo = JobRepository(session)
        job = repo.get_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        response = JobResponse(job)
        return response.__dict__
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=dict)
async def list_job_searches(session: Session = Depends(get_session)):
    """List all job searches."""
    try:
        repo = JobRepository(session)
        jobs = repo.get_all()

        return {
            "total": len(jobs),
            "jobs": [JobResponse(job).__dict__ for job in jobs],
        }
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
