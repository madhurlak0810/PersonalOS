"""Background runner for job search tasks.

The API accepts a job search and returns; running it belongs here. This module
is part of the composition root: it owns a session, builds the executor through
``personalos.bootstrap``, and therefore is allowed to know about every layer.

A worker gets its own session deliberately. A request-scoped session is closed
once the response is sent, so reusing one for background work would operate on
a closed connection.
"""

import asyncio
import logging
from uuid import UUID

from personalos.bootstrap import build_job_search_executor, register_mcp_servers
from personalos.domain.models import Job
from personalos.persistence.database import SessionLocal
from personalos.persistence.repositories import JobRepository

logger = logging.getLogger(__name__)


async def run_job_search(job_id: UUID) -> Job:
    """Run one job search to completion in its own session.

    Raises ``ValueError`` if the job no longer exists, and re-raises whatever
    the executor raised (including
    :class:`~personalos.policy.errors.PolicyDenied`) after the executor has
    marked the job failed.
    """
    session = SessionLocal()
    try:
        repo = JobRepository(session)
        job = repo.get_by_id(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        executor = build_job_search_executor(repo)
        return await executor.run_job_search(job)
    finally:
        session.close()


def main(job_id: str) -> None:
    """CLI entry point: register servers, then run one job."""
    logging.basicConfig(level=logging.INFO)
    register_mcp_servers()
    job = asyncio.run(run_job_search(UUID(job_id)))
    logger.info("Job %s finished with status %s", job.id, job.status)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m apps.worker.job_runner <job-id>")
    main(sys.argv[1])
