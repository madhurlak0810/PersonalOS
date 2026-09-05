"""Core executor for agentic job search loops.

The executor owns *how* a job search runs: the step sequence, the state it
carries, and how results land in the repository. It does not own *whether* a
step is allowed to touch the outside world. Every tool call is expressed as a
:class:`~personalos.policy.intents.ToolIntent` and handed to a
:class:`~personalos.tools.gateway.ToolGateway`, which is the only thing in this
call path that can reach an adapter.

That is why this module imports no MCP types at all: it cannot call a tool
without a policy decision, because it has nothing to call.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from personalos.domain.errors import InternalError, PersonalOSError
from personalos.domain.models import AgentState, Job, JobStatus
from personalos.persistence.repositories import JobRepository
from personalos.policy import IntentOrigin, ToolIntent
from personalos.tools.gateway import ToolGateway, ToolResult

logger = logging.getLogger(__name__)

#: Server the job search steps target. Which tools on it are permitted is the
#: policy layer's call, not this module's.
JOBS_SERVER = "jobs"

#: Cap on detail scrapes per run, to stay inside upstream rate limits.
MAX_SCRAPE_TARGETS = 20

#: Number of top matches persisted on the job record.
MAX_PERSISTED_MATCHES = 10


class JobSearchExecutor:
    """Executes the job search loop over a policy-enforcing tool gateway."""

    def __init__(self, repo: JobRepository, gateway: ToolGateway):
        """Initialize with a repository and a tool gateway.

        The gateway is required rather than defaulted: an executor with no
        gateway would have to reach for a global tool manager, which is exactly
        the policy bypass this boundary exists to prevent.
        """
        if gateway is None:
            raise ValueError(
                "JobSearchExecutor requires a ToolGateway; construct one via "
                "personalos.bootstrap.build_job_search_executor"
            )
        self.repo = repo
        self.gateway = gateway

    async def run_job_search(self, job: Job, agent_id: UUID | None = None) -> Job:
        """Run a job search task."""
        if agent_id is None:
            agent_id = uuid4()

        logger.info(f"Starting job search: {job.id} with agent: {agent_id}")

        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job = self.repo.update(job)

        try:
            # Step 1: Prepare search parameters
            logger.info(
                f"Step 1: Preparing search for keywords={job.keywords}, "
                f"locations={job.locations}"
            )
            state = AgentState(
                agent_id=agent_id,
                job_id=job.id,
                current_step="prepare",
                step_data={"keywords": job.keywords, "locations": job.locations},
            )

            # Step 2: Search job listings
            logger.info("Step 2: Searching job listings...")
            search_results = await self._search(job, state)
            state.history.append({"step": "search", "results_count": len(search_results)})
            state.current_step = "search"
            state.step_data = {"results_count": len(search_results)}

            # Step 3: Scrape and enrich results
            logger.info(
                f"Step 3: Scraping details from "
                f"{min(MAX_SCRAPE_TARGETS, len(search_results))} results..."
            )
            detailed_results = await self._enrich(job, state, search_results)
            state.history.append({"step": "scrape", "detailed_count": len(detailed_results)})
            state.current_step = "scrape"

            # Step 4: Filter and rank results
            logger.info("Step 4: Filtering and ranking results...")
            filtered_results = await self._filter(job, state, detailed_results)
            state.history.append({"step": "filter", "filtered_count": len(filtered_results)})
            state.current_step = "filter"
            state.step_data = {"filtered_count": len(filtered_results)}

            # Update job with results
            job.results = {"matches": filtered_results[:MAX_PERSISTED_MATCHES]}
            job.results_count = len(filtered_results)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job = self.repo.update(job)

            logger.info(
                f"Job search completed: {job.id}, found {job.results_count} matches"
            )
            return job

        except Exception as e:
            # A `PersonalOSError` already carries a stable code and a message
            # that is safe to persist and later surface over the API. Anything
            # else is wrapped as a generic, sanitized `InternalError` for that
            # purpose -- the original exception's message may contain details
            # (a DB error, a raw adapter string) that should stay in the log
            # line below, not in a field the API can return to a caller.
            error = e if isinstance(e, PersonalOSError) else InternalError()
            logger.exception(
                "Job search failed (job_id=%s, error_code=%s, context_id=%s)",
                job.id,
                error.code.value,
                error.context_id,
            )
            job.status = JobStatus.FAILED
            job.error_code = error.code.value
            job.error_message = error.message
            job.updated_at = datetime.utcnow()
            job = self.repo.update(job)
            raise

    # ------------------------------------------------------------------
    # Steps. Each one states an intent; the gateway decides and executes.
    # ------------------------------------------------------------------

    async def _search(self, job: Job, state: AgentState) -> list[dict[str, Any]]:
        """Search for listings matching the job criteria."""
        result = await self._dispatch(
            job,
            state,
            tool="search_jobs",
            arguments={
                "keywords": job.keywords,
                "locations": job.locations,
                "job_type": job.job_type,
                "limit": 100,
            },
        )
        return result.unwrap().get("jobs", [])

    async def _enrich(
        self, job: Job, state: AgentState, listings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Scrape details for the top listings.

        A failed scrape falls back to the search result: partial detail is
        worth more here than losing the listing entirely.
        """
        detailed: list[dict[str, Any]] = []
        for listing in listings[:MAX_SCRAPE_TARGETS]:
            result = await self._dispatch(
                job,
                state,
                tool="scrape_job_details",
                arguments={
                    "job_id": listing.get("id", ""),
                    "job_url": listing.get("url"),
                },
            )
            detailed.append(result.result if result.success else listing)
        return detailed

    async def _filter(
        self, job: Job, state: AgentState, listings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter and rank the enriched listings."""
        result = await self._dispatch(
            job,
            state,
            tool="filter_jobs",
            arguments={
                "jobs": listings,
                "salary_min": job.salary_min,
                "salary_max": job.salary_max,
                "experience_level": None,
                "remote_only": False,
            },
        )
        return result.unwrap().get("jobs", [])

    async def _dispatch(
        self,
        job: Job,
        state: AgentState,
        *,
        tool: str,
        arguments: dict[str, Any],
        mutating: bool = False,
    ) -> ToolResult:
        """Express one step as an intent and submit it to the gateway.

        Intents built here are ``SYSTEM`` origin: their shape is fixed by this
        module, not by model output. Anything a model proposes must be built as
        an ``LLM``-origin intent so policy can treat it as untrusted.
        """
        intent = ToolIntent(
            server=JOBS_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=mutating,
            requested_by=f"executor:job_search#{state.current_step}",
            job_id=job.id,
            agent_id=state.agent_id,
        )
        return await self.gateway.dispatch(intent)
