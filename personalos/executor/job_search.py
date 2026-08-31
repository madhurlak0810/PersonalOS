"""Core executor for agentic job search loops."""

import logging
from datetime import datetime
from uuid import UUID, uuid4

from personalos.domain.models import AgentState, Event, EventType, Job, JobStatus
from personalos.mcp.manager import get_mcp_manager
from personalos.persistence.repositories import JobRepository

logger = logging.getLogger(__name__)


class JobSearchExecutor:
    """Executes job search agents."""

    def __init__(self, repo: JobRepository):
        """Initialize executor with repository."""
        self.repo = repo
        self.mcp_manager = get_mcp_manager()

    async def run_job_search(self, job: Job, agent_id: UUID = None) -> Job:
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
            logger.info(f"Step 1: Preparing search for keywords={job.keywords}, locations={job.locations}")
            state = AgentState(
                agent_id=agent_id,
                job_id=job.id,
                current_step="prepare",
                step_data={"keywords": job.keywords, "locations": job.locations},
            )

            # Step 2: Search job listings using Jobs MCP Server
            logger.info("Step 2: Searching job listings via MCP...")
            search_result = await self.mcp_manager.execute_tool(
                "search_jobs",
                "jobs",
                keywords=job.keywords,
                locations=job.locations,
                job_type=job.job_type,
                limit=100,
            )

            if not search_result.get("success"):
                raise Exception(f"Search failed: {search_result.get('error')}")

            search_results = search_result.get("result", {}).get("jobs", [])
            state.history.append({"step": "search", "results_count": len(search_results)})
            state.current_step = "search"
            state.step_data = {"results_count": len(search_results)}

            # Step 3: Scrape and enrich results
            logger.info(f"Step 3: Scraping details from {min(20, len(search_results))} results...")
            detailed_results = []
            for search_job in search_results[:20]:  # Limit to top 20 to avoid rate limits
                scrape_result = await self.mcp_manager.execute_tool(
                    "scrape_job_details",
                    "jobs",
                    job_id=search_job.get("id", ""),
                    job_url=search_job.get("url"),
                )

                if scrape_result.get("success"):
                    detailed_results.append(scrape_result.get("result", {}))
                else:
                    # Fall back to search result if scraping fails
                    detailed_results.append(search_job)

            state.history.append({"step": "scrape", "detailed_count": len(detailed_results)})
            state.current_step = "scrape"

            # Step 4: Filter and rank results
            logger.info("Step 4: Filtering and ranking results...")
            filter_result = await self.mcp_manager.execute_tool(
                "filter_jobs",
                "jobs",
                jobs=detailed_results,
                salary_min=job.salary_min,
                salary_max=job.salary_max,
                experience_level=None,
                remote_only=False,
            )

            if not filter_result.get("success"):
                raise Exception(f"Filter failed: {filter_result.get('error')}")

            filtered_results = filter_result.get("result", {}).get("jobs", [])
            state.history.append({"step": "filter", "filtered_count": len(filtered_results)})
            state.current_step = "filter"
            state.step_data = {"filtered_count": len(filtered_results)}

            # Update job with results
            job.results = {"matches": filtered_results[:10]}  # Top 10
            job.results_count = len(filtered_results)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job = self.repo.update(job)

            logger.info(f"Job search completed: {job.id}, found {job.results_count} matches")
            return job

        except Exception as e:
            logger.error(f"Job search failed: {str(e)}", exc_info=True)
            job.status = JobStatus.FAILED
            job.updated_at = datetime.utcnow()
            job = self.repo.update(job)
            raise

    async def _search_jobs(self, keywords: list[str], locations: list[str]) -> list[dict]:
        """Search for job listings (legacy - now uses MCP)."""
        # Placeholder: would integrate with job search APIs (Indeed, LinkedIn, etc.)
        # For now, returns mock data
        results = []
        for keyword in keywords:
            for location in locations:
                results.append({
                    "title": f"{keyword} Position in {location}",
                    "location": location,
                    "company": "Example Corp",
                    "salary": "50000-80000",
                    "url": f"https://example.com/job/{keyword}/{location}",
                    "relevance_score": 0.8,
                })
        return results

    async def _filter_results(
        self,
        results: list[dict],
        salary_min: int = None,
        salary_max: int = None,
        job_type: str = None,
    ) -> list[dict]:
        """Filter results based on criteria (legacy - now uses MCP)."""
        filtered = results

        # Filter by job type if specified
        if job_type:
            filtered = [r for r in filtered if job_type.lower() in r.get("title", "").lower()]

        # Salary filtering would be more sophisticated in production
        if salary_min or salary_max:
            # Parse salary ranges and filter
            pass

        return filtered

    async def _rank_results(self, results: list[dict], keywords: list[str]) -> list[dict]:
        """Rank results by relevance (legacy - now uses MCP)."""
        # Calculate relevance scores based on keyword matches
        for result in results:
            keyword_matches = sum(
                1 for kw in keywords if kw.lower() in result.get("title", "").lower()
            )
            result["relevance_score"] = min(1.0, 0.5 + (keyword_matches * 0.2))

        # Sort by relevance score descending
        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
