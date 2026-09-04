"""Jobs MCP Server - Provides job search capabilities."""

import logging
from typing import Any

from personalos.domain.models import Intent, MutatingIntent
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.mcp.cache import get_cache
from personalos.persistence.idempotency import InMemoryOperationStore, OperationStore

logger = logging.getLogger(__name__)


class SearchJobsIntent(Intent):
    """Typed parameters for `search_jobs`."""

    keywords: list[str]
    locations: list[str]
    job_type: str | None = None
    limit: int = 50


class ScrapeJobDetailsIntent(Intent):
    """Typed parameters for `scrape_job_details`."""

    job_id: str
    job_url: str | None = None


class FilterJobsIntent(Intent):
    """Typed parameters for `filter_jobs`."""

    jobs: list[dict[str, Any]]
    salary_min: int | None = None
    salary_max: int | None = None
    experience_level: str | None = None
    remote_only: bool = False


class SaveFavoriteJobIntent(MutatingIntent):
    """Typed parameters for the mutating `save_favorite_job`."""

    job_id: str
    notes: str | None = None


class JobsMCPServer(MCPServer):
    """MCP Server for job search operations."""

    def __init__(self, operation_store: OperationStore | None = None):
        """Initialize Jobs MCP Server.

        Defaults to a process-local operation store so mutating tools are
        deduplicated out of the box; pass a SqlOperationStore to make that
        dedup durable across restarts and workers.
        """
        super().__init__(
            "jobs",
            "Job search and filtering capabilities",
            operation_store=operation_store or InMemoryOperationStore(),
        )
        self.cache = get_cache()
        self.initialize()

    def initialize(self):
        """Register all job search tools."""
        # Tool 1: Search jobs
        search_schema = ToolSchema(
            name="search_jobs",
            description="Search for job listings across multiple job boards",
            intent_type=SearchJobsIntent,
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Job keywords to search for",
                    },
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Job locations",
                    },
                    "job_type": {
                        "type": "string",
                        "enum": ["full-time", "part-time", "contract", "temporary"],
                        "description": "Type of job",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum results to return",
                    },
                },
            },
            required=["keywords", "locations"],
        )
        self.register_tool(search_schema, self._search_jobs)

        # Tool 2: Scrape job details
        scrape_schema = ToolSchema(
            name="scrape_job_details",
            description="Scrape detailed information from a job posting",
            intent_type=ScrapeJobDetailsIntent,
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Unique job ID",
                    },
                    "job_url": {
                        "type": "string",
                        "description": "URL to the job posting",
                    },
                },
            },
            required=["job_id"],
        )
        self.register_tool(scrape_schema, self._scrape_job_details)

        # Tool 3: Filter and rank jobs
        filter_schema = ToolSchema(
            name="filter_jobs",
            description="Filter and rank job listings based on criteria",
            intent_type=FilterJobsIntent,
            parameters={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "description": "List of job postings to filter",
                    },
                    "salary_min": {
                        "type": "integer",
                        "description": "Minimum salary",
                    },
                    "salary_max": {
                        "type": "integer",
                        "description": "Maximum salary",
                    },
                    "experience_level": {
                        "type": "string",
                        "enum": ["entry", "mid", "senior", "lead"],
                        "description": "Required experience level",
                    },
                    "remote_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Filter for remote positions only",
                    },
                },
            },
            required=["jobs"],
        )
        self.register_tool(filter_schema, self._filter_jobs)

        # Tool 4: Save favorite job
        favorite_schema = ToolSchema(
            name="save_favorite_job",
            description="Save a job to favorites for later review",
            intent_type=SaveFavoriteJobIntent,
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to save",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about the job",
                    },
                },
            },
            required=["job_id"],
        )
        self.register_tool(favorite_schema, self._save_favorite_job)

    async def _search_jobs(
        self,
        keywords: list[str],
        locations: list[str],
        job_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search for job listings."""
        logger.info(f"Searching jobs: keywords={keywords}, locations={locations}")

        # Placeholder: In production, would integrate with Indeed, LinkedIn, etc.
        # For now, return mock data
        results = []

        for i, keyword in enumerate(keywords):
            for j, location in enumerate(locations):
                for k in range(min(5, limit)):
                    job_id = f"job_{i}_{j}_{k}"
                    results.append({
                        "id": job_id,
                        "title": f"{keyword} Position in {location}",
                        "company": f"Company {k + 1}",
                        "location": location,
                        "job_type": job_type or "full-time",
                        "salary": f"${60000 + k * 10000}-${80000 + k * 10000}",
                        "posting_date": "2026-08-30",
                        "url": f"https://example.com/jobs/{job_id}",
                        "description_snippet": f"Looking for a {keyword} professional in {location}",
                        "relevance_score": 0.85 - (k * 0.05),
                    })

        logger.info(f"Found {len(results)} job listings")
        return {
            "total": len(results),
            "jobs": results[:limit],
            "search_params": {
                "keywords": keywords,
                "locations": locations,
                "job_type": job_type,
            },
        }

    async def _scrape_job_details(
        self, job_id: str, job_url: str | None = None
    ) -> dict[str, Any]:
        """Scrape detailed information from a job posting."""
        logger.info(f"Scraping job details: job_id={job_id}, url={job_url}")

        # Placeholder: Would use BeautifulSoup or Playwright to scrape HTML
        # For now, return mock detailed data
        return {
            "id": job_id,
            "title": "Senior Python Developer",
            "company": "TechCorp Inc.",
            "location": "Remote",
            "job_type": "full-time",
            "salary": "$120,000 - $160,000",
            "posting_date": "2026-08-25",
            "url": job_url or "https://example.com/jobs/job_123",
            "description": """
            We are looking for an experienced Python Developer to join our team.

            Responsibilities:
            - Develop and maintain Python applications
            - Collaborate with team members
            - Write clean, maintainable code
            - Participate in code reviews

            Requirements:
            - 5+ years of Python experience
            - Experience with FastAPI or Django
            - Strong SQL knowledge
            - Experience with Docker and Kubernetes

            Nice to have:
            - Machine learning experience
            - GraphQL knowledge
            - AWS or GCP experience
            """,
            "requirements": [
                "5+ years of Python experience",
                "Experience with FastAPI or Django",
                "Strong SQL knowledge",
                "Experience with Docker and Kubernetes",
            ],
            "benefits": [
                "Competitive salary",
                "Health insurance",
                "Remote work",
                "Professional development budget",
                "Flexible hours",
            ],
            "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
            "experience_level": "senior",
            "difficulty_match": 0.92,
        }

    async def _filter_jobs(
        self,
        jobs: list[dict[str, Any]],
        salary_min: int | None = None,
        salary_max: int | None = None,
        experience_level: str | None = None,
        remote_only: bool = False,
    ) -> dict[str, Any]:
        """Filter and rank job listings."""
        logger.info(f"Filtering {len(jobs)} jobs with criteria")

        filtered = jobs.copy()

        # Filter by remote
        if remote_only:
            filtered = [
                j for j in filtered if j.get("location", "").lower() == "remote"
            ]

        # Filter by experience level
        if experience_level:
            filtered = [
                j
                for j in filtered
                if j.get("experience_level", "") == experience_level
            ]

        # Filter by salary (mock parsing)
        if salary_min or salary_max:
            # In production, would parse salary ranges properly
            filtered = [
                j for j in filtered if _salary_in_range(j.get("salary", ""), salary_min, salary_max)
            ]

        # Rank by relevance score
        filtered_ranked = sorted(
            filtered, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        logger.info(f"Filtered to {len(filtered_ranked)} jobs")
        return {
            "total_filtered": len(filtered_ranked),
            "total_input": len(jobs),
            "filters_applied": {
                "salary_min": salary_min,
                "salary_max": salary_max,
                "experience_level": experience_level,
                "remote_only": remote_only,
            },
            "jobs": filtered_ranked,
        }

    async def _save_favorite_job(self, job_id: str, notes: str | None = None) -> dict[str, Any]:
        """Save a job to favorites."""
        logger.info(f"Saving job to favorites: job_id={job_id}")

        # In production, would save to database
        return {
            "success": True,
            "job_id": job_id,
            "saved_at": "2026-08-30T12:00:00Z",
            "notes": notes or "",
            "message": f"Job {job_id} saved to favorites",
        }


def _salary_in_range(salary_str: str, min_val: int | None, max_val: int | None) -> bool:
    """Check if salary string is within range (mock implementation)."""
    # In production, would parse salary properly
    # For now, simple heuristic
    if not min_val and not max_val:
        return True
    if "$" not in salary_str:
        return True
    return True  # Accept all for mock
