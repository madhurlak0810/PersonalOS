"""Integration tests for Job Search Executor with MCP."""

import pytest
from uuid import uuid4

from personalos.domain.models import ActionTarget, Job, JobStatus, ToolCallRequest
from personalos.executor import JobSearchExecutor
from personalos.mcp.manager import MCPServerManager, get_mcp_manager
from personalos.persistence.repositories import JobRepository
from mcp_servers.jobs.server import JobsMCPServer


class MockSession:
    """Mock database session for testing."""

    def __init__(self):
        self.jobs = {}
        self.committed_jobs = {}

    def add(self, obj):
        pass

    def commit(self):
        pass

    def close(self):
        pass

    def query(self, model):
        return MockQuery(self.jobs, model)


class MockQuery:
    """Mock SQLAlchemy query."""

    def __init__(self, jobs, model):
        self.jobs = jobs
        self.model = model

    def filter(self, *args):
        return self

    def first(self):
        return None


class MockJobRepository(JobRepository):
    """Mock job repository for testing."""

    def __init__(self):
        self.jobs = {}
        self.session = MockSession()

    def create(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    def update(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    def get_by_id(self, job_id):
        return self.jobs.get(job_id)


@pytest.mark.asyncio
async def test_job_search_executor_with_mcp():
    """Test JobSearchExecutor integration with MCP servers."""
    # Setup
    repo = MockJobRepository()
    manager = MCPServerManager()
    jobs_server = JobsMCPServer()
    manager.register_server(jobs_server)

    # Create executor
    executor = JobSearchExecutor(repo)
    executor.mcp_manager = manager

    # Create job search request
    job = Job(
        title="Search Python Developer Jobs",
        keywords=["Python", "FastAPI"],
        locations=["Remote", "NYC"],
        salary_min=100000,
        salary_max=150000,
        job_type="full-time",
    )

    # Execute job search
    result_job = await executor.run_job_search(job)

    # Verify results
    assert result_job.status == JobStatus.COMPLETED
    assert result_job.results_count > 0
    assert "matches" in result_job.results
    assert len(result_job.results["matches"]) > 0


@pytest.mark.asyncio
async def test_executor_mcp_search_step():
    """Test executor's search step with MCP."""
    repo = MockJobRepository()
    manager = MCPServerManager()
    manager.register_server(JobsMCPServer())

    executor = JobSearchExecutor(repo)
    executor.mcp_manager = manager

    # Test search tool directly
    search_result = await manager.execute_tool(
        ToolCallRequest(
            target=ActionTarget(server="jobs", tool="search_jobs"),
            params={"keywords": ["Python"], "locations": ["Remote"]},
        )
    )

    assert search_result.ok is True
    assert search_result.result["total"] > 0


@pytest.mark.asyncio
async def test_executor_mcp_filter_step():
    """Test executor's filter step with MCP."""
    repo = MockJobRepository()
    manager = MCPServerManager()
    manager.register_server(JobsMCPServer())

    executor = JobSearchExecutor(repo)
    executor.mcp_manager = manager

    # Get some jobs first
    search_result = await manager.execute_tool(
        ToolCallRequest(
            target=ActionTarget(server="jobs", tool="search_jobs"),
            params={"keywords": ["Python"], "locations": ["Remote"]},
        )
    )

    jobs = search_result.result["jobs"]

    # Filter them
    filter_result = await manager.execute_tool(
        ToolCallRequest(
            target=ActionTarget(server="jobs", tool="filter_jobs"),
            params={"jobs": jobs, "salary_min": 100000, "salary_max": 150000},
        )
    )

    assert filter_result.ok is True
    filtered_jobs = filter_result.result["jobs"]
    assert len(filtered_jobs) > 0
