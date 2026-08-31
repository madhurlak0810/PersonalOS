"""Tests for MCP Server implementations."""

import pytest

from mcp_servers.jobs.server import JobsMCPServer
from personalos.mcp.base import ToolSchema
from personalos.mcp.manager import MCPServerManager


def test_jobs_mcp_server_initialization():
    """Test Jobs MCP Server initializes correctly."""
    server = JobsMCPServer()

    assert server.name == "jobs"
    assert server.description == "Job search and filtering capabilities"
    assert len(server.get_tools()) > 0


def test_jobs_mcp_server_tools():
    """Test Jobs MCP Server has all required tools."""
    server = JobsMCPServer()
    tools = server.get_tools()
    tool_names = [t.name for t in tools]

    assert "search_jobs" in tool_names
    assert "scrape_job_details" in tool_names
    assert "filter_jobs" in tool_names
    assert "save_favorite_job" in tool_names


@pytest.mark.asyncio
async def test_search_jobs_tool():
    """Test search_jobs tool execution."""
    server = JobsMCPServer()

    result = await server.execute(
        "search_jobs",
        keywords=["Python", "Developer"],
        locations=["Remote", "NYC"],
    )

    assert result["success"] is True
    assert "result" in result
    assert "jobs" in result["result"]
    assert result["result"]["total"] > 0


@pytest.mark.asyncio
async def test_scrape_job_details_tool():
    """Test scrape_job_details tool execution."""
    server = JobsMCPServer()

    result = await server.execute(
        "scrape_job_details",
        job_id="job_123",
        job_url="https://example.com/jobs/job_123",
    )

    assert result["success"] is True
    assert "result" in result
    job_details = result["result"]
    assert "description" in job_details
    assert "requirements" in job_details
    assert "benefits" in job_details


@pytest.mark.asyncio
async def test_filter_jobs_tool():
    """Test filter_jobs tool execution."""
    server = JobsMCPServer()

    # First search for jobs
    search_result = await server.execute(
        "search_jobs",
        keywords=["Python"],
        locations=["Remote"],
    )

    jobs = search_result["result"]["jobs"]

    # Then filter them
    filter_result = await server.execute(
        "filter_jobs",
        jobs=jobs,
        salary_min=100000,
        salary_max=150000,
        remote_only=True,
    )

    assert filter_result["success"] is True
    assert "result" in filter_result
    assert "jobs" in filter_result["result"]


@pytest.mark.asyncio
async def test_save_favorite_job_tool():
    """Test save_favorite_job tool execution."""
    server = JobsMCPServer()

    result = await server.execute(
        "save_favorite_job",
        job_id="job_123",
        notes="Interesting opportunity",
    )

    assert result["success"] is True
    assert "result" in result
    saved_job = result["result"]
    assert saved_job["job_id"] == "job_123"
    assert saved_job["notes"] == "Interesting opportunity"


def test_mcp_server_manager():
    """Test MCP Server Manager."""
    manager = MCPServerManager()
    server = JobsMCPServer()

    manager.register_server(server)

    assert manager.get_server("jobs") is not None
    assert "jobs" in manager.list_servers()
    assert len(manager.get_all_tools()) > 0


@pytest.mark.asyncio
async def test_mcp_server_manager_execute_tool():
    """Test MCP Server Manager execute_tool."""
    manager = MCPServerManager()
    server = JobsMCPServer()
    manager.register_server(server)

    result = await manager.execute_tool(
        "search_jobs",
        "jobs",
        keywords=["Python"],
        locations=["Remote"],
    )

    assert result["success"] is True


def test_mcp_server_manager_get_server_info():
    """Test MCP Server Manager get_server_info."""
    manager = MCPServerManager()
    server = JobsMCPServer()
    manager.register_server(server)

    info = manager.get_server_info("jobs")

    assert info is not None
    assert info["name"] == "jobs"
    assert "tools" in info
    assert len(info["tools"]) > 0
