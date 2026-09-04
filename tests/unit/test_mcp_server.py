"""Tests for MCP Server implementations."""

from uuid import uuid4

import pytest

from mcp_servers.jobs.server import JobsMCPServer
from personalos.domain.models import (
    ActionTarget,
    ToolCallErrorCode,
    ToolCallRequest,
)
from personalos.mcp.manager import MCPServerManager


def request(tool: str, server: str = "jobs", **params) -> ToolCallRequest:
    """Build a ToolCallRequest against the jobs server."""
    return ToolCallRequest(target=ActionTarget(server=server, tool=tool), params=params)


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
        request("search_jobs", keywords=["Python", "Developer"], locations=["Remote", "NYC"])
    )

    assert result.ok is True
    assert result.error is None
    assert "jobs" in result.result
    assert result.result["total"] > 0


@pytest.mark.asyncio
async def test_scrape_job_details_tool():
    """Test scrape_job_details tool execution."""
    server = JobsMCPServer()

    result = await server.execute(
        request(
            "scrape_job_details",
            job_id="job_123",
            job_url="https://example.com/jobs/job_123",
        )
    )

    assert result.ok is True
    job_details = result.result
    assert "description" in job_details
    assert "requirements" in job_details
    assert "benefits" in job_details


@pytest.mark.asyncio
async def test_filter_jobs_tool():
    """Test filter_jobs tool execution."""
    server = JobsMCPServer()

    # First search for jobs
    search_result = await server.execute(
        request("search_jobs", keywords=["Python"], locations=["Remote"])
    )

    jobs = search_result.result["jobs"]

    # Then filter them
    filter_result = await server.execute(
        request(
            "filter_jobs",
            jobs=jobs,
            salary_min=100000,
            salary_max=150000,
            remote_only=True,
        )
    )

    assert filter_result.ok is True
    assert "jobs" in filter_result.result


@pytest.mark.asyncio
async def test_save_favorite_job_tool():
    """Test save_favorite_job tool execution."""
    server = JobsMCPServer()

    result = await server.execute(
        request(
            "save_favorite_job",
            job_id="job_123",
            notes="Interesting opportunity",
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is True
    assert result.replayed is False
    saved_job = result.result
    assert saved_job["job_id"] == "job_123"
    assert saved_job["notes"] == "Interesting opportunity"


@pytest.mark.asyncio
async def test_save_favorite_job_is_idempotent():
    """save_favorite_job mutates, so a retry must not save twice."""
    server = JobsMCPServer()
    key = str(uuid4())

    first = await server.execute(request("save_favorite_job", job_id="job_123", idempotency_key=key))
    second = await server.execute(request("save_favorite_job", job_id="job_123", idempotency_key=key))

    assert first.ok is True
    assert second.ok is True
    assert second.replayed is True
    assert second.result == first.result


@pytest.mark.asyncio
async def test_save_favorite_job_requires_idempotency_key():
    """The mutating tool refuses to run without an idempotency key."""
    server = JobsMCPServer()

    result = await server.execute(request("save_favorite_job", job_id="job_123"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert "idempotency_key" in result.error.message


@pytest.mark.asyncio
async def test_unknown_tool_is_typed_not_found():
    """Calling a tool the server doesn't have returns a typed not-found failure."""
    server = JobsMCPServer()

    result = await server.execute(request("does_not_exist"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_invalid_params_are_typed_validation_failure():
    """Missing required fields fail as a typed validation error, not an exception."""
    server = JobsMCPServer()

    result = await server.execute(request("search_jobs", keywords=["Python"]))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert result.error.details["errors"]


@pytest.mark.asyncio
async def test_unexpected_extra_field_is_rejected():
    """Intents forbid unknown fields, so typos surface as a validation failure."""
    server = JobsMCPServer()

    result = await server.execute(
        request("search_jobs", keywords=["Python"], locations=["Remote"], not_a_field=True)
    )

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR


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
        request("search_jobs", keywords=["Python"], locations=["Remote"])
    )

    assert result.ok is True


@pytest.mark.asyncio
async def test_mcp_server_manager_execute_tool_unknown_server():
    """An unknown server target is a typed failure, not a raised exception."""
    manager = MCPServerManager()

    result = await manager.execute_tool(request("search_jobs", server="ghost"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.SERVER_NOT_FOUND


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
