"""Tests for the MCP server base class and the server manager.

The behaviour under test belongs to `MCPServer` and `MCPServerManager`, not to
any particular server, so a minimal stub server stands in for a real one.
"""

from uuid import uuid4

import pytest

from personalos.domain.models import (
    ActionTarget,
    Intent,
    MutatingIntent,
    ToolCallErrorCode,
    ToolCallRequest,
)
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.mcp.manager import MCPServerManager
from personalos.persistence.idempotency import InMemoryOperationStore


class LookupIntent(Intent):
    """Typed parameters for the read-only `lookup` tool."""

    key: str
    limit: int = 10


class SaveIntent(MutatingIntent):
    """Typed parameters for the mutating `save` tool."""

    key: str
    notes: str | None = None


class StubMCPServer(MCPServer):
    """One read-only tool and one mutating tool, nothing else."""

    def __init__(self):
        super().__init__("stub", "Stub server", operation_store=InMemoryOperationStore())
        self.saves = 0
        self.initialize()

    def initialize(self):
        """Register the stub tools."""
        self.register_tool(
            ToolSchema(
                name="lookup",
                description="Look a key up",
                intent_type=LookupIntent,
                required=["key"],
            ),
            self._lookup,
        )
        self.register_tool(
            ToolSchema(
                name="save",
                description="Save a key",
                intent_type=SaveIntent,
                required=["key"],
            ),
            self._save,
        )

    async def _lookup(self, key: str, limit: int = 10) -> dict:
        return {"key": key, "limit": limit}

    async def _save(self, key: str, notes: str | None = None) -> dict:
        self.saves += 1
        return {"key": key, "notes": notes or "", "save": self.saves}


def request(tool: str, server: str = "stub", **params) -> ToolCallRequest:
    """Build a ToolCallRequest against the stub server."""
    return ToolCallRequest(target=ActionTarget(server=server, tool=tool), params=params)


def test_server_lists_its_registered_tools():
    """Every registered tool is exposed."""
    server = StubMCPServer()
    tools = {t.name: t for t in server.get_tools()}

    assert set(tools) == {"lookup", "save"}


@pytest.mark.asyncio
async def test_read_tool_executes():
    """A valid read call reaches its handler."""
    result = await StubMCPServer().execute(request("lookup", key="k1"))

    assert result.ok is True
    assert result.error is None
    assert result.result == {"key": "k1", "limit": 10}


@pytest.mark.asyncio
async def test_mutating_tool_is_idempotent():
    """A mutating tool retried under the same key does not run twice."""
    server = StubMCPServer()
    key = str(uuid4())

    first = await server.execute(request("save", key="k1", idempotency_key=key))
    second = await server.execute(request("save", key="k1", idempotency_key=key))

    assert first.ok is True
    assert first.replayed is False
    assert second.ok is True
    assert second.replayed is True
    assert second.result == first.result
    assert server.saves == 1


@pytest.mark.asyncio
async def test_mutating_tool_requires_idempotency_key():
    """The mutating tool refuses to run without an idempotency key."""
    result = await StubMCPServer().execute(request("save", key="k1"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert "idempotency_key" in result.error.message


@pytest.mark.asyncio
async def test_unknown_tool_is_typed_not_found():
    """Calling a tool the server doesn't have returns a typed not-found failure."""
    result = await StubMCPServer().execute(request("does_not_exist"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_invalid_params_are_typed_validation_failure():
    """Missing required fields fail as a typed validation error, not an exception."""
    result = await StubMCPServer().execute(request("lookup"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert result.error.details["errors"]


@pytest.mark.asyncio
async def test_unexpected_extra_field_is_rejected():
    """Intents forbid unknown fields, so typos surface as a validation failure."""
    result = await StubMCPServer().execute(request("lookup", key="k1", not_a_field=True))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR


def test_mcp_server_manager():
    """Test MCP Server Manager."""
    manager = MCPServerManager()
    manager.register_server(StubMCPServer())

    assert manager.get_server("stub") is not None
    assert "stub" in manager.list_servers()
    assert len(manager.get_all_tools()) > 0


@pytest.mark.asyncio
async def test_mcp_server_manager_execute_tool():
    """Test MCP Server Manager execute_tool."""
    manager = MCPServerManager()
    manager.register_server(StubMCPServer())

    result = await manager.execute_tool(request("lookup", key="k1"))

    assert result.ok is True


@pytest.mark.asyncio
async def test_mcp_server_manager_execute_tool_unknown_server():
    """An unknown server target is a typed failure, not a raised exception."""
    manager = MCPServerManager()

    result = await manager.execute_tool(request("lookup", server="ghost", key="k1"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.SERVER_NOT_FOUND


def test_mcp_server_manager_get_server_info():
    """Test MCP Server Manager get_server_info."""
    manager = MCPServerManager()
    manager.register_server(StubMCPServer())

    info = manager.get_server_info("stub")

    assert info is not None
    assert info["name"] == "stub"
    assert len(info["tools"]) == 2
