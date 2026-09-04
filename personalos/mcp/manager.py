"""MCP Server orchestration and management."""

import logging

from personalos.domain.models import (
    Tool,
    ToolCallErrorCode,
    ToolCallRequest,
    ToolCallResult,
)
from personalos.mcp.base import MCPServer

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages multiple MCP servers."""

    def __init__(self):
        """Initialize MCP server manager."""
        self._servers: dict[str, MCPServer] = {}

    def register_server(self, server: MCPServer):
        """Register an MCP server."""
        self._servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")

    def get_server(self, name: str) -> MCPServer | None:
        """Get MCP server by name."""
        return self._servers.get(name)

    def get_all_tools(self) -> list[Tool]:
        """Get all available tools from all servers."""
        tools = []
        for server in self._servers.values():
            tools.extend(server.get_tools())
        return tools

    async def execute_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """Execute a tool call against its target server."""
        server = self.get_server(request.target.server)
        if not server:
            return ToolCallResult.failed(
                request.target,
                ToolCallErrorCode.SERVER_NOT_FOUND,
                f"Server '{request.target.server}' not found",
            )

        return await server.execute(request)

    def list_servers(self) -> list[str]:
        """List all registered server names."""
        return list(self._servers.keys())

    def get_server_info(self, server_name: str) -> dict | None:
        """Get information about a server."""
        server = self.get_server(server_name)
        if not server:
            return None

        return {
            "name": server.name,
            "description": server.description,
            "tools": [{"name": t.name, "description": t.description} for t in server.get_tools()],
        }


# Global MCP server manager instance
_mcp_manager = MCPServerManager()


def get_mcp_manager() -> MCPServerManager:
    """Get the global MCP server manager."""
    return _mcp_manager
