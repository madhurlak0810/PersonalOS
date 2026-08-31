"""MCP Server orchestration and management."""

import logging
from typing import Dict, List, Optional

from personalos.domain.models import Tool
from personalos.mcp.base import MCPServer

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages multiple MCP servers."""

    def __init__(self):
        """Initialize MCP server manager."""
        self._servers: Dict[str, MCPServer] = {}

    def register_server(self, server: MCPServer):
        """Register an MCP server."""
        self._servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")

    def get_server(self, name: str) -> Optional[MCPServer]:
        """Get MCP server by name."""
        return self._servers.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Get all available tools from all servers."""
        tools = []
        for server in self._servers.values():
            tools.extend(server.get_tools())
        return tools

    async def execute_tool(self, tool_name: str, server_name: str, **kwargs) -> dict:
        """Execute a tool from a specific server."""
        server = self.get_server(server_name)
        if not server:
            return {"success": False, "error": f"Server '{server_name}' not found"}

        return await server.execute(tool_name, **kwargs)

    def list_servers(self) -> List[str]:
        """List all registered server names."""
        return list(self._servers.keys())

    def get_server_info(self, server_name: str) -> Optional[dict]:
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


def initialize_mcp_servers():
    """Initialize all MCP servers."""
    logger.info("Initializing MCP servers...")

    # Import and register Jobs MCP Server
    from mcp_servers.jobs.server import JobsMCPServer

    jobs_server = JobsMCPServer()
    _mcp_manager.register_server(jobs_server)

    # Placeholder for other servers
    # from mcp_servers.files.server import FilesMCPServer
    # files_server = FilesMCPServer()
    # _mcp_manager.register_server(files_server)

    logger.info(f"Initialized {len(_mcp_manager.list_servers())} MCP servers")
