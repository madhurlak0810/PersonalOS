"""Health of an MCP server: is it registered, and does it still answer?

An in-process server is healthy as long as it is registered with tools; one
in another process (`RemoteMCPServer`) is also pinged. Whether the account
behind the server's tools works is not this check's question -- the Gmail and
Calendar adapters ask that with a real call each.
"""

from personalos.domain.errors import RetryableFailure
from personalos.domain.health import HealthCheckResult, run_check
from personalos.mcp.manager import MCPServerManager


class MCPServerHealthCheck:
    """`HealthCheck` for one MCP server on a manager."""

    def __init__(self, manager: MCPServerManager, server: str, *, integration: str):
        """Take the manager the server is registered on, its name, and the integration name."""
        self.manager = manager
        self.server = server
        self.integration = integration

    async def health_check(self) -> HealthCheckResult:
        """Down when the server is not registered, lists no tools, or does not answer a ping."""

        async def probe() -> None:
            server = self.manager.get_server(self.server)
            if server is None:
                raise RetryableFailure(f"MCP server '{self.server}' is not registered")
            if not server.get_tools():
                raise RetryableFailure(f"MCP server '{self.server}' lists no tools")
            ping = getattr(server, "ping", None)
            if ping is not None:
                await ping()

        return await run_check(self.integration, probe)


__all__ = ["MCPServerHealthCheck"]
