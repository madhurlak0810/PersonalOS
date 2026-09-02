"""MCP adapter for the tool boundary.

Bridges the :class:`~personalos.tools.gateway.ToolInvoker` port to the MCP
server manager. This is the only place in the codebase that turns an approved
intent back into loose keyword arguments, and it refuses to do so for anything
that has not been through policy.
"""

import logging
from typing import Any

from personalos.mcp.manager import MCPServerManager
from personalos.policy import ApprovedIntent, PolicyViolation

logger = logging.getLogger(__name__)


class MCPToolInvoker:
    """Executes approved intents against registered MCP servers."""

    def __init__(self, manager: MCPServerManager):
        """Take the manager holding the registered servers."""
        self.manager = manager

    async def invoke(self, approved: ApprovedIntent) -> dict[str, Any]:
        """Run the intent on its target server.

        The type check is deliberate belt-and-braces: the gateway already
        rejects unapproved input, but this adapter is reachable from wiring
        code, and an un-vetted call here would defeat the whole boundary.
        """
        if not isinstance(approved, ApprovedIntent):
            raise PolicyViolation(
                f"MCPToolInvoker requires an ApprovedIntent, got "
                f"{type(approved).__name__}"
            )

        return await self.manager.execute_tool(
            approved.tool,
            approved.server,
            **approved.arguments,
        )


__all__ = ["MCPToolInvoker"]
