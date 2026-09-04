"""MCP adapter for the tool boundary.

Bridges the :class:`~personalos.tools.gateway.ToolInvoker` port to the MCP
server manager. This is the only place in the codebase that turns an approved
intent back into loose keyword arguments, and it refuses to do so for anything
that has not been through policy.
"""

import logging
from typing import Any

from personalos.domain.models import ActionTarget, ToolCallRequest
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

        The manager speaks a typed `ToolCallRequest`/`ToolCallResult` contract,
        not loose kwargs; this method is the translation point between that
        and the `{success, result, error, replayed}` payload shape the gateway
        expects back.
        """
        if not isinstance(approved, ApprovedIntent):
            raise PolicyViolation(
                f"MCPToolInvoker requires an ApprovedIntent, got "
                f"{type(approved).__name__}"
            )

        request = ToolCallRequest(
            target=ActionTarget(server=approved.server, tool=approved.tool),
            params=approved.arguments,
        )
        result = await self.manager.execute_tool(request)

        return {
            "success": result.ok,
            "result": result.result,
            "error": result.error.message if result.error else None,
            "replayed": bool(result.replayed),
        }


__all__ = ["MCPToolInvoker"]
