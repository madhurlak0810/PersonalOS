"""MCP adapter for the tool boundary.

Bridges the :class:`~personalos.tools.gateway.ToolInvoker` port to the MCP
server manager. This is the only place in the codebase that turns an approved
intent back into loose keyword arguments, and it refuses to do so for anything
that has not been through policy.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from personalos.domain.errors import ErrorCode
from personalos.domain.models import (
    ActionTarget,
    ToolAuthorization,
    ToolCallErrorCode,
    ToolCallRequest,
)
from personalos.mcp.manager import MCPServerManager
from personalos.policy import ApprovedIntent, PolicyViolation

logger = logging.getLogger(__name__)

#: Translates the MCP-specific `ToolCallErrorCode` into the shared
#: `ErrorCode` taxonomy, so a failure reads the same whether it reached the
#: gateway via this adapter or via any other `ToolInvoker`.
_ERROR_CODE_MAP: dict[ToolCallErrorCode, ErrorCode] = {
    ToolCallErrorCode.SERVER_NOT_FOUND: ErrorCode.NOT_FOUND,
    ToolCallErrorCode.TOOL_NOT_FOUND: ErrorCode.NOT_FOUND,
    ToolCallErrorCode.VALIDATION_ERROR: ErrorCode.VALIDATION,
    ToolCallErrorCode.MISSING_OPERATION_STORE: ErrorCode.INTERNAL,
    ToolCallErrorCode.MISSING_AUTHORIZATION: ErrorCode.INTERNAL,
    ToolCallErrorCode.IDEMPOTENCY_CONFLICT: ErrorCode.IDEMPOTENCY_CONFLICT,
    ToolCallErrorCode.EXECUTION_ERROR: ErrorCode.TOOL_FAILURE,
}


#: Builds the authorization context for an approved intent, or returns
#: `None` when the call acts on no account. Supplied per server by the
#: composition root -- it is configuration, never something an intent says.
Authorizer = Callable[[ApprovedIntent], ToolAuthorization | None]


class MCPToolInvoker:
    """Executes approved intents against registered MCP servers."""

    def __init__(
        self,
        manager: MCPServerManager,
        authorizers: Mapping[str, Authorizer] | None = None,
    ):
        """Take the manager holding the registered servers.

        `authorizers` maps a server name to what attaches that server's
        `ToolAuthorization`. It is attached here, on the gateway side of the
        boundary and after policy, so it rides beside the call's arguments
        rather than among them.
        """
        self.manager = manager
        self.authorizers = dict(authorizers or {})

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
                f"MCPToolInvoker requires an ApprovedIntent, got {type(approved).__name__}"
            )

        authorizer = self.authorizers.get(approved.server)
        request = ToolCallRequest(
            target=ActionTarget(server=approved.server, tool=approved.tool),
            params=approved.arguments,
            context=approved.context,
            authorization=authorizer(approved) if authorizer else None,
        )
        result = await self.manager.execute_tool(request)

        error_code = (
            _ERROR_CODE_MAP.get(result.error.code, ErrorCode.INTERNAL).value
            if result.error
            else None
        )
        return {
            "success": result.ok,
            "result": result.result,
            "error": result.error.message if result.error else None,
            "error_code": error_code,
            "replayed": bool(result.replayed),
        }


__all__ = ["Authorizer", "MCPToolInvoker"]
