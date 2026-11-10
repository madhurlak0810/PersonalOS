"""Google MCP Server - Gmail and Calendar tools behind the tool gateway.

Registers every `GoogleToolSpec` from `mcp_servers.google.tools` as an MCP
tool on a server named `google`, so the policy layer sees them as
`google.<tool>` -- the references `personalos.policy.permissions` classifies
and `personalos.policy.rules` allowlists per node:

==========================  =================  ==============================
tool                        permission class   Google scope
==========================  =================  ==============================
``gmail_search``            READ_EXTERNAL      ``gmail.readonly``
``gmail_read_message``      READ_EXTERNAL      ``gmail.readonly``
``gmail_list_history``      READ_EXTERNAL      ``gmail.readonly``
``gmail_get_profile``       READ_EXTERNAL      ``gmail.readonly``
``gmail_list_labels``       READ_EXTERNAL      ``gmail.readonly``
``gmail_create_draft``      WRITE_REVERSIBLE   ``gmail.compose``
``gmail_send_message``      WRITE_EXTERNAL     ``gmail.send``
``calendar_list_events``    READ_EXTERNAL      ``calendar.readonly``
``calendar_create_event``   WRITE_EXTERNAL     ``calendar.events``
``calendar_update_event``   WRITE_EXTERNAL     ``calendar.events``
``calendar_delete_event``   DESTRUCTIVE        ``calendar.events``
==========================  =================  ==============================

Every tool `requires_authorization`: a call with no gateway-attached
`ToolAuthorization` is refused before the handler runs, and the writes run
behind the idempotency guard like every other mutating MCP tool.

The tool set is fixed when the server is built. Nothing a tool returns is
read back into it, so no message content can add, remove or rename a tool.

This server listens on nothing. It is either registered in-process on the
`MCPServerManager`, or served over stdio (`personalos.mcp.stdio`) to the
process that spawned it; the browser and the HTTP API never reach it.
"""

import logging

from mcp_servers.google.tools import GOOGLE_TOOL_SPECS, GoogleWorkspaceTools
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.persistence.idempotency import InMemoryOperationStore, OperationStore

logger = logging.getLogger(__name__)

GOOGLE_SERVER_NAME = "google"


class GoogleMCPServer(MCPServer):
    """MCP Server for the connected Google account's Gmail and Calendar."""

    def __init__(
        self,
        tools: GoogleWorkspaceTools,
        operation_store: OperationStore | None = None,
    ):
        """Wrap the in-process tool implementations.

        `tools` is required: it carries the credential broker and the HTTP
        client, both of which are deployment decisions made at the
        composition root.
        """
        super().__init__(
            GOOGLE_SERVER_NAME,
            "Search, read, draft and send Gmail; list and change Google Calendar events",
            operation_store=operation_store or InMemoryOperationStore(),
        )
        self.tools = tools
        self.initialize()

    def initialize(self):
        """Register the Gmail and Calendar tools."""
        for spec in GOOGLE_TOOL_SPECS:
            parameters = spec.params_type.model_json_schema()
            schema = ToolSchema(
                name=spec.name,
                description=spec.description,
                intent_type=spec.intent_type,
                parameters=parameters,
                required=list(parameters.get("required", [])),
                response_schema=spec.output_type.model_json_schema(),
                requires_authorization=True,
            )
            self.register_tool(schema, spec.bind(self.tools))


__all__ = ["GOOGLE_SERVER_NAME", "GoogleMCPServer"]
