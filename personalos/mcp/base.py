"""Base MCP Server implementation."""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from personalos.domain.models import (
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENCY_KEY_MIN_LENGTH,
    InvalidIdempotencyKey,
    Tool,
)
from personalos.persistence.idempotency import (
    IdempotencyError,
    IdempotencyGuard,
    OperationStore,
)

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_PARAM = "idempotency_key"


class ToolSchema:
    """Schema for MCP tool parameters and response."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        mutating: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required or []
        self.response_schema = response_schema or {}
        self.mutating = mutating

        if mutating:
            self._require_idempotency_key()

    def _require_idempotency_key(self):
        """Add the idempotency key to the tool's contract.

        A mutating tool cannot be called without one, so the key is part of the
        advertised schema rather than something callers have to know about.
        """
        properties = self.parameters.setdefault("properties", {})
        properties[IDEMPOTENCY_KEY_PARAM] = {
            "type": "string",
            "minLength": IDEMPOTENCY_KEY_MIN_LENGTH,
            "maxLength": IDEMPOTENCY_KEY_MAX_LENGTH,
            "description": (
                "Unique key identifying this operation. Reuse the same key when "
                "retrying so the side effect happens at most once."
            ),
        }
        if IDEMPOTENCY_KEY_PARAM not in self.required:
            self.required.append(IDEMPOTENCY_KEY_PARAM)

    def to_domain_tool(self, mcp_server: str = "base") -> Tool:
        """Convert to domain Tool model."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            mcp_server=mcp_server,
        )


class MCPServer(ABC):
    """Base class for MCP servers."""

    def __init__(
        self,
        name: str,
        description: str,
        operation_store: Optional[OperationStore] = None,
    ):
        """Initialize MCP server.

        `operation_store` records mutating operations by idempotency key. It is
        required before any mutating tool can execute — without it there is no
        way to deduplicate a retry, so such calls are rejected rather than
        risking a repeated side effect.
        """
        self.name = name
        self.description = description
        self._tools: Dict[str, ToolSchema] = {}
        self._handlers: Dict[str, callable] = {}
        self.operation_store = operation_store
        self._guard = IdempotencyGuard(operation_store) if operation_store else None

    def set_operation_store(self, store: OperationStore):
        """Attach (or replace) the operation store backing idempotency."""
        self.operation_store = store
        self._guard = IdempotencyGuard(store)

    def register_tool(self, schema: ToolSchema, handler: callable):
        """Register a tool with its handler."""
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler
        logger.info(f"Registered tool '{schema.name}' on {self.name}")

    def get_tools(self) -> List[Tool]:
        """Get all available tools."""
        return [tool.to_domain_tool(self.name) for tool in self._tools.values()]

    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        return self._tools.get(tool_name)

    async def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool."""
        if tool_name not in self._handlers:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        handler = self._handlers[tool_name]
        schema = self._tools[tool_name]

        # Validate required parameters
        missing_params = [p for p in schema.required if p not in kwargs]
        if missing_params:
            return {
                "success": False,
                "error": f"Missing required parameters: {missing_params}",
            }

        if schema.mutating:
            return await self._execute_mutating(tool_name, schema, handler, kwargs)

        logger.info(f"Executing tool '{tool_name}' on {self.name}")
        try:
            result = handler(**kwargs)
            # Awaits any awaitable rather than testing the handler itself, so
            # callables whose __call__ is async are handled too - returning an
            # un-awaited coroutine as a "result" would hand callers an
            # unusable object instead of the tool's output.
            if inspect.isawaitable(result):
                result = await result
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing '{tool_name}': {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _execute_mutating(
        self,
        tool_name: str,
        schema: ToolSchema,
        handler: callable,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a mutating tool behind the idempotency guard."""
        if self._guard is None:
            return {
                "success": False,
                "error": (
                    f"Tool '{tool_name}' is mutating but no operation store is "
                    f"configured on server '{self.name}'; refusing to execute an "
                    f"un-deduplicated side effect"
                ),
            }

        params = {k: v for k, v in kwargs.items() if k != IDEMPOTENCY_KEY_PARAM}
        idempotency_key = kwargs.get(IDEMPOTENCY_KEY_PARAM)

        logger.info(f"Executing mutating tool '{tool_name}' on {self.name}")
        try:
            result, replayed = await self._guard.run(
                operation=f"{self.name}.{tool_name}",
                idempotency_key=idempotency_key,
                params=params,
                handler=handler,
            )
        except (InvalidIdempotencyKey, IdempotencyError) as e:
            logger.warning(f"Idempotency check rejected '{tool_name}': {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error executing '{tool_name}': {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "result": result,
            "idempotency_key": idempotency_key,
            "replayed": replayed,
        }

    @abstractmethod
    def initialize(self):
        """Initialize the MCP server (register tools, load configs, etc.)."""
        pass
