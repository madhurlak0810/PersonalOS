"""Base MCP Server implementation."""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import ValidationError

from personalos.domain.models import (
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENCY_KEY_MIN_LENGTH,
    ActionTarget,
    Intent,
    InvalidIdempotencyKey,
    MutatingIntent,
    Tool,
    ToolCallErrorCode,
    ToolCallRequest,
    ToolCallResult,
)
from personalos.persistence.idempotency import (
    IdempotencyError,
    IdempotencyGuard,
    OperationStore,
)

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_PARAM = "idempotency_key"


class ToolSchema:
    """Schema for an MCP tool: its description, typed contract, and JSON schema.

    `intent_type` is the single source of truth for what a call to this tool
    must look like — `params` on a `ToolCallRequest` are validated into it
    before the handler runs. Whether the tool is mutating is derived from
    that type (a `MutatingIntent` subclass) rather than tracked separately,
    so the two can't drift apart.
    """

    def __init__(
        self,
        name: str,
        description: str,
        intent_type: Type[Intent],
        parameters: Optional[Dict[str, Any]] = None,
        required: List[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.intent_type = intent_type
        self.mutating = issubclass(intent_type, MutatingIntent)
        self.parameters = parameters or {}
        self.required = required or []
        self.response_schema = response_schema or {}

        if self.mutating:
            self._require_idempotency_key()

    def _require_idempotency_key(self):
        """Advertise the idempotency key in the tool's descriptive JSON schema.

        A mutating tool cannot be called without one, so the key is part of the
        advertised schema rather than something callers have to know about.
        The actual enforcement happens via `intent_type` (a `MutatingIntent`),
        this only keeps the advertised schema honest.
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
        self._handlers: Dict[str, Callable] = {}
        self.operation_store = operation_store
        self._guard = IdempotencyGuard(operation_store) if operation_store else None

    def set_operation_store(self, store: OperationStore):
        """Attach (or replace) the operation store backing idempotency."""
        self.operation_store = store
        self._guard = IdempotencyGuard(store)

    def register_tool(self, schema: ToolSchema, handler: Callable):
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

    async def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """Execute a tool call against its typed contract."""
        tool_name = request.target.tool

        if tool_name not in self._handlers:
            return ToolCallResult.failed(
                request.target,
                ToolCallErrorCode.TOOL_NOT_FOUND,
                f"Tool '{tool_name}' not found on server '{self.name}'",
            )

        schema = self._tools[tool_name]
        handler = self._handlers[tool_name]

        try:
            intent = schema.intent_type.model_validate(request.params)
        except ValidationError as e:
            return ToolCallResult.failed(
                request.target,
                ToolCallErrorCode.VALIDATION_ERROR,
                f"Invalid parameters for '{tool_name}': {e}",
                details={"errors": e.errors()},
            )

        if schema.mutating:
            return await self._execute_mutating(request.target, schema, handler, intent)

        logger.info(f"Executing tool '{tool_name}' on {self.name}")
        try:
            result = handler(**intent.model_dump())
            # Awaits any awaitable rather than testing the handler itself, so
            # callables whose __call__ is async are handled too - returning an
            # un-awaited coroutine as a "result" would hand callers an
            # unusable object instead of the tool's output.
            if inspect.isawaitable(result):
                result = await result
            return ToolCallResult.succeeded(request.target, result)
        except Exception as e:
            logger.error(f"Error executing '{tool_name}': {str(e)}", exc_info=True)
            return ToolCallResult.failed(
                request.target, ToolCallErrorCode.EXECUTION_ERROR, str(e)
            )

    async def _execute_mutating(
        self,
        target: ActionTarget,
        schema: ToolSchema,
        handler: Callable,
        intent: MutatingIntent,
    ) -> ToolCallResult:
        """Execute a mutating tool behind the idempotency guard."""
        if self._guard is None:
            return ToolCallResult.failed(
                target,
                ToolCallErrorCode.MISSING_OPERATION_STORE,
                f"Tool '{target.tool}' is mutating but no operation store is "
                f"configured on server '{self.name}'; refusing to execute an "
                f"un-deduplicated side effect",
            )

        params = intent.side_effect_params()
        idempotency_key = intent.idempotency_key

        logger.info(f"Executing mutating tool '{target.tool}' on {self.name}")
        try:
            result, replayed = await self._guard.run(
                operation=f"{self.name}.{target.tool}",
                idempotency_key=idempotency_key,
                params=params,
                handler=handler,
            )
        except InvalidIdempotencyKey as e:
            logger.warning(f"Idempotency check rejected '{target.tool}': {str(e)}")
            return ToolCallResult.failed(target, ToolCallErrorCode.VALIDATION_ERROR, str(e))
        except IdempotencyError as e:
            logger.warning(f"Idempotency check rejected '{target.tool}': {str(e)}")
            return ToolCallResult.failed(target, ToolCallErrorCode.IDEMPOTENCY_CONFLICT, str(e))
        except Exception as e:
            logger.error(f"Error executing '{target.tool}': {str(e)}", exc_info=True)
            return ToolCallResult.failed(target, ToolCallErrorCode.EXECUTION_ERROR, str(e))

        try:
            return ToolCallResult.succeeded(
                target, result, idempotency_key=idempotency_key, replayed=replayed
            )
        except Exception as e:
            logger.error(f"Malformed result from '{target.tool}': {str(e)}", exc_info=True)
            return ToolCallResult.failed(target, ToolCallErrorCode.EXECUTION_ERROR, str(e))

    @abstractmethod
    def initialize(self):
        """Initialize the MCP server (register tools, load configs, etc.)."""
        pass
