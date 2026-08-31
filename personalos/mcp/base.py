"""Base MCP Server implementation."""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from personalos.domain.models import Tool

logger = logging.getLogger(__name__)


class ToolSchema:
    """Schema for MCP tool parameters and response."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required or []
        self.response_schema = response_schema or {}

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

    def __init__(self, name: str, description: str):
        """Initialize MCP server."""
        self.name = name
        self.description = description
        self._tools: Dict[str, ToolSchema] = {}
        self._handlers: Dict[str, callable] = {}

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

        logger.info(f"Executing tool '{tool_name}' on {self.name}")
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing '{tool_name}': {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    @abstractmethod
    def initialize(self):
        """Initialize the MCP server (register tools, load configs, etc.)."""
        pass
