"""Tool registry for agent tools."""

import logging
from collections.abc import Callable

from personalos.domain.models import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self):
        """Initialize tool registry."""
        self._tools: dict[str, Tool] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, tool: Tool, handler: Callable):
        """Register a tool."""
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_handler(self, name: str) -> Callable | None:
        """Get a tool handler by name."""
        return self._handlers.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    async def execute(self, tool_name: str, **kwargs) -> dict:
        """Execute a tool."""
        handler = self.get_handler(tool_name)
        if not handler:
            raise ValueError(f"Tool {tool_name} not found")

        logger.info(f"Executing tool: {tool_name}")
        try:
            if hasattr(handler, "__await__"):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}


# Global tool registry instance
_tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _tool_registry
