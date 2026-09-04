"""Composition root: the one module allowed to know about every layer.

Layers below cannot import each other across a boundary, so something has to
join them up. That job lives here (and in ``apps/``), which keeps the wiring
visible in a single file instead of hidden inside whichever layer happened to
need a collaborator.

See ``docs/ARCHITECTURE_BOUNDARIES.md``.
"""

import logging

from personalos.executor.job_search import JobSearchExecutor
from personalos.mcp.adapter import MCPToolInvoker
from personalos.mcp.manager import MCPServerManager, get_mcp_manager
from personalos.persistence.database import SessionLocal
from personalos.persistence.idempotency import OperationStore, SqlOperationStore
from personalos.persistence.repositories import JobRepository
from personalos.policy import PolicyEngine, default_policy_engine
from personalos.tools.gateway import PolicyEnforcingToolGateway, ToolGateway

logger = logging.getLogger(__name__)


def build_operation_store(session_factory=SessionLocal) -> OperationStore:
    """Build the durable operation store backing idempotency."""
    return SqlOperationStore(session_factory)


def register_mcp_servers(
    manager: MCPServerManager | None = None,
    operation_store: OperationStore | None = None,
) -> MCPServerManager:
    """Register the MCP servers this deployment exposes.

    Importing server implementations is the composition root's job: the
    ``personalos.mcp`` layer defines how to talk to a server and must not know
    which ones exist.
    """
    from mcp_servers.jobs.server import JobsMCPServer

    manager = manager or get_mcp_manager()
    manager.register_server(JobsMCPServer(operation_store=operation_store))
    logger.info("Registered MCP servers: %s", manager.list_servers())
    return manager


def build_tool_gateway(
    manager: MCPServerManager | None = None,
    policy: PolicyEngine | None = None,
) -> ToolGateway:
    """Build the gateway every executor is handed.

    Defaults to the default-deny policy engine, so a caller that forgets to
    pass one gets the restrictive engine rather than an open door.
    """
    return PolicyEnforcingToolGateway(
        policy=policy or default_policy_engine(),
        invoker=MCPToolInvoker(manager or get_mcp_manager()),
    )


def build_job_search_executor(
    repo: JobRepository,
    gateway: ToolGateway | None = None,
    policy: PolicyEngine | None = None,
) -> JobSearchExecutor:
    """Build the job search executor with its policy-enforcing gateway."""
    return JobSearchExecutor(repo, gateway or build_tool_gateway(policy=policy))


def initialize_mcp_servers() -> MCPServerManager:
    """Startup hook: register the MCP servers on the global manager."""
    logger.info("Initializing MCP servers...")
    return register_mcp_servers()


__all__ = [
    "build_operation_store",
    "register_mcp_servers",
    "build_tool_gateway",
    "build_job_search_executor",
    "initialize_mcp_servers",
]
