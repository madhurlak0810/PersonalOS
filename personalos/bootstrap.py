"""Composition root: the one module allowed to know about every layer.

Layers below cannot import each other across a boundary, so something has to
join them up. That job lives here (and in ``apps/``), which keeps the wiring
visible in a single file instead of hidden inside whichever layer happened to
need a collaborator.

See ``docs/ARCHITECTURE_BOUNDARIES.md``.
"""

import logging
from collections.abc import Callable
from uuid import UUID

from personalos.domain.workflow import (
    WorkflowThread,
    job_search_thread_id,
    supervisor_thread_id,
)
from personalos.executor.job_search import JobSearchExecutor
from personalos.graphs.job_search import JobSearchSubgraphRunner
from personalos.mcp.adapter import MCPToolInvoker
from personalos.mcp.manager import MCPServerManager, get_mcp_manager
from personalos.persistence.action_journal import ActionExecutorPort, JournaledActionExecutor
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.database import SessionLocal
from personalos.persistence.idempotency import OperationStore, SqlOperationStore
from personalos.persistence.leases import DEFAULT_LEASE_TTL_SECONDS, WorkflowLeaseStore
from personalos.persistence.pending_checkpoints import (
    PendingCheckpointStore,
    StorePendingCheckpointScheduler,
)
from personalos.persistence.repositories import JobRepository
from personalos.policy import PolicyEngine, default_policy_engine
from personalos.tools.gateway import PolicyEnforcingToolGateway, ToolGateway

logger = logging.getLogger(__name__)

#: `workflows.name` for the job search process. One workflow *definition*; each
#: pursuit of it gets its own `workflow_id` row, which is what a resume leases.
JOB_SEARCH_WORKFLOW = "job_search"

#: `workflows.name` for a Supervisor conversation.
SUPERVISOR_WORKFLOW = "supervisor"


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


# --- Durable orchestration ----------------------------------------------------
#
# Graphs compile against a `BaseCheckpointSaver` and default to
# `InMemorySaver`, which is a test fixture: it loses every thread when the
# process ends. These builders are what a real deployment passes instead, and
# they are here rather than in `personalos.graphs` because the graphs layer may
# not import `persistence` at all.


def build_workflow_thread_registry(
    session_factory: Callable[[], object] = SessionLocal,
) -> WorkflowThreadRegistry:
    """Build the registry that binds thread ids to the workflow they belong to."""
    return WorkflowThreadRegistry(session_factory)


def build_durable_checkpointer(
    session_factory: Callable[[], object] = SessionLocal,
    registry: WorkflowThreadRegistry | None = None,
) -> SqlAlchemyCheckpointSaver:
    """Build the database-backed checkpointer both graphs should run on.

    Pass the result as `checkpointer=` to `SupervisorGraph` and
    `JobSearchGraph`. Sharing one saver between them is deliberate: they run on
    separate threads but belong to the same workflow, so one store keeps a
    workflow's whole state findable by `workflow_id`.
    """
    return SqlAlchemyCheckpointSaver(
        session_factory, registry or build_workflow_thread_registry(session_factory)
    )


def build_workflow_lease_store(
    session_factory: Callable[[], object] = SessionLocal,
    *,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> WorkflowLeaseStore:
    """Build the lease store that keeps two workers off the same workflow."""
    return WorkflowLeaseStore(session_factory, ttl_seconds=ttl_seconds)


def build_journaled_action_executor(
    inner: ActionExecutorPort,
    session_factory: Callable[[], object] = SessionLocal,
    *,
    workflow_id: UUID | None = None,
) -> JournaledActionExecutor:
    """Wrap an `ActionExecutor` so a crash mid-action cannot double-submit.

    Every `action_executor` handed to a `JobSearchGraph` in a real deployment
    goes through this. The graph's approval node decides *whether* to act; this
    decides whether the act has already happened -- see
    `personalos.persistence.action_journal` for why the two are separate.
    """
    return JournaledActionExecutor(inner, session_factory, workflow_id=workflow_id)


def build_pending_checkpoint_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> PendingCheckpointStore:
    """Build the store behind durable, conditional waits.

    Shared by both halves of the feature on purpose: the graph writes waits
    through it and `apps.worker.checkpoint_monitor` sweeps the same rows. They
    are separate processes with no other connection -- a wait is scheduled by a
    run that has long since ended by the time it comes due -- and this table is
    the entirety of what passes between them.
    """
    return PendingCheckpointStore(session_factory)


def build_pending_checkpoint_scheduler(
    store: PendingCheckpointStore | None = None,
    session_factory: Callable[[], object] = SessionLocal,
) -> StorePendingCheckpointScheduler:
    """Bind `JobSearchGraph`'s `PendingCheckpointScheduler` port to the store.

    Pass the result as `checkpoint_scheduler=` when constructing the graph.
    Leaving it unwired is a deployment choice, not an omission to be papered
    over here: without a monitor process to sweep them, scheduled waits would
    sit in the table reading as follow-ups that are coming and never come.
    """
    return StorePendingCheckpointScheduler(store or build_pending_checkpoint_store(session_factory))


def register_job_search_thread(
    *,
    user_id: UUID,
    registry: WorkflowThreadRegistry,
    workflow_id: UUID | None = None,
    search_key: str | None = None,
    actor_id: str = "system",
    correlation_id: UUID | None = None,
) -> WorkflowThread:
    """Derive and register the stable thread a job search runs on.

    Deriving and registering belong together: a derived id that was never bound
    to a workflow run cannot be checkpointed (the durable saver refuses it), and
    a registered id that was not derived cannot be recomputed after a restart.
    Doing both in one call is what makes "resume this candidate's job search"
    answerable from nothing but the candidate.

    `search_key` distinguishes concurrent searches for the same candidate; omit
    it and the candidate has one long-running job search, which is the shape a
    single-user build actually has.
    """
    thread_id = job_search_thread_id(user_id, search_key)
    return registry.register(
        thread_id=thread_id,
        workflow_name=JOB_SEARCH_WORKFLOW,
        workflow_id=workflow_id,
        user_id=user_id,
        actor_id=actor_id,
        correlation_id=correlation_id,
    )


def register_supervisor_thread(
    *,
    conversation_key: str,
    registry: WorkflowThreadRegistry,
    workflow_id: UUID | None = None,
    user_id: UUID | None = None,
    actor_id: str = "system",
    correlation_id: UUID | None = None,
) -> WorkflowThread:
    """Derive and register the stable thread a Supervisor conversation runs on.

    Passing the job search's `workflow_id` here is how a conversation and the
    domain run it delegated to end up as one resumable business process with two
    threads, rather than two unrelated ones.
    """
    thread_id = supervisor_thread_id(conversation_key)
    return registry.register(
        thread_id=thread_id,
        workflow_name=SUPERVISOR_WORKFLOW,
        workflow_id=workflow_id,
        user_id=user_id,
        actor_id=actor_id,
        correlation_id=correlation_id,
    )


def build_job_search_subgraph_runner(
    compiled_subgraph,
    *,
    user_id: UUID,
    thread: WorkflowThread,
    prepare_application: bool = False,
) -> JobSearchSubgraphRunner:
    """Bind the Supervisor's `JobSubgraphRunner` port to a registered thread."""
    return JobSearchSubgraphRunner(
        compiled_subgraph,
        user_id=user_id,
        thread_id=thread.thread_id,
        workflow_id=thread.workflow_id,
        prepare_application=prepare_application,
    )


__all__ = [
    "JOB_SEARCH_WORKFLOW",
    "SUPERVISOR_WORKFLOW",
    "build_operation_store",
    "register_mcp_servers",
    "build_tool_gateway",
    "build_job_search_executor",
    "initialize_mcp_servers",
    "build_workflow_thread_registry",
    "build_durable_checkpointer",
    "build_workflow_lease_store",
    "build_journaled_action_executor",
    "build_pending_checkpoint_store",
    "build_pending_checkpoint_scheduler",
    "register_job_search_thread",
    "register_supervisor_thread",
    "build_job_search_subgraph_runner",
]
