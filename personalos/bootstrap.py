"""Composition root: the one module allowed to know about every layer.

Layers below cannot import each other across a boundary, so something has to
join them up. That job lives here (and in ``apps/``), which keeps the wiring
visible in a single file instead of hidden inside whichever layer happened to
need a collaborator.

See ``docs/ARCHITECTURE_BOUNDARIES.md``.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from uuid import UUID

from personalos.config import settings
from personalos.domain.credentials import CredentialExchangeFailed, CredentialRef
from personalos.domain.workflow import (
    WorkflowThread,
    job_search_thread_id,
    supervisor_thread_id,
)
from personalos.executor.artifact_prep import DocumentOverwriteExecutor, PolicyGatedDraftSink
from personalos.executor.calendar import (
    CalendarActionExecutor,
    CalendarClient,
    CalendarReconciler,
)
from personalos.executor.credentials import CredentialBroker
from personalos.executor.job_discovery import GatewayJobProvider
from personalos.executor.job_search import JobSearchExecutor
from personalos.executor.tool_executor import ProviderReconciler, ToolExecutor
from personalos.graphs.job_search import JobSearchSubgraphRunner
from personalos.mcp.adapter import MCPToolInvoker
from personalos.mcp.base import MCPServer
from personalos.mcp.manager import MCPServerManager, get_mcp_manager
from personalos.persistence.action_journal import ActionExecutorPort, JournaledActionExecutor
from personalos.persistence.application_lifecycle import ApplicationLifecycleStore
from personalos.persistence.artifact_drafts import SqlArtifactDraftStore
from personalos.persistence.checkpoint_conditions import SqlCheckpointConditionEvaluator
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.database import SessionLocal
from personalos.persistence.evidence import SqlEvidenceIndex, SqlEvidenceSource
from personalos.persistence.execution_ledger import ExecutionLedger
from personalos.persistence.idempotency import OperationStore, SqlOperationStore
from personalos.persistence.job_postings import SqlPostingCatalog
from personalos.persistence.leases import DEFAULT_LEASE_TTL_SECONDS, WorkflowLeaseStore
from personalos.persistence.pending_checkpoints import (
    PendingCheckpointStore,
    StorePendingCheckpointScheduler,
)
from personalos.persistence.policy_log import SqlPolicyDecisionLog
from personalos.persistence.recruiter_events import SqlRecruiterEventStore
from personalos.persistence.repositories import JobRepository
from personalos.policy import PolicyEngine, default_policy_engine
from personalos.providers import (
    JOB_PROVIDERS_SERVER,
    GreenhouseProvider,
    JobProvider,
    JobProviderInvoker,
)
from personalos.retrieval.artifact_prep import (
    DEFAULT_TOP_K,
    DraftWriter,
    Embedder,
    EvidenceSelector,
    TailoredPacketBuilder,
)
from personalos.retrieval.job_matching import HybridJobMatcher, ScoringConfig, SemanticAssessor
from personalos.secrets.exchange import (
    ApiKeyExchanger,
    GoogleOAuthTokenExchanger,
    TokenExchanger,
)
from personalos.secrets.store import KeyringSecretStore, SecretStore
from personalos.tools.gateway import (
    PolicyEnforcingToolGateway,
    RoutingToolInvoker,
    ToolGateway,
    ToolInvoker,
)

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
    if settings.mcp_files_enabled:
        manager.register_server(build_files_mcp_server(operation_store=operation_store))
    logger.info("Registered MCP servers: %s", manager.list_servers())
    return manager


def build_files_mcp_server(
    allowed_roots: Iterable[str] | None = None,
    operation_store: OperationStore | None = None,
) -> MCPServer:
    """Build the files server, confined to the configured allowed roots.

    The roots come from `FILES_ALLOWED_ROOTS` and from nowhere else: they are
    never an intent argument, so nothing a model emits can widen them. With
    none configured the server still registers and rejects every path.
    """
    from mcp_servers.files.sandbox import PathSandbox
    from mcp_servers.files.server import FilesMCPServer

    roots = settings.files_allowed_roots if allowed_roots is None else allowed_roots
    return FilesMCPServer(PathSandbox(roots), operation_store=operation_store)


def build_policy_engine(
    session_factory: Callable[[], object] = SessionLocal,
) -> PolicyEngine:
    """Build the default policy engine backed by the `policy_decisions` table.

    This is the engine a real deployment runs: every verdict is committed
    before the engine returns it, so no tool executes ahead of its decision
    row. `default_policy_engine()` on its own records nothing durable and is
    for tests and callers that have no database.
    """
    return default_policy_engine(decision_log=SqlPolicyDecisionLog(session_factory))


def build_tool_gateway(
    manager: MCPServerManager | None = None,
    policy: PolicyEngine | None = None,
    job_providers: Iterable[JobProvider] = (),
) -> ToolGateway:
    """Build the gateway every executor is handed.

    Defaults to the default-deny policy engine, so a caller that forgets to
    pass one gets the restrictive engine rather than an open door. Job
    providers, when given, are reachable as `job_providers.*` behind the same
    engine as every MCP tool.
    """
    invoker: ToolInvoker = MCPToolInvoker(manager or get_mcp_manager())
    providers = list(job_providers)
    if providers:
        invoker = RoutingToolInvoker(
            {JOB_PROVIDERS_SERVER: JobProviderInvoker(providers)}, default=invoker
        )
    return PolicyEnforcingToolGateway(policy=policy or default_policy_engine(), invoker=invoker)


def configured_job_providers() -> list[JobProvider]:
    """The real provider adapters this deployment has configured."""
    providers: list[JobProvider] = []
    if settings.greenhouse_boards:
        providers.append(GreenhouseProvider(settings.greenhouse_boards))
    return providers


def build_job_providers(
    providers: Iterable[JobProvider] | None = None,
    policy: PolicyEngine | None = None,
) -> list[GatewayJobProvider]:
    """Build the `providers` a `JobSearchGraph` is handed.

    Each is a gateway-backed stand-in for one adapter, so the graph holds no
    HTTP client and every provider call is a recorded `READ_EXTERNAL` decision.
    Pass `build_policy_engine()` to make those decisions durable.
    """
    adapters = configured_job_providers() if providers is None else list(providers)
    gateway = build_tool_gateway(policy=policy, job_providers=adapters)
    return [GatewayJobProvider(adapter.name, gateway) for adapter in adapters]


def build_posting_catalog(session_factory=SessionLocal) -> SqlPostingCatalog:
    """Bind `JobSearchGraph`'s `PostingCatalog` port to the `job_postings` table."""
    return SqlPostingCatalog(session_factory)


def build_job_matcher(
    assessor: SemanticAssessor,
    session_factory=SessionLocal,
    config: ScoringConfig | None = None,
) -> HybridJobMatcher:
    """Build the matcher a `JobSearchGraph` takes as both `scorer` and `evidence_checker`.

    `assessor` is required rather than defaulted so that which model reads
    posting text is chosen where the graph is assembled; pass
    `personalos.models.job_matching.anthropic_semantic_assessor()` for Claude.
    """
    return HybridJobMatcher(
        evidence_source=SqlEvidenceSource(session_factory), assessor=assessor, config=config
    )


def build_artifact_packet_builder(
    writer: DraftWriter,
    embedder: Embedder,
    session_factory=SessionLocal,
    *,
    policy: PolicyEngine | None = None,
    workflow_id: UUID | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = 0.0,
    context_terms: Iterable[str] = (),
) -> TailoredPacketBuilder:
    """Build the `packet_builder` a `JobSearchGraph` takes for tailored drafts.

    Evidence is selected from `evidence_chunks` by similarity to the posting,
    every draft is checked against that evidence before it is kept, and the
    survivors are stored as `artifact_versions` rows under a recorded
    `WRITE_REVERSIBLE` decision.

    `writer` and `embedder` are required rather than defaulted, as `assessor`
    is for `build_job_matcher`: pass
    `personalos.models.artifact_drafting.anthropic_draft_writer()` for Claude,
    and an embedder whose `model` is the one the evidence was ingested with.
    `context_terms` are names a draft may use without citing evidence -- the
    candidate's own name.
    """
    return TailoredPacketBuilder(
        selector=EvidenceSelector(
            embedder=embedder,
            index=SqlEvidenceIndex(session_factory),
            top_k=top_k,
            min_similarity=min_similarity,
        ),
        writer=writer,
        sink=PolicyGatedDraftSink(
            SqlArtifactDraftStore(session_factory),
            policy or build_policy_engine(session_factory),
            workflow_id=workflow_id,
        ),
        context_terms=tuple(context_terms),
    )


def build_document_overwrite_executor(
    gateway: ToolGateway | None = None,
    policy: PolicyEngine | None = None,
) -> DocumentOverwriteExecutor:
    """Build the adapter that carries out an approved `OVERWRITE_DOCUMENT` action.

    It dispatches `files.overwrite_document` through the gateway, so the files
    server's hash precondition and backup apply. Pass the result to
    `build_tool_executor` as `inner` -- that is what makes the overwrite
    at-most-once and audited against the approval it ran under.
    """
    return DocumentOverwriteExecutor(gateway or build_tool_gateway(policy=policy))


def build_job_search_executor(
    repo: JobRepository,
    gateway: ToolGateway | None = None,
    policy: PolicyEngine | None = None,
) -> JobSearchExecutor:
    """Build the job search executor with its policy-enforcing gateway."""
    return JobSearchExecutor(repo, gateway or build_tool_gateway(policy=policy))


# --- Credentials --------------------------------------------------------------
#
# Long-lived secrets are in the OS keychain; the database holds references to
# them. These builders are the only place the two are joined, and the broker
# they produce is handed to things that *act* -- an MCP server's tool handler,
# an action executor -- never to a graph or a model client.

#: Provider name of the Google OAuth exchanger, as it appears in a
#: `cred://google/<account>` reference. Gmail and Calendar share it.
GOOGLE_PROVIDER = "google"

#: Providers whose credential is a static API key rather than an OAuth grant.
API_KEY_PROVIDERS: tuple[str, ...] = ("greenhouse", "lever", "jobs")


def build_secret_store() -> SecretStore:
    """Build the OS-keychain secret store.

    Raises `SecretStoreUnavailable` when there is no protected backend. There
    is no fallback to an in-memory or file store here, on purpose: a
    deployment that cannot protect its refresh tokens should fail to start,
    not start with them somewhere else.
    """
    return KeyringSecretStore(settings.secret_store_service)


def build_credential_broker(
    store: SecretStore | None = None,
    exchangers: Mapping[str, TokenExchanger] | None = None,
) -> CredentialBroker:
    """Build the broker that exchanges credential references for access tokens.

    With no `exchangers` given, wires an API-key lease for each job provider
    and, when a Google OAuth client is configured, the Google exchanger. The
    OAuth client secret is itself read from the secret store, by reference.
    """
    store = store or build_secret_store()
    if exchangers is None:
        wired: dict[str, TokenExchanger] = {
            provider: ApiKeyExchanger() for provider in API_KEY_PROVIDERS
        }
        if settings.google_oauth_client_id:
            client_ref = CredentialRef.parse(settings.google_oauth_client_secret_ref)
            client_secret = store.get(client_ref)
            if client_secret is None:
                raise CredentialExchangeFailed(
                    f"GOOGLE_OAUTH_CLIENT_ID is set but no client secret is stored at {client_ref}"
                )
            wired[GOOGLE_PROVIDER] = GoogleOAuthTokenExchanger(
                settings.google_oauth_client_id, client_secret
            )
        exchangers = wired
    return CredentialBroker(store, exchangers)


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

    The graph's approval node decides *whether* to act; this decides whether
    the act has already happened -- see `personalos.persistence.action_journal`
    for why the two are separate. It journals and nothing more: a deployment
    binds `build_tool_executor` instead, which adds the policy decision, the
    audit trail and reconciliation on top of the same at-most-once rule.
    """
    return JournaledActionExecutor(inner, session_factory, workflow_id=workflow_id)


def build_tool_executor(
    inner: ActionExecutorPort,
    session_factory: Callable[[], object] = SessionLocal,
    *,
    workflow_id: UUID | None = None,
    policy: PolicyEngine | None = None,
    reconciler: ProviderReconciler | None = None,
) -> ToolExecutor:
    """Wrap a provider adapter in the executor every mutating action goes through.

    This is the `action_executor` a real deployment hands a `JobSearchGraph`.
    Each action is authorized by the policy engine, claimed under its
    idempotency key, executed at most once, and recorded in `tool_executions`
    and `audit_events` against the `policy_decisions` row that cleared it.

    `policy` defaults to the engine backed by `policy_decisions` on the same
    database, so the decision an audit row cites is a row that exists. Pass a
    `reconciler` for a provider that can be asked whether an action landed;
    without one, an action whose outcome was never recorded is never retried.
    """
    return ToolExecutor(
        inner,
        policy or build_policy_engine(session_factory),
        ExecutionLedger(session_factory, workflow_id=workflow_id),
        reconciler=reconciler,
    )


def build_calendar_action_executor(
    client: CalendarClient,
    session_factory: Callable[[], object] = SessionLocal,
    *,
    fallback: ActionExecutorPort | None = None,
    workflow_id: UUID | None = None,
    policy: PolicyEngine | None = None,
) -> ToolExecutor:
    """Build the `action_executor` for a graph that schedules interviews.

    Calendar creates and updates go to `client`; every other action kind goes
    to `fallback`. The pairing with `CalendarReconciler` is the point of
    building it here: a calendar write whose outcome was never recorded is
    looked up on the calendar -- by the idempotency key stamped on the event,
    or by event id -- before `ToolExecutor` is allowed to send it again.
    """
    return build_tool_executor(
        CalendarActionExecutor(client, fallback=fallback),
        session_factory,
        workflow_id=workflow_id,
        policy=policy,
        reconciler=CalendarReconciler(client),
    )


def build_checkpoint_condition_evaluator(
    session_factory: Callable[[], object] = SessionLocal,
) -> SqlCheckpointConditionEvaluator:
    """Build the evaluator `PendingCheckpointMonitor` re-asks conditions through.

    Answers from the rows the rest of the system has written since the wait
    was scheduled, which is what lets a follow-up cancel itself when the
    recruiter replies in the meantime.
    """
    return SqlCheckpointConditionEvaluator(session_factory)


def build_application_lifecycle_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> ApplicationLifecycleStore:
    """Build the store that moves applications along their lifecycle.

    Shared, like `build_pending_checkpoint_store`, by processes with no other
    connection: whatever acts on a recommended transition writes through it,
    and `apps.worker.stall_monitor` sweeps the same rows on its own schedule.
    """
    return ApplicationLifecycleStore(session_factory)


def build_recruiter_event_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> SqlRecruiterEventStore:
    """Build the store behind inbound recruiter mail.

    Pass the result to `JobSearchGraph` as both `application_directory=` and
    `recruiter_event_recorder=`, with an extractor from
    `personalos.models.recruiter_events` as `recruiter_event_extractor=`:
    `anthropic_recruiter_event_extractor()` for Claude, or
    `RuleBasedRecruiterEventExtractor()` to run on the deterministic rules alone.
    """
    return SqlRecruiterEventStore(session_factory)


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
    "build_files_mcp_server",
    "build_policy_engine",
    "build_tool_gateway",
    "configured_job_providers",
    "build_job_providers",
    "build_posting_catalog",
    "build_job_matcher",
    "build_artifact_packet_builder",
    "build_document_overwrite_executor",
    "build_job_search_executor",
    "GOOGLE_PROVIDER",
    "API_KEY_PROVIDERS",
    "build_secret_store",
    "build_credential_broker",
    "initialize_mcp_servers",
    "build_workflow_thread_registry",
    "build_durable_checkpointer",
    "build_workflow_lease_store",
    "build_journaled_action_executor",
    "build_tool_executor",
    "build_recruiter_event_store",
    "build_pending_checkpoint_store",
    "build_calendar_action_executor",
    "build_checkpoint_condition_evaluator",
    "build_pending_checkpoint_scheduler",
    "register_job_search_thread",
    "register_supervisor_thread",
    "build_job_search_subgraph_runner",
]
