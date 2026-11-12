"""Composition root: the one module allowed to know about every layer.

Layers below cannot import each other across a boundary, so something has to
join them up. That job lives here (and in ``apps/``), which keeps the wiring
visible in a single file instead of hidden inside whichever layer happened to
need a collaborator.

See ``docs/ARCHITECTURE_BOUNDARIES.md``.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import timedelta
from uuid import UUID

from personalos.config import settings
from personalos.domain.credentials import (
    CredentialExchangeFailed,
    CredentialReconnectRequired,
    CredentialRef,
    SecretValue,
)
from personalos.domain.errors import PortNotConfigured
from personalos.domain.events import EventEnvelope
from personalos.domain.job_search import ActionKind
from personalos.domain.mailbox import MailboxScope
from personalos.domain.models import ToolAuthorization
from personalos.domain.outbound_email import EMAIL_ACTION_KINDS
from personalos.domain.workflow import (
    WorkflowThread,
    application_transition_thread_id,
    event_handler_thread_id,
    job_search_thread_id,
    supervisor_thread_id,
)
from personalos.executor.artifact_prep import (
    DocumentOverwriteExecutor,
    PolicyGatedDraftSink,
    PolicyGatedReplyDraftStore,
)
from personalos.executor.calendar import (
    CALENDAR_ACTION_KINDS,
    CalendarActionExecutor,
    CalendarClient,
    CalendarReconciler,
)
from personalos.executor.credentials import CredentialBroker
from personalos.executor.document_reader import GatewayDocumentReader
from personalos.executor.gmail_inbox import GmailRecruiterInbox
from personalos.executor.gmail_outbound import GmailOutbound, GmailSendExecutor, GmailSentReconciler
from personalos.executor.google_calendar import CalendarChangeScanner, GoogleCalendar
from personalos.executor.google_connect import GoogleConnectionReader, GoogleConnector
from personalos.executor.job_discovery import GatewayJobProvider
from personalos.executor.tool_executor import (
    KindRoutingReconciler,
    ProviderReconciler,
    ToolExecutor,
)
from personalos.graphs.job_search import JobSearchSubgraphRunner
from personalos.mcp.adapter import Authorizer, MCPToolInvoker
from personalos.mcp.base import MCPServer
from personalos.mcp.manager import MCPServerManager, get_mcp_manager
from personalos.models.embedding import (
    EMBEDDING_PROVIDERS,
    EmbeddingsClientFactory,
    EmbeddingSpec,
)
from personalos.models.gateway import ModelConfig, ModelGateway
from personalos.persistence.action_journal import ActionExecutorPort, JournaledActionExecutor
from personalos.persistence.application_board import ApplicationBoardStore
from personalos.persistence.application_lifecycle import (
    ApplicationLifecycleStore,
    StoreApplicationTransitioner,
)
from personalos.persistence.approvals import ApprovalQueue
from personalos.persistence.artifact_drafts import SqlArtifactDraftStore
from personalos.persistence.audit_trail import AuditTrailStore
from personalos.persistence.checkpoint_conditions import SqlCheckpointConditionEvaluator
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.database import SessionLocal
from personalos.persistence.event_correlation import CorrelatedWaitLookup
from personalos.persistence.event_failures import EventFailureStore
from personalos.persistence.event_inbox import IdempotentConsumer
from personalos.persistence.event_outbox import EventOutbox
from personalos.persistence.evidence import (
    SqlEvidenceIndex,
    SqlEvidenceIngestionStore,
    SqlEvidenceSource,
)
from personalos.persistence.execution_ledger import ExecutionLedger
from personalos.persistence.idempotency import OperationStore, SqlOperationStore
from personalos.persistence.integration_health import IntegrationHealthStore
from personalos.persistence.job_postings import SqlPostingCatalog
from personalos.persistence.leases import DEFAULT_LEASE_TTL_SECONDS, WorkflowLeaseStore
from personalos.persistence.mailbox_cursors import SqlMailboxCursorStore
from personalos.persistence.outbox_emitter import OutboxEventEmitter
from personalos.persistence.pending_checkpoints import (
    PendingCheckpointStore,
    StorePendingCheckpointScheduler,
)
from personalos.persistence.policy_log import SqlPolicyDecisionLog
from personalos.persistence.recruiter_events import SqlRecruiterEventStore
from personalos.persistence.reembedding import SqlReembedStore
from personalos.persistence.reply_drafts import SqlReplyApplicationLookup, SqlReplyDraftStore
from personalos.persistence.search_profiles import SqlSearchProfileStore
from personalos.persistence.workflow_actions import WorkflowActionReader
from personalos.persistence.workflow_commands import WORKFLOW_COMMAND_EVENT, WorkflowCommandQueue
from personalos.persistence.workflow_failures import WorkflowFailureStore
from personalos.persistence.workflow_progress import WorkflowProgressLog
from personalos.persistence.workflow_status import WorkflowStatusReader
from personalos.policy import ApprovedIntent, PolicyEngine, default_policy_engine
from personalos.providers import (
    JOB_PROVIDERS_SERVER,
    AshbyProvider,
    GreenhouseProvider,
    JobProvider,
    JobProviderInvoker,
    LeverProvider,
    WorkdayProvider,
)
from personalos.retrieval.artifact_prep import (
    DEFAULT_TOP_K,
    DraftWriter,
    Embedder,
    EvidenceSelector,
    TailoredPacketBuilder,
    retrieval_query,
)
from personalos.retrieval.evidence_ingestion import ResumeIngestor
from personalos.retrieval.job_matching import HybridJobMatcher, ScoringConfig, SemanticAssessor
from personalos.retrieval.reembedding import Reembedder
from personalos.retrieval.reply_drafting import (
    EvidenceReplyGrounding,
    GroundedReplyDrafter,
    ReplyWriter,
)
from personalos.secrets.api_token import (
    FileApiTokenStore,
    KeychainApiTokenStore,
    LocalApiToken,
    default_token_path,
)
from personalos.secrets.exchange import (
    ApiKeyExchanger,
    GoogleOAuthTokenExchanger,
    TokenExchanger,
)
from personalos.secrets.google_oauth import GoogleOAuthClient
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

#: `workflows.name` for the runs started by events no waiting workflow claimed.
EVENT_HANDLER_WORKFLOW = "event_handler"


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
    manager = manager or get_mcp_manager()
    if settings.mcp_files_enabled:
        manager.register_server(build_files_mcp_server(operation_store=operation_store))
    if settings.mcp_google_enabled:
        manager.register_server(build_google_mcp_server(operation_store=operation_store))
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
    invoker: ToolInvoker = MCPToolInvoker(
        manager or get_mcp_manager(), authorizers={GOOGLE_PROVIDER: google_authorizer()}
    )
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
    if settings.lever_sites:
        providers.append(LeverProvider(settings.lever_sites))
    if settings.ashby_boards:
        providers.append(AshbyProvider(settings.ashby_boards))
    if settings.workday_sites:
        providers.append(WorkdayProvider(settings.workday_sites))
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


def build_search_profile_store(session_factory=SessionLocal) -> SqlSearchProfileStore:
    """Bind `JobSearchGraph`'s `SearchProfileStore` port to `candidate_profiles`."""
    return SqlSearchProfileStore(session_factory)


def build_event_emitter(session_factory=SessionLocal) -> OutboxEventEmitter:
    """Bind `JobSearchGraph`'s `EventEmitter` port to the outbox."""
    return OutboxEventEmitter(session_factory)


def build_model_gateway(
    config_path: str | None = None,
    *,
    embeddings_client_factory: EmbeddingsClientFactory | None = None,
) -> ModelGateway:
    """Build the `ModelGateway` every model-backed port is taken from.

    `config_path` (default: `settings.llm_gateway_config`) names a
    `ModelConfig` JSON file such as those in `evals/model_benchmarks/configs/`;
    with none set, every role runs on the deterministic `rules` provider.
    Changing models is changing this file, which is also what
    `python -m evals.harness --config` measures before the change ships.

    The gateway allows a hosted embedder only if `CLOUD_INFERENCE_ALLOWED` is
    set. `embeddings_client_factory` is the seam a test uses to stand in for
    the embedding server.
    """
    path = config_path if config_path is not None else settings.llm_gateway_config
    return ModelGateway(
        ModelConfig.load(path) if path else ModelConfig(name="rules"),
        embeddings_client_factory=embeddings_client_factory,
        cloud_inference_allowed=settings.cloud_inference_allowed,
    )


def build_job_matcher(
    assessor: SemanticAssessor,
    session_factory=SessionLocal,
    config: ScoringConfig | None = None,
) -> HybridJobMatcher:
    """Build the matcher a `JobSearchGraph` takes as both `scorer` and `evidence_checker`.

    `assessor` is required rather than defaulted so that which model reads
    posting text is chosen where the graph is assembled; pass
    `build_model_gateway().semantic_assessor()` for the configured model.
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


def embedding_spec_from_settings(
    provider: str | None = None,
    *,
    model: str | None = None,
    version: str | None = None,
    base_url: str | None = None,
) -> EmbeddingSpec | None:
    """The `EmbeddingSpec` the `EMBEDDING_*` settings name, with any argument overriding its setting.

    None when the provider is empty. Raises `ValueError` for a provider this
    build does not have, so a typo stops startup instead of leaving retrieval
    silently unconfigured.
    """
    provider = settings.embedding_provider if provider is None else provider
    name = provider.strip().lower()
    if not name:
        return None
    if name not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"unknown EMBEDDING_PROVIDER {provider!r}; this build has: "
            f"{', '.join(EMBEDDING_PROVIDERS)}"
        )
    return EmbeddingSpec(
        provider=name,
        model=(model if model is not None else settings.embedding_model) or None,
        version=(version if version is not None else settings.embedding_model_version) or None,
        base_url=(base_url if base_url is not None else settings.embedding_base_url) or None,
        api_key_env=settings.embedding_api_key_env or None,
    )


def build_embedder(
    provider: str | None = None,
    *,
    model: str | None = None,
    version: str | None = None,
    base_url: str | None = None,
    gateway: ModelGateway | None = None,
) -> Embedder | None:
    """The configured embedder, chosen by the `ModelGateway`. None if none is configured.

    Ingestion and artifact prep must share it: retrieval only compares
    vectors from the model the evidence was ingested with. An `embedding`
    block in the model configuration file wins. Otherwise the `EMBEDDING_*`
    settings are used, and any argument given overrides its setting.

    Raises `ValueError` for an unknown provider, and
    `personalos.models.embedding.CloudInferenceNotAllowed` for a hosted one
    while `CLOUD_INFERENCE_ALLOWED` is false.
    """
    gateway = gateway or build_model_gateway()
    overridden = any(value is not None for value in (provider, model, version, base_url))
    if gateway.config.embedding is not None and not overridden:
        return gateway.embedder()
    spec = embedding_spec_from_settings(provider, model=model, version=version, base_url=base_url)
    return gateway.embedder(spec) if spec is not None else None


def build_reembedder(
    session_factory=SessionLocal,
    *,
    embedder: Embedder | None = None,
    batch_size: int | None = None,
) -> Reembedder:
    """Build what `reembed` runs: the backfill onto `embedder` (default: the configured one)."""
    embedder = embedder or build_embedder()
    if embedder is None:
        raise PortNotConfigured(
            "re-embedding needs an embedder; set EMBEDDING_PROVIDER or pass --provider",
            port="embedder",
        )
    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    store = SqlReembedStore(session_factory, posting_text=retrieval_query)
    return Reembedder(embedder, store, **kwargs)


def build_resume_ingestor(
    session_factory=SessionLocal,
    *,
    gateway: ToolGateway | None = None,
    embedder: Embedder | None = None,
    requested_by: str = "evidence_ingestion",
) -> ResumeIngestor:
    """Build what `ingest-resume` and `POST /v1/evidence/documents` run.

    Files are read as `files.read_file` through `gateway` -- by default one
    over the global MCP manager with the durable policy engine -- so the files
    server's allowed roots (`FILES_ALLOWED_ROOTS`) bound what can be ingested.
    The embedder defaults to `build_embedder()`; with none configured this
    raises `PortNotConfigured` rather than ingest evidence nothing can search.
    """
    embedder = embedder or build_embedder()
    if embedder is None:
        raise PortNotConfigured(
            "resume ingestion needs an embedder; set EMBEDDING_PROVIDER (e.g. 'ollama')",
            port="embedder",
        )
    gateway = gateway or build_tool_gateway(policy=build_policy_engine(session_factory))
    return ResumeIngestor(
        reader=GatewayDocumentReader(gateway, requested_by=requested_by),
        embedder=embedder,
        store=SqlEvidenceIngestionStore(session_factory),
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


def build_local_api_token() -> LocalApiToken:
    """The local API token, in the store `API_TOKEN_STORE` names.

    `file` (the default) works on a headless machine with no keychain daemon;
    `keychain` refuses to run without a protected backend, like every other
    secret.
    """
    kind = settings.api_token_store.strip().lower()
    if kind == "keychain":
        return LocalApiToken(KeychainApiTokenStore(build_secret_store()))
    if kind != "file":
        raise ValueError(
            f"API_TOKEN_STORE must be 'file' or 'keychain', not {settings.api_token_store!r}"
        )
    path = settings.api_token_file.strip() or default_token_path()
    return LocalApiToken(FileApiTokenStore(path))


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


def google_oauth_client_secret_ref() -> CredentialRef:
    """Where this deployment's Google OAuth client secret is filed."""
    return CredentialRef.parse(settings.google_oauth_client_secret_ref)


def store_google_oauth_client_secret(
    secret: SecretValue, store: SecretStore | None = None
) -> CredentialRef:
    """File the Google OAuth client secret in the keychain; return its reference."""
    ref = google_oauth_client_secret_ref()
    (store or build_secret_store()).put(ref, secret)
    return ref


def build_google_oauth_client(store: SecretStore | None = None) -> GoogleOAuthClient:
    """Build the client for Google's consent, token and revocation endpoints.

    Raises `PortNotConfigured` until `GOOGLE_OAUTH_CLIENT_ID` is set and the
    client secret is in the keychain (`personalos.cli auth google-client`).
    """
    if not settings.google_oauth_client_id:
        raise PortNotConfigured(
            "GOOGLE_OAUTH_CLIENT_ID is not set; create a Desktop OAuth client and run "
            "`python -m personalos.cli auth google-client`",
            port="google_oauth_client",
        )
    ref = google_oauth_client_secret_ref()
    client_secret = (store or build_secret_store()).get(ref)
    if client_secret is None:
        raise PortNotConfigured(
            f"no Google OAuth client secret is stored at {ref}; run "
            "`python -m personalos.cli auth google-client`",
            port="google_oauth_client",
        )
    return GoogleOAuthClient(settings.google_oauth_client_id, client_secret)


def build_google_connector(
    session_factory: Callable[[], object] = SessionLocal,
    *,
    store: SecretStore | None = None,
    oauth: GoogleOAuthClient | None = None,
) -> GoogleConnector:
    """Build what `auth google` and `/v1/integrations/google/*` run the consent flow with."""
    store = store or build_secret_store()
    return GoogleConnector(
        oauth or build_google_oauth_client(store),
        store,
        session_factory,
        build_integration_health_store(session_factory),
        testing_app=settings.google_oauth_publishing_status.lower() != "production",
    )


def build_google_connection_reader(
    session_factory: Callable[[], object] = SessionLocal,
) -> GoogleConnectionReader:
    """Build the reader `auth google --status` and `GET /v1/integrations` use."""
    return GoogleConnectionReader(session_factory, build_integration_health_store(session_factory))


# --- Google MCP server --------------------------------------------------------


def build_google_mcp_server(
    broker: CredentialBroker | None = None,
    http=None,
    operation_store: OperationStore | None = None,
) -> MCPServer:
    """Build the Gmail and Calendar server.

    It holds the credential broker as its `AccessTokenProvider` port and
    exchanges a reference for a short-lived token on each call; the refresh
    token stays in the keychain, read only by the broker.
    """
    from mcp_servers.google.api import HttpxGoogleHttp
    from mcp_servers.google.server import GoogleMCPServer
    from mcp_servers.google.tools import GoogleWorkspaceTools

    tools = GoogleWorkspaceTools(broker or build_credential_broker(), http or HttpxGoogleHttp())
    return GoogleMCPServer(tools, operation_store=operation_store)


def connected_google_credential(
    session_factory: Callable[[], object] = SessionLocal,
    account: str | None = None,
) -> CredentialRef:
    """The `cred://google/...` reference the Google tools act on.

    `account` (default `GOOGLE_ACCOUNT`) picks one connected account by
    address; otherwise the most recently connected active one. Raises
    `CredentialReconnectRequired` when there is none.
    """
    account = settings.google_account if account is None else account
    connections = build_google_connection_reader(session_factory).connections()
    for connection in connections:
        if not account or connection.account.lower() == account.lower():
            return CredentialRef.parse(connection.credential_ref)
    wanted = f" for {account}" if account else ""
    raise CredentialReconnectRequired(
        f"no Google account is connected{wanted}; run `python -m personalos.cli auth google`",
        provider=GOOGLE_PROVIDER,
    )


def google_authorizer(
    credential: Callable[[], CredentialRef] | CredentialRef | str | None = None,
) -> Authorizer:
    """What `MCPToolInvoker` attaches to every `google.*` call, after policy.

    The credential is deployment configuration -- the connected account --
    resolved here at call time, never an intent argument. `principal` is the
    node the call was authorized for, and the decision row is cited.
    """

    def resolve() -> CredentialRef:
        if credential is None:
            return connected_google_credential()
        if callable(credential):
            return credential()
        return CredentialRef.parse(credential)

    def authorize(approved: ApprovedIntent) -> ToolAuthorization:
        return ToolAuthorization(
            credential_ref=str(resolve()),
            principal=approved.intent.requested_by,
            policy_decision_id=approved.decision.record_id,
        )

    return authorize


def serve_google_mcp_stdio() -> None:
    """Serve the Google MCP server on stdin/stdout (`personalos.cli mcp-google`).

    Logging goes to stderr first: stdout is the protocol.
    """
    import asyncio
    import sys

    from personalos.mcp.stdio import serve_stdio

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "stream", None) is sys.stdout:
            root.removeHandler(handler)
    if not any(getattr(h, "stream", None) is sys.stderr for h in root.handlers):
        root.addHandler(logging.StreamHandler(sys.stderr))
    asyncio.run(serve_stdio(build_google_mcp_server(operation_store=build_operation_store())))


async def connect_google_mcp_stdio(
    manager: MCPServerManager | None = None,
    command: Iterable[str] | None = None,
) -> MCPServer:
    """Spawn the Google server as a child over stdio and register its proxy.

    The alternative to registering it in-process: the same tools, the same
    gateway and policy, a process boundary between them and the caller.
    """
    import sys

    from personalos.mcp.stdio import MCPStdioClient, RemoteMCPServer

    client = await MCPStdioClient.spawn(
        list(command or (sys.executable, "-m", "personalos.cli", "mcp-google"))
    )
    remote = await RemoteMCPServer.connect(GOOGLE_PROVIDER, client)
    (manager or get_mcp_manager()).register_server(remote)
    return remote


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


def build_workflow_command_queue(
    session_factory: Callable[[], object] = SessionLocal,
    *,
    reclaim_after: timedelta | None = None,
) -> WorkflowCommandQueue:
    """Build the queue the API hands runs to and the worker claims them from.

    `reclaim_after` matters only to the worker, the one process that claims:
    see `build_worker_command_queue`.
    """
    return WorkflowCommandQueue(session_factory, reclaim_after=reclaim_after)


def build_worker_command_queue(
    session_factory: Callable[[], object] = SessionLocal,
    *,
    lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> WorkflowCommandQueue:
    """Build the queue a worker claims from, reclaiming what a dead worker left claimed.

    A claim is reclaimable once it is as old as the workflow lease's TTL, so a
    command is only handed on after the worker that claimed it has stopped
    renewing -- or never renewed -- the lease that let it run.
    """
    return build_workflow_command_queue(
        session_factory, reclaim_after=timedelta(seconds=lease_ttl_seconds)
    )


def build_workflow_status_reader(
    session_factory: Callable[[], object] = SessionLocal,
    registry: WorkflowThreadRegistry | None = None,
) -> WorkflowStatusReader:
    """Build the reader `GET /v1/workflows/{id}` answers from.

    Reads checkpoints through the same saver the graphs write with, so the
    status an API process reports is decoded exactly as a resuming worker
    would decode it -- without the API process ever compiling a graph.
    """
    registry = registry or build_workflow_thread_registry(session_factory)
    return WorkflowStatusReader(
        session_factory,
        build_durable_checkpointer(session_factory, registry),
        build_workflow_command_queue(session_factory),
    )


def build_workflow_action_reader(
    session_factory: Callable[[], object] = SessionLocal,
    registry: WorkflowThreadRegistry | None = None,
) -> WorkflowActionReader:
    """Build the reader that labels a workflow's actions RECOMMENDED ... EXECUTED.

    Decodes checkpoints through the same saver as the status reader, so an
    action's status is read from exactly the state a resuming worker sees.
    """
    registry = registry or build_workflow_thread_registry(session_factory)
    return WorkflowActionReader(
        session_factory, build_durable_checkpointer(session_factory, registry)
    )


def build_approval_queue(
    session_factory: Callable[[], object] = SessionLocal,
    registry: WorkflowThreadRegistry | None = None,
) -> ApprovalQueue:
    """Build the queue `GET /v1/approvals` lists and a decision is recorded through.

    Shares the status reader's view of what is parked and the command queue's
    resume path, so a decision made here resumes a workflow exactly as
    `POST /v1/workflows/{id}/resume` would.
    """
    registry = registry or build_workflow_thread_registry(session_factory)
    return ApprovalQueue(
        session_factory,
        build_workflow_status_reader(session_factory, registry),
        registry,
        build_workflow_command_queue(session_factory),
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
    reconcilers: Mapping[ActionKind, ProviderReconciler] | None = None,
) -> ToolExecutor:
    """Build the `action_executor` for a graph that schedules interviews.

    Calendar creates and updates go to `client`; every other action kind goes
    to `fallback`. The pairing with `CalendarReconciler` is the point of
    building it here: a calendar write whose outcome was never recorded is
    looked up on the calendar -- by the idempotency key stamped on the event,
    or by event id -- before `ToolExecutor` is allowed to send it again.
    `reconcilers` adds the same for the kinds `fallback` carries out (the
    Gmail Sent-folder lookup for sends, `build_gmail_send_reconcilers`).
    """
    calendar = CalendarReconciler(client)
    routes: dict[ActionKind, ProviderReconciler] = dict.fromkeys(CALENDAR_ACTION_KINDS, calendar)
    routes.update(reconcilers or {})
    return build_tool_executor(
        CalendarActionExecutor(client, fallback=fallback, workflow_id=workflow_id),
        session_factory,
        workflow_id=workflow_id,
        policy=policy,
        reconciler=KindRoutingReconciler(routes),
    )


def build_gmail_outbound(
    gateway: ToolGateway | None = None, *, policy: PolicyEngine | None = None
) -> GmailOutbound:
    """Build the Gmail drafts-and-sends client, over the `google.gmail_*` tools.

    Raises `PortNotConfigured` while the Google server is off.
    """
    if gateway is None:
        if not settings.mcp_google_enabled:
            raise PortNotConfigured(
                "MCP_GOOGLE_ENABLED is off; connect Google (`personalos.cli auth google "
                "--enable gmail.send --enable gmail.compose`) and enable it",
                port="mailbox",
            )
        gateway = build_tool_gateway(policy=policy or build_policy_engine())
    return GmailOutbound(gateway)


def build_gmail_send_executor(
    client: GmailOutbound, *, fallback: ActionExecutorPort | None = None
) -> GmailSendExecutor:
    """The provider adapter for `SEND_EMAIL` and `SEND_RECRUITER_MESSAGE` actions.

    Wrap it in a `ToolExecutor` with `build_gmail_send_reconcilers(client)`: a
    send whose outcome was never recorded is then looked up in the Sent folder
    before anything is sent again.
    """
    return GmailSendExecutor(client, fallback=fallback)


def build_gmail_send_reconcilers(client: GmailOutbound) -> dict[ActionKind, ProviderReconciler]:
    """`GmailSentReconciler` for every email action kind."""
    return dict.fromkeys(EMAIL_ACTION_KINDS, GmailSentReconciler(client))


def build_reply_drafter(
    writer: ReplyWriter,
    session_factory: Callable[[], object] = SessionLocal,
    *,
    embedder: Embedder | None = None,
    mailbox: GmailOutbound | None = None,
    policy: PolicyEngine | None = None,
    workflow_id: UUID | None = None,
    context_terms: Iterable[str] = (),
) -> GroundedReplyDrafter:
    """Build the `reply_drafter` a `JobSearchGraph` takes.

    `writer` is `build_model_gateway().reply_writer()`. Replies are grounded in
    the candidate's evidence (nearest the message by `embedder`, when given),
    checked against it, stored as `recruiter_reply` artifact versions under a
    recorded `WRITE_REVERSIBLE` decision, and -- with a `mailbox` -- saved as
    Gmail drafts. Nothing it builds can send.
    """
    return GroundedReplyDrafter(
        grounding=EvidenceReplyGrounding(
            SqlReplyApplicationLookup(session_factory),
            embedder=embedder,
            index=SqlEvidenceIndex(session_factory) if embedder is not None else None,
            evidence_source=SqlEvidenceSource(session_factory),
        ),
        writer=writer,
        store=PolicyGatedReplyDraftStore(
            SqlReplyDraftStore(session_factory),
            policy or build_policy_engine(session_factory),
            workflow_id=workflow_id,
        ),
        mailbox=mailbox,
        context_terms=tuple(context_terms),
    )


def build_google_calendar(
    gateway: ToolGateway | None = None,
    *,
    policy: PolicyEngine | None = None,
    time_zone: str | None = None,
    calendar_id: str | None = None,
) -> GoogleCalendar:
    """Build the calendar the Job Search graph reads and the calendar executor writes.

    One object serves both ports (`calendar_reader=` and the `client` of
    `build_calendar_action_executor`), so a plan made after a write sees it.
    Every call goes through `gateway` -- by default the durable-policy gateway
    over the global MCP manager -- to the `google.calendar_*` tools.

    Raises `PortNotConfigured` until the Google server is enabled and
    `CALENDAR_TIME_ZONE` names the IANA zone events are written in.
    """
    zone = time_zone if time_zone is not None else settings.calendar_time_zone
    if not zone:
        raise PortNotConfigured(
            "CALENDAR_TIME_ZONE is not set; name the IANA zone calendar events are written "
            "in, e.g. 'America/New_York'",
            port="calendar",
        )
    if gateway is None:
        if not settings.mcp_google_enabled:
            raise PortNotConfigured(
                "MCP_GOOGLE_ENABLED is off; connect Google (`personalos.cli auth google`) "
                "and enable it",
                port="calendar",
            )
        gateway = build_tool_gateway(policy=policy or build_policy_engine())
    return GoogleCalendar(
        gateway,
        time_zone=zone,
        calendar_id=calendar_id or settings.calendar_id,
    )


def build_calendar_change_scanner(
    calendar: GoogleCalendar,
    session_factory: Callable[[], object] = SessionLocal,
) -> CalendarChangeScanner:
    """Build the sweep that publishes `calendar.event_changed` for moved interviews.

    Publishes to the outbox, deduplicated per observed version of the event,
    where `build_outbox_dispatcher` routes it into a prep-block re-plan.
    """
    return CalendarChangeScanner(calendar, build_event_outbox(session_factory).publish)


def build_mailbox_scope(
    labels: Iterable[str] | None = None, queries: Iterable[str] | None = None
) -> MailboxScope:
    """The mailbox scope: `GMAIL_RECRUITER_LABELS` and `GMAIL_RECRUITER_QUERIES` by default."""
    return MailboxScope(
        labels=tuple(settings.gmail_recruiter_labels if labels is None else labels),
        queries=tuple(settings.gmail_recruiter_queries if queries is None else queries),
    )


def build_mailbox_cursor_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> SqlMailboxCursorStore:
    """The store the inbox poller's cursor -- and the events it moves with -- are written to."""
    return SqlMailboxCursorStore(session_factory)


def build_gmail_recruiter_inbox(
    gateway: ToolGateway | None = None,
    *,
    session_factory: Callable[[], object] = SessionLocal,
    policy: PolicyEngine | None = None,
    scope: MailboxScope | None = None,
    user_id: UUID | None = None,
    account: str | None = None,
) -> GmailRecruiterInbox:
    """Build the Gmail inbox: the poller the worker sweeps, and the classifier's reader.

    Every call goes through `gateway` -- by default the durable-policy gateway
    over the global MCP manager -- to the `google.gmail_*` tools. Mail is
    matched against `user_id`'s applications (default: `GMAIL_INBOX_ACTOR`'s).

    Raises `PortNotConfigured` while the scope is empty (nothing is in view),
    the Google server is off, or no Google account is connected.
    """
    scope = scope if scope is not None else build_mailbox_scope()
    if scope.is_empty:
        raise PortNotConfigured(
            "GMAIL_RECRUITER_LABELS and GMAIL_RECRUITER_QUERIES are both empty; name the "
            "labels or Gmail searches recruiter mail is found by",
            port="mailbox",
        )
    if gateway is None:
        if not settings.mcp_google_enabled:
            raise PortNotConfigured(
                "MCP_GOOGLE_ENABLED is off; connect Google (`personalos.cli auth google`) "
                "and enable it",
                port="mailbox",
            )
        gateway = build_tool_gateway(policy=policy or build_policy_engine())
    if account is None:
        try:
            account = connected_google_credential(session_factory).name
        except CredentialReconnectRequired as exc:
            raise PortNotConfigured(str(exc), port="mailbox") from exc
    if user_id is None:
        from personalos.domain.search_profiles import actor_user_id

        user_id = actor_user_id(settings.gmail_inbox_actor)
    return GmailRecruiterInbox(
        gateway,
        build_mailbox_cursor_store(session_factory),
        scope=scope,
        user_id=user_id,
        account=account,
        backfill_days=settings.gmail_backfill_days,
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


def build_application_board_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> ApplicationBoardStore:
    """Build the store `GET /v1/applications` reads the board and detail from.

    Also how a scored match is kept for the board (`record_match`) and how a
    board move with an external effect is recorded before its run is queued.
    """
    return ApplicationBoardStore(session_factory)


def build_audit_trail_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> AuditTrailStore:
    """Build the read-only store `GET /v1/audit` and `GET /v1/workflows/{id}/audit` read."""
    return AuditTrailStore(session_factory)


def build_workflow_progress_log(
    session_factory: Callable[[], object] = SessionLocal,
) -> WorkflowProgressLog:
    """Build the log a run's progress events go to, and `GET .../events` streams from."""
    return WorkflowProgressLog(session_factory)


def build_integration_health_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> IntegrationHealthStore:
    """Build the store that records which integrations need reconnecting."""
    return IntegrationHealthStore(session_factory)


def build_workflow_failure_store(
    session_factory: Callable[[], object] = SessionLocal,
    registry: WorkflowThreadRegistry | None = None,
) -> WorkflowFailureStore:
    """Build the store `GET /v1/workflows/{id}/failures`, retry and dismiss read."""
    registry = registry or build_workflow_thread_registry(session_factory)
    return WorkflowFailureStore(
        session_factory,
        build_workflow_status_reader(session_factory, registry),
        build_workflow_command_queue(session_factory),
        build_integration_health_store(session_factory),
    )


def build_application_transitioner(
    store: ApplicationLifecycleStore | None = None,
    session_factory: Callable[[], object] = SessionLocal,
) -> StoreApplicationTransitioner:
    """Bind `JobSearchGraph`'s `ApplicationTransitioner` port to the lifecycle store.

    Pass the result as `application_transitioner=` when constructing the
    graph. Without it, a board move that asks for an external effect is
    refused by the graph rather than proposed, since nothing could write the
    status change once the effect had run.
    """
    return StoreApplicationTransitioner(store or build_application_lifecycle_store(session_factory))


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


def register_application_transition_thread(
    *,
    user_id: UUID,
    application_id: UUID,
    registry: WorkflowThreadRegistry,
    actor_id: str = "system",
    correlation_id: UUID | None = None,
) -> WorkflowThread:
    """Derive and register the thread board moves on one application run on.

    A job-search thread with its own workflow, created on first use and reused
    after: see `application_transition_thread_id`.
    """
    thread_id = application_transition_thread_id(user_id, application_id)
    existing = registry.resolve(thread_id)
    if existing is not None:
        return existing
    return registry.register(
        thread_id=thread_id,
        workflow_name=JOB_SEARCH_WORKFLOW,
        workflow_id=registry.create_workflow(JOB_SEARCH_WORKFLOW),
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


def build_event_outbox(session_factory: Callable[[], object] = SessionLocal) -> EventOutbox:
    """Build the outbox event dispatchers claim from.

    Leaves `workflow.command` rows to the workflow-command worker: those run
    for as long as a graph does, under a workflow lease, and must not be
    reclaimed when this outbox's shorter visibility timeout runs out.
    """
    return EventOutbox(session_factory, exclude_types=(WORKFLOW_COMMAND_EVENT,))


def build_event_failure_store(
    session_factory: Callable[[], object] = SessionLocal,
) -> EventFailureStore:
    """Build the store the failures endpoints list and retry parked outbox events through."""
    return EventFailureStore(session_factory, build_event_outbox(session_factory))


def build_correlated_wait_lookup(
    session_factory: Callable[[], object] = SessionLocal,
    registry: WorkflowThreadRegistry | None = None,
) -> CorrelatedWaitLookup:
    """Build the lookup that finds the waiting workflow an event belongs to."""
    return CorrelatedWaitLookup(
        session_factory, build_workflow_status_reader(session_factory, registry)
    )


def build_idempotent_consumer(
    consumer: str, session_factory: Callable[[], object] = SessionLocal
) -> IdempotentConsumer:
    """Build the inbox `consumer` records the events it has applied in."""
    return IdempotentConsumer(session_factory, consumer)


def register_event_handler_thread(
    envelope: EventEnvelope, *, registry: WorkflowThreadRegistry
) -> WorkflowThread:
    """Derive and register the thread a new run for `envelope` goes on.

    Its own workflow per event, like `register_application_transition_thread`:
    sharing one would put every event-handler run under a single lease and
    serialize them. Reused when the event is redelivered, since the thread id
    is derived from its `dedupe_key`.
    """
    thread_id = event_handler_thread_id(envelope.type, envelope.dedupe_key)
    existing = registry.resolve(thread_id)
    if existing is not None:
        return existing
    return registry.register(
        thread_id=thread_id,
        workflow_name=EVENT_HANDLER_WORKFLOW,
        workflow_id=registry.create_workflow(EVENT_HANDLER_WORKFLOW),
        actor_id=f"event:{envelope.source}",
        correlation_id=envelope.correlation_id or envelope.event_id,
    )


def register_calendar_change_thread(
    envelope: EventEnvelope, user_id: UUID, *, registry: WorkflowThreadRegistry
) -> WorkflowThread:
    """Derive and register the job-search thread one calendar change is re-planned on.

    A job-search thread (it runs on the job-search graph) with its own
    workflow, one per observed move: the id is derived from the event's
    `dedupe_key`, so a redelivery finds the run its first delivery started,
    and a later move gets a run of its own rather than stacking onto one
    parked at an approval.
    """
    return _register_event_job_search_thread(envelope, user_id, "calendar_change", registry)


def register_inbound_email_thread(
    envelope: EventEnvelope, user_id: UUID, *, registry: WorkflowThreadRegistry
) -> WorkflowThread:
    """Derive and register the job-search thread one inbound email is classified on.

    One per message, for the reason a calendar change gets one: the id comes
    from the event's `dedupe_key` (the provider's message id), so a redelivery
    finds the first delivery's run, and a reply parked at approval on one
    message does not hold up the next.
    """
    return _register_event_job_search_thread(envelope, user_id, "email", registry)


def _register_event_job_search_thread(
    envelope: EventEnvelope, user_id: UUID, kind: str, registry: WorkflowThreadRegistry
) -> WorkflowThread:
    thread_id = job_search_thread_id(user_id, f"{kind}:{envelope.dedupe_key}")
    existing = registry.resolve(thread_id)
    if existing is not None:
        return existing
    return registry.register(
        thread_id=thread_id,
        workflow_name=JOB_SEARCH_WORKFLOW,
        workflow_id=registry.create_workflow(JOB_SEARCH_WORKFLOW),
        user_id=user_id,
        actor_id=f"event:{envelope.source}",
        correlation_id=envelope.correlation_id or envelope.event_id,
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
    "build_local_api_token",
    "build_model_gateway",
    "JOB_SEARCH_WORKFLOW",
    "SUPERVISOR_WORKFLOW",
    "EVENT_HANDLER_WORKFLOW",
    "build_event_outbox",
    "build_event_failure_store",
    "build_correlated_wait_lookup",
    "build_idempotent_consumer",
    "register_event_handler_thread",
    "build_operation_store",
    "register_mcp_servers",
    "build_files_mcp_server",
    "build_google_mcp_server",
    "connected_google_credential",
    "google_authorizer",
    "serve_google_mcp_stdio",
    "connect_google_mcp_stdio",
    "build_policy_engine",
    "build_tool_gateway",
    "configured_job_providers",
    "build_job_providers",
    "build_posting_catalog",
    "build_search_profile_store",
    "build_event_emitter",
    "build_job_matcher",
    "build_artifact_packet_builder",
    "EMBEDDING_PROVIDERS",
    "build_embedder",
    "embedding_spec_from_settings",
    "build_reembedder",
    "build_resume_ingestor",
    "build_document_overwrite_executor",
    "GOOGLE_PROVIDER",
    "API_KEY_PROVIDERS",
    "build_secret_store",
    "build_credential_broker",
    "initialize_mcp_servers",
    "build_workflow_thread_registry",
    "build_durable_checkpointer",
    "build_workflow_command_queue",
    "build_worker_command_queue",
    "build_workflow_status_reader",
    "build_workflow_action_reader",
    "build_approval_queue",
    "build_workflow_lease_store",
    "build_journaled_action_executor",
    "build_tool_executor",
    "build_application_board_store",
    "build_audit_trail_store",
    "build_integration_health_store",
    "build_workflow_progress_log",
    "build_workflow_failure_store",
    "build_application_transitioner",
    "register_application_transition_thread",
    "register_calendar_change_thread",
    "register_inbound_email_thread",
    "build_mailbox_scope",
    "build_mailbox_cursor_store",
    "build_gmail_recruiter_inbox",
    "build_recruiter_event_store",
    "build_pending_checkpoint_store",
    "build_calendar_action_executor",
    "build_google_calendar",
    "build_calendar_change_scanner",
    "build_checkpoint_condition_evaluator",
    "build_pending_checkpoint_scheduler",
    "register_job_search_thread",
    "register_supervisor_thread",
    "build_job_search_subgraph_runner",
]
