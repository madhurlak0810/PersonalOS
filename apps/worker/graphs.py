"""The graphs a worker runs unless the deployment names others.

`build_worker_graphs` is the default `settings.worker_graphs`. It compiles the
Supervisor and the Job Search subgraph on the durable Postgres checkpointer
-- one saver shared by both, so a workflow's threads are all found by its
`workflow_id` -- and binds every port that has a real implementation:

- the policy engine backed by `policy_decisions`, and the tool gateway behind it;
- the model gateway configured by `LLM_GATEWAY_CONFIG` (deterministic rules
  when unset) for intent routing, semantic assessment and recruiter mail;
- the persistence stores for profiles, postings, evidence, drafts, recruiter
  events, lifecycle transitions, durable waits and the outbox;
- the embedder `EMBEDDING_PROVIDER` names, the same one `ingest-resume`
  stores evidence vectors with;
- the Greenhouse boards in `GREENHOUSE_BOARDS`, through the gateway;
- Google Calendar, as both the calendar reader and the calendar executor's
  client, once `MCP_GOOGLE_ENABLED` and `CALENDAR_TIME_ZONE` are set;
- the Gmail inbox, as the reader `email.received` runs fetch their message
  through, once `MCP_GOOGLE_ENABLED` and a mailbox scope
  (`GMAIL_RECRUITER_LABELS` / `GMAIL_RECRUITER_QUERIES`) are set.

The ports with no adapter yet are bound to the `apps.worker.unconfigured`
stand-ins, which fail the step that reaches them with a retryable
`PortNotConfigured` rather than answer with nothing. `WorkerGraphs.unbound_ports`
lists them, and the worker logs that list at startup.

A deployment that wants other graphs names its own factory in `WORKER_GRAPHS`;
one that wants these graphs with a port or two swapped calls this with
`overrides`, keyed by `JobSearchGraph` constructor argument.
"""

import logging
import os
import socket
from collections.abc import Callable, Mapping
from typing import Any

from apps.worker.main import WorkerGraphs, WorkflowScopedJobSubgraph
from apps.worker.unconfigured import (
    NotConfiguredApplicationStore,
    NotConfiguredApplicationSubmitter,
    NotConfiguredCalendar,
    NotConfiguredDraftWriter,
    NotConfiguredEmbedder,
    NotConfiguredJobBoard,
    NotConfiguredMailbox,
    NotConfiguredReplyDrafter,
    _NotConfigured,
)
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    build_application_transitioner,
    build_artifact_packet_builder,
    build_calendar_action_executor,
    build_document_overwrite_executor,
    build_durable_checkpointer,
    build_embedder,
    build_event_emitter,
    build_gmail_recruiter_inbox,
    build_google_calendar,
    build_integration_health_store,
    build_job_matcher,
    build_job_providers,
    build_model_gateway,
    build_pending_checkpoint_scheduler,
    build_policy_engine,
    build_posting_catalog,
    build_recruiter_event_store,
    build_search_profile_store,
    build_tool_gateway,
    build_workflow_lease_store,
    build_workflow_thread_registry,
)
from personalos.domain.errors import PortNotConfigured
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    JobSearchContractError,
)
from personalos.domain.workflow import JOB_SEARCH_THREAD_NAMESPACE, SUPERVISOR_THREAD_NAMESPACE
from personalos.graphs.job_search import (
    ExtractingRecruiterClassifier,
    InterruptOnlyApprovalGate,
    JobSearchGraph,
)
from personalos.graphs.supervisor import SupervisorGraph
from personalos.persistence.leases import DEFAULT_LEASE_TTL_SECONDS

logger = logging.getLogger(__name__)


class ActionKindRouter:
    """Hands each approved action to the executor for its kind.

    What `ToolExecutor` wraps for the kinds `CalendarActionExecutor` passes
    on. A kind with no route is refused rather than guessed at.
    """

    def __init__(self, routes: Mapping[ActionKind, Any]):
        """Take the executor for each action kind."""
        self.routes = dict(routes)

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        executor = self.routes.get(intent.kind)
        if executor is None:
            raise JobSearchContractError(f"no executor is wired for '{intent.kind.value}' actions")
        return await executor.execute(intent, decision)


def default_job_search_ports(
    session_factory: Callable[[], Any],
) -> tuple[dict[str, Any], dict[str, list[_NotConfigured]]]:
    """`JobSearchGraph` keyword arguments for this build, and the stand-ins among them.

    The second value maps each argument to the `unconfigured` stand-ins bound
    under it, directly or inside an adapter (the calendar client inside the
    action executor), so replacing the argument also drops them from the
    unbound list.
    """
    policy = build_policy_engine(session_factory)
    models = build_model_gateway()
    matcher = build_job_matcher(models.semantic_assessor(), session_factory)
    recruiter_events = build_recruiter_event_store(session_factory)
    extractor = models.recruiter_event_extractor()

    providers: list[Any] = list(build_job_providers(policy=policy))
    board_stubs: list[_NotConfigured] = []
    if not providers:
        board_stubs.append(NotConfiguredJobBoard())
        providers = list(board_stubs)

    writer: Any = models.draft_writer()
    packet_stubs: list[_NotConfigured] = []
    embedder: Any = build_embedder()
    if embedder is None:
        embedder = NotConfiguredEmbedder()
        packet_stubs.append(embedder)
    if writer is None:
        writer = NotConfiguredDraftWriter()
        packet_stubs.append(writer)

    mailbox = NotConfiguredMailbox()
    inbound_reader: Any
    inbound_stubs: list[_NotConfigured] = []
    try:
        inbound_reader = build_gmail_recruiter_inbox(session_factory=session_factory, policy=policy)
    except PortNotConfigured as exc:
        logger.info("the Gmail inbox is not wired: %s", exc)
        inbound_reader = mailbox
        inbound_stubs.append(mailbox)
    calendar: Any
    calendar_stubs: list[_NotConfigured] = []
    try:
        calendar = build_google_calendar(policy=policy)
    except PortNotConfigured as exc:
        logger.info("calendar is not wired: %s", exc)
        calendar = NotConfiguredCalendar()
        calendar_stubs.append(calendar)
    submitter = NotConfiguredApplicationSubmitter()
    action_executor = build_calendar_action_executor(
        calendar,
        session_factory,
        policy=policy,
        fallback=ActionKindRouter(
            {
                ActionKind.SUBMIT_APPLICATION: submitter,
                ActionKind.SEND_RECRUITER_MESSAGE: mailbox,
                ActionKind.OVERWRITE_DOCUMENT: build_document_overwrite_executor(
                    build_tool_gateway(policy=policy)
                ),
            }
        ),
    )
    application_store = NotConfiguredApplicationStore()
    reply_drafter = NotConfiguredReplyDrafter()

    ports: dict[str, Any] = {
        "profile_store": build_search_profile_store(session_factory),
        "providers": providers,
        "posting_catalog": build_posting_catalog(session_factory),
        "scorer": matcher,
        "evidence_checker": matcher,
        "packet_builder": build_artifact_packet_builder(
            writer, embedder, session_factory, policy=policy
        ),
        "approval_gate": InterruptOnlyApprovalGate(),
        "action_executor": action_executor,
        "application_store": application_store,
        "event_emitter": build_event_emitter(session_factory),
        "recruiter_inbox": mailbox,
        "recruiter_classifier": ExtractingRecruiterClassifier(extractor),
        "checkpoint_scheduler": build_pending_checkpoint_scheduler(session_factory=session_factory),
        "recruiter_event_extractor": extractor,
        "application_directory": recruiter_events,
        "recruiter_event_recorder": recruiter_events,
        "reply_drafter": reply_drafter,
        "calendar_reader": calendar,
        "application_transitioner": build_application_transitioner(session_factory=session_factory),
        "inbound_message_reader": inbound_reader,
    }
    stubs = {
        "providers": board_stubs,
        "packet_builder": packet_stubs,
        "action_executor": [*calendar_stubs, submitter, mailbox],
        "application_store": [application_store],
        "recruiter_inbox": [mailbox],
        "reply_drafter": [reply_drafter],
        "calendar_reader": calendar_stubs,
        "inbound_message_reader": inbound_stubs,
    }
    return ports, stubs


def build_worker_graphs(
    session_factory: Callable[[], Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> WorkerGraphs:
    """The Supervisor and the Job Search subgraph, durable, with this build's ports.

    `overrides` replaces `JobSearchGraph` arguments by name; an argument it
    replaces no longer counts towards `unbound_ports`. `lease_ttl_seconds`
    must match the worker's, as `apps.worker.main.build_worker` explains.
    """
    registry = build_workflow_thread_registry(session_factory)
    checkpointer = build_durable_checkpointer(session_factory, registry)
    ports, stubs = default_job_search_ports(session_factory)
    for name, port in (overrides or {}).items():
        ports[name] = port
        stubs.pop(name, None)

    job_search = JobSearchGraph(**ports, checkpointer=checkpointer).build()
    supervisor = SupervisorGraph(
        build_model_gateway().intent_classifier(),
        WorkflowScopedJobSubgraph(job_search, registry, prepare_application=True),
        checkpointer=checkpointer,
    ).build()

    owner = f"worker:{socket.gethostname()}:{os.getpid()}"
    leases = build_workflow_lease_store(session_factory, ttl_seconds=lease_ttl_seconds)
    integrations = build_integration_health_store(session_factory)

    def runner(graph: Any) -> DurableWorkflowRunner:
        return DurableWorkflowRunner(
            graph, registry=registry, leases=leases, owner=owner, integrations=integrations
        )

    return WorkerGraphs(
        runners={
            SUPERVISOR_THREAD_NAMESPACE: runner(supervisor),
            JOB_SEARCH_THREAD_NAMESPACE: runner(job_search),
        },
        unbound_ports=tuple(sorted({stub.port for group in stubs.values() for stub in group})),
    )


__all__ = ["ActionKindRouter", "build_worker_graphs", "default_job_search_ports"]
