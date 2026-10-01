"""Acceptance tests for credential isolation, end to end.

Two claims, each tested against the real thing rather than a model of it:

1. **A full graph checkpoint contains no credential value.** A Job Search run
   executes on the database-backed checkpointer with an action executor that
   really does broker a token from a secret store, and whose provider then
   fails in the worst way available -- an error that quotes the request's
   `Authorization` header, the refresh token and the API key back. Every byte
   the run persisted is then searched.
2. **A tool error carrying an `Authorization` header is redacted in the log
   and in the trace.** A real MCP tool raises it; it travels the production
   path (gateway -> policy -> MCP adapter -> server) and is read back out of
   the log records, the `ToolResult`, and an exported span.

The secrets here are deliberately *shapeless* where it matters: a token that
looks like `ya29....` would be caught by a format rule, and the test would
pass without proving that the broker's registration of the actual value works.
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from personalos.bootstrap import build_tool_gateway
from personalos.domain.context import ExecutionContext
from personalos.domain.credentials import (
    AccessToken,
    AccessTokenProvider,
    CredentialRef,
    SecretValue,
)
from personalos.domain.job_search import ActionIntent, ActionReceipt, ApprovalDecision
from personalos.domain.models import Intent
from personalos.domain.redaction import REDACTED, default_redactor
from personalos.executor.credentials import CredentialBroker
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.mcp.manager import MCPServerManager
from personalos.observability.trace_redaction import RedactingSpanExporter
from personalos.persistence.action_journal import JournaledActionExecutor
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.models import (
    AuditEventModel,
    CheckpointModel,
    CheckpointWriteModel,
    EvidenceChunkModel,
    OperationModel,
)
from personalos.persistence.repositories import (
    AuditEventRepository,
    CredentialRepository,
    EvidenceChunkRepository,
)
from personalos.policy import IntentOrigin, ToolIntent
from personalos.secrets.exchange import ApiKeyExchanger
from personalos.secrets.store import InMemorySecretStore
from personalos.tools.gateway import ToolExecutionError
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes

GMAIL = CredentialRef("google", "candidate@example.com")
BOARD = CredentialRef("greenhouse", "default")

#: Long-lived secrets, as they sit in the keychain.
REFRESH_TOKEN = "shapeless-long-lived-gmail-refresh-value-7f3a"
BOARD_API_KEY = "shapeless-long-lived-job-board-key-value-91c2"
#: What the broker mints from the refresh token.
ACCESS_TOKEN = "shapeless-short-lived-access-value-c05e"

ALL_SECRETS = (REFRESH_TOKEN, BOARD_API_KEY, ACCESS_TOKEN)

#: A header value the broker never saw, so only the pattern rules can catch it.
UNREGISTERED_BEARER = "shapeless-header-the-broker-never-minted-44d1"


@pytest.fixture(autouse=True)
def _forget_registered_secrets():
    """Keep the process-wide redactor's known values from leaking across tests."""
    default_redactor().forget_secrets()
    yield
    default_redactor().forget_secrets()


class FakeGoogleExchanger:
    """Stands in for Google's token endpoint: refresh token in, access token out."""

    async def exchange(self, ref, secret, scopes):
        from datetime import datetime, timedelta, timezone

        assert secret.reveal() == REFRESH_TOKEN
        return AccessToken(
            ref,
            SecretValue(ACCESS_TOKEN),
            datetime.now(timezone.utc) + timedelta(minutes=30),
            tuple(scopes),
        )


def _broker() -> CredentialBroker:
    store = InMemorySecretStore()
    store.put(GMAIL, SecretValue(REFRESH_TOKEN))
    store.put(BOARD, SecretValue(BOARD_API_KEY))
    return CredentialBroker(store, {"google": FakeGoogleExchanger(), "greenhouse": ApiKeyExchanger()})


class LeakyProviderError(RuntimeError):
    """What a careless provider SDK raises: the request it failed to send, verbatim."""


class BrokeredSubmitter:
    """An action executor that brokers real tokens, against a provider that leaks them.

    It does everything right -- holds references, asks the broker at the moment
    of the call -- and then does the one thing wrong that redaction exists for:
    it copies the provider's error text straight into the receipt, which is
    graph state, which is checkpointed.
    """

    def __init__(self, tokens: AccessTokenProvider):
        self.tokens = tokens
        self.presented: list[dict[str, str]] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        assert decision.authorizes(intent)
        gmail = await self.tokens.exchange(GMAIL, ["gmail.send"])
        board = await self.tokens.exchange(BOARD)
        headers = {**gmail.authorization_header(), "X-Api-Key": board.secret.reveal()}
        self.presented.append(headers)
        try:
            raise LeakyProviderError(
                f"401 Unauthorized. Request headers: {headers!r}. "
                f"Tried refresh with refresh_token={REFRESH_TOKEN}. "
                f"Also saw Authorization: Bearer {UNREGISTERED_BEARER} "
                f"and Cookie: SID={UNREGISTERED_BEARER}; HttpOnly "
                f"after redirect https://accounts.example/cb?code={UNREGISTERED_BEARER}&state=s"
            )
        except LeakyProviderError as error:
            return ActionReceipt(action_id=intent.action_id, ok=False, detail=str(error))


def _persisted_bytes(factory, db_path: Path) -> dict[str, bytes]:
    """Everything the run wrote, decoded back to what was actually serialized.

    Checkpoint state is base64 inside a JSON column, so searching the row as
    stored would find nothing whether or not a secret was in it. Each envelope
    is decoded to the serializer's own bytes first.
    """
    found: dict[str, bytes] = {}
    session = factory()
    try:
        for row in session.query(CheckpointModel).all():
            found[f"checkpoint:{row.checkpoint_id}"] = base64.b64decode(row.checkpoint["data"])
            found[f"checkpoint-metadata:{row.checkpoint_id}"] = json.dumps(
                row.checkpoint_metadata
            ).encode()
        for row in session.query(CheckpointWriteModel).all():
            found[f"write:{row.checkpoint_id}:{row.task_id}:{row.idx}"] = base64.b64decode(
                row.value["data"]
            )
        for row in session.query(OperationModel).all():
            found[f"operation:{row.idempotency_key}"] = json.dumps(
                [row.result, row.error], default=str
            ).encode()
    finally:
        session.close()
    # And the database file itself, for anything the queries above did not name.
    found["sqlite-file"] = db_path.read_bytes()
    return found


def test_a_full_graph_checkpoint_contains_no_credential_value(tmp_path):
    from personalos.graphs.job_search import JobSearchGraph

    db_path = tmp_path / "workflow.db"
    factory = durable.session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)
    thread = registry.register(
        thread_id="job-search:credential-isolation",
        workflow_name=durable.WORKFLOW_NAME,
        user_id=fakes.USER_ID,
    )

    # The database knows the credentials exist -- by reference.
    session = factory()
    CredentialRepository(session).create(ref=GMAIL, kind="oauth_refresh_token", scopes=["gmail.send"])
    CredentialRepository(session).create(ref=BOARD, kind="api_key")
    session.close()

    submitter = BrokeredSubmitter(_broker())
    saver = SqlAlchemyCheckpointSaver(factory, registry)
    graph = JobSearchGraph(
        profile_store=fakes.FakeProfileStore(),
        providers=[fakes.FakeProvider()],
        scorer=fakes.FakeScorer(),
        evidence_checker=fakes.FakeEvidenceChecker(),
        packet_builder=fakes.FakePacketBuilder(),
        approval_gate=fakes.FakeApprovalGate(),
        action_executor=JournaledActionExecutor(
            submitter, factory, workflow_id=thread.workflow_id
        ),
        application_store=fakes.FakeApplicationStore(),
        event_emitter=fakes.FakeEventEmitter(),
        checkpointer=saver,
    ).build()

    final = asyncio.run(graph.ainvoke(durable.initial_state(), config=thread.config()))

    # The run really did use the credentials: the provider was presented with
    # the brokered token and the leased key, and nothing else.
    assert submitter.presented == [
        {"Authorization": f"Bearer {ACCESS_TOKEN}", "X-Api-Key": BOARD_API_KEY}
    ]

    persisted = _persisted_bytes(factory, db_path)
    checkpoints = [name for name in persisted if name.startswith("checkpoint:")]
    assert len(checkpoints) > 3, "expected a checkpoint per super-step"

    for name, blob in persisted.items():
        for secret in (*ALL_SECRETS, UNREGISTERED_BEARER):
            assert secret.encode() not in blob, f"credential value found in {name}"

    # Guard against the test passing vacuously: the leaky error *did* reach
    # checkpointed state, and what is there is its redacted form.
    serialized_state = b"".join(persisted[name] for name in checkpoints)
    assert b"401 Unauthorized" in serialized_state
    assert REDACTED.encode() in serialized_state

    # The state a resumed run would load is equally clean...
    loaded = saver.get_tuple(thread.config())
    assert not any(secret in repr(loaded.checkpoint) for secret in ALL_SECRETS)
    # ...and so is the result handed back to the caller (and on to any model).
    assert not any(secret in repr(final) for secret in (*ALL_SECRETS, UNREGISTERED_BEARER))
    assert REDACTED in repr(final["action_receipts"])


def test_a_secret_object_placed_in_state_fails_the_checkpoint_instead_of_persisting(tmp_path):
    """The structural half: a token is not a value graph state can hold."""
    factory = durable.session_factory(tmp_path / "cp.db")
    registry = WorkflowThreadRegistry(factory)
    registry.register(thread_id="t-1", workflow_name=durable.WORKFLOW_NAME)

    class PassThrough:
        """A redactor that redacts nothing, to show the serializer refuses on its own."""

        def redact(self, value: Any) -> Any:
            return value

    saver = SqlAlchemyCheckpointSaver(factory, registry, redactor=PassThrough())
    from langgraph.checkpoint.base import empty_checkpoint

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"token": SecretValue(ACCESS_TOKEN)}

    with pytest.raises(TypeError):
        saver.put({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}, checkpoint, {}, {})

    session = factory()
    assert session.query(CheckpointModel).count() == 0
    session.close()


def test_audit_events_and_the_vector_store_are_redacted_before_they_are_persisted(tmp_path):
    db_path = tmp_path / "audit.db"
    factory = durable.session_factory(db_path)
    default_redactor().register_secret(REFRESH_TOKEN)

    session = factory()
    AuditEventRepository(session).create(
        actor="executor:gmail",
        action=f"gmail.send failed with Authorization: Bearer {UNREGISTERED_BEARER}",
        target_ref=f"https://accounts.example/cb?code={UNREGISTERED_BEARER}&state=s",
        result="failure",
    )
    EvidenceChunkRepository(session).create(
        user_id=fakes.USER_ID,
        source_type="resume",
        chunk_text=f"Notes pasted from a terminal: export TOKEN={REFRESH_TOKEN} then deploy",
        embedding=[0.0] * 4,
        embedding_model="test",
    )
    session.close()

    session = factory()
    event = session.query(AuditEventModel).one()
    chunk = session.query(EvidenceChunkModel).one()
    session.close()

    assert event.action == f"gmail.send failed with Authorization: {REDACTED}"
    assert event.target_ref == f"https://accounts.example/cb?code={REDACTED}&state=s"
    assert chunk.chunk_text == f"Notes pasted from a terminal: export TOKEN={REDACTED} then deploy"
    raw = db_path.read_bytes()
    assert UNREGISTERED_BEARER.encode() not in raw and REFRESH_TOKEN.encode() not in raw


# --- A tool error with an Authorization header -------------------------------


class _SearchParams(Intent):
    keywords: list[str]
    locations: list[str]


class LeakyJobsServer(MCPServer):
    """A `jobs` server whose search tool fails by quoting its own request."""

    def __init__(self, tracer: trace.Tracer):
        super().__init__("jobs", "job board whose client leaks headers")
        self.tracer = tracer
        self.initialize()

    def initialize(self):
        self.register_tool(
            ToolSchema("search_jobs", "search", _SearchParams, required=["keywords", "locations"]),
            self._search,
        )

    async def _search(self, keywords: list[str], locations: list[str]):
        # Raising inside the span is what makes OpenTelemetry record the
        # exception -- message and stack trace -- as a span event.
        with self.tracer.start_as_current_span("jobs.search_jobs") as span:
            span.set_attribute("http.request.header.authorization", f"Bearer {UNREGISTERED_BEARER}")
            raise RuntimeError(
                "HTTP 401 from board. Request was: GET /jobs "
                f"Authorization: Bearer {UNREGISTERED_BEARER}; "
                f"Cookie: session={UNREGISTERED_BEARER}"
            )


def test_a_tool_error_with_an_authorization_header_is_redacted_in_logs_and_traces(caplog):
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exported)))

    manager = MCPServerManager()
    manager.register_server(LeakyJobsServer(provider.get_tracer("test")))
    gateway = build_tool_gateway(manager)
    intent = ToolIntent(
        server="jobs",
        tool="search_jobs",
        arguments={"keywords": ["python"], "locations": ["remote"]},
        origin=IntentOrigin.SYSTEM,
        requested_by="executor:test",
        context=ExecutionContext.new(),
    )

    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(gateway.dispatch(intent))

    # The tool really failed, with the leaky error, down the production path.
    assert result.success is False
    assert "HTTP 401 from board" in result.error

    # Logged: the message, and the traceback `exc_info=True` attached to it.
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "the tool failure was not logged"
    rendered = "\n".join(logging.Formatter().format(record) for record in caplog.records)
    assert "Traceback" in rendered and "RuntimeError" in rendered
    assert UNREGISTERED_BEARER not in rendered
    assert f"Authorization: {REDACTED}" in rendered
    assert f"Cookie: {REDACTED}" in rendered
    assert all(record.exc_info is None for record in caplog.records)

    # Traced: attributes, the recorded exception event, and the span status.
    (span,) = exported.get_finished_spans()
    dumped = span.to_json()
    assert "exception" in [event.name for event in span.events]
    assert "HTTP 401 from board" in dumped
    assert UNREGISTERED_BEARER not in dumped
    assert REDACTED in dumped

    # And what is handed back toward the executor -- and from there to graph
    # state and a model -- is the redacted error too.
    assert UNREGISTERED_BEARER not in result.error
    with pytest.raises(ToolExecutionError) as raised:
        result.unwrap()
    assert UNREGISTERED_BEARER not in str(raised.value)
    assert UNREGISTERED_BEARER not in repr(raised.value.details)
