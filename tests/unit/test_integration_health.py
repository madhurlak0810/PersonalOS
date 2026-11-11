"""Integration health: checks, degraded state, circuit breakers and recovery.

The acceptance criteria this file exists for:

1. **A revoked OAuth token degrades the Google integrations once, shows on
   `/failures`, and loops nothing.**
   `test_a_revoked_oauth_token_degrades_the_google_integrations_once_without_retry_loops`
   has Google refuse the token (HTTP 401) under the Gmail health check and the
   Gmail poll. Every Google integration goes `degraded` with a `reconnect`
   action, one transition each, and later sweeps neither probe nor poll again.
2. **With Gmail down, job discovery completes and follow-ups stay queued.**
   `test_with_gmail_down_job_discovery_completes_and_follow_ups_stay_pending`
   runs a real discovery-to-application workflow through the gated command
   worker while Gmail is down, then sweeps the follow-up it scheduled: the
   follow-up and an inbound-mail classification both wait, and both go ahead
   once Gmail recovers.
3. **Health recovers on its own after a successful check.**
   `test_a_down_integration_recovers_to_healthy_after_a_successful_check`.
"""

import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.api.routes.integrations import get_google_connection_reader
from apps.worker.health_monitor import GatedSweep, HealthGate, IntegrationHealthMonitor
from apps.worker.workflow_commands import process_next
from mcp_servers.google.api import raise_for_status
from personalos.domain.checkpoints import PendingCheckpointStatus
from personalos.domain.errors import ProviderPermanentFailure, RetryableFailure
from personalos.domain.health import (
    GMAIL,
    GOOGLE,
    GOOGLE_CALENDAR,
    GOOGLE_INTEGRATIONS,
    GOOGLE_MCP,
    HealthCheckResult,
    HealthStatus,
    model_integration,
    run_check,
)
from personalos.domain.models import OutboxEventStatus
from personalos.domain.workflow import (
    JOB_SEARCH_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
    job_search_thread_id,
)
from personalos.executor.google_connect import GoogleConnectionReader
from personalos.models.gateway import ModelConfig, ModelGateway, ModelSpec
from personalos.observability.health import HEALTH_CHANGED_EVENT
from personalos.persistence.integration_health import IntegrationHealthStore
from personalos.persistence.models import Base, OutboxEventModel
from personalos.persistence.workflow_commands import WorkflowCommandQueue
from personalos.providers.fake import FakeJobProvider
from tests.fixtures import job_search_fakes as fakes
from tests.graph_scenarios.test_pending_checkpoints import AT_TRIGGER, Deployment

SECRET = "ya29.a0AfH6SMBx-live-access-token-0123456789"


@pytest.fixture
def factory(tmp_path: Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{tmp_path / 'health.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


class ScriptedCheck:
    """A health check whose probe raises each scripted error in turn, then succeeds."""

    def __init__(self, integration: str, errors: list[Exception] | None = None):
        self.integration = integration
        self.errors = list(errors or [])
        self.calls = 0

    async def health_check(self) -> HealthCheckResult:
        self.calls += 1

        async def probe() -> None:
            if self.errors:
                raise self.errors.pop(0)

        return await run_check(self.integration, probe)


def revoked_token() -> Exception:
    """What the Google MCP server raises when Google rejects the token."""
    try:
        raise_for_status(401)
    except ProviderPermanentFailure as error:
        error.message = f"{error.message} (token {SECRET})"
        return error
    raise AssertionError("a 401 must raise")


def transitions(caplog, integration: str) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == HEALTH_CHANGED_EVENT
        and getattr(record, "integration", None) == integration
    ]


def api(factory) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_failure_services] = lambda: build_failure_services(factory)
    app.dependency_overrides[get_google_connection_reader] = lambda: GoogleConnectionReader(
        factory, IntegrationHealthStore(factory)
    )
    return TestClient(app)


# --- Acceptance 1: revoked OAuth ---------------------------------------------------


async def test_a_revoked_oauth_token_degrades_the_google_integrations_once_without_retry_loops(
    factory, caplog
):
    store = IntegrationHealthStore(factory)
    gmail = ScriptedCheck(GMAIL, [revoked_token() for _ in range(5)])
    calendar = ScriptedCheck(GOOGLE_CALENDAR, [revoked_token() for _ in range(5)])
    monitor = IntegrationHealthMonitor([gmail, calendar], store)
    polls: list[int] = []

    async def poll() -> None:
        polls.append(1)
        raise revoked_token()

    gmail_poll = GatedSweep(GMAIL, poll, store)

    with caplog.at_level(logging.INFO, logger="personalos.integration_health"):
        await monitor.sweep()
        # The poll scheduled in the same pass sees Gmail blocked, and does not call Google.
        assert await gmail_poll() is None
        for _ in range(3):
            report = await monitor.sweep()
            await gmail_poll()

    # Every Google integration is degraded, needing a reconnect.
    for name in GOOGLE_INTEGRATIONS:
        health = store.get(name)
        assert health.status == HealthStatus.DEGRADED.value, name
        assert health.reconnect_required and health.action == "reconnect"
        assert health.reason == ProviderPermanentFailure.REVOKED_AUTH
        assert SECRET not in (health.last_error or "")
        # Flipped once: one transition, however many times the refusal was seen.
        assert len(transitions(caplog, name)) == 1, name

    # No retry loop: after the first refusal nothing asks Google again.
    assert gmail.calls == 1
    assert calendar.calls == 0  # the first refusal already degraded the calendar
    assert polls == []
    assert set(report.skipped) == {GMAIL, GOOGLE_CALENDAR}
    assert HealthGate(store).blocked([GMAIL, GOOGLE_CALENDAR, GOOGLE_MCP])

    # Shown at /failures with a reconnect action, and under /v1/integrations.
    client = api(factory)
    failures = client.get("/v1/failures/integrations")
    assert failures.status_code == 200, failures.text
    listed = {item["provider"]: item for item in failures.json()["integrations"]}
    assert set(listed) == set(GOOGLE_INTEGRATIONS)
    assert all(item["action"] == "reconnect" for item in listed.values())
    assert SECRET not in failures.text

    integrations = {
        item["provider"]: item for item in client.get("/v1/integrations").json()["integrations"]
    }
    assert integrations[GOOGLE]["reconnect_required"] is True
    assert integrations[GMAIL]["health"] == "degraded"
    assert integrations[GMAIL]["action"] == "reconnect"
    assert integrations[GMAIL]["last_error"]


def test_a_reconnect_makes_every_google_integration_healthy_again(factory):
    store = IntegrationHealthStore(factory)
    store.mark_degraded(GOOGLE, reason=ProviderPermanentFailure.REVOKED_AUTH, detail="401")
    assert {h.provider for h in store.unhealthy()} == set(GOOGLE_INTEGRATIONS)

    store.mark_healthy(GOOGLE)

    assert store.unhealthy() == []
    assert HealthGate(store).blocked(GOOGLE_INTEGRATIONS) == []


def test_a_missing_scope_degrades_only_what_saw_it(factory):
    store = IntegrationHealthStore(factory)
    refusal = HealthCheckResult.from_exception(
        GOOGLE_CALENDAR,
        ProviderPermanentFailure(
            "no calendar scope", provider=GOOGLE, reason=ProviderPermanentFailure.MISSING_SCOPE
        ),
    )

    store.record(refusal)

    assert {h.provider for h in store.unhealthy()} == {GOOGLE_CALENDAR, GOOGLE}
    assert store.get(GMAIL) is None


# --- Acceptance 2: Gmail down --------------------------------------------------------


def _gmail_down(store: IntegrationHealthStore) -> None:
    for _ in range(store.down_after):
        store.record(HealthCheckResult.from_exception(GMAIL, RetryableFailure("timed out")))
    assert store.get(GMAIL).status == HealthStatus.DOWN.value


def _pending(factory) -> list[OutboxEventModel]:
    session = factory()
    try:
        return (
            session.query(OutboxEventModel)
            .filter(OutboxEventModel.status == OutboxEventStatus.PENDING.value)
            .all()
        )
    finally:
        session.close()


async def test_with_gmail_down_job_discovery_completes_and_follow_ups_stay_pending(tmp_path):
    deployment = Deployment(tmp_path, evaluator=fakes.ScriptedConditionEvaluator())
    factory = deployment.factory
    health = IntegrationHealthStore(factory)
    gate = HealthGate(health, model_integrations=[model_integration("ollama")])
    deployment.monitor.gate = gate
    queue = WorkflowCommandQueue(factory)
    runners = {JOB_SEARCH_THREAD_NAMESPACE: deployment.runner}
    _gmail_down(health)

    # A discovery run needs no Gmail: it is not held, and runs to an application.
    queue.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.START,
            workflow_id=deployment.thread.workflow_id,
            thread_id=deployment.thread.thread_id,
            actor_id="user:alice",
            correlation_id=uuid4(),
            graph_input={"user_id": str(fakes.USER_ID), "prepare_application": True},
        )
    )
    assert await process_next(queue, runners, gate=gate)
    state = await deployment.graph.aget_state(deployment.thread.config())
    assert not state.next, "the discovery run must finish, not park"
    assert state.values["application"]["status"] == "applied"
    assert deployment.searches == 1
    assert _pending(factory) == []

    # Classifying inbound mail needs Gmail: it is held in the queue, not failed.
    mail_thread = deployment.registry.register(
        thread_id=job_search_thread_id(fakes.USER_ID, "inbound-mail"),
        workflow_name="job_search",
        user_id=fakes.USER_ID,
    )
    mail = WorkflowCommand(
        command_id=uuid4(),
        kind=WorkflowCommandKind.START,
        workflow_id=mail_thread.workflow_id,
        thread_id=mail_thread.thread_id,
        actor_id="system",
        correlation_id=uuid4(),
        graph_input={"user_id": str(fakes.USER_ID), "inbound_message_refs": ["gmail:m-1"]},
    )
    assert gate.requirements(mail.graph_input) == {GMAIL, model_integration("ollama")}
    queue.enqueue(mail)
    assert await process_next(queue, runners, gate=gate)
    (held,) = _pending(factory)
    assert held.available_at is not None
    assert await process_next(queue, runners, gate=gate) is False  # not reclaimed at once

    # The follow-up that run scheduled comes due while Gmail is down: it stays pending.
    (wait,) = deployment.store.open_for_application(fakes.APPLICATION_ID)
    report = await deployment.monitor.sweep(now=AT_TRIGGER)
    assert report.deferred == (wait.checkpoint_id,)
    assert report.fired == ()
    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.PENDING

    # Gmail recovers on its next check, and the follow-up fires on the next sweep.
    await IntegrationHealthMonitor([ScriptedCheck(GMAIL)], health).sweep()
    assert gate.blocked([GMAIL]) == []
    report = await deployment.monitor.sweep(now=AT_TRIGGER + timedelta(minutes=5))
    assert report.fired == (wait.checkpoint_id,)


# --- Acceptance 3: recovery ----------------------------------------------------------


async def test_a_down_integration_recovers_to_healthy_after_a_successful_check(factory, caplog):
    store = IntegrationHealthStore(factory)
    board = ScriptedCheck("greenhouse", [RetryableFailure("HTTP 503") for _ in range(3)])
    monitor = IntegrationHealthMonitor([board], store)

    with caplog.at_level(logging.INFO, logger="personalos.integration_health"):
        statuses = [(await monitor.sweep()).results["greenhouse"] for _ in range(3)]
    assert statuses == ["degraded", "degraded", "down"]
    down = store.get("greenhouse")
    assert down.action == "retry" and not down.reconnect_required
    assert down.consecutive_failures == 3 and down.last_success_at is None
    assert HealthGate(store).blocked(["greenhouse"])

    with caplog.at_level(logging.INFO, logger="personalos.integration_health"):
        report = await monitor.sweep()

    assert report.results == {"greenhouse": "healthy"}
    healthy = store.get("greenhouse")
    assert healthy.status == "healthy" and healthy.action is None
    assert healthy.consecutive_failures == 0 and healthy.last_error is None
    assert healthy.last_success_at is not None
    assert HealthGate(store).blocked(["greenhouse"]) == []
    # Two changes on the way down (to degraded, then down), and one back.
    changes = [
        (r.previous_health_status, r.health_status) for r in transitions(caplog, "greenhouse")
    ]
    assert changes == [(None, "degraded"), ("degraded", "down"), ("down", "healthy")]


# --- Adapters --------------------------------------------------------------------------


async def test_a_check_that_hangs_counts_as_a_failure(factory):
    class Hangs:
        integration = "lever"

        async def health_check(self):
            await asyncio.sleep(10)

    report = await IntegrationHealthMonitor(
        [Hangs()], IntegrationHealthStore(factory), timeout_seconds=0.01
    ).sweep()

    assert report.results == {"lever": "degraded"}


async def test_a_job_board_is_checked_through_the_gateway(factory):
    from personalos.bootstrap import build_job_providers

    board = FakeJobProvider("fakeboard")
    (provider,) = build_job_providers([board])

    assert (await provider.health_check()).ok
    board.health_error = RetryableFailure("board is down")
    failed = await provider.health_check()

    assert not failed.ok and failed.action.value == "retry"
    assert board.health_calls == 2


async def test_a_job_board_without_a_check_of_its_own_counts_as_healthy():
    from personalos.bootstrap import build_job_providers

    class NoCheck:
        name = "plain"

        async def search(self, profile):
            return []

        async def get_job(self, source_job_id):
            return None

    (provider,) = build_job_providers([NoCheck()])

    assert (await provider.health_check()).ok


async def test_local_model_servers_are_probed_and_hosted_ones_are_not():
    probed: list[str] = []

    async def get(url: str) -> None:
        probed.append(url)
        raise RetryableFailure("connection refused")

    gateway = ModelGateway(
        ModelConfig(
            name="hybrid",
            default=ModelSpec(provider="ollama", model="llama3.1:8b"),
            roles={"draft_writer": ModelSpec(provider="anthropic", model="claude-opus-5")},
        ),
        chat_model_factory=lambda spec: object(),
    )

    (check,) = gateway.health_checks(get=get)
    result = await check.health_check()

    assert check.integration == "model.ollama"
    assert probed == ["http://localhost:11434/api/tags"]
    assert not result.ok and result.action.value == "retry"


async def test_the_hashing_embedder_is_always_healthy():
    from personalos.models.embedding import HashingEmbedder

    assert (await HashingEmbedder().health_check()).ok


async def test_the_google_mcp_check_is_down_when_the_server_is_not_registered():
    from personalos.mcp.health import MCPServerHealthCheck
    from personalos.mcp.manager import MCPServerManager

    result = await MCPServerHealthCheck(
        MCPServerManager(), "google", integration=GOOGLE_MCP
    ).health_check()

    assert not result.ok and result.integration == GOOGLE_MCP


def test_the_worker_checks_the_adapters_its_graphs_were_built_with(factory):
    from apps.worker import fake_providers
    from apps.worker.main import build_scheduler

    graphs = fake_providers.build_worker_graphs(factory)
    scheduler = build_scheduler(factory, graphs)

    assert fake_providers.FAKE_BOARD in {check.integration for check in graphs.health_checks}
    # Health runs first in a pass, so every other sweep sees what it found.
    assert scheduler.sweeps[0].name == "integration_health"
