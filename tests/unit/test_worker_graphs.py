"""The graphs a worker runs out of the box: `apps.worker.graphs.build_worker_graphs`.

The acceptance criteria this file exists for:

1. **The factory compiles durable graphs.**
   `test_the_default_factory_compiles_both_graphs_on_the_durable_checkpointer`
   checks both runners' graphs checkpoint to the database, not to memory.
2. **With nothing configured but the database, a queued job search runs.**
   `test_a_chat_queued_job_search_runs_to_completion_with_only_a_database`
   queues a conversation through `POST /v1/chat`, runs the real worker loop
   over the default graphs, and reads the workflow back as completed. With no
   job board configured the search finds nothing -- and says why.
3. **A step that needs an unbound port fails recoverably.**
   `test_a_step_that_needs_an_unbound_port_is_a_retryable_failure_not_a_crash`
   ingests a resume and gets a posting onto the shortlist, so the packet step
   retrieves evidence and asks the draft writer, which is not configured: the
   worker loop survives, and the step is listed on `/failures` as retryable,
   naming the port.
"""

import asyncio
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker import fake_providers
from apps.worker.graphs import build_worker_graphs
from apps.worker.main import WorkerGraphs, build_worker, load_graph_factory, log_wired_graphs
from apps.worker.unconfigured import (
    APPLICATION_SUBMITTER_PORT,
    DRAFT_WRITER_PORT,
    EMBEDDER_PORT,
    JOB_BOARDS_PORT,
    MAILBOX_PORT,
    NotConfiguredCalendar,
    NotConfiguredEmbedder,
    NotConfiguredMailbox,
)
from personalos import bootstrap
from personalos.config import Settings
from personalos.domain.artifacts import validate_draft
from personalos.domain.errors import FailureClass, PortNotConfigured
from personalos.domain.failures import classify, parse_stored_error
from personalos.domain.models import ArtifactType, OutboxEventStatus
from personalos.domain.workflow import JOB_SEARCH_THREAD_NAMESPACE, SUPERVISOR_THREAD_NAMESPACE
from personalos.persistence.checkpointer import SqlAlchemyCheckpointSaver
from personalos.persistence.models import Base, CandidateProfileModel, OutboxEventModel
from personalos.persistence.search_profiles import SqlSearchProfileStore
from personalos.persistence.workflow_commands import WORKFLOW_COMMAND_EVENT
from tests.fixtures import artifact_prep, evidence_ingestion
from tests.fixtures import job_search_fakes as fakes

DEFAULT_FACTORY = "apps.worker.graphs:build_worker_graphs"


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def factory(tmp_path: Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{tmp_path / 'worker.db'}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _client(factory) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    app.dependency_overrides[get_failure_services] = lambda: build_failure_services(factory)
    return TestClient(app)


def _chat(client: TestClient) -> dict:
    response = client.post(
        "/v1/chat", json={"message": "find me a python job", "user_id": str(fakes.USER_ID)}
    )
    assert response.status_code == 202, response.text
    return response.json()


def _commands_settled(factory) -> bool:
    session = factory()
    try:
        rows = (
            session.query(OutboxEventModel)
            .filter(OutboxEventModel.type == WORKFLOW_COMMAND_EVENT)
            .all()
        )
        unsettled = {OutboxEventStatus.PENDING.value, OutboxEventStatus.IN_PROGRESS.value}
        return bool(rows) and all(row.status not in unsettled for row in rows)
    finally:
        session.close()


def _run_worker_until_settled(factory, graphs: WorkerGraphs) -> None:
    """Run the real worker loop until every queued command has been settled."""
    worker = build_worker(factory, graphs, idle_seconds=0.01)

    async def run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        for _ in range(1000):
            if _commands_settled(factory) or task.done():
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(run())


# --- The factory ------------------------------------------------------------------


def test_the_default_factory_compiles_both_graphs_on_the_durable_checkpointer(factory):
    graphs = build_worker_graphs(factory)

    assert set(graphs.runners) == {SUPERVISOR_THREAD_NAMESPACE, JOB_SEARCH_THREAD_NAMESPACE}
    for runner in graphs.runners.values():
        assert isinstance(runner.graph.checkpointer, SqlAlchemyCheckpointSaver)
        assert not isinstance(runner.graph.checkpointer, InMemorySaver)
    # One saver for both, so a workflow's threads are all found by its id.
    supervisor, job_search = (
        graphs.runners[SUPERVISOR_THREAD_NAMESPACE],
        graphs.runners[JOB_SEARCH_THREAD_NAMESPACE],
    )
    assert supervisor.graph.checkpointer is job_search.graph.checkpointer
    assert graphs.follow_ups is job_search


def test_the_default_factory_names_every_port_it_bound_to_a_stand_in(factory):
    graphs = build_worker_graphs(factory)

    assert {JOB_BOARDS_PORT, MAILBOX_PORT, "calendar", DRAFT_WRITER_PORT, "reply_drafter"} <= set(
        graphs.unbound_ports
    )
    # The local default embedder is bound unless EMBEDDING_PROVIDER is cleared.
    assert EMBEDDER_PORT not in graphs.unbound_ports


def test_with_no_embedding_provider_the_embedder_is_a_stand_in(factory, monkeypatch):
    monkeypatch.setattr(bootstrap.settings, "embedding_provider", "")

    assert EMBEDDER_PORT in build_worker_graphs(factory).unbound_ports


def test_an_override_drops_the_port_it_replaces_from_the_unbound_list(factory):
    graphs = build_worker_graphs(factory, overrides={"providers": [fakes.FakeProvider()]})

    assert JOB_BOARDS_PORT not in graphs.unbound_ports
    assert MAILBOX_PORT in graphs.unbound_ports


def test_the_fake_provider_factory_binds_the_job_board_and_draft_writer(factory):
    # What the quick start and its smoke test run: everything else unbound,
    # so the walkthrough ends where the guide says it does.
    graphs = fake_providers.build_worker_graphs(factory)

    assert set(graphs.runners) == {SUPERVISOR_THREAD_NAMESPACE, JOB_SEARCH_THREAD_NAMESPACE}
    assert JOB_BOARDS_PORT not in graphs.unbound_ports
    assert DRAFT_WRITER_PORT not in graphs.unbound_ports
    assert APPLICATION_SUBMITTER_PORT in graphs.unbound_ports


@pytest.mark.parametrize("artifact_type", [ArtifactType.RESUME, ArtifactType.COVER_LETTER])
def test_extractive_drafts_restate_their_evidence_and_pass_validation(artifact_type):
    evidence = artifact_prep.records()

    proposal = asyncio.run(
        fake_providers.ExtractiveDraftWriter().write(
            artifact_type, artifact_prep.posting(), fakes.profile(), evidence
        )
    )

    assert [segment.text for segment in proposal.segments] == [r.text for r in evidence]
    assert validate_draft(proposal, evidence).ok


def test_the_setting_defaults_to_the_shipped_factory_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("WORKER_GRAPHS", raising=False)
    assert Settings(_env_file=None).worker_graphs == DEFAULT_FACTORY
    assert load_graph_factory(DEFAULT_FACTORY) is build_worker_graphs

    monkeypatch.setenv("WORKER_GRAPHS", "tests.fixtures.worker_process:build_graphs")
    spec = Settings(_env_file=None).worker_graphs
    from tests.fixtures.worker_process import build_graphs

    assert load_graph_factory(spec) is build_graphs


def test_startup_logs_the_wired_graphs_and_the_unbound_ports(factory, caplog):
    graphs = build_worker_graphs(factory)

    with caplog.at_level("INFO", logger="apps.worker.main"):
        log_wired_graphs(DEFAULT_FACTORY, graphs)

    assert "wired graphs from 'apps.worker.graphs:build_worker_graphs': job_search, supervisor" in (
        caplog.text
    )
    assert "unbound ports (steps that need them fail with a retryable PortNotConfigured): " in (
        caplog.text
    )
    assert DRAFT_WRITER_PORT in caplog.text and MAILBOX_PORT in caplog.text


# --- Stand-ins --------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: NotConfiguredEmbedder().embed("text"),
        lambda: NotConfiguredMailbox().fetch(fakes.USER_ID),
        lambda: NotConfiguredCalendar().get_event("evt-1"),
    ],
)
def test_a_stand_in_refuses_with_a_retryable_failure_naming_its_port(call):
    with pytest.raises(PortNotConfigured) as raised:
        asyncio.run(call())

    stored = parse_stored_error(repr(raised.value))
    assert classify(stored) == FailureClass.PROVIDER_TRANSIENT
    assert raised.value.retryable is False  # never retried automatically
    assert raised.value.port in raised.value.message


# --- End to end -------------------------------------------------------------------


def test_a_chat_queued_job_search_runs_to_completion_with_only_a_database(factory, caplog):
    client = _client(factory)
    queued = _chat(client)

    with caplog.at_level("WARNING"):
        _run_worker_until_settled(factory, build_worker_graphs(factory))

    status = client.get(f"/v1/workflows/{queued['workflow_id']}").json()
    assert status["status"] == "completed", status
    kinds = {thread["kind"] for thread in status["threads"]}
    assert kinds == {SUPERVISOR_THREAD_NAMESPACE, JOB_SEARCH_THREAD_NAMESPACE}
    # Nothing to search is not a silent empty result: the run says why.
    assert "GREENHOUSE_BOARDS" in caplog.text


def test_a_step_that_needs_an_unbound_port_is_a_retryable_failure_not_a_crash(factory, tmp_path):
    root = evidence_ingestion.allowed_root(tmp_path)
    asyncio.run(evidence_ingestion.ingestor(root, factory).ingest(fakes.USER_ID, "resume.md"))
    client = _client(factory)
    queued = _chat(client)
    graphs = build_worker_graphs(
        factory,
        overrides={
            "profile_store": fakes.FakeProfileStore(),
            "providers": [fakes.FakeProvider()],
            "scorer": fakes.FakeScorer(),
            "evidence_checker": fakes.FakeEvidenceChecker(),
        },
    )

    _run_worker_until_settled(factory, graphs)

    response = client.get(f"/v1/workflows/{queued['workflow_id']}/failures")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    failures = [
        f for f in body["failures"] if f["thread_id"].startswith(JOB_SEARCH_THREAD_NAMESPACE)
    ]
    assert len(failures) == 1, body
    failure = failures[0]
    assert failure["step"] == "prepare_application_packet"
    assert failure["error_type"] == "PortNotConfigured"
    assert failure["failure_class"] == FailureClass.PROVIDER_TRANSIENT.value
    assert failure["retryable"] is True
    assert "retry" in failure["actions"]
    assert DRAFT_WRITER_PORT in failure["last_error"]


# --- Adapters ---------------------------------------------------------------------


def test_the_profile_store_reads_the_latest_version_and_its_preferences(factory):
    session = factory()
    for version, roles in ((1, ["Analyst"]), (2, ["Backend Engineer"])):
        session.add(
            CandidateProfileModel(
                user_id=fakes.USER_ID,
                profile_version=version,
                target_roles=roles,
                target_locations=["Remote"],
                preferences={"keywords": ["python"], "remote_only": True, "unknown_key": 1},
            )
        )
    session.commit()
    session.close()

    profile = asyncio.run(SqlSearchProfileStore(factory).load(fakes.USER_ID))

    assert profile.profile_version == 2
    assert profile.target_roles == ("Backend Engineer",)
    assert profile.keywords == ("python",)
    assert profile.remote_only is True


def test_the_profile_store_falls_back_to_an_empty_profile(factory):
    user_id = UUID("22222222-2222-2222-2222-222222222222")

    profile = asyncio.run(SqlSearchProfileStore(factory).load(user_id))

    assert profile.user_id == user_id
    assert profile.target_roles == ()
