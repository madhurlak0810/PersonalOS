"""A fixture workflow records what every dashboard panel queries, in-process.

The compose smoke test (`scripts/smoke_observability.py`) checks each panel
against a live Prometheus. This is the same check without Docker, on every
test run: a workflow is started through the API and run by a worker against
one SQLite database, with an in-memory metric reader installed, and then every
non-alert panel in `scripts/build_dashboards.py` must find at least one of the
series it queries among what was recorded.

It also covers the stalled-workflow gauges, which the alert is built on, in
both directions:

- a command queued with no worker to run it is reported stalled
  (`no_progress`) once the stall threshold has passed, and stops being
  reported once a worker has run it;
- a run parked at an approval is *not* stalled while the approval is open,
  becomes stalled (`approval_overdue`) once its expiry and the grace period
  have passed without the expiry sweep, and stops when the sweep closes it --
  which also times the approval's wait.
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from apps.worker.main import WorkerGraphs, build_worker, prime_worker_metrics
from personalos.bootstrap import build_event_outbox, build_workflow_command_queue
from personalos.domain.checkpoints import ConditionKind
from personalos.domain.job_search import ActionKind, ApprovalRequest
from personalos.domain.operational_state import StallReason
from personalos.models.routing import KeywordIntentClassifier
from personalos.observability import metrics
from personalos.observability.health import record_outcome
from personalos.observability.metrics_setup import configure_metrics, reset_metrics
from personalos.observability.state_metrics import StateMetricsSampler
from personalos.persistence.operational_state import OperationalStateReader
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures import worker_process
from tests.graph_scenarios.test_pending_checkpoints import AT_TRIGGER, Deployment
from tests.unit.test_worker_process import Clock, _chat, _client, _database, _status

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import build_dashboards  # noqa: E402

STALL_AFTER = timedelta(minutes=15)
GRACE = timedelta(minutes=5)

#: Panels the chat workflow cannot fill, and the test here that does. The
#: fixture's approval is granted by a fake gate inside the graph, so no
#: approval is waited on; the expiry test below waits on one.
_FILLED_ELSEWHERE = {
    "Approval wait (p50, p95)": "test_an_approval_past_its_expiry_is_stalled_until_the_sweep_closes_it",
}


@pytest.fixture
def reader() -> InMemoryMetricReader:
    reader = InMemoryMetricReader()
    configure_metrics(reader=reader, install_global=False)
    try:
        yield reader
    finally:
        reset_metrics()


def collected(reader: InMemoryMetricReader) -> dict[str, list[dict]]:
    """Every metric recorded so far, by OpenTelemetry name, with each point's labels."""
    data = reader.get_metrics_data()
    found: dict[str, list[dict]] = {}
    for resource in data.resource_metrics if data else ():
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                found.setdefault(metric.name, []).extend(
                    dict(point.attributes) for point in metric.data.data_points
                )
    return found


def _state_sampler(factory, now) -> tuple[StateMetricsSampler, OperationalStateReader]:
    state = OperationalStateReader(
        factory,
        outbox=build_event_outbox(factory),
        commands=build_workflow_command_queue(factory),
        stall_after=STALL_AFTER,
        approval_grace=GRACE,
        clock=lambda: now(),
    )
    return StateMetricsSampler(state.snapshot).install(), state


_SERIES = re.compile(r"\b(personalos_[a-z0-9_]+)\b")


def _series_to_metric() -> dict[str, str]:
    return {series: spec.name for spec in metrics.CATALOG for series in spec.prometheus_series}


async def test_a_fixture_workflow_records_a_series_for_every_dashboard_panel(tmp_path, reader):
    db_path, factory = _database(tmp_path)
    client = _client(factory)
    times = {"now": datetime.utcnow()}
    sampler, _ = _state_sampler(factory, lambda: times["now"])
    try:
        accepted = _chat(client)
        # An oversized body: refused by the body cap, a Phase L2 limit hit.
        assert client.post("/v1/chat", content=b"x" * (2 * 1024 * 1024)).status_code == 413

        # No worker yet. Past the stall threshold the queued command is stalled.
        times["now"] = datetime.utcnow() + STALL_AFTER + timedelta(minutes=1)
        [stalled_workflow] = sampler.sample().stalled
        assert str(stalled_workflow.workflow_id) == accepted["workflow_id"]
        assert stalled_workflow.reason is StallReason.NO_PROGRESS
        stalled = collected(reader)["personalos.workflow.stalled_for"]
        assert stalled == [{"workflow_id": accepted["workflow_id"], "reason": "no_progress"}]

        log = durable.EventLog(tmp_path / "events.jsonl")
        graphs = worker_process.build_graphs(factory, log, lease_ttl_seconds=60)
        prime_worker_metrics(graphs)
        worker = build_worker(factory, graphs, idle_seconds=0.01)
        assert await worker.drain() == 1
        await worker.dispatcher.drain()
        await worker.scheduler.run_once()
        assert _status(client, accepted["workflow_id"])["status"] == "completed"

        # The workflow moved and finished: no longer stalled, at any clock.
        snapshot = sampler.sample()
        assert snapshot.stalled == ()
        assert "personalos.workflow.stalled_for" not in collected(reader)

        # The deterministic `rules` model gateway, which the quick start runs on,
        # and an integration health check, as the worker's health sweep records.
        KeywordIntentClassifier().classify("find me a python job")
        record_outcome("embedder", previous=None, status="healthy")

        recorded = collected(reader)
    finally:
        sampler.uninstall()

    to_metric = _series_to_metric()
    empty = set()
    for _uid, title, expr in build_dashboards.panel_queries():
        if title in build_dashboards.ALERT_ONLY_PANELS:
            continue
        names = {to_metric[s] for s in _SERIES.findall(expr) if s in to_metric}
        if not names & recorded.keys():
            empty.add(title)
    assert empty == set(_FILLED_ELSEWHERE)

    steps = recorded["personalos.workflow.step.duration"]
    assert {point["step_class"] for point in steps} == {"quick", "long_running"}
    assert {point["graph"] for point in steps} >= {"supervisor", "job_search"}
    assert {"route": "/v1/chat", "method": "POST", "status_class": "2xx"} in recorded[
        "personalos.api.request.duration"
    ]
    assert {"graph": "supervisor", "outcome": "completed"} in recorded["personalos.workflow.runs"]
    assert {"provider": "rules", "operation": "classify_intent", "outcome": "ok"} in recorded[
        "personalos.model.request.duration"
    ]


async def test_an_approval_past_its_expiry_is_stalled_until_the_sweep_closes_it(tmp_path, reader):
    evaluator = fakes.ScriptedConditionEvaluator({ConditionKind.RECRUITER_RESPONSE_RECEIVED: False})
    clock = Clock()
    deployment = Deployment(tmp_path, evaluator=evaluator, clock=clock)
    await deployment.apply()
    graphs = WorkerGraphs(runners={"job_search": deployment.runner}, evaluator=evaluator)
    clock.now = AT_TRIGGER + timedelta(days=1)
    worker = build_worker(deployment.factory, graphs, idle_seconds=0.01, clock=clock)
    await worker.scheduler.run_once()  # fires the follow-up, which parks at an approval
    state = await deployment.runner.inspect(deployment.thread)
    [request] = [
        ApprovalRequest.model_validate(raw)
        for raw in state.values["approval_requests"]
        if raw["kind"] == ActionKind.SEND_RECRUITER_MESSAGE.value
    ]
    sampler, _ = _state_sampler(deployment.factory, lambda: clock.now)
    try:
        # Waiting on a person inside the approval's window, however long: not a stall.
        clock.now = request.expires_at - timedelta(minutes=1)
        assert sampler.sample().stalled == ()

        # Past the expiry and the grace period with no sweep: stalled.
        clock.now = request.expires_at + GRACE + timedelta(minutes=1)
        [stalled] = sampler.sample().stalled
        assert stalled.workflow_id == deployment.thread.workflow_id
        assert stalled.reason is StallReason.APPROVAL_OVERDUE
        overdue = (GRACE + timedelta(minutes=1)).total_seconds()
        assert stalled.stalled_for_seconds == pytest.approx(overdue)
        assert collected(reader)["personalos.workflows.stalled"] == [
            {"reason": "no_progress"},
            {"reason": "approval_overdue"},
        ]

        # The sweep closes it and the run moves on: the stall clears.
        assert (await worker.scheduler.run_once())["approval_expiry"] == [request.request_id]
        await worker.drain()
        assert sampler.sample().stalled == ()
    finally:
        sampler.uninstall()

    waits = collected(reader)["personalos.approval.wait.duration"]
    assert {"outcome": "EXPIRED", "action_kind": "send_recruiter_message"} in waits


def test_the_operational_gauges_report_zeros_rather_than_nothing(tmp_path, reader):
    """Queue depth, backlog, parked and stalled counts are series from the first sample."""
    _, factory = _database(tmp_path)
    sampler, _ = _state_sampler(factory, datetime.utcnow)
    try:
        sampler.sample()
        recorded = collected(reader)
    finally:
        sampler.uninstall()

    assert recorded["personalos.worker.queue.depth"] == [
        {"state": "pending"},
        {"state": "in_progress"},
    ]
    assert recorded["personalos.outbox.parked"] == [
        {"status": "dead_letter"},
        {"status": "quarantined"},
    ]
    assert {"reason": "no_progress"} in recorded["personalos.workflows.stalled"]
    assert recorded["personalos.workflows.active"] == [{}]
    assert recorded["personalos.outbox.oldest_pending.age"] == [{}]


def test_a_failed_sample_empties_the_gauges_instead_of_reporting_stale_state(reader):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise RuntimeError("database unavailable")

    sampler = StateMetricsSampler(flaky).install()
    try:
        assert sampler.sample() is None
        assert "personalos.workflows.active" not in collected(reader)
    finally:
        sampler.uninstall()
