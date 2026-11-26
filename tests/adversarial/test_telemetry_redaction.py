"""No token, email body or other personal content reaches a metric label, a log line or a span.

A seeded fake secret (registered with the Phase D redactor, as the credential
broker registers every real one) and a fake recruiter email body are pushed
through every path telemetry is written from, with tracing, metrics and JSON
logging all switched on and captured in memory:

- a chat request whose message is the email body and the secret, run to the
  end by a worker;
- the secret as a bearer token, as a stream `?ticket=`, and as a path segment;
- a model that answers with the email body instead of the schema, so the
  validation error that comes back quotes it;
- that validation error logged, as a model-backed node logs one, described as
  the worker describes a failed step, and recorded on a span;
- the body and the secret recorded directly as metric labels and as an
  integration name, the way a careless call site would.

Then nothing captured -- any label value, any log line, any span name,
attribute, event or status -- may contain the secret or any fragment of the
body. The secret is removed by redaction; the body, which is not a credential
and which redaction does not know about, must never have been written at all.
"""

import io
import logging
import secrets
from datetime import timedelta
from uuid import UUID

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from apps.worker.main import build_worker
from personalos.bootstrap import build_event_outbox, build_workflow_command_queue
from personalos.domain.failures import describe_exception
from personalos.domain.redaction import default_redactor
from personalos.models.structured import invoke_structured
from personalos.observability import metrics, tracing
from personalos.observability.health import record_outcome
from personalos.observability.limits import record_limit_hit
from personalos.observability.metrics import Metric
from personalos.observability.metrics_setup import configure_metrics, reset_metrics
from personalos.observability.state_metrics import StateMetricsSampler
from personalos.observability.structured_logging import configure_logging
from personalos.observability.tracing_setup import configure_tracing, reset_tracing
from personalos.persistence.operational_state import OperationalStateReader
from tests.fixtures import durable_workflow as durable
from tests.fixtures import worker_process
from tests.unit.test_worker_process import _client, _database

#: A credential-shaped value, registered as the broker registers real ones.
SECRET = "ya29." + secrets.token_urlsafe(24)
#: A recruiter's email, as an inbound message or a pasted chat would carry it.
EMAIL_BODY = (
    "Hi Jordan,\n"
    "Thanks for applying to Northwind Traders. Could you do a call on Tuesday at 3pm?\n"
    "My direct line is 555-0142.\n"
    "Best regards,\nPriya Raman <priya.raman@northwind.example>"
)
#: Fragments of the body, any one of which appearing is a leak. Short ones
#: from each line, because a library quoting the body may elide its middle
#: (pydantic shows `'Hi Jordan,\nThanks for a...aman@northwind.example>'`).
BODY_FRAGMENTS = (
    "Hi Jordan",
    "Jordan",
    "northwind",
    "Northwind",
    "Thanks for applying to Northwind",
    "Northwind Traders",
    "call on Tuesday",
    "555-0142",
    "Priya Raman",
    "priya.raman@northwind.example",
)


class _Reply(BaseModel):
    verdict: str
    confidence: float


class _ModelThatQuotesTheEmail:
    """A structured model that answers with the email instead of the schema."""

    def invoke(self, _turns):
        return {"verdict": EMAIL_BODY, "confidence": EMAIL_BODY}


@pytest.fixture
def captured():
    """Tracing, metrics and JSON logs on, all captured in memory."""
    redactor = default_redactor()
    redactor.register_secret(SECRET)
    spans = InMemorySpanExporter()
    configure_tracing(exporter=spans, service_name="test", install_global=False)
    reader = InMemoryMetricReader()
    configure_metrics(reader=reader, install_global=False)
    stream = io.StringIO()
    handler = configure_logging(level="DEBUG", fmt="json", stream=stream)
    try:
        yield spans, reader, stream
    finally:
        logging.getLogger().removeHandler(handler)
        reset_metrics()
        reset_tracing()
        redactor.forget_secrets()


def _span_text(spans: InMemorySpanExporter) -> list[str]:
    texts = []
    for span in spans.get_finished_spans():
        texts.append(span.name)
        texts.extend(f"{k}={v}" for k, v in (span.attributes or {}).items())
        for event in span.events:
            texts.append(event.name)
            texts.extend(f"{k}={v}" for k, v in (event.attributes or {}).items())
        if span.status.description:
            texts.append(span.status.description)
    return texts


def _label_values(reader: InMemoryMetricReader) -> list[str]:
    data = reader.get_metrics_data()
    values = []
    for resource in data.resource_metrics if data else ():
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                for point in metric.data.data_points:
                    values.extend(f"{metric.name} {k}={v}" for k, v in point.attributes.items())
    return values


def _leaks(texts: list[str]) -> list[str]:
    needles = (SECRET, *BODY_FRAGMENTS)
    return [text[:200] for text in texts if any(needle in str(text) for needle in needles)]


async def test_no_secret_or_email_body_reaches_a_metric_label_log_line_or_span(tmp_path, captured):
    spans, reader, stream = captured
    _, factory = _database(tmp_path)
    client = _client(factory)
    state = OperationalStateReader(
        factory,
        outbox=build_event_outbox(factory),
        commands=build_workflow_command_queue(factory),
        stall_after=timedelta(minutes=15),
        approval_grace=timedelta(minutes=5),
    )
    sampler = StateMetricsSampler(state.snapshot).install()
    try:
        # User content as a chat message, run to the end by a worker.
        accepted = client.post("/v1/chat", json={"message": f"{EMAIL_BODY}\n{SECRET}"})
        assert accepted.status_code == 202, accepted.text
        graphs = worker_process.build_graphs(
            factory, durable.EventLog(tmp_path / "events.jsonl"), lease_ttl_seconds=60
        )
        worker = build_worker(factory, graphs, idle_seconds=0.01)
        assert await worker.drain() == 1
        await worker.dispatcher.drain()
        sampler.sample()

        # The secret as a bearer token, a stream ticket and a path segment.
        bare = _client(factory)
        bare.headers.pop("Authorization")
        assert (
            bare.get("/v1/approvals", headers={"Authorization": f"Bearer {SECRET}"}).status_code
            == 401
        )
        workflow_id = UUID(accepted.json()["workflow_id"])
        bare.get(f"/v1/workflows/{workflow_id}/events", params={"ticket": SECRET})
        client.get(f"/v1/workflows/{SECRET}")

        # A model that answers with the email, and the error that quotes it, logged.
        with pytest.raises(ValueError) as raised:
            invoke_structured(
                _ModelThatQuotesTheEmail(),
                [("system", "classify"), ("human", "the message")],
                _Reply,
                span="classify_reply",
                model="test-model",
            )
        logging.getLogger("personalos.models.test").warning(
            "the reply classification was invalid: %s", raised.value, exc_info=raised.value
        )
        # ... and as the worker logs a failed step: a description, pre-formatted.
        logging.getLogger("apps.worker.test").error(
            "thread failed at step(s) classify: %s", describe_exception(raised.value)
        )
        with tracing.span("node.classify") as failing:
            tracing.mark_failed(failing, raised.value)

        # A careless call site putting content straight into labels.
        metrics.add(Metric.API_LIMIT_HITS, route=EMAIL_BODY, limit=SECRET)
        metrics.observe(Metric.MODEL_REQUEST_DURATION, 0.1, provider=SECRET, operation=EMAIL_BODY)
        record_outcome(EMAIL_BODY[:60], previous=None, status="degraded", error=SECRET)
        record_limit_hit(actor_id=SECRET, route=f"/v1/{SECRET}", limit="body_size")
        sampler.sample()
    finally:
        sampler.uninstall()

    labels = _label_values(reader)
    logs = stream.getvalue().splitlines()
    span_text = _span_text(spans)

    assert labels and logs and span_text  # the paths above did write telemetry
    assert _leaks(labels) == []
    assert _leaks(logs) == []
    assert _leaks(span_text) == []


def test_unsafe_label_values_are_replaced_not_recorded():
    spec = Metric.WORKFLOW_STALLED_FOR

    assert metrics.clean_labels(spec, {"workflow_id": EMAIL_BODY, "reason": "no_progress"}) == {
        "workflow_id": metrics.INVALID,
        "reason": "no_progress",
    }
    assert metrics.clean_labels(Metric.API_LIMIT_HITS, {"route": "priya@northwind.example"}) == {
        "route": metrics.OTHER,
        "limit": metrics.UNKNOWN,
    }
    # Undeclared keys are dropped, whatever they hold.
    assert metrics.clean_labels(Metric.WORKFLOWS_ACTIVE, {"body": EMAIL_BODY}) == {}
