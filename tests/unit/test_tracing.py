"""Tests for the tracing API: workflow-keyed traces, propagation, and what a span may carry.

`tests/graph_scenarios/test_workflow_tracing.py` checks the end-to-end trace of
a real workflow. This file pins down the pieces it is built from:

- a workflow's spans land in the trace named by its id, under its root, from
  any process -- and a foreign trace that was current is linked, not lost;
- an event published inside a span is delivered inside a child of it, across
  the outbox;
- a span carries ids and labels, never a payload;
- tracing switched off changes nothing but the absence of spans.
"""

import asyncio
from datetime import datetime
from enum import Enum
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from apps.worker.outbox_dispatcher import OutboxDispatcher
from personalos.bootstrap import build_event_outbox
from personalos.domain.events import EventEnvelope
from personalos.observability import tracing
from personalos.observability.tracing import Attr
from personalos.observability.tracing_setup import configure_tracing, reset_tracing
from tests.fixtures.durable_workflow import session_factory


@pytest.fixture
def spans() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, service_name="test", install_global=False)
    try:
        yield exporter
    finally:
        reset_tracing()


class _Colour(Enum):
    RED = "red"


def _by_name(exporter: InMemorySpanExporter) -> dict:
    return {span.name: span for span in exporter.get_finished_spans()}


# --- Attributes -------------------------------------------------------------------


def test_clean_attributes_keeps_ids_and_labels_and_drops_payloads():
    workflow_id = uuid4()
    cleaned = tracing.clean_attributes(
        {
            "id": workflow_id,
            "colour": _Colour.RED,
            "count": 3,
            "flag": False,
            "when": datetime(2026, 1, 2, 3, 4, 5),
            "ids": [uuid4(), uuid4()],
            "none": None,
            "payload": {"body": "Dear hiring manager"},
            "model": EventEnvelope(
                type="t", source="s", occurred_at=datetime(2026, 1, 1), dedupe_key="k"
            ),
            "mixed_objects": [{"a": 1}],
        }
    )
    assert cleaned["id"] == str(workflow_id)
    assert cleaned["colour"] == "red"
    assert cleaned["count"] == 3 and cleaned["flag"] is False
    assert cleaned["when"] == "2026-01-02T03:04:05"
    assert len(cleaned["ids"]) == 2 and all(isinstance(i, str) for i in cleaned["ids"])
    assert {"none", "payload", "model", "mixed_objects"}.isdisjoint(cleaned)


# --- Workflow traces --------------------------------------------------------------


def test_a_workflow_root_span_takes_the_workflows_derived_ids(spans):
    workflow_id = uuid4()
    with tracing.workflow_span("api.chat", workflow_id=workflow_id, root=True):
        pass

    root = _by_name(spans)["api.chat"]
    assert root.parent is None
    assert root.context.trace_id == workflow_id.int
    assert root.context.span_id == tracing.workflow_root_span_id(workflow_id)
    assert tracing.format_trace_id(workflow_id) == workflow_id.hex


def test_a_later_span_with_no_context_joins_the_workflows_root(spans):
    """What an approval request or a recovered worker does: it knows only the workflow id."""
    workflow_id = uuid4()
    with tracing.workflow_span("api.chat", workflow_id=workflow_id, root=True):
        pass
    with tracing.workflow_span("approval.decide", workflow_id=workflow_id, actor_id="user:r"):
        with tracing.span("policy.evaluate"):
            pass

    named = _by_name(spans)
    decide, policy = named["approval.decide"], named["policy.evaluate"]
    assert decide.context.trace_id == workflow_id.int
    assert decide.parent.span_id == named["api.chat"].context.span_id
    assert policy.parent.span_id == decide.context.span_id
    # Scope ids reach the child through baggage.
    assert policy.attributes[Attr.WORKFLOW_ID] == str(workflow_id)
    assert policy.attributes[Attr.ACTOR_ID] == "user:r"


def test_a_foreign_trace_that_was_current_is_linked_not_joined(spans):
    workflow_id = uuid4()
    caller = SpanContext(
        trace_id=0xABC, span_id=0xDEF, is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED)
    )
    with trace.use_span(NonRecordingSpan(caller)):
        with tracing.workflow_span("api.workflow.resume", workflow_id=workflow_id):
            pass

    (resume,) = spans.get_finished_spans()
    assert resume.context.trace_id == workflow_id.int
    assert [link.context.trace_id for link in resume.links] == [0xABC]


def test_a_carrier_in_the_workflows_trace_is_the_parent(spans):
    """The worker's side of the outbox hand-off."""
    workflow_id = uuid4()
    with tracing.workflow_span("api.chat", workflow_id=workflow_id, root=True):
        with tracing.span("queue"):
            carrier = tracing.inject_trace_context()

    with tracing.workflow_span("worker.command.start", workflow_id=workflow_id, carrier=carrier):
        pass

    named = _by_name(spans)
    assert named["worker.command.start"].parent.span_id == named["queue"].context.span_id
    assert "baggage" in carrier and str(workflow_id) in carrier["baggage"]


def test_an_event_is_delivered_in_a_child_of_the_span_that_published_it(spans, tmp_path):
    factory = session_factory(tmp_path / "events.db")
    outbox = build_event_outbox(factory)
    workflow_id = uuid4()
    delivered: list[EventEnvelope] = []

    async def handler(envelope: EventEnvelope) -> None:
        delivered.append(envelope)
        with tracing.span("handler"):
            pass

    with tracing.workflow_span("producer", workflow_id=workflow_id, root=True):
        outbox.publish(
            EventEnvelope(
                type="application.created",
                source="test",
                occurred_at=datetime.utcnow(),
                dedupe_key=f"created:{uuid4()}",
            )
        )
    assert asyncio.run(OutboxDispatcher(outbox, {}, default=handler).process_next())

    named = _by_name(spans)
    publish, deliver = named["event.publish"], named["event.deliver"]
    assert delivered and delivered[0].trace_context["traceparent"]
    assert deliver.context.trace_id == workflow_id.int
    assert deliver.parent.span_id == publish.context.span_id
    assert named["handler"].parent.span_id == deliver.context.span_id
    assert deliver.attributes[Attr.EVENT_OUTCOME] == "acked"
    assert deliver.attributes[Attr.WORKFLOW_ID] == str(workflow_id)


def test_an_interrupted_node_is_marked_interrupted_not_failed(spans):
    from langgraph.errors import GraphInterrupt

    def parks(state):
        raise GraphInterrupt(())

    node = tracing.traced_node("approval_checkpoint", parks)
    with pytest.raises(GraphInterrupt):
        node({})

    (span,) = spans.get_finished_spans()
    assert span.name == "node.approval_checkpoint"
    assert span.attributes[Attr.INTERRUPTED] is True
    assert span.status.is_ok
    assert not span.events


def test_a_failed_node_records_the_failure(spans):
    async def fails(state):
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        asyncio.run(tracing.traced_node("search_providers", fails)({}))

    (span,) = spans.get_finished_spans()
    assert not span.status.is_ok
    assert [event.name for event in span.events] == ["exception"]


# --- Disabled ---------------------------------------------------------------------


def test_disabled_tracing_records_nothing_and_injects_nothing():
    assert configure_tracing(enabled=False) is None
    assert not tracing.enabled()
    with tracing.workflow_span("api.chat", workflow_id=uuid4(), root=True) as span:
        assert not span.is_recording()
        assert tracing.inject_trace_context() == {}


def test_settings_switch_tracing_without_code_changes(monkeypatch):
    from personalos.config import Settings
    from personalos.observability.tracing_setup import configure_tracing_from_settings

    monkeypatch.setenv("TRACING_ENABLED", "false")
    assert configure_tracing_from_settings("svc", Settings()) is None

    monkeypatch.setenv("TRACING_ENABLED", "true")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "personalos-test")
    monkeypatch.setattr(trace, "set_tracer_provider", lambda provider: None)
    try:
        provider = configure_tracing_from_settings("svc", Settings())
        assert provider is not None
        assert provider.resource.attributes["service.name"] == "personalos-test"
        assert tracing.enabled()
    finally:
        reset_tracing()
