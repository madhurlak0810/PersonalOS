"""Installing tracing: the SDK provider, where spans go, and the redaction in front of it.

The composition root calls `configure_tracing_from_settings` once at process
start (the API's `create_app`, the worker's `main`). What it installs is
decided by configuration alone, so tracing is switched on, pointed elsewhere,
or switched off without a code change:

- `TRACING_ENABLED=false` (the default) installs nothing. Every span the code
  opens is then OpenTelemetry's no-op span, and no trace context is written
  into commands or events.
- `OTEL_EXPORTER_OTLP_ENDPOINT` names the OTLP/HTTP collector spans are sent
  to (Jaeger, Tempo, an OpenTelemetry Collector, ...). Left empty with tracing
  on, spans are created -- and so propagated across the outbox -- but exported
  nowhere, which is what a test or a dry run wants.
- `OTEL_SERVICE_NAME` names the process in the backend; the API and the worker
  each default to their own.

Whatever the exporter, it is wrapped in `RedactingSpanExporter`, so no span
leaves the process with a credential in it -- the Phase D guarantee, applied
here once rather than at each call site.
"""

import logging
from typing import Any

from opentelemetry import baggage, trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from personalos.config import Settings, settings
from personalos.domain.redaction import Redactor
from personalos.observability import tracing
from personalos.observability.trace_redaction import RedactingSpanExporter

logger = logging.getLogger(__name__)


class WorkflowIdGenerator(RandomIdGenerator):
    """Random ids, except for a workflow's root span, which takes the ids it is derived to.

    That one exception is what lets a process that knows only a workflow id
    parent a span on the workflow's root (`tracing.workflow_root_context`).
    """

    #: Read by `tracing.workflow_span` before asking for a forced root.
    supports_forced_ids = True

    def generate_trace_id(self) -> int:
        """The forced trace id while a workflow root is being created; random otherwise."""
        forced = tracing.forced_ids()
        return forced[0] if forced else super().generate_trace_id()

    def generate_span_id(self) -> int:
        """The forced span id while a workflow root is being created; random otherwise."""
        forced = tracing.forced_ids()
        return forced[1] if forced else super().generate_span_id()


class ScopeAttributesSpanProcessor(SpanProcessor):
    """Stamps every span with the workflow scope ids in its parent context's baggage.

    So a span opened by code that knows nothing about workflows -- a policy
    evaluation, a ledger write -- still says which workflow, thread and request
    it served.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Copy scope baggage onto the span, without overwriting what it set itself."""
        present = span.attributes or {}
        for key in tracing.SCOPE_ATTRIBUTES:
            if key in present:
                continue
            value = baggage.get_baggage(key, parent_context)
            if value is not None:
                span.set_attribute(key, str(value))

    def on_end(self, span: ReadableSpan) -> None:
        """Nothing to do once a span ends."""


def _otlp_exporter(endpoint: str) -> SpanExporter | None:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-http "
            "is not installed; spans will not be exported"
        )
        return None
    if not endpoint.rstrip("/").endswith("/v1/traces"):
        endpoint = f"{endpoint.rstrip('/')}/v1/traces"
    exporter: SpanExporter = OTLPSpanExporter(endpoint=endpoint)
    return exporter


def configure_tracing(
    *,
    enabled: bool = True,
    service_name: str = "personalos",
    otlp_endpoint: str | None = None,
    exporter: SpanExporter | None = None,
    redactor: Redactor | None = None,
    install_global: bool = True,
    resource_attributes: dict[str, Any] | None = None,
) -> TracerProvider | None:
    """Install a tracer provider for PersonalOS spans; `None`, and nothing installed, if disabled.

    `exporter` is for tests (an in-memory exporter, exported synchronously);
    otherwise `otlp_endpoint` gets a batching OTLP exporter. Either way it is
    wrapped in `RedactingSpanExporter`. `install_global` also makes the
    provider OpenTelemetry's global one, which can happen once per process;
    a test passes `False` and calls `reset_tracing` after.

    Disabled is a no-op: it does not tear down a provider already installed.
    """
    if not enabled:
        return None

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, **(resource_attributes or {})}),
        id_generator=WorkflowIdGenerator(),
    )
    provider.add_span_processor(ScopeAttributesSpanProcessor())
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exporter, redactor)))
    elif otlp_endpoint:
        otlp = _otlp_exporter(otlp_endpoint)
        if otlp is not None:
            provider.add_span_processor(BatchSpanProcessor(RedactingSpanExporter(otlp, redactor)))

    tracing.set_tracer_provider(provider)
    if install_global:
        trace.set_tracer_provider(provider)
    logger.info(
        "tracing enabled for %s (exporter: %s)",
        service_name,
        "custom" if exporter is not None else (otlp_endpoint or "none"),
    )
    return provider


def configure_tracing_from_settings(
    service_name: str, config: Settings = settings
) -> TracerProvider | None:
    """Install tracing as the environment says: on or off, which endpoint, which name."""
    return configure_tracing(
        enabled=config.tracing_enabled,
        service_name=config.otel_service_name or service_name,
        otlp_endpoint=config.otel_exporter_otlp_endpoint or None,
        resource_attributes={"deployment.environment": config.app_env},
    )


def reset_tracing() -> None:
    """Stop creating PersonalOS spans from a configured provider."""
    provider = tracing.tracer_provider()
    tracing.set_tracer_provider(None)
    if isinstance(provider, TracerProvider):
        provider.shutdown()


__all__ = [
    "ScopeAttributesSpanProcessor",
    "WorkflowIdGenerator",
    "configure_tracing",
    "configure_tracing_from_settings",
    "reset_tracing",
]
