"""Trace redaction: no credential leaves the process inside a span.

Spans collect the same hazardous strings logs do -- `span.record_exception`
copies the exception message and the whole traceback into an event, and HTTP
instrumentation attaches request headers as attributes. A finished span is
immutable, so redaction happens at the last point before it leaves: the
exporter. `RedactingSpanExporter` wraps whichever exporter a deployment uses
and hands it redacted copies.
"""

from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status

from personalos.domain.redaction import Redactor, default_redactor


def redact_attributes(attributes: Any, redactor: Redactor | None = None) -> dict[str, Any]:
    """Redact a span's (or event's) attribute mapping."""
    redactor = redactor or default_redactor()
    return dict(redactor.redact(dict(attributes or {})))


def redact_span(span: ReadableSpan, redactor: Redactor | None = None) -> ReadableSpan:
    """Return a copy of a finished span with credentials removed."""
    redactor = redactor or default_redactor()
    status = span.status
    return ReadableSpan(
        name=redactor.redact_text(span.name),
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=redact_attributes(span.attributes, redactor),
        events=[
            Event(
                name=redactor.redact_text(event.name),
                attributes=redact_attributes(event.attributes, redactor),
                timestamp=event.timestamp,
            )
            for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        status=Status(
            status.status_code,
            redactor.redact_text(status.description) if status.description else None,
        ),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class RedactingSpanExporter(SpanExporter):
    """Wraps a span exporter so everything it exports has been redacted."""

    def __init__(self, inner: SpanExporter, redactor: Redactor | None = None):
        """Take the exporter that actually ships spans."""
        self._inner = inner
        self._redactor = redactor

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Redact, then delegate."""
        return self._inner.export([redact_span(span, self._redactor) for span in spans])

    def shutdown(self) -> None:
        """Delegate shutdown."""
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Delegate flush."""
        return self._inner.force_flush(timeout_millis)


__all__ = ["redact_attributes", "redact_span", "RedactingSpanExporter"]
