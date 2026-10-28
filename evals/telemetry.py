"""Spans, from a harness run or from a deployment, in one shape the metrics read.

The latency, tool-call validity and recovery metrics are computed from spans,
not from anything a suite reports about itself: a node's duration is what its
`node_span` measured, a model attempt's validity is the `schema_valid`
attribute `personalos.models.structured` set on its span. Two sources feed
them:

- **A harness run.** `capture()` installs an in-memory exporter for the length
  of the run, so every span the compiled graphs emit is kept.
- **Runtime telemetry.** `load_otlp_json()` reads what an OpenTelemetry
  Collector's `file` exporter writes (OTLP/JSON, one `ExportTraceServiceRequest`
  per line, or a single document), so the same metrics can be computed over a
  day of real workflows.

Either way the result is a list of `SpanRecord`s.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from personalos.observability.tracing_setup import configure_tracing, reset_tracing


@dataclass(frozen=True)
class SpanRecord:
    name: str
    start_ns: int
    end_ns: int
    ok: bool
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)


def from_readable(span: ReadableSpan) -> SpanRecord:
    return SpanRecord(
        name=span.name,
        start_ns=span.start_time or 0,
        end_ns=span.end_time or span.start_time or 0,
        ok=span.status.status_code is not StatusCode.ERROR,
        attributes=dict(span.attributes or {}),
    )


class SpanCapture:
    """What `capture()` yields: the spans ended so far."""

    def __init__(self, exporter: InMemorySpanExporter):
        self._exporter = exporter

    def spans(self) -> list[SpanRecord]:
        return [from_readable(span) for span in self._exporter.get_finished_spans()]


@contextmanager
def capture() -> Iterator[SpanCapture]:
    """Record every PersonalOS span ended inside the block."""
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, service_name="personalos-evals", install_global=False)
    try:
        yield SpanCapture(exporter)
    finally:
        reset_tracing()


# --- OTLP/JSON ----------------------------------------------------------------


def _any_value(value: dict[str, Any]) -> Any:
    """Decode an OTLP `AnyValue`. Integers arrive as strings, per the JSON mapping."""
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [_any_value(item) for item in value["arrayValue"].get("values", [])]
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    return None


def _otlp_span(raw: dict[str, Any]) -> SpanRecord:
    status = (raw.get("status") or {}).get("code")
    return SpanRecord(
        name=raw["name"],
        start_ns=int(raw.get("startTimeUnixNano", 0)),
        end_ns=int(raw.get("endTimeUnixNano", 0)),
        # 2 is STATUS_CODE_ERROR; a collector may write the enum by name.
        ok=status not in (2, "STATUS_CODE_ERROR"),
        attributes={item["key"]: _any_value(item["value"]) for item in raw.get("attributes", [])},
    )


def _spans_in(request: dict[str, Any]) -> Iterator[SpanRecord]:
    for resource in request.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for raw in scope.get("spans", []):
                yield _otlp_span(raw)


def load_otlp_json(path: str | Path) -> list[SpanRecord]:
    """Every span in an OTLP/JSON file: a single document or one request per line."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        requests = [json.loads(text)]
    except json.JSONDecodeError:
        requests = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [span for request in requests for span in _spans_in(request)]


def _encode_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_encode_value(item) for item in value]}}
    return {"stringValue": str(value)}


def write_otlp_json(path: str | Path, spans: list[SpanRecord]) -> None:
    """Write `spans` as one OTLP/JSON request, readable by `load_otlp_json`."""
    request = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "personalos-evals"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "personalos"},
                        "spans": [
                            {
                                "name": span.name,
                                "startTimeUnixNano": str(span.start_ns),
                                "endTimeUnixNano": str(span.end_ns),
                                "attributes": [
                                    {"key": key, "value": _encode_value(value)}
                                    for key, value in span.attributes.items()
                                ],
                                "status": {"code": 1 if span.ok else 2},
                            }
                            for span in spans
                        ],
                    }
                ],
            }
        ]
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request) + "\n", encoding="utf-8")


__all__ = [
    "SpanRecord",
    "SpanCapture",
    "capture",
    "from_readable",
    "load_otlp_json",
    "write_otlp_json",
]
