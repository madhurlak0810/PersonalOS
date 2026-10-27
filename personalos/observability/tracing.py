"""Tracing: one trace per workflow, from the request that started it to its last receipt.

A workflow outlives every process that touches it. The request that starts it
returns before anything runs; a worker runs it to an approval and lets go; days
later a second request answers the approval and another worker -- maybe on
another machine -- resumes it and executes what was approved. Ordinary span
parenting cannot connect those: the approval request arrives with no context
from the request that started the run. So the trace is keyed by the workflow
itself:

- **The trace id is the workflow id.** A UUID is 128 bits, exactly a trace id,
  so `workflow_trace_id(workflow_id)` is `workflow_id.int` and the trace a
  workflow's spans land in is the one an operator finds by pasting its id.
- **The root span id is derived from the workflow id** (`workflow_root_span_id`).
  Any process that knows the workflow id can parent a span on the workflow's
  root without having been told anything else; the request that creates the
  workflow emits that root span (`workflow_span(..., root=True)`).
- **Within a hand-off, context travels with the work.** A `WorkflowCommand` and
  an `EventEnvelope` each carry a W3C `traceparent`/`baggage` carrier
  (`inject_trace_context`), so the worker's span is a child of the span that
  queued it -- the request, or the event delivery -- not merely of the root.

`workflow_span` is how a span is scoped to a workflow: it joins the current
span when that is already in the workflow's trace, else the carrier it is
handed, else the workflow's root, and in the last two cases *links* whatever
foreign span was current (an HTTP caller's trace, the trace an inbound event
arrived in) rather than losing it. `workflow_id`, `correlation_id`,
`thread_id` and `actor_id` ride in baggage, and every span started under it is
stamped with them (`tracing_setup.ScopeAttributesSpanProcessor`), so a tool
span deep in an executor carries the same ids as the request span above it.

**No payload goes on a span.** `clean_attributes` keeps scalars and sequences
of scalars and drops anything else -- a dict, a model, a message body -- so a
span refers to an action, a command or an event by id and never embeds it.
What does reach a span (an exception message, a recorded traceback) is
redacted on its way out by `trace_redaction.RedactingSpanExporter`.

This module is the tracing *API*: it depends on `opentelemetry-api` alone and
is safe to import from any layer. With no provider configured every span is a
no-op. Installing the SDK provider, the exporter and the redaction around it is
`personalos.observability.tracing_setup`, which only the composition root calls.
"""

import functools
import hashlib
import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar
from uuid import UUID

from opentelemetry import baggage, propagate, trace
from opentelemetry import context as otel_context
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)

#: Instrumentation scope every PersonalOS span is created under.
TRACER_NAME = "personalos"


class Attr:
    """Span attribute names. One spelling each, shared by every layer that traces."""

    WORKFLOW_ID = "workflow_id"
    CORRELATION_ID = "correlation_id"
    THREAD_ID = "thread_id"
    ACTOR_ID = "actor_id"
    NODE = "node"
    TOOL_NAME = "tool_name"
    ACTION_ID = "action_id"
    ACTION_KIND = "action_kind"
    POLICY_DECISION = "policy_decision"
    POLICY_RULE = "policy_rule"
    POLICY_DECISION_ID = "policy_decision_id"
    ATTEMPT = "attempt"
    MAX_ATTEMPTS = "max_attempts"
    RETRY_DELAY = "retry_delay_seconds"
    COMMAND_ID = "command_id"
    COMMAND_KIND = "command_kind"
    EVENT_ID = "event_id"
    EVENT_TYPE = "event_type"
    EVENT_OUTCOME = "event_outcome"
    APPROVAL_ID = "approval_id"
    APPROVAL_OUTCOME = "approval_outcome"
    INTERRUPTED = "interrupted"
    INTERRUPT_IDS = "interrupt_ids"
    OUTCOME = "outcome"
    RECEIPT_OK = "receipt_ok"
    RECEIPT_REPLAYED = "receipt_replayed"
    EXTERNAL_REFERENCE = "external_reference"
    MODEL_OPERATION = "model_operation"
    MODEL_NAME = "model_name"
    GRAPH = "graph"


#: The ids that scope a workflow's spans. Carried in baggage, so they cross
#: the outbox with the trace context, and copied onto every span started under
#: them.
SCOPE_ATTRIBUTES = (Attr.WORKFLOW_ID, Attr.CORRELATION_ID, Attr.THREAD_ID, Attr.ACTOR_ID)

#: Prefix hashed with a workflow id to derive its root span id.
_ROOT_SPAN_SALT = b"personalos.workflow.root:"

#: Ids the next span must take, for a workflow's root span. Read by
#: `tracing_setup.WorkflowIdGenerator`; set only for the one `start_span` call
#: that creates the root.
_FORCED_IDS: ContextVar[tuple[int, int] | None] = ContextVar(
    "personalos_forced_trace_ids", default=None
)

#: The provider spans are created from, when one was configured here rather
#: than installed globally (tests configure one per test; the global can be
#: set only once per process).
_provider: trace.TracerProvider | None = None


# --- Provider ---------------------------------------------------------------------


def set_tracer_provider(provider: trace.TracerProvider | None) -> None:
    """Create PersonalOS spans from `provider`; `None` goes back to the global one."""
    global _provider
    _provider = provider


def tracer_provider() -> trace.TracerProvider:
    """The provider PersonalOS spans are created from."""
    return _provider or trace.get_tracer_provider()


def tracer() -> trace.Tracer:
    """The tracer PersonalOS spans are created with."""
    return tracer_provider().get_tracer(TRACER_NAME)


def enabled() -> bool:
    """True when spans are recorded: a real provider is configured, here or globally."""
    if _provider is not None:
        return True
    return not isinstance(
        trace.get_tracer_provider(), (trace.ProxyTracerProvider, trace.NoOpTracerProvider)
    )


def forced_ids() -> tuple[int, int] | None:
    """The `(trace_id, span_id)` the span being created must take, if any."""
    return _FORCED_IDS.get()


def _supports_forced_ids(provider: trace.TracerProvider) -> bool:
    return bool(getattr(getattr(provider, "id_generator", None), "supports_forced_ids", False))


# --- Workflow identity --------------------------------------------------------------


def workflow_trace_id(workflow_id: UUID) -> int:
    """The trace id of every span about `workflow_id`: the id itself, as 128 bits."""
    return workflow_id.int or 1


def workflow_root_span_id(workflow_id: UUID) -> int:
    """The span id of the workflow's root span, derivable by any process."""
    digest = hashlib.sha256(_ROOT_SPAN_SALT + workflow_id.bytes).digest()
    return int.from_bytes(digest[:8], "big") or 1


def workflow_root_context(workflow_id: UUID) -> SpanContext:
    """The workflow's root span, as a remote parent any process can attach to."""
    return SpanContext(
        trace_id=workflow_trace_id(workflow_id),
        span_id=workflow_root_span_id(workflow_id),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def format_trace_id(workflow_id: UUID) -> str:
    """The workflow's trace id as a tracing backend displays it (32 hex digits)."""
    return trace.format_trace_id(workflow_trace_id(workflow_id))


# --- Attributes -----------------------------------------------------------------------

_SCALARS = (str, bool, int, float)


def _scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, (UUID, datetime)):
        return str(value) if isinstance(value, UUID) else value.isoformat()
    return None


def clean_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only values a span may carry: ids, labels, counts, flags.

    `None` is dropped; a UUID, enum or datetime becomes its string form; a
    list or tuple is kept when every item is such a scalar. Anything else -- a
    mapping, a model, an arbitrary object -- is dropped rather than
    stringified, because its string form is a payload.
    """
    cleaned: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            items = [_scalar(item) for item in value]
            if items and all(item is not None for item in items):
                # One type per sequence, as OpenTelemetry requires.
                cleaned[key] = (
                    items if len({type(i) for i in items}) == 1 else [str(i) for i in items]
                )
            continue
        scalar = _scalar(value)
        if scalar is not None:
            cleaned[key] = scalar
    return cleaned


def policy_attributes(decision: Any) -> dict[str, Any]:
    """The attributes a policy verdict is traced with: what was decided, by which rule, where."""
    if decision is None:
        return {}
    return {
        Attr.POLICY_DECISION: getattr(decision, "decision", None),
        Attr.POLICY_RULE: getattr(decision, "rule", None),
        Attr.POLICY_DECISION_ID: getattr(decision, "record_id", None),
    }


def set_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
    """Set the cleaned `attributes` on `span`."""
    if span.is_recording():
        span.set_attributes(clean_attributes(attributes))


def add_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    """Record a point-in-time event on the current span."""
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, clean_attributes(attributes))


def current_span() -> Span:
    """The span active in this context (a no-op span when there is none)."""
    return trace.get_current_span()


# --- Propagation ------------------------------------------------------------------------


def inject_trace_context() -> dict[str, str]:
    """The current trace context and scope baggage as a W3C carrier; empty if not tracing."""
    if not enabled() or not trace.get_current_span().get_span_context().is_valid:
        return {}
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract_trace_context(carrier: Mapping[str, str] | None) -> otel_context.Context:
    """The context a carrier written by `inject_trace_context` describes."""
    return propagate.extract(dict(carrier or {}))


@contextmanager
def attached(ctx: otel_context.Context) -> Iterator[None]:
    """Make `ctx` current for the duration of the block."""
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


# --- Spans ------------------------------------------------------------------------------


@contextmanager
def _running(new: Span, ctx: otel_context.Context | None = None) -> Iterator[Span]:
    """Make `new` current (under `ctx`, if given) and end it when the block does.

    A failure is recorded on the span and re-raised. A LangGraph interrupt is
    not a failure -- the run is waiting -- so it ends the span marked
    `interrupted` instead of as an error that never happened.
    """
    with (
        attached(ctx) if ctx is not None else _nothing(),
        trace.use_span(
            new, end_on_exit=True, record_exception=False, set_status_on_exception=False
        ),
    ):
        try:
            yield new
        except BaseException as error:
            if _is_interrupt(error):
                set_attributes(new, {Attr.INTERRUPTED: True})
            elif isinstance(error, Exception):
                mark_failed(new, error)
            raise


@contextmanager
def _nothing() -> Iterator[None]:
    yield


@contextmanager
def span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Iterator[Span]:
    """A span under whatever is current. Exceptions are recorded, then re-raised."""
    new = tracer().start_span(name, kind=kind, attributes=clean_attributes(attributes))
    with _running(new) as current:
        yield current


def _scope(
    ctx: otel_context.Context, values: Mapping[str, Any]
) -> tuple[otel_context.Context, dict[str, Any]]:
    """Put the given scope ids into `ctx`'s baggage; return it and the full scope."""
    for key, value in clean_attributes(values).items():
        ctx = baggage.set_baggage(key, str(value), ctx)
    scope = {
        key: value
        for key in SCOPE_ATTRIBUTES
        if (value := baggage.get_baggage(key, ctx)) is not None
    }
    return ctx, scope


def _link(span_context: SpanContext) -> list[Link]:
    return [Link(span_context)] if span_context.is_valid else []


@contextmanager
def workflow_span(
    name: str,
    *,
    workflow_id: UUID,
    correlation_id: UUID | str | None = None,
    thread_id: str | None = None,
    actor_id: str | None = None,
    carrier: Mapping[str, str] | None = None,
    attributes: Mapping[str, Any] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    root: bool = False,
) -> Iterator[Span]:
    """A span in `workflow_id`'s trace, wherever this code is running.

    Parented on, in order of preference: the current span, when it is already
    in the workflow's trace; the span `carrier` names, when that is; else the
    workflow's root span -- which this call *is*, with `root=True` on the
    request that created the workflow. A current span or carrier from another
    trace is linked, not dropped.

    The scope ids given here (and any already in baggage) are set on this
    span and carried to every span under it. With tracing off this is a plain
    no-op span, and the context is left alone.
    """
    if not enabled():
        with span(name, kind=kind) as noop:
            yield noop
        return
    trace_id = workflow_trace_id(workflow_id)
    ctx = otel_context.get_current()
    links: list[Link] = []
    make_root = False

    current = trace.get_current_span(ctx).get_span_context()
    if not (current.is_valid and current.trace_id == trace_id):
        remote_ctx = extract_trace_context(carrier) if carrier else None
        remote = (
            trace.get_current_span(remote_ctx).get_span_context()
            if remote_ctx is not None
            else None
        )
        if (
            remote_ctx is not None
            and remote is not None
            and remote.is_valid
            and remote.trace_id == trace_id
        ):
            ctx = remote_ctx
        else:
            links = _link(current) + (_link(remote) if remote is not None else [])
            make_root = root and _supports_forced_ids(tracer_provider())
            # Carry the remote's baggage, if any, onto the workflow's root.
            base = remote_ctx if remote_ctx is not None else ctx
            parent = (
                trace.INVALID_SPAN
                if make_root
                else NonRecordingSpan(workflow_root_context(workflow_id))
            )
            ctx = trace.set_span_in_context(parent, base)

    ctx, scope = _scope(
        ctx,
        {
            Attr.WORKFLOW_ID: workflow_id,
            Attr.CORRELATION_ID: correlation_id,
            Attr.THREAD_ID: thread_id,
            Attr.ACTOR_ID: actor_id,
        },
    )
    span_attributes = {**scope, **clean_attributes(attributes)}

    if make_root:
        forced = _FORCED_IDS.set((trace_id, workflow_root_span_id(workflow_id)))
        try:
            new = tracer().start_span(
                name, context=ctx, kind=kind, attributes=span_attributes, links=links
            )
        finally:
            _FORCED_IDS.reset(forced)
    else:
        new = tracer().start_span(
            name, context=ctx, kind=kind, attributes=span_attributes, links=links
        )

    with _running(new, ctx) as opened:
        yield opened


def mark_failed(span_: Span, error: BaseException) -> None:
    """Record `error` on `span_` and mark it failed."""
    if span_.is_recording():
        span_.record_exception(error)
        span_.set_status(Status(StatusCode.ERROR, type(error).__name__))


# --- Graph nodes ------------------------------------------------------------------------


def _is_interrupt(error: BaseException) -> bool:
    """True for LangGraph's control-flow exceptions: an interrupt is a pause, not a failure."""
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:  # pragma: no cover - langgraph is a hard dependency
        return False
    return isinstance(error, GraphBubbleUp)


def _node_scope() -> dict[str, Any]:
    """The workflow and thread of the run this node belongs to, from its LangGraph config."""
    try:
        from langgraph.config import get_config

        configurable = (get_config() or {}).get("configurable") or {}
    except Exception:
        return {}
    return {
        "workflow_id": configurable.get("workflow_id"),
        "thread_id": configurable.get("thread_id"),
        "correlation_id": configurable.get("correlation_id"),
        "actor_id": configurable.get("actor_id"),
    }


def _as_uuid(value: Any) -> UUID | None:
    if isinstance(value, UUID) or value is None:
        return value
    try:
        return UUID(str(value))
    except ValueError:
        return None


@contextmanager
def node_span(node: str, graph: str | None = None) -> Iterator[Span]:
    """The span one graph node runs in, scoped to the run's workflow and thread."""
    scope = _node_scope()
    attributes = {Attr.NODE: node, Attr.GRAPH: graph}
    workflow_id = _as_uuid(scope.pop("workflow_id", None))
    if workflow_id is None:
        with span(f"node.{node}", attributes) as current:
            yield current
        return
    with workflow_span(
        f"node.{node}", workflow_id=workflow_id, attributes=attributes, **scope
    ) as current:
        yield current


F = TypeVar("F", bound=Callable[..., Any])


def traced_node(node: str, fn: F, *, graph: str | None = None) -> F:
    """Wrap a graph node so each run of it is a span named `node.<node>`.

    Works for sync and async nodes alike, and keeps the node's signature, so
    LangGraph injects the same arguments (`config`, ...) it would unwrapped.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            with node_span(node, graph):
                return await fn(*args, **kwargs)

        return run_async  # type: ignore[return-value]

    @functools.wraps(fn)
    def run_sync(*args: Any, **kwargs: Any) -> Any:
        with node_span(node, graph):
            return fn(*args, **kwargs)

    return run_sync  # type: ignore[return-value]


# --- Model calls ------------------------------------------------------------------------


@contextmanager
def model_span(operation: str, *, model: str | None = None, **attributes: Any) -> Iterator[Span]:
    """The span one model call runs in. Prompts and completions are never attributes."""
    with span(
        f"model.{operation}",
        {Attr.MODEL_OPERATION: operation, Attr.MODEL_NAME: model, **attributes},
        kind=SpanKind.CLIENT,
    ) as current:
        yield current


def model_name(model: Any) -> str | None:
    """The model a LangChain chat model is bound to, if it says."""
    for field in ("model_name", "model", "model_id"):
        value = getattr(model, field, None)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = [
    "Attr",
    "SCOPE_ATTRIBUTES",
    "TRACER_NAME",
    "add_event",
    "attached",
    "clean_attributes",
    "current_span",
    "enabled",
    "extract_trace_context",
    "forced_ids",
    "format_trace_id",
    "inject_trace_context",
    "mark_failed",
    "model_name",
    "model_span",
    "node_span",
    "policy_attributes",
    "set_attributes",
    "set_tracer_provider",
    "span",
    "traced_node",
    "tracer",
    "tracer_provider",
    "workflow_root_context",
    "workflow_root_span_id",
    "workflow_span",
    "workflow_trace_id",
]
