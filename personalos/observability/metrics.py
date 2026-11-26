"""Metrics: every metric PersonalOS emits, its labels, and the one way to record it.

This module is the single definition of the metric contract. `CATALOG` lists
every instrument by name, kind, unit and label set; `docs/METRICS.md` is the
same list in prose, and `tests/contracts/test_metrics_contract.py` fails when
the two disagree, when a dashboard or alert rule under `deploy/` queries a
series that is not in the catalog, or when a catalog metric is queried by no
dashboard. So renaming a metric breaks CI instead of silently emptying a panel.

**One spelling, two names.** Instruments are created under their OpenTelemetry
name (`personalos.api.request.duration`). The collector's Prometheus exporter
turns that into the series a dashboard queries: dots become underscores, a
unit of seconds appends `_seconds`, and a counter gets `_total`
(`personalos_api_request_duration_seconds`). `MetricSpec.prometheus_name` is
that translation, so tests and docs never spell it by hand.

**Labels are a closed set, and never user content.** Every value goes through
`_label_value` before it reaches an instrument:

- only the keys a metric declares are kept; anything else is dropped;
- `workflow_id` must parse as a UUID, or it becomes `invalid`;
- any other value must be a short identifier (letters, digits and `_ . : / { }
  -` and spaces, at most 128 characters) that the Phase D redactor finds no
  secret in, or it becomes `other`. That shape admits route templates, step,
  provider and integration names, and nothing with an `@`, a comma or a line
  break -- so not an email address, a message body or a resume line.

The SDK views installed by `metrics_setup` repeat the key allow-list, so a
label recorded around this module is dropped too.

**Recording never fails the caller.** A metric is an observation; an
exception from an instrument is logged and swallowed.

With no meter provider configured every instrument is OpenTelemetry's no-op.
`metrics_setup.configure_metrics_from_settings` installs the real one; tests
install their own with `metrics_setup.configure_metrics(reader=...)`.
"""

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from opentelemetry import metrics as otel_metrics

from personalos.domain.operational_state import StallReason
from personalos.domain.redaction import default_redactor

logger = logging.getLogger(__name__)

#: Instrumentation scope every PersonalOS instrument is created under.
METER_NAME = "personalos"

#: The value a label takes when what was recorded is not a safe identifier.
OTHER = "other"
#: The value `workflow_id` takes when what was recorded is not a UUID.
INVALID = "invalid"
#: The value a declared label takes when the call site did not supply it.
UNKNOWN = "unknown"

_MAX_LABEL_LENGTH = 128
_SAFE_LABEL = re.compile(rf"^[A-Za-z0-9_.:/{{}}\- ]{{1,{_MAX_LABEL_LENGTH}}}$")


class MetricKind(str, Enum):
    """The instrument behind a metric."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    #: An observable gauge: sampled through a registered source at each export.
    GAUGE = "gauge"


class Source(str, Enum):
    """Which process emits a metric, which is the `job` label it arrives under."""

    API = "api"
    WORKER = "worker"
    #: Sampled from the database by the API (`state_metrics`), so it is
    #: reported while the worker is down -- the case a stall alert exists for.
    STATE = "state"
    #: Recorded by either process; the dashboards aggregate across both.
    ANY = "any"


@dataclass(frozen=True)
class MetricSpec:
    """One metric: its name, kind, unit, labels and meaning."""

    name: str
    kind: MetricKind
    description: str
    labels: tuple[str, ...] = ()
    #: `s` for seconds, or empty. Only seconds are used, so the Prometheus
    #: suffix is fixed.
    unit: str = ""
    source: Source = Source.ANY
    #: Histogram bucket boundaries, in the metric's unit.
    buckets: tuple[float, ...] = ()

    @property
    def prometheus_name(self) -> str:
        """The series name the collector's Prometheus exporter gives this metric."""
        name = self.name.replace(".", "_")
        if self.unit == "s" and not name.endswith("_seconds"):
            name += "_seconds"
        if self.kind is MetricKind.COUNTER and not name.endswith("_total"):
            name += "_total"
        return name

    @property
    def prometheus_series(self) -> tuple[str, ...]:
        """Every series name a query may use for this metric."""
        if self.kind is MetricKind.HISTOGRAM:
            base = self.prometheus_name
            return (f"{base}_bucket", f"{base}_sum", f"{base}_count")
        return (self.prometheus_name,)


# --- Label vocabularies -------------------------------------------------------------
#
# Closed sets a label is drawn from, where the code defines one. Listed here so
# the docs and the contract tests read them from the same place as the code.


class WorkflowOutcome(str, Enum):
    """How one invocation of a graph ended."""

    COMPLETED = "completed"
    #: Stopped at an interrupt (an approval) and is waiting, not finished.
    PAUSED = "paused"
    FAILED = "failed"


class StepOutcome(str, Enum):
    """How one step (graph node) ended."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class StepClass(str, Enum):
    """Whether a step is expected to be quick or to wait on something slow."""

    QUICK = "quick"
    LONG_RUNNING = "long_running"


class RetrySource(str, Enum):
    """What was tried again."""

    #: A workflow command claimed again because its first worker never settled it.
    COMMAND_REDELIVERY = "command_redelivery"
    #: An operator's `POST .../retry` of a failed step.
    OPERATOR_RETRY = "operator_retry"
    #: An outbox event put back in line after a failed delivery.
    OUTBOX_DELIVERY = "outbox_delivery"
    #: A tool dispatch retried inside the executor after a transient failure.
    TOOL_DISPATCH = "tool_dispatch"


class ModelOutcome(str, Enum):
    """How one model call attempt ended."""

    OK = "ok"
    #: The provider answered, but not in the requested schema.
    SCHEMA_INVALID = "schema_invalid"
    ERROR = "error"


#: Steps that wait on a job board, a mailbox, a calendar, a model or a whole
#: subgraph, keyed `(graph, node)`. Every other step is `quick`. The dashboards
#: chart the two apart because their latencies differ by orders of magnitude,
#: and a quick step that slows down is a different problem from a slow one.
#: `tests/contracts/test_metrics_contract.py` checks each is a registered node.
LONG_RUNNING_STEPS: frozenset[tuple[str, str]] = frozenset(
    {
        ("supervisor", "classify_intent"),
        ("supervisor", "plan_work"),
        ("supervisor", "run_job_subgraph"),
        ("supervisor", "run_file_subgraph"),
        ("supervisor", "run_communications_subgraph"),
        ("supervisor", "run_calendar_subgraph"),
        ("job_search", "search_providers"),
        ("job_search", "score_candidates"),
        ("job_search", "prepare_application_packet"),
        ("job_search", "execute_approved_actions"),
        ("job_search", "classify_recruiter_events"),
        ("job_search", "draft_follow_up_for_triggered_checkpoint"),
        ("communications", "triage_inbox"),
        ("communications", "load_thread"),
        ("communications", "draft_reply"),
        ("communications", "execute_send"),
        ("calendar", "check_availability"),
        ("calendar", "execute_calendar_changes"),
        ("file_task", "plan_file_task"),
        ("file_task", "read_documents"),
        ("file_task", "apply_file_changes"),
    }
)


def step_class(graph: str | None, step: str | None) -> StepClass:
    """`long_running` for a step in `LONG_RUNNING_STEPS`, `quick` for any other."""
    return StepClass.LONG_RUNNING if (graph, step) in LONG_RUNNING_STEPS else StepClass.QUICK


# --- The catalog --------------------------------------------------------------------

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_STEP_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 900.0, 3600.0)
_MODEL_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
_WAIT_BUCKETS = (
    60.0,
    300.0,
    900.0,
    3600.0,
    4 * 3600.0,
    12 * 3600.0,
    86400.0,
    3 * 86400.0,
    7 * 86400.0,
)


class Metric:
    """Every metric, by role. `CATALOG` is these, in this order."""

    # API -----------------------------------------------------------------------
    API_REQUEST_DURATION = MetricSpec(
        "personalos.api.request.duration",
        MetricKind.HISTOGRAM,
        "Time to answer one API request, by route template, method and status class",
        labels=("route", "method", "status_class"),
        unit="s",
        source=Source.API,
        buckets=_LATENCY_BUCKETS,
    )
    API_LIMIT_HITS = MetricSpec(
        "personalos.api.limit_hits",
        MetricKind.COUNTER,
        "Requests refused by an API limit (Phase L2), by route and limit",
        labels=("route", "limit"),
        source=Source.API,
    )

    # Workflows -----------------------------------------------------------------
    WORKFLOW_RUNS = MetricSpec(
        "personalos.workflow.runs",
        MetricKind.COUNTER,
        "Graph invocations finished, by graph and outcome (completed, paused, failed)",
        labels=("graph", "outcome"),
        source=Source.WORKER,
    )
    WORKFLOW_RUN_DURATION = MetricSpec(
        "personalos.workflow.run.duration",
        MetricKind.HISTOGRAM,
        "Time one graph invocation ran, from claim to completion, pause or failure",
        labels=("graph", "outcome"),
        unit="s",
        source=Source.WORKER,
        buckets=_STEP_BUCKETS,
    )
    WORKFLOW_STEP_DURATION = MetricSpec(
        "personalos.workflow.step.duration",
        MetricKind.HISTOGRAM,
        "Time one step ran, by graph, step, step class (quick, long_running) and outcome",
        labels=("graph", "step", "step_class", "outcome"),
        unit="s",
        source=Source.WORKER,
        buckets=_STEP_BUCKETS,
    )
    APPROVAL_WAIT_DURATION = MetricSpec(
        "personalos.approval.wait.duration",
        MetricKind.HISTOGRAM,
        "Time from an approval being requested to it being decided or expired",
        labels=("outcome", "action_kind"),
        unit="s",
        source=Source.ANY,
        buckets=_WAIT_BUCKETS,
    )

    # Worker --------------------------------------------------------------------
    WORKER_COMMANDS = MetricSpec(
        "personalos.worker.commands",
        MetricKind.COUNTER,
        "Workflow commands the worker settled, by kind and outcome (done, failed, held, lease_busy)",
        labels=("kind", "outcome"),
        source=Source.WORKER,
    )
    WORKER_RETRIES = MetricSpec(
        "personalos.worker.retries",
        MetricKind.COUNTER,
        "Work tried again, by source (command_redelivery, operator_retry, outbox_delivery, "
        "tool_dispatch)",
        labels=("source",),
        source=Source.WORKER,
    )
    OUTBOX_DELIVERIES = MetricSpec(
        "personalos.outbox.deliveries",
        MetricKind.COUNTER,
        "Outbox event deliveries, by event type and outcome (acked, retry_scheduled, "
        "dead_lettered, quarantined, claim_lost)",
        labels=("event_type", "outcome"),
        source=Source.WORKER,
    )

    # Model gateway -------------------------------------------------------------
    MODEL_REQUEST_DURATION = MetricSpec(
        "personalos.model.request.duration",
        MetricKind.HISTOGRAM,
        "Time one model call attempt took, by provider, operation and outcome",
        labels=("provider", "operation", "outcome"),
        unit="s",
        source=Source.ANY,
        buckets=_MODEL_BUCKETS,
    )

    # Integration health (Phase J8) ---------------------------------------------
    INTEGRATION_HEALTH_CHECKS = MetricSpec(
        "personalos.integration.health.checks",
        MetricKind.COUNTER,
        "Integration health outcomes recorded, by integration, status and source",
        labels=("integration", "status", "source"),
        source=Source.ANY,
    )
    INTEGRATION_HEALTH_TRANSITIONS = MetricSpec(
        "personalos.integration.health.transitions",
        MetricKind.COUNTER,
        "Integration health state changes",
        labels=("integration", "from_status", "status"),
        source=Source.ANY,
    )
    INTEGRATION_HEALTH_STATUS = MetricSpec(
        "personalos.integration.health.status",
        MetricKind.GAUGE,
        "Integration health: 0 healthy, 1 degraded, 2 down",
        labels=("integration",),
        source=Source.ANY,
    )

    # Operational state, sampled from the database ------------------------------
    WORKER_QUEUE_DEPTH = MetricSpec(
        "personalos.worker.queue.depth",
        MetricKind.GAUGE,
        "Workflow commands not yet settled, by state (pending, in_progress)",
        labels=("state",),
        source=Source.STATE,
    )
    OUTBOX_BACKLOG = MetricSpec(
        "personalos.outbox.backlog",
        MetricKind.GAUGE,
        "Outbox events (other than workflow commands) not yet delivered, by state",
        labels=("state",),
        source=Source.STATE,
    )
    OUTBOX_OLDEST_PENDING_AGE = MetricSpec(
        "personalos.outbox.oldest_pending.age",
        MetricKind.GAUGE,
        "Age of the oldest undelivered outbox event; 0 when there is none",
        unit="s",
        source=Source.STATE,
    )
    OUTBOX_PARKED = MetricSpec(
        "personalos.outbox.parked",
        MetricKind.GAUGE,
        "Outbox events parked for an operator, by status (dead_letter, quarantined)",
        labels=("status",),
        source=Source.STATE,
    )
    WORKFLOWS_ACTIVE = MetricSpec(
        "personalos.workflows.active",
        MetricKind.GAUGE,
        "Workflows with a thread running or a command queued",
        source=Source.STATE,
    )
    WORKFLOWS_STALLED = MetricSpec(
        "personalos.workflows.stalled",
        MetricKind.GAUGE,
        "Stalled workflows, by reason (no_progress, approval_overdue); 0 when none",
        labels=("reason",),
        source=Source.STATE,
    )
    WORKFLOW_STALLED_FOR = MetricSpec(
        "personalos.workflow.stalled_for",
        MetricKind.GAUGE,
        "How long each stalled workflow has gone without progress (only stalled ones)",
        labels=("workflow_id", "reason"),
        unit="s",
        source=Source.STATE,
    )


CATALOG: tuple[MetricSpec, ...] = tuple(
    value for value in vars(Metric).values() if isinstance(value, MetricSpec)
)

# --- Labels -------------------------------------------------------------------------


def _label_value(key: str, value: Any) -> str:
    """`value` as a label value that cannot carry user content."""
    if value is None:
        return UNKNOWN
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if key == "workflow_id":
        try:
            return str(UUID(text))
        except ValueError:
            return INVALID
    if not _SAFE_LABEL.match(text) or default_redactor().contains_secret(text):
        return OTHER
    return text


def clean_labels(spec: MetricSpec, labels: Mapping[str, Any]) -> dict[str, str]:
    """The labels `spec` declares, each made safe; undeclared keys are dropped."""
    return {key: _label_value(key, labels.get(key)) for key in spec.labels}


# --- Provider and instruments -------------------------------------------------------

_lock = threading.RLock()
#: The provider instruments are created from, when one was configured here
#: rather than installed globally (tests configure one per test).
_provider: otel_metrics.MeterProvider | None = None
#: Instruments created from the current provider, by metric name.
_instruments: dict[str, Any] = {}
#: Sources each gauge is sampled from at export, by metric name.
_gauge_sources: dict[str, list[Callable[[], Iterable[tuple[float, Mapping[str, Any]]]]]] = {}


def set_meter_provider(provider: otel_metrics.MeterProvider | None) -> None:
    """Create PersonalOS instruments from `provider`; `None` goes back to the global one."""
    global _provider
    with _lock:
        _provider = provider
        _instruments.clear()
    ensure_instruments()


def meter_provider() -> otel_metrics.MeterProvider:
    """The provider PersonalOS instruments are created from."""
    return _provider or otel_metrics.get_meter_provider()


def _observe(spec: MetricSpec) -> Callable[[Any], list[otel_metrics.Observation]]:
    def callback(_options: Any) -> list[otel_metrics.Observation]:
        with _lock:
            sources = list(_gauge_sources.get(spec.name, ()))
        observations: list[otel_metrics.Observation] = []
        for source in sources:
            try:
                for value, labels in source():
                    observations.append(otel_metrics.Observation(value, clean_labels(spec, labels)))
            except Exception:  # noqa: BLE001 - one bad source must not hide the others
                logger.exception("metric source for %s failed", spec.name)
        return observations

    return callback


def _create(spec: MetricSpec) -> Any:
    meter = meter_provider().get_meter(METER_NAME)
    if spec.kind is MetricKind.COUNTER:
        return meter.create_counter(spec.name, unit=spec.unit, description=spec.description)
    if spec.kind is MetricKind.HISTOGRAM:
        return meter.create_histogram(spec.name, unit=spec.unit, description=spec.description)
    return meter.create_observable_gauge(
        spec.name, callbacks=[_observe(spec)], unit=spec.unit, description=spec.description
    )


def _instrument(spec: MetricSpec) -> Any:
    with _lock:
        instrument = _instruments.get(spec.name)
        if instrument is None:
            instrument = _instruments[spec.name] = _create(spec)
        return instrument


def ensure_instruments() -> None:
    """Create every catalog instrument now, so each gauge is observed from the first export."""
    for spec in CATALOG:
        _instrument(spec)


# --- Recording ----------------------------------------------------------------------


def add(spec: MetricSpec, value: float = 1, **labels: Any) -> None:
    """Add `value` to the counter `spec`."""
    if spec.kind is not MetricKind.COUNTER:
        raise TypeError(f"{spec.name} is a {spec.kind.value}, not a counter")
    try:
        _instrument(spec).add(value, clean_labels(spec, labels))
    except Exception:  # noqa: BLE001 - recording never fails the caller
        logger.exception("could not record %s", spec.name)


def observe(spec: MetricSpec, value: float, **labels: Any) -> None:
    """Record `value` in the histogram `spec`."""
    if spec.kind is not MetricKind.HISTOGRAM:
        raise TypeError(f"{spec.name} is a {spec.kind.value}, not a histogram")
    try:
        _instrument(spec).record(max(float(value), 0.0), clean_labels(spec, labels))
    except Exception:  # noqa: BLE001 - recording never fails the caller
        logger.exception("could not record %s", spec.name)


def prime(spec: MetricSpec, label_sets: Iterable[Mapping[str, Any]]) -> None:
    """Add 0 to counter `spec` for each label set, so its series exist before anything happens.

    A rate over a series that does not exist yet is no data, not zero; a
    panel or alert on a counter of rare events (dead letters, retries) needs
    the zero.
    """
    for labels in label_sets:
        add(spec, 0, **labels)


def register_gauge(
    spec: MetricSpec, source: Callable[[], Iterable[tuple[float, Mapping[str, Any]]]]
) -> Callable[[], None]:
    """Sample gauge `spec` from `source` at every export. Returns a function that unregisters it.

    `source` returns `(value, labels)` pairs; it is called on the exporter's
    thread and must be cheap -- read a snapshot, do not query a database.
    """
    if spec.kind is not MetricKind.GAUGE:
        raise TypeError(f"{spec.name} is a {spec.kind.value}, not a gauge")
    with _lock:
        _gauge_sources.setdefault(spec.name, []).append(source)

    def unregister() -> None:
        with _lock:
            sources = _gauge_sources.get(spec.name, [])
            if source in sources:
                sources.remove(source)

    return unregister


# --- Model providers ----------------------------------------------------------------

#: Model names the deterministic stand-ins report; their provider is `rules`.
RULES_MODELS = frozenset({"rules", "keyword", "extractive"})

_model_providers: dict[str, str] = {}


def register_model_provider(model: str, provider: str) -> None:
    """Say which provider serves `model`, for the `provider` label of model metrics.

    The model gateway registers each configured model as it is built; a call
    site then names only its model, as it already does for its span.
    """
    with _lock:
        _model_providers[model] = provider


def model_provider(model: str | None) -> str:
    """The provider registered for `model`: `rules` for a stand-in, `unknown` if not told."""
    if model is None:
        return UNKNOWN
    if model in RULES_MODELS:
        return "rules"
    with _lock:
        return _model_providers.get(model, UNKNOWN)


@dataclass
class _Timer:
    """A running timer: `elapsed()` seconds since it started."""

    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def start_timer() -> _Timer:
    """A timer for a span of work that does not fit a `with` block."""
    return _Timer()


__all__ = [
    "CATALOG",
    "INVALID",
    "LONG_RUNNING_STEPS",
    "METER_NAME",
    "OTHER",
    "UNKNOWN",
    "Metric",
    "MetricKind",
    "MetricSpec",
    "ModelOutcome",
    "RetrySource",
    "Source",
    "StallReason",
    "StepClass",
    "StepOutcome",
    "WorkflowOutcome",
    "add",
    "clean_labels",
    "ensure_instruments",
    "meter_provider",
    "model_provider",
    "observe",
    "prime",
    "register_gauge",
    "register_model_provider",
    "set_meter_provider",
    "start_timer",
    "step_class",
]
