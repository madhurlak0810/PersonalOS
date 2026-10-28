"""Quality metrics: what a harness run and its telemetry say about the system.

Everything here is a pure function of a results document (`results.json`, see
`evals.report`) and a list of spans (`evals.telemetry`), so a report can be
recomputed from files on disk, and the same span metrics can be computed over
an OpenTelemetry export from a deployment with no harness run at all.

`compute()` returns the `quality` section of a results document: one flat
`metrics` mapping, whose names are what `thresholds.json` and the run history
refer to, plus the detail behind each. A metric the inputs say nothing about
is `None`, never zero: a run with no model calls has no tool-call validity
rate, not a rate of zero.

| Metric | From | What |
| --- | --- | --- |
| `routing_accuracy` | `routing` cases | route and, where labelled, domain both right |
| `approval_precision` | `approvals` | approval prompts that were for consequential actions |
| `approval_recall` | `approvals` | consequential actions that could not run unattended |
| `external_write_approval_recall` | `approvals` | the same, for external writes only; must be 1.0 |
| `tool_call_validity_rate` | model spans | model attempts whose output passed its schema |
| `tool_selection_accuracy` | `scenarios` cases | runs that proposed exactly the expected actions |
| `workflow_completion_rate` | `scenarios` cases | runs that reached the expected end with no repair |
| `recovery_rate` | `recovery`, else spans | recoverable failures that finished by resuming |
| `mean_recovery_time_ms` | node spans | first failure to the successful retry of the same node |
| `*_p50_ms`, `*_p95_ms` | node and model spans | interactive vs long-running steps; model calls |
| `model_tokens_per_sec` | model spans | output tokens over model time, where usage was recorded |
| `job_match_precision_at_k` | `job_matching` | relevant postings in the top k |
| `event_classification_f1` | `recruiter_events` | macro F1 of event classification |
| `resume_evidence_grounding` | `job_matching`, `human_review` | matches and drafts whose every citation is on record |
| `duplicate_side_effect_rate` | `scenarios`, `recovery` | repeated side effects over all side effects; must be 0 |
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from personalos.graphs import job_search as js
from personalos.graphs import supervisor as sv
from personalos.observability.tracing import Attr

from .telemetry import SpanRecord

#: Which step each graph node belongs to, for latency. Nodes not listed
#: (persistence, approval, execution) are neither kind of step.
STEP_NODES: dict[str, tuple[str, ...]] = {
    "routing": (sv.LOAD_CONTEXT, sv.CLASSIFY_INTENT, sv.PLAN_WORK, sv.CLARIFY),
    "extraction": (js.CLASSIFY_RECRUITER_EVENTS,),
    "discovery": (
        js.LOAD_SEARCH_PROFILE,
        js.SEARCH_PROVIDERS,
        js.NORMALIZE_JOBS,
        js.DEDUPLICATE,
        js.HARD_FILTER,
        js.SCORE_CANDIDATES,
        js.EVIDENCE_CHECK,
        js.RANK,
        js.SHORTLIST,
    ),
    "drafting": (js.PREPARE_APPLICATION_PACKET, js.DRAFT_FOLLOW_UP),
}

#: A person is waiting on these.
INTERACTIVE_STEPS = ("routing", "extraction")
#: These run in the background.
LONG_RUNNING_STEPS = ("discovery", "drafting")

_STEP_OF_NODE = {node: step for step, nodes in STEP_NODES.items() for node in nodes}

#: Span attributes a model call's output token count may arrive under: the
#: OpenTelemetry GenAI convention first, then what common LangChain
#: instrumentations write.
OUTPUT_TOKEN_ATTRIBUTES = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.usage.completion_tokens",
)

#: The model "name" the keyword classifier reports. Not a model: excluded from
#: model latency so it does not flatter a configuration that routes on rules.
_NOT_A_MODEL = frozenset({"keyword"})


# --- Arithmetic -------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """The `q`th percentile (0-100) by linear interpolation; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    value = ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return round(value, 3)


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


# --- Approvals --------------------------------------------------------------


@dataclass(frozen=True)
class ActionObservation:
    """One action at a consequence boundary: its labels and what the system did.

    `consequential` and `external_write` are ground truth from a golden file.
    `prompted` is whether a person was asked to approve it; `gated` is whether
    it was kept from running unattended at all (asked, or refused outright).
    """

    action: str
    consequential: bool
    external_write: bool
    prompted: bool
    gated: bool


def approval_metrics(observations: Iterable[ActionObservation]) -> dict[str, Any]:
    """Approval precision and recall.

    - precision: of the approval prompts raised, the share that were for a
      consequential action -- how much of a reviewer's attention was earned;
    - recall: of the consequential actions, the share that were gated;
    - external-write recall: the same over external writes only, the one the
      system must never miss.
    """
    observations = list(observations)
    prompts = [o for o in observations if o.prompted]
    consequential = [o for o in observations if o.consequential]
    external = [o for o in observations if o.external_write]
    return {
        "approval_precision": _ratio(sum(o.consequential for o in prompts), len(prompts)),
        "approval_recall": _ratio(sum(o.gated for o in consequential), len(consequential)),
        "external_write_approval_recall": _ratio(sum(o.gated for o in external), len(external)),
        "approval_prompts": len(prompts),
        "consequential_actions": len(consequential),
        "external_writes": len(external),
        "ungated_consequential": sorted({o.action for o in consequential if not o.gated}),
        "unneeded_prompts": sorted({o.action for o in prompts if not o.consequential}),
    }


# --- Spans ------------------------------------------------------------------


def _node_spans(spans: Iterable[SpanRecord]) -> list[SpanRecord]:
    return [span for span in spans if span.get(Attr.NODE) is not None]


def _model_spans(spans: Iterable[SpanRecord]) -> list[SpanRecord]:
    return [
        span
        for span in spans
        if span.get(Attr.MODEL_OPERATION) is not None
        and span.get(Attr.MODEL_NAME) not in _NOT_A_MODEL
    ]


def _distribution(durations: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(durations),
        "p50_ms": percentile(durations, 50),
        "p95_ms": percentile(durations, 95),
    }


def _output_tokens(span: SpanRecord) -> int | None:
    for key in OUTPUT_TOKEN_ATTRIBUTES:
        value = span.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def latency_metrics(spans: Sequence[SpanRecord]) -> dict[str, Any]:
    """p50/p95 per step and per kind of step, from node spans; per model, from model spans.

    One node span is one execution of that step. A span that failed still
    took the time it took, so it counts.
    """
    by_step: dict[str, list[float]] = defaultdict(list)
    for span in _node_spans(spans):
        step = _STEP_OF_NODE.get(span.get(Attr.NODE))
        if step is not None:
            by_step[step].append(span.duration_ms)

    by_model: dict[str, list[SpanRecord]] = defaultdict(list)
    for span in _model_spans(spans):
        by_model[str(span.get(Attr.MODEL_NAME) or "unknown")].append(span)
    models = {}
    for name, calls in sorted(by_model.items()):
        tokens = [(t, s.duration_ms) for s in calls if (t := _output_tokens(s)) is not None]
        seconds = sum(ms for _, ms in tokens) / 1000
        models[name] = {
            **_distribution([s.duration_ms for s in calls]),
            "output_tokens": sum(t for t, _ in tokens) if tokens else None,
            "tokens_per_sec": round(sum(t for t, _ in tokens) / seconds, 2) if seconds else None,
        }
    all_model = [s for calls in by_model.values() for s in calls]
    token_spans = [(t, s.duration_ms) for s in all_model if (t := _output_tokens(s)) is not None]
    token_seconds = sum(ms for _, ms in token_spans) / 1000

    return {
        "steps": {step: _distribution(by_step.get(step, [])) for step in STEP_NODES},
        "interactive": _distribution([d for s in INTERACTIVE_STEPS for d in by_step.get(s, [])]),
        "long_running": _distribution([d for s in LONG_RUNNING_STEPS for d in by_step.get(s, [])]),
        "model": {
            **_distribution([s.duration_ms for s in all_model]),
            "tokens_per_sec": (
                round(sum(t for t, _ in token_spans) / token_seconds, 2) if token_seconds else None
            ),
            "by_model": models,
        },
    }


def tool_call_validity(spans: Sequence[SpanRecord]) -> dict[str, Any]:
    """Model attempts whose output validated against their schema.

    Read from the `schema_valid` attribute `personalos.models.structured` and
    the recruiter-event extractor set on each attempt's span. An attempt that
    failed at the provider never produced an output and is not counted.
    """
    attempts = [span for span in spans if span.get(Attr.SCHEMA_VALID) is not None]
    valid = sum(bool(span.get(Attr.SCHEMA_VALID)) for span in attempts)
    return {
        "attempts": len(attempts),
        "valid": valid,
        "tool_call_validity_rate": _ratio(valid, len(attempts)),
    }


def recovery_from_spans(spans: Sequence[SpanRecord]) -> dict[str, Any]:
    """Failures of a node and whether a later run of the same node, in the same thread, succeeded.

    A run of consecutive failures is one failure; it is recovered when the
    next run of that node in that thread succeeds, and its recovery time is
    from the start of the first failed run to the end of the one that worked.
    Spans with no thread id cannot be told apart across workflows and are
    left out.
    """
    runs: dict[tuple[str, str], list[SpanRecord]] = defaultdict(list)
    for span in _node_spans(spans):
        thread = span.get(Attr.THREAD_ID) or span.get(Attr.WORKFLOW_ID)
        if thread is not None:
            runs[(str(thread), str(span.get(Attr.NODE)))].append(span)

    failures, times = 0, []
    for sequence in runs.values():
        first_failure: SpanRecord | None = None
        for span in sorted(sequence, key=lambda s: s.start_ns):
            if not span.ok:
                if first_failure is None:
                    failures += 1
                    first_failure = span
            elif first_failure is not None:
                times.append((span.end_ns - first_failure.start_ns) / 1e6)
                first_failure = None
    return {
        "failures": failures,
        "recovered": len(times),
        "recovery_rate": _ratio(len(times), failures),
        "mean_recovery_time_ms": round(sum(times) / len(times), 3) if times else None,
    }


# --- Harness cases ----------------------------------------------------------


def _cases(document: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    return [case for case in document.get("cases", ()) if case["suite"] == suite]


def _scored(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [case for case in cases if case["status"] in ("pass", "fail", "error")]


def routing_metrics(document: dict[str, Any]) -> dict[str, Any]:
    """Routing decisions, overall and per kind of decision.

    The kinds: hand to a domain (`job_search`), turn away a request for a
    domain the system does not handle yet (`reject`: a clarify that must name
    the domain), and ask back about a request that names none (`clarify`).
    """
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for case in _scored(_cases(document, "routing")):
        expected = case["expected"]
        if expected["route"] == "job_search":
            kind = "job_search"
        else:
            kind = "reject" if expected.get("domain") else "clarify"
        by_kind[kind].append(case["status"] == "pass")
    every = [ok for oks in by_kind.values() for ok in oks]
    return {
        "routing_accuracy": _ratio(sum(every), len(every)),
        "by_decision": {
            kind: {"cases": len(oks), "accuracy": _ratio(sum(oks), len(oks))}
            for kind, oks in sorted(by_kind.items())
        },
    }


def tool_selection_accuracy(document: dict[str, Any]) -> float | None:
    """Scenario runs that proposed exactly the actions the golden case expects."""
    labelled = [
        case
        for case in _scored(_cases(document, "scenarios"))
        if "intents" in (case["expected"] or {}) and case["actual"]
    ]
    right = sum(
        sorted(case["actual"]["intents"]) == sorted(case["expected"]["intents"])
        for case in labelled
    )
    return _ratio(right, len(labelled))


def workflow_completion_rate(document: dict[str, Any]) -> float | None:
    """Scenario runs that reached the expected end state on their own."""
    scored = _scored(_cases(document, "scenarios"))
    return _ratio(sum(case["status"] == "pass" for case in scored), len(scored))


def _suite_metric(document: dict[str, Any], suite: str, name: str) -> Any:
    return document.get("suites", {}).get(suite, {}).get("metrics", {}).get(name)


def _grounding(document: dict[str, Any]) -> float | None:
    total = grounded = 0
    for suite, count, ok in (
        ("job_matching", "matches", "grounded_matches"),
        ("human_review", "drafts", "grounded_drafts"),
    ):
        total += _suite_metric(document, suite, count) or 0
        grounded += _suite_metric(document, suite, ok) or 0
    return _ratio(grounded, total)


def _duplicates(document: dict[str, Any]) -> dict[str, Any]:
    total = duplicates = 0
    seen = False
    for suite in document.get("suites", {}):
        effects = _suite_metric(document, suite, "side_effects")
        if effects is None:
            continue
        seen = True
        total += effects
        duplicates += _suite_metric(document, suite, "duplicate_side_effects") or 0
    rate = (round(duplicates / total, 4) if total else 0.0) if seen else None
    return {"side_effects": total, "duplicates": duplicates, "duplicate_side_effect_rate": rate}


# --- The report -------------------------------------------------------------


def compute(document: dict[str, Any] | None, spans: Sequence[SpanRecord]) -> dict[str, Any]:
    """The quality section for one run: flat headline `metrics`, and the detail behind them."""
    document = document or {}
    approvals = {
        name: _suite_metric(document, "approvals", name)
        for name in (
            "approval_precision",
            "approval_recall",
            "external_write_approval_recall",
            "approval_prompts",
            "consequential_actions",
            "external_writes",
            "ungated_consequential",
            "unneeded_prompts",
        )
    }
    routing = routing_metrics(document)
    latency = latency_metrics(spans)
    validity = tool_call_validity(spans)
    span_recovery = recovery_from_spans(spans)
    duplicates = _duplicates(document)
    suite_recovery = _suite_metric(document, "recovery", "recovery_rate")

    metrics = {
        "routing_accuracy": routing["routing_accuracy"],
        "approval_precision": approvals["approval_precision"],
        "approval_recall": approvals["approval_recall"],
        "external_write_approval_recall": approvals["external_write_approval_recall"],
        "tool_call_validity_rate": validity["tool_call_validity_rate"],
        "tool_selection_accuracy": tool_selection_accuracy(document),
        "workflow_completion_rate": workflow_completion_rate(document),
        "recovery_rate": (
            suite_recovery
            if _suite_metric(document, "recovery", "recoverable_failures")
            else span_recovery["recovery_rate"]
        ),
        "mean_recovery_time_ms": span_recovery["mean_recovery_time_ms"],
        "interactive_p50_ms": latency["interactive"]["p50_ms"],
        "interactive_p95_ms": latency["interactive"]["p95_ms"],
        "long_running_p50_ms": latency["long_running"]["p50_ms"],
        "long_running_p95_ms": latency["long_running"]["p95_ms"],
        "model_p50_ms": latency["model"]["p50_ms"],
        "model_p95_ms": latency["model"]["p95_ms"],
        "model_tokens_per_sec": latency["model"]["tokens_per_sec"],
        "job_match_precision_at_k": _suite_metric(
            document, "job_matching", "job_match_precision_at_k"
        ),
        "event_classification_f1": _suite_metric(
            document, "recruiter_events", "event_classification_f1"
        ),
        "resume_evidence_grounding": _grounding(document),
        "duplicate_side_effect_rate": duplicates["duplicate_side_effect_rate"],
    }
    return {
        "metrics": metrics,
        "routing": routing,
        "approvals": approvals,
        "tool_calls": validity,
        "recovery": {"suite_recovery_rate": suite_recovery, "spans": span_recovery},
        "latency": latency,
        "side_effects": duplicates,
        "spans": len(spans),
    }


# --- Thresholds -------------------------------------------------------------

THRESHOLDS = Path(__file__).parent / "thresholds.json"


@dataclass(frozen=True)
class Threshold:
    metric: str
    min: float | None = None
    max: float | None = None
    #: A critical breach fails the run; any other is a warning in the report.
    critical: bool = False
    why: str = ""

    def breach(self, value: Any) -> str | None:
        """Why `value` breaks this threshold, or None if it does not."""
        if value is None:
            # Missing data cannot show a critical property holds.
            return "no data" if self.critical else None
        if self.min is not None and value < self.min:
            return f"{value} < {self.min}"
        if self.max is not None and value > self.max:
            return f"{value} > {self.max}"
        return None


def load_thresholds(path: str | Path = THRESHOLDS) -> list[Threshold]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Threshold(**entry) for entry in raw["thresholds"]]


def check_thresholds(
    metrics: dict[str, Any], thresholds: Iterable[Threshold]
) -> list[dict[str, Any]]:
    """Every threshold `metrics` breaks, critical ones first."""
    breaches = []
    for threshold in thresholds:
        reason = threshold.breach(metrics.get(threshold.metric))
        if reason is not None:
            breaches.append(
                {
                    "metric": threshold.metric,
                    "value": metrics.get(threshold.metric),
                    "reason": reason,
                    "critical": threshold.critical,
                    "why": threshold.why,
                }
            )
    return sorted(breaches, key=lambda breach: not breach["critical"])


__all__ = [
    "STEP_NODES",
    "INTERACTIVE_STEPS",
    "LONG_RUNNING_STEPS",
    "OUTPUT_TOKEN_ATTRIBUTES",
    "THRESHOLDS",
    "ActionObservation",
    "Threshold",
    "approval_metrics",
    "check_thresholds",
    "compute",
    "latency_metrics",
    "load_thresholds",
    "percentile",
    "recovery_from_spans",
    "routing_metrics",
    "tool_call_validity",
    "tool_selection_accuracy",
    "workflow_completion_rate",
]
