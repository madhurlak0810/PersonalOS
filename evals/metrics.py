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
| `routing_accuracy_<domain>` | `routing` cases | the same, over one domain's golden file |
| `forbidden_route_count` | `routing` cases | cases that reached a route their `never_routes` names; must be 0 |
| `approval_precision` | `approvals`, `domain_workflows` | approval prompts that were for consequential actions |
| `approval_recall` | `approvals`, `domain_workflows` | consequential actions that could not run unattended |
| `external_write_approval_recall` | `approvals`, `domain_workflows` | the same, for external writes only; must be 1.0 |
| `<approval metric>_<domain>` | `approvals`, `domain_workflows` | each of the three, per domain; the worst suite counts |
| `tool_call_validity_rate` | model spans | model attempts whose output passed its schema |
| `tool_selection_accuracy[_<domain>]` | `scenarios`, `domain_workflows` | runs that proposed exactly the expected actions |
| `workflow_completion_rate[_<domain>]` | `scenarios`, `domain_workflows` | runs that reached the expected end with no repair |
| `recovery_rate[_<domain>]` | `recovery`, `domain_workflows`, else spans | recoverable failures that finished by resuming |
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

from personalos.graphs import calendar as cal
from personalos.graphs import communications as comms
from personalos.graphs import file_task as ft
from personalos.graphs import job_search as js
from personalos.graphs import supervisor as sv
from personalos.observability.tracing import Attr

from .telemetry import SpanRecord

#: Which step each graph node belongs to, for latency. A bare name matches
#: that node in any graph; a `(graph, node)` pair only in that graph, for the
#: names two subgraphs share (`parse_request`). Nodes not listed (persistence,
#: approval checkpoints) are neither kind of step.
STEP_NODES: dict[str, tuple[str | tuple[str, str], ...]] = {
    "routing": (sv.LOAD_CONTEXT, sv.CLASSIFY_INTENT, sv.PLAN_WORK, sv.CLARIFY, sv.PLAN_TASK),
    "extraction": (js.CLASSIFY_RECRUITER_EVENTS,),
    "calendar_read": (
        (cal.GRAPH_NAME, cal.PARSE_REQUEST),
        (cal.GRAPH_NAME, cal.CHECK_AVAILABILITY),
        (cal.GRAPH_NAME, cal.PLAN_CALENDAR_CHANGE),
    ),
    "communications_read": (
        (comms.GRAPH_NAME, comms.PARSE_REQUEST),
        (comms.GRAPH_NAME, comms.TRIAGE_INBOX),
        (comms.GRAPH_NAME, comms.LOAD_THREAD),
        (comms.GRAPH_NAME, comms.LINK_APPLICATION),
    ),
    "file_read": (
        (ft.GRAPH_NAME, ft.DISCOVER_FILES),
        (ft.GRAPH_NAME, ft.PLAN_FILE_TASK),
        (ft.GRAPH_NAME, ft.READ_DOCUMENTS),
    ),
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
    "drafting": (
        js.PREPARE_APPLICATION_PACKET,
        js.DRAFT_FOLLOW_UP,
        (comms.GRAPH_NAME, comms.DRAFT_REPLY),
    ),
    "external_writes": (
        (cal.GRAPH_NAME, cal.EXECUTE_CALENDAR_CHANGES),
        (comms.GRAPH_NAME, comms.EXECUTE_SEND),
        (ft.GRAPH_NAME, ft.APPLY_FILE_CHANGES),
        (ft.GRAPH_NAME, ft.VERIFY_FILE_CHANGES),
    ),
}

#: A person is waiting on these: routing, and the reads that answer a
#: question or build what a reviewer is shown.
INTERACTIVE_STEPS = ("routing", "extraction", "calendar_read", "communications_read", "file_read")
#: These run in the background, or after a reviewer has said yes.
LONG_RUNNING_STEPS = ("discovery", "drafting", "external_writes")

_STEP_OF_NODE: dict[str | tuple[str, str], str] = {
    node: step for step, nodes in STEP_NODES.items() for node in nodes
}


def _step_of(span: SpanRecord) -> str | None:
    node = span.get(Attr.NODE)
    return _STEP_OF_NODE.get((span.get(Attr.GRAPH), node)) or _STEP_OF_NODE.get(node)


#: The domains per-domain metrics are reported for, as `<metric>_<domain>`.
DOMAINS = ("job", "file", "communications", "calendar")

#: What a routing route is called in the confusion matrix: the domain it hands
#: work to, `plan` for a cross-domain plan, or `clarify`.
ROUTE_LABELS = {
    "job_search": "job",
    "file_task": "file",
    "communications": "communications",
    "calendar": "calendar",
    "plan": "plan",
    "clarify": "clarify",
}
CONFUSION_LABELS = ("job", "file", "communications", "calendar", "plan", "clarify")


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
        step = _step_of(span)
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


def _decision_kind(expected: dict[str, Any]) -> str:
    """What kind of routing decision a case asks for."""
    route = expected["route"]
    if route == "clarify":
        reason = expected.get("reason")
        if reason in ("ambiguous", "access_escalation", "multi_domain"):
            return reason
        return "reject" if expected.get("domain") else "clarify"
    return "plan" if route == "plan" else "hand_off"


def routing_metrics(document: dict[str, Any]) -> dict[str, Any]:
    """Routing decisions: overall, per domain, per kind of case, and as a confusion matrix.

    A case's domain is the golden file it came from (`expected.tags.domain`);
    the matrix counts expected route against actual route, both named by
    `ROUTE_LABELS`, so a calendar request handed to communications is the
    `calendar` row's `communications` cell. `forbidden_routes` counts cases
    that reached a route their `never_routes` names -- an unsafe outcome,
    counted apart from a merely wrong one.
    """
    by_kind: dict[str, list[bool]] = defaultdict(list)
    by_domain: dict[str, list[bool]] = defaultdict(list)
    by_case_kind: dict[str, list[bool]] = defaultdict(list)
    confusion = {row: dict.fromkeys(CONFUSION_LABELS, 0) for row in CONFUSION_LABELS}
    forbidden: list[str] = []
    for case in _scored(_cases(document, "routing")):
        expected = case["expected"]
        ok = case["status"] == "pass"
        tags = expected.get("tags") or {}
        by_kind[_decision_kind(expected)].append(ok)
        by_domain[tags.get("domain", "unlabelled")].append(ok)
        by_case_kind[tags.get("kind", "unlabelled")].append(ok)
        actual = case.get("actual") or {}
        if actual.get("route") in ROUTE_LABELS:
            confusion[ROUTE_LABELS[expected["route"]]][ROUTE_LABELS[actual["route"]]] += 1
        never = expected.get("never_routes") or ()
        if never and (not actual or actual.get("route") in never):
            forbidden.append(case["case_id"])
    every = [ok for oks in by_kind.values() for ok in oks]

    def table(groups: dict[str, list[bool]]) -> dict[str, Any]:
        return {
            name: {"cases": len(oks), "accuracy": _ratio(sum(oks), len(oks))}
            for name, oks in sorted(groups.items())
        }

    return {
        "routing_accuracy": _ratio(sum(every), len(every)),
        "by_decision": table(by_kind),
        "by_domain": table(by_domain),
        "by_case_kind": table(by_case_kind),
        "confusion": {"labels": list(CONFUSION_LABELS), "matrix": confusion},
        "forbidden_routes": sorted(forbidden),
    }


def _workflow_cases(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Scored end-to-end runs and their domain: Job Search scenarios, then the new domains'."""
    runs = [("job", case) for case in _scored(_cases(document, "scenarios"))]
    runs += [
        ((case["expected"] or {}).get("tags", {}).get("domain", "unlabelled"), case)
        for case in _scored(_cases(document, "domain_workflows"))
    ]
    return runs


def tool_selection_accuracy(document: dict[str, Any], domain: str | None = None) -> float | None:
    """Runs that proposed exactly the actions the golden case expects. Optionally one domain."""
    labelled = [
        case
        for case_domain, case in _workflow_cases(document)
        if (domain is None or case_domain == domain)
        and "intents" in (case["expected"] or {})
        and case["actual"]
    ]
    right = sum(
        sorted(case["actual"]["intents"]) == sorted(case["expected"]["intents"])
        for case in labelled
    )
    return _ratio(right, len(labelled))


def workflow_completion_rate(document: dict[str, Any], domain: str | None = None) -> float | None:
    """Runs that reached the expected end state on their own. Optionally one domain."""
    scored = [
        case
        for case_domain, case in _workflow_cases(document)
        if domain is None or case_domain == domain
    ]
    return _ratio(sum(case["status"] == "pass" for case in scored), len(scored))


def _recovery(document: dict[str, Any]) -> dict[str, Any]:
    """Recoverable failures that finished by resuming, from both suites that inject them."""
    finished: dict[str, list[bool]] = defaultdict(list)
    for case in _scored(_cases(document, "recovery")):
        if (case["expected"] or {}).get("recoverable") and case["actual"]:
            finished["job"].append(bool(case["actual"]["completed"]))
    for case in _scored(_cases(document, "domain_workflows")):
        expected = case["expected"] or {}
        if expected.get("recoverable") and case["actual"]:
            domain = expected.get("tags", {}).get("domain", "unlabelled")
            finished[domain].append(bool(case["actual"]["completed"]))
    every = [ok for oks in finished.values() for ok in oks]
    return {
        "recovery_rate": _ratio(sum(every), len(every)),
        "by_domain": {
            domain: _ratio(sum(oks), len(oks)) for domain, oks in sorted(finished.items())
        },
        "recoverable_failures": len(every),
    }


def _suite_metric(document: dict[str, Any], suite: str, name: str) -> Any:
    return document.get("suites", {}).get(suite, {}).get("metrics", {}).get(name)


#: Suites that measure approval gating: policy and the Job Search graph, and
#: the new domains' subgraphs end to end.
APPROVAL_SUITES = ("approvals", "domain_workflows")
_APPROVAL_RATES = ("approval_precision", "approval_recall", "external_write_approval_recall")


def _worst(values: Iterable[Any]) -> float | None:
    """The lowest of the values reported; None when no suite reported one."""
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _approvals(document: dict[str, Any]) -> dict[str, Any]:
    """Approval precision and recall, overall and per domain, across `APPROVAL_SUITES`.

    Each rate is the worst any suite reported: a recall of 1.0 at the policy
    boundary does not excuse a subgraph that ran a write unattended.
    """
    names = [*_APPROVAL_RATES, *(f"{n}_{d}" for n in _APPROVAL_RATES for d in DOMAINS)]
    combined: dict[str, Any] = {
        name: _worst(_suite_metric(document, suite, name) for suite in APPROVAL_SUITES)
        for name in names
    }
    for count in ("approval_prompts", "consequential_actions", "external_writes"):
        combined[count] = sum(
            _suite_metric(document, suite, count) or 0 for suite in APPROVAL_SUITES
        )
    for listed in ("ungated_consequential", "unneeded_prompts"):
        combined[listed] = sorted(
            {
                action
                for suite in APPROVAL_SUITES
                for action in _suite_metric(document, suite, listed) or ()
            }
        )
    return combined


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
    approvals = _approvals(document)
    routing = routing_metrics(document)
    latency = latency_metrics(spans)
    validity = tool_call_validity(spans)
    span_recovery = recovery_from_spans(spans)
    duplicates = _duplicates(document)

    recovery = _recovery(document)
    metrics = {
        "routing_accuracy": routing["routing_accuracy"],
        **{
            f"routing_accuracy_{domain}": (routing["by_domain"].get(domain) or {}).get("accuracy")
            for domain in DOMAINS
        },
        "forbidden_route_count": (
            len(routing["forbidden_routes"]) if routing["by_domain"] else None
        ),
        "approval_precision": approvals["approval_precision"],
        "approval_recall": approvals["approval_recall"],
        "external_write_approval_recall": approvals["external_write_approval_recall"],
        **{
            f"{name}_{domain}": approvals[f"{name}_{domain}"]
            for name in _APPROVAL_RATES
            for domain in DOMAINS
        },
        "tool_call_validity_rate": validity["tool_call_validity_rate"],
        "tool_selection_accuracy": tool_selection_accuracy(document),
        **{
            f"tool_selection_accuracy_{domain}": tool_selection_accuracy(document, domain)
            for domain in DOMAINS
        },
        "workflow_completion_rate": workflow_completion_rate(document),
        **{
            f"workflow_completion_rate_{domain}": workflow_completion_rate(document, domain)
            for domain in DOMAINS
        },
        "recovery_rate": (
            recovery["recovery_rate"]
            if recovery["recoverable_failures"]
            else span_recovery["recovery_rate"]
        ),
        **{f"recovery_rate_{domain}": recovery["by_domain"].get(domain) for domain in DOMAINS},
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
        "recovery": {**recovery, "spans": span_recovery},
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
    "DOMAINS",
    "CONFUSION_LABELS",
    "ROUTE_LABELS",
    "APPROVAL_SUITES",
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
