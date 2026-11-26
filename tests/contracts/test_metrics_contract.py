"""The metric contract: code, docs, dashboards and alert rules agree on every name and label.

A dashboard panel whose metric was renamed does not fail; it goes quiet. So
the names are checked here, in four directions:

- `docs/METRICS.md`'s catalog table lists exactly `metrics.CATALOG`, row for
  row (name, kind, unit, labels, source, meaning). Removing or renaming a
  metric in either place alone fails.
- Every series a dashboard panel (`scripts/build_dashboards.py`) or an alert
  rule (`deploy/prometheus/rules/`) queries is a catalog series, and every
  label it filters or groups by is one that metric has. Renaming a metric in
  the catalog and the docs but not in a panel fails.
- Every catalog metric is on at least one dashboard, and is recorded
  somewhere in the code.
- Every alert has a summary and a runbook section in docs/ALERTS.md; every
  long-running step is a registered node; the dashboard JSON is current.

The last tests show the checks bite: a catalog with one metric renamed, or a
doc with one row removed, is reported.
"""

import dataclasses
import re
import sys
from pathlib import Path

import pytest

from personalos.graphs.contracts import NODE_CONTRACTS
from personalos.observability import metrics
from personalos.observability.metrics import CATALOG, LONG_RUNNING_STEPS, MetricKind, MetricSpec

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import build_dashboards  # noqa: E402

METRICS_DOC = REPO / "docs" / "METRICS.md"
ALERTS_DOC = REPO / "docs" / "ALERTS.md"
RULES = sorted((REPO / "deploy" / "prometheus" / "rules").glob("*.yml"))

#: Labels Prometheus or the collector adds, and the histogram bucket bound.
_ADDED_LABELS = frozenset({"job", "instance", "le"})
#: Prometheus' own alert series, and its labels.
_ALERTS_SERIES = "ALERTS"
_ALERTS_LABELS = frozenset({"alertname", "alertstate", "severity"})

_SERIES = re.compile(r"\b(personalos_[a-z0-9_]+|ALERTS)\b")
_GROUPING = re.compile(r"\b(?:by|without)\s*\(([^)]*)\)")
_MATCHER = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=~|!~|!=|=)\s*\"")
_SELECTOR = re.compile(r"\{([^{}]*)\}")


# --- The checks ---------------------------------------------------------------------


def _doc_rows(text: str) -> list[list[str]]:
    start = text.index("<!-- metrics-catalog:start -->")
    end = text.index("<!-- metrics-catalog:end -->")
    rows = []
    for line in text[start:end].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].startswith("`"):
            rows.append(cells)
    return rows


def _spec_row(spec: MetricSpec) -> list[str]:
    return [
        f"`{spec.prometheus_name}`",
        f"`{spec.name}`",
        spec.kind.value,
        spec.unit or "—",
        ", ".join(f"`{label}`" for label in spec.labels) or "—",
        spec.source.value,
        spec.description,
    ]


def doc_problems(catalog: tuple[MetricSpec, ...], doc_text: str) -> list[str]:
    """Where docs/METRICS.md's catalog table and `catalog` disagree."""
    documented = {row[1]: row for row in _doc_rows(doc_text)}
    expected = {f"`{spec.name}`": _spec_row(spec) for spec in catalog}
    problems = [
        f"{name} is in the catalog but not documented"
        for name in expected.keys() - documented.keys()
    ]
    problems += [
        f"{name} is documented but not in the catalog"
        for name in documented.keys() - expected.keys()
    ]
    problems += [
        f"{name} is documented as {documented[name]} but the catalog says {expected[name]}"
        for name in expected.keys() & documented.keys()
        if documented[name] != expected[name]
    ]
    return problems


def _series_index(catalog: tuple[MetricSpec, ...]) -> dict[str, MetricSpec]:
    return {series: spec for spec in catalog for series in spec.prometheus_series}


def _labels_used(expr: str) -> set[str]:
    used: set[str] = set()
    for group in _GROUPING.findall(expr):
        used.update(label.strip() for label in group.split(",") if label.strip())
    for selector in _SELECTOR.findall(expr):
        used.update(_MATCHER.findall(selector))
    return used


def query_problems(where: str, expr: str, catalog: tuple[MetricSpec, ...]) -> list[str]:
    """Series in `expr` the catalog does not define, and labels none of them has."""
    index = _series_index(catalog)
    problems = []
    allowed = set(_ADDED_LABELS)
    for series in _SERIES.findall(expr):
        if series == _ALERTS_SERIES:
            allowed |= _ALERTS_LABELS | {label for spec in catalog for label in spec.labels}
            continue
        spec = index.get(series)
        if spec is None:
            problems.append(f"{where}: queries `{series}`, which is not a catalog series")
            continue
        allowed |= set(spec.labels)
    for label in sorted(_labels_used(expr) - allowed):
        problems.append(f"{where}: uses label `{label}`, which no metric it queries has")
    return problems


def _rules() -> list[dict]:
    return [
        rule
        for path in RULES
        for group in yaml.safe_load(path.read_text())["groups"]
        for rule in group["rules"]
    ]


def all_queries() -> list[tuple[str, str]]:
    """`(where, expr)` for every dashboard target and alert rule."""
    queries = [
        (f"dashboard {uid} / {title}", expr)
        for uid, title, expr in build_dashboards.panel_queries()
    ]
    queries += [(f"alert {rule['alert']}", rule["expr"]) for rule in _rules()]
    return queries


def dashboard_coverage_problems(catalog: tuple[MetricSpec, ...]) -> list[str]:
    queried = {
        series
        for _, _, expr in build_dashboards.panel_queries()
        for series in _SERIES.findall(expr)
    }
    return [
        f"{spec.name} is on no dashboard"
        for spec in catalog
        if not queried & set(spec.prometheus_series)
    ]


# --- The contract -------------------------------------------------------------------


def test_the_documented_catalog_is_the_catalog():
    assert doc_problems(CATALOG, METRICS_DOC.read_text()) == []


def test_every_series_a_dashboard_or_rule_queries_is_in_the_catalog_with_its_labels():
    problems = [
        problem for where, expr in all_queries() for problem in query_problems(where, expr, CATALOG)
    ]
    assert problems == []


def test_every_catalog_metric_is_on_a_dashboard():
    assert dashboard_coverage_problems(CATALOG) == []


def test_every_catalog_metric_is_recorded_somewhere():
    """A catalog entry nothing records would be a documented, permanently empty panel."""
    sources = "\n".join(
        path.read_text()
        for root in ("personalos", "apps")
        for path in (REPO / root).rglob("*.py")
        if path != REPO / "personalos" / "observability" / "metrics.py"
    )
    unrecorded = [
        name
        for name, value in vars(metrics.Metric).items()
        if isinstance(value, MetricSpec) and f"Metric.{name}" not in sources
    ]
    assert unrecorded == []


def test_metric_names_are_unique_and_follow_the_naming_rules():
    names = [spec.name for spec in CATALOG]
    series = [s for spec in CATALOG for s in spec.prometheus_series]
    assert len(names) == len(set(names))
    assert len(series) == len(set(series))
    for spec in CATALOG:
        assert re.fullmatch(r"personalos(\.[a-z_]+)+", spec.name), spec.name
        assert spec.unit in ("", "s"), spec.name
        assert spec.description, spec.name
        if spec.kind is MetricKind.HISTOGRAM:
            assert spec.unit == "s" and spec.buckets == tuple(sorted(spec.buckets)), spec.name


def test_no_metric_has_a_label_that_could_carry_user_content():
    forbidden = {
        "email",
        "subject",
        "body",
        "message",
        "text",
        "token",
        "user",
        "actor_id",
        "query",
    }
    for spec in CATALOG:
        assert not forbidden & set(spec.labels), spec.name


def test_the_prometheus_names_are_the_collectors_translation():
    by_name = {spec.name: spec for spec in CATALOG}

    assert by_name["personalos.api.request.duration"].prometheus_series == (
        "personalos_api_request_duration_seconds_bucket",
        "personalos_api_request_duration_seconds_sum",
        "personalos_api_request_duration_seconds_count",
    )
    assert by_name["personalos.api.limit_hits"].prometheus_name == "personalos_api_limit_hits_total"
    assert by_name["personalos.outbox.parked"].prometheus_name == "personalos_outbox_parked"
    assert (
        by_name["personalos.workflow.stalled_for"].prometheus_name
        == "personalos_workflow_stalled_for_seconds"
    )


def test_every_alert_has_a_summary_and_a_runbook_section():
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in ALERTS_DOC.read_text().splitlines()
        if line.startswith("## ")
    }
    rules = _rules()
    assert rules
    for rule in rules:
        name = rule["alert"]
        annotations = rule.get("annotations", {})
        assert annotations.get("summary"), name
        assert annotations.get("runbook") == f"docs/ALERTS.md#{name.lower()}", name
        assert name.lower() in headings, f"docs/ALERTS.md has no section for {name}"
        assert rule.get("labels", {}).get("severity") in ("warning", "critical"), name


def test_the_required_alerts_exist():
    names = {rule["alert"] for rule in _rules()}

    assert {
        "WorkflowStalled",
        "DeadLettersGrowing",
        "ApiErrorRateHigh",
        "IntegrationUnhealthy",
        "OutboxBacklogGrowing",
    } <= names


def test_every_long_running_step_is_a_registered_node():
    assert LONG_RUNNING_STEPS <= set(NODE_CONTRACTS)


def test_the_dashboard_json_is_generated_from_the_definitions():
    for path, text in build_dashboards.outputs().items():
        assert path.exists(), f"{path} is missing; run scripts/build_dashboards.py"
        assert path.read_text() == text, f"{path} is stale; run scripts/build_dashboards.py"


# --- The checks bite ----------------------------------------------------------------


def _renamed(spec: MetricSpec, name: str) -> tuple[MetricSpec, ...]:
    return tuple(dataclasses.replace(s, name=name) if s is spec else s for s in CATALOG)


def test_renaming_a_metric_in_the_catalog_alone_fails_the_doc_check():
    catalog = _renamed(metrics.Metric.OUTBOX_PARKED, "personalos.outbox.dead_letters")

    problems = doc_problems(catalog, METRICS_DOC.read_text())

    assert any("personalos.outbox.dead_letters" in p for p in problems)
    assert any("personalos.outbox.parked" in p for p in problems)


def test_removing_a_documented_metric_fails_the_doc_check():
    text = METRICS_DOC.read_text()
    row = next(line for line in text.splitlines() if "`personalos.workflow.stalled_for`" in line)

    problems = doc_problems(CATALOG, text.replace(row + "\n", ""))

    assert problems == ["`personalos.workflow.stalled_for` is in the catalog but not documented"]


def test_renaming_a_metric_without_its_dashboards_and_rules_fails_the_query_check():
    catalog = _renamed(metrics.Metric.WORKFLOW_STALLED_FOR, "personalos.workflow.idle")

    problems = [p for where, expr in all_queries() for p in query_problems(where, expr, catalog)]

    assert any("dashboard personalos-workflows / Stalled workflows" in p for p in problems)
    assert any("alert WorkflowStalled" in p for p in problems)


def test_a_label_a_metric_does_not_have_fails_the_query_check():
    problems = query_problems("test", "sum by (actor_id) (personalos_workflows_active)", CATALOG)

    assert problems == ["test: uses label `actor_id`, which no metric it queries has"]
