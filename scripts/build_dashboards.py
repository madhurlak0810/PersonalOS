"""Generate the Grafana dashboards in deploy/grafana/dashboards/ from one Python definition.

    python scripts/build_dashboards.py          # write the JSON files
    python scripts/build_dashboards.py --check  # exit 1 if they are out of date

The dashboards are code: each panel is a title and the PromQL it charts, here,
and the JSON Grafana loads is generated from that. Hand-edited JSON would drift
from the metric contract unnoticed; this way `tests/contracts/
test_metrics_contract.py` reads the same panel list (`panel_queries`) and
fails when a panel queries a series the metrics catalog does not define, and
`scripts/check_dashboards.py` runs every query against a live stack.

`$__rate_interval` and `$__range` are Grafana's own variables; the panel
check substitutes fixed windows for them. Panels in `ALERT_ONLY_PANELS` show
data only while an alert fires (the firing alerts, the stalled workflows); the
check expects data from them only while the stalled-workflow alert it
provokes is firing.

Standard library only.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "deploy" / "grafana" / "dashboards"
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}

#: Panels whose queries return data only while an alert is firing.
ALERT_ONLY_PANELS = frozenset({"Firing alerts", "Stalled workflows"})

_RATE = "$__rate_interval"


@dataclass(frozen=True)
class Target:
    expr: str
    legend: str = ""


@dataclass(frozen=True)
class Panel:
    title: str
    targets: tuple[Target, ...]
    kind: str = "timeseries"
    unit: str = "short"
    description: str = ""
    #: Grid width out of 24; height follows the kind.
    width: int = 12
    mappings: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Dashboard:
    uid: str
    title: str
    description: str
    panels: tuple[Panel, ...] = field(default_factory=tuple)


def _ratio_with_zero(numerator: str, denominator: str, by: str) -> str:
    """`numerator / denominator` by `by`, with 0 where the numerator has no series."""
    return (
        f"(sum by ({by}) ({numerator}) or sum by ({by}) ({denominator}) * 0)"
        f" / sum by ({by}) ({denominator})"
    )


def _p(quantile: float, metric: str, by: str, selector: str = "") -> str:
    return (
        f"histogram_quantile({quantile}, sum by ({by}, le) "
        f"(rate({metric}_bucket{selector}[{_RATE}])))"
    )


FIRING_ALERTS = Panel(
    "Firing alerts",
    (Target('ALERTS{alertstate="firing"}', "{{alertname}}"),),
    kind="table",
    width=24,
    description=(
        "Alerts firing now, from Prometheus. Each is also a line in "
        "`docker compose logs alert-receiver`; docs/ALERTS.md has the runbook."
    ),
)

_HEALTH_MAPPINGS = (
    {
        "type": "value",
        "options": {
            "0": {"text": "healthy", "color": "green", "index": 0},
            "1": {"text": "degraded", "color": "orange", "index": 1},
            "2": {"text": "down", "color": "red", "index": 2},
        },
    },
)

API = "personalos_api_request_duration_seconds"
RUNS = "personalos_workflow_runs_total"
MODEL = "personalos_model_request_duration_seconds"

DASHBOARDS: tuple[Dashboard, ...] = (
    Dashboard(
        "personalos-api",
        "PersonalOS · API",
        "Request rate, latency and error rate per route, and API limit hits.",
        (
            FIRING_ALERTS,
            Panel(
                "Requests per second by route",
                (Target(f"sum by (route) (rate({API}_count[{_RATE}]))", "{{route}}"),),
                unit="reqps",
            ),
            Panel(
                "p95 latency by route",
                (Target(_p(0.95, API, "route"), "{{route}}"),),
                unit="s",
            ),
            Panel(
                "Error rate by route (5xx share)",
                (
                    Target(
                        _ratio_with_zero(
                            f'rate({API}_count{{status_class="5xx"}}[{_RATE}])',
                            f"rate({API}_count[{_RATE}])",
                            "route",
                        ),
                        "{{route}}",
                    ),
                ),
                unit="percentunit",
                description="ApiErrorRateHigh fires above 5% for 10 minutes.",
            ),
            Panel(
                "Responses by status class",
                (
                    Target(
                        f"sum by (status_class) (rate({API}_count[{_RATE}]))", "{{status_class}}"
                    ),
                ),
                unit="reqps",
            ),
            Panel(
                "API limit hits by route and limit",
                (
                    Target(
                        f"sum by (route, limit) (rate(personalos_api_limit_hits_total[{_RATE}]))",
                        "{{limit}} {{route}}",
                    ),
                ),
                unit="reqps",
                width=24,
                description=(
                    "Requests refused by a Phase L2 limit (rate, size, concurrency, auth "
                    "throttle). Empty until the first refusal."
                ),
            ),
        ),
    ),
    Dashboard(
        "personalos-workflows",
        "PersonalOS · Workflows",
        "Throughput, duration and completion rate, step latency, approval wait, stalls.",
        (
            FIRING_ALERTS,
            Panel(
                "Active workflows",
                (Target("sum(personalos_workflows_active)"),),
                kind="stat",
                width=6,
            ),
            Panel(
                "Stalled workflows",
                (Target("max by (workflow_id, reason) (personalos_workflow_stalled_for_seconds)"),),
                kind="table",
                unit="s",
                width=12,
                description=(
                    "Workflows with no progress past WORKFLOW_STALL_SECONDS, or parked at an "
                    "approval past its expiry. WorkflowStalled fires for each after 1 minute."
                ),
            ),
            Panel(
                "Stalled workflow count by reason",
                (Target("sum by (reason) (personalos_workflows_stalled)", "{{reason}}"),),
                kind="stat",
                width=6,
            ),
            Panel(
                "Runs finished per second by graph and outcome",
                (
                    Target(
                        f"sum by (graph, outcome) (rate({RUNS}[{_RATE}]))",
                        "{{graph}} {{outcome}}",
                    ),
                ),
                unit="ops",
            ),
            Panel(
                "Completion rate by graph",
                (
                    Target(
                        f'sum by (graph) (rate({RUNS}{{outcome="completed"}}[{_RATE}]))'
                        f' / sum by (graph) (rate({RUNS}{{outcome=~"completed|failed"}}[{_RATE}]))',
                        "{{graph}}",
                    ),
                ),
                unit="percentunit",
                description="Completed runs over runs that finished (completed or failed).",
            ),
            Panel(
                "Run duration by graph (p50, p95)",
                (
                    Target(
                        _p(0.5, "personalos_workflow_run_duration_seconds", "graph"),
                        "p50 {{graph}}",
                    ),
                    Target(
                        _p(0.95, "personalos_workflow_run_duration_seconds", "graph"),
                        "p95 {{graph}}",
                    ),
                ),
                unit="s",
            ),
            Panel(
                "Approval wait (p50, p95)",
                (
                    Target(
                        _p(0.5, "personalos_approval_wait_duration_seconds", "outcome"),
                        "p50 {{outcome}}",
                    ),
                    Target(
                        _p(0.95, "personalos_approval_wait_duration_seconds", "outcome"),
                        "p95 {{outcome}}",
                    ),
                ),
                unit="s",
                description="From an approval being requested to it being approved, denied or expired.",
            ),
            Panel(
                "Step latency p95, quick steps",
                (
                    Target(
                        _p(
                            0.95,
                            "personalos_workflow_step_duration_seconds",
                            "graph, step",
                            '{step_class="quick"}',
                        ),
                        "{{graph}}.{{step}}",
                    ),
                ),
                unit="s",
            ),
            Panel(
                "Step latency p95, long-running steps",
                (
                    Target(
                        _p(
                            0.95,
                            "personalos_workflow_step_duration_seconds",
                            "graph, step",
                            '{step_class="long_running"}',
                        ),
                        "{{graph}}.{{step}}",
                    ),
                ),
                unit="s",
                description=(
                    "Steps that wait on a job board, mailbox, calendar, model or subgraph "
                    "(LONG_RUNNING_STEPS in personalos/observability/metrics.py)."
                ),
            ),
        ),
    ),
    Dashboard(
        "personalos-worker",
        "PersonalOS · Worker and outbox",
        "Queue depth, commands, retries, dead letters and the outbox backlog.",
        (
            FIRING_ALERTS,
            Panel(
                "Command queue depth",
                (Target("sum by (state) (personalos_worker_queue_depth)", "{{state}}"),),
            ),
            Panel(
                "Commands settled per second by kind and outcome",
                (
                    Target(
                        f"sum by (kind, outcome) (rate(personalos_worker_commands_total[{_RATE}]))",
                        "{{kind}} {{outcome}}",
                    ),
                ),
                unit="ops",
            ),
            Panel(
                "Retries per second by source",
                (
                    Target(
                        f"sum by (source) (rate(personalos_worker_retries_total[{_RATE}]))",
                        "{{source}}",
                    ),
                ),
                unit="ops",
            ),
            Panel(
                "Dead-lettered and quarantined events",
                (Target("sum by (status) (personalos_outbox_parked)", "{{status}}"),),
                description="DeadLettersGrowing fires when this goes up.",
            ),
            Panel(
                "Outbox backlog",
                (Target("sum by (state) (personalos_outbox_backlog)", "{{state}}"),),
            ),
            Panel(
                "Oldest undelivered outbox event",
                (Target("max(personalos_outbox_oldest_pending_age_seconds)", "age"),),
                unit="s",
                description="OutboxBacklogGrowing fires past 15 minutes.",
            ),
            Panel(
                "Outbox deliveries per second by outcome",
                (
                    Target(
                        f"sum by (outcome) (rate(personalos_outbox_deliveries_total[{_RATE}]))",
                        "{{outcome}}",
                    ),
                ),
                unit="ops",
                width=24,
            ),
        ),
    ),
    Dashboard(
        "personalos-models-integrations",
        "PersonalOS · Models and integrations",
        "Model gateway latency and error rate per provider; integration health (Phase J8).",
        (
            FIRING_ALERTS,
            Panel(
                "Model call p95 latency by provider",
                (Target(_p(0.95, MODEL, "provider"), "{{provider}}"),),
                unit="s",
            ),
            Panel(
                "Model error rate by provider",
                (
                    Target(
                        _ratio_with_zero(
                            f'rate({MODEL}_count{{outcome!="ok"}}[{_RATE}])',
                            f"rate({MODEL}_count[{_RATE}])",
                            "provider",
                        ),
                        "{{provider}}",
                    ),
                ),
                unit="percentunit",
                description="Errors and schema-invalid answers over all attempts.",
            ),
            Panel(
                "Model calls per second by provider and operation",
                (
                    Target(
                        f"sum by (provider, operation) (rate({MODEL}_count[{_RATE}]))",
                        "{{provider}} {{operation}}",
                    ),
                ),
                unit="ops",
                width=24,
            ),
            Panel(
                "Integration health",
                (
                    Target(
                        "max by (integration) (personalos_integration_health_status)",
                        "{{integration}}",
                    ),
                ),
                kind="stat",
                mappings=_HEALTH_MAPPINGS,
                description="IntegrationUnhealthy fires after 30 minutes degraded or down.",
            ),
            Panel(
                "Health check outcomes per second",
                (
                    Target(
                        "sum by (integration, status) "
                        f"(rate(personalos_integration_health_checks_total[{_RATE}]))",
                        "{{integration}} {{status}}",
                    ),
                ),
                unit="ops",
            ),
            Panel(
                "Health transitions in range",
                (
                    Target(
                        "sum by (integration, from_status, status) "
                        "(increase(personalos_integration_health_transitions_total[$__range]))",
                        "{{integration}} {{from_status}} -> {{status}}",
                    ),
                ),
                kind="table",
                width=24,
            ),
        ),
    ),
)


# --- Rendering ----------------------------------------------------------------------

_HEIGHT = {"timeseries": 8, "table": 7, "stat": 4}


def _panel_json(panel: Panel, panel_id: int, x: int, y: int) -> dict:
    field_config: dict = {"defaults": {"unit": panel.unit}, "overrides": []}
    if panel.mappings:
        field_config["defaults"]["mappings"] = list(panel.mappings)
    body: dict = {
        "id": panel_id,
        "type": panel.kind,
        "title": panel.title,
        "description": panel.description,
        "datasource": DATASOURCE,
        "gridPos": {"x": x, "y": y, "w": panel.width, "h": _HEIGHT[panel.kind]},
        "fieldConfig": field_config,
        "targets": [
            {
                "refId": chr(ord("A") + index),
                "datasource": DATASOURCE,
                "expr": target.expr,
                "legendFormat": target.legend,
                "instant": panel.kind in ("table", "stat"),
                "range": panel.kind == "timeseries",
            }
            for index, target in enumerate(panel.targets)
        ],
    }
    if panel.kind == "table":
        body["options"] = {"showHeader": True}
        body["transformations"] = [{"id": "labelsToFields", "options": {}}]
    elif panel.kind == "stat":
        body["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value"}
    else:
        body["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}}
    return body


def render(dashboard: Dashboard) -> dict:
    """The Grafana dashboard JSON for one definition."""
    panels = []
    x = y = row_height = 0
    for panel_id, panel in enumerate(dashboard.panels, start=1):
        if x + panel.width > 24:
            x, y, row_height = 0, y + row_height, 0
        panels.append(_panel_json(panel, panel_id, x, y))
        x += panel.width
        row_height = max(row_height, _HEIGHT[panel.kind])
    return {
        "uid": dashboard.uid,
        "title": dashboard.title,
        "description": dashboard.description,
        "tags": ["personalos"],
        "timezone": "browser",
        "editable": False,
        "graphTooltip": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "schemaVersion": 39,
        "version": 1,
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": panels,
    }


def outputs() -> dict[Path, str]:
    """Every dashboard file, by path, as it should be on disk."""
    return {
        OUT / f"{dashboard.uid}.json": json.dumps(render(dashboard), indent=2) + "\n"
        for dashboard in DASHBOARDS
    }


def panel_queries() -> list[tuple[str, str, str]]:
    """`(dashboard uid, panel title, expr)` for every target of every panel."""
    return [
        (dashboard.uid, panel.title, target.expr)
        for dashboard in DASHBOARDS
        for panel in dashboard.panels
        for target in panel.targets
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Fail if the files are stale.")
    args = parser.parse_args()
    expected = outputs()
    if args.check:
        stale = sorted(
            str(path.relative_to(REPO))
            for path, text in expected.items()
            if not path.exists() or path.read_text() != text
        )
        extra = sorted(
            str(path.relative_to(REPO)) for path in OUT.glob("*.json") if path not in expected
        )
        if stale or extra:
            print(
                "dashboards are out of date; run `python scripts/build_dashboards.py`:\n  "
                + "\n  ".join(stale + [f"{p} (not generated)" for p in extra]),
                file=sys.stderr,
            )
            return 1
        print(f"{len(expected)} dashboards are up to date")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in expected.items():
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
