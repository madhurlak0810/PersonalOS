"""Query every dashboard panel against a running stack and fail if one has no data.

    python scripts/check_dashboards.py                       # Grafana on 127.0.0.1:3000
    python scripts/check_dashboards.py --grafana http://127.0.0.1:3000
    python scripts/check_dashboards.py --alert-panels        # only the alert-only panels

Each panel's PromQL comes from `scripts/build_dashboards.py` (the same
definitions the dashboard JSON is generated from) and is sent through
Grafana's provisioned Prometheus datasource, so this also checks that Grafana
reaches Prometheus. A panel passes when at least one of its queries returns
a series. Grafana's `$__rate_interval` and `$__range` are replaced with fixed
windows.

Panels in `ALERT_ONLY_PANELS` (firing alerts, stalled workflows) only have
data while an alert fires, so they are skipped unless `--alert-panels` is
given; `scripts/smoke_observability.py` checks them while it holds the
stalled-workflow alert firing.

Exit status 0 when every checked panel has data, 1 otherwise. Needs `httpx`.
"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dashboards  # noqa: E402

#: What Grafana's own variables are replaced with.
SUBSTITUTIONS = {"$__rate_interval": "5m", "$__range": "1h", "$__interval": "1m"}


def promql(expr: str) -> str:
    """`expr` with Grafana's variables replaced, ready for Prometheus."""
    for variable, value in SUBSTITUTIONS.items():
        expr = expr.replace(variable, value)
    return expr


def query(client: httpx.Client, expr: str) -> list[dict]:
    """The instant-query result for `expr`, through Grafana's Prometheus datasource."""
    response = client.get(
        "/api/datasources/proxy/uid/prometheus/api/v1/query", params={"query": promql(expr)}
    )
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "success":
        raise RuntimeError(f"query failed: {body}")
    return body["data"]["result"]


def empty_panels(client: httpx.Client, *, alert_panels: bool) -> list[str]:
    """`dashboard / panel` for every checked panel none of whose queries returned a series."""
    panels: dict[tuple[str, str], bool] = {}
    for uid, title, expr in build_dashboards.panel_queries():
        if (title in build_dashboards.ALERT_ONLY_PANELS) != alert_panels:
            continue
        has_data = bool(query(client, expr))
        panels[(uid, title)] = panels.get((uid, title), False) or has_data
    return [f"{uid} / {title}" for (uid, title), ok in panels.items() if not ok]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grafana", default="http://127.0.0.1:3000")
    parser.add_argument(
        "--alert-panels",
        action="store_true",
        help="Check only the panels that have data while an alert fires.",
    )
    args = parser.parse_args()
    with httpx.Client(base_url=args.grafana, timeout=10) as client:
        empty = empty_panels(client, alert_panels=args.alert_panels)
    if empty:
        print("panels with no data:\n  " + "\n  ".join(empty), file=sys.stderr)
        return 1
    print("every checked panel has data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
