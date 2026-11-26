"""The monitoring profile end to end: every panel has data, and a stalled workflow alerts.

    python scripts/smoke_observability.py            # builds the images first
    python scripts/smoke_observability.py --no-build # uses the images already built

What docs/METRICS.md and docs/ALERTS.md promise, checked against a real stack:

1. `docker compose -f docker-compose.yml -f compose.quickstart.yml --profile
   monitoring up -d`, with a short stall threshold and fast sampling for the
   test (a temporary override file), and wait for every service to be healthy;
2. run the quick start's workflow through approval (`smoke_compose.run_workflow`)
   and refuse one oversized request, so an API limit hit is recorded;
3. query every dashboard panel through Grafana until each has data
   (`check_dashboards.empty_panels`);
4. **stall a workflow**: stop the worker, start a workflow, and wait for
   `WorkflowStalled` to fire for that workflow id in Prometheus *and* reach
   the alert receiver (its `GET /alerts` and an `alert.firing` log line);
   while it fires, the alert-only panels must have data too;
5. **resolve it**: start the worker again, which runs the workflow on to its
   approval, and wait for the alert to clear in Prometheus and at the
   receiver (`alert.resolved`);
6. `docker compose down -v`, printing every service's logs first on failure.

Needs Docker with Compose v2.24 or newer, and `httpx`. Uses the fixed project
name `personalos`, so the stack must not already be running.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_dashboards  # noqa: E402
import smoke_compose  # noqa: E402
from smoke_quickstart import SmokeFailure  # noqa: E402

API_PORT = int(os.environ.get("PERSONALOS_API_PORT", "8000"))
GRAFANA_PORT = int(os.environ.get("PERSONALOS_GRAFANA_PORT", "3000"))

#: The stall threshold the test runs with, instead of 15 minutes.
STALL_SECONDS = 20
#: The test's override: a short stall threshold and fast sampling and export.
OVERRIDE = f"""
services:
  api:
    environment:
      WORKFLOW_STALL_SECONDS: "{STALL_SECONDS}"
      METRICS_STATE_INTERVAL_SECONDS: "5"
      METRICS_EXPORT_INTERVAL_SECONDS: "5"
  worker:
    environment:
      METRICS_EXPORT_INTERVAL_SECONDS: "5"
"""

HEALTHY = ("postgres", "api", "worker", "prometheus", "alertmanager", "alert-receiver", "grafana")
PANELS_TIMEOUT_SECONDS = 180
ALERT_TIMEOUT_SECONDS = 300


def _wait(what: str, done: Callable[[], bool], timeout: float, every: float = 5) -> None:
    """Poll `done` until it is true, or fail after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    print(f"==> waiting for {what}", flush=True)
    while time.monotonic() < deadline:
        if done():
            return
        time.sleep(every)
    raise SmokeFailure(f"timed out after {timeout:.0f}s waiting for {what}")


def _stalled_alerts(grafana: httpx.Client, workflow_id: str) -> list[dict]:
    return check_dashboards.query(
        grafana,
        f'ALERTS{{alertname="WorkflowStalled", alertstate="firing", workflow_id="{workflow_id}"}}',
    )


def _receiver_firing() -> list[dict]:
    out = smoke_compose.compose(
        "exec",
        "-T",
        "alert-receiver",
        "python",
        "-c",
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:9095/alerts', timeout=5).read().decode())",
    )
    return json.loads(out.strip().splitlines()[-1])["firing"]


def _receiver_logged(event: str, workflow_id: str) -> bool:
    for line in smoke_compose.compose(
        "logs", "--no-color", "--no-log-prefix", "alert-receiver"
    ).splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == event and record.get("workflow_id") == workflow_id:
            return True
    return False


def check_panels(grafana: httpx.Client) -> None:
    """Step 3: every non-alert panel has data."""

    def all_have_data() -> bool:
        missing = check_dashboards.empty_panels(grafana, alert_panels=False)
        if missing:
            print(f"    still empty: {', '.join(missing)}", flush=True)
        return not missing

    _wait("every dashboard panel to have data", all_have_data, PANELS_TIMEOUT_SECONDS, every=10)
    print("    every dashboard panel has data", flush=True)


def stall_and_resolve(api: httpx.Client, grafana: httpx.Client) -> str:
    """Steps 4-5: a stalled workflow fires WorkflowStalled; moving it clears it."""
    smoke_compose.compose("stop", "worker")
    started = api.post("/v1/chat", json={"message": "find me remote python backend jobs"})
    if started.status_code != 202:
        raise SmokeFailure(f"POST /v1/chat: {started.status_code} {started.text}")
    workflow_id = started.json()["workflow_id"]
    print(f"    queued workflow {workflow_id} with no worker running", flush=True)

    _wait(
        f"WorkflowStalled to fire for {workflow_id}",
        lambda: bool(_stalled_alerts(grafana, workflow_id)),
        ALERT_TIMEOUT_SECONDS,
    )
    _wait(
        "the alert receiver to log it",
        lambda: any(
            alert["labels"].get("workflow_id") == workflow_id for alert in _receiver_firing()
        )
        and _receiver_logged("alert.firing", workflow_id),
        120,
    )
    print("    WorkflowStalled fired and reached the alert receiver", flush=True)
    empty = check_dashboards.empty_panels(grafana, alert_panels=True)
    if empty:
        raise SmokeFailure(f"alert panels with no data while an alert fires: {empty}")
    print("    the alert panels show it", flush=True)

    smoke_compose.compose("start", "worker")
    _wait(
        f"workflow {workflow_id} to reach its approval",
        lambda: api.get(f"/v1/workflows/{workflow_id}").json()["status"] == "waiting",
        ALERT_TIMEOUT_SECONDS,
    )
    _wait(
        "WorkflowStalled to clear",
        lambda: not _stalled_alerts(grafana, workflow_id),
        ALERT_TIMEOUT_SECONDS,
    )
    _wait(
        "the alert receiver to log the resolution",
        lambda: _receiver_logged("alert.resolved", workflow_id)
        and not any(a["labels"].get("workflow_id") == workflow_id for a in _receiver_firing()),
        120,
    )
    print("    WorkflowStalled cleared once the workflow moved", flush=True)
    return workflow_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-build", action="store_true", help="Use the images already built.")
    parser.add_argument("--keep", action="store_true", help="Leave the stack running.")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as override:
        override.write(OVERRIDE)
    smoke_compose.COMPOSE = [
        *smoke_compose.COMPOSE,
        "-f",
        override.name,
        "--profile",
        "monitoring",
    ]
    smoke_compose.HEALTHY = HEALTHY

    failed = False
    try:
        smoke_compose.compose("up", "-d", *([] if args.no_build else ["--build"]), capture=False)
        smoke_compose.wait_until_healthy()
        smoke_compose.run_workflow()
        with (
            httpx.Client(base_url=f"http://127.0.0.1:{API_PORT}", timeout=10) as api,
            httpx.Client(base_url=f"http://127.0.0.1:{GRAFANA_PORT}", timeout=10) as grafana,
        ):
            # Refused by the body cap before authentication: an API limit hit.
            if api.post("/v1/chat", content=b"x" * (2 * 1024 * 1024)).status_code != 413:
                raise SmokeFailure("an oversized request was not refused with 413")
            check_panels(grafana)
            api.headers["Authorization"] = (
                "Bearer " + smoke_compose._cli("api", "auth", "token").strip().splitlines()[-1]
            )
            workflow_id = stall_and_resolve(api, grafana)
        print(f"observability smoke test passed (stalled workflow {workflow_id})")
    except (SmokeFailure, httpx.HTTPError) as exc:
        failed = True
        print(f"\nOBSERVABILITY SMOKE TEST FAILED: {exc}", file=sys.stderr)
    finally:
        if failed:
            smoke_compose.compose("ps", "--all", check=False, capture=False)
            smoke_compose.compose("logs", "--no-color", check=False, capture=False)
        if not args.keep:
            smoke_compose.compose("down", "-v", "--remove-orphans", check=False, capture=False)
        os.unlink(override.name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
