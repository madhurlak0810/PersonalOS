"""Bring the compose stack up, check its security posture, run a workflow, tear down.

    python scripts/smoke_compose.py            # builds the images first
    python scripts/smoke_compose.py --no-build # uses the images already built

What `IMPLEMENTATION_GUIDE.md#containerized-quick-start` tells someone to do
with Docker, checked, so CI fails when it stops working:

1. `docker compose -f docker-compose.yml -f compose.quickstart.yml up -d`, then
   wait until the one-shot services (`db-password`, `migrate`) have exited 0
   and `postgres`, `api` and `worker` report healthy;
2. check the posture the compose file promises: the API published on
   127.0.0.1 only, no host port for Postgres (and nothing listening on the
   host's 5432), the API container running as a non-root user on a read-only
   root filesystem, and the collector answering its health check;
3. ingest the sample resume and create the sample profile with the CLI inside
   the worker container, read the API token with the CLI inside the API
   container;
4. queue the quick start's job search over the published port, approve it,
   and check it stops at the unbound submitter port, exactly as
   `scripts/smoke_quickstart.py` does;
5. `docker compose down -v`, whatever happened, printing every service's logs
   first if something failed.

Needs Docker with Compose v2.24 or newer, and `httpx` (the only non-stdlib
import, shared with `smoke_quickstart.py`). The stack uses the fixed project
name `personalos`, so it must not already be running.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_quickstart import SmokeFailure, _search_and_approve  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
COMPOSE = ["docker", "compose", "-f", "docker-compose.yml", "-f", "compose.quickstart.yml"]
API_PORT = int(os.environ.get("PERSONALOS_API_PORT", "8000"))
POSTGRES_PORT = 5432

#: Services that run once and must exit 0 before the API and worker start.
ONE_SHOT = ("db-password", "migrate")
#: Services whose health check must pass.
HEALTHY = ("postgres", "api", "worker")

STARTUP_TIMEOUT_SECONDS = 300


def compose(*args: str, check: bool = True, capture: bool = True) -> str:
    """Run a `docker compose` subcommand against the quick start stack."""
    print(f"==> docker compose {' '.join(args)}", flush=True)
    done = subprocess.run(
        [*COMPOSE, *args], cwd=REPO, capture_output=capture, text=True, check=False
    )
    if check and done.returncode != 0:
        raise SmokeFailure(
            f"docker compose {' '.join(args)} exited {done.returncode}\n"
            f"{done.stdout or ''}\n{done.stderr or ''}"
        )
    return done.stdout or ""


def _containers() -> dict[str, dict]:
    """Every container of the stack, by service, as `docker compose ps` reports it."""
    out = compose("ps", "--all", "--format", "json")
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    # Older Compose versions print one JSON array instead of one object per line.
    if len(rows) == 1 and isinstance(rows[0], list):
        rows = rows[0]
    return {row["Service"]: row for row in rows}


def wait_until_healthy() -> None:
    """Step 1's wait: one-shots exited 0, the long-running services healthy."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last = ""
    while time.monotonic() < deadline:
        containers = _containers()
        for name in ONE_SHOT:
            row = containers.get(name)
            if row and row.get("State") == "exited" and row.get("ExitCode", 0) != 0:
                raise SmokeFailure(f"one-shot service '{name}' exited {row['ExitCode']}")
        for name in HEALTHY:
            row = containers.get(name)
            if row and row.get("Health") == "unhealthy":
                raise SmokeFailure(f"service '{name}' is unhealthy")
        done = all(
            (row := containers.get(name)) and row.get("State") == "exited" for name in ONE_SHOT
        ) and all(
            (row := containers.get(name)) and row.get("Health") == "healthy" for name in HEALTHY
        )
        status = ", ".join(
            f"{name}={row.get('Health') or row.get('State')}"
            for name, row in sorted(containers.items())
        )
        if status != last:
            print(f"    {status}", flush=True)
            last = status
        if done:
            return
        time.sleep(2)
    raise SmokeFailure(f"timed out after {STARTUP_TIMEOUT_SECONDS}s waiting for a healthy stack")


def _inspect(service: str) -> dict:
    container = compose("ps", "--all", "-q", service).strip().splitlines()
    if not container:
        raise SmokeFailure(f"no container for service '{service}'")
    out = subprocess.run(
        ["docker", "inspect", container[0]], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)[0]


def _check_ports() -> None:
    """The API published on loopback only; Postgres not published, nothing on 5432."""
    bindings = _inspect("api")["HostConfig"]["PortBindings"] or {}
    if not bindings:
        raise SmokeFailure("the API publishes no port")
    for port, hosts in bindings.items():
        for host in hosts:
            if host.get("HostIp") != "127.0.0.1":
                raise SmokeFailure(f"the API publishes {port} on '{host.get('HostIp')}'")
    print("    the API is published on 127.0.0.1 only", flush=True)

    if _inspect("postgres")["HostConfig"]["PortBindings"]:
        raise SmokeFailure("postgres publishes a host port")
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(("127.0.0.1", POSTGRES_PORT)) == 0:
            raise SmokeFailure(f"something answers on the host's port {POSTGRES_PORT}")
    print("    postgres is not reachable from the host", flush=True)


def _check_user_and_filesystem(service: str) -> None:
    """A non-root user, and a root filesystem the service cannot write."""
    uid = compose("exec", "-T", service, "id", "-u").strip()
    if uid == "0" or not uid.isdigit():
        raise SmokeFailure(f"{service} runs as uid '{uid}'")
    if not _inspect(service)["HostConfig"]["ReadonlyRootfs"]:
        raise SmokeFailure(f"{service} has a writable root filesystem")
    wrote = subprocess.run(
        [*COMPOSE, "exec", "-T", service, "touch", "/app/write-probe"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if wrote.returncode == 0:
        raise SmokeFailure(f"{service} could write to /app")
    print(f"    {service} runs as uid {uid} on a read-only root filesystem", flush=True)


def check_posture() -> None:
    """Step 2: ports, users, read-only root filesystems, the collector."""
    _check_ports()
    for service in ("api", "worker"):
        _check_user_and_filesystem(service)
    # The collector has no shell to probe itself; probe it from the API.
    compose(
        "exec",
        "-T",
        "api",
        "python",
        "-c",
        "import urllib.request; urllib.request.urlopen('http://otel-collector:13133/', timeout=5)",
    )
    print("    the collector is healthy", flush=True)


def _cli(service: str, *args: str) -> str:
    return compose("exec", "-T", service, "python", "-m", "personalos.cli", *args)


def run_workflow() -> str:
    """Steps 3-4: the quick start's data, then its search and approval."""
    ingested = json.loads(_cli("worker", "ingest-resume", "resume.md"))
    if ingested["status"] != "indexed" or not ingested["document"]["evidence"]:
        raise SmokeFailure(f"the resume was not indexed: {ingested}")
    _cli("worker", "profile", "create", "/app/docs/quickstart/profile.json")

    token = _cli("api", "auth", "token").strip().splitlines()[-1]
    with httpx.Client(base_url=f"http://127.0.0.1:{API_PORT}", timeout=10) as client:
        if client.get("/health").status_code != 200:
            raise SmokeFailure("GET /health on the published port did not answer 200")
        if client.get("/v1/approvals").status_code != 401:
            raise SmokeFailure("the API answered a request that carried no token")
        client.headers["Authorization"] = f"Bearer {token}"
        return _search_and_approve(client, {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-build", action="store_true", help="Use the images already built.")
    parser.add_argument("--keep", action="store_true", help="Leave the stack running.")
    args = parser.parse_args()

    failed = False
    try:
        compose("up", "-d", *([] if args.no_build else ["--build"]), capture=False)
        wait_until_healthy()
        check_posture()
        workflow_id = run_workflow()
        print(f"compose smoke test passed (workflow {workflow_id})")
    except (SmokeFailure, httpx.HTTPError) as exc:
        failed = True
        print(f"\nCOMPOSE SMOKE TEST FAILED: {exc}", file=sys.stderr)
    finally:
        if failed:
            compose("ps", "--all", check=False, capture=False)
            compose("logs", "--no-color", check=False, capture=False)
        if not args.keep:
            compose("down", "-v", "--remove-orphans", check=False, capture=False)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
