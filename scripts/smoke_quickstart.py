"""Run the local quick start end to end, against the fake job board.

    python scripts/smoke_quickstart.py

Does what `IMPLEMENTATION_GUIDE.md#local-quick-start` tells a new contributor
to do, with the same commands and the same sample files, so a change that
breaks the walkthrough fails here instead of in someone's terminal:

1. migrate the database (`alembic upgrade head`);
2. ingest `docs/quickstart/resume.md` (`personalos.cli ingest-resume`);
3. create the search profile `docs/quickstart/profile.json` (`profile create`);
4. start the API and the worker as real processes, the worker on
   `apps.worker.fake_providers:build_worker_graphs`, and read the API token
   (`personalos.cli auth token`) the requests below send as a bearer token;
5. start a job search with `POST /v1/chat` and wait for its approval interrupt;
6. approve the pending action with `POST /v1/approvals/{id}/decision`;
7. check the run resumed and stopped where the guide says it does: at the
   unbound `application_submitter` port.

Needs `DATABASE_URL` pointing at a fresh PostgreSQL with pgvector (default:
the quick start's). Fresh, because the submission's idempotency key is derived
from the posting, so a second run against the same database stops at an
`IdempotencyConflict` instead. A `sqlite:///` URL skips Alembic and creates
the tables directly, for a check without Postgres; CI runs it on Postgres.
Exits non-zero on the first step that fails, with both processes' output.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
QUICKSTART = REPO / "docs" / "quickstart"
DEFAULT_DATABASE_URL = "postgresql://personalos:personalos@127.0.0.1:5432/personalos"
FAKE_GRAPHS = "apps.worker.fake_providers:build_worker_graphs"
#: What an approved submission fails on until a submitter adapter is bound.
UNBOUND_SUBMITTER = "application_submitter"

STARTUP_TIMEOUT_SECONDS = 60
WORKFLOW_TIMEOUT_SECONDS = 120


class SmokeFailure(RuntimeError):
    """A quick start step did not do what the guide says it does."""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _env(database_url: str, port: int, token_file: Path) -> dict[str, str]:
    """The quick start's minimum settings, over anything a local `.env` sets."""
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": database_url,
            "FILES_ALLOWED_ROOTS": json.dumps([str(QUICKSTART)]),
            "WORKER_GRAPHS": FAKE_GRAPHS,
            "API_PORT": str(port),
            # A throwaway token, never the contributor's own.
            "API_TOKEN_FILE": str(token_file),
            "API_TOKEN_STORE": "file",
            "API_AUTH_ENABLED": "true",
            "LOG_FORMAT": "text",
            # Pinned rather than left to `.env`: the smoke test never calls a
            # model or a real job board.
            "LLM_GATEWAY_CONFIG": "",
            "EMBEDDING_PROVIDER": "hashing",
            "GREENHOUSE_BOARDS": "{}",
            "TRACING_ENABLED": "false",
            "WORKER_IDLE_SECONDS": "0.2",
        }
    )
    env.pop("API_HOST", None)  # the default is the posture under test
    return env


def _run(step: str, args: list[str], env: dict[str, str]) -> str:
    print(f"==> {step}: {' '.join(args)}", flush=True)
    done = subprocess.run(
        [sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True
    )
    if done.returncode != 0:
        raise SmokeFailure(f"{step} exited {done.returncode}\n{done.stdout}\n{done.stderr}")
    return done.stdout


def _start(name: str, args: list[str], env: dict[str, str], logs: Path) -> subprocess.Popen:
    print(f"==> start {name}: {' '.join(args)}", flush=True)
    log = open(logs / f"{name}.log", "w")  # noqa: SIM115 - closed with the process
    return subprocess.Popen(
        [sys.executable, *args], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
    )


def _wait_for(what: str, probe, timeout: float, processes: dict[str, subprocess.Popen]):
    """Poll `probe` until it returns something truthy, a process dies, or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, process in processes.items():
            if process.poll() is not None:
                raise SmokeFailure(f"the {name} exited ({process.returncode}) waiting for {what}")
        result = probe()
        if result:
            return result
        time.sleep(0.5)
    raise SmokeFailure(f"timed out after {timeout:.0f}s waiting for {what}")


def _api_ready(client: httpx.Client) -> bool:
    try:
        return client.get("/health").status_code == 200
    except httpx.TransportError:
        return False


def _prepare(database_url: str, env: dict[str, str]) -> None:
    """Steps 1-3: the schema, the resume's evidence and the search profile."""
    if database_url.startswith("sqlite"):
        _run("create tables", ["-m", "personalos.cli", "db-init"], env)
    else:
        _run("migrate", ["-m", "alembic", "upgrade", "head"], env)

    ingested = json.loads(
        _run("ingest resume", ["-m", "personalos.cli", "ingest-resume", "resume.md"], env)
    )
    if ingested["status"] != "indexed" or not ingested["document"]["evidence"]:
        raise SmokeFailure(f"the resume was not indexed: {ingested}")

    profile = str(QUICKSTART / "profile.json")
    _run("create profile", ["-m", "personalos.cli", "profile", "create", profile], env)


def _search_and_approve(client: httpx.Client, processes: dict[str, subprocess.Popen]) -> str:
    """Steps 5-7, against the running API and worker; returns the workflow id."""
    print("==> POST /v1/chat", flush=True)
    started = client.post("/v1/chat", json={"message": "find me remote python backend jobs"})
    if started.status_code != 202:
        raise SmokeFailure(f"POST /v1/chat: {started.status_code} {started.text}")
    workflow_id = started.json()["workflow_id"]

    def status() -> dict:
        response = client.get(f"/v1/workflows/{workflow_id}")
        response.raise_for_status()
        return response.json()

    def failures() -> list[dict]:
        return client.get(f"/v1/workflows/{workflow_id}/failures").json()["failures"]

    print(f"==> waiting for workflow {workflow_id} to ask for approval", flush=True)
    waiting = _wait_for(
        "the approval interrupt",
        lambda: (s := status())["status"] in ("waiting", "failed", "completed") and s,
        WORKFLOW_TIMEOUT_SECONDS,
        processes,
    )
    if waiting["status"] != "waiting" or not waiting.get("pending_approval"):
        raise SmokeFailure(
            f"expected a pending approval, got status '{waiting['status']}'\n"
            f"{json.dumps(failures(), indent=2)}"
        )

    print("==> GET /v1/approvals", flush=True)
    approvals = [
        item
        for item in client.get("/v1/approvals").json()["approvals"]
        if item["workflow_id"] == workflow_id
    ]
    if len(approvals) != 1 or approvals[0]["action_type"] != "submit_application":
        raise SmokeFailure(f"expected one submit_application approval, got {approvals}")
    approval = approvals[0]

    print(f"==> POST /v1/approvals/{approval['id']}/decision", flush=True)
    decided = client.post(
        f"/v1/approvals/{approval['id']}/decision",
        json={"approved": True, "action_hash": approval["action_hash"]},
    )
    if decided.status_code != 200 or not decided.json()["resumed"]:
        raise SmokeFailure(f"approval decision: {decided.status_code} {decided.text}")

    print("==> waiting for the approved action to run", flush=True)
    _wait_for(
        "the resumed run to settle",
        lambda: status()["status"] in ("failed", "completed"),
        WORKFLOW_TIMEOUT_SECONDS,
        processes,
    )
    recorded = failures()
    if not any(
        f["step"] == "execute_approved_actions" and UNBOUND_SUBMITTER in (f["last_error"] or "")
        for f in recorded
    ):
        raise SmokeFailure(
            f"the approved submission did not stop at the unbound '{UNBOUND_SUBMITTER}' "
            "port as the guide says; if a submitter is now wired, update the quick start "
            f"and this check.\n{json.dumps(recorded, indent=2)}"
        )
    return workflow_id


def smoke(database_url: str, logs: Path) -> None:
    port = _free_port()
    env = _env(database_url, port, logs / "api-token")
    _prepare(database_url, env)

    processes = {
        "api": _start("api", ["-m", "personalos.cli", "api"], env, logs),
        "worker": _start("worker", ["-m", "personalos.cli", "worker"], env, logs),
    }
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            _wait_for("the API", lambda: _api_ready(client), STARTUP_TIMEOUT_SECONDS, processes)
            token = _run("read API token", ["-m", "personalos.cli", "auth", "token"], env)
            client.headers["Authorization"] = f"Bearer {token.strip().splitlines()[-1]}"
            if client.get("/v1/approvals", headers={"Authorization": ""}).status_code != 401:
                raise SmokeFailure("the API answered a request that carried no token")
            workflow_id = _search_and_approve(client, processes)
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()

    print(f"quick start smoke test passed (workflow {workflow_id})")


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    with tempfile.TemporaryDirectory(prefix="personalos-smoke-") as tmp:
        logs = Path(tmp)
        try:
            smoke(database_url, logs)
        except SmokeFailure as exc:
            print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
            for log in sorted(logs.glob("*.log")):
                print(f"\n--- {log.name} ---\n{log.read_text()}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
