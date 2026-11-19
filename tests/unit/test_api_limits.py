"""Per-actor rate limits, concurrency caps, the auth-failure throttle, request limits.

Each test drives the production app (`create_app()`), with the test bearer
verifier and its limits rebuilt over a fake clock, so a "window" passes by
advancing a number rather than by sleeping. Workflows are real rows in a
SQLite file: the active-workflow cap reads the command queue, and an
idempotent replay is judged by what is in the `operations` ledger.

Acceptance criteria covered:

- exceeding the workflow-start limit is a 429 with `Retry-After`, and a
  request after the window succeeds;
- the active-workflow cap refuses the next start, and frees up when a
  workflow's command is settled;
- a body over the cap is a 413 before the route runs; resume ingestion takes
  its larger cap and refuses beyond it;
- deeply nested or oversized JSON is a 422, quickly;
- repeated bad tokens from one address are throttled, while one typo does not
  lock out the right token;
- replaying an `Idempotency-Key` spends no budget and creates no workflow;
- `/health` is never limited, and no limit-hit log line or span carries a
  token or body content.
"""

import logging
import time
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from apps.api.limits import (
    MAX_PAGE_SIZE,
    AuthFailureThrottle,
    InMemoryRateLimitStore,
    LimitName,
    RateLimit,
    build_api_limits,
)
from apps.api.main import create_app
from apps.api.routes.events import get_progress_log
from apps.api.routes.evidence import get_resume_ingestor
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.api.routes.workflows import (
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENT_REPLAY_HEADER,
    build_workflow_services,
    get_workflow_services,
)
from personalos.bootstrap import build_workflow_command_queue
from personalos.config import Settings, settings
from personalos.observability.limits import LIMIT_HIT_EVENT, LIMIT_HIT_SPAN, LimitHitCounter
from personalos.observability.tracing_setup import configure_tracing, reset_tracing
from personalos.persistence.models import WorkflowRunModel
from tests.fixtures.auth import bearer, install_test_auth
from tests.fixtures.durable_workflow import session_factory


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limits(clock: FakeClock, **overrides):
    config = Settings(
        **{
            "api_rate_workflow_start_per_minute": 100,
            "api_max_active_workflows": 100,
            **overrides,
        }
    )
    limits = build_api_limits(config, clock=clock)
    limits.hits = LimitHitCounter()
    return limits


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def factory(tmp_path):
    return session_factory(tmp_path / "limits.db")


@pytest.fixture
def make_client(clock, factory):
    """A client over the production app whose limits are `overrides` of the defaults."""

    def build(**overrides) -> TestClient:
        app = install_test_auth(create_app())
        app.state.api_limits = _limits(clock, **overrides)
        app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
        app.dependency_overrides[get_failure_services] = lambda: build_failure_services(factory)
        app.dependency_overrides[get_resume_ingestor] = lambda: _MissingFileIngestor()
        return TestClient(app)

    return build


class _MissingFileIngestor:
    """Answers every ingest as a file that is not there, without touching disk or models."""

    async def ingest(self, user_id, path, *, display_name=None):
        from personalos.domain.errors import NotFound

        raise NotFound(f"no file '{path}'")


def _start(client: TestClient, actor: str = "alice", *, key: str | None = None, message="hi"):
    headers = bearer(actor)
    if key is not None:
        headers[IDEMPOTENCY_KEY_HEADER] = key
    return client.post("/v1/chat", json={"message": message}, headers=headers)


def _runs(factory) -> int:
    session = factory()
    try:
        return session.query(WorkflowRunModel).count()
    finally:
        session.close()


def _assert_rate_limited(response, limit: str) -> None:
    assert response.status_code == 429, response.text
    assert int(response.headers["Retry-After"]) >= 1
    body = response.json()
    assert body["error_code"] == "rate_limited"
    assert set(body) == {"error_code", "message", "context_id"}
    assert limit.replace("_", " ") in body["message"]


# --- Rates ------------------------------------------------------------------------


def test_exceeding_the_workflow_start_limit_is_a_429_until_the_window_passes(make_client, clock):
    client = make_client(api_rate_workflow_start_per_minute=2)

    assert _start(client).status_code == 202
    assert _start(client).status_code == 202
    refused = _start(client)

    _assert_rate_limited(refused, "workflow start")
    # One token comes back every 30s at 2 per minute.
    assert int(refused.headers["Retry-After"]) == 30

    clock.advance(30)
    assert _start(client).status_code == 202


def test_the_budget_is_the_verified_actors_not_a_header(make_client):
    client = make_client(api_rate_workflow_start_per_minute=1)

    assert _start(client, "alice").status_code == 202
    # Claiming to be someone else changes nothing: the token says alice.
    spoofed = client.post(
        "/v1/chat",
        json={"message": "hi"},
        headers={**bearer("alice"), "X-Actor-Id": "bob", "X-Forwarded-For": "10.9.9.9"},
    )
    _assert_rate_limited(spoofed, "workflow start")
    # Another actor has a budget of its own.
    assert _start(client, "bob").status_code == 202


@pytest.mark.parametrize(
    ("method", "path", "limit", "setting"),
    [
        (
            "POST",
            "/v1/approvals/00000000-0000-4000-8000-000000000001/decision",
            LimitName.APPROVAL_DECISION,
            "api_rate_approval_decision_per_minute",
        ),
        (
            "POST",
            "/v1/workflows/00000000-0000-4000-8000-000000000001/retry",
            LimitName.FAILURE_ACTION,
            "api_rate_failure_action_per_minute",
        ),
        (
            "POST",
            "/v1/workflows/00000000-0000-4000-8000-000000000001/dismiss",
            LimitName.FAILURE_ACTION,
            "api_rate_failure_action_per_minute",
        ),
        (
            "POST",
            "/v1/failures/events/00000000-0000-4000-8000-000000000001/retry",
            LimitName.FAILURE_ACTION,
            "api_rate_failure_action_per_minute",
        ),
        (
            "POST",
            "/v1/workflows/00000000-0000-4000-8000-000000000001/resume",
            LimitName.WORKFLOW_RESUME,
            "api_rate_workflow_resume_per_minute",
        ),
        (
            "POST",
            "/v1/evidence/documents",
            LimitName.EVIDENCE_INGEST,
            "api_rate_evidence_ingest_per_minute",
        ),
    ],
)
def test_every_limited_route_spends_its_own_budget(make_client, method, path, limit, setting):
    """The first request spends the only token, whatever it then answers; the next is a 429."""
    client = make_client(**{setting: 1})

    # FastAPI validates the body before route dependencies run, so each route
    # gets one that parses; what the route then answers does not matter.
    bodies = {
        LimitName.APPROVAL_DECISION: {"approved": True, "action_hash": "h"},
        LimitName.WORKFLOW_RESUME: {"event": {"ok": True}},
        LimitName.EVIDENCE_INGEST: {"path": "missing.md"},
    }
    kwargs = {"json": bodies[limit]} if limit in bodies else {}

    first = client.request(method, path, headers=bearer("alice"), **kwargs)
    second = client.request(method, path, headers=bearer("alice"), **kwargs)

    assert first.status_code != 429
    _assert_rate_limited(second, limit)


def test_token_bucket_refills_continuously_up_to_its_capacity():
    clock = FakeClock()
    store = InMemoryRateLimitStore(clock=clock)
    rate = RateLimit(capacity=3, period_seconds=60)

    assert [store.consume("k", rate).allowed for _ in range(4)] == [True, True, True, False]
    clock.advance(20)
    assert store.consume("k", rate).allowed
    assert not store.consume("k", rate).allowed
    clock.advance(3600)
    # Never more than the capacity, however long it sat.
    assert [store.consume("k", rate).allowed for _ in range(4)] == [True, True, True, False]


# --- Concurrency ------------------------------------------------------------------


def test_the_active_workflow_cap_refuses_the_next_start_and_frees_on_completion(
    make_client, factory
):
    client = make_client(api_max_active_workflows=2)

    assert _start(client).status_code == 202
    assert _start(client).status_code == 202
    refused = _start(client)
    _assert_rate_limited(refused, "active workflows")
    assert "at most 2" in refused.json()["message"]
    assert _runs(factory) == 2, "an over-cap start must not create a workflow"
    # Another actor is not held up by alice's work.
    assert _start(client, "bob").status_code == 202

    # The worker runs one of alice's commands to the end.
    queue = build_workflow_command_queue(factory)
    claimed = queue.claim_next()
    assert claimed is not None and claimed.command.actor_id == "alice"
    queue.mark_done(claimed.outbox_id)

    assert _start(client).status_code == 202


def test_the_event_stream_cap_refuses_one_more_and_frees_when_a_stream_ends(make_client):
    class _Log:
        def workflow_exists(self, workflow_id, *, access) -> bool:
            return True

        def events_after(self, workflow_id, after, *, access, limit):
            return []

    client = make_client(api_max_sse_connections=1)
    client.app.dependency_overrides[get_progress_log] = lambda: _Log()
    limits = client.app.state.api_limits
    path = "/v1/workflows/00000000-0000-4000-8000-000000000001/events?follow=false"

    # Streams that end give their slot back: one after another is fine.
    for _ in range(3):
        assert client.get(path, headers=bearer("alice")).status_code == 200
    assert limits.streams.held("alice") == 0

    # One stream held open (as a following stream would be) fills the cap.
    held = limits.open_stream("alice", route="test")
    _assert_rate_limited(client.get(path, headers=bearer("alice")), "event streams")
    assert client.get(path, headers=bearer("bob")).status_code == 200
    held.release()
    assert client.get(path, headers=bearer("alice")).status_code == 200


# --- Authentication failures ------------------------------------------------------


def test_repeated_bad_tokens_are_throttled_but_one_typo_is_not(make_client, clock):
    client = make_client(api_auth_free_failures=3, api_auth_backoff_base_seconds=2)
    whoami = "/v1/auth/whoami"
    typo = {"Authorization": "Bearer test-tokn:alice"}

    # A single typo, then the right token: no lockout.
    assert client.get(whoami, headers=typo).status_code == 401
    assert client.get(whoami, headers=bearer("alice")).status_code == 200

    # Four failures in a row: the fourth is past the free ones.
    for _ in range(4):
        assert client.get(whoami, headers=typo).status_code == 401
    throttled = client.get(whoami, headers=typo)
    assert throttled.status_code == 429
    assert throttled.json()["error_code"] == "rate_limited"
    assert throttled.headers["Retry-After"] == "2"
    # While waiting, not even the right token is checked.
    assert client.get(whoami, headers=bearer("alice")).status_code == 429

    # Each further failure doubles the wait.
    clock.advance(2)
    assert client.get(whoami, headers=typo).status_code == 401
    assert client.get(whoami, headers=typo).headers["Retry-After"] == "4"

    clock.advance(4)
    assert client.get(whoami, headers=bearer("alice")).status_code == 200
    # ...and a success clears the slate.
    assert client.get(whoami, headers=typo).status_code == 401
    assert client.get(whoami, headers=bearer("alice")).status_code == 200


def test_no_credential_at_all_is_not_counted_as_a_guess(make_client):
    client = make_client(api_auth_free_failures=1)

    for _ in range(10):
        assert client.get("/v1/auth/whoami").status_code == 401
    assert client.get("/v1/auth/whoami", headers=bearer("alice")).status_code == 200


def test_backoff_is_capped():
    clock = FakeClock()
    throttle = AuthFailureThrottle(free_failures=1, base_seconds=1, max_seconds=10, clock=clock)

    waits = [throttle.failed("1.2.3.4") for _ in range(8)]

    assert waits == [None, 1, 2, 4, 8, 10, 10, 10]
    assert throttle.retry_after("5.6.7.8") is None


# --- Request limits ---------------------------------------------------------------


class _SpyIngestor:
    def __init__(self) -> None:
        self.calls = 0

    async def ingest(self, user_id, path, *, display_name=None):  # pragma: no cover - never reached
        self.calls += 1
        raise AssertionError("the route ran")


def test_an_oversized_body_is_a_413_before_the_route_runs(make_client, monkeypatch):
    monkeypatch.setattr(settings, "api_max_body_bytes", 1024)
    monkeypatch.setattr(settings, "api_max_resume_body_bytes", 4096)
    client = make_client()
    spy = _SpyIngestor()
    client.app.dependency_overrides[get_resume_ingestor] = lambda: spy

    def body(size: int) -> bytes:
        prefix = b'{"path": ""}'
        return prefix + b" " * (size - len(prefix))

    headers = {"Content-Type": "application/json", **bearer("alice")}
    # Over the default cap, under the resume cap: through to validation (an
    # empty path is a 422), so the middleware let it pass.
    within = client.post("/v1/evidence/documents", content=body(3000), headers=headers)
    beyond = client.post("/v1/evidence/documents", content=body(4097), headers=headers)
    elsewhere = client.post("/v1/chat", content=body(3000), headers=headers)

    assert within.status_code == 422
    assert beyond.status_code == 413
    assert beyond.json()["error_code"] == "payload_too_large"
    assert elsewhere.status_code == 413
    assert spy.calls == 0


@pytest.mark.parametrize(
    ("body", "limit"),
    [
        (b"[" * 500_000 + b"]" * 500_000, "nests"),
        (b'{"message": "' + b"a" * 900_000 + b'"}', "string"),
        (b'{"message": "' + b'\\"' * 400_000, "string"),
        (b'{"message": "\xff\xfe"}', "UTF-8"),
    ],
    ids=["deeply-nested", "huge-string", "unterminated-escapes", "not-utf8"],
)
def test_hostile_json_is_a_422_without_excessive_work(make_client, body, limit):
    client = make_client()

    started = time.perf_counter()
    response = client.post(
        "/v1/chat",
        content=body,
        headers={"Content-Type": "application/json", **bearer("alice")},
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 422
    assert response.json()["error_code"] == "validation"
    assert limit in response.json()["message"]
    assert elapsed < 2.0, f"took {elapsed:.2f}s"


def test_json_inside_the_limits_still_parses(make_client):
    client = make_client()
    nested = {"event": {"a": [[[[{"b": "x" * 1000}]]]]}}

    response = client.post(
        "/v1/workflows/00000000-0000-4000-8000-000000000001/resume",
        json=nested,
        headers=bearer("alice"),
    )

    assert response.status_code == 404  # past the limits, to the route: no such workflow


def test_every_list_endpoint_bounds_its_page_size():
    """Read from the OpenAPI schema (`app.routes` is nested since FastAPI 0.14x)."""
    paths = create_app().openapi()["paths"]
    bounded = {
        path: param["schema"].get("maximum")
        for path, operations in paths.items()
        for method, operation in operations.items()
        if method == "get"
        for param in operation.get("parameters", [])
        if param["in"] == "query" and param["name"] == "limit"
    }
    assert {
        "/v1/approvals",
        "/v1/applications",
        "/v1/audit",
        "/v1/failures/events",
        "/v1/profile/versions",
    } <= set(bounded)
    assert all(m is not None and m <= MAX_PAGE_SIZE for m in bounded.values()), bounded


def test_a_page_size_over_the_maximum_is_refused(make_client):
    client = make_client()

    response = client.get(f"/v1/approvals?limit={MAX_PAGE_SIZE + 1}", headers=bearer("alice"))

    assert response.status_code == 422


# --- Idempotent replays -----------------------------------------------------------


def test_an_idempotent_replay_spends_no_budget_and_creates_no_workflow(make_client, factory):
    client = make_client(api_rate_workflow_start_per_minute=2)

    first = _start(client, key="key-a")
    replays = [_start(client, key="key-a") for _ in range(5)]

    assert first.status_code == 202
    assert IDEMPOTENT_REPLAY_HEADER not in first.headers
    for replay in replays:
        assert replay.status_code == 202
        assert replay.headers[IDEMPOTENT_REPLAY_HEADER] == "true"
        assert replay.json() == first.json()
        assert replay.headers["Location"] == f"/v1/workflows/{first.json()['workflow_id']}"
    assert _runs(factory) == 1

    # Five replays later the budget still holds its second token.
    assert _start(client, key="key-b").status_code == 202
    _assert_rate_limited(_start(client, key="key-c"), "workflow start")
    assert _runs(factory) == 2


def test_a_reused_key_with_a_different_body_is_a_409(make_client, factory):
    client = make_client()

    assert _start(client, key="k", message="one").status_code == 202
    reused = _start(client, key="k", message="two")

    assert reused.status_code == 409
    assert reused.json()["error_code"] == "idempotency_conflict"
    assert _runs(factory) == 1


def test_keys_are_per_actor(make_client, factory):
    client = make_client()

    alice = _start(client, "alice", key="same")
    bob = _start(client, "bob", key="same")

    assert alice.json()["workflow_id"] != bob.json()["workflow_id"]
    assert IDEMPOTENT_REPLAY_HEADER not in bob.headers


def test_a_refused_start_leaves_its_key_free_to_retry(make_client, clock):
    client = make_client(api_rate_workflow_start_per_minute=1)

    assert _start(client, key="a").status_code == 202
    _assert_rate_limited(_start(client, key="b"), "workflow start")
    clock.advance(60)
    retried = _start(client, key="b")

    assert retried.status_code == 202
    assert IDEMPOTENT_REPLAY_HEADER not in retried.headers
    UUID(retried.json()["workflow_id"])


# --- Health, and what a hit reveals -----------------------------------------------


def test_health_is_never_limited(make_client):
    client = make_client(api_auth_free_failures=1, api_auth_backoff_base_seconds=600)
    # Throttle this address, and fill every other budget.
    for _ in range(3):
        client.get("/v1/auth/whoami", headers={"Authorization": "Bearer wrong"})
    assert client.get("/v1/auth/whoami", headers=bearer("alice")).status_code == 429

    for _ in range(200):
        assert client.get("/health").status_code == 200


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, service_name="test", install_global=False)
    try:
        yield exporter
    finally:
        reset_tracing()


def test_limit_hits_are_reported_without_token_or_body(make_client, caplog, spans):
    # Not `test-token:...`, so the test verifier refuses it.
    secret_token = "pos_SECRET-TOKEN-VALUE-0123456789"
    body_marker = "BODY-CONTENT-MARKER"
    client = make_client(
        api_rate_workflow_start_per_minute=1,
        api_auth_free_failures=1,
    )
    caplog.set_level(logging.WARNING, logger="personalos.api.limits")

    # A rate limit, with a body.
    _start(client, message=body_marker)
    _assert_rate_limited(_start(client, message=body_marker), "workflow start")
    # JSON too deep and a body too big, both carrying the marker.
    client.post(
        "/v1/chat",
        content=b'{"message": "'
        + body_marker.encode()
        + b'", "x": '
        + b"[" * 100
        + b"]" * 100
        + b"}",
        headers={"Content-Type": "application/json", **bearer("alice")},
    )
    client.post(
        "/v1/chat",
        content=(body_marker * 200_000).encode(),
        headers={"Content-Type": "application/json", **bearer("alice")},
    )
    # The auth throttle, with a secret in the bearer.
    for _ in range(3):
        client.get(
            "/v1/auth/whoami?ticket=QUERY-MARKER",
            headers={"Authorization": f"Bearer {secret_token}"},
        )

    hits = [r for r in caplog.records if getattr(r, "event", None) == LIMIT_HIT_EVENT]
    assert {r.limit_name for r in hits} >= {
        LimitName.WORKFLOW_START,
        LimitName.JSON_DEPTH,
        LimitName.BODY_SIZE,
        LimitName.AUTH_FAILURES,
    }
    for record in hits:
        assert record.actor_id and record.route and record.limit_name
        rendered = repr(record.__dict__) + record.getMessage()
        for forbidden in (secret_token, body_marker, "QUERY-MARKER", "Bearer"):
            assert forbidden not in rendered

    limit_spans = [s for s in spans.get_finished_spans() if s.name == LIMIT_HIT_SPAN]
    assert len(limit_spans) == len(hits)
    for span in limit_spans:
        assert {"actor_id", "route", "limit"} <= set(span.attributes)
        rendered = repr(dict(span.attributes)) + repr([e.attributes for e in span.events])
        for forbidden in (secret_token, body_marker, "QUERY-MARKER"):
            assert forbidden not in rendered

    # The tally the failures view reads.
    tally = client.app.state.api_limits.hits.snapshot()
    assert any(h.limit == LimitName.WORKFLOW_START and h.actor_id == "alice" for h in tally)


def test_the_failures_view_shows_the_callers_hits_and_admin_sees_all(make_client):
    client = make_client(api_rate_workflow_start_per_minute=1)
    for actor in ("alice", "bob"):
        _start(client, actor)
        for _ in range(3):
            _start(client, actor)

    mine = client.get("/v1/failures/limits", headers=bearer("alice", scopes="read")).json()
    everyone = client.get("/v1/failures/limits", headers=bearer("alice")).json()

    assert mine["hits"] == [
        {
            "actor_id": "alice",
            "route": "POST /v1/chat",
            "limit": "workflow_start",
            "count": 3,
            "last_hit_at": mine["hits"][0]["last_hit_at"],
        }
    ]
    assert {h["actor_id"] for h in everyone["hits"]} == {"alice", "bob"}


def test_limits_can_be_turned_off(make_client):
    client = make_client(api_limits_enabled=False, api_rate_workflow_start_per_minute=1)

    assert all(_start(client).status_code == 202 for _ in range(3))
