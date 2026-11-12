"""API authentication: the local bearer token, stream tickets, bind safety, redaction.

These tests run the production wiring -- `create_app()` with the
`ApiAuth` it builds from settings, i.e. `LocalTokenVerifier` over a real
`0600` token file (pointed at a temp path by the suite-wide
`_isolated_api_token` fixture) -- rather than the test verifier the other
route tests install.

Acceptance criteria covered:

- every non-health route is a 401 without a token and with a bad one;
- the SSE endpoint rejects unauthenticated clients;
- the API refuses to bind 0.0.0.0 with auth disabled;
- a seeded fake token never appears in logs or span attributes.
"""

import json
import logging
import os
import re
import stat
from datetime import datetime, timezone
from uuid import UUID

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from apps.api.auth import (
    ApiAuth,
    LocalTokenVerifier,
    StreamTicketSigner,
    UnsafeBindAddress,
    build_api_auth,
    ensure_safe_bind,
    is_loopback_host,
)
from apps.api.main import create_app
from apps.api.routes.events import get_progress_log
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from personalos.config import settings
from personalos.domain.auth import API_TOKEN_PREFIX, AuthMethod, Principal
from personalos.domain.credentials import SecretValue
from personalos.domain.redaction import REDACTED, Redactor, default_redactor
from personalos.observability import tracing
from personalos.observability.tracing_setup import configure_tracing, reset_tracing
from personalos.secrets.api_token import FileApiTokenStore, LocalApiToken
from personalos.secrets.store import SecretStoreUnavailable
from tests.fixtures.durable_workflow import session_factory

#: A token of the real shape, fixed so a test can look for it in output.
SEEDED_TOKEN = API_TOKEN_PREFIX + "s33dedFakeT0kenForRedactionTests_0123456789"
BAD_TOKEN = API_TOKEN_PREFIX + "notTheRightTokenAtAll_000000000000000000000"

KNOWN_WORKFLOW = UUID("6f1e7b3a-0000-4000-8000-00000000a0a0")
OTHER_WORKFLOW = UUID("6f1e7b3a-0000-4000-8000-00000000b0b0")

#: The routes that answer without a bearer token. Adding to this set is a
#: security decision: say what authenticates the route instead.
UNAUTHENTICATED_PATHS = {
    "/health",
    # Google redirects the browser here; the one-time OAuth `state` minted
    # by an authenticated `/connect` is the credential. Covered by
    # `test_google_oauth_onboarding.py`, forged state included.
    "/v1/integrations/google/callback",
}


@pytest.fixture
def token_file(_isolated_api_token):
    """The configured token file, holding `SEEDED_TOKEN`."""
    FileApiTokenStore(_isolated_api_token).write(SecretValue(SEEDED_TOKEN))
    yield _isolated_api_token
    default_redactor().forget_secrets()


@pytest.fixture
def app(token_file):
    """The production app: authentication built from settings on first request."""
    return create_app()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


def auth_header(token: str = SEEDED_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class _FakeProgressLog:
    """Knows one workflow, which has no events yet."""

    def workflow_exists(self, workflow_id: UUID) -> bool:
        return workflow_id == KNOWN_WORKFLOW

    def events_after(self, workflow_id, after, *, limit):
        return []


def _concrete(path: str) -> str:
    """A route path with every `{param}` filled in with a well-formed value."""
    return re.sub(r"\{[^}]+\}", str(KNOWN_WORKFLOW), path)


def _protected_routes(app) -> list[tuple[str, str]]:
    """Every (method, path) the app serves, but /health.

    Read from the OpenAPI schema rather than `app.routes`, whose shape is a
    FastAPI internal (nested since 0.14x). Every route here is declared, so
    every route is in the schema.
    """
    paths = app.openapi()["paths"]
    routes = [
        (method.upper(), _concrete(path))
        for path, operations in paths.items()
        if path not in UNAUTHENTICATED_PATHS
        for method in operations
        if method in {"get", "post", "put", "patch", "delete"}
    ]
    assert len(routes) > 20, "expected every API route to be enumerated"
    assert set(paths) >= UNAUTHENTICATED_PATHS
    return routes


# --- Every route but /health ----------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-token"),
        pytest.param(auth_header(BAD_TOKEN), id="bad-token"),
        pytest.param({"Authorization": "Bearer "}, id="empty-token"),
        pytest.param({"Authorization": f"Basic {SEEDED_TOKEN}"}, id="wrong-scheme"),
        pytest.param({"Authorization": SEEDED_TOKEN}, id="no-scheme"),
        pytest.param({"X-Actor-Id": "api"}, id="self-declared-actor"),
    ],
)
def test_every_non_health_route_is_a_401_without_a_valid_token(app, headers):
    client = TestClient(app, raise_server_exceptions=False)
    for method, path in _protected_routes(app):
        response = client.request(method, path, headers=headers)
        assert response.status_code == 401, f"{method} {path}: {response.status_code}"
        assert response.json()["error_code"] == "unauthenticated"
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert SEEDED_TOKEN not in response.text
        assert BAD_TOKEN not in response.text


def test_the_long_lived_token_is_not_accepted_in_the_query_string(app):
    client = TestClient(app)
    for param in ("access_token", "token", "ticket"):
        response = client.get(f"/v1/auth/whoami?{param}={SEEDED_TOKEN}")
        assert response.status_code == 401, param
    events = client.get(f"/v1/workflows/{KNOWN_WORKFLOW}/events?ticket={SEEDED_TOKEN}")
    assert events.status_code == 401


def test_health_needs_no_token(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_the_right_token_authenticates_as_the_configured_actor(client):
    response = client.get("/v1/auth/whoami", headers=auth_header())
    assert response.status_code == 200
    assert response.json() == {"actor_id": settings.api_actor, "method": "local_token"}


def test_the_actor_comes_from_the_token_not_a_header(client):
    """`X-Actor-Id` used to name the actor; now it is ignored."""
    headers = {**auth_header(), "X-Actor-Id": "user:someone-else"}
    assert client.get("/v1/auth/whoami", headers=headers).json()["actor_id"] == "api"


def test_the_actor_reaches_a_workflow_from_the_token(app, tmp_path):
    """POST /v1/chat stamps the workflow with the verified actor, not the declared one."""
    factory = session_factory(tmp_path / "auth.db")
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    client = TestClient(app, headers=auth_header())

    started = client.post(
        "/v1/chat",
        json={"message": "find me backend jobs"},
        headers={"X-Actor-Id": "user:mallory"},
    )

    assert started.status_code == 202, started.text
    status = client.get(f"/v1/workflows/{started.json()['workflow_id']}").json()
    assert status["actor_id"] == "api"


def test_the_token_is_created_on_first_use(_isolated_api_token, monkeypatch):
    import apps.api.main as main_module

    monkeypatch.setattr(main_module, "init_db", lambda: None)
    monkeypatch.setattr(main_module, "initialize_mcp_servers", lambda: None)
    assert not _isolated_api_token.exists()
    client = TestClient(create_app())

    with client:  # the lifespan mints it
        pass

    token = _isolated_api_token.read_text().strip()
    assert token.startswith(API_TOKEN_PREFIX)
    assert stat.S_IMODE(os.stat(_isolated_api_token).st_mode) == 0o600
    assert client.get("/v1/auth/whoami", headers=auth_header(token)).status_code == 200


def test_a_token_file_others_can_read_is_refused(token_file):
    os.chmod(token_file, 0o644)
    with pytest.raises(SecretStoreUnavailable, match="accessible to other users"):
        FileApiTokenStore(token_file).read()


def test_rotation_takes_effect_without_a_restart(token_file):
    clock = [0.0]
    manager = LocalApiToken(FileApiTokenStore(token_file))
    verifier = LocalTokenVerifier(manager, actor_id="api", reload_seconds=5, clock=lambda: clock[0])
    assert verifier.verify(SEEDED_TOKEN) is not None

    rotated = manager.rotate().reveal()
    clock[0] += 5

    assert verifier.verify(SEEDED_TOKEN) is None
    assert verifier.verify(rotated) == Principal("api", AuthMethod.LOCAL_TOKEN)


# --- The event stream -----------------------------------------------------------


@pytest.fixture
def stream_client(app) -> TestClient:
    app.dependency_overrides[get_progress_log] = lambda: _FakeProgressLog()
    return TestClient(app)


def _events(workflow_id: UUID = KNOWN_WORKFLOW, query: str = "") -> str:
    return f"/v1/workflows/{workflow_id}/events?follow=false{query}"


def _ticket(client: TestClient, workflow_id: UUID = KNOWN_WORKFLOW) -> dict:
    response = client.post(
        "/v1/auth/stream-tickets", json={"workflow_id": str(workflow_id)}, headers=auth_header()
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_the_stream_rejects_an_unauthenticated_client(stream_client):
    response = stream_client.get(_events())
    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthenticated"


def test_the_stream_rejects_a_bad_token_and_a_forged_ticket(stream_client):
    assert stream_client.get(_events(), headers=auth_header(BAD_TOKEN)).status_code == 401
    forged = "pos_st_eyJhIjoiYXBpIiwidyI6IngiLCJlIjo5OTk5OTk5OTk5fQ.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    assert stream_client.get(_events(query=f"&ticket={forged}")).status_code == 401


def test_the_stream_accepts_the_bearer_header_for_a_fetch_based_client(stream_client):
    response = stream_client.get(_events(), headers=auth_header())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_an_event_source_opens_the_stream_with_a_ticket(stream_client):
    issued = _ticket(stream_client)
    assert issued["ticket"].startswith("pos_st_")
    assert SEEDED_TOKEN not in json.dumps(issued)
    assert (
        issued["stream_url"] == f"/v1/workflows/{KNOWN_WORKFLOW}/events?ticket={issued['ticket']}"
    )

    response = stream_client.get(issued["stream_url"] + "&follow=false")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_a_ticket_opens_only_its_own_workflow(stream_client):
    issued = _ticket(stream_client, KNOWN_WORKFLOW)
    response = stream_client.get(_events(OTHER_WORKFLOW, f"&ticket={issued['ticket']}"))
    assert response.status_code == 401


def test_a_ticket_is_not_a_bearer_token(stream_client):
    issued = _ticket(stream_client)
    response = stream_client.get("/v1/auth/whoami", headers=auth_header(issued["ticket"]))
    assert response.status_code == 401


def test_a_ticket_needs_a_token_to_get(stream_client):
    response = stream_client.post(
        "/v1/auth/stream-tickets", json={"workflow_id": str(KNOWN_WORKFLOW)}
    )
    assert response.status_code == 401


def test_a_ticket_expires():
    now = [1_000_000.0]
    signer = StreamTicketSigner(lambda: b"k" * 32, ttl_seconds=60, clock=lambda: now[0])
    principal = Principal("api", AuthMethod.LOCAL_TOKEN)
    issued = signer.issue(principal, KNOWN_WORKFLOW)
    assert issued.expires_at == datetime.fromtimestamp(1_000_060, tz=timezone.utc)

    assert signer.verify(issued.ticket, KNOWN_WORKFLOW) == Principal(
        "api", AuthMethod.STREAM_TICKET
    )
    now[0] += 60
    assert signer.verify(issued.ticket, KNOWN_WORKFLOW) is None


def test_rotating_the_token_voids_outstanding_tickets(token_file):
    clock = [0.0]
    manager = LocalApiToken(FileApiTokenStore(token_file))
    verifier = LocalTokenVerifier(manager, actor_id="api", reload_seconds=5, clock=lambda: clock[0])
    signer = StreamTicketSigner(verifier.ticket_key, ttl_seconds=60)
    issued = signer.issue(Principal("api", AuthMethod.LOCAL_TOKEN), KNOWN_WORKFLOW)
    assert signer.verify(issued.ticket, KNOWN_WORKFLOW) is not None

    manager.rotate()
    clock[0] += 5

    assert signer.verify(issued.ticket, KNOWN_WORKFLOW) is None


# --- Bind address ---------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "127.8.8.8", "::1", "[::1]", "localhost"])
def test_loopback_addresses_are_loopback(host):
    assert is_loopback_host(host)
    ensure_safe_bind(host, auth_enabled=False)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.1", "myhost.lan"])
def test_the_api_refuses_a_non_loopback_bind_with_auth_disabled(host):
    assert not is_loopback_host(host)
    with pytest.raises(UnsafeBindAddress, match="API_AUTH_ENABLED=false"):
        ensure_safe_bind(host, auth_enabled=False)
    ensure_safe_bind(host, auth_enabled=True)


def test_the_cli_refuses_to_serve_0_0_0_0_with_auth_disabled(monkeypatch):
    import personalos.cli as cli_module

    served = []
    monkeypatch.setattr(settings, "api_auth_enabled", False)
    monkeypatch.setattr(cli_module.uvicorn, "run", lambda *a, **k: served.append(k))
    monkeypatch.setattr(cli_module, "init_db", lambda: None)

    result = CliRunner().invoke(cli_module.cli, ["api", "--host", "0.0.0.0"])

    assert result.exit_code != 0
    assert "refusing to listen on '0.0.0.0'" in result.output
    assert served == []


def test_the_cli_serves_0_0_0_0_with_auth_enabled(monkeypatch, _isolated_api_token):
    import personalos.cli as cli_module

    served = []
    monkeypatch.setattr(cli_module.uvicorn, "run", lambda *a, **k: served.append(k))
    monkeypatch.setattr(cli_module, "init_db", lambda: None)
    monkeypatch.setattr(settings, "api_host", settings.api_host)
    monkeypatch.setenv("API_HOST", "127.0.0.1")

    result = CliRunner().invoke(cli_module.cli, ["api", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert served and served[0]["host"] == "0.0.0.0"
    assert _isolated_api_token.exists()  # minted on first run


def test_the_app_lifespan_refuses_a_non_loopback_host_with_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "api_auth_enabled", False)
    monkeypatch.setattr(settings, "api_host", "0.0.0.0")
    with pytest.raises(UnsafeBindAddress), TestClient(create_app()):
        pass


def test_with_auth_disabled_every_request_is_the_configured_actor(monkeypatch):
    monkeypatch.setattr(settings, "api_auth_enabled", False)
    response = TestClient(create_app()).get(
        "/v1/auth/whoami", headers={"X-Actor-Id": "user:mallory"}
    )
    assert response.json() == {"actor_id": settings.api_actor, "method": "disabled"}


# --- The CLI ----------------------------------------------------------------------


def _last_line(output: str) -> str:
    """The token: the one line on stdout (click 8.1's runner mixes stderr in)."""
    return output.strip().splitlines()[-1]


def test_cli_auth_token_prints_and_rotates_the_token(_isolated_api_token):
    from personalos.cli import cli

    runner = CliRunner()
    first = runner.invoke(cli, ["auth", "token"])
    assert first.exit_code == 0, first.output
    token = _last_line(first.stdout)
    assert token.startswith(API_TOKEN_PREFIX)
    assert _isolated_api_token.read_text().strip() == token

    again = runner.invoke(cli, ["auth", "token"])
    assert _last_line(again.stdout) == token

    rotated = runner.invoke(cli, ["auth", "token", "--rotate"])
    assert rotated.exit_code == 0
    assert _last_line(rotated.stdout) not in ("", token)
    assert _isolated_api_token.read_text().strip() == _last_line(rotated.stdout)
    default_redactor().forget_secrets()


# --- Redaction --------------------------------------------------------------------


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, service_name="test", install_global=False)
    try:
        yield exporter
    finally:
        reset_tracing()


def test_a_seeded_token_never_appears_in_logs_or_span_attributes(app, tmp_path, spans, caplog):
    factory = session_factory(tmp_path / "redaction.db")
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    app.dependency_overrides[get_progress_log] = lambda: _FakeProgressLog()
    client = TestClient(app, raise_server_exceptions=False)
    caplog.set_level(logging.DEBUG)

    # Real traffic: authenticated, rejected, and a ticketed stream.
    assert client.get("/v1/auth/whoami", headers=auth_header()).status_code == 200
    assert client.get("/v1/auth/whoami", headers=auth_header(BAD_TOKEN)).status_code == 401
    started = client.post(
        "/v1/chat", json={"message": "find me backend jobs"}, headers=auth_header()
    )
    assert started.status_code == 202, started.text
    ticket = _ticket(client)["ticket"]
    assert client.get(_events(query=f"&ticket={ticket}")).status_code == 200

    # The careless call sites the redactor exists for: a header dumped into a
    # log line, a token inside an exception, a header copied onto a span.
    log = logging.getLogger("tests.api_auth")
    log.warning("request headers: %s", {"Authorization": f"Bearer {SEEDED_TOKEN}"})
    log.info("GET /v1/workflows/x/events?ticket=%s", ticket)
    try:
        raise RuntimeError(f"upstream rejected {SEEDED_TOKEN}")
    except RuntimeError:
        log.exception("call failed")
    with tracing.span(
        "http.request",
        {"http.request.header.authorization": f"Bearer {SEEDED_TOKEN}", "note": SEEDED_TOKEN},
    ) as span:
        span.add_event("retry", {"url": f"/v1/workflows/x/events?ticket={ticket}"})

    logged = "\n".join(
        [record.getMessage() for record in caplog.records]
        + [record.exc_text or "" for record in caplog.records]
        + [caplog.text]
    )
    finished = spans.get_finished_spans()
    assert finished, "the test should have produced spans"
    exported = "\n".join(span.to_json() for span in finished)
    attributes = json.dumps(
        [
            {
                "attributes": dict(span.attributes or {}),
                "events": [dict(event.attributes or {}) for event in span.events],
            }
            for span in finished
        ]
    )

    assert "call failed" in logged and REDACTED in logged
    for secret in (SEEDED_TOKEN, BAD_TOKEN, ticket):
        assert secret not in logged
        assert secret not in exported
        assert secret not in attributes


def test_the_token_is_redacted_by_shape_even_when_never_registered():
    """Another process's token (the CLI rotating it) is caught by its prefix alone."""
    fresh = Redactor()
    payload = {
        "note": f"token was {SEEDED_TOKEN}",
        "url": "/x?ticket=pos_st_abcdefghij.klmnopqrstuv",
    }
    redacted = json.dumps(fresh.redact(payload))
    assert SEEDED_TOKEN not in redacted
    assert "pos_st_abcdefghij" not in redacted


def test_an_audit_payload_carrying_the_token_is_redacted(token_file):
    """Audit rows go through the process-wide redactor, which knows the loaded token."""
    auth = build_api_auth()
    assert isinstance(auth, ApiAuth)
    assert auth.verifier.verify(SEEDED_TOKEN) is not None  # loads, and so registers, it
    assert SEEDED_TOKEN in default_redactor()._known

    redacted = default_redactor().redact(
        {"actor_id": "api", "detail": f"leaked {SEEDED_TOKEN}", "nested": [SEEDED_TOKEN]}
    )

    assert SEEDED_TOKEN not in json.dumps(redacted)
    assert redacted["actor_id"] == "api"


def test_a_new_token_is_known_to_the_redactor_before_it_is_stored(tmp_path):
    redactor = Redactor()
    manager = LocalApiToken(FileApiTokenStore(tmp_path / "t"), redactor=redactor)
    token, created = manager.load_or_create()
    assert created
    assert token.reveal() in redactor._known


def test_principal_repr_is_safe():
    assert repr(Principal("api", AuthMethod.LOCAL_TOKEN)) == (
        "Principal(actor_id='api', method=local_token)"
    )
