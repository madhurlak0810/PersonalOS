"""The HTTP layer under hostile requests: CORS, security headers, body limits.

The browser UI calls the API from its own origin, so the API's first line of
defence is the HTTP layer itself (`apps.api.security`). These tests run the
production app (`create_app()`), with the test bearer verifier installed, and
throw at it what a hostile page or client would: an origin that is not the
UI's, a preflight asking for more than the UI needs, a body bigger than the
cap, and JSON that does not parse.

Acceptance criteria covered:

- a request from a disallowed origin gets no CORS allow headers;
- the configured UI origin completes a preflight and an authenticated request;
- the security headers are present on representative responses.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.security import (
    API_CONTENT_SECURITY_POLICY,
    DOCS_CONTENT_SECURITY_POLICY,
    SECURITY_HEADERS,
    validate_cors_origins,
)
from personalos.config import settings
from tests.fixtures.auth import bearer, install_test_auth

UI_ORIGIN = "http://localhost:5173"

#: Origins a hostile page could be served from. None of them is the UI.
DISALLOWED_ORIGINS = {
    "other-site": "https://evil.example",
    "other-port": "http://localhost:8080",
    "other-scheme": "https://localhost:5173",
    "suffix-trick": "http://localhost:5173.evil.example",
    "prefix-trick": "http://evil-localhost:5173",
    "loopback-ip-instead": "http://127.0.0.1:5173",
    "sandboxed-iframe": "null",
    "file-url": "file://",
}

#: A small cap, so the oversized-body tests do not allocate megabytes.
MAX_BODY_BYTES = 4096


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(settings, "api_cors_origins", [UI_ORIGIN])
    monkeypatch.setattr(settings, "api_max_body_bytes", MAX_BODY_BYTES)
    return install_test_auth(create_app())


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


def _cors_headers(response) -> dict[str, str]:
    return {
        name.lower(): value
        for name, value in response.headers.items()
        if name.lower().startswith("access-control-")
    }


def _preflight(
    client,
    *,
    origin=UI_ORIGIN,
    method="POST",
    headers="authorization,content-type",
    path="/v1/auth/stream-tickets",
    extra=None,
):
    request_headers = {"Origin": origin, "Access-Control-Request-Method": method}
    if headers is not None:
        request_headers["Access-Control-Request-Headers"] = headers
    request_headers.update(extra or {})
    return client.options(path, headers=request_headers)


def _assert_security_headers(response, csp=API_CONTENT_SECURITY_POLICY):
    for name, value in SECURITY_HEADERS.items():
        if name == "Content-Security-Policy":
            continue
        assert response.headers.get(name) == value, name
    assert response.headers.get("Content-Security-Policy") == csp


# --- The configured UI origin ---------------------------------------------------


def test_ui_origin_completes_a_preflight_and_an_authenticated_request(client):
    preflight = _preflight(client)

    assert preflight.status_code == 200
    cors = _cors_headers(preflight)
    assert cors["access-control-allow-origin"] == UI_ORIGIN
    allowed_methods = {m.strip() for m in cors["access-control-allow-methods"].split(",")}
    assert allowed_methods == {"GET", "POST", "PUT"}
    allowed_headers = {h.strip().lower() for h in cors["access-control-allow-headers"].split(",")}
    assert {"authorization", "content-type", "last-event-id"} <= allowed_headers
    # Bearer tokens, never cookies: credentials mode is not allowed.
    assert "access-control-allow-credentials" not in cors

    response = client.post(
        "/v1/auth/stream-tickets",
        json={"workflow_id": str(uuid4())},
        headers={"Origin": UI_ORIGIN, **bearer("candidate")},
    )

    assert response.status_code == 201
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN
    assert "access-control-allow-credentials" not in _cors_headers(response)
    # The UI can read the correlation id to quote it in a bug report.
    assert "x-correlation-id" in response.headers["access-control-expose-headers"].lower()
    assert "origin" in response.headers.get("vary", "").lower()


def test_ui_origin_can_preflight_the_event_stream_with_last_event_id(client):
    preflight = _preflight(
        client,
        method="GET",
        headers="authorization,last-event-id",
        path=f"/v1/workflows/{uuid4()}/events",
    )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == UI_ORIGIN


def test_ui_origin_gets_cors_headers_on_an_error_too(client):
    """A 401 the UI cannot read is a 401 it cannot explain to the user."""
    response = client.get("/v1/auth/whoami", headers={"Origin": UI_ORIGIN})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN


# --- Disallowed origins ---------------------------------------------------------


@pytest.mark.parametrize("origin", DISALLOWED_ORIGINS.values(), ids=DISALLOWED_ORIGINS.keys())
def test_disallowed_origin_gets_no_cors_allow_headers_on_a_request(client, origin):
    response = client.get("/v1/auth/whoami", headers={"Origin": origin, **bearer()})

    # The server still answers (CORS is the browser's to enforce), but with
    # nothing that would let the browser hand the response to the page.
    assert _cors_headers(response) == {}


@pytest.mark.parametrize("origin", DISALLOWED_ORIGINS.values(), ids=DISALLOWED_ORIGINS.keys())
def test_disallowed_origin_preflight_is_refused_and_discloses_nothing(client, origin):
    response = _preflight(client, origin=origin)

    assert response.status_code == 400
    # Not even the allowed methods and headers: Starlette's stock middleware
    # sends those on a refusal, ours does not.
    assert _cors_headers(response) == {}


# --- Preflight abuse ------------------------------------------------------------


@pytest.mark.parametrize("method", ["DELETE", "PATCH", "TRACE", "CONNECT", "post"])
def test_preflight_for_a_method_the_ui_does_not_use_is_refused(client, method):
    response = _preflight(client, method=method)

    assert response.status_code == 400
    assert _cors_headers(response) == {}


@pytest.mark.parametrize(
    "headers",
    [
        "x-actor-id",
        "authorization,x-actor-id",
        "cookie",
        "x-correlation-id",
        "authorization," + ",".join(f"x-filler-{i}" for i in range(200)),
    ],
    ids=["actor-override", "smuggled-after-allowed", "cookie", "correlation-id", "header-flood"],
)
def test_preflight_for_a_header_the_ui_does_not_send_is_refused(client, headers):
    response = _preflight(client, headers=headers)

    assert response.status_code == 400
    assert _cors_headers(response) == {}


def test_preflight_asking_for_private_network_access_is_refused(client):
    """A public page asking to reach this loopback API (Private Network Access)."""
    response = _preflight(client, extra={"Access-Control-Request-Private-Network": "true"})

    assert response.status_code == 400
    assert _cors_headers(response) == {}


def test_options_without_a_request_method_is_not_a_preflight(client):
    """An `OPTIONS` that is not a real preflight falls through to the routes."""
    response = client.options("/v1/auth/whoami", headers={"Origin": "https://evil.example"})

    assert response.status_code in (401, 405)
    assert _cors_headers(response) == {}


def test_preflight_never_reaches_authentication_or_the_route(client):
    """A preflight carries no token; it is answered before auth, with no body leaked."""
    response = _preflight(client, path="/v1/auth/whoami", method="GET", headers="authorization")

    assert response.status_code == 200
    assert response.text == "OK"


# --- Configuration ----------------------------------------------------------------


@pytest.mark.parametrize(
    "origins",
    [
        ["*"],
        ["http://*.example.com"],
        [UI_ORIGIN, "*"],
        ["null"],
        ["localhost:5173"],
        ["http://localhost:5173/"],
        ["http://localhost:5173/app"],
        ["http://user@localhost:5173"],
        ["ftp://localhost:5173"],
        ["http://"],
    ],
)
def test_misconfigured_origins_are_refused_at_startup(monkeypatch, origins):
    monkeypatch.setattr(settings, "api_cors_origins", origins)

    with pytest.raises(ValueError):
        create_app()


def test_origins_are_normalised_and_deduplicated():
    assert validate_cors_origins(["HTTP://LocalHost:5173", " http://localhost:5173 "]) == (
        UI_ORIGIN,
    )


def test_default_allows_only_the_local_dev_ui():
    from personalos.config import Settings

    assert Settings().api_cors_origins == [UI_ORIGIN]


def test_a_configured_origin_is_allowed_and_the_default_then_is_not(monkeypatch):
    monkeypatch.setattr(settings, "api_cors_origins", ["https://ui.internal.example"])
    client = TestClient(install_test_auth(create_app()))

    allowed = _preflight(client, origin="https://ui.internal.example")
    default = _preflight(client, origin=UI_ORIGIN)

    assert allowed.headers["access-control-allow-origin"] == "https://ui.internal.example"
    assert default.status_code == 400
    assert _cors_headers(default) == {}


# --- Security headers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "status"),
    [
        ("GET", "/health", {}, 200),
        ("GET", "/v1/auth/whoami", {"headers": bearer()}, 200),
        ("GET", "/v1/auth/whoami", {}, 401),
        ("GET", "/v1/no-such-route", {"headers": bearer()}, 404),
        ("POST", "/v1/auth/stream-tickets", {"headers": bearer(), "json": {}}, 422),
        ("GET", "/openapi.json", {}, 200),
    ],
    ids=["health", "authenticated", "unauthenticated", "not-found", "validation", "openapi"],
)
def test_security_headers_are_on_representative_responses(client, method, path, kwargs, status):
    response = client.request(method, path, **kwargs)

    assert response.status_code == status
    _assert_security_headers(response)


def test_security_headers_are_on_preflights_allowed_and_refused(client):
    _assert_security_headers(_preflight(client))
    _assert_security_headers(_preflight(client, origin="https://evil.example"))


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_docs_pages_get_their_own_csp_and_still_cannot_be_framed(client, path):
    response = client.get(path)

    assert response.status_code == 200
    _assert_security_headers(response, csp=DOCS_CONTENT_SECURITY_POLICY)
    assert "frame-ancestors 'none'" in DOCS_CONTENT_SECURITY_POLICY


def test_security_headers_are_on_an_unhandled_500(app):
    """The catch-all 500 is sent outside every middleware; it sets them itself."""

    @app.get("/v1/explode")
    def explode():
        raise RuntimeError("boom")

    response = TestClient(app, raise_server_exceptions=False).get("/v1/explode")

    assert response.status_code == 500
    assert response.json()["error_code"] == "internal"
    _assert_security_headers(response)


def test_a_route_cannot_weaken_the_security_headers(app):
    from fastapi.responses import JSONResponse

    @app.get("/v1/framable")
    def framable():
        return JSONResponse(
            {}, headers={"X-Frame-Options": "ALLOWALL", "Content-Security-Policy": "default-src *"}
        )

    response = TestClient(app).get("/v1/framable")

    _assert_security_headers(response)


# --- Oversized bodies --------------------------------------------------------------


def _oversized_json() -> bytes:
    return b'{"workflow_id": "' + b"a" * (MAX_BODY_BYTES * 2) + b'"}'


def test_oversized_body_with_a_content_length_is_refused_before_it_is_read(client):
    response = client.post(
        "/v1/auth/stream-tickets",
        content=_oversized_json(),
        headers={"Content-Type": "application/json", "Origin": UI_ORIGIN, **bearer()},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "payload_too_large"
    assert set(body) == {"error_code", "message", "context_id"}
    _assert_security_headers(response)
    # The UI can still read why its request was refused.
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN
    assert "x-correlation-id" in response.headers


def test_oversized_chunked_body_is_refused_once_it_passes_the_cap(client):
    """No `Content-Length` to check up front: the body is counted as it arrives."""

    def chunks():
        yield b'{"workflow_id": "'
        for _ in range(8):
            yield b"a" * 1024
        yield b'"}'

    response = client.post(
        "/v1/auth/stream-tickets",
        content=chunks(),
        headers={"Content-Type": "application/json", **bearer()},
    )

    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"
    _assert_security_headers(response)


def test_oversized_body_is_refused_even_without_a_token(client):
    """The cap is checked before authentication: an anonymous flood costs nothing."""
    response = client.post(
        "/v1/auth/stream-tickets",
        content=_oversized_json(),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413


def test_a_body_at_the_cap_is_accepted(client):
    payload = b'{"workflow_id": "%s"}' % str(uuid4()).encode()
    padded = payload + b" " * (MAX_BODY_BYTES - len(payload))
    assert len(padded) == MAX_BODY_BYTES

    response = client.post(
        "/v1/auth/stream-tickets",
        content=padded,
        headers={"Content-Type": "application/json", **bearer()},
    )

    assert response.status_code == 201


def test_a_non_positive_cap_is_refused_at_startup(monkeypatch):
    monkeypatch.setattr(settings, "api_max_body_bytes", 0)

    with pytest.raises(ValueError):
        # Starlette builds the middleware stack on the first request.
        TestClient(create_app()).get("/health")


# --- Malformed JSON ----------------------------------------------------------------

MALFORMED_BODIES = {
    "truncated": b'{"workflow_id": "',
    "trailing-garbage": b'{"workflow_id": "%s"} junk' % str(uuid4()).encode(),
    "single-quotes": b"{'workflow_id': 'x'}",
    "not-utf8": b'{"workflow_id": "\xff\xfe"}',
    "nan-literal": b'{"workflow_id": NaN}',
    "deeply-nested": b"[" * 1500 + b"]" * 1500,
    "bare-string": b'"just a string"',
    "array-not-object": b'["workflow_id"]',
    "wrong-type": b'{"workflow_id": {"$ne": null}}',
    "empty-object": b"{}",
}


#: Bodies the JSON decoder itself gives up on (bytes that are not UTF-8), which
#: FastAPI answers with its own 400 "error parsing the body" rather than a
#: `RequestValidationError`. Still a refusal, still not a 500.
UNDECODABLE = {"not-utf8"}

#: Bodies whose outcome depends on the Python version: the JSON decoder's
#: recursion limit is lower on older interpreters (3.11 gives up on 1500
#: levels, 3.14 parses them). Either answer -- FastAPI's 400 or the 422
#: envelope -- is a clean refusal; what must never happen is a 500.
DEPTH_DEPENDENT = {"deeply-nested"}


@pytest.mark.parametrize("name", MALFORMED_BODIES.keys())
def test_malformed_json_is_refused_cleanly_not_a_500(client, name):
    body = MALFORMED_BODIES[name]
    response = client.post(
        "/v1/auth/stream-tickets",
        content=body,
        headers={"Content-Type": "application/json", **bearer()},
    )

    _assert_security_headers(response)
    if name in DEPTH_DEPENDENT:
        assert response.status_code in (400, 422)
        assert body.decode() not in response.text
        return
    if name in UNDECODABLE:
        assert response.status_code == 400
        assert response.json() == {"detail": "There was an error parsing the body"}
        return
    assert response.status_code == 422
    envelope = response.json()
    assert envelope["error_code"] == "validation"
    # Only the stable fields: the parser's view of the input is not echoed.
    assert set(envelope) == {"error_code", "message", "context_id"}


def test_json_sent_as_another_content_type_is_not_parsed(client):
    response = client.post(
        "/v1/auth/stream-tickets",
        content=b'{"workflow_id": "%s"}' % str(uuid4()).encode(),
        headers={"Content-Type": "text/plain", **bearer()},
    )

    # A CORS "simple request" (text/plain skips the preflight) must not be
    # able to drive a JSON route.
    assert response.status_code == 422
