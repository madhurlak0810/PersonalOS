"""Tests for the API app factory and its error envelope.

Building the app is itself an assertion: FastAPI rejects a route whose body
annotation is not a valid Pydantic field type while the decorator runs, so a
broken route fails here rather than at deploy time.

The legacy `/api/v1/jobs` routes are gone. Job searches start through
`/v1/workflows`, which runs them under the policy engine and audit trail;
those routes are covered in `test_workflow_routes.py`.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from apps.api.errors import register_error_handlers
from apps.api.main import create_app
from personalos.bootstrap import register_mcp_servers
from personalos.domain.errors import NotFound
from personalos.mcp.manager import MCPServerManager


class _Body(BaseModel):
    title: str


@pytest.fixture
def client():
    """Client over a bare app with the same error handlers `create_app()` wires on."""
    app = FastAPI()

    @app.post("/things")
    async def create_thing(body: _Body):
        return {"title": body.title}

    @app.get("/things/{thing_id}")
    async def get_thing(thing_id: str):
        raise NotFound(f"thing '{thing_id}' not found", details={"thing_id": thing_id})

    @app.get("/boom")
    async def boom():
        raise RuntimeError("connection to db-primary-7 refused: password auth failed")

    register_error_handlers(app)
    return TestClient(app)


def test_app_factory_builds():
    """create_app() succeeds and registers the workflow routes."""
    paths = set(create_app().openapi()["paths"])
    assert "/v1/workflows/{workflow_id}" in paths


def test_legacy_jobs_routes_are_not_registered():
    """Nothing is mounted under the old `/api/v1/jobs` prefix."""
    paths = create_app().openapi()["paths"]
    assert not [path for path in paths if path.startswith("/api/v1/jobs")]


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/v1/jobs/"),
        ("get", "/api/v1/jobs/"),
        ("get", "/api/v1/jobs/6f1e7b3a-0000-4000-8000-000000000000"),
    ],
)
def test_legacy_jobs_route_is_a_404(method, path):
    """The old route cannot start a job search: it no longer exists."""
    response = TestClient(create_app()).request(method, path, json={"title": "Python roles"})
    assert response.status_code == 404


def test_startup_does_not_register_a_jobs_mcp_server():
    """The mock jobs server is no longer registered; real providers live behind the gateway."""
    manager = register_mcp_servers(MCPServerManager())
    assert "jobs" not in manager.list_servers()


def test_validation_failure_uses_the_error_envelope(client):
    """A request that fails body validation is a 422 with the stable envelope."""
    response = client.post("/things", json={})
    assert response.status_code == 422

    body = response.json()
    assert body["error_code"] == "validation"
    assert body["context_id"]
    assert "message" in body


def test_domain_not_found_is_a_404_envelope(client):
    """A missing resource is not reported as a server error, and carries the stable envelope."""
    response = client.get("/things/t1")
    assert response.status_code == 404

    body = response.json()
    assert body["error_code"] == "not_found"
    assert body["context_id"]
    assert set(body) == {"error_code", "message", "context_id"}


def test_unexpected_failure_is_sanitized_before_it_reaches_the_client(client):
    """An unclassified exception becomes a generic envelope, not a leaked string.

    The internal detail (here, a made-up DB failure message) must never reach
    the response body; only the stack trace, logged internally, should carry it.

    Starlette's `ServerErrorMiddleware` re-raises an exception caught by a bare
    `Exception` handler after sending its response, so that a real ASGI server
    still logs it -- `TestClient` mirrors that by re-raising too unless told
    not to. Production callers only ever see the sanitized response sent
    before that re-raise; this test asserts on that response.
    """
    non_raising_client = TestClient(client.app, raise_server_exceptions=False)
    response = non_raising_client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "internal"
    assert "db-primary-7" not in body["message"]
    assert "password" not in body["message"]
    assert body["context_id"]
    assert set(body) == {"error_code", "message", "context_id"}
