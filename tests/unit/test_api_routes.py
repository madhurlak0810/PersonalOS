"""Tests for the job search API.

The app used to fail at import: FastAPI rejects a route whose body annotation
is not a valid Pydantic field type, and the request/response classes were plain
classes. These tests cover that directly -- building the app is the assertion
-- as well as the round trip through the endpoints.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.errors import register_error_handlers
from apps.api.main import create_app
from apps.api.routes import jobs as jobs_routes
from personalos.persistence import get_session
from personalos.persistence.models import Base


@pytest.fixture
def session_factory(tmp_path):
    """Session factory over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture
def client(session_factory):
    """Client over a router-only app, so no lifespan or real database is used.

    Registers the same error handlers `create_app()` wires on, so a route
    raising a domain error is turned into the sanitized envelope here too,
    rather than propagating as a raw exception through the test client.
    """
    app = FastAPI()
    app.include_router(jobs_routes.router, prefix="/api/v1/jobs")
    register_error_handlers(app)

    def override_session():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = override_session
    return TestClient(app)


def test_app_factory_builds():
    """create_app() succeeds and registers the job routes.

    This is the regression test for the import failure: an invalid body
    annotation raises FastAPIError while the decorator runs, so a broken route
    fails here rather than at deploy time.
    """
    paths = set(create_app().openapi()["paths"])
    assert {"/api/v1/jobs/", "/api/v1/jobs/{job_id}"} <= paths


def test_openapi_schema_is_generated():
    """Request and response models produce a usable schema."""
    schema = create_app().openapi()
    assert "JobCreateRequest" in schema["components"]["schemas"]
    assert "JobResponse" in schema["components"]["schemas"]


def test_create_job_search_returns_a_summary(client):
    """A created search is persisted and echoed back as pending."""
    response = client.post(
        "/api/v1/jobs/",
        json={
            "title": "Python roles",
            "keywords": ["python"],
            "locations": ["remote"],
            "salary_min": 100000,
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["title"] == "Python roles"
    assert body["status"] == "pending"
    assert body["id"]


def test_create_rejects_a_request_missing_its_title(client):
    """Validation now happens, because the request model is a real model."""
    response = client.post("/api/v1/jobs/", json={"keywords": ["python"]})
    assert response.status_code == 422

    body = response.json()
    assert body["error_code"] == "validation"
    assert body["context_id"]
    assert "message" in body


def test_get_job_search_returns_the_full_view(client):
    """A created job can be read back with its full field set."""
    created = client.post(
        "/api/v1/jobs/",
        json={"title": "Python roles", "keywords": ["python"], "locations": ["remote"]},
    ).json()

    response = client.get(f"/api/v1/jobs/{created['id']}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == created["id"]
    assert body["keywords"] == ["python"]
    assert body["results_count"] == 0
    assert body["started_at"] is None


def test_get_unknown_job_is_a_404(client):
    """A missing job is not reported as a server error, and carries the stable envelope."""
    response = client.get("/api/v1/jobs/6f1e7b3a-0000-4000-8000-000000000000")
    assert response.status_code == 404

    body = response.json()
    assert body["error_code"] == "not_found"
    assert body["context_id"]
    assert set(body) == {"error_code", "message", "context_id"}


def test_unexpected_failure_is_sanitized_before_it_reaches_the_client(client, monkeypatch):
    """An unclassified exception becomes a generic envelope, not a leaked string.

    The internal detail (here, a made-up DB failure message) must never reach
    the response body; only the stack trace, logged internally, should carry it.

    Starlette's `ServerErrorMiddleware` re-raises an exception caught by a bare
    `Exception` handler after sending its response, so that a real ASGI server
    still logs it -- `TestClient` mirrors that by re-raising too unless told
    not to. Production callers only ever see the sanitized response sent
    before that re-raise; this test asserts on that response.
    """
    from fastapi.testclient import TestClient

    from personalos.persistence.repositories import JobRepository

    def boom(self):
        raise RuntimeError("connection to db-primary-7 refused: password auth failed")

    monkeypatch.setattr(JobRepository, "get_all", boom)

    non_raising_client = TestClient(client.app, raise_server_exceptions=False)
    response = non_raising_client.get("/api/v1/jobs/")

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "internal"
    assert "db-primary-7" not in body["message"]
    assert "password" not in body["message"]
    assert body["context_id"]
    assert set(body) == {"error_code", "message", "context_id"}


def test_list_job_searches(client):
    """Listing returns every stored search."""
    for title in ("first", "second"):
        client.post("/api/v1/jobs/", json={"title": title, "keywords": ["python"]})

    response = client.get("/api/v1/jobs/")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 2
    assert {job["title"] for job in body["jobs"]} == {"first", "second"}
