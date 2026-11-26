"""`personalos.api.request.duration`: one point per request, labelled by route template."""

from uuid import uuid4

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from apps.api.metrics import UNMATCHED_ROUTE, RequestMetricsMiddleware, route_template
from personalos.observability.metrics_setup import configure_metrics, reset_metrics
from tests.unit.test_worker_process import _client, _database


@pytest.fixture
def reader() -> InMemoryMetricReader:
    reader = InMemoryMetricReader()
    configure_metrics(reader=reader, install_global=False)
    try:
        yield reader
    finally:
        reset_metrics()


def _points(reader: InMemoryMetricReader) -> list[dict]:
    data = reader.get_metrics_data()
    return [
        dict(point.attributes)
        for resource in (data.resource_metrics if data else ())
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == "personalos.api.request.duration"
        for point in metric.data.data_points
    ]


def test_requests_are_labelled_with_the_full_route_template_never_the_path(tmp_path, reader):
    _, factory = _database(tmp_path)
    client = _client(factory)
    workflow_id = uuid4()

    client.get("/health")
    client.get(f"/v1/workflows/{workflow_id}?ticket=not-a-label")
    client.get("/no/such/route/priya@example.com")

    points = _points(reader)
    assert {"route": "/health", "method": "GET", "status_class": "2xx"} in points
    assert {
        "route": "/v1/workflows/{workflow_id}",
        "method": "GET",
        "status_class": "4xx",
    } in points
    assert {"route": UNMATCHED_ROUTE, "method": "GET", "status_class": "4xx"} in points
    labels = " ".join(str(value) for point in points for value in point.values())
    assert str(workflow_id) not in labels
    assert "ticket" not in labels and "priya" not in labels


def test_a_handler_that_raises_is_recorded_as_a_5xx():
    router = APIRouter()

    @router.get("/boom/{thing_id}")
    def boom(thing_id: str):
        raise RuntimeError("boom")

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.add_middleware(RequestMetricsMiddleware)
    reader = InMemoryMetricReader()
    configure_metrics(reader=reader, install_global=False)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/v1/boom/42").status_code == 500
        assert _points(reader) == [
            {"route": "/v1/boom/{thing_id}", "method": "GET", "status_class": "5xx"}
        ]
    finally:
        reset_metrics()


def test_route_template_restores_the_router_prefix():
    class Route:
        def __init__(self, path):
            self.path = path

    assert route_template({"route": Route("/chat"), "path": "/v1/chat"}) == "/v1/chat"
    assert (
        route_template({"route": Route("/workflows/{workflow_id}"), "path": "/v1/workflows/abc"})
        == "/v1/workflows/{workflow_id}"
    )
    assert route_template({"route": Route("/health"), "path": "/health"}) == "/health"
    assert route_template({"path": "/v1/anything"}) == UNMATCHED_ROUTE
