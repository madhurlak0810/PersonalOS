"""The local alert receiver: Alertmanager's webhook in, one log line per alert out."""

import http.client
import io
import json
import logging
import threading
import urllib.error
import urllib.request

import pytest

from apps.alerts.receiver import ALERT_FIRING_EVENT, ALERT_RESOLVED_EVENT, AlertBook, serve
from personalos.observability.structured_logging import configure_logging

WORKFLOW_ID = "6f1c2e9a-1b2c-4d3e-8f90-123456789abc"


def _notification(status: str) -> dict:
    """An Alertmanager webhook body (version 4), as `deploy/alertmanager` sends it."""
    return {
        "version": "4",
        "status": status,
        "receiver": "local",
        "alerts": [
            {
                "status": status,
                "labels": {
                    "alertname": "WorkflowStalled",
                    "severity": "critical",
                    "workflow_id": WORKFLOW_ID,
                    "reason": "no_progress",
                    # Not on the allow-list: dropped, whatever it holds.
                    "note": "Dear Priya, thanks for the call",
                },
                "annotations": {
                    "summary": f"Workflow {WORKFLOW_ID} is stalled (no_progress) for 16m 0s",
                    "runbook": "docs/ALERTS.md#workflowstalled",
                },
                "startsAt": "2026-11-26T08:00:00Z",
                "fingerprint": "abc123",
            }
        ],
    }


@pytest.fixture
def server():
    book = AlertBook()
    httpd = serve("127.0.0.1", 0, book)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", book
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def logs():
    stream = io.StringIO()
    handler = configure_logging(level="INFO", fmt="json", stream=stream)
    try:
        yield stream
    finally:
        logging.getLogger().removeHandler(handler)


def _post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{url}/alerts", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _get(url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{url}{path}", timeout=5) as response:
        return json.loads(response.read())


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_a_firing_alert_is_logged_and_listed_and_its_resolution_clears_it(server, logs):
    url, _book = server

    assert _post(url, json.dumps(_notification("firing")).encode()) == (200, {"received": 1})
    [firing] = _get(url, "/alerts")["firing"]
    assert firing["labels"] == {
        "alertname": "WorkflowStalled",
        "severity": "critical",
        "workflow_id": WORKFLOW_ID,
        "reason": "no_progress",
    }
    assert firing["runbook"] == "docs/ALERTS.md#workflowstalled"

    assert _post(url, json.dumps(_notification("resolved")).encode())[0] == 200
    assert _get(url, "/alerts")["firing"] == []

    lines = [
        line for line in _lines(logs) if line["event"] in (ALERT_FIRING_EVENT, ALERT_RESOLVED_EVENT)
    ]
    assert [(line["event"], line["level"]) for line in lines] == [
        (ALERT_FIRING_EVENT, "WARNING"),
        (ALERT_RESOLVED_EVENT, "INFO"),
    ]
    assert lines[0]["alertname"] == "WorkflowStalled"
    assert lines[0]["workflow_id"] == WORKFLOW_ID
    assert lines[0]["runbook"] == "docs/ALERTS.md#workflowstalled"
    assert "Priya" not in json.dumps(lines)


def test_a_malformed_or_oversized_notification_is_refused(server):
    url, _book = server

    assert _post(url, b"not json")[0] == 400
    assert _post(url, json.dumps({"alerts": "nope"}).encode())[0] == 400
    # Refused from the declared length alone, before any of the body is read.
    host, port = url.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=5)
    connection.putrequest("POST", "/alerts")
    connection.putheader("Content-Length", str(1024 * 1024 + 1))
    connection.endheaders()
    assert connection.getresponse().status == 413
    connection.close()


def test_health_answers(server):
    url, _book = server

    assert _get(url, "/health") == {"status": "ok"}
