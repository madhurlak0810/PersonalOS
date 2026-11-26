"""The local alert receiver: where Alertmanager delivers when nothing external is configured.

A single-user stack should alert without a mail server, a chat workspace or a
paging service. So Alertmanager's default receiver (`deploy/alertmanager/
alertmanager.yml`) is this process, a webhook on the stack's internal network,
and what it does with a notification is:

- **write one log line per alert**, through the same structured logging and
  redaction as the API and worker: `alert.firing` at WARNING or
  `alert.resolved` at INFO, with `alertname`, `severity`, `summary`, `runbook`
  and the alert's identifying labels (`workflow_id`, `integration`, `reason`,
  `route`). `docker compose logs -f alert-receiver` is the notification feed;
- **keep the alerts currently firing**, served as JSON on `GET /alerts`, which
  the compose smoke test reads and an operator can `curl` from inside the
  network.

The Grafana dashboards show the same alerts in their "Firing alerts" panels,
from Prometheus' own `ALERTS` series, so an alert is visible there even if
this process is down. An external channel (mail, Slack, a pager) is a second
Alertmanager receiver; see docs/ALERTS.md.

Only labels from `_LABELS` are logged or served. They come from PersonalOS
metrics, whose labels are already restricted to ids and code-defined names
(`personalos.observability.metrics`); the allow-list keeps anything an edited
rule might add out of the log.

    python -m apps.alerts.receiver [--host 0.0.0.0] [--port 9095]

Standard library only: no framework, so nothing beyond the API image's own
dependencies is needed to run it.
"""

import argparse
import json
import logging
import threading
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from personalos.observability.structured_logging import configure_logging_from_settings

logger = logging.getLogger("apps.alerts")

#: `event` of the line a firing alert writes.
ALERT_FIRING_EVENT = "alert.firing"
#: `event` of the line a resolved alert writes.
ALERT_RESOLVED_EVENT = "alert.resolved"

#: Alert labels that are logged and served; anything else is dropped.
_LABELS = ("alertname", "severity", "workflow_id", "integration", "reason", "route", "job")
#: Annotations that are logged and served.
_ANNOTATIONS = ("summary", "runbook")

#: A notification larger than this is refused; Alertmanager's are a few KB.
MAX_BODY_BYTES = 1024 * 1024
#: Port the receiver listens on, inside the stack's network.
DEFAULT_PORT = 9095


def _pick(values: Any, keys: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        return {}
    return {key: str(values[key])[:500] for key in keys if key in values}


class AlertBook:
    """The alerts currently firing, by fingerprint, as the notifications reported them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._firing: dict[str, dict[str, Any]] = {}

    def receive(self, notification: Mapping[str, Any]) -> int:
        """Log each alert in an Alertmanager webhook notification and track it. Returns how many."""
        alerts = notification.get("alerts")
        if not isinstance(alerts, list):
            raise ValueError("a webhook notification carries an 'alerts' list")
        for alert in alerts:
            if isinstance(alert, Mapping):
                self._one(alert)
        return len(alerts)

    def firing(self) -> list[dict[str, Any]]:
        """Every alert firing now, oldest first."""
        with self._lock:
            return sorted(self._firing.values(), key=lambda alert: alert.get("starts_at", ""))

    def _one(self, alert: Mapping[str, Any]) -> None:
        labels = _pick(alert.get("labels"), _LABELS)
        annotations = _pick(alert.get("annotations"), _ANNOTATIONS)
        status = "resolved" if alert.get("status") == "resolved" else "firing"
        fingerprint = str(alert.get("fingerprint") or sorted(labels.items()))
        record = {
            "status": status,
            "labels": labels,
            **annotations,
            "starts_at": str(alert.get("startsAt") or ""),
        }
        with self._lock:
            if status == "firing":
                self._firing[fingerprint] = record
            else:
                self._firing.pop(fingerprint, None)

        name = labels.get("alertname", "unnamed")
        extra = {
            "event": ALERT_FIRING_EVENT if status == "firing" else ALERT_RESOLVED_EVENT,
            "alertname": name,
            "alert_status": status,
            "severity": labels.get("severity"),
            "summary": annotations.get("summary"),
            "runbook": annotations.get("runbook"),
            **{key: value for key, value in labels.items() if key not in ("alertname", "severity")},
        }
        if status == "firing":
            logger.warning(
                "ALERT %s [%s]: %s (runbook: %s)",
                name,
                labels.get("severity", "unknown"),
                annotations.get("summary", "no summary"),
                annotations.get("runbook", "docs/ALERTS.md"),
                extra=extra,
            )
        else:
            logger.info("resolved: %s (%s)", name, annotations.get("summary", ""), extra=extra)


def _receive(book: AlertBook, declared_length: str | None, body: Any) -> tuple[HTTPStatus, dict]:
    """Read one notification of `declared_length` bytes from `body` into `book`."""
    try:
        length = int(declared_length or 0)
    except ValueError:
        length = -1
    if length < 0 or length > MAX_BODY_BYTES:
        return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"}
    try:
        count = book.receive(json.loads(body.read(length) or b"{}"))
    except (ValueError, TypeError) as exc:
        return HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__}
    return HTTPStatus.OK, {"received": count}


def make_handler(book: AlertBook) -> type[BaseHTTPRequestHandler]:
    """The request handler, bound to `book`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "personalos-alerts"
        sys_version = ""

        def _send(self, status: HTTPStatus, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - the http.server spelling
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/alerts":
                self._send(HTTPStatus.OK, {"firing": book.firing()})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - the http.server spelling
            if self.path != "/alerts":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(*_receive(book, self.headers.get("Content-Length"), self.rfile))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # The access log would be one unstructured line per request; the
            # alerts themselves are what is logged.
            return

    return Handler


def serve(host: str, port: int, book: AlertBook | None = None) -> ThreadingHTTPServer:
    """A server for `book`, bound and ready; call `serve_forever` on it."""
    return ThreadingHTTPServer((host, port), make_handler(book or AlertBook()))


def main(argv: list[str] | None = None) -> None:
    """Run the receiver until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    configure_logging_from_settings()
    server = serve(args.host, args.port)
    logger.info("alert receiver listening on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        pass
    finally:
        server.server_close()


__all__ = ["ALERT_FIRING_EVENT", "ALERT_RESOLVED_EVENT", "AlertBook", "main", "serve"]


if __name__ == "__main__":  # pragma: no cover - the process entry point
    main()
