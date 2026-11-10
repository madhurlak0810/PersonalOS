"""A one-shot HTTP listener on 127.0.0.1 that catches an OAuth redirect.

The installed-app flow (RFC 8252 section 7.3): the consent page redirects the
user's browser to `http://127.0.0.1:<port>/callback?code=...&state=...`, and
this listener is what answers. It binds the loopback address only -- never
`0.0.0.0` -- on a port the OS picks, accepts the first request to its path,
and stops.

The code it catches is wrapped in a `SecretValue` at once, and the request
line is never logged: `http.server` would otherwise print it, code and all,
to stderr.
"""

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from personalos.domain.credentials import SecretValue

LOOPBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/callback"

_DONE_PAGE = (
    "<!doctype html><meta charset=utf-8><title>PersonalOS</title>"
    "<p>{message}</p><p>You can close this tab and return to the terminal.</p>"
)


@dataclass(frozen=True)
class RedirectResult:
    """What the redirect carried."""

    state: str | None
    code: SecretValue | None
    #: Google's `error` parameter, e.g. `access_denied` when the user declined.
    error: str | None


class RedirectTimeout(TimeoutError):
    """No redirect arrived before the deadline."""


class LoopbackRedirectReceiver:
    """Listen on `127.0.0.1` for one OAuth redirect.

    Use as a context manager, so the socket is closed however the flow ends:

        with LoopbackRedirectReceiver() as receiver:
            open_browser(client.authorization_url(redirect_uri=receiver.redirect_uri, ...))
            result = receiver.wait(timeout=300)
    """

    def __init__(self, port: int = 0, path: str = CALLBACK_PATH):
        """Bind now, so `redirect_uri` names a port that is already ours."""
        self.path = path
        self._result: RedirectResult | None = None
        self._received = threading.Event()
        receiver = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                parts = urlsplit(self.path)
                if parts.path != receiver.path or receiver._received.is_set():
                    self.send_error(404)
                    return
                query = parse_qs(parts.query)
                code = _first(query, "code")
                error = _first(query, "error")
                receiver._result = RedirectResult(
                    state=_first(query, "state"),
                    code=SecretValue(code) if code else None,
                    error=error[:64] if error else None,
                )
                message = (
                    "Consent was not given."
                    if error or not code
                    else "PersonalOS received consent."
                )
                body = _DONE_PAGE.format(message=message).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                receiver._received.set()

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                """Silenced: the request line holds the authorization code."""

        self._server = HTTPServer((LOOPBACK_HOST, port), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The port the OS assigned."""
        return self._server.server_address[1]

    @property
    def redirect_uri(self) -> str:
        """The URI to register as the consent's redirect."""
        return f"http://{LOOPBACK_HOST}:{self.port}{self.path}"

    def __enter__(self) -> "LoopbackRedirectReceiver":
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="oauth-loopback",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def wait(self, timeout: float | None = None) -> RedirectResult:
        """Block until the redirect arrives; raise `RedirectTimeout` if it does not."""
        if not self._received.wait(timeout):
            raise RedirectTimeout("no OAuth redirect arrived before the timeout")
        assert self._result is not None
        return self._result

    def close(self) -> None:
        """Stop listening."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()


def _first(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return values[0] if values and values[0] else None


__all__ = [
    "LOOPBACK_HOST",
    "CALLBACK_PATH",
    "RedirectResult",
    "RedirectTimeout",
    "LoopbackRedirectReceiver",
]
