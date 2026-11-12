"""The API's HTTP-layer defences: CORS, security headers, and a body-size cap.

A browser UI served from its own origin (Phase M) calls this API
cross-origin. Three things keep that from over-exposing the API:

- **CORS is an allow-list from configuration** (`API_CORS_ORIGINS`, by
  default only the local dev UI at `http://localhost:5173`). A wildcard is
  refused outright rather than merely when credentials are on: the API
  authenticates with a bearer token, never a cookie, so `allow_credentials`
  is off as well, and nothing needs `*`. Only the methods the routes serve
  (GET, POST, PUT) and the headers the UI sends (`Authorization`,
  `Content-Type`, `Last-Event-ID`) pass a preflight. A preflight that fails
  carries no `Access-Control-Allow-*` header at all (Starlette's stock
  middleware still sends the allowed methods and headers on a refusal), so a
  disallowed origin learns nothing about the policy.
- **Every response carries the security headers** (`SECURITY_HEADERS`): a
  `Content-Security-Policy` that loads nothing and cannot be framed,
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` and
  `Referrer-Policy: no-referrer`. The API answers in JSON, so it needs no
  more. The interactive docs (`/docs`, `/redoc`) are the one HTML the API
  serves; they get `DOCS_CONTENT_SECURITY_POLICY`, which admits the CDN their
  scripts come from and nothing else. A UI server can send the same set by
  using `SECURITY_HEADERS`, swapping in a CSP that lists its own assets.
- **A request body is capped** (`API_MAX_BODY_BYTES`, 1 MiB). Nothing the
  API accepts comes near it: documents are ingested by path, not uploaded. A
  declared `Content-Length` over the cap is refused before the body is read;
  a chunked body is counted as it arrives and refused once it passes the cap.

Ordering (see `apps.api.main.create_app`): the security headers are
outermost, so preflights, CORS refusals and the 413 all carry them; CORS is
next, so a refusal from anything inside it is still readable by an allowed
UI. Responses from the catch-all 500 handler are produced outside every
middleware (by Starlette's `ServerErrorMiddleware`), so that handler applies
the headers itself (`apps.api.errors`).
"""

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from personalos.domain.errors import PayloadTooLarge

#: The only methods the API's routes serve. `OPTIONS` is the preflight
#: itself and never needs to be allowed by one.
CORS_ALLOWED_METHODS = ("GET", "POST", "PUT")

#: The only request headers the UI sends: the bearer token, a JSON body's
#: type, and the SSE resume cursor. Not `X-Actor-Id` (ignored anyway), not
#: `X-Correlation-Id` (a browser has no correlation id of its own to send).
CORS_ALLOWED_HEADERS = ("Authorization", "Content-Type", "Last-Event-ID")

#: Response headers a cross-origin script may read beyond the CORS-safelisted
#: ones: the correlation id, so the UI can quote it when reporting an error.
CORS_EXPOSED_HEADERS = ("X-Correlation-Id",)

#: How long a browser may cache a successful preflight, in seconds.
CORS_MAX_AGE_SECONDS = 600

#: The API answers in JSON: a response that a browser renders anyway (a
#: mistyped error page, say) may load nothing, run nothing, and be framed by
#: no one.
API_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)

#: For FastAPI's Swagger UI and ReDoc pages: their bundles come from
#: jsDelivr and they bootstrap with an inline script; they fetch the
#: OpenAPI schema from this origin. Nothing else is admitted, and they
#: cannot be framed either.
DOCS_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.redoc.ly; "
    "connect-src 'self'; "
    "worker-src blob:; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)

#: The paths that serve `DOCS_CONTENT_SECURITY_POLICY` instead.
DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})

#: Sent on every response, the CSP aside (which depends on the path).
SECURITY_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": API_CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    # Legacy twin of `frame-ancestors 'none'`, for browsers that predate it.
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

#: The default request body cap: 1 MiB.
DEFAULT_MAX_BODY_BYTES = 1024 * 1024


def validate_cors_origins(origins: Iterable[str]) -> tuple[str, ...]:
    """The configured origins, checked to be exact `scheme://host[:port]` values.

    Raises `ValueError` for a wildcard, `null`, a non-HTTP scheme, a
    path, query, fragment or credentials, or anything else that is not a
    plain origin. Exact origins are all `CORSMiddleware` compares against,
    so a malformed entry would otherwise just never match -- or, for `*`,
    match everything.
    """
    checked: list[str] = []
    for raw in origins:
        origin = raw.strip()
        if "*" in origin:
            raise ValueError(
                f"API_CORS_ORIGINS may not contain a wildcard ({origin!r}); list each UI origin"
            )
        parts = urlsplit(origin)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path
            or parts.query
            or parts.fragment
            or origin.lower() != f"{parts.scheme}://{parts.netloc.lower()}"
        ):
            raise ValueError(
                f"API_CORS_ORIGINS entry {origin!r} is not an origin "
                "(expected scheme://host[:port], no path or trailing slash)"
            )
        # Browsers send the scheme and host lower-cased.
        checked.append(origin.lower())
    return tuple(dict.fromkeys(checked))


class StrictCORSMiddleware:
    """`CORSMiddleware`, but a response to anyone but an allowed origin discloses nothing.

    Starlette answers a failed preflight with a 400 that still lists the
    allowed methods, headers and max-age, and (depending on the version) adds
    `Access-Control-Expose-Headers` to responses for any origin. A browser
    would block the request regardless, but there is no reason to tell a
    disallowed origin what an allowed one may do.

    It wraps Starlette's middleware rather than subclassing it, filtering at
    the ASGI boundary, so it depends on nothing but Starlette's constructor:
    the internals it would otherwise override change between releases.
    """

    def __init__(self, app: ASGIApp, *, allow_origins: Sequence[str]) -> None:
        self.allowed_origins = frozenset(validate_cors_origins(allow_origins))
        self.cors = CORSMiddleware(
            app,
            allow_origins=sorted(self.allowed_origins),
            allow_methods=CORS_ALLOWED_METHODS,
            allow_headers=CORS_ALLOWED_HEADERS,
            allow_credentials=False,
            expose_headers=CORS_EXPOSED_HEADERS,
            max_age=CORS_MAX_AGE_SECONDS,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.cors(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin_allowed = headers.get("origin") in self.allowed_origins
        is_preflight = scope["method"] == "OPTIONS" and "access-control-request-method" in headers

        async def filtered_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                refused_preflight = is_preflight and message["status"] != 200
                if not origin_allowed or refused_preflight:
                    message.setdefault("headers", [])
                    _strip_cors_headers(MutableHeaders(scope=message))
            await send(message)

        await self.cors(scope, receive, filtered_send)


def _strip_cors_headers(headers: MutableHeaders) -> None:
    for name in {n for n in headers.keys() if n.lower().startswith("access-control-")}:
        del headers[name]


def apply_security_headers(headers: MutableMapping[str, str], path: str) -> None:
    """Set `SECURITY_HEADERS` on `headers`, with the CSP `path` calls for.

    Overwrites: a route cannot opt out of these by setting its own.
    """
    for name, value in SECURITY_HEADERS.items():
        headers[name] = value
    if path in DOCS_PATHS:
        headers["Content-Security-Policy"] = DOCS_CONTENT_SECURITY_POLICY


class SecurityHeadersMiddleware:
    """Adds `SECURITY_HEADERS` to every HTTP response, streamed ones included."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                apply_security_headers(MutableHeaders(scope=message), path)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _BodyTooLarge(Exception):
    """Raised from inside `receive` once a streamed body passes the cap."""


class BodySizeLimitMiddleware:
    """Refuses a request whose body is over `max_bytes` with a 413 envelope.

    A streamed body that passes the cap mid-read raises inside `receive`.
    FastAPI turns any error reading the body into its own 400, so rather than
    rely on the exception reaching here, whatever response the app then
    produces is discarded and the 413 sent in its place.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("API_MAX_BODY_BYTES must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self._declares_too_much(scope):
            await self._refuse(send)
            return

        guard = _BodyGuard(self.max_bytes, receive, send, self._refuse)
        try:
            await self.app(scope, guard.receive, guard.send)
        except Exception:
            if not guard.exceeded:
                raise
        await guard.finish()

    def _declares_too_much(self, scope: Scope) -> bool:
        declared = Headers(scope=scope).get("content-length")
        if declared is None:
            return False
        try:
            return int(declared) > self.max_bytes
        except ValueError:
            return False  # The server rejects a malformed length itself.

    async def _refuse(self, send: Send) -> None:
        error = PayloadTooLarge(details={"max_bytes": self.max_bytes})
        body = json.dumps(error.to_envelope()).encode()
        await send(
            {
                "type": "http.response.start",
                "status": error.http_status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    # The rest of the body is never read; don't keep the socket.
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyGuard:
    """One request's view of `BodySizeLimitMiddleware`: counts, then swaps the response."""

    def __init__(self, max_bytes: int, receive: Receive, send: Send, refuse) -> None:
        self.max_bytes = max_bytes
        self._receive = receive
        self._send = send
        self._refuse = refuse
        self.received = 0
        self.exceeded = False
        self.refused = False

    async def receive(self) -> Message:
        message = await self._receive()
        if message["type"] == "http.request":
            self.received += len(message.get("body", b""))
            if self.received > self.max_bytes:
                self.exceeded = True
                raise _BodyTooLarge
        return message

    async def send(self, message: Message) -> None:
        if not self.exceeded:
            await self._send(message)
        else:
            # Whatever the app answers an overrun with is replaced by the 413.
            await self.finish()

    async def finish(self) -> None:
        if self.exceeded and not self.refused:
            self.refused = True
            await self._refuse(self._send)


__all__ = [
    "API_CONTENT_SECURITY_POLICY",
    "CORS_ALLOWED_HEADERS",
    "CORS_ALLOWED_METHODS",
    "CORS_EXPOSED_HEADERS",
    "DEFAULT_MAX_BODY_BYTES",
    "DOCS_CONTENT_SECURITY_POLICY",
    "SECURITY_HEADERS",
    "BodySizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "StrictCORSMiddleware",
    "apply_security_headers",
    "validate_cors_origins",
]
