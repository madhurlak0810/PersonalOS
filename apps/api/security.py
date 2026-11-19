"""The API's HTTP-layer defences: CORS, security headers, body and JSON limits.

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
- **A request body is capped** (`API_MAX_BODY_BYTES`, 1 MiB), except resume
  ingestion (`API_MAX_RESUME_BODY_BYTES`, 10 MiB, its route alone). A
  declared `Content-Length` over the cap is refused before the body is read;
  a chunked body is counted as it arrives and refused once it passes the cap.
- **A JSON body's shape is checked before it is parsed**
  (`JsonLimitsMiddleware`): not UTF-8, nested deeper than
  `API_MAX_JSON_DEPTH`, or holding a string longer than
  `API_MAX_JSON_STRING_BYTES` is a 422, found in one linear pass over the
  raw bytes. See docs/LIMITS.md.

Ordering (see `apps.api.main.create_app`): the security headers are
outermost, so preflights, CORS refusals and the 413 all carry them; CORS is
next, so a refusal from anything inside it is still readable by an allowed
UI. Responses from the catch-all 500 handler are produced outside every
middleware (by Starlette's `ServerErrorMiddleware`), so that handler applies
the headers itself (`apps.api.errors`).
"""

import json
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from personalos.domain.errors import PayloadTooLarge, PersonalOSError, ValidationFailed
from personalos.observability.limits import normalize_path, record_limit_hit

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


#: Paths no body limit applies to: the liveness probe.
LIMIT_EXEMPT_PATHS = frozenset({"/health"})


def _refusal(error: PersonalOSError) -> tuple[Message, Message]:
    """The two ASGI messages of an error-envelope response sent from a middleware."""
    body = json.dumps(error.to_envelope()).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if isinstance(error, PayloadTooLarge):
        # The rest of the body is never read; don't keep the socket.
        headers.append((b"connection", b"close"))
    return (
        {"type": "http.response.start", "status": error.http_status, "headers": headers},
        {"type": "http.response.body", "body": body},
    )


def _route(scope: Scope) -> str:
    return f"{scope.get('method', '-')} {normalize_path(scope.get('path', ''))}"


class BodySizeLimitMiddleware:
    """Refuses a request whose body is over its cap with a 413 envelope.

    The cap is `max_bytes`, unless `route_caps` names the request's method
    and path -- resume ingestion's larger one. A streamed body that passes the
    cap mid-read raises inside `receive`. FastAPI turns any error reading the
    body into its own 400, so rather than rely on the exception reaching here,
    whatever response the app then produces is discarded and the 413 sent in
    its place. `/health` is never capped.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = DEFAULT_MAX_BODY_BYTES,
        route_caps: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("API_MAX_BODY_BYTES must be positive")
        caps = dict(route_caps or {})
        if any(cap <= 0 for cap in caps.values()):
            raise ValueError("a per-route body cap must be positive")
        self.app = app
        self.max_bytes = max_bytes
        self.route_caps = {(method.upper(), path): cap for (method, path), cap in caps.items()}

    def cap_for(self, scope: Scope) -> int:
        """The body cap for this request's method and path."""
        return self.route_caps.get(
            (str(scope.get("method", "")).upper(), scope.get("path", "")), self.max_bytes
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in LIMIT_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        cap = self.cap_for(scope)

        async def refuse(send_to: Send) -> None:
            record_limit_hit(actor_id=None, route=_route(scope), limit="body_size")
            for message in _refusal(PayloadTooLarge(details={"max_bytes": cap})):
                await send_to(message)

        if self._declares_too_much(scope, cap):
            await refuse(send)
            return

        guard = _BodyGuard(cap, receive, send, refuse)
        try:
            await self.app(scope, guard.receive, guard.send)
        except Exception:
            if not guard.exceeded:
                raise
        await guard.finish()

    @staticmethod
    def _declares_too_much(scope: Scope, cap: int) -> bool:
        declared = Headers(scope=scope).get("content-length")
        if declared is None:
            return False
        try:
            return int(declared) > cap
        except ValueError:
            return False  # The server rejects a malformed length itself.


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


# --- JSON shape --------------------------------------------------------------------

#: Defaults for `JsonLimitsMiddleware`.
DEFAULT_MAX_JSON_DEPTH = 32
DEFAULT_MAX_JSON_STRING_BYTES = 64 * 1024

#: One JSON string token, escapes included -- or, unterminated, everything to
#: the end of the body. Because a match that starts can never fail (it ends at
#: a closing quote, the end, or a lone trailing backslash), the scan is linear
#: whatever the input: no quote is ever re-tried as the start of a string.
_JSON_STRING = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*(?:"|\\?\Z)', re.DOTALL)
#: Every byte that is not a bracket, for `bytes.translate` to delete.
_NOT_BRACKETS = bytes(b for b in range(256) if b not in b"[]{}")
#: +1 for an opening bracket, -1 for a closing one, by byte value.
_DEPTH_DELTA = [0] * 256
for _open in b"[{":
    _DEPTH_DELTA[_open] = 1
for _close in b"]}":
    _DEPTH_DELTA[_close] = -1


@dataclass(frozen=True)
class JsonShapeViolation:
    """Why a JSON body was refused before parsing."""

    limit: str
    message: str


def check_json_shape(
    body: bytes, *, max_depth: int, max_string_bytes: int
) -> JsonShapeViolation | None:
    """Refuse a body that is not UTF-8, nests too deep, or has too long a string.

    A linear scan of the raw bytes, done before any parser sees them, so a
    body built to exhaust the parser's recursion or allocate a huge string
    costs one pass over at most the body cap. Strings are blanked out first
    so a bracket inside one does not count as nesting.
    """
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        return JsonShapeViolation("json_encoding", "request body is not valid UTF-8 JSON")

    longest = 0

    def blank(match: re.Match[bytes]) -> bytes:
        nonlocal longest
        longest = max(longest, match.end() - match.start() - 2)
        return b'""'

    structure = _JSON_STRING.sub(blank, body)
    if longest > max_string_bytes:
        return JsonShapeViolation(
            "json_string_length",
            f"request JSON has a string of {longest} bytes; at most {max_string_bytes} are allowed",
        )
    brackets = structure.translate(None, _NOT_BRACKETS)
    depth = max(accumulate(map(_DEPTH_DELTA.__getitem__, brackets)), default=0)
    if depth > max_depth:
        return JsonShapeViolation(
            "json_depth",
            f"request JSON nests {depth} levels deep; at most {max_depth} are allowed",
        )
    return None


def _is_json(scope: Scope) -> bool:
    content_type = Headers(scope=scope).get("content-type", "")
    media = content_type.split(";", 1)[0].strip().lower()
    return media == "application/json" or (
        media.startswith("application/") and media.endswith("+json")
    )


class JsonLimitsMiddleware:
    """A 422 for a JSON body that is not UTF-8, nests too deep or holds too long a string.

    Sits inside `BodySizeLimitMiddleware`, so the body it buffers is already
    bounded by the cap. Only a body sent as JSON is examined; anything else
    passes untouched (and no route parses it as JSON). `/health` is exempt.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_depth: int = DEFAULT_MAX_JSON_DEPTH,
        max_string_bytes: int = DEFAULT_MAX_JSON_STRING_BYTES,
    ) -> None:
        if max_depth <= 0 or max_string_bytes <= 0:
            raise ValueError("API_MAX_JSON_DEPTH and API_MAX_JSON_STRING_BYTES must be positive")
        self.app = app
        self.max_depth = max_depth
        self.max_string_bytes = max_string_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") in LIMIT_EXEMPT_PATHS
            or scope.get("method") in ("GET", "HEAD", "OPTIONS")
            or not _is_json(scope)
        ):
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # The client went away mid-body; let the app see the disconnect.
                await self.app(scope, _replaying(b"".join(chunks), receive, message), send)
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)

        violation = check_json_shape(
            body, max_depth=self.max_depth, max_string_bytes=self.max_string_bytes
        )
        if violation is not None:
            record_limit_hit(actor_id=None, route=_route(scope), limit=violation.limit)
            error = ValidationFailed(violation.message, details={"limit": violation.limit})
            for out in _refusal(error):
                await send(out)
            return
        await self.app(scope, _replaying(body, receive), send)


def _replaying(body: bytes, receive: Receive, pending: Message | None = None) -> Receive:
    """A `receive` that hands over `body` once, then defers to the real one."""
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        if pending is not None:
            return pending
        return await receive()

    return replay


__all__ = [
    "API_CONTENT_SECURITY_POLICY",
    "CORS_ALLOWED_HEADERS",
    "CORS_ALLOWED_METHODS",
    "CORS_EXPOSED_HEADERS",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_JSON_DEPTH",
    "DEFAULT_MAX_JSON_STRING_BYTES",
    "LIMIT_EXEMPT_PATHS",
    "DOCS_CONTENT_SECURITY_POLICY",
    "SECURITY_HEADERS",
    "BodySizeLimitMiddleware",
    "JsonLimitsMiddleware",
    "JsonShapeViolation",
    "SecurityHeadersMiddleware",
    "StrictCORSMiddleware",
    "apply_security_headers",
    "check_json_shape",
    "validate_cors_origins",
]
