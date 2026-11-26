"""Request latency and error rate per route: `personalos.api.request.duration`.

`RequestMetricsMiddleware` is plain ASGI and the outermost layer of the app,
so the time it records covers every other middleware -- the security
headers, CORS, the body cap, authentication -- and the status it records is
the one the client got, refusals included.

The `route` label is the **template** the request matched
(`/v1/workflows/{workflow_id}`), read from the route FastAPI puts on the scope
once routing is done -- never the path, which carries ids, and never the query
string, where a stream ticket travels. A request that matched no route (a
404, a refusal made before routing) is labelled `{unmatched}`, so a scan of
random paths cannot grow the metric's cardinality. `status_class` is `2xx`,
`4xx`, `5xx` and so on; the error-rate panels divide the `5xx` share by the
total.
"""

from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from personalos.observability import metrics
from personalos.observability.metrics import Metric

#: The `route` label of a request no route matched.
UNMATCHED_ROUTE = "{unmatched}"

#: Methods recorded as themselves; anything else is `other`.
_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def route_template(scope: Scope) -> str:
    """The full template of the route `scope` matched (`/v1/chat`), or `{unmatched}`.

    The route FastAPI leaves on the scope is the one inside its router, whose
    template lacks the prefix the router was included under (`/chat`). The
    prefix is static, so it is the request path's leading segments, as many
    as the path has beyond the template's. A template with a `:path`
    parameter, which can span segments, is reported without a prefix.
    """
    path = getattr(scope.get("route"), "path", None)
    if not isinstance(path, str) or not path:
        return UNMATCHED_ROUTE
    if ":path}" in path:
        return path
    template = path.rstrip("/").split("/")
    request_path = str(scope.get("path") or "").rstrip("/").split("/")
    extra = len(request_path) - len(template)
    if extra <= 0:
        return path
    return "/".join(request_path[: extra + 1]) + path


class RequestMetricsMiddleware:
    """Times every HTTP request into `personalos.api.request.duration`."""

    def __init__(self, app: ASGIApp):
        """Wrap the rest of the application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status: dict[str, Any] = {"code": 500}

        async def send_and_note(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = int(message["status"])
            await send(message)

        timer = metrics.start_timer()
        try:
            await self.app(scope, receive, send_and_note)
        finally:
            method = scope.get("method", "")
            metrics.observe(
                Metric.API_REQUEST_DURATION,
                timer.elapsed(),
                route=route_template(scope),
                method=method if method in _METHODS else metrics.OTHER,
                status_class=f"{status['code'] // 100}xx",
            )


__all__ = ["RequestMetricsMiddleware", "UNMATCHED_ROUTE", "route_template"]
