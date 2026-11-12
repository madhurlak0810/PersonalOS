"""FastAPI application for PersonalOS API."""

import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Request
from opentelemetry import baggage
from starlette.middleware.base import BaseHTTPMiddleware

from apps.api.auth import (
    build_api_auth,
    ensure_safe_bind,
    require_principal,
    require_stream_principal,
)
from apps.api.errors import register_error_handlers
from apps.api.security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    StrictCORSMiddleware,
    validate_cors_origins,
)
from personalos.bootstrap import initialize_mcp_servers
from personalos.config import settings
from personalos.observability import tracing
from personalos.observability.structured_logging import (
    configure_logging_from_settings,
    log_scope,
)
from personalos.observability.tracing_setup import configure_tracing_from_settings
from personalos.persistence import init_db

logger = logging.getLogger(__name__)

#: Header a caller may supply to correlate their own logs with ours, and that
#: we echo back so a fresh id we minted is visible to them too.
CORRELATION_ID_HEADER = "X-Correlation-Id"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Assigns every request a correlation id, the seed of its execution context.

    Stashed on `request.state.correlation_id` so a route handler can fold it
    into the `ExecutionContext` of whatever it creates (a job, a workflow run),
    tying the two together from the very first hop. A malformed caller-supplied
    id is replaced rather than rejected: a bad correlation id should not be why
    a request fails.

    It is also the request's tracing context: a caller's W3C `traceparent` is
    extracted, so the workflow span a route opens links back to the caller's
    trace, and the correlation id goes into baggage, so every span the request
    produces carries it -- and the request's log scope, so every log line
    written while handling it carries it too.
    """

    async def dispatch(self, request: Request, call_next):
        supplied = request.headers.get(CORRELATION_ID_HEADER)
        try:
            correlation_id = UUID(supplied) if supplied else uuid4()
        except ValueError:
            correlation_id = uuid4()
        request.state.correlation_id = correlation_id

        ctx = baggage.set_baggage(
            tracing.Attr.CORRELATION_ID,
            str(correlation_id),
            tracing.extract_trace_context(request.headers),
        )
        with tracing.attached(ctx), log_scope(correlation_id=correlation_id):
            response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = str(correlation_id)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting PersonalOS API...")
    # Refuses API_AUTH_ENABLED=false on a non-loopback API_HOST. The CLI checks
    # its own --host before this; this covers `uvicorn apps.api.main:app`.
    ensure_safe_bind(settings.api_host, auth_enabled=settings.api_auth_enabled)
    if getattr(app.state, "api_auth", None) is None:
        app.state.api_auth = build_api_auth()
    ensure_token = getattr(app.state.api_auth.verifier, "ensure_token", None)
    if app.state.api_auth.enabled and ensure_token is not None and ensure_token():
        logger.info("Created the API token; print it with `python -m personalos.cli auth token`")
    configure_tracing_from_settings("personalos-api")
    init_db()
    logger.info("Database initialized")
    initialize_mcp_servers()
    logger.info("MCP servers initialized")
    yield
    # Shutdown
    logger.info("Shutting down PersonalOS API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    configure_logging_from_settings()
    app = FastAPI(
        title=settings.app_name,
        description="PersonalOS - Local-first agentic assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Starlette runs the most recently added middleware first, so these are
    # listed innermost to outermost (see `apps.api.security`):
    #
    # - the body cap, innermost, so its 413 still gets a correlation id;
    # - the correlation id, assigned for every route to see;
    # - CORS, restricted to `API_CORS_ORIGINS`, wrapping every response so an
    #   allowed UI can read errors too;
    # - the security headers, outermost, so preflights and CORS refusals
    #   carry them as well.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.api_max_body_bytes)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        StrictCORSMiddleware,
        # Checked here, not on first request: a bad entry fails startup.
        allow_origins=validate_cors_origins(settings.api_cors_origins),
    )
    app.add_middleware(SecurityHeadersMiddleware)

    # Include routers. Every router but health and the OAuth callback is
    # mounted behind `require_principal`; `tests/unit/test_api_auth.py` checks every
    # registered route, so a router added here without it fails the suite.
    from apps.api.routes import (
        applications,
        approvals,
        audit,
        auth,
        events,
        evidence,
        failures,
        health,
        integrations,
        profiles,
        workflows,
    )

    app.include_router(health.router, tags=["health"])
    # Google's OAuth redirect: the browser has no bearer token, so the
    # one-time `state` an authenticated `/connect` minted is its credential.
    app.include_router(integrations.callback_router, prefix="/v1", tags=["integrations"])

    authenticated = [Depends(require_principal)]
    for router, tag in (
        (auth.router, "auth"),
        (workflows.router, "workflows"),
        (approvals.router, "approvals"),
        (applications.router, "applications"),
        (audit.router, "audit"),
        (failures.router, "failures"),
        (profiles.router, "profiles"),
        (evidence.router, "evidence"),
        (integrations.router, "integrations"),
    ):
        app.include_router(router, prefix="/v1", tags=[tag], dependencies=authenticated)

    # The event stream also takes `?ticket=`, for a browser EventSource that
    # cannot send Authorization; see `apps.api.auth`.
    app.include_router(
        events.router,
        prefix="/v1",
        tags=["events"],
        dependencies=[Depends(require_stream_principal)],
    )

    register_error_handlers(app)

    return app


app = create_app()
