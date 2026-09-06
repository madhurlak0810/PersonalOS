"""FastAPI application for PersonalOS API."""

import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from apps.api.errors import register_error_handlers
from personalos.bootstrap import initialize_mcp_servers
from personalos.config import settings
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
    """

    async def dispatch(self, request: Request, call_next):
        supplied = request.headers.get(CORRELATION_ID_HEADER)
        try:
            correlation_id = UUID(supplied) if supplied else uuid4()
        except ValueError:
            correlation_id = uuid4()
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = str(correlation_id)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting PersonalOS API...")
    init_db()
    logger.info("Database initialized")
    initialize_mcp_servers()
    logger.info("MCP servers initialized")
    yield
    # Shutdown
    logger.info("Shutting down PersonalOS API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="PersonalOS - Local-first agentic assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Assigns request.state.correlation_id for every route to see. Added
    # before CORS so CORS ends up outermost (Starlette runs the most
    # recently added middleware first) and still wraps every response,
    # including ones this middleware's own error path might produce.
    app.add_middleware(CorrelationIdMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from apps.api.routes import jobs

    app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])

    register_error_handlers(app)

    return app


app = create_app()
