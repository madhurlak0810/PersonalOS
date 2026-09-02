"""FastAPI application for PersonalOS API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from personalos.bootstrap import initialize_mcp_servers
from personalos.config import settings
from personalos.persistence import init_db

logger = logging.getLogger(__name__)


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

    return app


app = create_app()
