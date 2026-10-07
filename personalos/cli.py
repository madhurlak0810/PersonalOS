"""CLI entry points for PersonalOS."""

import logging

import click
import uvicorn

from personalos.config import settings
from personalos.persistence import init_db

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """PersonalOS CLI."""
    pass


@cli.command()
@click.option("--host", default=settings.api_host, help="API host")
@click.option("--port", default=settings.api_port, help="API port")
@click.option("--workers", default=settings.api_workers, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def api(host: str, port: int, workers: int, reload: bool):
    """Run the PersonalOS API server."""
    logger.info(f"Starting PersonalOS API on {host}:{port}")
    init_db()

    uvicorn.run(
        "apps.api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
    )


@cli.command()
def db_init():
    """Initialize database."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")


@cli.command()
def worker():
    """Run the background worker."""
    logger.info("Starting PersonalOS worker...")
    # Placeholder for worker implementation
    logger.info("Worker not yet implemented")


@cli.command()
@click.option("--once", is_flag=True, help="Run a single sweep and exit (for cron).")
def stall_monitor(once: bool):
    """Move applications with no activity past the stall window to STALLED."""
    import asyncio

    from apps.worker.stall_monitor import StalledApplicationMonitor
    from personalos.bootstrap import build_application_lifecycle_store

    monitor = StalledApplicationMonitor(store=build_application_lifecycle_store())
    if once:
        report = monitor.sweep()
        logger.info("Stall sweep: %s stalled, %s skipped", len(report.stalled), len(report.skipped))
        return
    asyncio.run(monitor.run_forever())


if __name__ == "__main__":
    cli()
