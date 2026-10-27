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
@click.option(
    "--graphs",
    default=None,
    help="Graph factory as module:callable (default: WORKER_GRAPHS).",
)
def worker(graphs: str | None):
    """Run the worker: queued workflow runs, outbox delivery and scheduled sweeps."""
    from apps.worker.main import main

    logger.info("Starting PersonalOS worker...")
    main(graphs)


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


@cli.command()
@click.argument("workflow_id", type=click.UUID)
@click.option("--json", "as_json", is_flag=True, help="Print the API response body instead.")
def workflow_status(workflow_id, as_json: bool):
    """Show a workflow: plan, what was done, what is only suggested, what is waiting."""
    from apps.api.routes.workflows import WorkflowStatusResponse, get_workflow_services
    from personalos.cli_views import render_workflow

    services = get_workflow_services()
    snapshot = services.reader.read(workflow_id)
    if snapshot is None:
        raise click.ClickException(f"workflow '{workflow_id}' not found")
    view = WorkflowStatusResponse.from_snapshot(snapshot, services.actions.read(workflow_id))
    click.echo(view.model_dump_json(indent=2) if as_json else render_workflow(view))


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Print the API response body instead.")
def approvals(as_json: bool):
    """List every action waiting for approval, with what approving it would do."""
    from apps.api.routes.approvals import PendingApprovalItem, PendingApprovalsResponse
    from apps.api.routes.workflows import get_workflow_services
    from personalos.cli_views import render_approvals

    items = [
        PendingApprovalItem.from_pending(approval)
        for approval in get_workflow_services().approvals.list_pending()
    ]
    if as_json:
        click.echo(PendingApprovalsResponse(approvals=items).model_dump_json(indent=2))
    else:
        click.echo(render_approvals(items))


if __name__ == "__main__":
    cli()
