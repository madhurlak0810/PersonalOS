"""CLI entry points for PersonalOS."""

import logging

import click
import uvicorn

from personalos.config import settings
from personalos.observability.structured_logging import configure_logging_from_settings
from personalos.persistence import init_db

configure_logging_from_settings()
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
        # Leave uvicorn's loggers handler-less, so they propagate to the root
        # handler the app installs and come out structured like everything else.
        log_config=None,
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


# --- Search profile ---------------------------------------------------------------
#
# The same shapes as `/v1/profile`: a create or update reads a
# `CandidateProfileInput` as JSON, and every read prints the API's response
# body, so `profile show`'s `profile` can be fed back to `profile update`.
# Scoped by `--actor` the way the API is scoped by `X-Actor-Id`, and just as
# unauthenticated until Phase L.

_actor_option = click.option(
    "--actor",
    "actor_id",
    default="api",
    show_default=True,
    help="Whose profile: the same actor the API reads from X-Actor-Id.",
)


def _read_profile_input(source):
    from pydantic import ValidationError

    from personalos.domain.search_profiles import CandidateProfileInput

    try:
        return CandidateProfileInput.model_validate_json(source.read())
    except ValidationError as exc:
        raise click.ClickException(f"invalid profile:\n{exc}") from exc


def _profile_call(call):
    """Run a store call, turning its typed refusals into CLI errors."""
    from personalos.domain.errors import PersonalOSError

    try:
        return call()
    except PersonalOSError as exc:
        raise click.ClickException(exc.message) from exc


@cli.group()
def profile():
    """Create, show and update your job-search profile. Updates add a version."""


@profile.command("create")
@click.argument("source", type=click.File("r"), default="-")
@_actor_option
def profile_create(source, actor_id: str):
    """Write version 1 of the profile, from a JSON file (or stdin)."""
    from apps.api.routes.profiles import get_profile_store
    from personalos.domain.search_profiles import actor_user_id

    body = _read_profile_input(source)
    created = _profile_call(
        lambda: get_profile_store().create(actor_user_id(actor_id), body, display_name=actor_id)
    )
    click.echo(created.model_dump_json(indent=2))


@profile.command("update")
@click.argument("source", type=click.File("r"), default="-")
@_actor_option
def profile_update(source, actor_id: str):
    """Write the next version of the profile, from a JSON file (or stdin)."""
    from apps.api.routes.profiles import get_profile_store
    from personalos.domain.search_profiles import actor_user_id

    body = _read_profile_input(source)
    updated = _profile_call(lambda: get_profile_store().update(actor_user_id(actor_id), body))
    click.echo(updated.model_dump_json(indent=2))


@profile.command("show")
@click.option("--version", "profile_version", type=click.IntRange(min=1), default=None)
@_actor_option
def profile_show(profile_version: int | None, actor_id: str):
    """Print the latest profile version, or the one named by --version."""
    from apps.api.routes.profiles import get_profile_store
    from personalos.domain.search_profiles import actor_user_id

    shown = _profile_call(lambda: get_profile_store().get(actor_user_id(actor_id), profile_version))
    click.echo(shown.model_dump_json(indent=2))


@profile.command("versions")
@_actor_option
def profile_versions(actor_id: str):
    """Print every version of the profile, newest first."""
    from apps.api.routes.profiles import get_profile_store
    from personalos.domain.search_profiles import actor_user_id

    click.echo(get_profile_store().versions(actor_user_id(actor_id)).model_dump_json(indent=2))


# --- Evidence ---------------------------------------------------------------------


@cli.command("ingest-resume")
@click.argument("path")
@click.option(
    "--actor",
    "actor_id",
    default="api",
    show_default=True,
    help="Whose evidence: the same actor the API reads from X-Actor-Id.",
)
def ingest_resume(path: str, actor_id: str):
    """Load the resume at PATH (PDF, DOCX, Markdown or text) into the evidence index.

    PATH is read through the sandboxed files tool, so it must be inside
    FILES_ALLOWED_ROOTS; a relative PATH is resolved against the first root,
    not the current directory. Re-running on an unchanged file changes nothing.
    Prints the same body as `POST /v1/evidence/documents`.
    """
    import asyncio

    from personalos.bootstrap import build_resume_ingestor, initialize_mcp_servers
    from personalos.domain.search_profiles import actor_user_id

    init_db()
    initialize_mcp_servers()
    ingestor = _profile_call(lambda: build_resume_ingestor(requested_by="cli:ingest_resume"))
    result = _profile_call(
        lambda: asyncio.run(ingestor.ingest(actor_user_id(actor_id), path, display_name=actor_id))
    )
    click.echo(result.model_dump_json(indent=2))


# --- Integrations -----------------------------------------------------------------
#
# `auth google` is the only way a Google refresh token is ever obtained: it
# runs the consent on a loopback listener and files the token in the OS
# keychain. The database, the logs and the output hold only its `cred://`
# reference.


@cli.command("mcp-google")
def mcp_google():
    """Serve the Gmail and Calendar MCP server over stdin/stdout.

    For a parent process to spawn; it opens no port.
    """
    from personalos.bootstrap import serve_google_mcp_stdio

    serve_google_mcp_stdio()


@cli.group()
def auth():
    """Connect, inspect and disconnect integrations."""


@auth.command("google")
@click.option("--status", "show_status", is_flag=True, help="Show the connection; change nothing.")
@click.option("--revoke", is_flag=True, help="Revoke at Google and delete the keychain entry.")
@click.option(
    "--enable",
    "enable",
    multiple=True,
    metavar="CAPABILITY",
    help=(
        "Ask for a write capability on top of the existing read-only connection: "
        "gmail.compose, gmail.send or calendar.write. Repeatable."
    ),
)
@click.option("--account", default=None, help="With --revoke: disconnect only this account.")
@click.option("--no-browser", is_flag=True, help="Print the consent URL instead of opening it.")
@click.option("--port", default=0, type=click.IntRange(0, 65535), help="Loopback port (0: any).")
@click.option("--timeout", default=300, type=click.IntRange(min=10), help="Seconds to wait.")
@click.option("--json", "as_json", is_flag=True, help="Print the API response body instead.")
def auth_google(
    show_status: bool,
    revoke: bool,
    enable: tuple[str, ...],
    account: str | None,
    no_browser: bool,
    port: int,
    timeout: int,
    as_json: bool,
):
    """Connect a Google account (read-only by default), show it, or disconnect it.

    With no options, opens Google's consent page for Gmail and Calendar
    read-only access, catches the redirect on 127.0.0.1, and stores the refresh
    token in the OS keychain. Write access (`--enable gmail.send`, ...) is a
    separate consent, asked for only once a read-only connection exists.
    """
    import asyncio
    import webbrowser

    from personalos.bootstrap import build_google_connection_reader, build_google_connector
    from personalos.cli_views import render_integration, render_revocations
    from personalos.domain.integrations import parse_capabilities
    from personalos.executor.google_connect import run_loopback_consent

    if sum((show_status, revoke, bool(enable))) > 1:
        raise click.UsageError("use one of --status, --revoke or --enable at a time")
    if account and not revoke:
        raise click.UsageError("--account applies only to --revoke")
    try:
        capabilities = parse_capabilities(enable)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--enable") from exc

    if show_status:
        status = build_google_connection_reader().status()
        click.echo(status.model_dump_json(indent=2) if as_json else render_integration(status))
        return

    connector = _profile_call(build_google_connector)

    if revoke:
        results = _profile_call(lambda: asyncio.run(connector.revoke(account)))
        if as_json:
            from apps.api.routes.integrations import GoogleRevokeResponse

            click.echo(GoogleRevokeResponse(results=results).model_dump_json(indent=2))
        else:
            click.echo(render_revocations(results))
        return

    def open_browser(url: str) -> None:
        click.echo("Grant access in your browser. If it did not open, visit:\n", err=True)
        click.echo(f"  {url}\n", err=True)
        if not no_browser:
            webbrowser.open(url)

    _profile_call(
        lambda: asyncio.run(
            run_loopback_consent(
                connector,
                enable=capabilities,
                open_browser=open_browser,
                timeout=timeout,
                port=port,
            )
        )
    )
    status = connector.status()
    click.echo(status.model_dump_json(indent=2) if as_json else render_integration(status))


@auth.command("google-client")
@click.option(
    "--from-json",
    "client_json",
    type=click.File("r"),
    default=None,
    help="The client_secret_*.json downloaded from the Google Cloud console.",
)
def auth_google_client(client_json):
    """Store this deployment's Google OAuth client secret in the OS keychain.

    Create a "Desktop app" OAuth client in the Google Cloud console, then run
    this with its downloaded JSON (or paste the secret when prompted). The
    client id is not secret: put it in `.env` as GOOGLE_OAUTH_CLIENT_ID.
    """
    import json

    from personalos.bootstrap import store_google_oauth_client_secret
    from personalos.domain.credentials import SecretValue

    client_id = None
    if client_json is not None:
        try:
            document = json.load(client_json)
            client = document.get("installed") or document.get("web") or {}
            client_id, secret = client["client_id"], client["client_secret"]
        except (ValueError, KeyError, AttributeError) as exc:
            raise click.ClickException(
                "not a Google OAuth client JSON (expected installed.client_id and client_secret)"
            ) from exc
    else:
        secret = click.prompt("Client secret", hide_input=True)

    ref = _profile_call(lambda: store_google_oauth_client_secret(SecretValue(secret)))
    click.echo(f"Stored the OAuth client secret in the OS keychain at {ref}.")
    configured = settings.google_oauth_client_id
    if client_id and configured != client_id:
        click.echo(f"Now set GOOGLE_OAUTH_CLIENT_ID={client_id} in .env.")
    elif not configured:
        click.echo("Now set GOOGLE_OAUTH_CLIENT_ID in .env to the client's id.")


if __name__ == "__main__":
    cli()
