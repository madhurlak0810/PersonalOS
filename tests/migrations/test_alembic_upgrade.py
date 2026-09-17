"""Runs the actual Alembic migration chain against Postgres.

Every other test in this repo builds its schema with
`Base.metadata.create_all()` against SQLite (see e.g.
`tests/unit/test_pgvector_embedding_tables.py`), which never executes a
single line of `migrations/versions/*.py`. These tests are the only place
that does: they drive `alembic upgrade head` (and `downgrade`) against a
disposable Postgres database, the same way a real deployment would, so a
migration that is broken, out of order, or not idempotent fails here instead
of in production.

Skipped automatically when no Postgres is reachable at
`personalos.config.settings.database_url` (e.g. running `pytest` locally
without a database) -- CI provides one as a service container, see
`.github/workflows/ci.yml`.
"""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

from personalos.config import settings

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = {
    "users",
    "workflows",
    "workflow_runs",
    "workflow_steps",
    "checkpoints",
    "approvals",
    "job_postings",
    "candidate_profiles",
    "applications",
    "artifact_versions",
    "communication_events",
    "tool_executions",
    "policy_decisions",
    "audit_events",
    "outbox_events",
    "event_log",
    "application_status_view",
    "evidence_chunks",
    "job_posting_embeddings",
    "message_embeddings",
}


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return cfg


@pytest.fixture
def empty_database():
    """Reset `settings.database_url` to a blank `public` schema.

    Skips rather than fails when no Postgres is reachable, so the rest of
    the suite still runs on a machine without Docker.
    """
    engine = create_engine(settings.database_url)
    if engine.dialect.name != "postgresql":
        pytest.skip("Migration tests require a Postgres DATABASE_URL")
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip(f"No reachable Postgres at {settings.database_url!r}")

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    yield engine
    engine.dispose()


def test_upgrade_head_from_empty_database(empty_database):
    """`alembic upgrade head` succeeds from a genuinely empty database."""
    command.upgrade(_alembic_config(), "head")

    inspector = inspect(empty_database)
    assert EXPECTED_TABLES <= set(inspector.get_table_names())

    with empty_database.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    script = ScriptDirectory.from_config(_alembic_config())
    assert version == script.get_current_head()


def test_upgrade_head_is_idempotent(empty_database):
    """Re-running `upgrade head` against a database already at head is a no-op."""
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.upgrade(cfg, "head")  # must not raise (e.g. "relation already exists")

    inspector = inspect(empty_database)
    assert EXPECTED_TABLES <= set(inspector.get_table_names())


def test_downgrade_and_reupgrade_round_trips(empty_database):
    """Every migration's `downgrade()` reverses its `upgrade()` cleanly.

    Exercises the one code path nothing else in the suite touches: the
    per-revision `downgrade()` functions, including the Postgres enum-type
    cleanup each does explicitly (see e.g. `202609090001`'s downgrade).
    """
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    inspector = inspect(empty_database)
    assert not (EXPECTED_TABLES & set(inspector.get_table_names()))

    command.upgrade(cfg, "head")
    inspector = inspect(empty_database)
    assert EXPECTED_TABLES <= set(inspector.get_table_names())
