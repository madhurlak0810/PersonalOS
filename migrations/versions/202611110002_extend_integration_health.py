"""Extend integration_health with scheduled checks: down, last success, failure streaks.

Until now a row was written only when a provider refused permanently. The
worker now checks every adapter on a schedule and records each outcome, so
the table gains:

- `down` in `integration_health_status`: transient failures several times in
  a row, with the circuit open until a check succeeds;
- `reconnect_required`: the failure was permanent, and only a reconnect
  fixes it;
- `consecutive_failures`, `last_success_at`, `last_checked_at`.

Existing degraded rows were all permanent refusals, so they are backfilled
with `reconnect_required` set.

The enum is rebuilt rather than extended with `ALTER TYPE ... ADD VALUE`, for
the reason given in `202610010002`.

Revision ID: 202611110002
Revises: 202611110001
Create Date: 2026-11-11

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "202611110002"
down_revision = "202611110001"
branch_labels = None
depends_on = None

_TYPE = "integration_health_status"
_OLD = ("healthy", "degraded")
_NEW = ("healthy", "degraded", "down")


def _rebuild(values: tuple[str, ...]) -> None:
    """Replace the Postgres enum behind `integration_health.status` with `values`."""
    if op.get_bind().dialect.name != "postgresql":
        # Other dialects store the type as plain text; there is no type to alter.
        return
    labels = ", ".join(f"'{value}'" for value in values)
    # The column default names the old type, so it is dropped across the swap.
    op.execute("ALTER TABLE integration_health ALTER COLUMN status DROP DEFAULT")
    op.execute(f"ALTER TYPE {_TYPE} RENAME TO {_TYPE}_old")
    op.execute(f"CREATE TYPE {_TYPE} AS ENUM ({labels})")
    op.execute(
        f"ALTER TABLE integration_health ALTER COLUMN status TYPE {_TYPE} "
        f"USING status::text::{_TYPE}"
    )
    op.execute(f"DROP TYPE {_TYPE}_old")
    op.execute("ALTER TABLE integration_health ALTER COLUMN status SET DEFAULT 'healthy'")


def upgrade() -> None:
    _rebuild(_NEW)
    with op.batch_alter_table("integration_health") as batch:
        batch.add_column(
            sa.Column("reconnect_required", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.add_column(
            sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("last_success_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("last_checked_at", sa.DateTime(), nullable=True))
    op.execute("UPDATE integration_health SET reconnect_required = true WHERE status = 'degraded'")


def downgrade() -> None:
    # `down` has no place in the old vocabulary. A down integration was
    # failing transiently, which the old table never recorded: it was healthy
    # as far as it knew.
    op.execute(
        "UPDATE integration_health SET status = 'healthy', reason = NULL, detail = NULL, "
        "degraded_at = NULL WHERE status = 'down' OR "
        "(status = 'degraded' AND reconnect_required = false)"
    )
    with op.batch_alter_table("integration_health") as batch:
        batch.drop_column("last_checked_at")
        batch.drop_column("last_success_at")
        batch.drop_column("consecutive_failures")
        batch.drop_column("reconnect_required")
    _rebuild(_OLD)
