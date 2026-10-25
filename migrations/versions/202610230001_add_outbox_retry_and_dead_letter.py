"""Add outbox retry backoff, dead-lettering and quarantine.

`outbox_events.status` gains `dead_letter` (a row that used up its retries)
and `quarantined` (a row whose payload no delivery could succeed with). Four
columns carry what retrying and surfacing them needs:

- `available_at`: not claimable before then -- the backoff after a failed
  delivery, instead of the row sitting claimed until its lease runs out;
- `last_error`: the last delivery failure, redacted;
- `failed_at`: when the row was dead-lettered or quarantined;
- `workflow_id`: the workflow a dead-lettered event belongs to, so the
  workflow's failures view can list it. Indexed; no foreign key.

The enum is rebuilt rather than extended with `ALTER TYPE ... ADD VALUE`, for
the reason given in `202610010002`.

Revision ID: 202610230001
Revises: 202610220001
Create Date: 2026-10-23

"""

import sqlalchemy as sa
from alembic import op

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202610230001"
down_revision = "202610220001"
branch_labels = None
depends_on = None

_STATUS_TYPE = "outbox_event_status"
_OLD_STATUSES = ("pending", "in_progress", "dispatched", "failed")
_NEW_STATUSES = (*_OLD_STATUSES, "dead_letter", "quarantined")


def _rebuild_status_type(values: tuple[str, ...]) -> None:
    """Replace the Postgres enum behind `outbox_events.status` with `values`."""
    if op.get_bind().dialect.name != "postgresql":
        # Other dialects store the status as plain text; there is no type to alter.
        return
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"ALTER TYPE {_STATUS_TYPE} RENAME TO {_STATUS_TYPE}_old")
    op.execute(f"CREATE TYPE {_STATUS_TYPE} AS ENUM ({labels})")
    op.execute("ALTER TABLE outbox_events ALTER COLUMN status DROP DEFAULT")
    op.execute(
        f"ALTER TABLE outbox_events ALTER COLUMN status TYPE {_STATUS_TYPE} "
        f"USING status::text::{_STATUS_TYPE}"
    )
    op.execute("ALTER TABLE outbox_events ALTER COLUMN status SET DEFAULT 'pending'")
    op.execute(f"DROP TYPE {_STATUS_TYPE}_old")


def upgrade() -> None:
    _rebuild_status_type(_NEW_STATUSES)
    op.add_column("outbox_events", sa.Column("available_at", sa.DateTime(), nullable=True))
    op.add_column("outbox_events", sa.Column("last_error", sa.Text(), nullable=True))
    op.add_column("outbox_events", sa.Column("failed_at", sa.DateTime(), nullable=True))
    op.add_column("outbox_events", sa.Column("workflow_id", GUID(), nullable=True))
    op.create_index("ix_outbox_events_workflow_id", "outbox_events", ["workflow_id"])


def downgrade() -> None:
    # Both new states were "given up on"; before them, that was `failed`.
    op.execute(
        "UPDATE outbox_events SET status = 'failed' WHERE status IN ('dead_letter', 'quarantined')"
    )
    op.drop_index("ix_outbox_events_workflow_id", table_name="outbox_events")
    op.drop_column("outbox_events", "workflow_id")
    op.drop_column("outbox_events", "failed_at")
    op.drop_column("outbox_events", "last_error")
    op.drop_column("outbox_events", "available_at")
    _rebuild_status_type(_OLD_STATUSES)
