"""Add outbox claim leases and the consumed_events inbox.

Two halves of at-least-once event dispatch:

- `outbox_events.claimed_at` / `attempts`: a claim is now a lease. A worker
  that crashes between claiming a row and marking it dispatched no longer
  strands it IN_PROGRESS forever -- once the claim is older than the
  dispatcher's visibility timeout the row is claimable again, and `attempts`
  counts how many times that has happened.
- `consumed_events`: one row per event a consumer has applied, unique on
  `(consumer, dedupe_key)` and written in the same transaction as the effect,
  so a redelivered event cannot apply twice.

Revision ID: 202610220001
Revises: 202610200001
Create Date: 2026-10-22

"""

import sqlalchemy as sa
from alembic import op

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202610220001"
down_revision = "202610200001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("outbox_events", sa.Column("claimed_at", sa.DateTime(), nullable=True))
    op.add_column(
        "outbox_events",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "consumed_events",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("consumer", sa.String(length=100), nullable=False),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("event_id", GUID(), nullable=False),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("consumed_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "consumer", "dedupe_key", name="uq_consumed_events_consumer_dedupe_key"
        ),
    )


def downgrade() -> None:
    op.drop_table("consumed_events")
    op.drop_column("outbox_events", "attempts")
    op.drop_column("outbox_events", "claimed_at")
