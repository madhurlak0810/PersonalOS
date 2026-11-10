"""Add mailbox_cursors: how far the Gmail inbox poller has read.

One row per provider and account: the Gmail history id every change at or
below which has been considered, and the fingerprint of the scope rules
(labels and queries) it was read under, so a change of rules triggers a
resync. A position, not content: no column here holds anything from a message.

Revision ID: 202611100001
Revises: 202611090001
Create Date: 2026-11-10

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "202611100001"
down_revision = "202611090001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mailbox_cursors",
        sa.Column("provider", sa.String(length=63), primary_key=True),
        sa.Column("account", sa.String(length=320), primary_key=True),
        sa.Column("history_id", sa.String(length=20), nullable=False),
        sa.Column("scope_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("polled_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("mailbox_cursors")
