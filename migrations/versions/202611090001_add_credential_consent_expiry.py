"""Add credentials.consent_expires_at: when a stored OAuth grant stops working.

Google gives an unverified app in "Testing" publishing status 7-day refresh
tokens, so a connection made through one silently dies a week later. Recording
when lets `personalos.cli auth google --status` and `GET /v1/integrations` say
so in advance. Null when the grant does not expire on a clock.

A timestamp, not a credential: the table still has no column a token could go in.

Revision ID: 202611090001
Revises: 202611040001
Create Date: 2026-11-09

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "202611090001"
down_revision = "202611040001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("credentials", sa.Column("consent_expires_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("credentials", "consent_expires_at")
