"""Add integration_health: which integrations need reconnecting.

A provider that refuses permanently -- revoked OAuth, a missing scope -- is not
fixed by retrying, so the workflow that hit it records the integration as
degraded here and the failure it left offers a reconnect instead of a retry.
One row per provider; a provider with no row has never failed that way.

Revision ID: 202610200001
Revises: 202610070002
Create Date: 2026-10-20

"""

import sqlalchemy as sa
from alembic import op

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202610200001"
down_revision = "202610070002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "integration_health",
        sa.Column("provider", sa.String(length=63), primary_key=True),
        sa.Column(
            "status",
            sa.Enum("healthy", "degraded", name="integration_health_status"),
            nullable=False,
            server_default="healthy",
        ),
        sa.Column("reason", sa.String(length=63), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("degraded_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("integration_health")
    bind = op.get_bind()
    sa.Enum(name="integration_health_status").drop(bind, checkfirst=True)
