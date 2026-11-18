"""Record who decided an approval on the approval itself.

Until now the deciding actor was kept only in the audit trail (the
`approval.<outcome>` event's `actor`), and `approvals.approved_by_user_id`
was never written. Phase L makes the decider part of the approval record:
`approvals.decided_by` is the verified actor that approved or rejected it,
stored beside the `action_hash` it is bound to, so the record alone says who
said yes to exactly what.

Existing decisions are backfilled from their audit event.

Revision ID: 202611180001
Revises: 202611110002
Create Date: 2026-11-18

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "202611180001"
down_revision = "202611110002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("approvals") as batch:
        batch.add_column(sa.Column("decided_by", sa.String(length=255), nullable=True))
    op.execute(
        """
        UPDATE approvals SET decided_by = (
            SELECT audit_events.actor FROM audit_events
            WHERE audit_events.approval_ref = CAST(approvals.id AS VARCHAR)
              AND audit_events.action LIKE 'approval.%'
              AND audit_events.action NOT LIKE 'approval.refused.%'
            ORDER BY audit_events.timestamp ASC
            LIMIT 1
        )
        WHERE decided_by IS NULL AND status <> 'pending'
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("approvals") as batch:
        batch.drop_column("decided_by")
