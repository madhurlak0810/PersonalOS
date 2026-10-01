"""Tie tool_executions and audit_events back to what authorized them.

`tool_executions` gains the policy decision and approval each call was made
under, the fingerprint of the action its idempotency key was claimed for, an
attempt counter, and an `unknown` status for a claim that was left with no
recorded outcome and has to be reconciled against the provider before anything
is re-executed.

`audit_events` gains `policy_decision_id` and `operation_id`, so an audit row
references the exact `policy_decisions` row and execution it describes. The
existing `policy_decision` text column stays as the snapshot it was designed
to be; the new columns add the link without replacing it.

The enum is rebuilt rather than extended with `ALTER TYPE ... ADD VALUE`: a
value added that way cannot be used until the transaction that added it
commits, and Alembic runs the whole chain in one transaction.

Revision ID: 202610010002
Revises: 202610010001
Create Date: 2026-10-01

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202610010002"
down_revision = "202610010001"
branch_labels = None
depends_on = None

_STATUS_TYPE = "tool_execution_status"
_OLD_STATUSES = ("in_progress", "completed", "failed")
_NEW_STATUSES = (*_OLD_STATUSES, "unknown")


def _rebuild_status_type(values: tuple[str, ...]) -> None:
    """Replace the Postgres enum behind `tool_executions.status` with `values`."""
    if op.get_bind().dialect.name != "postgresql":
        # Other dialects store the status as plain text; there is no type to alter.
        return
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"ALTER TYPE {_STATUS_TYPE} RENAME TO {_STATUS_TYPE}_old")
    op.execute(f"CREATE TYPE {_STATUS_TYPE} AS ENUM ({labels})")
    op.execute("ALTER TABLE tool_executions ALTER COLUMN status DROP DEFAULT")
    op.execute(
        f"ALTER TABLE tool_executions ALTER COLUMN status TYPE {_STATUS_TYPE} "
        f"USING status::text::{_STATUS_TYPE}"
    )
    op.execute("ALTER TABLE tool_executions ALTER COLUMN status SET DEFAULT 'in_progress'")
    op.execute(f"DROP TYPE {_STATUS_TYPE}_old")


def upgrade() -> None:
    _rebuild_status_type(_NEW_STATUSES)

    op.add_column(
        "tool_executions", sa.Column("request_fingerprint", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "tool_executions",
        sa.Column(
            "policy_decision_id", GUID(), sa.ForeignKey("policy_decisions.id"), nullable=True
        ),
    )
    op.add_column(
        "tool_executions", sa.Column("approval_ref", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "tool_executions", sa.Column("approved_by", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "tool_executions",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"),
    )

    op.add_column(
        "audit_events",
        sa.Column(
            "policy_decision_id", GUID(), sa.ForeignKey("policy_decisions.id"), nullable=True
        ),
    )
    op.add_column(
        "audit_events",
        sa.Column(
            "operation_id", GUID(), sa.ForeignKey("tool_executions.operation_id"), nullable=True
        ),
    )
    op.add_column("audit_events", sa.Column("approval_ref", sa.String(length=255), nullable=True))
    op.create_index("ix_audit_events_operation_id", "audit_events", ["operation_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_operation_id", table_name="audit_events")
    op.drop_column("audit_events", "approval_ref")
    op.drop_column("audit_events", "operation_id")
    op.drop_column("audit_events", "policy_decision_id")

    op.drop_column("tool_executions", "attempts")
    op.drop_column("tool_executions", "approved_by")
    op.drop_column("tool_executions", "approval_ref")
    op.drop_column("tool_executions", "policy_decision_id")
    op.drop_column("tool_executions", "request_fingerprint")

    # `unknown` has no equivalent in the old vocabulary. `failed` is the
    # closest: like `unknown`, it is never retried automatically.
    op.execute("UPDATE tool_executions SET status = 'failed' WHERE status = 'unknown'")
    _rebuild_status_type(_OLD_STATUSES)
