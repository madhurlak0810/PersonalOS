"""Add tool_executions, policy_decisions, and audit_events tables.

The tables that make every mutating action inspectable and auditable:
tool_executions (one row per tool-call attempt, keyed by idempotency key),
policy_decisions (the verdict reached on a proposed action), and audit_events
(the append-only record of what actually happened).

`tool_executions.idempotency_key` is unique so a retried tool call returns
the stored `receipt_json` instead of re-executing, mirroring the
`operations` table added before Alembic-managed migrations existed (see
`202609090001`). `audit_events` is append-only in application code --
`personalos.persistence.repositories.AuditEventRepository` exposes only
`create` and reads, never update or delete.

Primary keys and foreign keys use `GUID`, matching prior migrations, so this
migration produces the same DDL `Base.metadata.create_all()` would.

Revision ID: 202609110001
Revises: 202609100002
Create Date: 2026-09-11

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609110001"
down_revision = "202609100002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_executions",
        sa.Column("operation_id", GUID(), primary_key=True),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False, unique=True),
        sa.Column(
            "status",
            sa.Enum("in_progress", "completed", "failed", name="tool_execution_status"),
            nullable=False,
            server_default="in_progress",
        ),
        sa.Column("receipt_json", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_tool_executions_workflow_id", "tool_executions", ["workflow_id"])

    op.create_table(
        "policy_decisions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("principal", sa.String(length=255), nullable=False),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("tool", sa.String(length=255), nullable=False),
        sa.Column("args_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "decision",
            sa.Enum("allow", "deny", "require_approval", name="policy_decision_outcome"),
            nullable=False,
        ),
        sa.Column("requested_scopes", sa.JSON(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_policy_decisions_workflow_id", "policy_decisions", ["workflow_id"])

    op.create_table(
        "audit_events",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("actor", sa.String(length=255), nullable=False),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("action", sa.String(length=255), nullable=False),
        sa.Column("target_ref", sa.String(length=500), nullable=False),
        # Snapshot of the policy verdict, stored as plain text rather than a
        # foreign key to `policy_decisions` or a shared enum type, so this row
        # stays a stable historical record even if that table's vocabulary
        # changes later.
        sa.Column("policy_decision", sa.String(length=32), nullable=True),
        sa.Column(
            "result",
            sa.Enum("success", "failure", name="audit_event_result"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_audit_events_workflow_id", "audit_events", ["workflow_id"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("policy_decisions")
    op.drop_table("tool_executions")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    bind = op.get_bind()
    sa.Enum(name="audit_event_result").drop(bind, checkfirst=True)
    sa.Enum(name="policy_decision_outcome").drop(bind, checkfirst=True)
    sa.Enum(name="tool_execution_status").drop(bind, checkfirst=True)
