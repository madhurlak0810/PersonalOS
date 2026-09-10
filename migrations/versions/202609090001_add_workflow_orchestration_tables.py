"""Add core workflow orchestration and persistence schema.

Adds the six tables every LangGraph-based workflow needs, independent of any
particular domain: users, workflows, workflow_runs, workflow_steps,
checkpoints and approvals. `approvals.action_hash` stores the exact
fingerprint of the action it was granted for (see
`personalos.policy.intents.fingerprint_intent`), so a changed action
fingerprints differently and can never reuse an older approval.

Primary keys and foreign keys use `GUID`, the same type decorator the ORM
models use, so this migration produces identical DDL to
`Base.metadata.create_all()` (native `uuid` on Postgres, `CHAR(36)`
elsewhere) instead of drifting from it.

This is the first Alembic-managed migration in the project; the pre-existing
`jobs`, `events`, `operations` and `agent_states` tables are created via
`Base.metadata.create_all()` and are out of scope here.

Revision ID: 202609090001
Revises:
Create Date: 2026-09-09

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609090001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=True, unique=True),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "workflows",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_workflows_name", "workflows", ["name"])

    op.create_table(
        "workflow_runs",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=False),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("thread_id", sa.String(length=255), nullable=False, unique=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "running", "completed", "failed", "cancelled",
                name="workflow_run_status",
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("correlation_id", GUID(), nullable=True),
        sa.Column("actor_id", sa.String(length=255), nullable=False, server_default="system"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_workflow_runs_workflow_id", "workflow_runs", ["workflow_id"])

    op.create_table(
        "workflow_steps",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "workflow_run_id", GUID(), sa.ForeignKey("workflow_runs.id"), nullable=False
        ),
        sa.Column("step_name", sa.String(length=255), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "running", "completed", "failed", "skipped",
                name="workflow_step_status",
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("input_data", sa.JSON(), nullable=True),
        sa.Column("output_data", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_workflow_steps_workflow_run_id", "workflow_steps", ["workflow_run_id"])

    op.create_table(
        "checkpoints",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=False),
        sa.Column(
            "workflow_run_id", GUID(), sa.ForeignKey("workflow_runs.id"), nullable=True
        ),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.String(length=255), nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=255), nullable=True),
        sa.Column("checkpoint", sa.JSON(), nullable=False),
        sa.Column("checkpoint_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # Declared inline rather than via a later op.create_unique_constraint:
        # SQLite cannot ALTER a table to add a constraint, only create one with
        # the table, and this migration must run against both SQLite (tests,
        # local dev) and Postgres (production).
        sa.UniqueConstraint(
            "thread_id", "checkpoint_ns", "checkpoint_id",
            name="uq_checkpoints_thread_ns_checkpoint",
        ),
    )
    op.create_index("ix_checkpoints_workflow_id", "checkpoints", ["workflow_id"])
    op.create_index("ix_checkpoints_thread_id", "checkpoints", ["thread_id"])

    op.create_table(
        "approvals",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "workflow_run_id", GUID(), sa.ForeignKey("workflow_runs.id"), nullable=True
        ),
        sa.Column(
            "workflow_step_id", GUID(), sa.ForeignKey("workflow_steps.id"), nullable=True
        ),
        sa.Column("action_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "approved", "denied", name="approval_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("requested_by_user_id", GUID(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("approved_by_user_id", GUID(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_approvals_action_hash", "approvals", ["action_hash"])


def downgrade() -> None:
    op.drop_table("approvals")
    op.drop_table("checkpoints")
    op.drop_table("workflow_steps")
    op.drop_table("workflow_runs")
    op.drop_table("workflows")
    op.drop_table("users")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    bind = op.get_bind()
    sa.Enum(name="workflow_run_status").drop(bind, checkfirst=True)
    sa.Enum(name="workflow_step_status").drop(bind, checkfirst=True)
    sa.Enum(name="approval_status").drop(bind, checkfirst=True)
