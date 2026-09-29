"""Add checkpoint_writes and workflow_leases for durable resume.

Two tables the durable LangGraph checkpointer needs beyond the `checkpoints`
table added in 202609090001.

`checkpoint_writes` stores the writes each task produced against a checkpoint.
A checkpoint alone records the state at a super-step boundary; the writes record
which tasks *inside* the in-flight super-step had already finished when the
process died. Without them, a resume re-runs every task in that step, which for
a step that performed an outward-facing action means doing it twice -- exactly
what `personalos.persistence.action_journal` and this table exist together to
prevent. Keyed like LangGraph's own write tuple:
`(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`.

`workflow_leases` is one row per workflow, and its unique constraint on
`workflow_id` is the primitive that stops two workers resuming the same
workflow. The constraint matters more than the row: `SELECT ... FOR UPDATE` is a
no-op on SQLite (tests, local dev), so a guarantee that depended on row locking
would hold in production and quietly not hold anywhere else. See
`personalos.persistence.leases`.

Revision ID: 202609290001
Revises: 202609150001
Create Date: 2026-09-29

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609290001"
down_revision = "202609150001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "checkpoint_writes",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.String(length=255), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("channel", sa.String(length=255), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("task_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        # Declared inline rather than via a later op.create_unique_constraint:
        # SQLite cannot ALTER a table to add a constraint, only create one with
        # the table, and this migration runs against both SQLite and Postgres.
        sa.UniqueConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            name="uq_checkpoint_writes_thread_ns_checkpoint_task_idx",
        ),
    )
    op.create_index(
        "ix_checkpoint_writes_thread_ns_checkpoint",
        "checkpoint_writes",
        ["thread_id", "checkpoint_ns", "checkpoint_id"],
    )

    op.create_table(
        "workflow_leases",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=True),
        sa.Column("owner", sa.String(length=255), nullable=False),
        sa.Column("lease_token", GUID(), nullable=False),
        sa.Column("acquired_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("released_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # One lease per workflow. This is the mutual-exclusion primitive, not a
        # data-hygiene constraint: the worker that loses the insert race gets an
        # IntegrityError instead of a second lease.
        sa.UniqueConstraint("workflow_id", name="uq_workflow_leases_workflow_id"),
    )
    op.create_index("ix_workflow_leases_expires_at", "workflow_leases", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_workflow_leases_expires_at", table_name="workflow_leases")
    op.drop_table("workflow_leases")
    op.drop_index("ix_checkpoint_writes_thread_ns_checkpoint", table_name="checkpoint_writes")
    op.drop_table("checkpoint_writes")
