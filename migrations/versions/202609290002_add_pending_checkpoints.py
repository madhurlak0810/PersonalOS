"""Add pending_checkpoints for durable conditional waits.

One table, and the whole point of it is that nothing else is needed: a wait of
the form "seven days after applying, if no recruiter response exists, draft a
follow-up" is a row here and not a sleeping task, an open connection or a
scheduled future. A deploy, a crash or a weekend takes all three of those away;
it does not take the row away.

Three columns carry the design (see `personalos.domain.checkpoints`):

- `condition_kind` / `condition_subject_id` / `condition_since` store the
  condition *declaratively*, so it is asked again at trigger time rather than
  answered once at creation time -- a condition resolved when the wait was set
  up answers a question about the wrong moment.
- `trigger_at` is when the checkpoint becomes actionable.
- `expires_at` is when it must never act again, and it is a stored column
  rather than a derived one on purpose: it is what stops a monitor that was
  down for a week from sending a week-late follow-up, and what stops a
  checkpoint nobody swept from sitting `pending` forever.

`dedupe_key` is unique so the graph branch that schedules a follow-up can be
re-entered (a resumed run replaying its last super-step, a second recruiter
message) without stacking a second reminder on the same application -- the same
role `outbox_events.dedupe_key` plays for the outbox.

`application_id` carries no foreign key deliberately: checkpoints are scheduled
from graph state, which holds ids rather than rows, and the sweep never joins
to the application.

Revision ID: 202609290002
Revises: 202609290001
Create Date: 2026-09-29

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609290002"
down_revision = "202609290001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pending_checkpoints",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("application_id", GUID(), nullable=False),
        sa.Column("kind", sa.String(length=50), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("workflow_id", GUID(), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("condition_kind", sa.String(length=50), nullable=False),
        sa.Column("condition_subject_id", GUID(), nullable=False),
        sa.Column("condition_since", sa.DateTime(), nullable=True),
        sa.Column("condition_params", sa.JSON(), nullable=False),
        sa.Column("trigger_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "resolved",
                "fired",
                "expired",
                "cancelled",
                name="pending_checkpoint_status",
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("closed_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # Declared inline rather than via a later op.create_unique_constraint:
        # SQLite cannot ALTER a table to add a constraint, only create one with
        # the table, and this migration runs against both SQLite and Postgres.
        sa.UniqueConstraint("dedupe_key", name="uq_pending_checkpoints_dedupe_key"),
    )
    # The sweep's index. Leading with `status` keeps the scan off the closed
    # rows, which are the ones that accumulate: a checkpoint is closed once and
    # kept forever, so "what is still waiting?" must not get slower as the
    # history grows.
    op.create_index(
        "ix_pending_checkpoints_status_trigger_at",
        "pending_checkpoints",
        ["status", "trigger_at"],
    )
    op.create_index(
        "ix_pending_checkpoints_application_id",
        "pending_checkpoints",
        ["application_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_pending_checkpoints_application_id", table_name="pending_checkpoints")
    op.drop_index(
        "ix_pending_checkpoints_status_trigger_at", table_name="pending_checkpoints"
    )
    op.drop_table("pending_checkpoints")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    bind = op.get_bind()
    sa.Enum(name="pending_checkpoint_status").drop(bind, checkfirst=True)
