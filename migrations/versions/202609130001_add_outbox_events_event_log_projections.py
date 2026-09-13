"""Add outbox_events, event_log, and application_status_view tables.

The transactional-outbox pattern plus an event-sourced read model:
outbox_events (one row per outbound message, dispatched by a worker that
claims it before publishing), event_log (the immutable, append-only history
of domain events), and application_status_view (a mutable projection of an
application's current status, recomputed from event_log rather than written
directly).

Outbox rows are written in the same DB transaction as the domain mutation
that produced them -- see `personalos.persistence.repositories.
OutboxEventRepository` and the `commit` parameter on
`ApplicationRepository.update_status` -- so a committed domain change can
never lose its corresponding outbound message. `event_log` rows are never
updated or deleted in application code; only `application_status_view` is
ever recomputed.

Primary keys and foreign keys use `GUID`, matching prior migrations, so this
migration produces the same DDL `Base.metadata.create_all()` would.

Revision ID: 202609130001
Revises: 202609110001
Create Date: 2026-09-13

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609130001"
down_revision = "202609110001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "outbox_events",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("type", sa.String(length=255), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("dedupe_key", sa.String(length=255), nullable=True, unique=True),
        sa.Column(
            "status",
            sa.Enum("pending", "in_progress", "dispatched", "failed", name="outbox_event_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_outbox_events_status", "outbox_events", ["status"])

    op.create_table(
        "event_log",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("aggregate_type", sa.String(length=100), nullable=False),
        sa.Column("aggregate_id", GUID(), nullable=False),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_event_log_aggregate_id", "event_log", ["aggregate_id"])
    op.create_index(
        "ix_event_log_aggregate_type_aggregate_id",
        "event_log",
        ["aggregate_type", "aggregate_id"],
    )

    op.create_table(
        "application_status_view",
        sa.Column("application_id", GUID(), sa.ForeignKey("applications.id"), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("last_event_id", GUID(), sa.ForeignKey("event_log.id"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("application_status_view")
    op.drop_table("event_log")
    op.drop_table("outbox_events")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    bind = op.get_bind()
    sa.Enum(name="outbox_event_status").drop(bind, checkfirst=True)
