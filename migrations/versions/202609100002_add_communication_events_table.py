"""Add communication_events table.

Captures recruiter-side signals tied to an application — an interview
invite, rejection, offer, etc. — even though the standalone Communications
Agent that will classify inbound messages is out of scope for this build;
this table only stores the classification once it's known.

`application_id` is a foreign key to `applications`. `provider_message_id`
identifies the source message (e.g. an email Message-ID) and, paired with
`application_id`, is unique so the same message ingested twice collapses
onto one row rather than duplicating.

Primary key and foreign key use `GUID`, matching `202609090001` and
`202609100001`, so this migration produces the same DDL
`Base.metadata.create_all()` would.

Revision ID: 202609100002
Revises: 202609100001
Create Date: 2026-09-10

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609100002"
down_revision = "202609100001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "communication_events",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "application_id", GUID(), sa.ForeignKey("applications.id"), nullable=False
        ),
        sa.Column(
            "classification",
            sa.Enum(
                "recruiter_response",
                "interview_invite",
                "rejection",
                "offer",
                "action_required",
                "general_update",
                name="communication_event_classification",
            ),
            nullable=False,
        ),
        sa.Column("provider_message_id", sa.String(length=255), nullable=True),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # Declared inline rather than via a later op.create_unique_constraint:
        # SQLite cannot ALTER a table to add a constraint, only create one with
        # the table, and this migration must run against both SQLite (tests,
        # local dev) and Postgres (production).
        sa.UniqueConstraint(
            "application_id",
            "provider_message_id",
            name="uq_communication_events_app_provider_message",
        ),
    )
    op.create_index(
        "ix_communication_events_application_id",
        "communication_events",
        ["application_id"],
    )


def downgrade() -> None:
    op.drop_table("communication_events")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    bind = op.get_bind()
    sa.Enum(name="communication_event_classification").drop(bind, checkfirst=True)
