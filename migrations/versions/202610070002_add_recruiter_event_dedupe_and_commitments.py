"""Add communication_events.dedupe_key, the `unrelated` classification, and commitments.

`communication_events.dedupe_key` is what makes recording an inbound message
idempotent. The existing unique constraint is per application, which stops one
application getting the same message twice but not the same message being
attached to two applications by two deliveries that correlated differently.
The new key is derived from the provider message id alone (see
`personalos.domain.recruiter_events.communication_dedupe_key`) and is unique
across the table. It is a unique index rather than a constraint because SQLite
cannot add a constraint to an existing table.

Existing rows are backfilled where that is unambiguous (and the id is short
enough for the key to be the id itself rather than its hash). A provider message id
already recorded against more than one application keeps a NULL key on every
one of its rows: picking a winner here would be choosing which application the
message was really about, and that is not a migration's decision.

`communication_event_classification` gains `unrelated`, rebuilt rather than
extended with `ALTER TYPE ... ADD VALUE` for the reason given in
`202610010002`.

`commitments` holds what a message commits anyone to: who owes what, by when,
on what condition, always tied to the `communication_events` row it came from.

Revision ID: 202610070002
Revises: 202610070001
Create Date: 2026-10-07

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202610070002"
down_revision = "202610070001"
branch_labels = None
depends_on = None

_CLASSIFICATION_TYPE = "communication_event_classification"
_OLD_CLASSIFICATIONS = (
    "recruiter_response",
    "interview_invite",
    "rejection",
    "offer",
    "action_required",
    "general_update",
)
_NEW_CLASSIFICATIONS = (*_OLD_CLASSIFICATIONS, "unrelated")


def _rebuild_classification_type(values: tuple[str, ...]) -> None:
    """Replace the Postgres enum behind `communication_events.classification`."""
    if op.get_bind().dialect.name != "postgresql":
        # Other dialects store the classification as plain text.
        return
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"ALTER TYPE {_CLASSIFICATION_TYPE} RENAME TO {_CLASSIFICATION_TYPE}_old")
    op.execute(f"CREATE TYPE {_CLASSIFICATION_TYPE} AS ENUM ({labels})")
    op.execute(
        f"ALTER TABLE communication_events ALTER COLUMN classification "
        f"TYPE {_CLASSIFICATION_TYPE} USING classification::text::{_CLASSIFICATION_TYPE}"
    )
    op.execute(f"DROP TYPE {_CLASSIFICATION_TYPE}_old")


def upgrade() -> None:
    _rebuild_classification_type(_NEW_CLASSIFICATIONS)

    op.add_column(
        "communication_events", sa.Column("dedupe_key", sa.String(length=300), nullable=True)
    )
    op.execute(
        "UPDATE communication_events SET dedupe_key = 'message:' || provider_message_id "
        "WHERE provider_message_id IS NOT NULL AND length(provider_message_id) <= 200 "
        "AND provider_message_id NOT IN ("
        "SELECT provider_message_id FROM ("
        "SELECT provider_message_id FROM communication_events "
        "WHERE provider_message_id IS NOT NULL "
        "GROUP BY provider_message_id HAVING COUNT(*) > 1) AS shared)"
    )
    op.create_index(
        "uq_communication_events_dedupe_key",
        "communication_events",
        ["dedupe_key"],
        unique=True,
    )

    op.create_table(
        "commitments",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "communication_event_id",
            GUID(),
            sa.ForeignKey("communication_events.id"),
            nullable=False,
        ),
        sa.Column("application_id", GUID(), sa.ForeignKey("applications.id"), nullable=False),
        sa.Column(
            "actor", sa.Enum("user", "external_person", name="commitment_actor"), nullable=False
        ),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("due_at", sa.DateTime(), nullable=True),
        sa.Column("condition", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source_message_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_commitments_application_id", "commitments", ["application_id"])
    op.create_index(
        "ix_commitments_communication_event_id", "commitments", ["communication_event_id"]
    )
    op.create_index("ix_commitments_due_at", "commitments", ["due_at"])


def downgrade() -> None:
    op.drop_table("commitments")
    # See `202609100002`: the enum type outlives the table that used it.
    sa.Enum(name="commitment_actor").drop(op.get_bind(), checkfirst=True)

    op.drop_index("uq_communication_events_dedupe_key", table_name="communication_events")
    op.drop_column("communication_events", "dedupe_key")

    # The old vocabulary has nowhere to put an unrelated message, and such a
    # row was only ever recorded because a reviewer tied it to an application
    # by hand; `general_update` is the closest thing that says nothing more.
    op.execute(
        "UPDATE communication_events SET classification = 'general_update' "
        "WHERE classification = 'unrelated'"
    )
    _rebuild_classification_type(_OLD_CLASSIFICATIONS)
