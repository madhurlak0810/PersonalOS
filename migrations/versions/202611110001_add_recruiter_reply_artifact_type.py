"""Let `artifact_versions` hold drafted recruiter replies.

`artifact_versions.artifact_type` gains `recruiter_reply`: a reply drafted for
a recruiter's message is stored, with the evidence it cites and the message it
answers, before anyone decides whether to send it.

The enum is rebuilt rather than extended with `ALTER TYPE ... ADD VALUE`, for
the reason given in `202610010002`.

Revision ID: 202611110001
Revises: 202611100001
Create Date: 2026-11-11

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "202611110001"
down_revision = "202611100001"
branch_labels = None
depends_on = None

_TYPE = "artifact_type"
_OLD = ("resume", "cover_letter")
_NEW = ("resume", "cover_letter", "recruiter_reply")


def _rebuild(values: tuple[str, ...]) -> None:
    """Replace the Postgres enum behind `artifact_versions.artifact_type` with `values`."""
    if op.get_bind().dialect.name != "postgresql":
        # Other dialects store the type as plain text; there is no type to alter.
        return
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"ALTER TYPE {_TYPE} RENAME TO {_TYPE}_old")
    op.execute(f"CREATE TYPE {_TYPE} AS ENUM ({labels})")
    op.execute(
        f"ALTER TABLE artifact_versions ALTER COLUMN artifact_type TYPE {_TYPE} "
        f"USING artifact_type::text::{_TYPE}"
    )
    op.execute(f"DROP TYPE {_TYPE}_old")


def upgrade() -> None:
    _rebuild(_NEW)


def downgrade() -> None:
    # Replies have no place in the old vocabulary. They are drafts, never the
    # record of anything sent, so they are dropped rather than relabelled as a
    # resume or cover letter they are not.
    op.execute("DELETE FROM artifact_versions WHERE artifact_type = 'recruiter_reply'")
    _rebuild(_OLD)
