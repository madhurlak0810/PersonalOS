"""Extend applications with the full lifecycle: new states, held-from, last activity.

`applications.status` gains `response`, `accepted`, `declined`,
`follow_up_pending` and `stalled`. As before, the column only constrains the
value to a known state; which moves are legal is
`personalos.domain.models.validate_application_status_transition`.

Two columns carry what the status alone cannot. `resume_status` is the state a
held application (`follow_up_pending`, `stalled`) was held from, which is what
decides where it may go next. `last_activity_at` is what the scheduled stall
check (`apps.worker.stall_monitor`) measures the stall window against;
existing rows are backfilled from `updated_at`, the closest thing they have to
a last-activity time, so none of them reads as either stalled since forever or
exempt from stalling.

The enum is rebuilt rather than extended with `ALTER TYPE ... ADD VALUE`, for
the reason given in `202610010002`.

Revision ID: 202610070001
Revises: 202610010002
Create Date: 2026-10-07

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "202610070001"
down_revision = "202610010002"
branch_labels = None
depends_on = None

_STATUS_TYPE = "application_status"
_OLD_STATUSES = (
    "discovered",
    "saved",
    "preparing",
    "ready_to_apply",
    "applied",
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
    "skipped",
)
_NEW_STATUSES = (
    "discovered",
    "saved",
    "preparing",
    "ready_to_apply",
    "applied",
    "response",
    "interviewing",
    "offer",
    "accepted",
    "declined",
    "rejected",
    "withdrawn",
    "skipped",
    "follow_up_pending",
    "stalled",
)

#: Where a row in a state the old vocabulary lacks lands on downgrade. The
#: nearest state that existed, never a more advanced one: `accepted` and
#: `declined` were both an `offer`, a `response` or a pending follow-up was
#: still `applied`.
_DOWNGRADE_STATUS = {
    "response": "applied",
    "follow_up_pending": "applied",
    "accepted": "offer",
    "declined": "offer",
}


def _rebuild_status_type(values: tuple[str, ...]) -> None:
    """Replace the Postgres enum behind `applications.status` with `values`."""
    if op.get_bind().dialect.name != "postgresql":
        # Other dialects store the status as plain text; there is no type to alter.
        return
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"ALTER TYPE {_STATUS_TYPE} RENAME TO {_STATUS_TYPE}_old")
    op.execute(f"CREATE TYPE {_STATUS_TYPE} AS ENUM ({labels})")
    op.execute("ALTER TABLE applications ALTER COLUMN status DROP DEFAULT")
    op.execute(
        f"ALTER TABLE applications ALTER COLUMN status TYPE {_STATUS_TYPE} "
        f"USING status::text::{_STATUS_TYPE}"
    )
    op.execute("ALTER TABLE applications ALTER COLUMN status SET DEFAULT 'discovered'")
    op.execute(f"DROP TYPE {_STATUS_TYPE}_old")


def upgrade() -> None:
    _rebuild_status_type(_NEW_STATUSES)

    op.add_column(
        "applications", sa.Column("resume_status", sa.String(length=32), nullable=True)
    )
    op.add_column("applications", sa.Column("last_activity_at", sa.DateTime(), nullable=True))
    op.execute("UPDATE applications SET last_activity_at = updated_at")
    op.create_index(
        "ix_applications_status_last_activity_at",
        "applications",
        ["status", "last_activity_at"],
    )


def downgrade() -> None:
    # A stalled row goes back to the state it stalled from; then every state
    # the old vocabulary lacks through the table above. `event_log` is left
    # alone: it is history, and the transitions happened. A `stalled` row with
    # no `resume_status` is not something the lifecycle can produce, so it is
    # not guessed at here -- the enum rebuild below fails on it instead.
    resume_status = "resume_status"
    if op.get_bind().dialect.name == "postgresql":
        resume_status = f"CAST(resume_status AS {_STATUS_TYPE})"
    op.execute(
        f"UPDATE applications SET status = {resume_status} "
        "WHERE status = 'stalled' AND resume_status IS NOT NULL"
    )
    for status, fallback in _DOWNGRADE_STATUS.items():
        op.execute(f"UPDATE applications SET status = '{fallback}' WHERE status = '{status}'")

    op.drop_index("ix_applications_status_last_activity_at", table_name="applications")
    op.drop_column("applications", "last_activity_at")
    op.drop_column("applications", "resume_status")
    _rebuild_status_type(_OLD_STATUSES)
