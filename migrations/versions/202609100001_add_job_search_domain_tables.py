"""Add the canonical job-search domain schema.

Adds the four tables the job-search workflow needs on top of the generic
orchestration schema from `202609090001`: job_postings (what was found),
candidate_profiles (what the user is looking for, versioned), applications
(one user's tracked pursuit of one posting, with a status lifecycle enforced
in application code — see `personalos.domain.models.
validate_application_status_transition` — never by this column definition),
and artifact_versions (the tailored resume/cover-letter drafts generated for
an application, each required to cite the evidence it was built from).

`job_postings.dedupe_key` carries the unique constraint that prevents the
same posting from being stored twice; `description_hash` and
`normalized_json` are the inputs a caller derives it from.

Primary keys and foreign keys use `GUID`, matching `202609090001`, so this
migration produces the same DDL `Base.metadata.create_all()` would.

Revision ID: 202609100001
Revises: 202609090001
Create Date: 2026-09-10

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202609100001"
down_revision = "202609090001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_postings",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("source_job_id", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("company", sa.String(length=255), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("normalized_json", sa.JSON(), nullable=False),
        sa.Column("description_hash", sa.String(length=64), nullable=False),
        sa.Column("dedupe_key", sa.String(length=150), nullable=False),
        sa.Column("posted_at", sa.DateTime(), nullable=True),
        sa.Column("discovered_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # Declared inline rather than via a later op.create_unique_constraint:
        # SQLite cannot ALTER a table to add a constraint, only create one with
        # the table, and this migration must run against both SQLite (tests,
        # local dev) and Postgres (production).
        sa.UniqueConstraint("dedupe_key", name="uq_job_postings_dedupe_key"),
    )
    op.create_index("ix_job_postings_source", "job_postings", ["source"])
    op.create_index("ix_job_postings_company", "job_postings", ["company"])

    op.create_table(
        "candidate_profiles",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("target_roles", sa.JSON(), nullable=False),
        sa.Column("target_locations", sa.JSON(), nullable=False),
        sa.Column("preferences", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "user_id", "profile_version", name="uq_candidate_profiles_user_version"
        ),
    )
    op.create_index("ix_candidate_profiles_user_id", "candidate_profiles", ["user_id"])

    op.create_table(
        "applications",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "job_posting_id", GUID(), sa.ForeignKey("job_postings.id"), nullable=False
        ),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "candidate_profile_id",
            GUID(),
            sa.ForeignKey("candidate_profiles.id"),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.Enum(
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
                name="application_status",
            ),
            nullable=False,
            server_default="discovered",
        ),
        sa.Column("resume_doc_id", sa.String(length=255), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "job_posting_id", "user_id", name="uq_applications_job_posting_user"
        ),
    )
    op.create_index("ix_applications_user_id", "applications", ["user_id"])
    op.create_index("ix_applications_status", "applications", ["status"])

    op.create_table(
        "artifact_versions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "application_id", GUID(), sa.ForeignKey("applications.id"), nullable=False
        ),
        sa.Column(
            "artifact_type",
            sa.Enum("resume", "cover_letter", name="artifact_type"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("doc_id", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        # No evidence linkage, no row: `ArtifactVersionRepository.create` runs
        # this through `validate_evidence_links` before insert, but NOT NULL
        # here guards against anything that writes to this table directly.
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("generated_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "application_id",
            "artifact_type",
            "version",
            name="uq_artifact_versions_app_type_version",
        ),
    )
    op.create_index(
        "ix_artifact_versions_application_id", "artifact_versions", ["application_id"]
    )


def downgrade() -> None:
    op.drop_table("artifact_versions")
    op.drop_table("applications")
    op.drop_table("candidate_profiles")
    op.drop_table("job_postings")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    bind = op.get_bind()
    sa.Enum(name="application_status").drop(bind, checkfirst=True)
    sa.Enum(name="artifact_type").drop(bind, checkfirst=True)
