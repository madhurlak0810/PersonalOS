"""Add `documents` and let evidence chunks outlive their document's edits.

Resume ingestion (the `ingest-resume` CLI command, `POST /v1/evidence/documents`)
records each file it loaded in `documents` -- path, the SHA-256 of what was
read, its format, the embedder its chunks were embedded with, and when -- and
points every chunk it cut at that row through `evidence_chunks.document_id`.

`evidence_chunks.retired_at` marks a chunk a newer version of its document no
longer contains but an `artifact_versions` row still cites. The chunk stays so
the citation resolves; readers skip it as evidence. For the same reason the
uniqueness of a chunk's position becomes a partial unique index over active
chunks only: a retired chunk must not hold the slot the new version needs.

Revision ID: 202611040001
Revises: 202610230001
Create Date: 2026-11-04

"""

import sqlalchemy as sa
from alembic import op

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202611040001"
down_revision = "202610230001"
branch_labels = None
depends_on = None

_POSITION_COLUMNS = ["user_id", "source_type", "source_ref", "chunk_index"]
_ACTIVE = sa.text("retired_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_version", sa.String(length=50), nullable=False),
        sa.Column("indexed_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("user_id", "path", name="uq_documents_user_path"),
    )

    op.add_column(
        "evidence_chunks",
        sa.Column("document_id", GUID(), sa.ForeignKey("documents.id"), nullable=True),
    )
    op.add_column("evidence_chunks", sa.Column("retired_at", sa.DateTime(), nullable=True))
    op.create_index("ix_evidence_chunks_document_id", "evidence_chunks", ["document_id"])

    op.drop_constraint("uq_evidence_chunks_user_source_chunk", "evidence_chunks", type_="unique")
    op.create_index(
        "uq_evidence_chunks_active_source_chunk",
        "evidence_chunks",
        _POSITION_COLUMNS,
        unique=True,
        postgresql_where=_ACTIVE,
        sqlite_where=_ACTIVE,
    )


def downgrade() -> None:
    # Retired chunks have no place in the old schema: without `retired_at`
    # they would read as evidence again, and they may collide on position.
    op.execute("DELETE FROM evidence_chunks WHERE retired_at IS NOT NULL")
    op.drop_index("uq_evidence_chunks_active_source_chunk", table_name="evidence_chunks")
    op.create_unique_constraint(
        "uq_evidence_chunks_user_source_chunk", "evidence_chunks", _POSITION_COLUMNS
    )
    op.drop_index("ix_evidence_chunks_document_id", table_name="evidence_chunks")
    op.drop_column("evidence_chunks", "retired_at")
    op.drop_column("evidence_chunks", "document_id")
    op.drop_table("documents")
