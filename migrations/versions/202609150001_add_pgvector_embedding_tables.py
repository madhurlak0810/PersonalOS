"""Add pgvector extension and embedding tables for semantic retrieval.

Three embedding tables backing evidence-grounded job matching and semantic
scoring/dedup: evidence_chunks (resume/project text cut into retrievable
chunks), job_posting_embeddings (one embedding per posting per embedding
model, for similarity scoring and near-duplicate detection), and
message_embeddings (recruiter message text, for semantic search over
communication history).

Every row carries `embedding_model`/`embedding_version` alongside its vector,
and each embedding table's uniqueness constraint is scoped to include those
columns, so re-embedding with a new model or version adds rows rather than
overwriting the old ones -- old and new embeddings can coexist until callers
have moved off the old model.

Enables the pgvector extension and, on Postgres only, an ivfflat index per
embedding column using `vector_cosine_ops` (matching the `cosine_distance`
comparator `personalos.persistence.repositories` queries with). SQLite (used
in tests) has no such extension or index type -- see
`personalos.persistence.models.Vector` for how the column type itself
degrades on that dialect, and the repositories for how nearest-neighbor
search falls back to Python there instead.

Revision ID: 202609150001
Revises: 202609130001
Create Date: 2026-09-15

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import EMBEDDING_DIMENSION, GUID, Vector

# revision identifiers, used by Alembic.
revision = "202609150001"
down_revision = "202609130001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "evidence_chunks",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "source_type",
            sa.Enum("resume", "project", name="evidence_source_type"),
            nullable=False,
        ),
        sa.Column("source_ref", sa.String(length=255), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_version", sa.String(length=50), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_evidence_chunks_user_id", "evidence_chunks", ["user_id"])
    op.create_unique_constraint(
        "uq_evidence_chunks_user_source_chunk",
        "evidence_chunks",
        ["user_id", "source_type", "source_ref", "chunk_index"],
    )

    op.create_table(
        "job_posting_embeddings",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("job_posting_id", GUID(), sa.ForeignKey("job_postings.id"), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_version", sa.String(length=50), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_job_posting_embeddings_job_posting_id", "job_posting_embeddings", ["job_posting_id"]
    )
    op.create_unique_constraint(
        "uq_job_posting_embeddings_posting_model_version",
        "job_posting_embeddings",
        ["job_posting_id", "embedding_model", "embedding_version"],
    )

    op.create_table(
        "message_embeddings",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "communication_event_id",
            GUID(),
            sa.ForeignKey("communication_events.id"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_version", sa.String(length=50), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_message_embeddings_communication_event_id",
        "message_embeddings",
        ["communication_event_id"],
    )
    op.create_unique_constraint(
        "uq_message_embeddings_event_model_version",
        "message_embeddings",
        ["communication_event_id", "embedding_model", "embedding_version"],
    )

    if is_postgres:
        op.execute(
            "CREATE INDEX ix_evidence_chunks_embedding_cosine ON evidence_chunks "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        op.execute(
            "CREATE INDEX ix_job_posting_embeddings_embedding_cosine ON job_posting_embeddings "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        op.execute(
            "CREATE INDEX ix_message_embeddings_embedding_cosine ON message_embeddings "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_table("message_embeddings")
    op.drop_table("job_posting_embeddings")
    op.drop_table("evidence_chunks")
    # Postgres enum types created by sa.Enum(...) in create_table outlive
    # drop_table and must be dropped explicitly, or re-running this migration
    # fails with "type already exists".
    sa.Enum(name="evidence_source_type").drop(bind, checkfirst=True)
    # The pgvector extension is left in place: other tables/migrations may
    # depend on it, and CREATE EXTENSION is cluster-wide, not owned by this
    # migration alone.
