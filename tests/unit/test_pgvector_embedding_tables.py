"""Tests for the evidence_chunks, job_posting_embeddings, and message_embeddings schema.

Covers the acceptance criteria from the pgvector migration: the three tables
exist with the documented shape, an embedding can be inserted and retrieved,
and nearest-neighbor search returns only rows above a similarity threshold,
ranked most-similar first. These tests run against SQLite (see
`personalos.persistence.models.Vector`), which exercises the same repository
contract as Postgres+pgvector through the non-Postgres fallback path in
`personalos.persistence.repositories` -- the dialect branch itself is the
same pattern already used by `OutboxEventRepository.claim_next`.
"""

import math
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from personalos.persistence.models import Base, UserModel
from personalos.persistence.repositories import (
    ApplicationRepository,
    CommunicationEventRepository,
    EvidenceChunkRepository,
    JobPostingEmbeddingRepository,
    JobPostingRepository,
    MessageEmbeddingRepository,
)

EMBEDDING_MODEL = "text-embedding-3-small"


def _open(db_path):
    """A fresh engine + session over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory(), engine


def _unit_vector(angle_radians: float, dim: int = 8) -> list[float]:
    """A deterministic, non-degenerate embedding-shaped vector for testing.

    Two vectors built from nearby angles are cosine-similar; two built from
    angles ~90 degrees apart are not. `dim` stays small (real embeddings are
    1536-wide) since these tests only exercise similarity ranking, not the
    production dimension.
    """
    vec = [math.cos(angle_radians + i) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _make_application(session):
    user = UserModel(email=f"candidate-{id(session)}@example.com")
    session.add(user)
    session.commit()
    posting = JobPostingRepository(session).create(
        source="linkedin",
        title="Staff Engineer",
        company="Acme",
        description_hash="a" * 64,
        dedupe_key=f"acme:staff-engineer:{id(session)}",
    )
    application = ApplicationRepository(session).create(
        job_posting_id=posting.id, user_id=user.id
    )
    return user, posting, application


# ----------------------------------------------------------------------
# Migration shape: tables and the documented indexes/constraints exist.
# ----------------------------------------------------------------------


def test_schema_creates_all_three_tables_with_documented_shape(tmp_path):
    """evidence_chunks, job_posting_embeddings, and message_embeddings all exist as expected."""
    _, engine = _open(tmp_path / "pgvector.db")
    try:
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert {
            "evidence_chunks",
            "job_posting_embeddings",
            "message_embeddings",
        } <= table_names

        evidence_columns = {c["name"] for c in inspector.get_columns("evidence_chunks")}
        assert {
            "id",
            "user_id",
            "source_type",
            "source_ref",
            "chunk_text",
            "chunk_index",
            "embedding",
            "embedding_model",
            "embedding_version",
            "created_at",
            "updated_at",
        } == evidence_columns
        evidence_unique_columns = {
            column
            for uc in inspector.get_unique_constraints("evidence_chunks")
            for column in uc["column_names"]
        }
        assert {"user_id", "source_type", "source_ref", "chunk_index"} <= evidence_unique_columns

        posting_columns = {c["name"] for c in inspector.get_columns("job_posting_embeddings")}
        assert {
            "id",
            "job_posting_id",
            "embedding",
            "embedding_model",
            "embedding_version",
            "created_at",
        } == posting_columns
        posting_unique_columns = {
            column
            for uc in inspector.get_unique_constraints("job_posting_embeddings")
            for column in uc["column_names"]
        }
        assert {"job_posting_id", "embedding_model", "embedding_version"} <= posting_unique_columns

        message_columns = {c["name"] for c in inspector.get_columns("message_embeddings")}
        assert {
            "id",
            "communication_event_id",
            "embedding",
            "embedding_model",
            "embedding_version",
            "created_at",
        } == message_columns
    finally:
        engine.dispose()


# ----------------------------------------------------------------------
# evidence_chunks: insert + nearest-neighbor retrieval above a threshold.
# ----------------------------------------------------------------------


def test_evidence_chunk_similarity_search_returns_only_matches_above_threshold(tmp_path):
    """A near-duplicate chunk ranks first and above threshold; an unrelated one is excluded."""
    session, engine = _open(tmp_path / "pgvector.db")
    try:
        user = UserModel(email="candidate@example.com")
        session.add(user)
        session.commit()
        repo = EvidenceChunkRepository(session)

        query_embedding = _unit_vector(0.0)
        close_chunk = repo.create(
            user_id=user.id,
            source_type="resume",
            source_ref="resume-v1",
            chunk_index=0,
            chunk_text="Led migration of the payments service to a new event-driven architecture.",
            embedding=_unit_vector(0.01),
            embedding_model=EMBEDDING_MODEL,
        )
        repo.create(
            user_id=user.id,
            source_type="project",
            source_ref="side-project",
            chunk_index=0,
            chunk_text="Built a weekend hiking-trail recommendation app.",
            embedding=_unit_vector(math.pi / 2),
            embedding_model=EMBEDDING_MODEL,
        )

        results = repo.find_similar(
            query_embedding=query_embedding,
            embedding_model=EMBEDDING_MODEL,
            top_k=5,
            min_similarity=0.9,
        )

        assert len(results) == 1
        matched_chunk, similarity = results[0]
        assert matched_chunk.id == close_chunk.id
        assert similarity >= 0.9
    finally:
        session.close()
        engine.dispose()


def test_evidence_chunk_duplicate_ingest_is_rejected(tmp_path):
    """The uniqueness constraint stops the same chunk from being embedded twice."""
    session, engine = _open(tmp_path / "pgvector.db")
    try:
        user = UserModel(email="candidate@example.com")
        session.add(user)
        session.commit()
        repo = EvidenceChunkRepository(session)

        repo.create(
            user_id=user.id,
            source_type="resume",
            source_ref="resume-v1",
            chunk_index=0,
            chunk_text="Led migration of the payments service.",
            embedding=_unit_vector(0.0),
            embedding_model=EMBEDDING_MODEL,
        )
        with pytest.raises(IntegrityError):
            repo.create(
                user_id=user.id,
                source_type="resume",
                source_ref="resume-v1",
                chunk_index=0,
                chunk_text="Led migration of the payments service (retry).",
                embedding=_unit_vector(0.02),
                embedding_model=EMBEDDING_MODEL,
            )
    finally:
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# job_posting_embeddings: per-model versioning and similarity ranking.
# ----------------------------------------------------------------------


def test_job_posting_embedding_supports_multiple_models_without_clobbering(tmp_path):
    """Re-embedding a posting under a new model adds a row rather than overwriting the old one."""
    session, engine = _open(tmp_path / "pgvector.db")
    try:
        _, posting, _ = _make_application(session)
        repo = JobPostingEmbeddingRepository(session)

        repo.create(
            job_posting_id=posting.id,
            embedding=_unit_vector(0.0),
            embedding_model="text-embedding-ada-002",
        )
        repo.create(
            job_posting_id=posting.id,
            embedding=_unit_vector(0.0),
            embedding_model="text-embedding-3-small",
        )

        old_model_results = repo.find_similar(
            query_embedding=_unit_vector(0.0),
            embedding_model="text-embedding-ada-002",
            min_similarity=0.0,
        )
        new_model_results = repo.find_similar(
            query_embedding=_unit_vector(0.0),
            embedding_model="text-embedding-3-small",
            min_similarity=0.0,
        )
        assert len(old_model_results) == 1
        assert len(new_model_results) == 1
        assert old_model_results[0][0].id != new_model_results[0][0].id
    finally:
        session.close()
        engine.dispose()


def test_job_posting_embedding_ranks_most_similar_first(tmp_path):
    """Multiple candidates above threshold come back ordered by descending similarity."""
    session, engine = _open(tmp_path / "pgvector.db")
    try:
        _, posting_a, _ = _make_application(session)
        posting_b = JobPostingRepository(session).create(
            source="linkedin",
            title="Senior Engineer",
            company="Acme",
            description_hash="b" * 64,
            dedupe_key="acme:senior-engineer",
        )
        repo = JobPostingEmbeddingRepository(session)

        query_embedding = _unit_vector(0.0)
        repo.create(
            job_posting_id=posting_a.id,
            embedding=_unit_vector(0.3),
            embedding_model=EMBEDDING_MODEL,
        )
        repo.create(
            job_posting_id=posting_b.id,
            embedding=_unit_vector(0.05),
            embedding_model=EMBEDDING_MODEL,
        )

        results = repo.find_similar(
            query_embedding=query_embedding,
            embedding_model=EMBEDDING_MODEL,
            min_similarity=0.5,
        )

        assert [r[0].job_posting_id for r in results] == [posting_b.id, posting_a.id]
        assert results[0][1] >= results[1][1]
    finally:
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# message_embeddings: semantic search over recruiter communications.
# ----------------------------------------------------------------------


def test_message_embedding_similarity_search_above_threshold(tmp_path):
    """A recruiter message's embedding is retrievable by semantic similarity."""
    session, engine = _open(tmp_path / "pgvector.db")
    try:
        _, _, application = _make_application(session)
        event = CommunicationEventRepository(session).create(
            application_id=application.id,
            classification="interview_invite",
            occurred_at=datetime(2026, 9, 10, 9, 0),
        )
        repo = MessageEmbeddingRepository(session)
        repo.create(
            communication_event_id=event.id,
            embedding=_unit_vector(0.0),
            embedding_model=EMBEDDING_MODEL,
        )

        results = repo.find_similar(
            query_embedding=_unit_vector(0.02),
            embedding_model=EMBEDDING_MODEL,
            min_similarity=0.9,
        )

        assert len(results) == 1
        assert results[0][0].communication_event_id == event.id
    finally:
        session.close()
        engine.dispose()
