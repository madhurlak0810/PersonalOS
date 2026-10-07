"""The candidate's citable record, as job matching reads it.

One `EvidenceRecord` per `evidence_chunks` row, with the row id as the
`evidence_id` a match cites -- so a citation in a `JobMatch` is a key into
this table, and "does this evidence exist" is a lookup rather than a judgement.

`SqlEvidenceIndex` is the same table read by similarity instead of in full:
the nearest chunks to a query embedding, through pgvector on Postgres.
"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.artifacts import RetrievedEvidence
from personalos.domain.job_search import EvidenceRecord
from personalos.persistence.models import EvidenceChunkModel
from personalos.persistence.repositories import EvidenceChunkRepository


def _to_record(row: EvidenceChunkModel) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(row.id),
        source_type=row.source_type,
        source_ref=row.source_ref,
        text=row.chunk_text,
    )


class SqlEvidenceSource:
    """Binds job matching's `EvidenceSource` port to the `evidence_chunks` table."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    async def load(self, user_id: UUID) -> list[EvidenceRecord]:
        """Every non-empty chunk on file for the user, in a stable order."""
        session = self.session_factory()
        try:
            rows = (
                session.query(EvidenceChunkModel)
                .filter(EvidenceChunkModel.user_id == user_id)
                .order_by(
                    EvidenceChunkModel.source_type,
                    EvidenceChunkModel.source_ref,
                    EvidenceChunkModel.chunk_index,
                    EvidenceChunkModel.id,
                )
                .all()
            )
            return [_to_record(row) for row in rows if row.chunk_text.strip()]
        finally:
            session.close()


class SqlEvidenceIndex:
    """Binds artifact prep's `EvidenceIndex` port to nearest-neighbour search.

    Scoped to one user and one embedding model on every call: vectors from
    different models are not comparable, and another candidate's record is
    never evidence for this one.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    async def search(
        self,
        *,
        user_id: UUID,
        query_embedding: list[float],
        embedding_model: str,
        top_k: int,
        min_similarity: float,
    ) -> list[RetrievedEvidence]:
        """The user's chunks nearest `query_embedding`, most similar first."""
        session = self.session_factory()
        try:
            rows = EvidenceChunkRepository(session).find_similar(
                query_embedding=query_embedding,
                embedding_model=embedding_model,
                user_id=user_id,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            return [
                RetrievedEvidence(record=_to_record(row), similarity=float(similarity))
                for row, similarity in rows
                if row.chunk_text.strip()
            ]
        finally:
            session.close()


__all__ = ["SqlEvidenceSource", "SqlEvidenceIndex"]
