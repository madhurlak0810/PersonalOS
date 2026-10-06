"""The candidate's citable record, as job matching reads it.

One `EvidenceRecord` per `evidence_chunks` row, with the row id as the
`evidence_id` a match cites -- so a citation in a `JobMatch` is a key into
this table, and "does this evidence exist" is a lookup rather than a judgement.
"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.job_search import EvidenceRecord
from personalos.persistence.models import EvidenceChunkModel


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
            return [
                EvidenceRecord(
                    evidence_id=str(row.id),
                    source_type=row.source_type,
                    source_ref=row.source_ref,
                    text=row.chunk_text,
                )
                for row in rows
                if row.chunk_text.strip()
            ]
        finally:
            session.close()


__all__ = ["SqlEvidenceSource"]
