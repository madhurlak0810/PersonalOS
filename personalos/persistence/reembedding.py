"""The vector tables as the re-embedding backfill reads and writes them.

`SqlReembedStore` is the store behind `personalos.retrieval.reembedding.Reembedder`.
For each `VectorTable` it lists what has no vector from the target model and
version yet, in id order and one page at a time, and it writes one batch of
new vectors per transaction. A run that dies mid-way loses at most the batch
it was embedding. Everything committed before that is no longer pending, so
the next run skips it.

Evidence vectors are overwritten in place, so the chunk ids that
`artifact_versions` cite never change (see `personalos.domain.embeddings`). Once
every chunk of a document is on the target model, the `documents` row is moved
to it as well, so a later `ingest-resume` of the unchanged file is still a
no-op.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import and_, exists, not_
from sqlalchemy.orm import Session

from personalos.domain.embeddings import PendingVector, VectorTable
from personalos.domain.job_search import NormalizedPosting
from personalos.persistence.models import (
    CommunicationEventModel,
    DocumentModel,
    EvidenceChunkModel,
    JobPostingEmbeddingModel,
    JobPostingModel,
    MessageEmbeddingModel,
)

#: Turns a stored posting into the text its vector is embedded from.
PostingText = Callable[[NormalizedPosting], str]


def _default_posting_text(posting: NormalizedPosting) -> str:
    return "\n".join(
        part for part in (posting.title, ", ".join(posting.skills), posting.description) if part
    )


class SqlReembedStore:
    """`ReembedStore` over `evidence_chunks`, `job_posting_embeddings` and `message_embeddings`."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        posting_text: PostingText | None = None,
    ):
        """Bind the database and the text a posting is embedded from.

        The composition root passes `personalos.retrieval.artifact_prep.retrieval_query`,
        the text artifact prep embeds a posting as, so a backfilled posting
        vector is comparable with the query vectors.
        """
        self.session_factory = session_factory
        self.posting_text = posting_text or _default_posting_text

    # --- Reads -----------------------------------------------------------------------

    def count_pending(self, table: VectorTable, *, model: str, version: str) -> int:
        """How many rows of `table` have no vector from `model`/`version` yet."""
        session = self.session_factory()
        try:
            return self._pending_query(session, table, model, version).count()
        finally:
            session.close()

    def pending(
        self,
        table: VectorTable,
        *,
        model: str,
        version: str,
        after: UUID | None,
        limit: int,
    ) -> list[PendingVector]:
        """Up to `limit` pending rows with ids after `after`, in id order."""
        session = self.session_factory()
        try:
            query = self._pending_query(session, table, model, version)
            column = _key_column(table)
            if after is not None:
                query = query.filter(column > after)
            rows = query.order_by(column).limit(limit).all()
            return [PendingVector(key=row.id, text=self._text(table, row)) for row in rows]
        finally:
            session.close()

    def _pending_query(self, session: Session, table: VectorTable, model: str, version: str):
        if table is VectorTable.EVIDENCE:
            return session.query(EvidenceChunkModel).filter(
                not_(
                    and_(
                        EvidenceChunkModel.embedding_model == model,
                        EvidenceChunkModel.embedding_version == version,
                    )
                )
            )
        if table is VectorTable.POSTINGS:
            vectors, source, foreign_key = (
                JobPostingEmbeddingModel,
                JobPostingModel,
                JobPostingEmbeddingModel.job_posting_id,
            )
        else:
            vectors, source, foreign_key = (
                MessageEmbeddingModel,
                CommunicationEventModel,
                MessageEmbeddingModel.communication_event_id,
            )
        # Only what was embedded before: a backfill re-embeds, it does not
        # embed postings or messages that never had a vector.
        has_any = exists().where(foreign_key == source.id)
        has_target = exists().where(
            foreign_key == source.id,
            vectors.embedding_model == model,
            vectors.embedding_version == version,
        )
        return session.query(source).filter(has_any, ~has_target)

    def _text(self, table: VectorTable, row) -> str:
        if table is VectorTable.EVIDENCE:
            return row.chunk_text or ""
        if table is VectorTable.POSTINGS:
            try:
                posting = NormalizedPosting.model_validate(row.normalized_json or {})
            except ValidationError:
                return "\n".join(part for part in (row.title, row.company) if part)
            return self.posting_text(posting)
        metadata = row.metadata_json or {}
        return "\n".join(
            str(metadata[field]) for field in ("subject", "summary") if metadata.get(field)
        )

    # --- Writes ----------------------------------------------------------------------

    def write(
        self,
        table: VectorTable,
        vectors: Sequence[tuple[UUID, list[float]]],
        *,
        model: str,
        version: str,
    ) -> int:
        """Store one batch of vectors in one transaction; returns how many were written."""
        session = self.session_factory()
        try:
            now = datetime.utcnow()
            if table is VectorTable.EVIDENCE:
                written = _write_evidence(session, vectors, model, version, now)
            else:
                written = _write_additive(session, table, vectors, model, version, now)
            session.commit()
            return written
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _key_column(table: VectorTable):
    return {
        VectorTable.EVIDENCE: EvidenceChunkModel.id,
        VectorTable.POSTINGS: JobPostingModel.id,
        VectorTable.MESSAGES: CommunicationEventModel.id,
    }[table]


def _write_evidence(
    session: Session,
    vectors: Sequence[tuple[UUID, list[float]]],
    model: str,
    version: str,
    now: datetime,
) -> int:
    """Overwrite each chunk's vector in place, keeping its id (the cited evidence id)."""
    written = 0
    documents: set[UUID] = set()
    for evidence_id, vector in vectors:
        row = session.get(EvidenceChunkModel, evidence_id)
        if row is None:  # deleted by a re-ingest since it was listed
            continue
        row.embedding = vector
        row.embedding_model, row.embedding_version = model, version
        row.updated_at = now
        written += 1
        if row.document_id is not None:
            documents.add(row.document_id)
    session.flush()

    for document_id in documents:
        behind = (
            session.query(EvidenceChunkModel.id)
            .filter(
                EvidenceChunkModel.document_id == document_id,
                not_(
                    and_(
                        EvidenceChunkModel.embedding_model == model,
                        EvidenceChunkModel.embedding_version == version,
                    )
                ),
            )
            .first()
        )
        document = session.get(DocumentModel, document_id)
        if behind is None and document is not None:
            document.embedding_model, document.embedding_version = model, version
            document.updated_at = now
    return written


def _write_additive(
    session: Session,
    table: VectorTable,
    vectors: Sequence[tuple[UUID, list[float]]],
    model: str,
    version: str,
    now: datetime,
) -> int:
    """Add a row per source for the new model, leaving every old row as it is."""
    if table is VectorTable.POSTINGS:
        row_type, foreign_key = JobPostingEmbeddingModel, "job_posting_id"
    else:
        row_type, foreign_key = MessageEmbeddingModel, "communication_event_id"
    written = 0
    for source_id, vector in vectors:
        duplicate = (
            session.query(row_type.id)
            .filter(
                getattr(row_type, foreign_key) == source_id,
                row_type.embedding_model == model,
                row_type.embedding_version == version,
            )
            .first()
        )
        if duplicate is not None:
            continue
        session.add(
            row_type(
                **{foreign_key: source_id},
                embedding=vector,
                embedding_model=model,
                embedding_version=version,
                created_at=now,
            )
        )
        written += 1
    session.flush()
    return written


__all__ = ["PostingText", "SqlReembedStore"]
