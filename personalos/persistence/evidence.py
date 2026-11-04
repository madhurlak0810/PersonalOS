"""The candidate's citable record, as job matching reads it.

One `EvidenceRecord` per `evidence_chunks` row, with the row id as the
`evidence_id` a match cites -- so a citation in a `JobMatch` is a key into
this table, and "does this evidence exist" is a lookup rather than a judgement.

`SqlEvidenceIndex` is the same table read by similarity instead of in full:
the nearest chunks to a query embedding, through pgvector on Postgres.

`SqlEvidenceIngestionStore` is the write side for resume ingestion: the
`documents` row for each ingested file, and the chunks cut from it, replaced
version by version without losing a chunk an artifact version cites.

Retired chunks (`retired_at` set) are never evidence: both readers skip them.
"""

from collections.abc import Callable, Collection, Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.artifacts import RetrievedEvidence
from personalos.domain.errors import Conflict
from personalos.domain.evidence import (
    DocumentKind,
    EmbeddedChunk,
    IndexedDocument,
    IndexedEvidence,
    IngestionResult,
    IngestionStatus,
)
from personalos.domain.job_search import EvidenceRecord
from personalos.domain.redaction import redact_text
from personalos.persistence.models import (
    ApplicationModel,
    ArtifactVersionModel,
    DocumentModel,
    EvidenceChunkModel,
    UserModel,
)
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
                .filter(
                    EvidenceChunkModel.user_id == user_id,
                    EvidenceChunkModel.retired_at.is_(None),
                )
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


class SqlEvidenceIngestionStore:
    """The `documents` table and the chunks ingestion cuts into `evidence_chunks`.

    `sync` replaces one document's chunks with a new set in one transaction.
    Chunks are matched by evidence id, which ingestion derives from the text,
    so a chunk the new version still contains keeps its row -- and its id --
    and only moves to its new position. A chunk the new version dropped is
    deleted, or retired if any of the candidate's artifact versions cites it.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    def document(self, user_id: UUID, path: str) -> IndexedDocument | None:
        """The ingested document at `path` and its active evidence, if on record."""
        session = self.session_factory()
        try:
            row = (
                session.query(DocumentModel)
                .filter(DocumentModel.user_id == user_id, DocumentModel.path == path)
                .one_or_none()
            )
            return None if row is None else _indexed(session, row)
        finally:
            session.close()

    def embedded(
        self,
        user_id: UUID,
        evidence_ids: Collection[UUID],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> set[UUID]:
        """Which of `evidence_ids` already have a vector from this model and version."""
        if not evidence_ids:
            return set()
        session = self.session_factory()
        try:
            rows = (
                session.query(EvidenceChunkModel.id)
                .filter(
                    EvidenceChunkModel.user_id == user_id,
                    EvidenceChunkModel.id.in_(list(evidence_ids)),
                    EvidenceChunkModel.embedding_model == embedding_model,
                    EvidenceChunkModel.embedding_version == embedding_version,
                )
                .all()
            )
            return {row_id for (row_id,) in rows}
        finally:
            session.close()

    def sync(
        self,
        *,
        user_id: UUID,
        path: str,
        sha256: str,
        kind: DocumentKind,
        embedding_model: str,
        embedding_version: str,
        chunks: Sequence[EmbeddedChunk],
        display_name: str | None = None,
    ) -> IngestionResult:
        """Make `chunks` the document's active evidence, and record the document.

        A chunk with `embedding=None` must already be on record with a vector
        from `embedding_model`/`embedding_version` (see `embedded`).
        """
        session = self.session_factory()
        try:
            now = datetime.utcnow()
            document, status = _record_document(
                session,
                user_id=user_id,
                path=path,
                sha256=sha256,
                kind=kind,
                embedding=(embedding_model, embedding_version),
                display_name=display_name,
                now=now,
            )
            wanted = {chunk.evidence_id: chunk for chunk in chunks}
            rows = {
                row.id: row
                for row in session.query(EvidenceChunkModel)
                .filter(
                    (EvidenceChunkModel.document_id == document.id)
                    | EvidenceChunkModel.id.in_(list(wanted))
                )
                .all()
            }
            retired, deleted = _drop_unwanted(session, user_id, rows, wanted, now)
            added, kept = _place_wanted(
                session, document, rows, wanted, (embedding_model, embedding_version), now
            )
            session.commit()
            return IngestionResult(
                status=status,
                document=_indexed(session, document),
                added=added,
                kept=kept,
                retired=retired,
                deleted=deleted,
            )
        except IntegrityError as exc:
            session.rollback()
            raise Conflict(
                f"evidence from {path!r} collides with evidence already on record "
                f"for this candidate"
            ) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _record_document(
    session: Session,
    *,
    user_id: UUID,
    path: str,
    sha256: str,
    kind: DocumentKind,
    embedding: tuple[str, str],
    display_name: str | None,
    now: datetime,
) -> tuple[DocumentModel, IngestionStatus]:
    """Create or update the `documents` row (and the user it belongs to)."""
    if session.get(UserModel, user_id) is None:
        session.add(UserModel(id=user_id, display_name=display_name))
        session.flush()
    document = (
        session.query(DocumentModel)
        .filter(DocumentModel.user_id == user_id, DocumentModel.path == path)
        .one_or_none()
    )
    status = IngestionStatus.INDEXED if document is None else IngestionStatus.UPDATED
    if document is None:
        document = DocumentModel(user_id=user_id, path=path, created_at=now)
        session.add(document)
    document.sha256 = sha256
    document.kind = kind.value
    document.embedding_model, document.embedding_version = embedding
    document.indexed_at = now
    document.updated_at = now
    session.flush()
    return document, status


def _drop_unwanted(
    session: Session,
    user_id: UUID,
    rows: dict[UUID, EvidenceChunkModel],
    wanted: dict[UUID, EmbeddedChunk],
    now: datetime,
) -> tuple[int, int]:
    """Retire the active chunks the new version dropped if cited, else delete them."""
    dropped = [row for row in rows.values() if row.id not in wanted and row.retired_at is None]
    cited = _cited_evidence_ids(session, user_id) if dropped else set()
    retired = deleted = 0
    for row in dropped:
        if str(row.id) in cited:
            row.retired_at = now
            row.updated_at = now
            retired += 1
        else:
            session.delete(row)
            deleted += 1
    session.flush()
    return retired, deleted


def _place_wanted(
    session: Session,
    document: DocumentModel,
    rows: dict[UUID, EvidenceChunkModel],
    wanted: dict[UUID, EmbeddedChunk],
    embedding: tuple[str, str],
    now: datetime,
) -> tuple[int, int]:
    """Insert the new chunks and move the kept ones to their new positions."""
    # Park the chunks being kept off every real position first: two of them
    # trading places would otherwise collide on the unique position index
    # halfway through the update.
    kept_rows = [rows[evidence_id] for evidence_id in wanted if evidence_id in rows]
    for offset, row in enumerate(kept_rows):
        row.chunk_index = -1 - offset
    session.flush()

    added = kept = 0
    for chunk in wanted.values():
        row = rows.get(chunk.evidence_id)
        if row is None:
            if chunk.embedding is None:
                raise ValueError(f"new chunk {chunk.evidence_id} has no embedding")
            row = EvidenceChunkModel(
                id=chunk.evidence_id,
                user_id=document.user_id,
                chunk_text=redact_text(chunk.text),
                created_at=now,
            )
            session.add(row)
            added += 1
        elif row.retired_at is not None:
            added += 1
        else:
            kept += 1

        if chunk.embedding is not None:
            row.embedding = chunk.embedding
            row.embedding_model, row.embedding_version = embedding
        elif (row.embedding_model, row.embedding_version) != embedding:
            raise ValueError(
                f"chunk {chunk.evidence_id} has no embedding from {embedding[0]}/{embedding[1]}"
            )
        row.retired_at = None
        row.document_id = document.id
        row.source_type = chunk.source_type.value
        row.source_ref = chunk.source_ref
        row.chunk_index = chunk.chunk_index
        row.updated_at = now
    return added, kept


def _cited_evidence_ids(session: Session, user_id: UUID) -> set[str]:
    """Every evidence ref any of the candidate's artifact versions cites."""
    rows = (
        session.query(ArtifactVersionModel.evidence)
        .join(ApplicationModel, ApplicationModel.id == ArtifactVersionModel.application_id)
        .filter(ApplicationModel.user_id == user_id)
        .all()
    )
    return {
        str(item["ref"])
        for (evidence,) in rows
        for item in (evidence or [])
        if isinstance(item, dict) and item.get("ref")
    }


def _indexed(session: Session, document: DocumentModel) -> IndexedDocument:
    chunks = (
        session.query(EvidenceChunkModel)
        .filter(
            EvidenceChunkModel.document_id == document.id,
            EvidenceChunkModel.retired_at.is_(None),
        )
        .order_by(EvidenceChunkModel.chunk_index)
        .all()
    )
    return IndexedDocument(
        document_id=document.id,
        user_id=document.user_id,
        path=document.path,
        sha256=document.sha256,
        kind=DocumentKind(document.kind),
        embedding_model=document.embedding_model,
        embedding_version=document.embedding_version,
        indexed_at=document.indexed_at,
        evidence=tuple(
            IndexedEvidence(
                evidence_id=chunk.id,
                source_type=chunk.source_type,
                source_ref=chunk.source_ref,
                chunk_index=chunk.chunk_index,
                embedding_model=chunk.embedding_model,
                embedding_version=chunk.embedding_version,
            )
            for chunk in chunks
        ),
    )


__all__ = ["SqlEvidenceSource", "SqlEvidenceIndex", "SqlEvidenceIngestionStore"]
