"""Load a resume into the evidence index that job matching and artifact prep cite.

`ResumeIngestor.ingest` takes a path and a candidate and

1. **reads** the file through the `DocumentReader` port -- in production the
   sandboxed files tool, so a path outside the allowed roots is refused before
   a byte is read;
2. **stops** if the document on record for that path has the same SHA-256 and
   was embedded by the same embedder model and version: an unchanged file is a
   no-op, down to `indexed_at`;
3. **chunks** it (`personalos.retrieval.resume_chunking`) and gives each chunk
   its stable evidence id (`personalos.domain.evidence.evidence_id_for`);
4. **embeds** only the chunks that have no vector from this embedder yet, so
   an edit to one bullet costs one embedding call, not one per chunk;
5. **syncs** the document's chunks to the new set through the
   `EvidenceIngestionStore` port, which keeps the rows of unchanged chunks and
   never deletes one an artifact version cites.

Every vector row records the embedder's `model` and `version`. Retrieval
pins the model, so whatever embedder ingests the evidence must be the one
artifact prep embeds postings with -- `personalos.bootstrap.build_embedder`
is where both get theirs.
"""

import logging
from collections.abc import Collection, Sequence
from pathlib import PurePath
from typing import Protocol
from uuid import UUID

from personalos.domain.errors import ValidationFailed
from personalos.domain.evidence import (
    DocumentKind,
    EmbeddedChunk,
    IndexedDocument,
    IngestionResult,
    IngestionStatus,
    SourceDocument,
    evidence_id_for,
)
from personalos.persistence.models import EMBEDDING_DIMENSION
from personalos.retrieval.artifact_prep import Embedder
from personalos.retrieval.resume_chunking import chunk_resume, detect_kind, extract_text

logger = logging.getLogger(__name__)


class DocumentReader(Protocol):
    """Reads one file the candidate named, inside whatever bounds the deployment sets."""

    async def read(self, path: str) -> SourceDocument:
        """The file's canonical path, bytes and SHA-256."""
        ...


class EvidenceIngestionStore(Protocol):
    """Where ingested documents and their chunks are kept."""

    def document(self, user_id: UUID, path: str) -> IndexedDocument | None:
        """The document on record at `path`, if any."""
        ...

    def embedded(
        self,
        user_id: UUID,
        evidence_ids: Collection[UUID],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> set[UUID]:
        """Which of `evidence_ids` already have a vector from this model and version."""
        ...

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
        """Make `chunks` the document's evidence and record the document."""
        ...


class ResumeIngestor:
    """Reads, chunks, embeds and indexes one resume file at a time."""

    def __init__(
        self,
        *,
        reader: DocumentReader,
        embedder: Embedder,
        store: EvidenceIngestionStore,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        """Bind the reader, the embedder and the store.

        `dimension` is the width every vector must have: the
        `evidence_chunks.embedding` column's, unless a test says otherwise.
        """
        if reader is None or embedder is None or store is None:
            raise ValueError("ResumeIngestor requires a reader, an embedder and a store")
        self.reader = reader
        self.embedder = embedder
        self.store = store
        self.dimension = dimension

    async def ingest(
        self, user_id: UUID, path: str, *, display_name: str | None = None
    ) -> IngestionResult:
        """Index the resume at `path` as `user_id`'s evidence; a no-op if nothing changed."""
        source = await self.reader.read(path)
        model, version = self.embedder.model, self.embedder.version

        previous = self.store.document(user_id, source.path)
        if (
            previous is not None
            and previous.sha256 == source.sha256
            and (previous.embedding_model, previous.embedding_version) == (model, version)
        ):
            logger.info("resume %s is unchanged since %s", source.path, previous.indexed_at)
            return IngestionResult(status=IngestionStatus.UNCHANGED, document=previous)

        kind = detect_kind(source.path, source.content)
        chunks = chunk_resume(
            extract_text(source.content, kind), document_name=PurePath(source.path).name
        )
        if not chunks:
            raise ValidationFailed(f"no text could be read from {PurePath(source.path).name!r}")

        ids = [evidence_id_for(user_id, source.path, chunk.text) for chunk in chunks]
        have = self.store.embedded(user_id, ids, embedding_model=model, embedding_version=version)
        embedded: list[EmbeddedChunk] = []
        for index, (evidence_id, chunk) in enumerate(zip(ids, chunks, strict=True)):
            vector = None
            if evidence_id not in have:
                vector = list(await self.embedder.embed(chunk.text))
                if len(vector) != self.dimension:
                    raise ValidationFailed(
                        f"embedder {model} returned {len(vector)} dimensions; "
                        f"the evidence index stores {self.dimension}"
                    )
            embedded.append(
                EmbeddedChunk(
                    evidence_id=evidence_id,
                    kind=chunk.kind,
                    source_ref=chunk.source_ref,
                    chunk_index=index,
                    text=chunk.text,
                    embedding=vector,
                )
            )

        result = self.store.sync(
            user_id=user_id,
            path=source.path,
            sha256=source.sha256,
            kind=kind,
            embedding_model=model,
            embedding_version=version,
            chunks=embedded,
            display_name=display_name,
        )
        logger.info(
            "resume %s %s: %s added, %s kept, %s retired, %s deleted",
            source.path,
            result.status.value,
            result.added,
            result.kept,
            result.retired,
            result.deleted,
        )
        return result


__all__ = ["DocumentReader", "EvidenceIngestionStore", "ResumeIngestor"]
