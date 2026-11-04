"""Resume ingestion: what a loaded document is, and what loading it changed.

A resume enters the evidence index as one `documents` row (where it was read
from, the SHA-256 of what was read, its format) and the `evidence_chunks` cut
from it. Each chunk's `evidence_id` is derived from the candidate, the
document's path and the chunk's text, not drawn at random, so cutting the same
text from the same file always names the same record:

- re-ingesting an unchanged file finds every id already on record and changes
  nothing;
- re-ingesting an edited file keeps the ids of the chunks that did not change,
  so a citation of an unedited bullet survives an edit elsewhere in the resume.

A chunk that an edit removes is deleted, unless an `artifact_versions` row
cites it: then it is *retired* -- kept, so the citation still resolves, but no
longer offered to job matching or artifact prep as evidence.
"""

import hashlib
import re
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from personalos.domain.models import EvidenceSourceType

#: Namespace for `evidence_id_for`. Fixed forever: changing it renames every
#: evidence record ever ingested and orphans every citation of one.
EVIDENCE_ID_NAMESPACE = UUID("6f1d3c2a-8e4b-5a7c-9d0e-1f2a3b4c5d6e")


class DocumentKind(str, Enum):
    """The formats a resume can be ingested from."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"


class ChunkKind(str, Enum):
    """What part of a resume a chunk was cut from."""

    #: A whole section with no finer structure: a summary, education.
    SECTION = "section"
    #: One position under an experience section.
    ROLE = "role"
    #: One project, from a projects section. Stored as `project` evidence.
    PROJECT = "project"
    #: A skills section, or one slice of a long one.
    SKILLS = "skills"


class IngestionStatus(str, Enum):
    """What an ingest did to the index."""

    #: The path had never been ingested for this candidate.
    INDEXED = "indexed"
    #: The path was on record with different content (or a different embedder).
    UPDATED = "updated"
    #: The path was on record with this exact content and embedder: a no-op.
    UNCHANGED = "unchanged"


def normalize_chunk_text(text: str) -> str:
    """Whitespace-insensitive form of a chunk, which its id is derived from."""
    return re.sub(r"\s+", " ", text).strip()


def evidence_id_for(user_id: UUID, document_path: str, text: str) -> UUID:
    """The stable id of the chunk with this text, in this document, for this candidate."""
    digest = hashlib.sha256(normalize_chunk_text(text).encode("utf-8")).hexdigest()
    return uuid5(EVIDENCE_ID_NAMESPACE, f"{user_id}\n{document_path}\n{digest}")


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResumeChunk(_Value):
    """One piece of a resume, cut and labelled, before it is embedded."""

    kind: ChunkKind
    #: Where in the document it came from, e.g. `resume.md#experience/acme-corp`.
    source_ref: str
    text: str

    @property
    def source_type(self) -> EvidenceSourceType:
        """Projects are `project` evidence; everything else is `resume` evidence."""
        return (
            EvidenceSourceType.PROJECT
            if self.kind == ChunkKind.PROJECT
            else EvidenceSourceType.RESUME
        )


class EmbeddedChunk(_Value):
    """A chunk with its stable id, position and vector: one `evidence_chunks` row."""

    evidence_id: UUID
    kind: ChunkKind
    source_ref: str
    chunk_index: int = Field(ge=0)
    text: str
    #: None when the row on record already has this text embedded by the
    #: same model and version, so the vector it has is kept.
    embedding: list[float] | None = None

    @property
    def source_type(self) -> EvidenceSourceType:
        """As `ResumeChunk.source_type`."""
        return (
            EvidenceSourceType.PROJECT
            if self.kind == ChunkKind.PROJECT
            else EvidenceSourceType.RESUME
        )


class SourceDocument(_Value):
    """A file as read through the sandboxed files tool."""

    #: Canonical path, as the files sandbox resolved it.
    path: str
    content: bytes
    sha256: str


class IndexedEvidence(_Value):
    """One active evidence record of an ingested document."""

    evidence_id: UUID
    source_type: EvidenceSourceType
    source_ref: str | None
    chunk_index: int
    embedding_model: str
    embedding_version: str


class IndexedDocument(_Value):
    """A `documents` row and the evidence currently cut from it."""

    document_id: UUID
    user_id: UUID
    path: str
    sha256: str
    kind: DocumentKind
    embedding_model: str
    embedding_version: str
    indexed_at: datetime
    evidence: tuple[IndexedEvidence, ...] = ()


class IngestionResult(_Value):
    """What one ingest of one document did."""

    status: IngestionStatus
    document: IndexedDocument
    #: Chunks inserted, or brought back from retirement.
    added: int = 0
    #: Chunks already on record that the new version still contains.
    kept: int = 0
    #: Chunks the new version dropped that an artifact version cites: kept, inactive.
    retired: int = 0
    #: Chunks the new version dropped that nothing cites.
    deleted: int = 0


__all__ = [
    "EVIDENCE_ID_NAMESPACE",
    "DocumentKind",
    "ChunkKind",
    "IngestionStatus",
    "normalize_chunk_text",
    "evidence_id_for",
    "ResumeChunk",
    "EmbeddedChunk",
    "SourceDocument",
    "IndexedEvidence",
    "IndexedDocument",
    "IngestionResult",
]
