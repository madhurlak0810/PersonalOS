"""Re-embedding: which vector tables there are, and what a backfill did to them.

Vectors from different embedding models are not comparable, so changing the
embedder means embedding everything again (`personalos.retrieval.reembedding`).
The three vector tables do this in two different ways:

- **evidence** (`evidence_chunks`): one vector per chunk, overwritten in place.
  The row's id is the evidence id an `artifact_versions` row cites, so the
  row is kept and only its vector, model and version change. Every citation
  still resolves, retired chunks included.
- **postings** and **messages** (`job_posting_embeddings`,
  `message_embeddings`): one row per (source, model, version). A new row is
  added for the new model, and the old one is left for anything still
  querying under the old model.

Either way, a row that is already at the target model and version is not
pending. That is what makes the backfill resumable: after a crash it is run
again, and it skips everything an earlier run committed.
"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VectorTable(str, Enum):
    """The tables that store embeddings, by the name `cli reembed --only` takes."""

    EVIDENCE = "evidence"
    POSTINGS = "postings"
    MESSAGES = "messages"


@dataclass(frozen=True)
class PendingVector:
    """One thing that has no vector from the target model yet.

    `key` is the row the vector is written to (for evidence, the chunk's
    evidence id) or the source it is written for (a posting, a communication
    event). `text` is what gets embedded.
    """

    key: UUID
    text: str


class TableReembedReport(BaseModel):
    """What one table's backfill found and did."""

    model_config = ConfigDict(frozen=True)

    table: VectorTable
    #: Rows not yet at the target model and version when the run started.
    pending: int = 0
    #: Vectors written by this run.
    reembedded: int = 0
    #: Rows with no text to embed. They stay pending and are reported again.
    skipped: int = 0
    #: Batches committed by this run.
    batches: int = 0


class ReembedReport(BaseModel):
    """What a backfill run did, per table."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    embedding_model: str
    embedding_version: str
    dry_run: bool = False
    tables: list[TableReembedReport] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether every pending row with text now has a vector from the target model."""
        return not self.dry_run and all(
            table.reembedded + table.skipped >= table.pending for table in self.tables
        )


__all__ = ["VectorTable", "PendingVector", "TableReembedReport", "ReembedReport"]
