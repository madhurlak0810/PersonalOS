"""Re-embed stored vectors with a new embedder: `cli reembed`.

Vectors are keyed by model, so retrieval under a new embedder sees nothing
until what is stored has been embedded again. `Reembedder.run` does that for
resume and project evidence, job postings and recruiter messages, one table
at a time:

1. list a batch of rows that have no vector from the target `model`/`version`
   (`ReembedStore.pending`, in id order);
2. embed their text, in one request where the embedder has `embed_many`;
3. write the batch in one transaction (`ReembedStore.write`).

It is resumable because of step 1. A row that is already at the target is not
pending, so a run that crashed is simply run again, and it starts where the
last committed batch ended. Evidence is overwritten in place, so every
evidence id an `artifact_versions` row cites stays the same
(`personalos.domain.embeddings`).

Run it with the new embedder before (or right after) switching
`EMBEDDING_PROVIDER`. Retrieval under the new model only finds rows that
have been re-embedded.
"""

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol
from uuid import UUID

from personalos.domain.embeddings import (
    PendingVector,
    ReembedReport,
    TableReembedReport,
    VectorTable,
)
from personalos.domain.errors import ValidationFailed
from personalos.persistence.models import EMBEDDING_DIMENSION
from personalos.retrieval.artifact_prep import Embedder

logger = logging.getLogger(__name__)

#: Rows embedded and committed together.
DEFAULT_BATCH_SIZE = 64


class ReembedStore(Protocol):
    """The vector tables, as the backfill reads and writes them."""

    def count_pending(self, table: VectorTable, *, model: str, version: str) -> int:
        """How many rows of `table` have no vector from `model`/`version` yet."""
        ...

    def pending(
        self,
        table: VectorTable,
        *,
        model: str,
        version: str,
        after: UUID | None,
        limit: int,
    ) -> Sequence[PendingVector]:
        """Up to `limit` pending rows with keys after `after`, in key order."""
        ...

    def write(
        self,
        table: VectorTable,
        vectors: Sequence[tuple[UUID, list[float]]],
        *,
        model: str,
        version: str,
    ) -> int:
        """Store one batch in one transaction; returns how many were written."""
        ...


class Reembedder:
    """Moves every stored vector onto one embedder, batch by batch."""

    def __init__(
        self,
        embedder: Embedder,
        store: ReembedStore,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        """Bind the target embedder and the store."""
        if embedder is None or store is None:
            raise ValueError("Reembedder requires an embedder and a store")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size
        self.dimension = dimension

    async def run(
        self,
        tables: Iterable[VectorTable] = tuple(VectorTable),
        *,
        dry_run: bool = False,
    ) -> ReembedReport:
        """Re-embed every pending row of `tables`; with `dry_run`, only count them."""
        model, version = self.embedder.model, self.embedder.version
        reports = []
        for table in dict.fromkeys(tables):
            pending = self.store.count_pending(table, model=model, version=version)
            if dry_run or not pending:
                reports.append(TableReembedReport(table=table, pending=pending))
                continue
            reports.append(await self._run_table(table, pending, model, version))
        return ReembedReport(
            embedding_model=model, embedding_version=version, dry_run=dry_run, tables=reports
        )

    async def _run_table(
        self, table: VectorTable, pending: int, model: str, version: str
    ) -> TableReembedReport:
        reembedded = skipped = batches = 0
        after: UUID | None = None
        while True:
            rows = list(
                self.store.pending(
                    table, model=model, version=version, after=after, limit=self.batch_size
                )
            )
            if not rows:
                break
            after = rows[-1].key
            embeddable = [row for row in rows if row.text.strip()]
            skipped += len(rows) - len(embeddable)
            if not embeddable:
                continue
            vectors = await self._embed([row.text for row in embeddable])
            reembedded += self.store.write(
                table,
                [(row.key, vector) for row, vector in zip(embeddable, vectors, strict=True)],
                model=model,
                version=version,
            )
            batches += 1
            logger.info(
                "re-embedded %s/%s %s row(s) with %s@%s",
                reembedded,
                pending,
                table.value,
                model,
                version,
            )
        return TableReembedReport(
            table=table, pending=pending, reembedded=reembedded, skipped=skipped, batches=batches
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        embed_many = getattr(self.embedder, "embed_many", None)
        if embed_many is not None:
            vectors = [list(vector) for vector in await embed_many(texts)]
        else:
            vectors = [list(await self.embedder.embed(text)) for text in texts]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValidationFailed(
                    f"embedder {self.embedder.model} returned {len(vector)} dimensions; "
                    f"the vector columns store {self.dimension}"
                )
        return vectors


__all__ = ["DEFAULT_BATCH_SIZE", "ReembedStore", "Reembedder"]
