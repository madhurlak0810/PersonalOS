"""The durable catalog of discovered job postings.

Where cross-provider and cross-run dedupe is actually settled. The graph's
`deduplicate` node collapses duplicates *within* one run; this is what makes
the second run of the same search, or a second provider on a later day, land
on the row the first one created -- via the unique constraint on
`job_postings.dedupe_key`, not via a read-then-write that two workers could
both win.
"""

from collections.abc import Callable, Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.job_search import NormalizedPosting, PersistedPosting
from personalos.persistence.models import JobPostingModel


class SqlPostingCatalog:
    """Binds the graph's `PostingCatalog` port to the `job_postings` table.

    Takes a session factory rather than a session, as `PendingCheckpointStore`
    does: each posting is its own short transaction, so one losing an insert
    race rolls back only itself.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    async def record(self, postings: Sequence[NormalizedPosting]) -> list[PersistedPosting]:
        """Store each posting once, returning the row every one resolved to.

        The existing row wins whole: a posting seen again is not re-attributed
        to whichever provider returned it most recently.
        """
        session = self.session_factory()
        try:
            return [self._record_one(session, posting) for posting in postings]
        finally:
            session.close()

    def _record_one(self, session: Session, posting: NormalizedPosting) -> PersistedPosting:
        key = posting.dedupe_key
        existing = self._row_for_key(session, key)
        if existing is None:
            row = _to_row(posting)
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                # Lost the insert race on `dedupe_key`; the winner's row is the
                # one this posting resolves to.
                session.rollback()
                existing = self._row_for_key(session, key)
                if existing is None:  # pragma: no cover - a unique violation implies a row
                    raise
            else:
                return PersistedPosting(job_posting_id=row.id, dedupe_key=key, created=True)
        return PersistedPosting(job_posting_id=existing.id, dedupe_key=key, created=False)

    @staticmethod
    def _row_for_key(session: Session, dedupe_key: str) -> JobPostingModel | None:
        return (
            session.query(JobPostingModel).filter(JobPostingModel.dedupe_key == dedupe_key).first()
        )


def _to_row(posting: NormalizedPosting) -> JobPostingModel:
    """Flatten a posting into its row, clipped to the column widths.

    Clipped rather than rejected: the text is whatever a provider sent, and an
    over-long title should cost its tail, not the posting.
    """
    return JobPostingModel(
        source=posting.source[:100],
        source_job_id=posting.source_job_id[:255] if posting.source_job_id else None,
        title=posting.title[:500],
        company=posting.company[:255],
        location=posting.location[:255] if posting.location else None,
        url=posting.url,
        raw_json=posting.raw,
        normalized_json=posting.model_dump(mode="json", exclude={"raw"}),
        description_hash=posting.description_hash,
        dedupe_key=posting.dedupe_key,
        posted_at=posting.posted_at,
    )


__all__ = ["SqlPostingCatalog"]
