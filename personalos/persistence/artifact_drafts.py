"""Tailored drafts as `artifact_versions` rows.

A draft is stored the moment it has been validated, before anyone has decided
whether to send it: it is a row in this system's own database, superseded by
the next version rather than overwritten, which is what makes creating one
reversible. Each row's `evidence` lists the evidence ids the draft was written
from, so "what is this sentence based on" is answerable from the row alone.

`artifact_versions.application_id` is not nullable, so a draft needs an
application to hang from. One is created in `DISCOVERED` if the candidate has
none for this posting yet; its status is left alone otherwise. Moving an
application through its lifecycle is `ApplicationRepository.update_status`'s
job, and storing a draft is not a lifecycle event.
"""

from collections.abc import Callable, Sequence
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.job_search import ArtifactDraft, NormalizedPosting
from personalos.domain.models import validate_evidence_links
from personalos.persistence.job_postings import SqlPostingCatalog
from personalos.persistence.models import ApplicationModel, ArtifactVersionModel


class SqlArtifactDraftStore:
    """Binds artifact prep's `DraftStore` port to the `artifact_versions` table."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory
        self._postings = SqlPostingCatalog(session_factory)

    async def save(
        self,
        *,
        user_id: UUID,
        posting: NormalizedPosting,
        drafts: Sequence[ArtifactDraft],
        generated_by: str,
    ) -> list[ArtifactDraft]:
        """Store each draft as the next version of its type; return them with row ids.

        Storing the same draft twice returns the row it already has. A run
        that is retried after this step regenerates nothing new, and should
        not leave a second identical version behind for having tried.
        """
        (persisted,) = await self._postings.record([posting])
        session = self.session_factory()
        try:
            application = self._application(session, persisted.job_posting_id, user_id)
            stored: list[ArtifactDraft] = []
            for draft in drafts:
                evidence = validate_evidence_links(
                    [ref.model_dump(mode="json") for ref in draft.evidence]
                )
                latest = (
                    session.query(ArtifactVersionModel)
                    .filter(
                        ArtifactVersionModel.application_id == application.id,
                        ArtifactVersionModel.artifact_type == draft.artifact_type.value,
                    )
                    .order_by(ArtifactVersionModel.version.desc())
                    .first()
                )
                if (
                    latest is not None
                    and latest.content == draft.content
                    and latest.evidence == evidence
                ):
                    row = latest
                else:
                    row = ArtifactVersionModel(
                        application_id=application.id,
                        artifact_type=draft.artifact_type.value,
                        version=(latest.version + 1) if latest is not None else 1,
                        content=draft.content,
                        evidence=evidence,
                        generated_by=generated_by[:255],
                    )
                    session.add(row)
                    session.flush()
                stored.append(draft.model_copy(update={"artifact_version_id": row.id}))
            session.commit()
            return stored
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _application(session: Session, job_posting_id: UUID, user_id: UUID) -> ApplicationModel:
        """The user's application for this posting, created if there is none."""

        def existing() -> ApplicationModel | None:
            return (
                session.query(ApplicationModel)
                .filter(
                    ApplicationModel.job_posting_id == job_posting_id,
                    ApplicationModel.user_id == user_id,
                )
                .first()
            )

        application = existing()
        if application is not None:
            return application
        application = ApplicationModel(job_posting_id=job_posting_id, user_id=user_id)
        session.add(application)
        try:
            session.commit()
        except IntegrityError:
            # Lost the insert race on (job_posting_id, user_id); use the winner's row.
            session.rollback()
            application = existing()
            if application is None:  # pragma: no cover - a unique violation implies a row
                raise
        return application


__all__ = ["SqlArtifactDraftStore"]
