"""Reply drafting's two tables: the application a reply is about, and where the reply is kept.

`SqlReplyApplicationLookup` reads the application a recruiter's message was
tied to, with its posting's company, role and skills -- what a reply may name
without citing evidence, and what it may not claim without citing it.

`SqlReplyDraftStore` keeps a reply as the next `recruiter_reply` version of
the application's `artifact_versions`, with the evidence it cites and the
message it answers in `evidence`. Storing the same reply twice returns the
row it already has.
"""

from collections.abc import Callable, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.job_search import EvidenceRef, JobSearchContractError
from personalos.domain.models import ArtifactType, validate_evidence_links
from personalos.domain.outbound_email import ReplyApplication
from personalos.persistence.models import ApplicationModel, ArtifactVersionModel, JobPostingModel


class SqlReplyApplicationLookup:
    """`applications` joined to `job_postings`, as a `ReplyApplication`."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    async def lookup(self, application_id: UUID) -> ReplyApplication:
        """The application, or `JobSearchContractError` if there is none."""
        session = self.session_factory()
        try:
            row = (
                session.query(ApplicationModel, JobPostingModel)
                .join(JobPostingModel, JobPostingModel.id == ApplicationModel.job_posting_id)
                .filter(ApplicationModel.id == application_id)
                .one_or_none()
            )
            if row is None:
                raise JobSearchContractError(f"no application {application_id} on file")
            application, posting = row
            return ReplyApplication(
                application_id=application.id,
                user_id=application.user_id,
                company=posting.company,
                title=posting.title,
                location=posting.location,
                skills=tuple(
                    str(skill) for skill in (posting.normalized_json or {}).get("skills") or ()
                ),
            )
        finally:
            session.close()


class SqlReplyDraftStore:
    """`ReplyDraftStore` over `artifact_versions`."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    async def save(
        self,
        *,
        application_id: UUID,
        content: str,
        evidence: Sequence[EvidenceRef],
        generated_by: str,
    ) -> UUID:
        """Store the reply as the next version; return the row id."""
        links = validate_evidence_links([ref.model_dump(mode="json") for ref in evidence])
        kind = ArtifactType.RECRUITER_REPLY.value
        session = self.session_factory()
        try:
            latest = (
                session.query(ArtifactVersionModel)
                .filter(
                    ArtifactVersionModel.application_id == application_id,
                    ArtifactVersionModel.artifact_type == kind,
                )
                .order_by(ArtifactVersionModel.version.desc())
                .first()
            )
            if latest is not None and latest.content == content and latest.evidence == links:
                return latest.id
            row = ArtifactVersionModel(
                application_id=application_id,
                artifact_type=kind,
                version=(latest.version + 1) if latest is not None else 1,
                content=content,
                evidence=links,
                generated_by=generated_by[:255],
            )
            session.add(row)
            session.commit()
            return row.id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


__all__ = ["SqlReplyApplicationLookup", "SqlReplyDraftStore"]
