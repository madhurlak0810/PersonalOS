"""Tests for the job-search domain schema: postings, profiles, applications,
artifacts, and recruiter-side communication events.

Covers the acceptance criteria from the schema migrations: the tables are
created with the documented constraints (dedupe uniqueness on job_postings,
FK/uniqueness on the rest), an application's status can only move along the
documented lifecycle — never set directly to an arbitrary state — and a
communication_events row is queryable by its application_id.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from datetime import datetime

from personalos.domain.models import (
    ApplicationStatus,
    CommunicationEventClassification,
    InvalidApplicationTransition,
    InvalidEvidenceLinkage,
    validate_application_status_transition,
    validate_evidence_links,
)
from personalos.persistence.models import Base
from personalos.persistence.repositories import (
    ApplicationRepository,
    ArtifactVersionRepository,
    CandidateProfileRepository,
    CommunicationEventRepository,
    JobPostingRepository,
)


def _open(db_path):
    """A fresh engine + session over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory(), engine


# ----------------------------------------------------------------------
# Domain-level transition validation: no database involved.
# ----------------------------------------------------------------------


def test_discovered_to_offer_is_rejected():
    """The documented lifecycle has no direct DISCOVERED -> OFFER edge."""
    with pytest.raises(InvalidApplicationTransition):
        validate_application_status_transition(
            ApplicationStatus.DISCOVERED, ApplicationStatus.OFFER
        )


@pytest.mark.parametrize(
    "current,new",
    [
        (ApplicationStatus.DISCOVERED, ApplicationStatus.SAVED),
        (ApplicationStatus.SAVED, ApplicationStatus.PREPARING),
        (ApplicationStatus.PREPARING, ApplicationStatus.READY_TO_APPLY),
        (ApplicationStatus.READY_TO_APPLY, ApplicationStatus.APPLIED),
        (ApplicationStatus.APPLIED, ApplicationStatus.INTERVIEWING),
        (ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER),
        (ApplicationStatus.INTERVIEWING, ApplicationStatus.REJECTED),
        (ApplicationStatus.APPLIED, ApplicationStatus.WITHDRAWN),
        (ApplicationStatus.DISCOVERED, ApplicationStatus.SKIPPED),
    ],
)
def test_documented_transitions_are_accepted(current, new):
    """Every edge on the documented lifecycle is accepted."""
    assert validate_application_status_transition(current, new) == new


@pytest.mark.parametrize(
    "current,new",
    [
        (ApplicationStatus.REJECTED, ApplicationStatus.APPLIED),
        (ApplicationStatus.WITHDRAWN, ApplicationStatus.DISCOVERED),
        (ApplicationStatus.SKIPPED, ApplicationStatus.SAVED),
        (ApplicationStatus.APPLIED, ApplicationStatus.DISCOVERED),
        (ApplicationStatus.DISCOVERED, ApplicationStatus.DISCOVERED),
    ],
)
def test_undocumented_transitions_are_rejected(current, new):
    """Skipping steps, reversing, or re-asserting the current status all fail."""
    with pytest.raises(InvalidApplicationTransition):
        validate_application_status_transition(current, new)


def test_evidence_links_require_at_least_one_entry():
    """An artifact version cannot be generated with no cited evidence."""
    with pytest.raises(InvalidEvidenceLinkage):
        validate_evidence_links([])


def test_evidence_links_require_type_and_ref():
    """Each evidence entry must identify what it points at, not just its kind."""
    with pytest.raises(InvalidEvidenceLinkage):
        validate_evidence_links([{"type": "resume_section"}])


def test_evidence_links_pass_through_when_valid():
    evidence = [{"type": "project", "ref": "personalos-job-search"}]
    assert validate_evidence_links(evidence) == evidence


# ----------------------------------------------------------------------
# Schema + repository behavior against a real (SQLite) database.
# ----------------------------------------------------------------------


def test_job_posting_dedupe_key_is_unique(tmp_path):
    """Inserting two postings with the same dedupe_key is rejected by the schema."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        repo = JobPostingRepository(session)
        repo.create(
            source="linkedin",
            title="Staff Engineer",
            company="Acme",
            description_hash="a" * 64,
            dedupe_key="acme:staff-engineer:a" * 3,
        )
        with pytest.raises(IntegrityError):
            repo.create(
                source="indeed",
                title="Staff Engineer (re-posted)",
                company="Acme",
                description_hash="a" * 64,
                dedupe_key="acme:staff-engineer:a" * 3,
            )
    finally:
        session.close()
        engine.dispose()


def test_application_status_update_rejects_invalid_transition(tmp_path):
    """ApplicationRepository.update_status refuses DISCOVERED -> OFFER end-to-end."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        from personalos.persistence.models import UserModel

        user = UserModel(email="candidate@example.com")
        session.add(user)
        session.commit()

        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Staff Engineer",
            company="Acme",
            description_hash="b" * 64,
            dedupe_key="acme:staff-engineer:b" * 3,
        )
        application = ApplicationRepository(session).create(
            job_posting_id=posting.id, user_id=user.id
        )
        assert application.status == ApplicationStatus.DISCOVERED.value

        with pytest.raises(InvalidApplicationTransition):
            ApplicationRepository(session).update_status(
                application.id, ApplicationStatus.OFFER
            )

        # The rejected attempt left the row untouched.
        refetched = ApplicationRepository(session).get_by_id(application.id)
        assert refetched.status == ApplicationStatus.DISCOVERED.value
    finally:
        session.close()
        engine.dispose()


def test_application_status_update_follows_documented_path(tmp_path):
    """A sequence of documented moves succeeds and is persisted."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        from personalos.persistence.models import UserModel

        user = UserModel(email="candidate2@example.com")
        session.add(user)
        session.commit()

        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Senior Engineer",
            company="Globex",
            description_hash="c" * 64,
            dedupe_key="globex:senior-engineer:c" * 3,
        )
        repo = ApplicationRepository(session)
        application = repo.create(job_posting_id=posting.id, user_id=user.id)

        for next_status in (
            ApplicationStatus.SAVED,
            ApplicationStatus.PREPARING,
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.APPLIED,
        ):
            application = repo.update_status(application.id, next_status)
            assert application.status == next_status.value

        assert application.applied_at is not None
    finally:
        session.close()
        engine.dispose()


def test_artifact_version_requires_evidence(tmp_path):
    """ArtifactVersionRepository.create refuses to write a row with no evidence."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        from personalos.persistence.models import UserModel

        user = UserModel(email="candidate3@example.com")
        session.add(user)
        session.commit()

        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Platform Engineer",
            company="Initech",
            description_hash="d" * 64,
            dedupe_key="initech:platform-engineer:d" * 3,
        )
        application = ApplicationRepository(session).create(
            job_posting_id=posting.id, user_id=user.id
        )

        with pytest.raises(InvalidEvidenceLinkage):
            ArtifactVersionRepository(session).create(
                application_id=application.id,
                artifact_type="resume",
                evidence=[],
            )

        artifact = ArtifactVersionRepository(session).create(
            application_id=application.id,
            artifact_type="resume",
            evidence=[{"type": "resume_section", "ref": "experience.personalos"}],
        )
        assert artifact.evidence == [
            {"type": "resume_section", "ref": "experience.personalos"}
        ]
    finally:
        session.close()
        engine.dispose()


def test_candidate_profile_versions_are_unique_per_user(tmp_path):
    """A user cannot have two profiles at the same profile_version."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        from personalos.persistence.models import UserModel

        user = UserModel(email="candidate4@example.com")
        session.add(user)
        session.commit()

        repo = CandidateProfileRepository(session)
        repo.create(user_id=user.id, profile_version=1, target_roles=["Staff Engineer"])
        with pytest.raises(IntegrityError):
            repo.create(user_id=user.id, profile_version=1, target_roles=["Principal Engineer"])
    finally:
        session.close()
        engine.dispose()


def test_communication_event_is_queryable_by_application_id(tmp_path):
    """Inserting an INTERVIEW_INVITE event makes it findable via its application_id."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        from personalos.persistence.models import UserModel

        user = UserModel(email="candidate5@example.com")
        session.add(user)
        session.commit()

        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Backend Engineer",
            company="Umbrella",
            description_hash="e" * 64,
            dedupe_key="umbrella:backend-engineer:e" * 3,
        )
        application = ApplicationRepository(session).create(
            job_posting_id=posting.id, user_id=user.id
        )

        occurred_at = datetime(2026, 9, 10, 14, 30)
        CommunicationEventRepository(session).create(
            application_id=application.id,
            classification=CommunicationEventClassification.INTERVIEW_INVITE.value,
            occurred_at=occurred_at,
            provider_message_id="msg-12345",
            metadata_json={"subject": "Interview availability?"},
        )

        events = CommunicationEventRepository(session).get_by_application_id(application.id)
        assert len(events) == 1
        event = events[0]
        assert event.application_id == application.id
        assert event.classification == CommunicationEventClassification.INTERVIEW_INVITE.value
        assert event.provider_message_id == "msg-12345"
        assert event.occurred_at == occurred_at
        assert event.metadata_json == {"subject": "Interview availability?"}
    finally:
        session.close()
        engine.dispose()


def test_communication_event_duplicate_provider_message_is_rejected(tmp_path):
    """The same provider_message_id for an application cannot be inserted twice."""
    session, engine = _open(tmp_path / "jobs.db")
    try:
        from personalos.persistence.models import UserModel

        user = UserModel(email="candidate6@example.com")
        session.add(user)
        session.commit()

        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Backend Engineer II",
            company="Umbrella",
            description_hash="f" * 64,
            dedupe_key="umbrella:backend-engineer-ii:f" * 3,
        )
        application = ApplicationRepository(session).create(
            job_posting_id=posting.id, user_id=user.id
        )

        repo = CommunicationEventRepository(session)
        repo.create(
            application_id=application.id,
            classification=CommunicationEventClassification.GENERAL_UPDATE.value,
            occurred_at=datetime(2026, 9, 10, 9, 0),
            provider_message_id="msg-dup",
        )
        with pytest.raises(IntegrityError):
            repo.create(
                application_id=application.id,
                classification=CommunicationEventClassification.REJECTION.value,
                occurred_at=datetime(2026, 9, 11, 9, 0),
                provider_message_id="msg-dup",
            )
    finally:
        session.close()
        engine.dispose()
