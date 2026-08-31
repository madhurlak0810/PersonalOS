"""Tests for JobRepository against a real database session."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.domain.models import Job, JobStatus
from personalos.persistence.models import Base
from personalos.persistence.repositories import JobRepository


@pytest.fixture
def session(tmp_path):
    """Real SQLAlchemy session over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def test_create_round_trips_job(session):
    """A created job comes back with its fields intact."""
    repo = JobRepository(session)
    job = Job(
        title="Python Developer",
        description="Find Python roles",
        keywords=["python", "fastapi"],
        locations=["Remote"],
        salary_min=120000,
        job_type="full-time",
    )

    created = repo.create(job)
    fetched = repo.get_by_id(job.id)

    assert created.id == job.id
    assert fetched is not None
    assert fetched.title == "Python Developer"
    assert fetched.keywords == ["python", "fastapi"]
    assert fetched.status == JobStatus.PENDING


def test_metadata_is_persisted(session):
    """Job metadata must actually reach the database.

    The column is `job_metadata` because `metadata` is reserved on SQLAlchemy's
    declarative base; assigning `metadata` silently discarded the value and read
    back SQLAlchemy's own MetaData object.
    """
    repo = JobRepository(session)
    job = Job(title="Job with metadata", metadata={"source": "cli", "attempt": 2})

    created = repo.create(job)
    assert created.metadata == {"source": "cli", "attempt": 2}

    session.expire_all()
    fetched = repo.get_by_id(job.id)
    assert fetched.metadata == {"source": "cli", "attempt": 2}


def test_update_persists_metadata_and_results(session):
    """Updating a job persists metadata, results and status."""
    repo = JobRepository(session)
    job = Job(title="Job to update", metadata={"stage": "initial"})
    repo.create(job)

    job.status = JobStatus.COMPLETED
    job.results_count = 3
    job.results = {"jobs": ["a", "b", "c"]}
    job.metadata = {"stage": "final", "runs": 1}
    repo.update(job)

    session.expire_all()
    fetched = repo.get_by_id(job.id)

    assert fetched.status == JobStatus.COMPLETED
    assert fetched.results_count == 3
    assert fetched.results == {"jobs": ["a", "b", "c"]}
    assert fetched.metadata == {"stage": "final", "runs": 1}


def test_empty_metadata_defaults_to_dict(session):
    """A job with no metadata reads back as an empty dict, not a MetaData object."""
    repo = JobRepository(session)
    job = Job(title="No metadata")
    repo.create(job)

    session.expire_all()
    fetched = repo.get_by_id(job.id)

    assert fetched.metadata == {}


def test_delete_removes_job(session):
    """Deleting a job removes it; deleting again reports failure."""
    repo = JobRepository(session)
    job = Job(title="Job to delete")
    repo.create(job)

    assert repo.delete(job.id) is True
    assert repo.get_by_id(job.id) is None
    assert repo.delete(job.id) is False


def test_update_unknown_job_raises(session):
    """Updating a job that was never created is an error."""
    repo = JobRepository(session)

    with pytest.raises(ValueError):
        repo.update(Job(title="Never created"))
