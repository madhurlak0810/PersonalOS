"""Test job search functionality."""

import pytest

from personalos.domain.models import Job, JobStatus


@pytest.fixture
def mock_session():
    """Mock database session."""
    # This is a placeholder - in a real test, you'd use SQLite or a test DB
    class MockSession:
        pass

    return MockSession()


def test_job_creation():
    """Test creating a job search."""
    job = Job(
        title="Search Python Developer Jobs",
        keywords=["Python", "Developer"],
        locations=["New York", "Remote"],
        salary_min=80000,
        salary_max=120000,
        job_type="full-time",
    )

    assert job.title == "Search Python Developer Jobs"
    assert job.status == JobStatus.PENDING
    assert job.keywords == ["Python", "Developer"]
    assert len(job.locations) == 2
    assert job.salary_min == 80000
    assert job.created_at is not None


def test_job_status_transitions():
    """Test job status transitions."""
    job = Job(
        title="Test Job",
        keywords=["test"],
        locations=["Remote"],
    )

    # Initial status
    assert job.status == JobStatus.PENDING

    # Transition to running
    job.status = JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING

    # Transition to completed
    job.status = JobStatus.COMPLETED
    assert job.status == JobStatus.COMPLETED


def test_job_with_results():
    """Test job with results."""
    job = Job(
        title="Test Job",
        keywords=["test"],
        locations=["Remote"],
    )

    results = [
        {
            "title": "Python Developer",
            "company": "TechCorp",
            "location": "Remote",
            "salary": "100000-130000",
            "relevance_score": 0.95,
        },
        {
            "title": "Senior Python Developer",
            "company": "StartupXYZ",
            "location": "Remote",
            "salary": "120000-150000",
            "relevance_score": 0.92,
        },
    ]

    job.results = {"matches": results}
    job.results_count = len(results)
    job.status = JobStatus.COMPLETED

    assert job.results_count == 2
    assert len(job.results["matches"]) == 2
    assert job.status == JobStatus.COMPLETED
