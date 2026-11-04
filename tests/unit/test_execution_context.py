"""Context continuity: one `ExecutionContext` should reach every hop.

These tests pin the traceability contract: the workflow_id, run_id,
correlation_id, and actor_id set on a `Job` must round-trip through
persistence unchanged.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.domain.context import ExecutionContext
from personalos.domain.models import Job
from personalos.persistence.models import Base
from personalos.persistence.repositories import JobRepository

# ----------------------------------------------------------------------
# ExecutionContext itself
# ----------------------------------------------------------------------


def test_new_context_mints_distinct_ids_with_a_default_actor():
    """Two fresh contexts never collide, and default to the system actor."""
    first = ExecutionContext.new()
    second = ExecutionContext.new()

    assert first.actor_id == "system"
    assert len({first.workflow_id, second.workflow_id}) == 2
    assert len({first.run_id, second.run_id}) == 2
    assert len({first.correlation_id, second.correlation_id}) == 2


def test_context_is_immutable():
    """A context is carried, not edited, as it propagates."""
    context = ExecutionContext.new(actor_id="user:42")

    with pytest.raises(ValidationError):
        context.actor_id = "someone-else"


def test_job_gets_a_context_by_default():
    """A job built without an explicit context still gets a real one."""
    job = Job(title="Python roles", keywords=["python"])
    assert isinstance(job.context, ExecutionContext)


# ----------------------------------------------------------------------
# Persistence: context round-trips through the database
# ----------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'context.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def test_context_round_trips_through_the_repository(session):
    """A job's context survives a create/fetch cycle unchanged.

    This is what lets a worker, loading the job in its own process, run with
    the same workflow/correlation identity the API request minted.
    """
    repo = JobRepository(session)
    context = ExecutionContext(
        workflow_id=uuid4(),
        run_id=uuid4(),
        correlation_id=uuid4(),
        actor_id="user:alice",
    )
    job = repo.create(Job(title="Python roles", keywords=["python"], context=context))

    fetched = repo.get_by_id(job.id)

    assert fetched.context == context
