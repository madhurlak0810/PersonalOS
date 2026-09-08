"""Context continuity: one `ExecutionContext` should reach every hop.

These tests pin the traceability contract end to end: the same workflow_id,
run_id, correlation_id, and actor_id set on a `Job` must show up on every
`ToolIntent` the executor dispatches, on every `Event` it publishes, and must
round-trip through persistence unchanged -- so a worker loading the job in a
separate process still runs with the same identity the API request minted.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.domain.context import ExecutionContext
from personalos.domain.models import Event, EventType, Job
from personalos.executor import JobSearchExecutor
from personalos.persistence.models import Base
from personalos.persistence.repositories import JobRepository
from personalos.policy import (
    ApprovalGrant,
    PolicyDenied,
    PolicyEngine,
    ToolIntent,
    default_policy_engine,
)
from personalos.tools.gateway import ToolGateway, ToolResult

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
# Executor: intents and events carry the job's context
# ----------------------------------------------------------------------


class InMemoryJobRepository(JobRepository):
    """Repository double that keeps jobs in a dict."""

    def __init__(self):
        self.jobs: dict[Any, Job] = {}

    def create(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    def update(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    def get_by_id(self, job_id) -> Job | None:
        return self.jobs.get(job_id)


class RecordingGateway(ToolGateway):
    """Gateway double that records intents and runs the real policy engine."""

    def __init__(self, policy: PolicyEngine | None = None, payloads=None):
        self.policy = policy or default_policy_engine()
        self.payloads = payloads or {}
        self.intents: list[ToolIntent] = []

    async def dispatch(self, intent: ToolIntent, approval: ApprovalGrant | None = None) -> ToolResult:
        approved = self.policy.authorize(intent, approval)
        self.intents.append(intent)
        payload = self.payloads.get(intent.tool, {"success": True, "result": {}})
        return ToolResult.from_adapter_payload(approved, payload)


class RecordingEventBus:
    """Event bus double that just remembers what was published."""

    def __init__(self):
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)


def make_job(**overrides) -> Job:
    defaults = {
        "title": "Python roles",
        "keywords": ["python"],
        "locations": ["remote"],
        "salary_min": 100000,
        "salary_max": 150000,
        "job_type": "full-time",
    }
    defaults.update(overrides)
    return Job(**defaults)


def default_payloads():
    return {
        "search_jobs": {
            "success": True,
            "result": {"jobs": [{"id": "j1", "url": "https://example.com/j1"}]},
        },
        "scrape_job_details": {
            "success": True,
            "result": {"id": "j1", "title": "Python Developer"},
        },
        "filter_jobs": {
            "success": True,
            "result": {"jobs": [{"id": "j1", "title": "Python Developer"}]},
        },
    }


@pytest.mark.asyncio
async def test_every_dispatched_intent_carries_the_jobs_context():
    """The context set on the job is the context on every tool call it makes."""
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(payloads=default_payloads())
    context = ExecutionContext.new(actor_id="user:alice")
    job = repo.create(make_job(context=context))

    await JobSearchExecutor(repo, gateway).run_job_search(job)

    assert gateway.intents, "expected the run to dispatch intents"
    for intent in gateway.intents:
        assert intent.context == context


@pytest.mark.asyncio
async def test_lifecycle_events_carry_the_jobs_context():
    """JOB_STARTED and JOB_COMPLETED are published with the job's context."""
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(payloads=default_payloads())
    bus = RecordingEventBus()
    context = ExecutionContext.new(actor_id="user:alice")
    job = repo.create(make_job(context=context))

    await JobSearchExecutor(repo, gateway, event_bus=bus).run_job_search(job)

    event_types = [e.event_type for e in bus.events]
    assert event_types == [EventType.JOB_STARTED, EventType.JOB_COMPLETED]
    for event in bus.events:
        assert event.context == context
        assert event.job_id == job.id


@pytest.mark.asyncio
async def test_a_failed_run_still_publishes_with_the_jobs_context():
    """A denial fails the job, but the failure event still carries its context."""
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(policy=PolicyEngine(), payloads=default_payloads())
    bus = RecordingEventBus()
    context = ExecutionContext.new(actor_id="user:alice")
    job = repo.create(make_job(context=context))

    with pytest.raises(PolicyDenied):
        await JobSearchExecutor(repo, gateway, event_bus=bus).run_job_search(job)

    event_types = [e.event_type for e in bus.events]
    assert event_types == [EventType.JOB_STARTED, EventType.JOB_FAILED]
    for event in bus.events:
        assert event.context == context


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
