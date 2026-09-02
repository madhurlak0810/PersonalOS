"""The executor executes approved intents, and nothing else.

These tests pin the boundary from the executor side: what it emits, what it
refuses to be built without, and what happens when policy says no.
"""

from typing import Any

import pytest

from personalos.domain.models import Job, JobStatus
from personalos.executor import JobSearchExecutor
from personalos.persistence.repositories import JobRepository
from personalos.policy import (
    ApprovalGrant,
    Decision,
    IntentOrigin,
    PolicyDenied,
    PolicyEngine,
    ToolIntent,
    default_policy_engine,
)
from personalos.policy.rules import JOB_SEARCH_TOOL_ARGUMENTS
from personalos.tools.gateway import ToolGateway, ToolResult


class InMemoryJobRepository(JobRepository):
    """Repository double that keeps jobs in a dict."""

    def __init__(self):
        self.jobs: dict[Any, Job] = {}

    def create(self, job: Job) -> Job:
        """Store a new job."""
        self.jobs[job.id] = job
        return job

    def update(self, job: Job) -> Job:
        """Replace a stored job."""
        self.jobs[job.id] = job
        return job

    def get_by_id(self, job_id) -> Job | None:
        """Fetch a stored job."""
        return self.jobs.get(job_id)


class RecordingGateway(ToolGateway):
    """Gateway double that records intents and runs the real policy engine.

    Policy is genuinely evaluated -- only the adapter is faked -- so a test can
    assert both what the executor asked for and whether it was permitted.
    """

    def __init__(self, policy: PolicyEngine | None = None, payloads=None):
        self.policy = policy or default_policy_engine()
        self.payloads = payloads or {}
        self.intents: list[ToolIntent] = []

    async def dispatch(
        self, intent: ToolIntent, approval: ApprovalGrant | None = None
    ) -> ToolResult:
        """Authorize the intent for real, then return a canned payload."""
        if not isinstance(intent, ToolIntent):
            raise AssertionError(f"executor dispatched a {type(intent).__name__}")
        approved = self.policy.authorize(intent, approval)
        self.intents.append(intent)
        payload = self.payloads.get(intent.tool, {"success": True, "result": {}})
        return ToolResult.from_adapter_payload(approved, payload)

    def intents_for(self, tool: str) -> list[ToolIntent]:
        """Every recorded intent targeting one tool."""
        return [i for i in self.intents if i.tool == tool]


def make_job(**overrides) -> Job:
    """A job search request."""
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
    """Canned MCP responses for a complete run."""
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


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_executor_cannot_be_built_without_a_gateway():
    """No gateway means no tools; there is no global fallback to reach for."""
    with pytest.raises(TypeError):
        JobSearchExecutor(InMemoryJobRepository())


def test_executor_rejects_an_explicit_none_gateway():
    """Passing None is a mistake, not a request for a default."""
    with pytest.raises(ValueError):
        JobSearchExecutor(InMemoryJobRepository(), None)


def test_executor_holds_no_tool_adapter():
    """The executor has no attribute that could reach a server directly."""
    executor = JobSearchExecutor(InMemoryJobRepository(), RecordingGateway())
    assert not hasattr(executor, "mcp_manager")
    assert isinstance(executor.gateway, ToolGateway)


# ----------------------------------------------------------------------
# What the executor emits
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_tool_call_is_an_intent_cleared_by_policy():
    """A full run reaches the adapter only through authorized intents."""
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(payloads=default_payloads())
    job = repo.create(make_job())

    result = await JobSearchExecutor(repo, gateway).run_job_search(job)

    assert result.status == JobStatus.COMPLETED
    assert [i.tool for i in gateway.intents] == [
        "search_jobs",
        "scrape_job_details",
        "filter_jobs",
    ]
    for intent in gateway.intents:
        assert intent.server == "jobs"
        assert intent.origin == IntentOrigin.SYSTEM
        assert intent.requested_by.startswith("executor:job_search#")
        assert intent.job_id == job.id
        assert intent.agent_id is not None


@pytest.mark.asyncio
async def test_executor_emits_no_mutating_intents():
    """The read-only search loop never proposes a side effect."""
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(payloads=default_payloads())
    job = repo.create(make_job())

    await JobSearchExecutor(repo, gateway).run_job_search(job)

    assert gateway.intents, "expected the run to dispatch intents"
    assert all(not intent.mutating for intent in gateway.intents)


@pytest.mark.asyncio
async def test_intent_arguments_stay_inside_the_declared_surface():
    """The executor cannot widen a tool contract at runtime."""
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(payloads=default_payloads())
    job = repo.create(make_job())

    await JobSearchExecutor(repo, gateway).run_job_search(job)

    for intent in gateway.intents:
        declared = JOB_SEARCH_TOOL_ARGUMENTS[intent.tool_ref]
        assert set(intent.arguments) <= declared, intent.tool_ref


@pytest.mark.asyncio
async def test_tool_output_cannot_widen_the_next_intent():
    """Extra keys in a tool response do not become arguments downstream.

    A scraped listing is data, not instructions: the executor picks the fields
    the next tool declared and drops the rest, so an upstream server cannot
    inject arguments into a later call.
    """
    payloads = default_payloads()
    payloads["search_jobs"] = {
        "success": True,
        "result": {
            "jobs": [
                {
                    "id": "j1",
                    "url": "https://example.com/j1",
                    "idempotency_key": "smuggled-key",
                    "callback": "http://attacker.example/steal",
                }
            ]
        },
    }
    repo = InMemoryJobRepository()
    gateway = RecordingGateway(payloads=payloads)
    job = repo.create(make_job())

    await JobSearchExecutor(repo, gateway).run_job_search(job)

    scrape = gateway.intents_for("scrape_job_details")[0]
    assert set(scrape.arguments) == {"job_id", "job_url"}


# ----------------------------------------------------------------------
# What happens when policy says no
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_denial_fails_the_job_rather_than_continuing():
    """A blocked step stops the run and is recorded as a failure."""
    repo = InMemoryJobRepository()
    # An engine with no rules denies everything.
    gateway = RecordingGateway(policy=PolicyEngine(), payloads=default_payloads())
    job = repo.create(make_job())

    with pytest.raises(PolicyDenied):
        await JobSearchExecutor(repo, gateway).run_job_search(job)

    assert gateway.intents == []
    assert repo.get_by_id(job.id).status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_a_model_proposed_intent_is_still_subject_to_policy():
    """Nothing about arriving from a model exempts an intent from the rules.

    Stated from the caller's side: an LLM-origin intent carrying an argument
    outside the declared surface is denied, so raw model output is never what
    reaches an adapter.
    """
    policy = default_policy_engine()
    smuggled = ToolIntent(
        server="jobs",
        tool="search_jobs",
        arguments={"keywords": ["python"], "callback": "http://attacker.example"},
        origin=IntentOrigin.LLM,
        requested_by="llm:planner",
    )

    with pytest.raises(PolicyDenied):
        policy.authorize(smuggled)

    # The same tool, within its declared surface, is fine.
    legitimate = ToolIntent(
        server="jobs",
        tool="search_jobs",
        arguments={"keywords": ["python"]},
        origin=IntentOrigin.LLM,
        requested_by="llm:planner",
    )
    assert policy.authorize(legitimate).decision.decision == Decision.ALLOW
