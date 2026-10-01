"""Permission classes, the decision log, and the terminality of policy verdicts.

Covers the acceptance criteria for the policy engine: one representative
action per permission class reaches the expected outcome under default
configuration, every `evaluate()` call leaves a `policy_decisions` row before
its tool runs, and a DENY is never retried by the executor's retry logic.
"""

from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.domain.errors import RetryableFailure
from personalos.executor import JobSearchExecutor
from personalos.executor.retry import dispatch_with_retry, is_retryable
from personalos.persistence.models import Base, PolicyDecisionModel
from personalos.persistence.policy_log import SqlPolicyDecisionLog
from personalos.persistence.repositories import WorkflowRepository
from personalos.policy import (
    DEFAULT_CLASS_OUTCOMES,
    DEFAULT_TOOL_PERMISSIONS,
    ApprovalRequired,
    Decision,
    IntentOrigin,
    PermissionClass,
    PolicyDecision,
    PolicyDenied,
    PolicyEngine,
    Provenance,
    ToolIntent,
    default_policy_engine,
)
from personalos.tools.gateway import PolicyEnforcingToolGateway, ToolGateway, ToolResult
from tests.unit.test_executor_policy_boundary import InMemoryJobRepository, make_job

ARGS_HASH = "a" * 64
SYSTEM = Provenance(origin=IntentOrigin.SYSTEM, requested_by="test")
LLM = Provenance(origin=IntentOrigin.LLM, requested_by="graph:job_search")


class RecordingLog:
    """Decision log double that keeps rows in a list."""

    def __init__(self):
        self.rows: list[dict] = []

    def record(self, **row) -> None:
        self.rows.append(row)


def evaluate(engine: PolicyEngine, tool: str, provenance: Provenance = SYSTEM, scopes=None):
    """Evaluate one tool with the scopes it is classified as consuming."""
    if scopes is None:
        scopes = sorted(DEFAULT_TOOL_PERMISSIONS[tool].scopes)
    return engine.evaluate("user:1", None, tool, ARGS_HASH, scopes, provenance)


# ----------------------------------------------------------------------
# One representative action per permission class, default configuration
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,permission_class,expected",
    [
        ("jobs.filter_jobs", PermissionClass.READ_LOCAL, Decision.ALLOW),
        ("jobs.search_jobs", PermissionClass.READ_EXTERNAL, Decision.ALLOW),
        ("google.gmail_create_draft", PermissionClass.WRITE_REVERSIBLE, Decision.ALLOW),
        ("google.gmail_send_message", PermissionClass.WRITE_EXTERNAL, Decision.REQUIRE_APPROVAL),
        ("files.delete_file", PermissionClass.DESTRUCTIVE, Decision.REQUIRE_APPROVAL),
        ("google.update_credentials", PermissionClass.SENSITIVE, Decision.DENY),
    ],
)
def test_each_permission_class_has_its_default_outcome(tool, permission_class, expected):
    """The class of the tool, not the caller, decides the outcome."""
    assert DEFAULT_TOOL_PERMISSIONS[tool].permission_class is permission_class
    assert evaluate(default_policy_engine(), tool) is expected


def test_every_permission_class_has_a_default_outcome_and_a_tool():
    """No class is defined without deciding what it costs or what belongs to it."""
    assert set(DEFAULT_CLASS_OUTCOMES) == set(PermissionClass)
    classified = {p.permission_class for p in DEFAULT_TOOL_PERMISSIONS.values()}
    assert classified == set(PermissionClass)


def test_unclassified_tool_is_denied():
    """A tool nobody assigned a class to cannot run."""
    assert evaluate(default_policy_engine(), "jobs.delete_everything", scopes=[]) is Decision.DENY


def test_action_without_provenance_is_denied():
    """An unattributed action is denied even when its class would allow it."""
    assert evaluate(default_policy_engine(), "jobs.search_jobs", Provenance()) is Decision.DENY


def test_scope_beyond_what_the_tool_declares_is_denied():
    """A read cannot ride along with a request for a send scope."""
    decision = evaluate(
        default_policy_engine(),
        "jobs.search_jobs",
        scopes=["jobs:read", "communications:send"],
    )
    assert decision is Decision.DENY


def test_model_proposed_reversible_write_waits_for_approval():
    """What system code may do unattended, a model may only propose."""
    engine = default_policy_engine()
    assert evaluate(engine, "google.gmail_create_draft", LLM) is Decision.REQUIRE_APPROVAL
    assert evaluate(engine, "jobs.search_jobs", LLM) is Decision.ALLOW


def test_sensitive_is_never_delegated_to_the_llm_even_when_reconfigured():
    """No class_outcomes override can hand a credential change to a model."""
    engine = PolicyEngine(class_outcomes={PermissionClass.SENSITIVE: Decision.ALLOW})
    assert evaluate(engine, "google.update_credentials", LLM) is Decision.DENY


def test_permission_class_tightens_the_intent_path_too():
    """An allowlisted intent is still held to its class's outcome."""

    class _AllowAll:
        name = "allow_all"

        def evaluate(self, intent):
            return PolicyDecision(
                intent_id=intent.intent_id,
                tool_ref=intent.tool_ref,
                decision=Decision.ALLOW,
                rule=self.name,
                reason="test rule",
            )

    engine = PolicyEngine([_AllowAll()])
    send = ToolIntent(server="google", tool="gmail_send_message", requested_by="test")
    with pytest.raises(ApprovalRequired):
        engine.authorize(send)
    with pytest.raises(PolicyDenied):
        engine.authorize(ToolIntent(server="google", tool="update_credentials", requested_by="test"))


# ----------------------------------------------------------------------
# Every evaluate() is recorded, before the tool executes
# ----------------------------------------------------------------------


def test_every_outcome_is_recorded():
    """ALLOW, REQUIRE_APPROVAL and DENY each leave a row."""
    log = RecordingLog()
    engine = default_policy_engine(decision_log=log)
    workflow_id = uuid4()

    for tool in ("jobs.search_jobs", "google.gmail_send_message", "google.update_credentials"):
        scopes = sorted(DEFAULT_TOOL_PERMISSIONS[tool].scopes)
        engine.evaluate("user:1", workflow_id, tool, ARGS_HASH, scopes, SYSTEM)

    assert [row["decision"] for row in log.rows] == ["allow", "require_approval", "deny"]
    assert log.rows[0] == {
        "principal": "user:1",
        "workflow_id": workflow_id,
        "tool": "jobs.search_jobs",
        "args_hash": ARGS_HASH,
        "decision": "allow",
        "requested_scopes": ["jobs:read"],
    }


def test_evaluate_writes_a_policy_decisions_row(tmp_path):
    """The engine is backed by the real table, linked to its workflow."""
    db = create_engine(f"sqlite:///{tmp_path / 'policy.db'}")
    Base.metadata.create_all(db)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=db)
    session = factory()
    workflow_id = WorkflowRepository(session).create(name="job_search").id
    engine = default_policy_engine(decision_log=SqlPolicyDecisionLog(factory))

    engine.evaluate(
        "user:1", workflow_id, "google.gmail_send_message", ARGS_HASH,
        ["communications:send"], SYSTEM,
    )
    # A run that never registered a workflow is still recorded, unlinked.
    engine.evaluate("user:1", uuid4(), "google.update_credentials", ARGS_HASH, [], SYSTEM)

    rows = session.query(PolicyDecisionModel).order_by(PolicyDecisionModel.decided_at).all()
    assert [(r.tool, r.decision, r.workflow_id) for r in rows] == [
        ("google.gmail_send_message", "require_approval", workflow_id),
        ("google.update_credentials", "deny", None),
    ]
    assert rows[0].principal == "user:1"
    assert rows[0].args_hash == ARGS_HASH
    assert rows[0].requested_scopes == ["communications:send"]
    session.close()
    db.dispose()


class _OrderInvoker:
    """Invoker double that notes how many decisions existed when it ran."""

    def __init__(self, log: RecordingLog):
        self.log = log
        self.rows_seen_at_invoke: list[int] = []

    async def invoke(self, approved):
        self.rows_seen_at_invoke.append(len(self.log.rows))
        return {"success": True, "result": {}}


def search_intent(**overrides) -> ToolIntent:
    """A well-formed, allowlisted read intent."""
    defaults = {
        "server": "jobs",
        "tool": "search_jobs",
        "arguments": {"keywords": ["python"]},
        "requested_by": "test",
    }
    defaults.update(overrides)
    return ToolIntent(**defaults)


async def test_decision_row_exists_before_the_tool_executes():
    """By the time the adapter runs, the verdict that cleared it is recorded."""
    log = RecordingLog()
    invoker = _OrderInvoker(log)
    gateway = PolicyEnforcingToolGateway(default_policy_engine(decision_log=log), invoker)
    intent = search_intent()

    await gateway.dispatch(intent)

    assert invoker.rows_seen_at_invoke == [1]
    assert log.rows[0]["decision"] == "allow"
    assert log.rows[0]["tool"] == "jobs.search_jobs"
    assert log.rows[0]["args_hash"] == intent.fingerprint()
    assert log.rows[0]["workflow_id"] == intent.context.workflow_id


async def test_denied_dispatch_is_recorded_and_never_reaches_the_tool():
    """A denial leaves a row and no execution."""
    log = RecordingLog()
    invoker = _OrderInvoker(log)
    gateway = PolicyEnforcingToolGateway(default_policy_engine(decision_log=log), invoker)

    with pytest.raises(PolicyDenied):
        await gateway.dispatch(search_intent(tool="delete_everything"))

    assert [row["decision"] for row in log.rows] == ["deny"]
    assert invoker.rows_seen_at_invoke == []


async def test_tool_does_not_run_when_the_decision_cannot_be_recorded():
    """No row, no execution: a failing log fails closed."""

    class _BrokenLog:
        def record(self, **row):
            raise RuntimeError("database unavailable")

    invoker = _OrderInvoker(RecordingLog())
    gateway = PolicyEnforcingToolGateway(
        default_policy_engine(decision_log=_BrokenLog()), invoker
    )

    with pytest.raises(RuntimeError):
        await gateway.dispatch(search_intent())
    assert invoker.rows_seen_at_invoke == []


# ----------------------------------------------------------------------
# DENY and REQUIRE_APPROVAL are terminal for automatic retries
# ----------------------------------------------------------------------


class CountingGateway(ToolGateway):
    """Gateway double that counts dispatches and raises scripted errors."""

    def __init__(self, policy: PolicyEngine, errors=()):
        self.policy = policy
        self.errors = list(errors)
        self.dispatches = 0

    async def dispatch(self, intent, approval=None):
        self.dispatches += 1
        approved = self.policy.authorize(intent, approval)
        if self.errors:
            raise self.errors.pop(0)
        return ToolResult.from_adapter_payload(approved, {"success": True, "result": {}})


async def _no_sleep(_delay: float) -> None:
    return None


async def test_deny_is_never_retried():
    """A denied dispatch is attempted exactly once."""
    log = RecordingLog()
    gateway = CountingGateway(default_policy_engine(decision_log=log))

    with pytest.raises(PolicyDenied):
        await dispatch_with_retry(
            gateway, search_intent(tool="delete_everything"), max_attempts=5, sleep=_no_sleep
        )

    assert gateway.dispatches == 1
    assert [row["decision"] for row in log.rows] == ["deny"]


async def test_require_approval_is_never_retried():
    """An action waiting on a human is not re-asked in a loop."""
    gateway = CountingGateway(default_policy_engine())
    intent = search_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j1", "idempotency_key": "k" * 12},
        mutating=True,
    )

    with pytest.raises(ApprovalRequired):
        await dispatch_with_retry(gateway, intent, max_attempts=5, sleep=_no_sleep)

    assert gateway.dispatches == 1


async def test_transient_failure_is_retried():
    """The retry logic does retry; it is policy verdicts it refuses."""
    gateway = CountingGateway(default_policy_engine(), errors=[RetryableFailure()] * 2)

    result = await dispatch_with_retry(gateway, search_intent(), sleep=_no_sleep)

    assert result.success
    assert gateway.dispatches == 3


def test_policy_verdicts_are_not_retryable_even_if_mislabelled():
    """Terminality is a property of the type, not of a flag someone can flip."""

    class _Mislabelled(PolicyDenied):
        retryable = True

    decision = default_policy_engine().evaluate_intent(search_intent(tool="nope"))
    assert not is_retryable(PolicyDenied(decision))
    assert not is_retryable(_Mislabelled(decision))
    assert is_retryable(RetryableFailure())


async def test_executor_does_not_retry_a_denied_step():
    """End to end: the executor's own dispatch path gives a denial one attempt."""
    repo = InMemoryJobRepository()
    # An engine with no rules denies everything.
    gateway = CountingGateway(PolicyEngine())
    job = repo.create(make_job())

    with pytest.raises(PolicyDenied):
        await JobSearchExecutor(repo, gateway).run_job_search(job)

    assert gateway.dispatches == 1
