"""Tests for recoverable failures: `GET .../failures`, `POST .../retry`, `POST .../dismiss`.

The acceptance criteria this file exists for:

1. **Retrying a step whose operation already SUCCEEDED returns the stored
   receipt and causes no duplicate side effect.**
   `test_retrying_a_step_whose_send_already_succeeded_replays_its_receipt`.
2. **A policy-denied action does not offer a retry.**
   `test_a_policy_denied_action_is_denied_not_failed_and_offers_no_retry`.
3. **A transient failure's retry resumes from the failed step, not from the
   start.** `test_a_transient_failure_retry_resumes_from_the_failed_step`.

Everything below the HTTP layer is real: a three-step graph
(`prepare -> send -> finish`) on the durable checkpointer, whose `send` step
goes through a real `ToolExecutor` and execution ledger, run by
`process_next` the way the worker runs it. Only the provider is a spy, so a
side effect is something that can be counted.
"""

import asyncio
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    build_durable_checkpointer,
    build_integration_health_store,
    build_tool_executor,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.domain.errors import (
    ApprovalRequiredError,
    FailureClass,
    HumanTimeout,
    ModelSemanticFailure,
    ModelTransientFailure,
    PartialSideEffect,
    PersonalOSError,
    PolicyDeniedError,
    ProviderPermanentFailure,
)
from personalos.domain.failures import (
    MAX_FAILURE_ATTEMPTS,
    FailedOperation,
    FailureStatus,
    OperationRecovery,
    assess_failure,
    classify,
    parse_stored_error,
    sanitize_error,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalVerdict,
)
from personalos.domain.workflow import (
    JOB_SEARCH_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
)
from personalos.executor.retry import is_retryable
from personalos.executor.tool_executor import ReconcileOutcome, Reconciliation
from personalos.persistence.leases import WorkflowLeaseUnavailable
from personalos.persistence.models import AuditEventModel, Base, ToolExecutionModel
from personalos.policy import Decision, PermissionClass, PolicyEngine
from personalos.providers.base import ProviderUnavailable

USER_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000cc")
SECRET = "sk-live-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "failures.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


def api_process(db_path: Path) -> TestClient:
    """A fresh API instance over the database, sharing nothing but the file."""
    engine = create_engine(f"sqlite:///{db_path}")
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app = create_app()
    app.dependency_overrides[get_failure_services] = lambda: build_failure_services(factory)
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


class SpyProvider:
    """The provider behind `send`: counts every side effect it performs."""

    def __init__(self, *, errors: list[Exception] | None = None):
        self.errors = list(errors or [])
        self.calls: list[str] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        if self.errors:
            raise self.errors.pop(0)
        self.calls.append(intent.idempotency_key)
        return ActionReceipt(
            action_id=intent.action_id, ok=True, external_reference=f"ext-{len(self.calls)}"
        )


class ScriptedReconciler:
    """Answers that the action landed, and records what it was asked about."""

    def __init__(self):
        self.asked: list[str] = []

    async def reconcile(self, intent: ActionIntent) -> Reconciliation:
        self.asked.append(intent.idempotency_key)
        return Reconciliation(
            ReconcileOutcome.APPLIED,
            ActionReceipt(action_id=intent.action_id, ok=True, external_reference="ext-found"),
        )


class _State(TypedDict, total=False):
    prepared: bool
    receipt: dict[str, Any] | None
    finished: bool


class Harness:
    """One workflow on one thread, its graph, and a worker to run its commands.

    `crash_after_send` and `finish_errors` make the `send` and `finish` steps
    fail that many times; `runs` counts every time each step was entered.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        provider: SpyProvider | None = None,
        policy: PolicyEngine | None = None,
        reconciler: ScriptedReconciler | None = None,
        crash_after_send: int = 0,
        finish_errors: list[Exception] | None = None,
    ):
        engine = create_engine(f"sqlite:///{db_path}")
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.registry = build_workflow_thread_registry(self.factory)
        self.queue = build_workflow_command_queue(self.factory)
        self.provider = provider or SpyProvider()
        self.crash_after_send = crash_after_send
        self.finish_errors = list(finish_errors or [])
        self.runs = {"prepare": 0, "send": 0, "finish": 0}
        self.receipts: list[dict[str, Any]] = []

        self.workflow_id = self.registry.create_workflow(JOB_SEARCH_WORKFLOW)
        self.thread = register_job_search_thread(
            user_id=USER_ID, registry=self.registry, workflow_id=self.workflow_id
        )
        self.executor = build_tool_executor(
            self.provider,
            self.factory,
            workflow_id=self.workflow_id,
            policy=policy,
            reconciler=reconciler,
        )
        self.intent = ActionIntent(
            kind=ActionKind.SUBMIT_APPLICATION,
            target="https://example.test/acme/backend",
            summary="Submit an application to Acme for Backend Engineer",
            payload={"dedupe_key": "acme:backend"},
            idempotency_key="submit-acme-backend",
        )
        self.decision = ApprovalDecision(
            action_id=self.intent.action_id,
            action_fingerprint=self.intent.fingerprint(),
            verdict=ApprovalVerdict.APPROVED,
            decided_by="reviewer@example.test",
        )

        graph = StateGraph(_State)
        graph.add_node("prepare", self._prepare)
        graph.add_node("send", self._send)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "send")
        graph.add_edge("send", "finish")
        graph.add_edge("finish", END)
        self.graph = graph.compile(
            checkpointer=build_durable_checkpointer(self.factory, self.registry)
        )
        self.runner = DurableWorkflowRunner(
            self.graph,
            registry=self.registry,
            leases=build_workflow_lease_store(self.factory),
            integrations=build_integration_health_store(self.factory),
        )

    def _prepare(self, state: _State) -> dict[str, Any]:
        self.runs["prepare"] += 1
        return {"prepared": True}

    async def _send(self, state: _State) -> dict[str, Any]:
        self.runs["send"] += 1
        receipt = await self.executor.execute(self.intent, self.decision)
        self.receipts.append(receipt.model_dump(mode="json"))
        if self.crash_after_send:
            self.crash_after_send -= 1
            raise RuntimeError("the worker crashed after the send, before the step finished")
        return {"receipt": receipt.model_dump(mode="json")}

    def _finish(self, state: _State) -> dict[str, Any]:
        self.runs["finish"] += 1
        if self.finish_errors:
            raise self.finish_errors.pop(0)
        return {"finished": True}

    def start(self) -> None:
        self.queue.enqueue(
            WorkflowCommand(
                command_id=uuid4(),
                kind=WorkflowCommandKind.START,
                workflow_id=self.workflow_id,
                thread_id=self.thread.thread_id,
                actor_id="user:alice",
                correlation_id=uuid4(),
                graph_input={},
            )
        )

    def run_next(self) -> bool:
        return asyncio.run(process_next(self.queue, {JOB_SEARCH_THREAD_NAMESPACE: self.runner}))

    def started_and_failed(self) -> "Harness":
        self.start()
        with pytest.raises((RuntimeError, OSError, PersonalOSError)):
            self.run_next()
        return self

    def state(self) -> dict[str, Any]:
        return asyncio.run(self.graph.aget_state(self.thread.config())).values

    def rows(self, model) -> list[dict[str, Any]]:
        session = self.factory()
        try:
            return [row.to_dict() for row in session.query(model).all()]
        finally:
            session.close()


def failures(client: TestClient, workflow_id: UUID, **params) -> dict[str, Any]:
    response = client.get(f"/v1/workflows/{workflow_id}/failures", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def only_failure(client: TestClient, workflow_id: UUID) -> dict[str, Any]:
    [failure] = failures(client, workflow_id)["failures"]
    return failure


# --- Acceptance -----------------------------------------------------------------


def test_retrying_a_step_whose_send_already_succeeded_replays_its_receipt(db_path):
    """Acceptance 1: the send landed, the step died after it; the retry sends nothing."""
    harness = Harness(db_path, crash_after_send=1).started_and_failed()
    assert harness.provider.calls == ["submit-acme-backend"]
    [stored] = [r for r in harness.rows(ToolExecutionModel) if r["status"] == "completed"]

    client = api_process(db_path)
    failure = only_failure(client, harness.workflow_id)
    # The error itself was unclassified, but what matters for its retry is
    # that the step had already sent something.
    assert failure["failure_class"] == "partial_side_effect"
    assert failure["cause_class"] == "unclassified"
    assert failure["step"] == "send"
    assert failure["retryable"] is True
    [operation] = failure["operations"]
    assert operation["status"] == "completed"
    assert operation["recovery"] == "replay_stored_receipt"
    assert operation["receipt"] == stored["receipt_json"]

    response = client.post(
        f"/v1/workflows/{harness.workflow_id}/retry", headers={"X-Actor-Id": "user:alice"}
    )
    assert response.status_code == 202, response.text
    retry = response.json()
    assert retry["failure_id"] == failure["failure_id"]
    assert retry["step"] == "send"
    assert retry["attempt"] == 2
    assert retry["operations"][0]["receipt"] == stored["receipt_json"]

    assert harness.run_next()

    # No duplicate side effect: still one provider call, and the re-run step
    # got back exactly the receipt the first attempt stored.
    assert harness.provider.calls == ["submit-acme-backend"]
    assert harness.receipts[1]["external_reference"] == stored["receipt_json"]["external_reference"]
    assert harness.state()["receipt"]["external_reference"] == "ext-1"
    assert harness.state()["finished"] is True
    assert len(harness.rows(ToolExecutionModel)) == 1
    assert failures(client, harness.workflow_id) == {
        "workflow_id": str(harness.workflow_id),
        "status": "completed",
        "failures": [],
    }


def test_a_policy_denied_action_is_denied_not_failed_and_offers_no_retry(db_path):
    """Acceptance 2: a denial is an answer; it can be acknowledged, never retried."""
    deny_writes = PolicyEngine(class_outcomes={PermissionClass.WRITE_EXTERNAL: Decision.DENY})
    harness = Harness(db_path, policy=deny_writes).started_and_failed()
    client = api_process(db_path)

    failure = only_failure(client, harness.workflow_id)
    assert failure["failure_class"] == "policy_denial"
    assert failure["status"] == "denied"
    assert failure["retryable"] is False
    assert failure["actions"] == ["dismiss"]
    assert "policy" in failure["not_retryable_reason"]

    response = client.post(f"/v1/workflows/{harness.workflow_id}/retry")
    assert response.status_code == 409
    assert response.json()["error_code"] == "conflict"
    assert harness.queue.queued_for(harness.workflow_id) == []
    assert harness.provider.calls == []

    dismissed = client.post(
        f"/v1/workflows/{harness.workflow_id}/dismiss", headers={"X-Actor-Id": "user:alice"}
    )
    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["status"] == "dismissed"
    assert dismissed.json()["dismissed_by"] == "user:alice"
    assert failures(client, harness.workflow_id)["failures"] == []
    [closed] = failures(client, harness.workflow_id, include_dismissed=True)["failures"]
    assert closed["status"] == "dismissed"
    assert closed["actions"] == []


def test_a_transient_failure_retry_resumes_from_the_failed_step(db_path):
    """Acceptance 3: only the step that failed runs again; nothing before it does."""
    harness = Harness(
        db_path, finish_errors=[ProviderUnavailable("job board timed out")]
    ).started_and_failed()
    client = api_process(db_path)

    failure = only_failure(client, harness.workflow_id)
    assert failure["failure_class"] == "provider_transient"
    assert failure["step"] == "finish"
    assert failure["attempts"] == 1
    assert failure["actions"] == ["retry", "dismiss"]
    # The send finished in an earlier step, so it is not this step's to redo.
    assert failure["operations"] == []

    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 202
    assert client.get(f"/v1/workflows/{harness.workflow_id}").json()["status"] == "queued"
    assert harness.run_next()

    assert harness.runs == {"prepare": 1, "send": 1, "finish": 2}
    assert harness.provider.calls == ["submit-acme-backend"]
    assert harness.state()["finished"] is True
    assert client.get(f"/v1/workflows/{harness.workflow_id}").json()["status"] == "completed"


# --- Retry policy ---------------------------------------------------------------


def test_a_send_whose_outcome_is_unknown_is_reconciled_before_anything_is_sent(db_path):
    """The provider raised mid-send: the retry asks it first, and does not send again."""
    reconciler = ScriptedReconciler()
    harness = Harness(
        db_path,
        provider=SpyProvider(errors=[ConnectionResetError("connection reset by peer")]),
        reconciler=reconciler,
    ).started_and_failed()
    client = api_process(db_path)

    failure = only_failure(client, harness.workflow_id)
    assert failure["failure_class"] == "provider_transient"
    [operation] = failure["operations"]
    assert operation["status"] == "failed"
    assert operation["recovery"] == "reconcile_before_send"

    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 202
    assert harness.run_next()

    assert reconciler.asked == ["submit-acme-backend"]
    assert harness.provider.calls == []
    assert harness.state()["receipt"]["external_reference"] == "ext-found"


def test_a_failed_retry_is_the_same_failure_with_one_more_attempt(db_path):
    errors = [ProviderUnavailable("timed out"), ProviderUnavailable("timed out again")]
    harness = Harness(db_path, finish_errors=errors).started_and_failed()
    client = api_process(db_path)
    first = only_failure(client, harness.workflow_id)

    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 202
    with pytest.raises(ProviderUnavailable):
        harness.run_next()

    again = only_failure(client, harness.workflow_id)
    assert again["failure_id"] == first["failure_id"]
    assert again["attempts"] == 2
    assert "timed out again" in again["last_error"]
    assert harness.runs["prepare"] == 1


def test_a_step_out_of_attempts_can_only_be_dismissed(db_path):
    errors = [ProviderUnavailable("timed out") for _ in range(MAX_FAILURE_ATTEMPTS)]
    harness = Harness(db_path, finish_errors=errors).started_and_failed()
    client = api_process(db_path)
    for _ in range(MAX_FAILURE_ATTEMPTS - 1):
        assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 202
        with pytest.raises(ProviderUnavailable):
            harness.run_next()

    failure = only_failure(client, harness.workflow_id)
    assert failure["attempts"] == MAX_FAILURE_ATTEMPTS
    assert failure["retryable"] is False
    assert failure["actions"] == ["dismiss"]
    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 409


def test_a_second_retry_while_one_is_queued_is_refused(db_path):
    harness = Harness(db_path, finish_errors=[ProviderUnavailable("x")]).started_and_failed()
    client = api_process(db_path)

    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 202
    assert only_failure(client, harness.workflow_id)["status"] == "retry_queued"
    second = client.post(f"/v1/workflows/{harness.workflow_id}/retry")
    assert second.status_code == 409
    assert "already queued" in second.json()["message"]
    assert client.post(f"/v1/workflows/{harness.workflow_id}/dismiss").status_code == 409
    assert len(harness.queue.queued_for(harness.workflow_id)) == 1


def test_a_provider_permanent_failure_degrades_the_integration_and_asks_for_reconnect(db_path):
    refusal = ProviderPermanentFailure(
        f"google refused the token {SECRET}: insufficient scope",
        provider="google",
        reason=ProviderPermanentFailure.MISSING_SCOPE,
    )
    harness = Harness(db_path, provider=SpyProvider(errors=[refusal])).started_and_failed()
    client = api_process(db_path)

    failure = only_failure(client, harness.workflow_id)
    assert failure["failure_class"] == "provider_permanent"
    assert failure["retryable"] is False
    assert failure["actions"] == ["reconnect", "dismiss"]
    assert failure["integration"]["provider"] == "google"
    assert failure["integration"]["status"] == "degraded"
    assert failure["integration"]["reason"] == "missing_scope"
    assert failure["integration"]["reconnect_required"] is True
    assert SECRET not in failure["last_error"]
    assert build_integration_health_store(harness.factory).get("google").status == "degraded"

    response = client.post(f"/v1/workflows/{harness.workflow_id}/retry")
    assert response.status_code == 409
    assert "reconnect" in response.json()["message"]
    # Not retried automatically either: nothing loops on a permanent refusal.
    assert is_retryable(refusal) is False


def test_retry_and_dismiss_are_recorded_in_the_audit_trail(db_path):
    harness = Harness(db_path, finish_errors=[ProviderUnavailable("x")]).started_and_failed()
    client = api_process(db_path)
    failure_id = only_failure(client, harness.workflow_id)["failure_id"]

    client.post(f"/v1/workflows/{harness.workflow_id}/retry", headers={"X-Actor-Id": "user:a"})
    actions = [
        (row["actor"], row["action"], row["target_ref"])
        for row in harness.rows(AuditEventModel)
        if row["action"].startswith("workflow.failure.")
    ]
    assert actions == [("user:a", "workflow.failure.retry_requested", f"failure:{failure_id}")]


def test_failure_errors_are_redacted_in_the_status_endpoint_too(db_path):
    harness = Harness(
        db_path, finish_errors=[RuntimeError(f"upstream said Authorization: Bearer {SECRET}")]
    ).started_and_failed()
    status = api_process(db_path).get(f"/v1/workflows/{harness.workflow_id}").json()
    [failure] = status["recoverable_failures"]
    assert SECRET not in failure["message"]
    assert SECRET not in only_failure(api_process(db_path), harness.workflow_id)["last_error"]


# --- Selection and errors -------------------------------------------------------


def test_an_unknown_workflow_is_a_404(db_path):
    client = api_process(db_path)
    assert client.get(f"/v1/workflows/{uuid4()}/failures").status_code == 404
    assert client.post(f"/v1/workflows/{uuid4()}/retry").status_code == 404
    assert client.post(f"/v1/workflows/{uuid4()}/dismiss").status_code == 404


def test_an_unknown_failure_id_is_a_404(db_path):
    harness = Harness(db_path, finish_errors=[ProviderUnavailable("x")]).started_and_failed()
    response = api_process(db_path).post(
        f"/v1/workflows/{harness.workflow_id}/retry", json={"failure_id": str(uuid4())}
    )
    assert response.status_code == 404


def test_retrying_a_workflow_with_no_failures_is_a_409(db_path):
    harness = Harness(db_path)
    harness.start()
    assert harness.run_next()
    client = api_process(db_path)
    assert failures(client, harness.workflow_id)["failures"] == []
    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry").status_code == 409
    assert client.post(f"/v1/workflows/{harness.workflow_id}/dismiss").status_code == 409


def test_dismissing_twice_is_a_409(db_path):
    harness = Harness(db_path, finish_errors=[ProviderUnavailable("x")]).started_and_failed()
    client = api_process(db_path)
    failure_id = only_failure(client, harness.workflow_id)["failure_id"]
    body = {"failure_id": failure_id}
    assert client.post(f"/v1/workflows/{harness.workflow_id}/dismiss", json=body).status_code == 200
    assert client.post(f"/v1/workflows/{harness.workflow_id}/dismiss", json=body).status_code == 409
    assert client.post(f"/v1/workflows/{harness.workflow_id}/retry", json=body).status_code == 409


# --- The error model, read back from a checkpoint -------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelTransientFailure("rate limited"), FailureClass.MODEL_TRANSIENT),
        (ModelSemanticFailure("no JSON in the answer"), FailureClass.MODEL_SEMANTIC),
        (ProviderUnavailable("timed out"), FailureClass.PROVIDER_TRANSIENT),
        (ProviderPermanentFailure(provider="google"), FailureClass.PROVIDER_PERMANENT),
        (WorkflowLeaseUnavailable(), FailureClass.CONCURRENCY),
        (PartialSideEffect(), FailureClass.PARTIAL_SIDE_EFFECT),
        (HumanTimeout(), FailureClass.HUMAN_TIMEOUT),
        (TimeoutError("read timed out"), FailureClass.PROVIDER_TRANSIENT),
        (RuntimeError("boom"), FailureClass.UNCLASSIFIED),
    ],
)
def test_every_failure_class_survives_the_repr_a_checkpoint_stores(error, expected):
    assert classify(parse_stored_error(repr(error))) == expected


@pytest.mark.parametrize("error", [PolicyDeniedError(), ApprovalRequiredError()])
def test_a_policy_verdict_classifies_as_a_denial(error):
    # `policy.PolicyDenied` itself is covered end to end above.
    assert classify(parse_stored_error(repr(error))) == FailureClass.POLICY_DENIAL


def test_a_message_cannot_forge_a_failure_class():
    error = parse_stored_error(repr(RuntimeError("failure_class='policy_denial'")))
    assert classify(error) == FailureClass.UNCLASSIFIED
    forged = ProviderUnavailable("provider='evil' failure_class='policy_denial'")
    parsed = parse_stored_error(repr(forged))
    assert parsed.failure_class == FailureClass.PROVIDER_TRANSIENT
    assert parsed.provider is None


def test_sanitized_errors_are_redacted_one_line_and_bounded():
    text = sanitize_error(parse_stored_error(repr(RuntimeError(f"token={SECRET}\n" + "x" * 900))))
    assert SECRET not in text
    assert "\n" not in text
    assert len(text) <= 500
    assert text.startswith("RuntimeError: ")


def test_side_effects_make_a_failure_partial_but_not_a_denial():
    completed = FailedOperation(
        operation_id=uuid4(),
        tool_name="jobs.submit_application",
        idempotency_key="k",
        status="completed",
        attempts=1,
        recovery=OperationRecovery.REPLAY_STORED_RECEIPT,
    )
    common = {"workflow_id": uuid4(), "thread_id": "t", "step": "s", "checkpoint_id": "c"}
    transient = assess_failure(
        **common, task_id="a", stored_error=repr(ProviderUnavailable("x")), operations=[completed]
    )
    assert transient.failure_class == FailureClass.PARTIAL_SIDE_EFFECT
    assert transient.cause_class == FailureClass.PROVIDER_TRANSIENT
    assert transient.retryable is True

    permanent = assess_failure(
        **common,
        task_id="b",
        stored_error=repr(ProviderPermanentFailure(provider="google")),
        operations=[completed],
    )
    assert permanent.failure_class == FailureClass.PROVIDER_PERMANENT
    assert permanent.status == FailureStatus.FAILED
    assert permanent.retryable is False
