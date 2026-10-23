"""Worker reliability: leases, backoff, dead-lettering, quarantine, failure events.

The acceptance criteria this file exists for:

1. **A worker killed mid-task has its lease expire, and another worker picks
   the task up.** `test_a_killed_worker_s_lease_expires_and_another_worker_takes_over`
   kills a real worker process with SIGKILL in the middle of a delivery;
   `test_a_cancelled_delivery_leaves_its_lease_to_lapse` does the same
   in-process, against a hand-moved clock.
2. **A transient failure retries with increasing delay and succeeds on a later
   attempt.** `test_a_transient_failure_backs_off_further_each_time_then_succeeds`.
3. **A poison payload is quarantined on first failure and surfaced as a
   recoverable failure.** `test_a_schema_invalid_event_is_quarantined_and_surfaced_for_retry`
   and `test_a_handler_rejecting_the_payload_quarantines_it_without_a_retry`.

Around those: the per-class retry budgets (reads, drafts, sends, calendar
writes), `Retry-After`, and the `tool.operation.failed` event `ToolExecutor`
stages when an action fails for good.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.worker.outbox_dispatcher import OutboxDispatcher
from personalos.bootstrap import build_policy_engine
from personalos.domain.errors import (
    InternalError,
    ProviderPermanentFailure,
    RetryableFailure,
    ValidationFailed,
)
from personalos.domain.events import ENVELOPE_KEY, EventEnvelope, EventType
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalVerdict,
)
from personalos.domain.models import OutboxEventStatus, ToolExecutionStatus
from personalos.domain.retry_policy import (
    CALENDAR_WRITE,
    DRAFT,
    EVENT_DELIVERY,
    READ,
    SEND,
    OperationClass,
    RetryPolicy,
    retry_after_seconds,
)
from personalos.executor.retry import dispatch_with_retry, retry_policy_for
from personalos.executor.tool_executor import ReconcileOutcome, Reconciliation, ToolExecutor
from personalos.persistence.event_failures import EVENT_RETRY_REQUESTED, event_target
from personalos.persistence.event_outbox import EventOutbox, FailureOutcome
from personalos.persistence.execution_ledger import ExecutionLedger
from personalos.persistence.models import (
    AuditEventModel,
    Base,
    OutboxEventModel,
    ToolExecutionModel,
    WorkflowModel,
)
from personalos.persistence.repositories import OutboxEventRepository
from personalos.persistence.workflow_commands import WORKFLOW_COMMAND_EVENT
from personalos.policy import ToolIntent
from personalos.tools.gateway import ToolResult

REPO_ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 10, 23, 9, 0)
TIMEOUT = timedelta(minutes=5)


# --- Fixtures -------------------------------------------------------------------


class Clock:
    """A clock a test moves by hand."""

    def __init__(self, now: datetime = T0):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "reliability.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def factory(db_path):
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    engine.dispose()


@pytest.fixture
def clock():
    return Clock()


def _outbox(factory, clock, **kwargs) -> EventOutbox:
    kwargs.setdefault("rand", lambda: 0.5)
    return EventOutbox(
        factory,
        exclude_types=(WORKFLOW_COMMAND_EVENT,),
        visibility_timeout=TIMEOUT,
        clock=clock,
        **kwargs,
    )


def _envelope(**overrides: Any) -> EventEnvelope:
    fields: dict[str, Any] = {
        "type": "test.happened",
        "source": "test",
        "occurred_at": T0,
        "dedupe_key": f"test:{uuid4()}",
        "payload": {},
    }
    fields.update(overrides)
    return EventEnvelope(**fields)


def _row(factory, outbox_id: UUID) -> OutboxEventModel:
    session = factory()
    try:
        row = session.get(OutboxEventModel, outbox_id)
        session.expunge(row)
        return row
    finally:
        session.close()


def _workflow(factory) -> UUID:
    session = factory()
    try:
        workflow = WorkflowModel(name="job_search")
        session.add(workflow)
        session.commit()
        return workflow.id
    finally:
        session.close()


def _api(db_path: Path) -> TestClient:
    engine = create_engine(f"sqlite:///{db_path}")
    api_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app = create_app()
    app.dependency_overrides[get_failure_services] = lambda: build_failure_services(api_factory)
    return TestClient(app)


class Recorder:
    """A handler that fails with each of `errors` in turn, then succeeds."""

    def __init__(self, *errors: BaseException):
        self.errors = list(errors)
        self.calls: list[EventEnvelope] = []

    async def __call__(self, envelope: EventEnvelope) -> None:
        self.calls.append(envelope)
        if self.errors:
            raise self.errors.pop(0)


# --- Retry policy -----------------------------------------------------------------


def test_each_operation_class_has_the_documented_budget():
    assert READ.max_retries == 3
    assert DRAFT.max_retries == 2
    assert SEND.max_retries == 0
    assert CALENDAR_WRITE.max_retries == 1 and CALENDAR_WRITE.requires_reconciliation
    assert not SEND.allows_retry(1)
    assert CALENDAR_WRITE.allows_retry(1) and not CALENDAR_WRITE.allows_retry(2)


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("jobs.search_jobs", OperationClass.READ),
        ("google.gmail_read_message", OperationClass.READ),
        ("google.gmail_create_draft", OperationClass.DRAFT),
        ("artifacts.create_draft", OperationClass.DRAFT),
        ("google.gmail_send_message", OperationClass.SEND),
        ("jobs.submit_application", OperationClass.SEND),
        ("files.overwrite_document", OperationClass.SEND),
        ("google.calendar_create_event", OperationClass.CALENDAR_WRITE),
        ("google.calendar_update_event", OperationClass.CALENDAR_WRITE),
        ("nobody.knows_this_tool", OperationClass.SEND),
    ],
)
def test_tools_get_the_policy_of_their_permission_class(tool, expected):
    assert retry_policy_for(tool).operation_class == expected


@pytest.mark.parametrize("jitter", [0.0, 0.25, 0.5, 0.99, 1.0])
def test_backoff_never_shrinks_and_stays_in_its_window(jitter):
    policy = RetryPolicy(OperationClass.READ, max_retries=8, base_delay=1.0, max_delay=40.0)
    delays = [policy.delay(n, rand=lambda: jitter) for n in range(1, 9)]
    assert delays == sorted(delays)
    for n, delay in enumerate(delays, start=1):
        window = min(40.0, 2 ** (n - 1))
        assert window / 2 <= delay <= window


def test_backoff_spreads_workers_that_failed_together():
    """Same retry, different draws: the jitter is real."""
    assert READ.delay(2, rand=lambda: 0.0) < READ.delay(2, rand=lambda: 1.0)


def test_retry_after_is_a_floor_not_a_ceiling():
    assert READ.delay(1, retry_after=7.5, rand=lambda: 0.5) == 7.5
    assert READ.delay(1, retry_after=0.0, rand=lambda: 1.0) == READ.base_delay


def test_retry_after_is_read_off_the_error():
    assert retry_after_seconds(RetryableFailure(retry_after=12)) == 12.0
    assert retry_after_seconds(RetryableFailure()) is None
    assert retry_after_seconds(RuntimeError("no header")) is None


def test_an_adapter_s_retry_after_reaches_the_raised_error():
    result = ToolResult(
        intent_id=uuid4(), tool_ref="jobs.search_jobs", success=False, error="429", retry_after=9
    )
    with pytest.raises(Exception) as raised:
        result.unwrap()
    assert retry_after_seconds(raised.value) == 9.0


# --- dispatch_with_retry --------------------------------------------------------------


class FlakyGateway:
    """Fails with each of `errors` in turn, then succeeds."""

    def __init__(self, *errors: Exception):
        self.errors = list(errors)
        self.dispatches = 0

    async def dispatch(self, intent: ToolIntent, approval=None) -> ToolResult:
        self.dispatches += 1
        if self.errors:
            raise self.errors.pop(0)
        return ToolResult(intent_id=intent.intent_id, tool_ref=intent.tool_ref, success=True)


def _tool_intent(server: str = "jobs", tool: str = "search_jobs") -> ToolIntent:
    return ToolIntent(server=server, tool=tool, arguments={"keywords": ["x"]}, requested_by="test")


async def test_a_read_backs_off_further_on_each_retry_and_succeeds():
    slept: list[float] = []

    async def sleep(delay: float) -> None:
        slept.append(delay)

    gateway = FlakyGateway(RetryableFailure(), RetryableFailure(), RetryableFailure())
    result = await dispatch_with_retry(gateway, _tool_intent(), sleep=sleep, rand=lambda: 0.5)

    assert result.success
    assert gateway.dispatches == 4
    assert len(slept) == 3 and slept[0] < slept[1] < slept[2]


async def test_a_read_gives_up_after_its_budget():
    gateway = FlakyGateway(*[RetryableFailure() for _ in range(10)])
    with pytest.raises(RetryableFailure):
        await dispatch_with_retry(gateway, _tool_intent(), sleep=_no_sleep)
    assert gateway.dispatches == READ.max_attempts


async def test_a_send_is_never_retried_blindly():
    gateway = FlakyGateway(RetryableFailure())
    with pytest.raises(RetryableFailure):
        await dispatch_with_retry(
            gateway, _tool_intent("google", "gmail_send_message"), sleep=_no_sleep
        )
    assert gateway.dispatches == 1


async def test_a_calendar_write_is_not_retried_where_it_cannot_be_reconciled():
    gateway = FlakyGateway(RetryableFailure())
    with pytest.raises(RetryableFailure):
        await dispatch_with_retry(
            gateway, _tool_intent("google", "calendar_create_event"), sleep=_no_sleep
        )
    assert gateway.dispatches == 1


async def test_a_draft_gets_two_retries():
    gateway = FlakyGateway(*[RetryableFailure() for _ in range(5)])
    with pytest.raises(RetryableFailure):
        await dispatch_with_retry(
            gateway, _tool_intent("google", "gmail_create_draft"), sleep=_no_sleep
        )
    assert gateway.dispatches == 3


async def test_dispatch_waits_out_a_retry_after():
    slept: list[float] = []

    async def sleep(delay: float) -> None:
        slept.append(delay)

    gateway = FlakyGateway(RetryableFailure(retry_after=4))
    await dispatch_with_retry(gateway, _tool_intent(), sleep=sleep)
    assert slept == [4.0]


async def test_dispatch_hands_back_a_retry_after_too_long_to_wait_out():
    gateway = FlakyGateway(RetryableFailure(retry_after=3600))
    with pytest.raises(RetryableFailure):
        await dispatch_with_retry(gateway, _tool_intent(), sleep=_no_sleep)
    assert gateway.dispatches == 1


async def _no_sleep(_delay: float) -> None:
    return None


# --- Acceptance 1: leases ---------------------------------------------------------------

_KILLABLE_WORKER = """
import asyncio, pathlib, sys
from datetime import timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from apps.worker.outbox_dispatcher import OutboxDispatcher
from personalos.persistence.event_outbox import EventOutbox

db, marker, lease = sys.argv[1], sys.argv[2], float(sys.argv[3])
engine = create_engine(f"sqlite:///{db}", connect_args={"timeout": 30})
factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def hang(envelope):
    pathlib.Path(marker).write_text(str(envelope.event_id))
    await asyncio.sleep(3600)

outbox = EventOutbox(factory, visibility_timeout=timedelta(seconds=lease))
asyncio.run(OutboxDispatcher(outbox, {}, default=hang).process_next())
"""


def test_a_killed_worker_s_lease_expires_and_another_worker_takes_over(factory, db_path, tmp_path):
    """Acceptance 1, with a real process killed with SIGKILL mid-delivery."""
    lease = timedelta(seconds=2)
    envelope = _envelope()
    EventOutbox(factory).publish(envelope)

    marker = tmp_path / "claimed"
    worker = subprocess.Popen(
        [sys.executable, "-c", _KILLABLE_WORKER, str(db_path), str(marker), str(lease.seconds)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert worker.poll() is None, "the worker exited before claiming the event"
            assert time.monotonic() < deadline, "the worker never claimed the event"
            time.sleep(0.05)
    finally:
        worker.send_signal(signal.SIGKILL)
        worker.wait(timeout=30)

    # Killed mid-handler: nothing settled the row, so it is still leased.
    killed_claim = _row(factory, envelope.event_id)
    assert killed_claim.status == OutboxEventStatus.IN_PROGRESS.value
    assert killed_claim.attempts == 1

    survivor = Recorder()
    dispatcher = OutboxDispatcher(
        EventOutbox(factory, visibility_timeout=lease), {}, default=survivor
    )
    deadline = time.monotonic() + 30
    while not asyncio.run(dispatcher.process_next()):
        assert time.monotonic() < deadline, "the lease never expired"
        time.sleep(0.1)
    taken_over_at = datetime.utcnow()

    assert [event.event_id for event in survivor.calls] == [envelope.event_id]
    assert taken_over_at >= killed_claim.claimed_at + lease
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DISPATCHED.value
    assert row.attempts == 2


def test_a_cancelled_delivery_leaves_its_lease_to_lapse(factory, clock):
    """Acceptance 1, in-process: a delivery cancelled mid-handler settles nothing."""
    envelope = _envelope()
    _outbox(factory, clock).publish(envelope)
    started = asyncio.Event()

    async def hang(event: EventEnvelope) -> None:
        started.set()
        await asyncio.sleep(3600)

    async def kill_mid_delivery() -> None:
        task = asyncio.create_task(
            OutboxDispatcher(_outbox(factory, clock), {}, default=hang).process_next()
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(kill_mid_delivery())
    assert _row(factory, envelope.event_id).status == OutboxEventStatus.IN_PROGRESS.value

    survivor = Recorder()
    dispatcher = OutboxDispatcher(_outbox(factory, clock), {}, default=survivor)
    clock.advance(TIMEOUT - timedelta(seconds=1))
    assert asyncio.run(dispatcher.process_next()) is False, "the lease was still live"

    clock.advance(timedelta(seconds=2))
    assert asyncio.run(dispatcher.process_next()) is True
    assert len(survivor.calls) == 1
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DISPATCHED.value
    assert row.attempts == 2


def test_an_event_that_keeps_killing_its_worker_is_dead_lettered(factory, clock):
    """Lapsed leases count against the budget, so a crash loop ends."""
    envelope = _envelope()
    outbox = _outbox(factory, clock, max_attempts=2)
    outbox.publish(envelope)

    for _ in range(2):
        assert outbox.claim() is not None  # ...and the worker dies
        clock.advance(TIMEOUT + timedelta(seconds=1))

    assert outbox.claim() is None
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DEAD_LETTER.value
    assert "lease lapsed" in row.last_error


# --- Acceptance 2: backoff ---------------------------------------------------------------


def test_a_transient_failure_backs_off_further_each_time_then_succeeds(factory, clock):
    """Acceptance 2: each retry waits longer than the last; the third delivery lands."""
    envelope = _envelope()
    outbox = _outbox(factory, clock)
    outbox.publish(envelope)
    handler = Recorder(RetryableFailure("provider 503"), RetryableFailure("provider 503"))
    dispatcher = OutboxDispatcher(outbox, {}, default=handler)

    delays: list[timedelta] = []
    for _ in range(2):
        assert asyncio.run(dispatcher.process_next()) is True
        row = _row(factory, envelope.event_id)
        assert row.status == OutboxEventStatus.PENDING.value
        assert row.last_error == "provider 503"
        delays.append(row.available_at - clock.now)

        # Not a moment early.
        clock.now = row.available_at - timedelta(milliseconds=1)
        assert asyncio.run(dispatcher.process_next()) is False
        clock.now = row.available_at

    assert delays[0] < delays[1]
    assert asyncio.run(dispatcher.process_next()) is True
    assert len(handler.calls) == 3
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DISPATCHED.value
    assert row.attempts == 3


def test_backoff_honors_a_handler_s_retry_after(factory, clock):
    envelope = _envelope()
    outbox = _outbox(factory, clock)
    outbox.publish(envelope)
    dispatcher = OutboxDispatcher(
        outbox, {}, default=Recorder(RetryableFailure("rate limited", retry_after=900))
    )

    assert asyncio.run(dispatcher.process_next()) is True
    assert _row(factory, envelope.event_id).available_at == T0 + timedelta(seconds=900)


def test_a_transient_failure_dead_letters_once_its_retries_are_spent(factory, clock):
    envelope = _envelope()
    outbox = _outbox(factory, clock)
    outbox.publish(envelope)
    handler = Recorder(*[RetryableFailure(f"down, try {n}") for n in range(10)])
    dispatcher = OutboxDispatcher(outbox, {}, default=handler)

    for _ in range(EVENT_DELIVERY.max_attempts):
        assert asyncio.run(dispatcher.process_next()) is True
        row = _row(factory, envelope.event_id)
        if row.available_at:
            clock.now = row.available_at

    assert len(handler.calls) == EVENT_DELIVERY.max_attempts
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DEAD_LETTER.value
    assert row.last_error == f"down, try {EVENT_DELIVERY.max_attempts - 1}"
    assert asyncio.run(dispatcher.process_next()) is False


def test_a_terminal_failure_is_dead_lettered_without_retrying(factory, clock):
    envelope = _envelope()
    outbox = _outbox(factory, clock)
    outbox.publish(envelope)
    claimed = outbox.claim()

    failure = outbox.fail(claimed, ProviderPermanentFailure(provider="gmail"))

    assert failure.outcome == FailureOutcome.DEAD_LETTERED
    assert _row(factory, envelope.event_id).status == OutboxEventStatus.DEAD_LETTER.value


def test_a_late_failure_from_a_lapsed_claim_writes_nothing(factory, clock):
    envelope = _envelope()
    outbox = _outbox(factory, clock)
    outbox.publish(envelope)
    stale = outbox.claim()
    clock.advance(TIMEOUT + timedelta(seconds=1))
    assert outbox.claim() is not None

    assert outbox.fail(stale, RetryableFailure()).outcome == FailureOutcome.CLAIM_LOST
    assert _row(factory, envelope.event_id).status == OutboxEventStatus.IN_PROGRESS.value


# --- Acceptance 3: poison events -----------------------------------------------------------


def _stage_raw(factory, payload: dict, *, type: str = "test.happened") -> UUID:
    session = factory()
    try:
        return OutboxEventRepository(session).create(type=type, payload=payload).id
    finally:
        session.close()


def test_a_schema_invalid_event_is_quarantined_and_surfaced_for_retry(factory, clock, db_path):
    """Acceptance 3: quarantined on its first claim, listed as recoverable, retried by hand."""
    workflow_id = _workflow(factory)
    poison_id = _stage_raw(
        factory,
        {
            ENVELOPE_KEY: {
                "type": "test.happened",
                "source": "",  # blank: violates the envelope contract
                "occurred_at": T0.isoformat(),
                "dedupe_key": "test:poison",
                "payload": {"workflow_id": str(workflow_id)},
            }
        },
    )
    healthy = _envelope()
    _outbox(factory, clock).publish(healthy)
    handler = Recorder()
    dispatcher = OutboxDispatcher(_outbox(factory, clock), {}, default=handler)

    # The poison row is skipped past on the way to the healthy one, never handed out.
    assert asyncio.run(dispatcher.process_next()) is True
    assert [event.event_id for event in handler.calls] == [healthy.event_id]
    row = _row(factory, poison_id)
    assert row.status == OutboxEventStatus.QUARANTINED.value
    assert row.attempts == 1
    assert "source" in row.last_error
    assert row.workflow_id == workflow_id
    assert asyncio.run(dispatcher.process_next()) is False

    client = _api(db_path)
    listed = client.get("/v1/failures/events").json()["failures"]
    assert [failure["failure_id"] for failure in listed] == [str(poison_id)]
    (failure,) = listed
    assert failure["disposition"] == "quarantined"
    assert failure["retryable"] is True
    assert failure["actions"] == ["retry"]
    assert failure["status"] == "failed"
    assert failure["workflow_id"] == str(workflow_id)

    retried = client.post(
        f"/v1/failures/events/{poison_id}/retry", headers={"X-Actor-Id": "operator"}
    )
    assert retried.status_code == 202, retried.text
    assert retried.json()["status"] == "retry_queued"
    row = _row(factory, poison_id)
    assert row.status == OutboxEventStatus.PENDING.value
    assert row.attempts == 0
    assert client.get("/v1/failures/events").json()["failures"] == []

    session = factory()
    try:
        (audit,) = (
            session.query(AuditEventModel)
            .filter(AuditEventModel.action == EVENT_RETRY_REQUESTED)
            .all()
        )
        assert audit.actor == "operator"
        assert audit.target_ref == event_target(poison_id)
        assert audit.workflow_id == workflow_id
    finally:
        session.close()

    # Still malformed: quarantined again on its next claim, not retried in a loop.
    assert asyncio.run(dispatcher.process_next()) is False
    assert _row(factory, poison_id).status == OutboxEventStatus.QUARANTINED.value


def test_a_handler_rejecting_the_payload_quarantines_it_without_a_retry(factory, clock, db_path):
    """Acceptance 3, handler side: a `ValidationFailed` is poison, not transient."""
    workflow_id = _workflow(factory)
    envelope = _envelope(payload={"workflow_id": str(workflow_id)})
    _outbox(factory, clock).publish(envelope)
    handler = Recorder(ValidationFailed("payload is missing application_id"))
    dispatcher = OutboxDispatcher(_outbox(factory, clock), {}, default=handler)

    assert asyncio.run(dispatcher.process_next()) is True
    clock.advance(timedelta(days=1))
    assert asyncio.run(dispatcher.process_next()) is False
    assert len(handler.calls) == 1
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.QUARANTINED.value
    assert row.last_error == "payload is missing application_id"

    # Surfaced on the workflow's own failures view...
    client = _api(db_path)
    services = build_failure_services(factory)
    (failure,) = services.events.failures(workflow_id=workflow_id)
    assert failure.failure_id == envelope.event_id
    assert failure.event_id == envelope.event_id
    assert failure.disposition.value == "quarantined"
    assert services.events.failures(workflow_id=uuid4()) == []

    # ...and once the handler is fixed, a manual retry delivers it.
    assert client.post(f"/v1/failures/events/{envelope.event_id}/retry").status_code == 202
    assert asyncio.run(dispatcher.process_next()) is True
    assert len(handler.calls) == 2
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DISPATCHED.value
    assert row.last_error == "payload is missing application_id"


def test_only_a_parked_event_can_be_retried(factory, clock, db_path):
    envelope = _envelope()
    _outbox(factory, clock).publish(envelope)
    client = _api(db_path)

    pending = client.post(f"/v1/failures/events/{envelope.event_id}/retry")
    assert pending.status_code == 409, pending.text
    missing = client.post(f"/v1/failures/events/{uuid4()}/retry")
    assert missing.status_code == 404, missing.text


# --- ToolExecutor: per-class retries and tool.operation.failed ------------------------------


class FlakyProvider:
    """Fails with each of `errors` in turn, then performs the action."""

    def __init__(self, *errors: Exception):
        self.errors = list(errors)
        self.attempts = 0
        self.performed: list[str] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        self.performed.append(intent.idempotency_key)
        return ActionReceipt(action_id=intent.action_id, ok=True, external_reference="ext-1")


class ScriptedReconciler:
    def __init__(self, outcome: ReconcileOutcome):
        self.outcome = outcome
        self.asked = 0

    async def reconcile(self, intent: ActionIntent) -> Reconciliation:
        self.asked += 1
        receipt = None
        if self.outcome == ReconcileOutcome.APPLIED:
            receipt = ActionReceipt(
                action_id=intent.action_id, ok=True, external_reference="ext-found"
            )
        return Reconciliation(self.outcome, receipt)


def _calendar_intent() -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.CREATE_CALENDAR_EVENT,
        target="calendar: new event 'Interview: Acme'",
        summary="Add 'Interview: Acme' to the calendar",
        payload={"title": "Interview: Acme"},
        idempotency_key=f"cal-create:{uuid4()}",
    )


def _send_intent() -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.SEND_RECRUITER_MESSAGE,
        target="recruiter@example.test",
        summary="Reply to the Acme recruiter",
        payload={"body": "Thanks!"},
        idempotency_key=f"send:{uuid4()}",
    )


def _approve(intent: ActionIntent) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
    )


def _executor(factory, provider, *, reconciler=None, slept=None, workflow_id=None):
    async def sleep(delay: float) -> None:
        if slept is not None:
            slept.append(delay)

    return ToolExecutor(
        provider,
        build_policy_engine(factory),
        ExecutionLedger(factory, workflow_id=workflow_id),
        reconciler=reconciler,
        sleep=sleep,
        rand=lambda: 0.5,
    )


def _execution(factory, key: str) -> ToolExecutionModel:
    session = factory()
    try:
        row = session.query(ToolExecutionModel).filter_by(idempotency_key=key).one()
        session.expunge(row)
        return row
    finally:
        session.close()


def _failure_events(factory) -> list[EventEnvelope]:
    session = factory()
    try:
        rows = session.query(OutboxEventModel).filter_by(type=EventType.TOOL_OPERATION_FAILED)
        return [EventEnvelope.model_validate(row.payload_json[ENVELOPE_KEY]) for row in rows]
    finally:
        session.close()


async def test_a_failed_send_is_not_retried_and_announces_its_failure(factory):
    workflow_id = _workflow(factory)
    provider = FlakyProvider(RetryableFailure("smtp timed out"))
    intent = _send_intent()

    with pytest.raises(RetryableFailure):
        await _executor(factory, provider, workflow_id=workflow_id).execute(
            intent, _approve(intent)
        )

    assert provider.attempts == 1
    execution = _execution(factory, intent.idempotency_key)
    assert execution.status == ToolExecutionStatus.FAILED.value
    (event,) = _failure_events(factory)
    assert event.source == "tool_executor"
    assert event.subject_ref == f"tool_execution:{execution.operation_id}"
    assert event.payload == {
        "kind": "send_recruiter_message",
        "tool": "google.gmail_send_message",
        "target": "recruiter@example.test",
        "idempotency_key": intent.idempotency_key,
        "workflow_id": str(workflow_id),
        "operation_class": "send",
        "failure_class": "unclassified",
        "error_code": "retryable",
        "retryable": True,
        "error": "smtp timed out",
        "operation_id": str(execution.operation_id),
        "attempts": 1,
    }


async def test_a_failure_event_is_dispatched_like_any_other(factory, clock):
    intent = _send_intent()
    with pytest.raises(InternalError):
        await _executor(factory, FlakyProvider(InternalError("boom"))).execute(
            intent, _approve(intent)
        )

    handler = Recorder()
    dispatcher = OutboxDispatcher(
        _outbox(factory, clock), {EventType.TOOL_OPERATION_FAILED: handler}
    )
    assert await dispatcher.drain() == 1
    assert handler.calls[0].payload["failure_class"] == "unclassified"
    assert handler.calls[0].payload["retryable"] is False


async def test_a_calendar_write_is_retried_once_after_reconciliation_says_it_did_not_land(factory):
    slept: list[float] = []
    provider = FlakyProvider(RetryableFailure("calendar 503", retry_after=3))
    reconciler = ScriptedReconciler(ReconcileOutcome.NOT_APPLIED)
    intent = _calendar_intent()

    receipt = await _executor(factory, provider, reconciler=reconciler, slept=slept).execute(
        intent, _approve(intent)
    )

    assert receipt.ok and receipt.external_reference == "ext-1"
    assert provider.attempts == 2
    assert reconciler.asked == 1
    assert slept == [3.0]
    execution = _execution(factory, intent.idempotency_key)
    assert execution.status == ToolExecutionStatus.COMPLETED.value
    assert execution.attempts == 2
    assert _failure_events(factory) == []


async def test_a_calendar_write_found_at_the_provider_is_recorded_not_retried(factory):
    provider = FlakyProvider(RetryableFailure("timed out reading the response"))
    reconciler = ScriptedReconciler(ReconcileOutcome.APPLIED)
    intent = _calendar_intent()

    receipt = await _executor(factory, provider, reconciler=reconciler).execute(
        intent, _approve(intent)
    )

    assert receipt.external_reference == "ext-found"
    assert provider.attempts == 1
    assert _execution(factory, intent.idempotency_key).status == ToolExecutionStatus.COMPLETED.value


@pytest.mark.parametrize("reconciler", [None, ScriptedReconciler(ReconcileOutcome.INDETERMINATE)])
async def test_a_calendar_write_is_not_retried_without_a_confirmed_miss(factory, reconciler):
    provider = FlakyProvider(RetryableFailure("calendar 503"))
    intent = _calendar_intent()

    with pytest.raises(RetryableFailure):
        await _executor(factory, provider, reconciler=reconciler).execute(intent, _approve(intent))

    assert provider.attempts == 1
    (event,) = _failure_events(factory)
    assert event.payload["operation_class"] == "calendar_write"


async def test_a_calendar_write_gets_one_retry_and_no_more(factory):
    provider = FlakyProvider(RetryableFailure("503"), RetryableFailure("503 again"))
    intent = _calendar_intent()

    with pytest.raises(RetryableFailure):
        await _executor(
            factory, provider, reconciler=ScriptedReconciler(ReconcileOutcome.NOT_APPLIED)
        ).execute(intent, _approve(intent))

    assert provider.attempts == 2
    (event,) = _failure_events(factory)
    assert event.payload["attempts"] == 2
    assert event.payload["error"] == "503 again"
