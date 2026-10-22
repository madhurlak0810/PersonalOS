"""Durable async event dispatch through `outbox_events`.

Covers the transactional-outbox path end to end:

- the `EventEnvelope` contract (normalized, minimal, secret-free payloads);
- staging an event in the producer's own transaction, so the event exists if
  and only if the mutation committed;
- concurrent dispatchers never delivering the same row twice (SQLite here;
  the `SKIP LOCKED` variant runs against Postgres when one is reachable);
- a duplicate event (same `dedupe_key`) producing exactly one application
  transition;
- a dispatcher that crashes between claiming and acknowledging: the row is
  redelivered once its lease lapses, and the idempotent consumer applies
  nothing twice;
- routing: an event resumes the correlated workflow waiting on it
  (`Command(resume=...)`), or starts an event-handler run when none is.
"""

import asyncio
import operator
import threading
import time
from datetime import datetime, timedelta
from typing import Annotated, Any, TypedDict
from uuid import UUID, uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from apps.worker.event_routing import (
    APPLICATION_TRANSITION_CONSUMER,
    ApplicationTransitionHandler,
    EventRouter,
    RouteKind,
)
from apps.worker.outbox_dispatcher import OutboxDispatcher, build_outbox_dispatcher
from apps.worker.workflow_commands import process_next as process_next_command
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    build_correlated_wait_lookup,
    build_durable_checkpointer,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_event_handler_thread,
)
from personalos.config import settings
from personalos.domain.events import ENVELOPE_KEY, EventEnvelope, EventType
from personalos.domain.models import ApplicationStatus, OutboxEventStatus
from personalos.domain.workflow import (
    EVENT_HANDLER_THREAD_NAMESPACE,
    WorkflowCommandKind,
    WorkflowThread,
    derive_thread_id,
)
from personalos.persistence.event_inbox import IdempotentConsumer
from personalos.persistence.event_outbox import (
    LEGACY_SOURCE,
    EventOutbox,
    envelope_from_row,
    stage_event,
)
from personalos.persistence.models import (
    Base,
    ConsumedEventModel,
    EventLogModel,
    OutboxEventModel,
    UserModel,
)
from personalos.persistence.repositories import (
    ApplicationRepository,
    JobPostingRepository,
    OutboxEventRepository,
)
from personalos.persistence.workflow_commands import WORKFLOW_COMMAND_EVENT

T0 = datetime(2026, 10, 22, 9, 0)
TIMEOUT = timedelta(minutes=5)


# --- Fixtures -------------------------------------------------------------------


class Clock:
    """A clock a test moves by hand, so a lease can lapse without waiting for it."""

    def __init__(self, now: datetime = T0):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def factory(tmp_path):
    # A busy timeout so concurrent sessions queue on SQLite's file lock instead
    # of failing with "database is locked".
    engine = create_engine(f"sqlite:///{tmp_path / 'outbox.db'}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    engine.dispose()


@pytest.fixture
def clock():
    return Clock()


def _outbox(factory, clock, **kwargs) -> EventOutbox:
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


def _make_application(factory) -> UUID:
    session = factory()
    try:
        user = UserModel(email=f"candidate-{uuid4()}@example.com")
        session.add(user)
        session.commit()
        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Staff Engineer",
            company="Acme",
            description_hash="a" * 64,
            dedupe_key=f"acme:staff-engineer:{uuid4()}",
        )
        return ApplicationRepository(session).create(job_posting_id=posting.id, user_id=user.id).id
    finally:
        session.close()


def _transition_request(application_id: UUID, *, dedupe_key: str, **overrides) -> EventEnvelope:
    return _envelope(
        type=EventType.APPLICATION_TRANSITION_REQUESTED,
        source="api",
        subject_ref=f"application:{application_id}",
        dedupe_key=dedupe_key,
        payload={"application_id": str(application_id), "status": "saved", "reason": "liked it"},
        **overrides,
    )


def _status(factory, application_id: UUID) -> str:
    session = factory()
    try:
        return ApplicationRepository(session).get_by_id(application_id).status
    finally:
        session.close()


def _transitions(factory, application_id: UUID) -> int:
    session = factory()
    try:
        return (
            session.query(EventLogModel)
            .filter(
                EventLogModel.aggregate_id == application_id,
                EventLogModel.event_type == "application.status_changed",
            )
            .count()
        )
    finally:
        session.close()


def _row(factory, outbox_id: UUID) -> OutboxEventModel:
    session = factory()
    try:
        row = session.get(OutboxEventModel, outbox_id)
        session.expunge(row)
        return row
    finally:
        session.close()


def _transition_handler(factory) -> ApplicationTransitionHandler:
    return ApplicationTransitionHandler(
        IdempotentConsumer(factory, APPLICATION_TRANSITION_CONSUMER)
    )


# --- The envelope ---------------------------------------------------------------


def test_envelope_payload_is_redacted_and_normalized():
    envelope = _envelope(
        payload={
            "application_id": uuid4(),
            "at": datetime(2026, 1, 1),
            "access_token": "ya29.super-secret-token-value",
        }
    )
    assert envelope.payload["access_token"] == "[REDACTED]"
    # Round-tripped through JSON: what a consumer in another process reads.
    assert isinstance(envelope.payload["application_id"], str)
    assert isinstance(envelope.payload["at"], str)


@pytest.mark.parametrize("field", ["type", "source", "dedupe_key"])
def test_envelope_rejects_blank_identity_fields(field):
    with pytest.raises(ValueError):
        _envelope(**{field: "   "})


def test_envelope_rejects_a_document_sized_payload():
    with pytest.raises(ValueError, match="reference"):
        _envelope(payload={"body": "x" * 20_000})


def test_envelope_subject_names_the_id_of_its_kind():
    application_id = uuid4()
    envelope = _envelope(subject_ref=f"application:{application_id}")
    assert envelope.subject("application") == str(application_id)
    assert envelope.subject("job") is None


# --- Publishing: one transaction with the mutation -------------------------------


def test_event_is_committed_with_the_mutation_it_describes(factory):
    application_id = _make_application(factory)
    envelope = _transition_request(application_id, dedupe_key="move:1")

    session = factory()
    try:
        ApplicationRepository(session).update_status(
            application_id, ApplicationStatus.SAVED, commit=False
        )
        stage_event(session, envelope)
        session.commit()
    finally:
        session.close()

    assert _status(factory, application_id) == ApplicationStatus.SAVED.value
    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.PENDING.value
    assert row.dedupe_key == "move:1"
    assert envelope_from_row(row) == envelope


def test_rolled_back_mutation_leaves_no_event(factory):
    application_id = _make_application(factory)
    session = factory()
    try:
        ApplicationRepository(session).update_status(
            application_id, ApplicationStatus.SAVED, commit=False
        )
        stage_event(session, _transition_request(application_id, dedupe_key="move:1"))
        session.rollback()
        assert session.query(OutboxEventModel).count() == 0
    finally:
        session.close()
    assert _status(factory, application_id) == ApplicationStatus.DISCOVERED.value


def test_restaging_the_same_fact_is_a_no_op(factory):
    first = _envelope(dedupe_key="fact:1")
    retry = _envelope(dedupe_key="fact:1")  # a producer re-run mints a new event_id
    session = factory()
    try:
        stage_event(session, first)
        session.commit()
        assert stage_event(session, retry).id == first.event_id
        session.commit()
        assert session.query(OutboxEventModel).count() == 1
    finally:
        session.close()


def test_rows_staged_without_an_envelope_are_read_as_one(factory):
    session = factory()
    try:
        row = OutboxEventRepository(session).create(
            type="application.created", payload={"application_id": "a1"}
        )
        envelope = envelope_from_row(row)
    finally:
        session.close()
    assert envelope.event_id == row.id
    assert envelope.type == "application.created"
    assert envelope.source == LEGACY_SOURCE
    assert envelope.dedupe_key == f"outbox:{row.id}"
    assert envelope.payload == {"application_id": "a1"}


# --- Claiming: concurrent workers never share a row -------------------------------


def _race_dispatchers(factory, outbox_for, *, workers: int, rows: int) -> list[list[UUID]]:
    """Run `workers` dispatchers on their own threads until the outbox is empty."""
    for n in range(rows):
        _outbox(factory, Clock()).publish(_envelope(dedupe_key=f"race:{n}"))

    seen: list[list[UUID]] = [[] for _ in range(workers)]
    errors: list[BaseException] = []
    barrier = threading.Barrier(workers)

    def work(index: int) -> None:
        async def record(envelope: EventEnvelope) -> None:
            seen[index].append(envelope.event_id)
            # Hold the row for a moment, blocking this thread, so the other
            # workers poll while it is claimed rather than after it is settled.
            time.sleep(0.005)

        try:
            barrier.wait(timeout=10)
            asyncio.run(OutboxDispatcher(outbox_for(), {}, default=record).drain())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    return seen


def _assert_each_row_delivered_once(factory, seen: list[list[UUID]], rows: int) -> None:
    delivered = [event_id for worker in seen for event_id in worker]
    assert len(delivered) == rows
    assert len(set(delivered)) == rows, "a row was delivered to more than one worker"
    # Otherwise the race was never run: one worker drained the outbox alone.
    assert sum(1 for worker in seen if worker) > 1
    session = factory()
    try:
        statuses = {status for (status,) in session.query(OutboxEventModel.status)}
    finally:
        session.close()
    assert statuses == {OutboxEventStatus.DISPATCHED.value}


def test_concurrent_workers_never_process_the_same_row(factory):
    seen = _race_dispatchers(factory, lambda: _outbox(factory, Clock()), workers=4, rows=40)
    _assert_each_row_delivered_once(factory, seen, rows=40)


@pytest.fixture
def postgres_factory():
    """A session factory over a scratch schema on the configured Postgres, or a skip."""
    engine = create_engine(settings.database_url, pool_size=10)
    if engine.dialect.name != "postgresql":
        pytest.skip("SKIP LOCKED test requires a Postgres DATABASE_URL")
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip(f"No reachable Postgres at {settings.database_url!r}")
    schema = f"outbox_test_{uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    scoped = engine.execution_options(schema_translate_map={None: schema})
    Base.metadata.create_all(
        scoped, tables=[OutboxEventModel.__table__, ConsumedEventModel.__table__]
    )
    try:
        yield sessionmaker(autocommit=False, autoflush=False, bind=scoped)
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        engine.dispose()


def test_concurrent_workers_skip_locked_rows_on_postgres(postgres_factory):
    seen = _race_dispatchers(
        postgres_factory, lambda: _outbox(postgres_factory, Clock()), workers=4, rows=40
    )
    _assert_each_row_delivered_once(postgres_factory, seen, rows=40)


def test_dispatcher_leaves_workflow_commands_to_their_own_worker(factory, clock):
    session = factory()
    try:
        OutboxEventRepository(session).create(type=WORKFLOW_COMMAND_EVENT, payload={})
    finally:
        session.close()
    assert _outbox(factory, clock).claim() is None


# --- Consuming: a duplicate applies once --------------------------------------------


def test_duplicate_event_results_in_exactly_one_transition(factory):
    application_id = _make_application(factory)
    handler = _transition_handler(factory)

    # Same fact, two deliveries -- even with distinct event ids, as a producer
    # retry would mint.
    first = asyncio.run(handler(_transition_request(application_id, dedupe_key="move:1")))
    second = asyncio.run(handler(_transition_request(application_id, dedupe_key="move:1")))

    assert first.applied and not second.applied
    assert _status(factory, application_id) == ApplicationStatus.SAVED.value
    assert _transitions(factory, application_id) == 1


def test_racing_duplicate_deliveries_apply_once(factory):
    application_id = _make_application(factory)
    envelope = _transition_request(application_id, dedupe_key="move:1")
    results: list = []
    errors: list = []
    barrier = threading.Barrier(2)

    def deliver() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(asyncio.run(_transition_handler(factory)(envelope)))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=deliver) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    # The loser either saw the inbox row, or lost the commit to it; either way
    # nothing of its delivery survives. The only acceptable error is the
    # guarded status UPDATE refusing a move made under it, which also rolls
    # back -- and the outbox would simply redeliver it.
    applied = [r for r in results if r.applied]
    assert len(applied) == 1
    assert all(type(e).__name__ == "ApplicationTransitionConflict" for e in errors)
    assert _transitions(factory, application_id) == 1


def test_refused_transition_is_consumed_not_retried(factory):
    application_id = _make_application(factory)
    envelope = _transition_request(application_id, dedupe_key="move:offer")
    envelope = envelope.model_copy(
        update={"payload": {"application_id": str(application_id), "status": "offer"}}
    )
    result = asyncio.run(_transition_handler(factory)(envelope))
    assert result.applied and result.result is None
    assert _status(factory, application_id) == ApplicationStatus.DISCOVERED.value


def test_duplicate_through_the_dispatcher_moves_the_application_once(factory, clock):
    application_id = _make_application(factory)
    outbox = _outbox(factory, clock)
    dispatcher = OutboxDispatcher(
        outbox, {EventType.APPLICATION_TRANSITION_REQUESTED: _transition_handler(factory)}
    )
    outbox.publish(_transition_request(application_id, dedupe_key="move:1"))
    assert asyncio.run(dispatcher.drain()) == 1

    # The producer runs again after a crash it could not tell apart from failure.
    outbox.publish(_transition_request(application_id, dedupe_key="move:1"))
    assert asyncio.run(dispatcher.drain()) == 0

    assert _transitions(factory, application_id) == 1


# --- Crashing between claim and ack ---------------------------------------------------


def test_crash_after_effect_before_ack_is_redelivered_without_a_second_effect(factory, clock):
    application_id = _make_application(factory)
    handler = _transition_handler(factory)
    envelope = _transition_request(application_id, dedupe_key="move:1")
    _outbox(factory, clock).publish(envelope)

    # Worker A claims and applies the event, then dies before acknowledging it.
    crashed = _outbox(factory, clock)
    claimed = crashed.claim()
    assert claimed is not None and claimed.attempt == 1
    assert asyncio.run(handler(claimed.envelope)).applied

    # While A's lease is live, nobody else is handed the row.
    survivor = OutboxDispatcher(
        _outbox(factory, clock), {EventType.APPLICATION_TRANSITION_REQUESTED: handler}
    )
    assert asyncio.run(survivor.process_next()) is False

    # Once it lapses, worker B gets it again, and the consumer applies nothing.
    clock.advance(TIMEOUT + timedelta(seconds=1))
    assert asyncio.run(survivor.process_next()) is True

    row = _row(factory, envelope.event_id)
    assert row.status == OutboxEventStatus.DISPATCHED.value
    assert row.attempts == 2
    assert _status(factory, application_id) == ApplicationStatus.SAVED.value
    assert _transitions(factory, application_id) == 1

    # A's late ack is refused: the claim it holds is no longer the row's.
    assert crashed.ack(claimed) is False


def test_crash_before_effect_is_redelivered_and_applied_once(factory, clock):
    application_id = _make_application(factory)
    envelope = _transition_request(application_id, dedupe_key="move:1")
    _outbox(factory, clock).publish(envelope)
    assert _outbox(factory, clock).claim() is not None  # claimed, then the worker dies

    clock.advance(TIMEOUT + timedelta(seconds=1))
    dispatcher = OutboxDispatcher(
        _outbox(factory, clock),
        {EventType.APPLICATION_TRANSITION_REQUESTED: _transition_handler(factory)},
    )
    assert asyncio.run(dispatcher.drain()) == 1
    assert _transitions(factory, application_id) == 1
    assert _row(factory, envelope.event_id).status == OutboxEventStatus.DISPATCHED.value


def test_failing_handler_retries_after_the_lease_then_gives_up(factory, clock):
    envelope = _envelope()
    outbox = _outbox(factory, clock, max_attempts=2)
    outbox.publish(envelope)
    calls: list[int] = []

    async def broken(event: EventEnvelope) -> None:
        calls.append(1)
        raise RuntimeError("downstream is down")

    dispatcher = OutboxDispatcher(outbox, {}, default=broken)
    assert asyncio.run(dispatcher.process_next()) is True
    # Not retried on the spot: the lease spaces retries out.
    assert asyncio.run(dispatcher.process_next()) is False
    assert _row(factory, envelope.event_id).status == OutboxEventStatus.IN_PROGRESS.value

    clock.advance(TIMEOUT + timedelta(seconds=1))
    assert asyncio.run(dispatcher.process_next()) is True
    assert len(calls) == 2
    assert _row(factory, envelope.event_id).status == OutboxEventStatus.FAILED.value


def test_malformed_event_fails_at_once(factory, clock):
    application_id = _make_application(factory)
    outbox = _outbox(factory, clock)
    bad = _transition_request(application_id, dedupe_key="move:bad").model_copy(
        update={"payload": {"status": "not-a-status"}, "subject_ref": None}
    )
    outbox.publish(bad)
    dispatcher = OutboxDispatcher(
        outbox, {EventType.APPLICATION_TRANSITION_REQUESTED: _transition_handler(factory)}
    )
    assert asyncio.run(dispatcher.process_next()) is True
    assert _row(factory, bad.event_id).status == OutboxEventStatus.FAILED.value


# --- Routing: resume the waiting workflow, or start a run ------------------------------


class _State(TypedDict, total=False):
    mode: str
    event: dict
    handled: Annotated[list, operator.add]


class EventGraph:
    """A graph that either waits for an event at an interrupt, or handles one it was started with."""

    def __init__(self, factory):
        self.registry = build_workflow_thread_registry(factory)
        graph = StateGraph(_State)
        graph.add_node("park", self._park)
        graph.add_node("handle", self._handle)
        graph.add_edge(START, "park")
        graph.add_edge("park", "handle")
        graph.add_edge("handle", END)
        self.graph = graph.compile(checkpointer=build_durable_checkpointer(factory, self.registry))
        self.runner = DurableWorkflowRunner(
            self.graph, registry=self.registry, leases=build_workflow_lease_store(factory)
        )
        self.commands = build_workflow_command_queue(factory)
        self.router = EventRouter(
            self.commands,
            build_correlated_wait_lookup(factory, self.registry),
            lambda envelope: register_event_handler_thread(envelope, registry=self.registry),
        )

    @staticmethod
    def _park(state: _State) -> dict[str, Any]:
        if state.get("mode") != "wait":
            return {}
        answer = interrupt({"waiting_for": "recruiter reply"})
        return {"event": answer["event"]}

    @staticmethod
    def _handle(state: _State) -> dict[str, Any]:
        return {"handled": [state["event"]["dedupe_key"]]}

    def run_commands(self) -> int:
        ran = 0
        while asyncio.run(
            process_next_command(self.commands, {EVENT_HANDLER_THREAD_NAMESPACE: self.runner})
        ):
            ran += 1
        return ran

    def values(self, thread: WorkflowThread) -> dict[str, Any]:
        return asyncio.run(self.runner.inspect(thread)).values


def test_event_resumes_the_correlated_workflow_waiting_on_it(factory, clock):
    app = EventGraph(factory)
    correlation_id = uuid4()
    thread = app.registry.register(
        thread_id=derive_thread_id(EVENT_HANDLER_THREAD_NAMESPACE, "waiting"),
        workflow_name="waiting",
        workflow_id=app.registry.create_workflow("waiting"),
        correlation_id=correlation_id,
    )
    asyncio.run(app.runner.start(thread, {"mode": "wait"}))
    assert not app.values(thread).get("handled")  # parked at the interrupt

    outbox = _outbox(factory, clock)
    dispatcher = OutboxDispatcher(outbox, {}, default=app.router)
    envelope = _envelope(dedupe_key="reply:1", correlation_id=correlation_id)
    outbox.publish(envelope)
    assert asyncio.run(dispatcher.drain()) == 1

    queued = app.commands.queued_for(thread.workflow_id)
    assert [c.kind for c in queued] == [WorkflowCommandKind.RESUME]
    assert queued[0].thread_id == thread.thread_id
    assert queued[0].graph_input["event"]["dedupe_key"] == "reply:1"

    assert app.run_commands() == 1
    assert app.values(thread)["handled"] == ["reply:1"]


def test_event_with_nothing_waiting_starts_an_event_handler_run(factory, clock):
    app = EventGraph(factory)
    outbox = _outbox(factory, clock)
    dispatcher = OutboxDispatcher(outbox, {}, default=app.router)
    envelope = _envelope(dedupe_key="reply:2", correlation_id=uuid4())
    outbox.publish(envelope)

    assert asyncio.run(dispatcher.drain()) == 1
    assert app.run_commands() == 1

    thread = register_event_handler_thread(envelope, registry=app.registry)
    assert thread.thread_id.startswith(f"{EVENT_HANDLER_THREAD_NAMESPACE}:")
    assert app.values(thread)["handled"] == ["reply:2"]


def test_redelivered_event_is_routed_once(factory, clock):
    app = EventGraph(factory)
    envelope = _envelope(dedupe_key="reply:3")

    first = asyncio.run(app.router(envelope))
    again = asyncio.run(app.router(envelope.model_copy(update={"event_id": uuid4()})))

    assert first.kind is RouteKind.STARTED
    assert again.kind is RouteKind.ALREADY_ROUTED
    assert again.workflow_id == first.workflow_id
    assert len(app.commands.queued_for(first.workflow_id)) == 1


def test_types_that_start_no_run_are_left_unrouted(factory):
    app = EventGraph(factory)
    router = EventRouter(app.commands, app.router.waits, app.router.starter, start_types=())
    assert asyncio.run(router(_envelope())).kind is RouteKind.UNROUTED


def test_built_dispatcher_applies_transition_requests(factory, clock):
    application_id = _make_application(factory)
    dispatcher = build_outbox_dispatcher(factory)
    dispatcher.outbox.clock = clock
    dispatcher.outbox.publish(_transition_request(application_id, dedupe_key="move:1"))
    dispatcher.outbox.publish(_envelope())  # unrouted: nothing waits, starts nothing

    assert asyncio.run(dispatcher.drain()) == 2
    assert _status(factory, application_id) == ApplicationStatus.SAVED.value
    session = factory()
    try:
        assert session.query(ConsumedEventModel).count() == 1
        assert {s for (s,) in session.query(OutboxEventModel.status)} == {
            OutboxEventStatus.DISPATCHED.value
        }
        stored = session.query(OutboxEventModel).first().payload_json
        assert ENVELOPE_KEY in stored
    finally:
        session.close()
