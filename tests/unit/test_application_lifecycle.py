"""Tests for the durable application lifecycle state machine.

Covers the acceptance criteria: every valid transition succeeds and every
invalid one (e.g. DISCOVERED -> OFFER) is rejected; an application's state
survives a process restart and is correct with no chat history anywhere; and
an application with no activity past the stall window is moved to STALLED by
the scheduled check exactly once.
"""

import threading
from collections import deque
from datetime import datetime, timedelta
from itertools import count
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.worker.stall_monitor import StalledApplicationMonitor
from personalos.domain.models import (
    ALLOWED_APPLICATION_TRANSITIONS,
    APPLICATION_STATUS_CHANGED_EVENT,
    HOLDING_APPLICATION_STATUSES,
    STALL_MONITOR_ACTOR,
    STALLABLE_APPLICATION_STATUSES,
    TERMINAL_APPLICATION_STATUSES,
    ApplicationStatus,
    ApplicationTransitionRecommendation,
    InvalidApplicationTransition,
    allowed_application_transitions,
    validate_application_status_transition,
)
from personalos.persistence.application_lifecycle import ApplicationLifecycleStore
from personalos.persistence.models import (
    ApplicationModel,
    ApplicationStatusViewModel,
    Base,
    EventLogModel,
    UserModel,
)
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository

S = ApplicationStatus
T0 = datetime(2026, 10, 1, 9, 0, 0)
WINDOW = timedelta(days=14)


class TickingClock:
    """A clock that moves one second per reading, so events have a total order."""

    def __init__(self, start: datetime = T0):
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def _memory_factory():
    """One shared in-memory SQLite database, for tests that make many transitions."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine


def _file_factory(db_path):
    """A session factory over a file-backed SQLite database; see `_open_shared` elsewhere."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine


_serial = count()


def _new_application(factory, *, created_at: datetime = T0):
    """A fresh DISCOVERED application, last active at `created_at`."""
    n = next(_serial)
    session = factory()
    try:
        user = UserModel(email=f"candidate-{n}-{uuid4()}@example.com")
        session.add(user)
        session.commit()
        posting = JobPostingRepository(session).create(
            source="linkedin",
            title="Staff Engineer",
            company="Acme",
            description_hash="a" * 64,
            dedupe_key=f"acme:staff-engineer:{uuid4()}",
        )
        application = ApplicationRepository(session).create(
            job_posting_id=posting.id, user_id=user.id
        )
        application.last_activity_at = created_at
        session.commit()
        return application.id
    finally:
        session.close()


def _status_events(factory, application_id):
    session = factory()
    try:
        rows = (
            session.query(EventLogModel)
            .filter(
                EventLogModel.aggregate_id == application_id,
                EventLogModel.event_type == APPLICATION_STATUS_CHANGED_EVENT,
            )
            .order_by(EventLogModel.occurred_at)
            .all()
        )
        return [row.payload_json for row in rows]
    finally:
        session.close()


def _reachable_states():
    """Every (status, resume_status) an application can be in, with a path to each.

    Derived by walking the state machine from DISCOVERED rather than listed,
    so a state added to the lifecycle is covered without touching this file.
    """
    start = (S.DISCOVERED, None)
    paths = {start: ()}
    queue = deque([start])
    while queue:
        status, held_from = queue.popleft()
        for target in allowed_application_transitions(status, resume_status=held_from):
            if target in HOLDING_APPLICATION_STATUSES:
                next_held = held_from if status in HOLDING_APPLICATION_STATUSES else status
            else:
                next_held = None
            state = (target, next_held)
            if state not in paths:
                paths[state] = (*paths[(status, held_from)], target)
                queue.append(state)
    return paths


REACHABLE = _reachable_states()
ALL_MOVES = [(state, target) for state in REACHABLE for target in S]


def _move_id(move):
    (status, held_from), target = move
    origin = f"{status.value}[{held_from.value}]" if held_from else status.value
    return f"{origin}->{target.value}"


# ----------------------------------------------------------------------
# The state machine itself
# ----------------------------------------------------------------------


def test_every_status_is_reachable_and_accounted_for():
    """No state is an orphan, and only the holding states lack a fixed successor row."""
    assert {status for status, _ in REACHABLE} == set(S)
    assert set(ALLOWED_APPLICATION_TRANSITIONS) == set(S) - HOLDING_APPLICATION_STATUSES
    assert TERMINAL_APPLICATION_STATUSES == {
        S.ACCEPTED,
        S.DECLINED,
        S.REJECTED,
        S.WITHDRAWN,
        S.SKIPPED,
    }


def test_documented_happy_path_is_valid_end_to_end():
    """DISCOVERED -> ... -> APPLIED -> RESPONSE -> INTERVIEWING -> OFFER -> ACCEPTED | DECLINED."""
    path = [
        S.DISCOVERED,
        S.SAVED,
        S.PREPARING,
        S.READY_TO_APPLY,
        S.APPLIED,
        S.RESPONSE,
        S.INTERVIEWING,
        S.OFFER,
    ]
    for current, new in zip(path, path[1:], strict=False):
        assert validate_application_status_transition(current, new) is new
    assert validate_application_status_transition(S.OFFER, S.ACCEPTED) is S.ACCEPTED
    assert validate_application_status_transition(S.OFFER, S.DECLINED) is S.DECLINED
    # RESPONSE is optional.
    assert validate_application_status_transition(S.APPLIED, S.INTERVIEWING) is S.INTERVIEWING


def test_a_held_application_cannot_skip_ahead():
    """Stalling is not a shortcut: STALLED leads only where the held-from state led."""
    assert allowed_application_transitions(S.STALLED, resume_status=S.SAVED) == {
        S.SAVED,
        S.PREPARING,
        S.SKIPPED,
    }
    with pytest.raises(InvalidApplicationTransition):
        validate_application_status_transition(S.STALLED, S.OFFER, resume_status=S.SAVED)
    # With nothing recorded to resume to, there is nowhere to go.
    assert allowed_application_transitions(S.STALLED) == frozenset()


@pytest.mark.parametrize("move", ALL_MOVES, ids=_move_id)
def test_every_transition_is_accepted_or_rejected_by_the_store(move):
    """Exhaustive: from every reachable state, each of the fifteen targets.

    A valid move succeeds, emits exactly one event and moves the projection to
    it. An invalid one raises, leaves the status where it was and emits
    nothing.
    """
    (status, held_from), target = move
    factory, engine = _memory_factory()
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        application_id = _new_application(factory)
        for step in REACHABLE[(status, held_from)]:
            store.transition(application_id, step)
        events_before = _status_events(factory, application_id)

        if target in allowed_application_transitions(status, resume_status=held_from):
            state = store.transition(application_id, target, reason="because")
            assert state.status is target
            events = _status_events(factory, application_id)
            assert len(events) == len(events_before) + 1
            assert events[-1]["from"] == status.value
            assert events[-1]["to"] == target.value
            assert events[-1]["reason"] == "because"

            session = factory()
            try:
                view = session.get(ApplicationStatusViewModel, application_id)
                assert view.status == target.value
                assert view.last_event_id == state.last_event_id is not None
            finally:
                session.close()
        else:
            with pytest.raises(InvalidApplicationTransition):
                store.transition(application_id, target)
            state = store.get(application_id)
            assert (state.status, state.resume_status) == (status, held_from)
            assert _status_events(factory, application_id) == events_before
    finally:
        engine.dispose()


def test_discovered_to_offer_is_rejected_by_the_store():
    """The example from the issue, spelled out."""
    factory, engine = _memory_factory()
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        application_id = _new_application(factory)
        with pytest.raises(InvalidApplicationTransition):
            store.transition(application_id, S.OFFER)
        assert store.get(application_id).status is S.DISCOVERED
        assert _status_events(factory, application_id) == []
    finally:
        engine.dispose()


# ----------------------------------------------------------------------
# An LLM may recommend, never set
# ----------------------------------------------------------------------


def test_a_valid_recommendation_is_applied_and_attributed():
    factory, engine = _memory_factory()
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        application_id = _new_application(factory)
        state = store.recommend(
            ApplicationTransitionRecommendation(
                application_id=application_id, to="Saved", reason="strong match"
            )
        )
        assert state.status is S.SAVED
        (event,) = _status_events(factory, application_id)
        assert event["actor"] == "llm"
        assert event["reason"] == "strong match"
    finally:
        engine.dispose()


@pytest.mark.parametrize("to", ["offer", "hired", "", "stalled", "OFFER; DROP TABLE"])
def test_a_recommendation_cannot_set_an_arbitrary_status(to):
    """Off-path, unknown and system-only statuses are all refused, and nothing is written."""
    factory, engine = _memory_factory()
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        application_id = _new_application(factory)
        store.transition(application_id, S.SAVED)

        with pytest.raises(InvalidApplicationTransition):
            store.recommend(
                ApplicationTransitionRecommendation(application_id=application_id, to=to)
            )
        assert store.get(application_id).status is S.SAVED
        assert len(_status_events(factory, application_id)) == 1
    finally:
        engine.dispose()


# ----------------------------------------------------------------------
# Durability: the database is the whole story
# ----------------------------------------------------------------------


def test_state_survives_a_restart_with_no_chat_history(tmp_path):
    """A second process, given only the database file, reads the correct state.

    Nothing is carried across but the path: the first engine is disposed, and
    the second store has no graph, no checkpointer, no thread and no
    transcript to consult.
    """
    db_path = tmp_path / "lifecycle.db"

    factory, engine = _file_factory(db_path)
    store = ApplicationLifecycleStore(factory, clock=TickingClock())
    application_id = _new_application(factory)
    path = (S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED, S.RESPONSE, S.FOLLOW_UP_PENDING)
    for step in path:
        last = store.transition(application_id, step)
    del store
    engine.dispose()

    restarted_factory, restarted_engine = _file_factory(db_path)
    try:
        restarted = ApplicationLifecycleStore(restarted_factory)
        state = restarted.get(application_id)
        assert state.status is S.FOLLOW_UP_PENDING
        assert state.resume_status is S.RESPONSE
        assert state.last_activity_at == last.last_activity_at
        assert state.last_event_id == last.last_event_id

        # The projection agrees with the immutable log it was built from.
        history = [event["payload_json"]["to"] for event in restarted.history(application_id)]
        assert history == [step.value for step in path]

        # And the lifecycle carries on from where it was, rules intact.
        with pytest.raises(InvalidApplicationTransition):
            restarted.transition(application_id, S.OFFER)
        assert restarted.transition(application_id, S.INTERVIEWING).status is S.INTERVIEWING
    finally:
        restarted_engine.dispose()


def test_a_failed_transition_leaves_no_event_behind(tmp_path):
    """Status, event and projection are one transaction: roll one back, lose all three."""
    factory, engine = _file_factory(tmp_path / "lifecycle.db")
    try:
        application_id = _new_application(factory)
        session = factory()
        try:
            ApplicationRepository(session).update_status(application_id, S.SAVED, commit=False)
            session.rollback()
        finally:
            session.close()

        assert ApplicationLifecycleStore(factory).get(application_id).status is S.DISCOVERED
        assert _status_events(factory, application_id) == []
    finally:
        engine.dispose()


# ----------------------------------------------------------------------
# STALLED: a scheduled check, exactly once
# ----------------------------------------------------------------------


def _applied_application(factory, store):
    application_id = _new_application(factory)
    for step in (S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED):
        state = store.transition(application_id, step)
    return application_id, state.last_activity_at


def _stall_events(factory, application_id):
    return [e for e in _status_events(factory, application_id) if e["to"] == S.STALLED.value]


def test_quiet_application_is_stalled_exactly_once(tmp_path):
    """Past the window it stalls; every later sweep leaves it, and the log, alone."""
    factory, engine = _file_factory(tmp_path / "lifecycle.db")
    try:
        clock = TickingClock()
        store = ApplicationLifecycleStore(factory, clock=clock)
        monitor = StalledApplicationMonitor(store=store, window=WINDOW, clock=clock)
        application_id, last_activity = _applied_application(factory, store)

        # Inside the window: nothing.
        assert monitor.sweep(now=last_activity + WINDOW - timedelta(seconds=1)).stalled == ()
        assert store.get(application_id).status is S.APPLIED

        first = monitor.sweep(now=last_activity + WINDOW)
        assert first.stalled == (application_id,)
        state = store.get(application_id)
        assert state.status is S.STALLED
        assert state.resume_status is S.APPLIED
        # Stalling is not activity.
        assert state.last_activity_at == last_activity

        for days in (1, 30, 365):
            again = monitor.sweep(now=last_activity + WINDOW + timedelta(days=days))
            assert again.stalled == () and again.skipped == ()

        (event,) = _stall_events(factory, application_id)
        assert event["from"] == S.APPLIED.value
        assert event["actor"] == STALL_MONITOR_ACTOR
    finally:
        engine.dispose()


def test_two_monitors_sweeping_at_once_stall_it_once(tmp_path):
    """Both select the same quiet application; the guarded update lets one through."""
    factory, engine = _file_factory(tmp_path / "lifecycle.db")
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        application_id, last_activity = _applied_application(factory, store)
        now = last_activity + WINDOW + timedelta(hours=1)

        barrier = threading.Barrier(2)
        reports = []

        def sweep():
            monitor = StalledApplicationMonitor(
                store=ApplicationLifecycleStore(factory), window=WINDOW
            )
            barrier.wait()
            reports.append(monitor.sweep(now=now))

        threads = [threading.Thread(target=sweep) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(len(report.stalled) for report in reports) == 1
        assert len(_stall_events(factory, application_id)) == 1
        assert store.get(application_id).status is S.STALLED
    finally:
        engine.dispose()


def test_a_candidate_that_became_active_is_not_stalled(tmp_path):
    """Selected as quiet, active by the time it is written: the stall loses."""
    factory, engine = _file_factory(tmp_path / "lifecycle.db")
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        application_id, last_activity = _applied_application(factory, store)
        now = last_activity + WINDOW + timedelta(hours=1)
        cutoff = now - WINDOW
        assert store.quiet_since(cutoff) == [application_id]

        store.record_activity(application_id, now=now)

        assert store.mark_stalled(application_id, quiet_since=cutoff, now=now) is False
        assert store.get(application_id).status is S.APPLIED
        assert _stall_events(factory, application_id) == []
    finally:
        engine.dispose()


def test_activity_restarts_the_window_and_resumes_a_stalled_application(tmp_path):
    factory, engine = _file_factory(tmp_path / "lifecycle.db")
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        monitor = StalledApplicationMonitor(store=store, window=WINDOW)
        application_id, last_activity = _applied_application(factory, store)

        # Activity on day 10 pushes the stall from day 14 to day 24.
        day_10 = last_activity + timedelta(days=10)
        store.record_activity(application_id, now=day_10)
        assert monitor.sweep(now=last_activity + timedelta(days=15)).stalled == ()
        assert monitor.sweep(now=day_10 + WINDOW).stalled == (application_id,)

        # Activity on a stalled application puts it back where it was...
        day_40 = last_activity + timedelta(days=40)
        resumed = store.record_activity(application_id, reason="recruiter replied", now=day_40)
        assert resumed.status is S.APPLIED
        assert resumed.resume_status is None
        assert resumed.last_activity_at == day_40

        # ...from where a second, separate silence can stall it again.
        assert monitor.sweep(now=day_40 + WINDOW).stalled == (application_id,)
        assert len(_stall_events(factory, application_id)) == 2
    finally:
        engine.dispose()


def test_only_applications_still_in_play_can_stall(tmp_path):
    """Untouched leads and finished applications are never stalled, however old."""
    factory, engine = _file_factory(tmp_path / "lifecycle.db")
    try:
        store = ApplicationLifecycleStore(factory, clock=TickingClock())
        monitor = StalledApplicationMonitor(store=store, window=WINDOW)

        by_status = {}
        for (status, _), path in REACHABLE.items():
            if status is S.STALLED:
                continue
            application_id = _new_application(factory)
            for step in path:
                store.transition(application_id, step)
            by_status[application_id] = status

        report = monitor.sweep(now=T0 + timedelta(days=365), limit=len(by_status))

        assert report.skipped == ()
        assert {by_status[a] for a in report.stalled} == STALLABLE_APPLICATION_STATUSES
        assert set(report.stalled) == {
            a for a, status in by_status.items() if status in STALLABLE_APPLICATION_STATUSES
        }
        session = factory()
        try:
            untouched = (
                session.query(ApplicationModel)
                .filter(ApplicationModel.id.notin_(report.stalled))
                .all()
            )
            assert {S(row.status) for row in untouched} == (
                TERMINAL_APPLICATION_STATUSES | {S.DISCOVERED}
            )
        finally:
            session.close()
    finally:
        engine.dispose()


def test_the_stall_window_comes_from_configuration(monkeypatch):
    from personalos.config import settings

    monkeypatch.setattr(settings, "application_stall_window_days", 3)
    factory, engine = _memory_factory()
    try:
        monitor = StalledApplicationMonitor(store=ApplicationLifecycleStore(factory))
        assert monitor.window == timedelta(days=3)
        with pytest.raises(ValueError):
            StalledApplicationMonitor(
                store=ApplicationLifecycleStore(factory), window=timedelta(0)
            )
    finally:
        engine.dispose()
