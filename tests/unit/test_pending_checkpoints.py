"""Contract tests for durable conditional waits: the value, the rules, the store.

Three layers, tested separately because they fail separately:

- **`PendingCheckpoint`'s invariants.** A wait that expires before it triggers,
  or that names no thread, is unactionable in a way nothing downstream can
  detect -- it just quietly never fires. So it is refused where it is built.
- **`decide_checkpoint`.** The whole policy, as a pure function. Every
  interesting case here is a combination of a condition and a moment, and all
  of them are reachable without a clock, a database or a graph.
- **`PendingCheckpointStore`.** What the rules run against: idempotent
  scheduling, a `due` query that does not evaluate anything, and a close that
  exactly one of two racing callers wins.

The end-to-end behaviour -- a wait that resolves silently, one that fires and
resumes the right graph path, one that expires unfired -- is in
`tests/graph_scenarios/test_pending_checkpoints.py`.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.worker.checkpoint_monitor import NeverMetConditionEvaluator
from personalos.domain.checkpoints import (
    DEFAULT_CHECKPOINT_GRACE,
    CheckpointCondition,
    CheckpointContractError,
    CheckpointOutcome,
    ConditionKind,
    PendingCheckpoint,
    PendingCheckpointStatus,
    condition_for_kind,
    decide_checkpoint,
    follow_up_dedupe_key,
)
from personalos.domain.job_search import FollowUpKind
from personalos.persistence.models import Base
from personalos.persistence.pending_checkpoints import PendingCheckpointStore

NOW = datetime(2026, 9, 29, 12, 0, 0)
APPLICATION_ID = uuid4()
THREAD_ID = "job_search:abcdef0123456789abcdef0123456789"


def wait(**overrides) -> PendingCheckpoint:
    """A pending wait due in seven days, as the follow-up branch schedules one."""
    defaults = {
        "application_id": APPLICATION_ID,
        "kind": FollowUpKind.NO_RESPONSE,
        "due_at": NOW + timedelta(days=7),
        "reason": "no recruiter response received yet",
        "thread_id": THREAD_ID,
        "created_at": NOW,
    }
    defaults.update(overrides)
    return PendingCheckpoint.for_follow_up(**defaults)


@pytest.fixture
def store(tmp_path):
    """A store over a file-backed SQLite database, with a fixed clock."""
    engine = create_engine(f"sqlite:///{tmp_path / 'checkpoints.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return PendingCheckpointStore(factory, clock=lambda: NOW)


# --- The value ---------------------------------------------------------------


def test_a_wait_carries_its_condition_its_trigger_and_an_explicit_expiry():
    """All three, and none of them derived from a process that is still running."""
    checkpoint = wait()

    assert checkpoint.condition.kind == ConditionKind.RECRUITER_RESPONSE_RECEIVED
    assert checkpoint.condition.subject_id == APPLICATION_ID
    # Bounded in time, so a reply from before the wait existed cannot resolve it.
    assert checkpoint.condition.since == NOW
    assert checkpoint.trigger_at == NOW + timedelta(days=7)
    assert checkpoint.expires_at == NOW + timedelta(days=7) + DEFAULT_CHECKPOINT_GRACE
    assert checkpoint.status == PendingCheckpointStatus.PENDING
    # The thread is a lookup key, not a handle: it is a string, and nothing in
    # the value refers to a graph, a run or a connection.
    assert checkpoint.thread_id == THREAD_ID


def test_a_wait_that_expires_before_it_triggers_is_refused():
    """It could only ever be written off, so it is not a wait at all."""
    with pytest.raises(ValidationError, match="must expire after it triggers"):
        PendingCheckpoint(
            application_id=APPLICATION_ID,
            kind=FollowUpKind.NO_RESPONSE,
            condition=condition_for_kind(FollowUpKind.NO_RESPONSE, APPLICATION_ID),
            reason="no response",
            thread_id=THREAD_ID,
            created_at=NOW,
            trigger_at=NOW + timedelta(days=7),
            expires_at=NOW + timedelta(days=7),
            dedupe_key="follow_up:x",
        )


def test_a_wait_with_no_thread_to_resume_is_refused():
    """A checkpoint with nowhere to come back to can only ever expire."""
    with pytest.raises(ValidationError, match="must name the thread"):
        wait(thread_id="   ")


def test_every_follow_up_kind_says_what_would_make_it_unnecessary():
    """No kind may be scheduled without a resolution condition.

    The same failure mode `risk_profile_for` guards: a new kind with no entry
    would either never be cancellable or -- worse, if it defaulted -- always be.
    """
    for kind in FollowUpKind:
        condition = condition_for_kind(kind, APPLICATION_ID)
        assert isinstance(condition, CheckpointCondition)
        assert condition.subject_id == APPLICATION_ID


def test_closing_a_wait_records_the_status_and_the_reason_and_leaves_the_original():
    """Values are immutable, so a close produces a new one."""
    checkpoint = wait()

    closed = checkpoint.closed_as(
        PendingCheckpointStatus.RESOLVED, at=NOW + timedelta(days=2), reason="recruiter replied"
    )

    assert closed.status == PendingCheckpointStatus.RESOLVED
    assert closed.closed_reason == "recruiter replied"
    assert closed.closed_at == NOW + timedelta(days=2)
    assert checkpoint.status == PendingCheckpointStatus.PENDING


# --- The rules ---------------------------------------------------------------


def test_a_wait_before_its_trigger_does_nothing():
    """Not yet actionable, and nothing is asked of the world on its behalf."""
    assert (
        decide_checkpoint(checkpoint=wait(), condition_met=False, now=NOW + timedelta(days=1))
        is CheckpointOutcome.WAIT
    )


def test_a_met_condition_resolves_the_wait_even_before_its_trigger():
    """The reason for waiting went away, so there is nothing left to do."""
    assert (
        decide_checkpoint(checkpoint=wait(), condition_met=True, now=NOW + timedelta(days=1))
        is CheckpointOutcome.RESOLVE
    )


def test_a_wait_whose_condition_is_still_unmet_at_its_trigger_fires():
    """The case the whole mechanism exists for."""
    assert (
        decide_checkpoint(checkpoint=wait(), condition_met=False, now=NOW + timedelta(days=7))
        is CheckpointOutcome.FIRE
    )


def test_expiry_beats_the_trigger():
    """A monitor that was down for a week must not send a week-late follow-up.

    This is the rule that makes `expires_at` worth storing separately: by this
    moment the trigger has long passed, and a design without an explicit expiry
    would fire.
    """
    checkpoint = wait()
    late = checkpoint.expires_at + timedelta(days=1)

    assert (
        decide_checkpoint(checkpoint=checkpoint, condition_met=False, now=late)
        is CheckpointOutcome.EXPIRE
    )


def test_a_met_condition_beats_expiry():
    """Resolved and expired are both silences, and they mean opposite things.

    Recording a wait as `EXPIRED` when the recruiter had in fact replied would
    report a success as a miss, and an operator reading the table would go
    looking for a bug that is not there.
    """
    checkpoint = wait()

    assert (
        decide_checkpoint(
            checkpoint=checkpoint, condition_met=True, now=checkpoint.expires_at + timedelta(days=1)
        )
        is CheckpointOutcome.RESOLVE
    )


def test_deciding_about_an_already_closed_wait_is_refused():
    """A second decision on a fired checkpoint is a second follow-up."""
    fired = wait().closed_as(PendingCheckpointStatus.FIRED, at=NOW, reason="fired")

    with pytest.raises(CheckpointContractError, match="already fired"):
        decide_checkpoint(checkpoint=fired, condition_met=False, now=NOW + timedelta(days=8))


# --- The store ---------------------------------------------------------------


def test_scheduling_the_same_wait_twice_keeps_the_first_one(store):
    """Re-entering the branch that schedules a follow-up must not stack a second.

    And the *first* wait wins whole, including its trigger date: re-dating it on
    every replay would push the trigger forever into the future, which is the
    subtle version of never firing.
    """
    first = store.schedule(wait())
    second = store.schedule(wait(created_at=NOW + timedelta(days=1)))

    assert second.checkpoint_id == first.checkpoint_id
    assert second.trigger_at == first.trigger_at
    assert len(store.open_for_application(APPLICATION_ID)) == 1
    assert first.dedupe_key == follow_up_dedupe_key(APPLICATION_ID, FollowUpKind.NO_RESPONSE)


def test_a_stored_wait_round_trips_through_the_database(store):
    """Including its condition, which is the part that has to survive to be re-asked."""
    stored = store.schedule(wait())

    read_back = store.get(stored.checkpoint_id)

    assert read_back == stored
    assert read_back.condition.kind == ConditionKind.RECRUITER_RESPONSE_RECEIVED
    assert read_back.condition.since == NOW


def test_due_returns_nothing_before_the_trigger_and_the_wait_after_it(store):
    """The sweep's query, and it evaluates no condition on the way past.

    That matters more than it looks: a `due` that pre-filtered on the condition
    would be asking the question at the wrong moment for every checkpoint that
    is not yet actionable, which is exactly the creation-time evaluation this
    design exists to avoid.
    """
    stored = store.schedule(wait())

    assert store.due(now=NOW + timedelta(days=6)) == []
    assert [c.checkpoint_id for c in store.due(now=NOW + timedelta(days=7))] == [
        stored.checkpoint_id
    ]


def test_an_expired_wait_is_still_returned_by_due_so_something_can_close_it(store):
    """Otherwise it sits `pending` forever, which is the third acceptance criterion."""
    stored = store.schedule(wait())

    due = store.due(now=stored.expires_at + timedelta(days=30))

    assert [c.checkpoint_id for c in due] == [stored.checkpoint_id]


def test_only_one_of_two_callers_closing_the_same_wait_wins(store):
    """The exclusion primitive: two monitors, one guarded UPDATE, one winner.

    Without it both would go on to start the graph path, and the candidate
    would get nudged twice about the same silence.
    """
    stored = store.schedule(wait())

    first = store.close(stored, PendingCheckpointStatus.FIRED, reason="trigger reached")
    second = store.close(stored, PendingCheckpointStatus.FIRED, reason="trigger reached")

    assert first is True
    assert second is False
    assert store.get(stored.checkpoint_id).status == PendingCheckpointStatus.FIRED


def test_closing_a_wait_as_pending_is_refused(store):
    """"Closed, still waiting" is not a state, and would silently reopen a decision."""
    stored = store.schedule(wait())

    with pytest.raises(ValueError, match="terminal status"):
        store.close(stored, PendingCheckpointStatus.PENDING, reason="nope")


def test_an_event_resolves_the_waits_it_makes_unnecessary(store):
    """The event-driven early close, for when the resolution is noticed as it happens."""
    stored = store.schedule(wait())
    other = store.schedule(
        wait(kind=FollowUpKind.INTERVIEW_PREP, reason="prepare for the interview")
    )

    closed = store.resolve_matching(
        condition_kind=ConditionKind.RECRUITER_RESPONSE_RECEIVED,
        subject_id=APPLICATION_ID,
        reason="recruiter replied",
        occurred_at=NOW + timedelta(days=2),
    )

    assert closed == [stored.checkpoint_id]
    assert store.get(stored.checkpoint_id).status == PendingCheckpointStatus.RESOLVED
    # The interview-prep wait asks a different question and is untouched.
    assert store.get(other.checkpoint_id).status == PendingCheckpointStatus.PENDING


def test_an_event_older_than_the_wait_does_not_resolve_it(store):
    """`condition.since` is what stops last month's reply cancelling this week's wait."""
    stored = store.schedule(wait())

    closed = store.resolve_matching(
        condition_kind=ConditionKind.RECRUITER_RESPONSE_RECEIVED,
        subject_id=APPLICATION_ID,
        reason="stale reply",
        occurred_at=NOW - timedelta(days=30),
    )

    assert closed == []
    assert store.get(stored.checkpoint_id).status == PendingCheckpointStatus.PENDING


def test_closed_waits_are_kept_so_a_silence_can_be_explained(store):
    """"Why was no follow-up sent?" is answerable only if the row is still there."""
    stored = store.schedule(wait())
    store.close(stored, PendingCheckpointStatus.RESOLVED, reason="recruiter replied on day 2")

    assert store.open_for_application(APPLICATION_ID) == []
    history = store.history_for_application(APPLICATION_ID)
    assert [c.status for c in history] == [PendingCheckpointStatus.RESOLVED]
    assert history[0].closed_reason == "recruiter replied on day 2"


# --- The default evaluator ----------------------------------------------------


async def test_the_default_evaluator_never_reports_a_condition_met():
    """A deployment with nothing to ask follows up rather than staying silent.

    The failure modes are not symmetric, which is why this default is chosen on
    purpose rather than inferred: an evaluator that answered `True` when it
    could not tell would silently cancel every follow-up in the system, while
    one that answers `False` produces at most a redundant nudge -- and the nudge
    still goes past a human at the approval interrupt before it is sent.
    """
    evaluator = NeverMetConditionEvaluator()

    assert (await evaluator.is_met(wait().condition)) is False
