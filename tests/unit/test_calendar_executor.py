"""Calendar writes do not duplicate an event on a retry.

The acceptance criterion is the first test: a calendar create succeeds, its
response is lost, and the retry finds the event already there instead of
inserting a second one. It runs through the real `ToolExecutor` and its ledger
on a real database, because that is where the retry decision is made -- the
first attempt is on record as failed, and only the reconciler's answer permits
doing anything about it.

The rest pin the two halves on their own: `CalendarActionExecutor` looks before
it writes, and `CalendarReconciler` answers from the calendar and nothing else.
"""

from datetime import datetime, timedelta

import pytest

from personalos.bootstrap import build_calendar_action_executor
from personalos.domain.interview_scheduling import (
    PAYLOAD_ENDS_AT,
    PAYLOAD_EVENT_ID,
    PAYLOAD_PROPERTIES,
    PAYLOAD_STARTS_AT,
    PAYLOAD_TITLE,
    PROP_APPLICATION_ID,
    PROP_IDEMPOTENCY_KEY,
    PROP_KEY,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ApprovalDecision,
    ApprovalVerdict,
    JobSearchContractError,
)
from personalos.domain.models import ToolExecutionStatus
from personalos.executor.calendar import CalendarActionExecutor, CalendarReconciler
from personalos.executor.tool_executor import ReconcileOutcome
from personalos.persistence.models import ToolExecutionModel
from personalos.policy import ApprovalRequired
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.durable_workflow import session_factory

STARTS = datetime(2026, 10, 1, 14, 0)
ENDS = STARTS + timedelta(hours=1)
KEY = f"interview:{fakes.APPLICATION_ID}"


def create_intent(**payload) -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.CREATE_CALENDAR_EVENT,
        target="calendar: new event 'Interview: Acme'",
        summary="Add 'Interview: Acme' to the calendar",
        payload={
            PAYLOAD_TITLE: "Interview: Acme",
            PAYLOAD_STARTS_AT: STARTS.isoformat(),
            PAYLOAD_ENDS_AT: ENDS.isoformat(),
            PAYLOAD_PROPERTIES: {PROP_APPLICATION_ID: str(fakes.APPLICATION_ID), PROP_KEY: KEY},
            **payload,
        },
        idempotency_key=f"cal-create:{KEY}:20261001T1400",
    )


def update_intent(event_id: str, starts: datetime) -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.UPDATE_CALENDAR_EVENT,
        target=f"calendar: event {event_id}",
        summary="Move 'Interview: Acme'",
        payload={
            PAYLOAD_EVENT_ID: event_id,
            PAYLOAD_TITLE: "Interview: Acme",
            PAYLOAD_STARTS_AT: starts.isoformat(),
            PAYLOAD_ENDS_AT: (starts + timedelta(hours=1)).isoformat(),
        },
        idempotency_key=f"cal-update:{event_id}:1:{starts:%Y%m%dT%H%M}",
    )


def approved(intent: ActionIntent) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
    )


def execution_status(factory, key: str) -> str:
    session = factory()
    try:
        return (
            session.query(ToolExecutionModel)
            .filter(ToolExecutionModel.idempotency_key == key)
            .one()
            .status
        )
    finally:
        session.close()


# --- Acceptance: the create landed, the response did not ----------------------


async def test_a_create_whose_response_was_lost_is_reconciled_not_duplicated(tmp_path):
    """The insert succeeded and the caller never heard. The retry must not insert again.

    The first attempt raises after the calendar has stored the event, which is
    recorded as a failure with no receipt. The retry -- a fresh executor on a
    fresh session factory, as a restarted worker would be -- finds that record,
    asks the calendar for an event stamped with this action's idempotency key,
    and returns the one that is there.
    """
    calendar = fakes.FakeCalendar()
    calendar.lose_next_create_response = True
    intent = create_intent()

    first = build_calendar_action_executor(calendar, session_factory(tmp_path / "cal.db"))
    with pytest.raises(fakes.LostResponse):
        await first.execute(intent, approved(intent))

    assert len(calendar.events) == 1, "the insert did land"
    (event_id,) = calendar.events

    factory = session_factory(tmp_path / "cal.db")
    receipt = await build_calendar_action_executor(calendar, factory).execute(
        intent, approved(intent)
    )

    assert receipt.ok is True
    assert receipt.external_reference == event_id
    # One event, one insert: the retry created nothing.
    assert list(calendar.events) == [event_id]
    assert calendar.created == [event_id]
    assert execution_status(factory, intent.idempotency_key) == (
        ToolExecutionStatus.COMPLETED.value
    )

    # And a third attempt replays the stored receipt without asking anyone.
    again = await build_calendar_action_executor(calendar, factory).execute(
        intent, approved(intent)
    )
    assert again.external_reference == event_id
    assert calendar.created == [event_id]


async def test_a_create_that_never_reached_the_calendar_is_executed_on_retry(tmp_path):
    """The other half: reconciliation finds nothing, so the retry is the first insert."""

    class Unreachable(fakes.FakeCalendar):
        down = True

        async def create_event(self, **fields):
            if self.down:
                raise ConnectionError("calendar unreachable")
            return await super().create_event(**fields)

    calendar = Unreachable()
    intent = create_intent()
    factory = session_factory(tmp_path / "cal.db")

    with pytest.raises(ConnectionError):
        await build_calendar_action_executor(calendar, factory).execute(intent, approved(intent))
    assert calendar.events == {}

    calendar.down = False
    receipt = await build_calendar_action_executor(calendar, factory).execute(
        intent, approved(intent)
    )

    assert receipt.ok is True
    assert len(calendar.events) == 1


async def test_a_retry_is_not_attempted_when_the_calendar_cannot_be_asked(tmp_path):
    """No answer from the provider is not permission to write again."""
    calendar = fakes.FakeCalendar()
    calendar.lose_next_create_response = True
    intent = create_intent()
    factory = session_factory(tmp_path / "cal.db")
    with pytest.raises(fakes.LostResponse):
        await build_calendar_action_executor(calendar, factory).execute(intent, approved(intent))

    async def unavailable(name, value):
        raise ConnectionError("calendar unreachable")

    calendar.find_by_property = unavailable
    receipt = await build_calendar_action_executor(calendar, factory).execute(
        intent, approved(intent)
    )

    assert receipt.ok is False
    assert len(calendar.created) == 1


async def test_a_calendar_write_is_a_write_external_action_and_needs_approval(tmp_path):
    calendar = fakes.FakeCalendar()
    intent = create_intent()
    pending = approved(intent).model_copy(update={"verdict": ApprovalVerdict.PENDING})

    with pytest.raises(ApprovalRequired):
        await build_calendar_action_executor(
            calendar, session_factory(tmp_path / "cal.db")
        ).execute(intent, pending)

    assert calendar.events == {}


# --- The executor on its own ---------------------------------------------------


async def test_a_created_event_is_stamped_with_its_idempotency_key_and_links():
    calendar = fakes.FakeCalendar()
    intent = create_intent()

    receipt = await CalendarActionExecutor(calendar).execute(intent, approved(intent))

    event = calendar.events[receipt.external_reference]
    assert (event.starts_at, event.ends_at) == (STARTS, ENDS)
    assert event.properties == {
        PROP_APPLICATION_ID: str(fakes.APPLICATION_ID),
        PROP_KEY: KEY,
        PROP_IDEMPOTENCY_KEY: intent.idempotency_key,
    }


async def test_the_executor_itself_looks_before_it_creates():
    """Without a ledger in front of it at all, a repeated create is still one event."""
    calendar = fakes.FakeCalendar()
    calendar.lose_next_create_response = True
    executor = CalendarActionExecutor(calendar)
    intent = create_intent()

    with pytest.raises(fakes.LostResponse):
        await executor.execute(intent, approved(intent))
    receipt = await executor.execute(intent, approved(intent))

    assert len(calendar.events) == 1
    assert receipt.external_reference in calendar.events


async def test_an_update_already_applied_is_not_applied_again():
    calendar = fakes.FakeCalendar()
    created = create_intent()
    event_id = (
        await CalendarActionExecutor(calendar).execute(created, approved(created))
    ).external_reference
    move = update_intent(event_id, STARTS + timedelta(days=1))
    executor = CalendarActionExecutor(calendar)

    first = await executor.execute(move, approved(move))
    second = await executor.execute(move, approved(move))

    assert first.external_reference == second.external_reference == event_id
    assert calendar.updated == [event_id]
    assert calendar.events[event_id].starts_at == STARTS + timedelta(days=1)
    # The move kept the event's identity: it is still findable as ours.
    assert calendar.events[event_id].properties[PROP_KEY] == KEY


async def test_an_update_to_an_event_that_is_gone_fails_without_creating_one():
    calendar = fakes.FakeCalendar()
    move = update_intent("evt-missing", STARTS)

    receipt = await CalendarActionExecutor(calendar).execute(move, approved(move))

    assert receipt.ok is False
    assert calendar.events == {}


async def test_other_action_kinds_go_to_the_fallback_or_are_refused():
    fallback = fakes.FakeActionExecutor()
    intent = fakes.submit_intent()

    await CalendarActionExecutor(fakes.FakeCalendar(), fallback=fallback).execute(
        intent, approved(intent)
    )
    assert [i.kind for i, _ in fallback.executed] == [ActionKind.SUBMIT_APPLICATION]

    with pytest.raises(JobSearchContractError, match="fallback"):
        await CalendarActionExecutor(fakes.FakeCalendar()).execute(intent, approved(intent))


# --- The reconciler on its own -------------------------------------------------


async def test_the_reconciler_answers_from_the_calendar():
    calendar = fakes.FakeCalendar()
    reconciler = CalendarReconciler(calendar)
    create = create_intent()

    assert (await reconciler.reconcile(create)).outcome is ReconcileOutcome.NOT_APPLIED

    event_id = (
        await CalendarActionExecutor(calendar).execute(create, approved(create))
    ).external_reference
    found = await reconciler.reconcile(create)
    assert found.outcome is ReconcileOutcome.APPLIED
    assert found.receipt.external_reference == event_id

    move = update_intent(event_id, STARTS + timedelta(days=1))
    assert (await reconciler.reconcile(move)).outcome is ReconcileOutcome.NOT_APPLIED
    await CalendarActionExecutor(calendar).execute(move, approved(move))
    assert (await reconciler.reconcile(move)).outcome is ReconcileOutcome.APPLIED

    # Not a calendar action: it has nothing to say, and says so.
    other = fakes.submit_intent()
    assert (await reconciler.reconcile(other)).outcome is ReconcileOutcome.INDETERMINATE
