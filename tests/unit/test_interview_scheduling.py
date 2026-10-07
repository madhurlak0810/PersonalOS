"""`plan_interview_schedule` and its helpers, as pure values.

Every case is a calendar, an interview time and a `now`; nothing here touches
a graph or a store. `fakes.NOW` is Monday 28 September 2026, 12:00.
"""

from datetime import datetime, timedelta

import pytest

from personalos.domain.interview_scheduling import (
    PROP_APPLICATION_ID,
    PROP_INTERVIEW_KEY,
    PROP_KEY,
    PROP_ROLE,
    CalendarEvent,
    CalendarEventRole,
    ConflictKind,
    InterviewRequest,
    ScheduleOp,
    WorkingHours,
    calendar_create_key,
    calendar_update_key,
    event_properties,
    interview_event_key,
    interview_reminder_dedupe_key,
    interview_request_from_event,
    interview_time_from,
    plan_interview_schedule,
    prep_block_key,
)
from personalos.domain.recruiter_events import Commitment, CommitmentActor
from tests.fixtures import job_search_fakes as fakes

APP = fakes.APPLICATION_ID
NOW = fakes.NOW
#: Thursday 14:00, three days out.
INTERVIEW = datetime(2026, 10, 1, 14, 0)
INTERVIEW_KEY = interview_event_key(APP)
DEEP = prep_block_key(APP, "deep_prep")
WARM = prep_block_key(APP, "warm_up")


def request(starts_at: datetime = INTERVIEW, **overrides) -> InterviewRequest:
    return InterviewRequest(application_id=APP, starts_at=starts_at, **overrides)


def booked(event_id: str, start: datetime, end: datetime, **overrides) -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id, title=event_id, starts_at=start, ends_at=end, **overrides
    )


def ours(event_id: str, key: str, start: datetime, end: datetime) -> CalendarEvent:
    role = CalendarEventRole.INTERVIEW if key == INTERVIEW_KEY else CalendarEventRole.PREP
    return CalendarEvent(
        event_id=event_id,
        title="Interview" if role is CalendarEventRole.INTERVIEW else "Interview prep",
        starts_at=start,
        ends_at=end,
        properties={PROP_APPLICATION_ID: str(APP), PROP_KEY: key, PROP_ROLE: role.value},
        etag="7",
    )


def scheduled(plan) -> list[CalendarEvent]:
    """The calendar as it stands once every change in `plan` has been applied."""
    return [
        ours(change.event_id or f"evt-{n}", change.logical_key, change.starts_at, change.ends_at)
        for n, change in enumerate(plan.changes)
    ]


def by_key(plan) -> dict:
    return {change.logical_key: change for change in plan.changes}


# --- A first invite ------------------------------------------------------------


def test_a_first_invite_plans_the_interview_and_two_prep_blocks():
    plan = plan_interview_schedule(request(), [], now=NOW)

    changes = by_key(plan)
    assert {c.op for c in plan.changes} == {ScheduleOp.CREATE}
    assert (changes[INTERVIEW_KEY].starts_at, changes[INTERVIEW_KEY].ends_at) == (
        INTERVIEW,
        INTERVIEW + timedelta(hours=1),
    )
    # The long block: the evening before, inside working hours.
    assert (changes[DEEP].starts_at, changes[DEEP].ends_at) == (
        datetime(2026, 9, 30, 18, 30),
        datetime(2026, 9, 30, 20, 0),
    )
    # The short one: ending a quarter of an hour before the interview.
    assert (changes[WARM].starts_at, changes[WARM].ends_at) == (
        datetime(2026, 10, 1, 13, 15),
        datetime(2026, 10, 1, 13, 45),
    )
    assert plan.conflicts == ()
    assert plan.writes == plan.changes


def test_prep_blocks_are_placed_around_what_is_already_booked():
    calendar = [
        booked("dinner", datetime(2026, 9, 30, 18, 0), datetime(2026, 9, 30, 20, 0)),
        booked("standup", datetime(2026, 10, 1, 13, 0), datetime(2026, 10, 1, 13, 30)),
    ]

    changes = by_key(plan_interview_schedule(request(), calendar, now=NOW))

    # Each lands in the latest gap before the thing in its way.
    assert changes[DEEP].ends_at == datetime(2026, 9, 30, 18, 0)
    assert changes[WARM].ends_at == datetime(2026, 10, 1, 13, 0)
    for change in (changes[DEEP], changes[WARM]):
        assert not any(
            event.starts_at < change.ends_at and event.ends_at > change.starts_at
            for event in calendar
        )


def test_an_event_marked_free_blocks_nothing():
    calendar = [
        booked("fyi", datetime(2026, 9, 30, 18, 0), datetime(2026, 9, 30, 20, 0), busy=False)
    ]

    changes = by_key(plan_interview_schedule(request(), calendar, now=NOW))

    assert changes[DEEP].ends_at == datetime(2026, 9, 30, 20, 0)


def test_working_hours_follow_the_candidates_offset():
    # 08:00-20:00 at UTC-7 is 15:00-03:00 UTC.
    hours = WorkingHours(utc_offset=timedelta(hours=-7))

    changes = by_key(plan_interview_schedule(request(), [], now=NOW, working_hours=hours))

    assert changes[DEEP].ends_at == datetime(2026, 10, 1, 2, 0)


def test_something_booked_over_the_interview_is_reported_not_hidden():
    clash = booked("offsite", INTERVIEW - timedelta(minutes=30), INTERVIEW + timedelta(hours=2))

    plan = plan_interview_schedule(request(), [clash], now=NOW)

    # The interview is still proposed: the recruiter set the time.
    assert by_key(plan)[INTERVIEW_KEY].op is ScheduleOp.CREATE
    (conflict,) = [c for c in plan.conflicts if c.kind is ConflictKind.INTERVIEW_OVERLAP]
    assert conflict.event_id == "offsite"


def test_a_prep_block_with_nowhere_to_go_is_a_conflict():
    wall = booked("conference", NOW, INTERVIEW)

    plan = plan_interview_schedule(request(), [wall], now=NOW)

    assert set(by_key(plan)) == {INTERVIEW_KEY}
    assert {c.logical_key for c in plan.conflicts if c.kind is ConflictKind.PREP_UNPLACEABLE} == {
        DEEP,
        WARM,
    }


def test_an_interview_already_past_schedules_nothing():
    plan = plan_interview_schedule(request(NOW - timedelta(hours=1)), [], now=NOW)

    assert plan.changes == ()
    assert [c.kind for c in plan.conflicts] == [ConflictKind.INTERVIEW_IN_PAST]
    assert plan.reminder_at(NOW) is None


# --- Reconciling against what is already there ---------------------------------


def test_planning_again_against_its_own_result_changes_nothing():
    first = plan_interview_schedule(request(), [], now=NOW)

    again = plan_interview_schedule(request(), scheduled(first), now=NOW)

    assert {c.op for c in again.changes} == {ScheduleOp.KEEP}
    assert again.writes == ()
    assert all(c.event_id for c in again.changes)


def test_a_changed_interview_time_moves_the_interview_and_its_prep_blocks():
    calendar = scheduled(plan_interview_schedule(request(), [], now=NOW))
    moved_to = datetime(2026, 10, 5, 10, 0)

    plan = plan_interview_schedule(request(moved_to), calendar, now=NOW)

    changes = by_key(plan)
    # Moved, not duplicated: every change names the event already on the calendar.
    assert {c.op for c in plan.changes} == {ScheduleOp.UPDATE}
    assert {c.event_id for c in plan.changes} == {e.event_id for e in calendar}
    assert changes[INTERVIEW_KEY].starts_at == moved_to
    assert changes[INTERVIEW_KEY].previous_starts_at == INTERVIEW
    assert changes[DEEP].ends_at == datetime(2026, 10, 4, 20, 0)
    assert changes[WARM].ends_at == moved_to - timedelta(minutes=15)
    for key in (DEEP, WARM):
        assert changes[key].ends_at <= moved_to


def test_a_small_move_leaves_a_prep_block_that_still_fits():
    calendar = scheduled(plan_interview_schedule(request(), [], now=NOW))

    plan = plan_interview_schedule(request(INTERVIEW + timedelta(minutes=30)), calendar, now=NOW)

    changes = by_key(plan)
    assert changes[INTERVIEW_KEY].op is ScheduleOp.UPDATE
    assert changes[DEEP].op is ScheduleOp.KEEP
    assert changes[WARM].op is ScheduleOp.KEEP


def test_a_prep_block_something_was_booked_over_is_moved():
    calendar = scheduled(plan_interview_schedule(request(), [], now=NOW))
    calendar.append(booked("dentist", datetime(2026, 9, 30, 19, 0), datetime(2026, 9, 30, 20, 0)))

    changes = by_key(plan_interview_schedule(request(), calendar, now=NOW))

    assert changes[INTERVIEW_KEY].op is ScheduleOp.KEEP
    assert changes[WARM].op is ScheduleOp.KEEP
    assert changes[DEEP].op is ScheduleOp.UPDATE
    assert changes[DEEP].ends_at == datetime(2026, 9, 30, 19, 0)


def test_a_prep_block_already_over_is_left_alone():
    calendar = scheduled(plan_interview_schedule(request(), [], now=NOW))
    later = datetime(2026, 10, 1, 9, 0)  # the morning of the interview

    plan = plan_interview_schedule(request(datetime(2026, 10, 8, 14, 0)), calendar, now=later)

    assert by_key(plan)[DEEP].op is ScheduleOp.KEEP


def test_another_applications_events_are_just_busy_time():
    other = CalendarEvent(
        event_id="other-interview",
        starts_at=datetime(2026, 9, 30, 18, 0),
        ends_at=datetime(2026, 9, 30, 20, 0),
        properties={PROP_APPLICATION_ID: "someone-else", PROP_KEY: "interview:someone-else"},
    )

    plan = plan_interview_schedule(request(), [other], now=NOW)

    assert {c.op for c in plan.changes} == {ScheduleOp.CREATE}
    assert by_key(plan)[DEEP].ends_at == datetime(2026, 9, 30, 18, 0)


# --- Reminders, keys and properties --------------------------------------------


def test_the_reminder_is_the_start_of_the_first_prep_block():
    plan = plan_interview_schedule(request(), [], now=NOW)

    assert plan.reminder_at(NOW) == datetime(2026, 9, 30, 18, 30)
    # With the long block behind us, the short one.
    assert plan.reminder_at(datetime(2026, 10, 1, 9, 0)) == datetime(2026, 10, 1, 13, 15)
    # And never at or after the interview itself.
    assert plan.reminder_at(INTERVIEW) is None


def test_keys_are_stable_for_the_same_write_and_differ_for_a_different_one():
    first = by_key(plan_interview_schedule(request(), [], now=NOW))
    again = by_key(plan_interview_schedule(request(), [], now=NOW))
    assert calendar_create_key(first[DEEP]) == calendar_create_key(again[DEEP])
    assert calendar_create_key(first[DEEP]) != calendar_create_key(first[WARM])

    calendar = scheduled(plan_interview_schedule(request(), [], now=NOW))
    to_monday = by_key(
        plan_interview_schedule(request(datetime(2026, 10, 5, 10, 0)), calendar, now=NOW)
    )
    to_tuesday = by_key(
        plan_interview_schedule(request(datetime(2026, 10, 6, 10, 0)), calendar, now=NOW)
    )
    assert calendar_update_key(to_monday[INTERVIEW_KEY]) != calendar_update_key(
        to_tuesday[INTERVIEW_KEY]
    )
    assert interview_reminder_dedupe_key(APP, INTERVIEW) != interview_reminder_dedupe_key(
        APP, INTERVIEW + timedelta(days=1)
    )


def test_a_prep_block_is_stamped_with_the_interview_it_belongs_to():
    changes = by_key(plan_interview_schedule(request(), [], now=NOW))

    prep = event_properties(APP, changes[DEEP])
    assert prep == {
        PROP_APPLICATION_ID: str(APP),
        PROP_KEY: DEEP,
        PROP_ROLE: "prep",
        PROP_INTERVIEW_KEY: INTERVIEW_KEY,
    }
    assert PROP_INTERVIEW_KEY not in event_properties(APP, changes[INTERVIEW_KEY])


# --- Reading the interview time ------------------------------------------------


def commitment(action: str, due_at: datetime | None, confidence: float = 0.8) -> Commitment:
    return Commitment(
        actor=CommitmentActor.EXTERNAL_PERSON,
        action=action,
        due_at=due_at,
        confidence=confidence,
        source_message_id="msg-1",
    )


@pytest.mark.parametrize(
    "action",
    [
        "Technical interview with the team on Thursday at 2pm",
        "Phone screen with Priya",
        "Onsite at the Northwind office",
    ],
)
def test_a_commitment_that_reads_like_the_interview_gives_its_time(action):
    assert interview_time_from((commitment(action, INTERVIEW),)) == INTERVIEW


@pytest.mark.parametrize(
    "action",
    [
        "Please send your availability for an interview by Friday",
        "Complete the take-home exercise",
        "We will get back to you next week",
    ],
)
def test_arranging_an_interview_is_not_an_interview_time(action):
    assert interview_time_from((commitment(action, INTERVIEW),)) is None


def test_an_interview_with_no_stated_time_gives_none():
    assert interview_time_from((commitment("Interview with the team", None),)) is None


def test_the_most_confident_reading_of_the_time_wins():
    guess = commitment("Interview, maybe Wednesday", INTERVIEW - timedelta(days=1), 0.4)
    stated = commitment("Interview on Thursday at 2pm", INTERVIEW, 0.9)

    assert interview_time_from((guess, stated)) == INTERVIEW


def test_only_one_of_our_interview_events_becomes_a_request():
    moved = ours("evt-1", INTERVIEW_KEY, INTERVIEW, INTERVIEW + timedelta(minutes=45))

    from_event = interview_request_from_event(moved)

    assert from_event.application_id == APP
    assert from_event.starts_at == INTERVIEW
    assert from_event.duration == timedelta(minutes=45)
    assert interview_request_from_event(ours("evt-2", DEEP, NOW, INTERVIEW)) is None
    assert interview_request_from_event(booked("lunch", NOW, INTERVIEW)) is None
