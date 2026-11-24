"""Time-phrase resolution, zones and DST, and local availability -- as pure functions."""

from datetime import datetime, time, timedelta

import pytest

from personalos.domain.calendar import (
    CalendarContractError,
    calendar_write_intent,
    iana_zone,
    local_free_windows,
    localize,
    parse_calendar_request,
    parse_duration,
    propose_slots,
    resolve_when,
)
from personalos.domain.interview_scheduling import BusyInterval, CalendarEventRole
from personalos.domain.job_search import ActionKind

NY = iana_zone("America/New_York")
#: Thursday 5 March 2026, 10:00 EST. New York springs forward on Sunday 8 March.
BEFORE_SPRING = datetime(2026, 3, 5, 15, 0)
#: Thursday 29 October 2026, 10:00 EDT. New York falls back on Sunday 1 November.
BEFORE_FALL = datetime(2026, 10, 29, 14, 0)


@pytest.mark.parametrize("name", ["EST", "PST", "", "Mars/Olympus", "GMT+5"])
def test_only_iana_area_location_zones_are_accepted(name):
    with pytest.raises(CalendarContractError):
        iana_zone(name)


def test_utc_is_accepted():
    assert iana_zone("UTC").key == "UTC"


# --- DST ---------------------------------------------------------------------


def test_a_monday_after_spring_forward_is_four_hours_behind_utc():
    resolved = resolve_when("next monday at 9am", now=BEFORE_SPRING, zone=NY)
    assert resolved.starts_at == datetime(2026, 3, 9, 13, 0)
    assert resolved.local_start.utcoffset() == timedelta(hours=-4)


def test_a_monday_after_fall_back_is_five_hours_behind_utc():
    resolved = resolve_when("next monday at 9am", now=BEFORE_FALL, zone=NY)
    assert resolved.starts_at == datetime(2026, 11, 2, 14, 0)
    assert resolved.local_start.utcoffset() == timedelta(hours=-5)


def test_a_wall_time_in_the_spring_gap_moves_forward_with_a_note():
    local, note = localize(datetime(2026, 3, 8, 2, 30), NY)
    assert local.strftime("%H:%M") == "03:30"
    assert local.utcoffset() == timedelta(hours=-4)
    assert note and "does not exist" in note


def test_a_wall_time_in_the_fall_overlap_takes_the_first_with_a_note():
    local, note = localize(datetime(2026, 11, 1, 1, 30), NY)
    assert local.utcoffset() == timedelta(hours=-4)
    assert note and "happens twice" in note


def test_an_event_spanning_spring_forward_keeps_its_real_length():
    resolved = resolve_when(
        "2026-03-08 at 1:30am for 2 hours", now=BEFORE_SPRING, zone=NY, duration=timedelta(hours=2)
    )
    assert resolved.duration == timedelta(hours=2)
    # 01:30 EST + 2h of real time = 04:30 EDT on the wall.
    assert resolved.local_end.strftime("%H:%M") == "04:30"


def test_a_whole_day_on_the_spring_forward_date_is_twenty_three_hours():
    resolved = resolve_when("2026-03-08", now=BEFORE_SPRING, zone=NY)
    assert resolved.ends_at - resolved.starts_at == timedelta(hours=23)


# --- Phrases -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "local"),
    [
        ("thursday", "2026-03-05"),  # today counts for a bare weekday
        ("next thursday", "2026-03-12"),
        ("friday", "2026-03-06"),
        ("tomorrow at 3pm", "2026-03-06T15:00"),
        ("march 9 at 9:30am", "2026-03-09T09:30"),
        ("9 march at 14:00", "2026-03-09T14:00"),
        ("2026-03-10 at noon", "2026-03-10T12:00"),
    ],
)
def test_phrases_resolve_to_the_expected_local_day_or_time(phrase, local):
    resolved = resolve_when(phrase, now=BEFORE_SPRING, zone=NY)
    assert resolved.local_start.isoformat().startswith(local) or (
        resolved.local_start.date().isoformat() == local
    )


def test_a_part_of_day_bounds_the_window():
    resolved = resolve_when("next tuesday afternoon", now=BEFORE_SPRING, zone=NY)
    assert (resolved.local_start.time(), resolved.local_end.time()) == (time(12), time(17))
    assert resolved.exact is False


def test_a_time_with_no_day_that_has_passed_is_tomorrow_and_says_so():
    resolved = resolve_when("at 8am", now=BEFORE_SPRING, zone=NY)
    assert resolved.local_start.date().isoformat() == "2026-03-06"
    assert any("has passed" in note for note in resolved.notes)


def test_a_month_day_already_past_this_year_is_next_year():
    resolved = resolve_when("january 5 at 9am", now=BEFORE_SPRING, zone=NY)
    assert resolved.local_start.year == 2027
    assert any("no year given" in note for note in resolved.notes)


def test_no_day_and_no_time_is_none():
    assert resolve_when("set something up with Bob", now=BEFORE_SPRING, zone=NY) is None


@pytest.mark.parametrize(
    ("text", "minutes"),
    [("for 45 minutes", 45), ("a 30-minute call", 30), ("for 1.5 hours", 90), ("2 hrs", 120)],
)
def test_durations(text, minutes):
    assert parse_duration(text) == timedelta(minutes=minutes)


def test_a_move_names_the_event_and_where_it_goes():
    request = parse_calendar_request(
        "move my Acme interview to Friday at 3pm", now=BEFORE_SPRING, time_zone="America/New_York"
    )
    assert request.event_hint == "acme interview"
    assert request.when is not None and request.when.local_start.strftime("%a %H:%M") == "Fri 15:00"


def test_attendees_and_title_are_read_from_a_create():
    request = parse_calendar_request(
        "schedule a call with Priya@Acme.example tomorrow at 2pm titled 'Acme sync'",
        now=BEFORE_SPRING,
        time_zone="America/New_York",
    )
    assert request.attendees == ("priya@acme.example",)
    assert request.title == "Acme sync"


# --- Availability --------------------------------------------------------------------


def test_free_windows_are_cut_to_the_local_working_day_on_each_side_of_dst():
    """Fri 6 Mar (EST) and Mon 9 Mar (EDT): 09:00-17:00 local is a different UTC span."""
    windows = local_free_windows(
        (),
        start=datetime(2026, 3, 6, 5, 0),
        end=datetime(2026, 3, 10, 4, 0),
        zone=NY,
    )
    spans = [(w.starts_at, w.ends_at) for w in windows]
    assert (datetime(2026, 3, 6, 14), datetime(2026, 3, 6, 22)) in spans
    assert (datetime(2026, 3, 9, 13), datetime(2026, 3, 9, 21)) in spans


def test_busy_time_splits_a_window():
    busy = (BusyInterval(starts_at=datetime(2026, 3, 6, 16), ends_at=datetime(2026, 3, 6, 17)),)
    windows = local_free_windows(
        busy, start=datetime(2026, 3, 6, 5), end=datetime(2026, 3, 7, 5), zone=NY
    )
    assert [(w.local_start.strftime("%H:%M"), w.local_end.strftime("%H:%M")) for w in windows] == [
        ("09:00", "11:00"),
        ("12:00", "17:00"),
    ]


def test_proposals_land_on_the_local_half_hour():
    busy = (BusyInterval(starts_at=datetime(2026, 3, 6, 14), ends_at=datetime(2026, 3, 6, 14, 10)),)
    windows = local_free_windows(
        busy, start=datetime(2026, 3, 6, 5), end=datetime(2026, 3, 7, 5), zone=NY
    )
    (slot,) = propose_slots(windows, duration=timedelta(minutes=30), zone=NY)
    assert slot.local_start.strftime("%H:%M") == "09:30"


# --- Write intents ---------------------------------------------------------------------


def test_a_move_intent_names_what_changes_in_local_time():
    intent = calendar_write_intent(
        kind=ActionKind.UPDATE_CALENDAR_EVENT,
        title="Acme sync",
        starts_at=datetime(2026, 3, 9, 13),
        ends_at=datetime(2026, 3, 9, 14),
        zone=NY,
        requested_by="graph:calendar#plan_calendar_change",
        idempotency_key="cal-update:e1:1:20260309T1300",
        logical_key="event:e1",
        role=CalendarEventRole.EVENT,
        now=BEFORE_SPRING,
        event_id="e1",
        previous=(datetime(2026, 3, 6, 14), datetime(2026, 3, 6, 15)),
    )
    assert intent.payload["changes"][0] == {
        "field": "start",
        "from": "2026-03-06T09:00:00-05:00",
        "to": "2026-03-09T09:00:00-04:00",
    }
    assert "America/New_York" in intent.summary


def test_an_update_without_its_previous_time_is_refused():
    with pytest.raises(CalendarContractError):
        calendar_write_intent(
            kind=ActionKind.UPDATE_CALENDAR_EVENT,
            title="x",
            starts_at=datetime(2026, 3, 9, 13),
            ends_at=datetime(2026, 3, 9, 14),
            zone=NY,
            requested_by="graph:calendar#plan_calendar_change",
            idempotency_key="k",
            logical_key=None,
            role=None,
            now=BEFORE_SPRING,
            event_id="e1",
        )
