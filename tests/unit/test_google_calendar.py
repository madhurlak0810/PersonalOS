"""`GoogleCalendar` over the real Google MCP tools, against a fake Google API.

The production path, end to end: `CalendarActionExecutor` and
`CalendarReconciler` inside `ToolExecutor` (the ledger on a real database),
writing through `GoogleCalendar`, which dispatches `google.calendar_*` intents
through the policy gateway to the Google MCP server -- in-process and over
MCP stdio -- whose HTTP port is `FakeGoogleApi`. The production per-node
allowlists are in force.

The acceptance criteria this file exists for:

- a create that lands at Google but whose response is lost is reconciled by
  its idempotency stamp, and the retry creates no second event;
- an event PersonalOS did not create (no stamp) is never patched or deleted,
  whatever an approved action says.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from personalos.bootstrap import build_calendar_action_executor
from personalos.domain.errors import ValidationFailed
from personalos.domain.interview_scheduling import (
    PAYLOAD_ENDS_AT,
    PAYLOAD_EVENT_ID,
    PAYLOAD_PROPERTIES,
    PAYLOAD_STARTS_AT,
    PAYLOAD_TIME_ZONE,
    PAYLOAD_TITLE,
    PROP_APPLICATION_ID,
    PROP_IDEMPOTENCY_KEY,
    PROP_KEY,
    PROP_ROLE,
    PROP_SCHEDULED_STARTS_AT,
    PROP_WORKFLOW_ID,
    BusyInterval,
    WorkingHours,
    availability_windows,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ApprovalDecision,
    ApprovalVerdict,
)
from personalos.domain.models import ToolExecutionStatus
from personalos.executor.calendar import CalendarActionExecutor, WriteAuthority
from personalos.executor.google_calendar import GoogleCalendar, to_utc_naive
from personalos.persistence.models import ToolExecutionModel
from personalos.policy import PolicyViolation
from personalos.policy.rules import GOOGLE_NODE_TOOL_ALLOWLISTS
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.durable_workflow import session_factory
from tests.fixtures.fake_google import VARIANTS, GoogleHarness

SCHEDULER = "graph:job_search#plan_interview_schedule"
ZONE = "America/New_York"
#: 10:00 in New York on 1 October 2026 (EDT, UTC-4).
STARTS = datetime(2026, 10, 1, 14, 0)
ENDS = STARTS + timedelta(hours=1)
KEY = f"interview:{fakes.APPLICATION_ID}"


@pytest.fixture(params=VARIANTS)
async def google(request):
    harness = await GoogleHarness(
        request.param, node_allowlists=GOOGLE_NODE_TOOL_ALLOWLISTS
    ).start()
    try:
        yield harness
    finally:
        await harness.stop()


def calendar_for(google: GoogleHarness, zone: str = ZONE) -> GoogleCalendar:
    return GoogleCalendar(google.gateway, time_zone=zone)


def create_intent(**payload) -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.CREATE_CALENDAR_EVENT,
        target="calendar: new event 'Interview: Acme'",
        summary="Add 'Interview: Acme' to the calendar",
        payload={
            "application_id": str(fakes.APPLICATION_ID),
            PAYLOAD_TITLE: "Interview: Acme",
            PAYLOAD_STARTS_AT: STARTS.isoformat(),
            PAYLOAD_ENDS_AT: ENDS.isoformat(),
            PAYLOAD_PROPERTIES: {
                PROP_APPLICATION_ID: str(fakes.APPLICATION_ID),
                PROP_KEY: KEY,
                PROP_ROLE: "interview",
            },
            **payload,
        },
        idempotency_key=f"cal-create:{KEY}:20261001T1400",
        requested_by=SCHEDULER,
    )


def event_intent(kind: ActionKind, event_id: str, starts: datetime = STARTS) -> ActionIntent:
    return ActionIntent(
        kind=kind,
        target=f"calendar: event {event_id}",
        summary=f"{kind.value} {event_id}",
        payload={
            PAYLOAD_EVENT_ID: event_id,
            PAYLOAD_TITLE: "Interview: Acme",
            PAYLOAD_STARTS_AT: starts.isoformat(),
            PAYLOAD_ENDS_AT: (starts + timedelta(hours=1)).isoformat(),
        },
        idempotency_key=f"cal-{kind.value}:{event_id}:{starts:%Y%m%dT%H%M}",
        requested_by=SCHEDULER,
    )


def approved(intent: ActionIntent) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
    )


def inserts(google: GoogleHarness) -> list:
    return [c for c in google.api.calls if c.method == "POST" and c.url.endswith("/events")]


def writes(google: GoogleHarness) -> list:
    return [c for c in google.api.calls if c.method in ("PATCH", "DELETE")]


def foreign_event(google: GoogleHarness, event_id: str = "dentist") -> None:
    """An event the candidate made: no PersonalOS stamp at all."""
    google.api.add_event(
        "primary",
        {
            "id": event_id,
            "etag": '"e1"',
            "summary": "Dentist",
            "start": {"dateTime": "2026-10-01T09:00:00-04:00", "timeZone": ZONE},
            "end": {"dateTime": "2026-10-01T10:00:00-04:00", "timeZone": ZONE},
        },
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


# --- Acceptance: the insert landed, its response did not ----------------------


async def test_a_lost_create_response_is_reconciled_by_stamp_and_not_duplicated(google, tmp_path):
    """Google stores the event and answers 504. The one permitted retry reconciles first.

    `ToolExecutor` sees a retryable failure; the calendar retry policy allows
    one more attempt only after reconciliation, and `CalendarReconciler` --
    listing by the idempotency stamp through `calendar_list_events` -- finds
    the event that landed. Its receipt is recorded; nothing is inserted again.
    """
    google.api.fail_next_insert_after_commit = True
    factory = session_factory(tmp_path / "gcal.db")
    workflow_id = uuid4()
    intent = create_intent()

    executor = build_calendar_action_executor(
        calendar_for(google), factory, workflow_id=workflow_id
    )
    receipt = await executor.execute(intent, approved(intent))

    (event_id,) = google.api.events["primary"]
    assert receipt.ok is True
    assert receipt.external_reference == event_id
    assert len(inserts(google)) == 1, "the retry reconciled instead of inserting again"
    assert execution_status(factory, intent.idempotency_key) == (
        ToolExecutionStatus.COMPLETED.value
    )

    stored = google.api.events["primary"][event_id]
    private = stored["extendedProperties"]["private"]
    assert private[PROP_IDEMPOTENCY_KEY] == intent.idempotency_key
    assert private[PROP_WORKFLOW_ID] == str(workflow_id)
    assert private[PROP_APPLICATION_ID] == str(fakes.APPLICATION_ID)

    # A restarted worker replays the stored receipt without asking Google.
    calls = len(google.api.calls)
    again = await build_calendar_action_executor(
        calendar_for(google), session_factory(tmp_path / "gcal.db"), workflow_id=workflow_id
    ).execute(intent, approved(intent))
    assert again.external_reference == event_id
    assert len(google.api.calls) == calls


async def test_a_create_that_never_reached_google_is_made_once_on_retry(google, tmp_path):
    def unavailable(call):
        google.api.overrides.clear()
        from mcp_servers.google.api import raise_for_status

        raise_for_status(503)

    google.api.override("POST", r".*/calendars/primary/events", unavailable)
    intent = create_intent()

    receipt = await build_calendar_action_executor(
        calendar_for(google), session_factory(tmp_path / "gcal.db")
    ).execute(intent, approved(intent))

    assert receipt.ok is True
    assert list(google.api.events["primary"]) == [receipt.external_reference]


# --- Acceptance: events PersonalOS did not create are left alone --------------


@pytest.mark.parametrize(
    "kind", [ActionKind.UPDATE_CALENDAR_EVENT, ActionKind.CANCEL_CALENDAR_EVENT]
)
async def test_an_unstamped_event_is_never_modified_or_deleted(google, tmp_path, kind):
    """Even an approved update or cancel aimed at it is refused before any write is sent."""
    foreign_event(google)
    before = dict(google.api.events["primary"]["dentist"])
    intent = event_intent(kind, "dentist", STARTS + timedelta(days=1))

    receipt = await build_calendar_action_executor(
        calendar_for(google), session_factory(tmp_path / "gcal.db")
    ).execute(intent, approved(intent))

    assert receipt.ok is False
    assert "not created by PersonalOS" in (receipt.detail or "")
    assert writes(google) == []
    assert google.api.events["primary"]["dentist"] == before


async def test_a_stamped_event_can_be_moved_and_cancelled(google, tmp_path):
    factory = session_factory(tmp_path / "gcal.db")
    calendar = calendar_for(google)
    create = create_intent()
    created = await build_calendar_action_executor(calendar, factory).execute(
        create, approved(create)
    )
    event_id = created.external_reference

    moved_to = STARTS + timedelta(days=2)
    move = event_intent(ActionKind.UPDATE_CALENDAR_EVENT, event_id, moved_to)
    moved = await build_calendar_action_executor(calendar, factory).execute(move, approved(move))
    assert moved.ok is True
    event = await calendar.get_event(event_id)
    assert (event.starts_at, event.ends_at) == (moved_to, moved_to + timedelta(hours=1))
    # The PATCH merged the new stamps and kept the links.
    assert event.properties[PROP_KEY] == KEY
    assert event.properties[PROP_IDEMPOTENCY_KEY] == move.idempotency_key
    assert event.properties[PROP_SCHEDULED_STARTS_AT] == f"{moved_to:%Y%m%dT%H%M}"

    cancel = event_intent(ActionKind.CANCEL_CALENDAR_EVENT, event_id, moved_to)
    cancelled = await build_calendar_action_executor(calendar, factory).execute(
        cancel, approved(cancel)
    )
    assert cancelled.ok is True
    assert event_id not in google.api.events["primary"]
    assert await calendar.get_event(event_id) is None


async def test_a_write_without_its_approval_is_never_dispatched(google):
    calendar = calendar_for(google)
    intent = create_intent()
    rejected = approved(intent).model_copy(update={"verdict": ApprovalVerdict.REJECTED})

    with pytest.raises(PolicyViolation):
        await calendar.create_event(
            title="Interview",
            starts_at=STARTS,
            ends_at=ENDS,
            description="",
            properties={},
            authority=WriteAuthority(intent, rejected),
        )
    with pytest.raises(PolicyViolation):
        await CalendarActionExecutor(calendar).execute(intent, rejected)
    assert inserts(google) == []


# --- Reads, free/busy and availability need no approval -----------------------


async def test_reads_go_through_without_approval(google):
    foreign_event(google)
    calendar = calendar_for(google)

    (event,) = await calendar.events_between(fakes.USER_ID, STARTS - timedelta(days=1), ENDS)
    assert event.event_id == "dentist"
    assert await calendar.get_event("missing") is None
    assert await calendar.find_by_property(PROP_KEY, KEY) == []
    assert await calendar.free_busy(STARTS - timedelta(days=1), ENDS)


async def test_free_busy_and_availability_windows(google):
    foreign_event(google)  # 13:00-14:00 UTC
    calendar = calendar_for(google)
    day_start, day_end = datetime(2026, 10, 1, 12, 0), datetime(2026, 10, 1, 16, 0)

    busy = await calendar.free_busy(day_start, day_end)
    assert busy == (
        BusyInterval(starts_at=datetime(2026, 10, 1, 13, 0), ends_at=datetime(2026, 10, 1, 14, 0)),
    )
    windows = await calendar.availability(day_start, day_end, min_duration=timedelta(minutes=45))
    assert [(w.starts_at, w.ends_at) for w in windows] == [
        (day_start, datetime(2026, 10, 1, 13, 0)),
        (datetime(2026, 10, 1, 14, 0), day_end),
    ]


async def test_a_free_busy_error_is_not_read_as_free(google):
    calendar = calendar_for(google)
    with pytest.raises(ValidationFailed):
        await calendar.free_busy(STARTS, ENDS, calendar_ids=["someone-else@example.test"])


# --- Time zones ---------------------------------------------------------------


async def test_writes_send_utc_instants_and_an_iana_zone(google, tmp_path):
    intent = create_intent()
    await build_calendar_action_executor(
        calendar_for(google), session_factory(tmp_path / "gcal.db")
    ).execute(intent, approved(intent))

    (call,) = inserts(google)
    assert call.json["start"] == {"dateTime": "2026-10-01T14:00:00+00:00", "timeZone": ZONE}
    assert call.json["end"] == {"dateTime": "2026-10-01T15:00:00+00:00", "timeZone": ZONE}


async def test_an_intent_may_name_its_own_zone(google, tmp_path):
    intent = create_intent(**{PAYLOAD_TIME_ZONE: "Europe/Berlin"})
    await build_calendar_action_executor(
        calendar_for(google), session_factory(tmp_path / "gcal.db")
    ).execute(intent, approved(intent))
    (call,) = inserts(google)
    assert call.json["start"]["timeZone"] == "Europe/Berlin"
    assert call.json["start"]["dateTime"] == "2026-10-01T14:00:00+00:00"


async def test_reads_convert_from_the_offset_google_gives(google):
    foreign_event(google)
    google.api.add_event(
        "primary",
        {
            "id": "holiday",
            "summary": "Day off",
            "start": {"date": "2026-10-02"},
            "end": {"date": "2026-10-03"},
        },
    )
    calendar = calendar_for(google)

    dentist = await calendar.get_event("dentist")
    assert (dentist.starts_at, dentist.ends_at) == (
        datetime(2026, 10, 1, 13, 0),
        datetime(2026, 10, 1, 14, 0),
    )
    assert dentist.time_zone == ZONE
    # An all-day event is the day in the calendar's zone, not the server's.
    holiday = await calendar.get_event("holiday")
    assert (holiday.starts_at, holiday.ends_at) == (
        datetime(2026, 10, 2, 4, 0),
        datetime(2026, 10, 3, 4, 0),
    )


def test_a_naive_time_from_outside_is_refused_not_guessed():
    with pytest.raises(ValidationFailed):
        to_utc_naive(datetime(2026, 10, 1, 9, 0))
    assert to_utc_naive(datetime(2026, 10, 1, 9, 0, tzinfo=timezone(timedelta(hours=-4)))) == (
        datetime(2026, 10, 1, 13, 0)
    )


def test_the_calendar_needs_a_real_iana_zone(google):
    for bad in ("EST", "+05:00", "Mars/Olympus_Mons", ""):
        with pytest.raises(ValueError):
            GoogleCalendar(google.gateway, time_zone=bad)


def test_the_tool_contract_refuses_a_zone_that_is_not_iana():
    from mcp_servers.google.schemas import CalendarCreateParams

    with pytest.raises(ValueError):
        CalendarCreateParams(
            summary="x",
            start=datetime(2026, 10, 1, 9, tzinfo=timezone.utc),
            end=datetime(2026, 10, 1, 10, tzinfo=timezone.utc),
            time_zone="EST5EDT; DROP",
        )


# --- availability_windows on its own --------------------------------------------


def test_availability_merges_busy_time_and_respects_working_hours():
    start, end = datetime(2026, 10, 1, 0, 0), datetime(2026, 10, 2, 0, 0)
    busy = [
        BusyInterval(starts_at=datetime(2026, 10, 1, 13), ends_at=datetime(2026, 10, 1, 14)),
        BusyInterval(starts_at=datetime(2026, 10, 1, 13, 30), ends_at=datetime(2026, 10, 1, 15)),
        BusyInterval(starts_at=datetime(2026, 10, 1, 15), ends_at=datetime(2026, 10, 1, 15, 10)),
    ]
    # 09:00-17:00 in New York (UTC-4) is 13:00-21:00 UTC.
    hours = WorkingHours(utc_offset=timedelta(hours=-4)).model_copy(
        update={"start": datetime(2000, 1, 1, 9).time(), "end": datetime(2000, 1, 1, 17).time()}
    )
    windows = availability_windows(
        busy, start=start, end=end, min_duration=timedelta(minutes=30), working_hours=hours
    )
    assert [(w.starts_at, w.ends_at) for w in windows] == [
        (datetime(2026, 10, 1, 15, 10), datetime(2026, 10, 1, 21, 0)),
    ]
