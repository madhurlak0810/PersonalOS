"""The Calendar subgraph's boundaries: availability, approval, exactly-once, conflicts, zones.

Most tests run over the production path to Google -- the policy gateway with
the production per-node allowlists, the Google MCP server, a fake Google API,
and the calendar executor under `ToolExecutor` with its ledger on SQLite
(`tests.fixtures.calendar.build_google_world`). The rest use the in-memory
calendar, where the point is the graph's own decision.

The acceptance criteria this file exists for:

1. 'when am I free Thursday' returns windows computed from fake free/busy data
   in the configured time zone.
2. An event create without an approval record is denied; with approval it
   creates exactly one event, even when the first response is lost.
3. A request that would double-book returns a conflict result and no write.
4. An event created outside PersonalOS is never modified or deleted by the
   executor.
5. A DST-boundary date resolves to the correct absolute time.

Plus: a moved interview reconciles its linked prep blocks in the same
approval; an application of another user is refused; the Calendar
subgraph's identities reach their calendar tools and nothing else; and the
Supervisor routes calendar requests here.
"""

from datetime import datetime, timedelta
from typing import Any

import pytest
from langgraph.types import Command

from personalos.domain.calendar import (
    PAYLOAD_ATTENDEES,
    PAYLOAD_CHANGES,
    PAYLOAD_LOCAL_ENDS_AT,
    PAYLOAD_LOCAL_STARTS_AT,
    PAYLOAD_RESOLUTION_NOTES,
    CalendarChangePlan,
    CalendarOutcome,
    TimeSlot,
)
from personalos.domain.interview_scheduling import (
    PAYLOAD_EVENT_ID,
    PAYLOAD_TIME_ZONE,
    PAYLOAD_TITLE,
    PROP_IDEMPOTENCY_KEY,
    BusyInterval,
    CalendarEventRole,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ApprovalDecision,
    ApprovalVerdict,
    RefusalReason,
)
from personalos.domain.routing import RouteDomain
from personalos.graphs import calendar as cal
from personalos.graphs.calendar import CalendarSubgraphRunner
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.routing import KeywordIntentClassifier
from personalos.policy import ApprovalRequired, IntentOrigin, ToolIntent
from personalos.policy.engine import Decision
from tests.fixtures import calendar as cf


@pytest.fixture
def google(tmp_path):
    return cf.build_google_world(tmp_path / "calendar.db")


def plan_of(final: dict[str, Any]) -> CalendarChangePlan:
    return CalendarChangePlan.model_validate(final["change_plan"])


def approved(intent: ActionIntent) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=ApprovalVerdict.APPROVED,
        decided_by="sam",
    )


# --- 1. Availability from free/busy, in the configured zone --------------------------


async def test_when_am_i_free_thursday_answers_from_free_busy_in_the_configured_zone(google):
    """Thursday 26 Nov 2026 in New York (EST, UTC-5), working day 09:00-17:00 local.

    Two bookings: the dentist 10:00-11:00 local, and a block 13:30-14:30 local
    that only free/busy knows of. The windows are the gaps between them, cut
    to the local working day -- 14:00-22:00 UTC, not 09:00-17:00 UTC.
    """
    google.add_event("dentist", "Dentist", cf.utc(2026, 11, 26, 15), cf.utc(2026, 11, 26, 16))
    google.add_event(
        "lunch", "Team lunch", cf.utc(2026, 11, 26, 18, 30), cf.utc(2026, 11, 26, 19, 30)
    )

    final = await google.graph.build().ainvoke(cf.start("when am I free Thursday"), cf.thread())

    windows = [TimeSlot.model_validate(raw) for raw in final["availability"]]
    assert [(w.local_start.strftime("%H:%M"), w.local_end.strftime("%H:%M")) for w in windows] == [
        ("09:00", "10:00"),
        ("11:00", "13:30"),
        ("14:30", "17:00"),
    ]
    assert {w.time_zone for w in windows} == {cf.ZONE}
    assert {w.local_start.utcoffset() for w in windows} == {timedelta(hours=-5)}
    assert [(w.starts_at, w.ends_at) for w in windows] == [
        (cf.utc(2026, 11, 26, 14), cf.utc(2026, 11, 26, 15)),
        (cf.utc(2026, 11, 26, 16), cf.utc(2026, 11, 26, 18, 30)),
        (cf.utc(2026, 11, 26, 19, 30), cf.utc(2026, 11, 26, 22)),
    ]
    assert final["errors"] == []
    # Read, never written; and the read was free/busy, not a guess from listings.
    assert google.writes() == []
    assert any(c.url.endswith("/freeBusy") for c in google.google.api.calls)


async def test_free_busy_from_another_calendar_counts_as_busy():
    """A booking free/busy reports but no listed event explains still blocks the window."""
    world = cf.build_world()
    world.calendar.extra_busy.append(
        BusyInterval(starts_at=cf.utc(2026, 11, 26, 14), ends_at=cf.utc(2026, 11, 26, 20))
    )

    final = await world.graph.build().ainvoke(cf.start("when am I free Thursday"), cf.thread())

    windows = [TimeSlot.model_validate(raw) for raw in final["availability"]]
    assert [(w.local_start.strftime("%H:%M"), w.local_end.strftime("%H:%M")) for w in windows] == [
        ("15:00", "17:00")
    ]


async def test_proposed_interview_times_come_from_free_busy_one_per_day():
    world = cf.build_world(
        [cf.external("busy-mon", "All hands", cf.utc(2026, 11, 30, 14), cf.utc(2026, 11, 30, 17))]
    )

    final = await world.graph.build().ainvoke(
        cf.start("propose interview times next week for 45 minutes"), cf.thread()
    )

    proposals = [TimeSlot.model_validate(raw) for raw in final["proposals"]]
    assert [p.local_start.strftime("%a %H:%M") for p in proposals] == [
        "Mon 12:00",
        "Tue 09:00",
        "Wed 09:00",
        "Thu 09:00",
        "Fri 09:00",
    ]
    assert all(p.duration == timedelta(minutes=45) for p in proposals)
    assert world.calendar.created == []


# --- 2. No create without approval; exactly one with it, even after a lost response ---


async def test_a_create_without_an_approval_record_is_denied(google):
    """Three ways to have no approval, and none of them reaches Google."""
    final = await (compiled := google.graph.build()).ainvoke(
        cf.start("schedule 'Acme sync' Thursday at 2pm"), config := cf.thread()
    )
    (request,) = cf.interrupt_requests(final)
    assert request["kind"] == ActionKind.CREATE_CALENDAR_EVENT.value

    # (a) Resumed with no decision at all: refused at the graph, nothing sent.
    resumed = await compiled.ainvoke(Command(resume=[]), config)
    (refusal,) = resumed["approval_refusals"]
    assert refusal["reason"] == RefusalReason.NOT_APPROVED.value
    assert [r["ok"] for r in resumed["action_receipts"]] == [False]

    # (b) An intent handed straight to the executor with only a pending
    # decision: policy holds the WRITE_EXTERNAL create for approval.
    intent = ActionIntent.model_validate(final["pending_actions"][0])
    pending = approved(intent).model_copy(update={"verdict": ApprovalVerdict.PENDING})
    with pytest.raises(ApprovalRequired):
        await google.graph.action_executor.execute(intent, pending)

    # (c) And the tool itself, called through the gateway with no grant.
    tool = ToolIntent(
        server="google",
        tool="calendar_create_event",
        arguments={
            "calendar_id": "primary",
            "summary": "Acme sync",
            "start": "2026-11-26T19:00:00+00:00",
            "end": "2026-11-26T20:00:00+00:00",
            "idempotency_key": "k-unapproved",
        },
        origin=IntentOrigin.SYSTEM,
        mutating=True,
        requested_by=cal.requester(cal.PLAN_CALENDAR_CHANGE),
    )
    assert google.google.policy.evaluate_intent(tool).decision is Decision.REQUIRE_APPROVAL

    assert google.inserts() == []
    assert google.events() == {}


async def test_a_rejected_create_is_not_written():
    world = cf.build_world()
    compiled, config = world.graph.build(), cf.thread()
    parked = await compiled.ainvoke(cf.start("schedule 'Acme sync' Thursday at 2pm"), config)
    (request,) = cf.interrupt_requests(parked)
    rejection = cf.approve(request).model_copy(update={"verdict": ApprovalVerdict.REJECTED})

    final = await compiled.ainvoke(Command(resume=[rejection.model_dump(mode="json")]), config)

    assert [r["ok"] for r in final["action_receipts"]] == [False]
    assert world.calendar.created == []


async def test_an_approved_create_whose_response_was_lost_creates_exactly_one_event(google):
    """Google stores the event and answers 504. The retry reconciles by stamp first.

    `ToolExecutor` records the attempt with no outcome; before it may try
    again, `CalendarReconciler` lists events by this action's idempotency key
    (stamped in `extendedProperties.private`), finds the one that landed, and
    that becomes the receipt. One insert in total.
    """
    google.google.api.fail_next_insert_after_commit = True
    compiled, config = google.graph.build(), cf.thread()
    parked = await compiled.ainvoke(
        cf.start("schedule a call with priya@acme.example Thursday at 2pm titled 'Acme sync'"),
        config,
    )

    final = await compiled.ainvoke(Command(resume=cf.approve_all(parked)), config)

    (receipt,) = final["action_receipts"]
    assert receipt["ok"] is True
    (event_id,) = google.events()
    assert receipt["external_reference"] == event_id
    assert len(google.inserts()) == 1, "the retry reconciled instead of inserting again"
    stored = google.events()[event_id]
    intent = ActionIntent.model_validate(parked["pending_actions"][0])
    assert stored["extendedProperties"]["private"][PROP_IDEMPOTENCY_KEY] == intent.idempotency_key
    assert [a["email"] for a in stored["attendees"]] == ["priya@acme.example"]
    assert stored["start"]["timeZone"] == cf.ZONE

    # A restarted worker re-running the step replays the receipt; Google is not asked to insert.
    replay = await google.graph.action_executor.execute(intent, approved(intent))
    assert replay.external_reference == event_id
    assert len(google.inserts()) == 1


async def test_a_lost_response_that_fails_the_step_is_reconciled_when_the_step_reruns(tmp_path):
    """The executor raises; the run stops at the write. Re-running it finds the event.

    In-memory calendar under the real `ToolExecutor`: the first attempt is
    recorded with no outcome, and the resumed run asks the calendar before
    it writes again.
    """
    from personalos.bootstrap import build_calendar_action_executor
    from tests.fixtures import job_search_fakes as fakes
    from tests.fixtures.durable_workflow import session_factory

    world = cf.build_world()
    world.graph.action_executor = build_calendar_action_executor(
        world.calendar, session_factory(tmp_path / "ledger.db")
    )
    world.calendar.lose_next_create_response = True
    compiled, config = world.graph.build(), cf.thread()
    parked = await compiled.ainvoke(cf.start("schedule 'Acme sync' Thursday at 2pm"), config)

    with pytest.raises(fakes.LostResponse):
        await compiled.ainvoke(Command(resume=cf.approve_all(parked)), config)
    assert len(world.calendar.created) == 1, "the insert did land"

    final = await compiled.ainvoke(None, config)

    (receipt,) = final["action_receipts"]
    assert receipt["ok"] is True
    assert list(world.calendar.events) == world.calendar.created
    assert len(world.calendar.created) == 1


async def test_the_approval_payload_shows_title_attendees_and_zoned_times(google):
    final = await google.graph.build().ainvoke(
        cf.start(
            "schedule a call with priya@acme.example next tuesday afternoon titled 'Acme sync'"
        ),
        cf.thread(),
    )

    (request,) = cf.interrupt_requests(final)
    payload = request["payload"]
    assert payload[PAYLOAD_TITLE] == "Acme sync"
    assert payload[PAYLOAD_ATTENDEES] == ["priya@acme.example"]
    assert payload[PAYLOAD_TIME_ZONE] == cf.ZONE
    # Asked on Monday 23 Nov: 'next tuesday' is the first Tuesday after today,
    # and the note says so -- that is the reading a reviewer must be able to
    # catch. With nothing booked, the afternoon's first free hour is 12:00 local.
    assert payload[PAYLOAD_LOCAL_STARTS_AT] == "2026-11-24T12:00:00-05:00"
    assert payload[PAYLOAD_LOCAL_ENDS_AT] == "2026-11-24T13:00:00-05:00"
    assert payload["starts_at"] == "2026-11-24T17:00:00"
    notes = " | ".join(payload[PAYLOAD_RESOLUTION_NOTES])
    assert "'next tuesday' read as Tue 24 Nov 2026 (the first one after today)" in notes
    assert "picked the earliest free 60-minute slot" in notes
    # The hash covers all of it: what the reviewer read is what executes.
    assert request["preview"]["recipient"] == "priya@acme.example"
    assert "America/New_York" in request["preview"]["body_diff"]


# --- 3. A double-booking is a typed conflict and no write -------------------------------


async def test_a_double_booking_returns_a_conflict_result_and_no_write(google):
    google.add_event("dentist", "Dentist", cf.utc(2026, 11, 26, 15), cf.utc(2026, 11, 26, 16))

    final = await google.graph.build().ainvoke(
        cf.start("schedule 'Acme sync' Thursday at 10:30am"), cf.thread()
    )

    plan = plan_of(final)
    assert plan.outcome is CalendarOutcome.CONFLICT
    assert plan.conflict is not None
    assert [(c.event_id, c.title) for c in plan.conflict.conflicts_with] == [("dentist", "Dentist")]
    assert plan.conflict.requested.local_start.strftime("%H:%M") == "10:30"
    # Alternatives, nearest first in time, none overlapping the dentist.
    assert plan.conflict.alternatives
    for slot in plan.conflict.alternatives:
        assert not (
            slot.starts_at < cf.utc(2026, 11, 26, 16) and slot.ends_at > cf.utc(2026, 11, 26, 15)
        )
        assert slot.duration == timedelta(hours=1)
    assert final["pending_actions"] == []
    assert "__interrupt__" not in final
    assert google.writes() == []


async def test_moving_an_event_onto_a_taken_slot_is_a_conflict_not_a_move():
    world = cf.build_world(
        [
            cf.managed("sync-1", "Acme sync", cf.utc(2026, 11, 25, 19), cf.utc(2026, 11, 25, 20)),
            cf.external("ext-1", "Dentist", cf.utc(2026, 11, 27, 20), cf.utc(2026, 11, 27, 21)),
        ]
    )

    final = await world.graph.build().ainvoke(
        cf.start("move my Acme sync to Friday at 3pm"), cf.thread()
    )

    plan = plan_of(final)
    assert plan.outcome is CalendarOutcome.CONFLICT
    assert [c.event_id for c in plan.conflict.conflicts_with] == ["ext-1"]
    assert final["pending_actions"] == []
    assert world.calendar.updated == []


async def test_a_conflict_with_free_busy_alone_is_still_a_conflict():
    world = cf.build_world()
    world.calendar.extra_busy.append(
        BusyInterval(starts_at=cf.utc(2026, 11, 26, 19), ends_at=cf.utc(2026, 11, 26, 20))
    )

    final = await world.graph.build().ainvoke(
        cf.start("schedule 'Acme sync' Thursday at 2pm"), cf.thread()
    )

    plan = plan_of(final)
    assert plan.outcome is CalendarOutcome.CONFLICT
    assert [c.event_id for c in plan.conflict.conflicts_with] == [None]
    assert world.calendar.created == []


# --- 4. Events created outside PersonalOS are never modified or deleted ----------------


@pytest.mark.parametrize(
    "request_text",
    ["move my dentist appointment to Friday at 3pm", "cancel my dentist appointment"],
)
async def test_an_event_created_outside_personalos_is_never_planned_for_change(
    google, request_text
):
    google.add_event("dentist", "Dentist", cf.utc(2026, 11, 26, 15), cf.utc(2026, 11, 26, 16))
    before = dict(google.events()["dentist"])

    final = await google.graph.build().ainvoke(cf.start(request_text), cf.thread())

    plan = plan_of(final)
    assert plan.outcome is CalendarOutcome.NOT_MANAGED
    assert plan.event is not None and plan.event.event_id == "dentist"
    assert plan.event.managed is False
    assert final["pending_actions"] == []
    assert google.writes() == []
    assert google.events()["dentist"] == before


@pytest.mark.parametrize(
    "kind", [ActionKind.UPDATE_CALENDAR_EVENT, ActionKind.CANCEL_CALENDAR_EVENT]
)
async def test_even_an_approved_write_aimed_at_an_outside_event_is_refused_by_the_executor(
    google, kind
):
    """Forge the intent the graph would never plan, approve it, execute it: still refused."""
    google.add_event("dentist", "Dentist", cf.utc(2026, 11, 26, 15), cf.utc(2026, 11, 26, 16))
    before = dict(google.events()["dentist"])
    intent = ActionIntent(
        kind=kind,
        target="calendar: event dentist ('Dentist')",
        summary="change the dentist",
        payload={
            PAYLOAD_EVENT_ID: "dentist",
            PAYLOAD_TITLE: "Dentist",
            "starts_at": cf.utc(2026, 11, 27, 15).isoformat(),
            "ends_at": cf.utc(2026, 11, 27, 16).isoformat(),
        },
        idempotency_key=f"forged-{kind.value}:dentist",
        requested_by=cal.requester(cal.PLAN_CALENDAR_CHANGE),
    )

    receipt = await google.graph.action_executor.execute(intent, approved(intent))

    assert receipt.ok is False
    assert "not created by PersonalOS" in (receipt.detail or "")
    assert google.writes() == []
    assert google.events()["dentist"] == before


async def test_an_interview_move_leaves_an_unstamped_event_alone_and_plans_around_it():
    world = cf.build_world(
        [
            *cf.interview_with_prep(cf.utc(2026, 11, 26, 19)),
            cf.external("ext-1", "Focus time", cf.utc(2026, 11, 27, 15), cf.utc(2026, 11, 27, 16)),
        ]
    )

    final = await world.graph.build().ainvoke(
        cf.start("move my Acme interview to Friday at 3pm"), cf.thread()
    )

    event_ids = {
        ActionIntent.model_validate(raw).payload.get(PAYLOAD_EVENT_ID)
        for raw in final["pending_actions"]
    }
    assert "ext-1" not in event_ids


# --- 5. Time zones and DST ------------------------------------------------------------------


async def test_a_dst_boundary_date_resolves_to_the_correct_absolute_time():
    """New York springs forward on Sun 8 Mar 2026. 'next Monday at 9am' asked on the
    Thursday before is 09:00 EDT (UTC-4) = 13:00 UTC -- not 14:00, which is what the
    offset in force when the question was asked (EST, UTC-5) would give."""
    asked = datetime(2026, 3, 5, 15, 0)  # Thu 5 Mar, 10:00 EST
    world = cf.build_world(clock_at=asked)

    final = await world.graph.build().ainvoke(
        cf.start("schedule 'Acme sync' next monday at 9am"), cf.thread()
    )

    (request,) = cf.interrupt_requests(final)
    payload = request["payload"]
    assert payload["starts_at"] == "2026-03-09T13:00:00"
    assert payload["ends_at"] == "2026-03-09T14:00:00"
    assert payload[PAYLOAD_LOCAL_STARTS_AT] == "2026-03-09T09:00:00-04:00"
    assert any("EDT" in note for note in payload[PAYLOAD_RESOLUTION_NOTES])


async def test_a_time_inside_the_dst_gap_is_moved_forward_and_says_so():
    asked = datetime(2026, 3, 5, 15, 0)
    world = cf.build_world(clock_at=asked)

    final = await world.graph.build().ainvoke(
        cf.start("schedule 'Early run' 2026-03-08 at 2:30am"), cf.thread()
    )

    (request,) = cf.interrupt_requests(final)
    payload = request["payload"]
    # 02:30 does not exist that morning; it is read as 03:30 EDT = 07:30 UTC.
    assert payload["starts_at"] == "2026-03-08T07:30:00"
    assert payload[PAYLOAD_LOCAL_STARTS_AT] == "2026-03-08T03:30:00-04:00"
    assert any("does not exist" in note for note in payload[PAYLOAD_RESOLUTION_NOTES])


async def test_availability_across_a_dst_change_keeps_the_local_working_day():
    """Fri 6 Mar is EST, Mon 9 Mar is EDT: 09:00 local is 14:00 UTC then 13:00 UTC."""
    world = cf.build_world(clock_at=datetime(2026, 3, 5, 15, 0))

    final = await world.graph.build().ainvoke(cf.start("when am I free next week"), cf.thread())

    first = TimeSlot.model_validate(final["availability"][0])
    assert first.local_start.strftime("%a %H:%M") == "Mon 09:00"
    assert first.local_start.utcoffset() == timedelta(hours=-4)
    assert first.starts_at == datetime(2026, 3, 9, 13, 0)

    before = await world.graph.build().ainvoke(cf.start("when am I free friday"), cf.thread())
    friday = TimeSlot.model_validate(before["availability"][0])
    assert friday.starts_at == datetime(2026, 3, 6, 14, 0)


# --- Applications and prep blocks ------------------------------------------------------


async def test_moving_a_linked_interview_reconciles_its_prep_blocks_in_one_approval():
    """The interview moves a day later; both prep blocks move with it, in the same batch."""
    interview_at = cf.utc(2026, 11, 26, 19)  # Thu 14:00 EST
    world = cf.build_world(cf.interview_with_prep(interview_at))
    compiled, config = world.graph.build(), cf.thread()

    parked = await compiled.ainvoke(cf.start("move my Acme interview to Friday at 2pm"), config)

    plan = plan_of(parked)
    assert plan.outcome is CalendarOutcome.PROPOSED
    assert plan.application_id == cf.APPLICATION_ID
    requests = cf.interrupt_requests(parked)
    by_event = {r["payload"][PAYLOAD_EVENT_ID]: r["payload"] for r in requests}
    assert set(by_event) == {"int-1", "prep-deep", "prep-warm"}
    interview = by_event["int-1"]
    assert interview["starts_at"] == "2026-11-27T19:00:00"
    assert interview[PAYLOAD_ATTENDEES] == ["recruiter@acme.example"]
    assert interview[PAYLOAD_CHANGES][0] == {
        "field": "start",
        "from": "2026-11-26T14:00:00-05:00",
        "to": "2026-11-27T14:00:00-05:00",
    }
    # The prep blocks are re-planned against the new time.
    for key in ("prep-deep", "prep-warm"):
        assert by_event[key]["interview_starts_at"] == "2026-11-27T19:00:00"
        assert datetime.fromisoformat(by_event[key]["ends_at"]) <= cf.utc(2026, 11, 27, 19)

    final = await compiled.ainvoke(Command(resume=cf.approve_all(parked)), config)

    assert all(r["ok"] for r in final["action_receipts"])
    assert world.calendar.events["int-1"].starts_at == cf.utc(2026, 11, 27, 19)
    assert world.calendar.events["prep-warm"].ends_at <= cf.utc(2026, 11, 27, 19)
    assert world.calendar.events["prep-deep"].starts_at > cf.utc(2026, 11, 26, 19)
    assert world.calendar.created == []


async def test_scheduling_an_interview_for_an_application_creates_it_with_prep_blocks():
    world = cf.build_world()

    final = await world.graph.build().ainvoke(
        cf.start("schedule the Acme interview Friday at 2pm"), cf.thread()
    )

    plan = plan_of(final)
    assert plan.outcome is CalendarOutcome.PROPOSED
    assert plan.application_id == cf.APPLICATION_ID
    roles = sorted(
        ActionIntent.model_validate(raw).payload["role"] for raw in final["pending_actions"]
    )
    assert roles == [
        CalendarEventRole.INTERVIEW.value,
        CalendarEventRole.PREP.value,
        CalendarEventRole.PREP.value,
    ]


async def test_cancelling_an_interview_cancels_its_prep_blocks_too():
    world = cf.build_world(cf.interview_with_prep(cf.utc(2026, 11, 26, 19)))
    compiled, config = world.graph.build(), cf.thread()

    parked = await compiled.ainvoke(cf.start("cancel my Acme interview"), config)

    kinds = {r["kind"] for r in cf.interrupt_requests(parked)}
    assert kinds == {ActionKind.CANCEL_CALENDAR_EVENT.value}
    assert len(cf.interrupt_requests(parked)) == 3
    await compiled.ainvoke(Command(resume=cf.approve_all(parked)), config)
    assert sorted(world.calendar.cancelled) == ["int-1", "prep-deep", "prep-warm"]


async def test_an_application_belonging_to_someone_else_is_refused():
    world = cf.build_world()

    final = await world.graph.build().ainvoke(
        cf.start(f"schedule an interview for {cf.OTHER_USERS_APPLICATION} Friday at 2pm"),
        cf.thread(),
    )

    assert plan_of(final).outcome is CalendarOutcome.APPLICATION_NOT_FOUND
    assert final["pending_actions"] == []


# --- Scoping ------------------------------------------------------------------------------


_WRITES = {
    "calendar_create_event",
    "calendar_update_event",
    "calendar_delete_event",
    "gmail_send_message",
}


def _intent(tool: str, requested_by: str, **arguments) -> ToolIntent:
    return ToolIntent(
        server="google",
        tool=tool,
        arguments=arguments,
        origin=IntentOrigin.SYSTEM,
        mutating=tool in _WRITES,
        requested_by=requested_by,
    )


@pytest.mark.parametrize(
    ("requested_by", "tool"),
    [
        (cal.requester(cal.READ_CALENDAR), "calendar_create_event"),
        (cal.requester(cal.READ_CALENDAR), "calendar_delete_event"),
        (cal.requester(cal.READ_CALENDAR), "gmail_search"),
        (cal.requester(cal.PLAN_CALENDAR_CHANGE), "gmail_send_message"),
        (cal.requester(cal.CHECK_AVAILABILITY), "calendar_freebusy"),
    ],
)
def test_calendar_identities_reach_only_their_own_tools(google, requested_by, tool):
    decision = google.google.policy.evaluate_intent(_intent(tool, requested_by))
    assert decision.decision is Decision.DENY


def test_the_calendar_reader_may_read_without_approval(google):
    decision = google.google.policy.evaluate_intent(
        _intent(
            "calendar_freebusy",
            cal.requester(cal.READ_CALENDAR),
            calendar_ids=["primary"],
            time_min="2026-11-26T00:00:00+00:00",
            time_max="2026-11-27T00:00:00+00:00",
        )
    )
    assert decision.decision is Decision.ALLOW


def test_a_delete_from_the_planner_is_held_for_approval(google):
    decision = google.google.policy.evaluate_intent(
        ToolIntent(
            server="google",
            tool="calendar_delete_event",
            arguments={"calendar_id": "primary", "event_id": "evt1", "idempotency_key": "k-1"},
            origin=IntentOrigin.SYSTEM,
            mutating=True,
            requested_by=cal.requester(cal.PLAN_CALENDAR_CHANGE),
        )
    )
    assert decision.decision is Decision.REQUIRE_APPROVAL


# --- Routing ------------------------------------------------------------------------------


class RecordingSubgraph:
    def __init__(self):
        self.calls: list[Any] = []

    async def __call__(self, task_dag, state):
        self.calls.append(task_dag)
        return {}


async def test_a_calendar_question_routes_to_the_calendar_subgraph(google):
    google.add_event("dentist", "Dentist", cf.utc(2026, 11, 26, 15), cf.utc(2026, 11, 26, 16))
    job = RecordingSubgraph()
    supervisor = SupervisorGraph(
        KeywordIntentClassifier(),
        job,
        calendar_subgraph=CalendarSubgraphRunner(
            google.graph.build(), user_id=cf.USER_ID, thread_id=f"calendar-{cf.USER_ID}"
        ),
    ).build()

    final = await supervisor.ainvoke({"message": "When am I free Thursday?"}, cf.thread("sup"))

    assert final["route_decision"]["domain"] == RouteDomain.CALENDAR.value
    assert job.calls == []
    result = final["result"]
    assert result["domain"] == "calendar" and result["task"] == "availability"
    assert result["time_zone"] == cf.ZONE
    assert len(result["availability"]) == 2
    assert result["awaiting_approval"] is False
