"""Interview scheduling through the compiled graph: invite, approval, calendar.

The acceptance criterion this file exists for:

**A changed interview time updates or reschedules the linked prep blocks.**
`test_a_changed_interview_time_moves_the_interview_and_its_prep_blocks` takes
an invite through approval so the interview and its two prep blocks are on the
calendar, then delivers a second message moving the interview. The run that
follows proposes *updates* to those same three events; once approved, the
calendar still holds exactly three, the prep blocks sit before the new time,
and the reminder for the old time has been cancelled.

Around it: nothing reaches the calendar before a reviewer says so, a rejected
write is not made, and the same holds when the interview is moved on the
calendar rather than by a message.

The calendar is one `FakeCalendar`, read by the graph's planner and written by
the real `CalendarActionExecutor`, so a plan made after a write sees that
write.
"""

from datetime import datetime, timedelta

from langgraph.types import Command

from personalos.domain.interview_scheduling import (
    PROP_INTERVIEW_KEY,
    interview_event_key,
    prep_block_key,
)
from personalos.domain.job_search import (
    ActionKind,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    FollowUpKind,
    JobSearchEventType,
)
from personalos.domain.recruiter_events import CommitmentActor, ExtractedCommitment
from personalos.domain.workflow import recruiter_inbox_thread_id
from personalos.executor.calendar import CalendarActionExecutor
from personalos.graphs.job_search import APPROVAL_CHECKPOINT, JobSearchGraph
from tests.fixtures import job_search_fakes as fakes

APP = fakes.APPLICATION_ID
#: `fakes.NOW` is Monday 28 September 2026, 12:00.
FIRST_TIME = datetime(2026, 10, 1, 14, 0)
NEW_TIME = datetime(2026, 10, 6, 11, 0)
INTERVIEW = interview_event_key(APP)
DEEP = prep_block_key(APP, "deep_prep")
WARM = prep_block_key(APP, "warm_up")
CONFIG = {"configurable": {"thread_id": recruiter_inbox_thread_id(fakes.USER_ID)}}


def invite_at(starts_at: datetime):
    """An extraction of an invite that fixes the interview's time."""
    return fakes.extraction(
        commitments=[
            ExtractedCommitment(
                actor=CommitmentActor.EXTERNAL_PERSON,
                action="Technical interview with the team",
                due_at=starts_at,
                confidence=0.9,
            )
        ]
    )


class Deployment:
    """A compiled graph whose calendar reads and writes hit the same calendar."""

    def __init__(self):
        self.calendar = fakes.FakeCalendar()
        self.extractor = fakes.FakeRecruiterEventExtractor(
            by_message={"invite": invite_at(FIRST_TIME), "moved": invite_at(NEW_TIME)}
        )
        self.emitter = fakes.FakeEventEmitter()
        self.scheduler = fakes.FakePendingCheckpointScheduler()
        self.graph = JobSearchGraph(
            profile_store=fakes.FakeProfileStore(),
            providers=[fakes.FakeProvider()],
            scorer=fakes.FakeScorer(),
            evidence_checker=fakes.FakeEvidenceChecker(),
            packet_builder=fakes.FakePacketBuilder(),
            approval_gate=fakes.NoStandingApprovalGate(),
            action_executor=CalendarActionExecutor(self.calendar),
            application_store=fakes.FakeApplicationStore(),
            event_emitter=self.emitter,
            recruiter_event_extractor=self.extractor,
            application_directory=fakes.FakeApplicationDirectory(),
            recruiter_event_recorder=fakes.FakeRecruiterEventRecorder(),
            calendar_reader=self.calendar,
            checkpoint_scheduler=self.scheduler,
            clock=lambda: fakes.NOW,
        ).build()

    async def deliver(self, message_id: str) -> list[ApprovalRequest]:
        """Run one inbound message; return the requests the run parked on."""
        await self.graph.ainvoke(
            {
                "user_id": str(fakes.USER_ID),
                "inbound_messages": [fakes.inbound_message(message_id).model_dump(mode="json")],
            },
            CONFIG,
        )
        return await self.outstanding()

    async def outstanding(self) -> list[ApprovalRequest]:
        state = await self.graph.aget_state(CONFIG)
        if state.next != (APPROVAL_CHECKPOINT,):
            return []
        pending = {raw["action_id"] for raw in state.values["pending_actions"]}
        return [
            ApprovalRequest.model_validate(raw)
            for raw in state.values["approval_requests"]
            if raw["action_id"] in pending
        ]

    async def answer(self, requests, verdict=ApprovalVerdict.APPROVED) -> dict:
        decisions = [
            ApprovalDecision(
                action_id=request.action_id,
                action_fingerprint=request.action_hash,
                verdict=verdict,
                decided_by="reviewer@example.test",
                request_id=request.request_id,
            ).model_dump(mode="json")
            for request in requests
        ]
        return await self.graph.ainvoke(Command(resume=decisions), CONFIG)

    def times(self) -> dict[str, tuple[datetime, datetime]]:
        """Every event this system put on the calendar, by logical key."""
        return {
            event.logical_key: (event.starts_at, event.ends_at)
            for event in self.calendar.events.values()
            if event.logical_key
        }


async def scheduled() -> Deployment:
    """A deployment whose first invite has been approved onto the calendar."""
    deployment = Deployment()
    await deployment.answer(await deployment.deliver("invite"))
    assert set(deployment.times()) == {INTERVIEW, DEEP, WARM}
    return deployment


# --- An invite -----------------------------------------------------------------


async def test_an_invite_proposes_the_interview_and_prep_blocks_and_waits_for_approval():
    """Creating calendar events is an external write. Nothing happens until approved."""
    deployment = Deployment()
    # Something already in the diary where the long prep block would go.
    dinner = deployment.calendar.add(
        "Dinner", datetime(2026, 9, 30, 18, 0), datetime(2026, 9, 30, 20, 0)
    )

    requests = await deployment.deliver("invite")

    assert [r.kind for r in requests] == [ActionKind.CREATE_CALENDAR_EVENT] * 3
    assert all(r.requested_scopes == ("calendar:write",) for r in requests)
    # Parked: the calendar holds only what was already in it.
    assert list(deployment.calendar.events) == [dinner.event_id]
    # But the proposal was announced, and the reminder exists, regardless.
    assert JobSearchEventType.INTERVIEW_SCHEDULE_PROPOSED.value in deployment.emitter.types()
    (reminder,) = deployment.scheduler.waits()
    assert reminder.kind == FollowUpKind.INTERVIEW_PREP

    await deployment.answer(requests)

    times = deployment.times()
    assert times[INTERVIEW] == (FIRST_TIME, FIRST_TIME + timedelta(hours=1))
    # The prep block went around the dinner rather than on top of it.
    assert times[DEEP] == (datetime(2026, 9, 30, 16, 30), dinner.starts_at)
    assert times[WARM][1] == FIRST_TIME - timedelta(minutes=15)
    # And each prep block names the interview it is for.
    for key in (DEEP, WARM):
        assert deployment.calendar.by_key(key).properties[PROP_INTERVIEW_KEY] == INTERVIEW
    assert await deployment.outstanding() == []


async def test_a_rejected_proposal_writes_nothing():
    deployment = Deployment()

    await deployment.answer(await deployment.deliver("invite"), ApprovalVerdict.REJECTED)

    assert deployment.calendar.events == {}


async def test_the_same_invite_delivered_twice_proposes_nothing_new():
    deployment = await scheduled()

    assert await deployment.deliver("invite") == []
    assert len(deployment.calendar.events) == 3


# --- Acceptance: a changed interview time -------------------------------------


async def test_a_changed_interview_time_moves_the_interview_and_its_prep_blocks():
    """The recruiter moves the interview; its prep blocks move with it.

    Updates, not creates: the three events are found on the calendar by the
    properties they were stamped with, so the plan for the new time is a plan
    to move them. Nothing is duplicated and nothing is left behind at the old
    time.
    """
    deployment = await scheduled()
    before = {key: deployment.calendar.by_key(key).event_id for key in (INTERVIEW, DEEP, WARM)}
    old_times = deployment.times()
    (old_reminder,) = deployment.scheduler.waits()

    requests = await deployment.deliver("moved")

    assert [r.kind for r in requests] == [ActionKind.UPDATE_CALENDAR_EVENT] * 3
    # Still an external write: the calendar has not moved yet.
    assert deployment.times() == old_times

    await deployment.answer(requests)

    times = deployment.times()
    # The same three events, each somewhere new.
    assert len(deployment.calendar.events) == 3
    assert {
        key: deployment.calendar.by_key(key).event_id for key in (INTERVIEW, DEEP, WARM)
    } == before
    assert times[INTERVIEW] == (NEW_TIME, NEW_TIME + timedelta(hours=1))
    assert times[DEEP] == (datetime(2026, 10, 5, 18, 30), datetime(2026, 10, 5, 20, 0))
    assert times[WARM] == (NEW_TIME - timedelta(minutes=45), NEW_TIME - timedelta(minutes=15))
    for key in (INTERVIEW, DEEP, WARM):
        assert times[key] != old_times[key]

    # The reminder followed the interview, and the old one will not fire.
    (reminder,) = deployment.scheduler.waits()
    assert reminder.dedupe_key != old_reminder.dedupe_key
    assert reminder.trigger_at == times[DEEP][0]
    assert reminder.expires_at == NEW_TIME
    assert list(deployment.scheduler.cancelled) == [old_reminder.dedupe_key]
    assert (
        deployment.emitter.types().count(JobSearchEventType.INTERVIEW_SCHEDULE_PROPOSED.value) == 2
    )


async def test_an_interview_moved_on_the_calendar_reschedules_its_prep_blocks():
    """The other way the time changes: the candidate drags the event."""
    deployment = await scheduled()
    interview = deployment.calendar.by_key(INTERVIEW)
    moved = await deployment.calendar.update_event(
        interview.event_id,
        title=interview.title,
        starts_at=NEW_TIME,
        ends_at=NEW_TIME + timedelta(hours=1),
        properties={},
    )

    await deployment.graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "calendar_changes": [moved.model_dump(mode="json")]},
        CONFIG,
    )
    requests = await deployment.outstanding()

    # The interview is already where it is; only the prep blocks need moving.
    assert [r.kind for r in requests] == [ActionKind.UPDATE_CALENDAR_EVENT] * 2
    await deployment.answer(requests)

    times = deployment.times()
    assert len(deployment.calendar.events) == 3
    assert times[DEEP][1] <= NEW_TIME - timedelta(hours=12)
    assert times[WARM][1] == NEW_TIME - timedelta(minutes=15)
    # Consumed: the input channel will not re-route this thread's next run.
    assert (await deployment.graph.aget_state(CONFIG)).values["calendar_changes"] == []


# --- Reminders -----------------------------------------------------------------


async def test_a_due_reminder_is_emitted_as_an_event_and_proposes_nothing():
    deployment = await scheduled()
    (reminder,) = deployment.scheduler.waits()

    final = await deployment.graph.ainvoke(
        {"fired_checkpoints": [reminder.model_dump(mode="json")]}, CONFIG
    )

    assert deployment.emitter.types()[-1] == JobSearchEventType.INTERVIEW_REMINDER.value
    event = deployment.emitter.events[-1]
    assert event.aggregate_id == APP
    assert event.payload["interview_starts_at"] == FIRST_TIME.isoformat()
    assert len(event.payload["prep_blocks"]) == 2
    assert final["pending_actions"] == []
    assert await deployment.outstanding() == []
