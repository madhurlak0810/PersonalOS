"""A Calendar subgraph over an in-memory calendar with fake free/busy.

`FakeCalendar` (from `job_search_fakes`) is the reader the graph plans against
*and* the client the calendar executor writes through, over one set of
events, so a plan made after a write sees that write. Its `free_busy` is
computed from its busy events plus `extra_busy` -- bookings on another
calendar that the event listing does not show.

The clock is pinned to `NOW`, a Monday morning in New York, and the graph's
zone is `ZONE`. Nothing here reads the machine's zone.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from personalos.domain.interview_scheduling import (
    PROP_APPLICATION_ID,
    PROP_IDEMPOTENCY_KEY,
    PROP_INTERVIEW_KEY,
    PROP_KEY,
    PROP_ROLE,
    CalendarEvent,
    CalendarEventRole,
    interview_event_key,
    prep_block_key,
)
from personalos.domain.job_search import ApprovalDecision, ApprovalVerdict
from personalos.domain.models import ApplicationStatus
from personalos.domain.recruiter_events import ApplicationCandidate
from personalos.executor.calendar import CalendarActionExecutor
from personalos.graphs.calendar import CalendarGraph
from tests.fixtures import job_search_fakes as fakes

ZONE = "America/New_York"
#: Monday 23 November 2026, 09:00 EST (UTC-5).
NOW = datetime(2026, 11, 23, 14, 0)
USER_ID = UUID("00000000-0000-0000-0000-0000000000c1")
APPLICATION_ID = UUID("00000000-0000-0000-0000-0000000000a1")
OTHER_USERS_APPLICATION = UUID("00000000-0000-0000-0000-0000000000a2")


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute)


def application(application_id: UUID = APPLICATION_ID, company: str = "Acme"):
    return ApplicationCandidate(
        application_id=application_id,
        company=company,
        title="Backend Engineer",
        status=ApplicationStatus.RESPONSE,
    )


def external(event_id: str, title: str, starts_at: datetime, ends_at: datetime) -> CalendarEvent:
    """An event someone else put on the calendar: no PersonalOS stamp."""
    return CalendarEvent(event_id=event_id, title=title, starts_at=starts_at, ends_at=ends_at)


def managed(
    event_id: str,
    title: str,
    starts_at: datetime,
    ends_at: datetime,
    *,
    key: str | None = None,
    role: CalendarEventRole = CalendarEventRole.EVENT,
    application_id: UUID | None = None,
    interview_key: str | None = None,
    attendees: tuple[str, ...] = (),
) -> CalendarEvent:
    """An event PersonalOS created: it carries the idempotency stamp."""
    properties = {
        PROP_IDEMPOTENCY_KEY: f"cal-create:{key or event_id}",
        PROP_KEY: key or f"event:{event_id}",
        PROP_ROLE: role.value,
    }
    if application_id is not None:
        properties[PROP_APPLICATION_ID] = str(application_id)
    if interview_key is not None:
        properties[PROP_INTERVIEW_KEY] = interview_key
    return CalendarEvent(
        event_id=event_id,
        title=title,
        starts_at=starts_at,
        ends_at=ends_at,
        properties=properties,
        etag="1",
        attendees=attendees,
    )


def interview_with_prep(
    starts_at: datetime, application_id: UUID = APPLICATION_ID
) -> list[CalendarEvent]:
    """A stamped interview and its two prep blocks, as the Phase E planner would place them."""
    key = interview_event_key(application_id)
    return [
        managed(
            "int-1",
            "Interview: Acme",
            starts_at,
            starts_at + timedelta(hours=1),
            key=key,
            role=CalendarEventRole.INTERVIEW,
            application_id=application_id,
            attendees=("recruiter@acme.example",),
        ),
        managed(
            "prep-deep",
            "Interview prep",
            starts_at - timedelta(hours=20),
            starts_at - timedelta(hours=18, minutes=30),
            key=prep_block_key(application_id, "deep_prep"),
            role=CalendarEventRole.PREP,
            application_id=application_id,
            interview_key=key,
        ),
        managed(
            "prep-warm",
            "Interview warm-up",
            starts_at - timedelta(minutes=45),
            starts_at - timedelta(minutes=15),
            key=prep_block_key(application_id, "warm_up"),
            role=CalendarEventRole.PREP,
            application_id=application_id,
            interview_key=key,
        ),
    ]


@dataclass
class CalendarWorld:
    calendar: fakes.FakeCalendar
    graph: CalendarGraph
    directory: fakes.FakeApplicationDirectory


def build_world(
    events: list[CalendarEvent] | None = None,
    *,
    action_executor: Any = None,
    candidates: list[ApplicationCandidate] | None = None,
    clock_at: datetime = NOW,
    time_zone: str = ZONE,
) -> CalendarWorld:
    calendar = fakes.FakeCalendar(events or [])
    directory = fakes.FakeApplicationDirectory(
        candidates if candidates is not None else [application()]
    )
    graph = CalendarGraph(
        calendar,
        time_zone=time_zone,
        action_executor=action_executor or CalendarActionExecutor(calendar),
        application_directory=directory,
        clock=lambda: clock_at,
    )
    return CalendarWorld(calendar=calendar, graph=graph, directory=directory)


def thread(name: str = "calendar") -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{name}-{uuid4()}"}}


def start(request: str, **state: Any) -> dict[str, Any]:
    return {"request": request, "user_id": str(USER_ID), **state}


def approve(request: dict[str, Any], *, decided_by: str = "sam") -> ApprovalDecision:
    """A person's approval of one `ApprovalRequest` in an interrupt payload."""
    return ApprovalDecision(
        action_id=request["action_id"],
        action_fingerprint=request["action_hash"],
        verdict=ApprovalVerdict.APPROVED,
        decided_by=decided_by,
        decided_at=NOW,
        request_id=request["request_id"],
    )


def approve_all(final: dict[str, Any]) -> list[dict[str, Any]]:
    return [approve(r).model_dump(mode="json") for r in interrupt_requests(final)]


def interrupt_requests(final: dict[str, Any]) -> list[dict[str, Any]]:
    (pending,) = final["__interrupt__"]
    return pending.value["requests"]


# --- The real Google tool path ------------------------------------------------------


@dataclass
class GoogleCalendarWorld:
    """The Calendar subgraph over the production path to Google.

    Policy gateway with the production per-node allowlists, the Google MCP
    server in-process, a fake Google API -- and the calendar executor under
    `ToolExecutor` with its ledger on SQLite, as the worker builds it.
    """

    google: Any
    reader: Any
    graph: CalendarGraph
    factory: Any

    def events(self) -> dict[str, dict[str, Any]]:
        return self.google.api.events.setdefault("primary", {})

    def inserts(self) -> list[Any]:
        return [
            c for c in self.google.api.calls if c.method == "POST" and c.url.endswith("/events")
        ]

    def writes(self) -> list[Any]:
        return [
            c
            for c in self.google.api.calls
            if c.method in ("POST", "PATCH", "DELETE") and "/events" in c.url
        ]

    def add_event(
        self,
        event_id: str,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        *,
        private: dict[str, str] | None = None,
    ) -> None:
        """Put an event on the fake Google calendar. Times are naive UTC."""
        body: dict[str, Any] = {
            "id": event_id,
            "etag": f'"{event_id}-1"',
            "summary": title,
            "start": {"dateTime": f"{starts_at.isoformat()}+00:00", "timeZone": ZONE},
            "end": {"dateTime": f"{ends_at.isoformat()}+00:00", "timeZone": ZONE},
        }
        if private:
            body["extendedProperties"] = {"private": dict(private)}
        self.google.api.add_event("primary", body)


def build_google_world(db_path, *, clock_at: datetime = NOW) -> GoogleCalendarWorld:
    from personalos.bootstrap import build_calendar_action_executor
    from personalos.executor.google_calendar import GoogleCalendar
    from personalos.graphs.calendar import READ_CALENDAR, requester
    from personalos.policy.rules import GOOGLE_NODE_TOOL_ALLOWLISTS
    from tests.fixtures.durable_workflow import session_factory
    from tests.fixtures.fake_google import GoogleHarness

    google = GoogleHarness("in_process", node_allowlists=GOOGLE_NODE_TOOL_ALLOWLISTS)
    google.manager.register_server(google.server)
    google.api.events.setdefault("primary", {})
    reader = GoogleCalendar(google.gateway, time_zone=ZONE, requested_by=requester(READ_CALENDAR))
    factory = session_factory(db_path)
    graph = CalendarGraph(
        reader,
        time_zone=ZONE,
        action_executor=build_calendar_action_executor(reader, factory),
        application_directory=fakes.FakeApplicationDirectory([application()]),
        clock=lambda: clock_at,
    )
    return GoogleCalendarWorld(google=google, reader=reader, graph=graph, factory=factory)
