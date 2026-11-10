"""The candidate's Google Calendar, over the Google MCP tools.

`GoogleCalendar` is the production implementation of two ports:

- the Job Search graph's `CalendarReader` (`events_between`,
  `events_for_application`), plus free/busy and availability windows; and
- the executor's `CalendarClient` (`find_by_property`, `get_event`,
  `create_event`, `update_event`, `cancel_event`).

It holds no token and no HTTP client. Every call is a `google.calendar_*`
`ToolIntent` dispatched through the tool gateway, so policy records a decision
for each one and the Google MCP server brokers a token scoped to that one
call. Reads are `READ_EXTERNAL` and need no approval. Writes are
`WRITE_EXTERNAL` (create, update) or `DESTRUCTIVE` (delete): each is made
under the `WriteAuthority` the calendar executor hands over -- the approved
`ActionIntent` and its `ApprovalDecision` -- restated as the `ApprovalGrant`
the gateway checks, with the action's idempotency key as the MCP server's.
There is no path through this class to an unapproved write.

**Time.** This system keeps times as naive UTC. Every instant sent to Google
is made explicitly UTC-aware first, every instant read back is converted to
UTC from the offset Google gave it, and every write names an IANA zone
(`time_zone`) for the event to be stored in. Nothing here reads the server's
local zone: an instant without an offset from Google is a malformed response,
not something to guess about, and an all-day date is resolved in the event's
own zone (or the configured one).

**Failures.** A failed dispatch raises the shared taxonomy error
(`ToolResult.unwrap`), so `ToolExecutor` sees a retryable failure as one and
applies the calendar retry policy -- one retry, after reconciliation. A
free/busy answer with an error for a calendar raises too: an unknown calendar
is not a free one.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from personalos.domain.errors import ValidationFailed
from personalos.domain.events import EventEnvelope, EventType
from personalos.domain.interview_scheduling import (
    PROP_APPLICATION_ID,
    PROP_ROLE,
    AvailabilityWindow,
    BusyInterval,
    CalendarEvent,
    CalendarEventRole,
    WorkingHours,
    availability_windows,
    calendar_change_dedupe_key,
    changed_interviews,
)
from personalos.executor.calendar import WriteAuthority
from personalos.policy import ApprovalGrant, IntentOrigin, PolicyViolation, ToolIntent
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

GOOGLE_SERVER = "google"

#: The node every calendar read is made for. It is the one node the policy's
#: node allowlist lets reach the calendar tools.
DEFAULT_REQUESTED_BY = "graph:job_search#plan_interview_schedule"

#: Bound on pages followed for one listing (at 100 events a page).
MAX_LIST_PAGES = 20
PAGE_SIZE = 100


def _zone(name: str) -> ZoneInfo:
    """An IANA Area/Location zone (or UTC). Legacy aliases such as `EST` are refused."""
    if not name or ("/" not in name and name != "UTC"):
        raise ValueError(f"'{name}' is not an IANA Area/Location time zone")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"'{name}' is not an IANA time zone") from exc


def to_utc_naive(value: datetime) -> datetime:
    """An instant as this system stores it: naive UTC.

    An aware value is converted from its own offset. A naive one is refused:
    a bare wall-clock time could be anywhere, and this module never assumes
    the server's local zone.
    """
    if value.tzinfo is None:
        raise ValidationFailed("refusing a time with no UTC offset; it could be in any zone")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def as_utc(value: datetime) -> datetime:
    """A system time (naive UTC, or already aware) as an explicit UTC instant."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _instant(raw: str, *, all_day: bool, zone: ZoneInfo) -> datetime:
    if all_day:
        day = date.fromisoformat(raw)
        return to_utc_naive(datetime.combine(day, time.min, tzinfo=zone))
    return to_utc_naive(datetime.fromisoformat(raw))


class GoogleCalendar:
    """`CalendarReader` and `CalendarClient` over the `google.calendar_*` tools."""

    def __init__(
        self,
        gateway: ToolGateway,
        *,
        time_zone: str,
        calendar_id: str = "primary",
        requested_by: str = DEFAULT_REQUESTED_BY,
    ):
        """Take the gateway, and the IANA zone events are stored and read in.

        `time_zone` is required, not defaulted, so a deployment states it: it
        decides which day an all-day event falls on and how a written event
        behaves across a DST change.
        """
        if gateway is None:
            raise ValueError("GoogleCalendar requires a ToolGateway")
        self.gateway = gateway
        self.zone = _zone(time_zone)
        self.time_zone = time_zone
        self.calendar_id = calendar_id
        self.requested_by = requested_by

    # --- dispatch -------------------------------------------------------

    async def _read(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=False,
            requested_by=self.requested_by,
        )
        result = await self.gateway.dispatch(intent)
        return result.unwrap() or {}

    async def _write(
        self, tool: str, arguments: dict[str, Any], authority: WriteAuthority
    ) -> dict[str, Any]:
        if authority is None or authority.decision is None:
            raise PolicyViolation(f"google.{tool} needs the approval it is made under; none given")
        action, decision = authority.intent, authority.decision
        if not decision.authorizes(action):
            raise PolicyViolation(
                f"google.{tool} for action {action.action_id}: the approval does not "
                f"authorize this action"
            )
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool=tool,
            arguments={**arguments, "idempotency_key": action.idempotency_key},
            origin=IntentOrigin.SYSTEM,
            mutating=True,
            requested_by=action.requested_by or self.requested_by,
        )
        grant = ApprovalGrant(
            intent_id=intent.intent_id,
            intent_fingerprint=intent.fingerprint(),
            approved_by=decision.decided_by,
            note=f"approval of action {action.action_id} ({action.fingerprint()})",
        )
        result = await self.gateway.dispatch(intent, grant)
        return result.unwrap() or {}

    # --- shaping --------------------------------------------------------

    def _event(self, raw: dict[str, Any]) -> CalendarEvent | None:
        """A tool result's event as this system's `CalendarEvent`; None if it has no times."""
        start, end = raw.get("start"), raw.get("end")
        if not start or not end:
            return None
        zone_name = raw.get("time_zone")
        zone = self.zone
        if zone_name:
            try:
                zone = ZoneInfo(zone_name)
            except (ZoneInfoNotFoundError, ValueError):
                zone_name = None
        all_day = bool(raw.get("all_day"))
        try:
            starts_at = _instant(start, all_day=all_day, zone=zone)
            ends_at = _instant(end, all_day=all_day, zone=zone)
        except ValueError as exc:
            raise ValidationFailed(
                f"calendar event {raw.get('event_id')} has unreadable times"
            ) from exc
        return CalendarEvent(
            event_id=str(raw["event_id"]),
            title=str(raw.get("summary") or ""),
            starts_at=starts_at,
            ends_at=ends_at,
            properties=dict(raw.get("private_properties") or {}),
            busy=bool(raw.get("busy", True)),
            etag=raw.get("etag"),
            time_zone=zone_name or None,
        )

    async def _list(self, **filters: Any) -> list[CalendarEvent]:
        events: list[CalendarEvent] = []
        page_token: str | None = None
        for _ in range(MAX_LIST_PAGES):
            arguments: dict[str, Any] = {
                "calendar_id": self.calendar_id,
                "max_results": PAGE_SIZE,
                "time_zone": self.time_zone,
                **filters,
            }
            if page_token:
                arguments["page_token"] = page_token
            listing = await self._read("calendar_list_events", arguments)
            for raw in listing.get("events") or []:
                if raw.get("status") == "cancelled":
                    continue
                event = self._event(raw)
                if event is not None:
                    events.append(event)
            page_token = listing.get("next_page_token")
            if not page_token:
                return events
        logger.warning(
            "calendar listing %s stopped after %s pages; later events were not read",
            filters,
            MAX_LIST_PAGES,
        )
        return events

    # --- CalendarReader -------------------------------------------------

    async def events_between(
        self, user_id: UUID, start: datetime, end: datetime
    ) -> Sequence[CalendarEvent]:
        """Every event overlapping the window."""
        return await self._list(
            time_min=as_utc(start).isoformat(), time_max=as_utc(end).isoformat()
        )

    async def events_for_application(
        self, user_id: UUID, application_id: UUID
    ) -> Sequence[CalendarEvent]:
        """Every event stamped with the application, wherever it sits."""
        return await self.find_by_property(PROP_APPLICATION_ID, str(application_id))

    async def free_busy(
        self, start: datetime, end: datetime, *, calendar_ids: Sequence[str] | None = None
    ) -> tuple[BusyInterval, ...]:
        """Busy intervals across `calendar_ids` (default: this calendar). Naive UTC."""
        ids = list(calendar_ids or [self.calendar_id])
        answer = await self._read(
            "calendar_freebusy",
            {
                "calendar_ids": ids,
                "time_min": as_utc(start).isoformat(),
                "time_max": as_utc(end).isoformat(),
                "time_zone": self.time_zone,
            },
        )
        busy: list[BusyInterval] = []
        for calendar_id in ids:
            calendar = (answer.get("calendars") or {}).get(calendar_id)
            if calendar is None or calendar.get("errors"):
                reasons = ", ".join((calendar or {}).get("errors") or ["missing from the answer"])
                raise ValidationFailed(
                    f"free/busy could not be read for calendar '{calendar_id}': {reasons}"
                )
            for interval in calendar.get("busy") or []:
                busy.append(
                    BusyInterval(
                        starts_at=to_utc_naive(datetime.fromisoformat(interval["start"])),
                        ends_at=to_utc_naive(datetime.fromisoformat(interval["end"])),
                    )
                )
        return tuple(sorted(busy, key=lambda b: b.starts_at))

    async def availability(
        self,
        start: datetime,
        end: datetime,
        *,
        min_duration: timedelta = timedelta(minutes=30),
        working_hours: WorkingHours | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> tuple[AvailabilityWindow, ...]:
        """Free windows between `start` and `end`, from free/busy."""
        busy = await self.free_busy(start, end, calendar_ids=calendar_ids)
        return availability_windows(
            busy,
            start=to_utc_naive(as_utc(start)),
            end=to_utc_naive(as_utc(end)),
            min_duration=min_duration,
            working_hours=working_hours,
        )

    # --- CalendarClient -------------------------------------------------

    async def find_by_property(self, name: str, value: str) -> Sequence[CalendarEvent]:
        """Events whose private extended property `name` equals `value`."""
        return await self._list(private_properties={name: value})

    async def get_event(self, event_id: str) -> CalendarEvent | None:
        """One event by id, or `None` if the calendar has no such event."""
        answer = await self._read(
            "calendar_get_event",
            {"calendar_id": self.calendar_id, "event_id": event_id, "time_zone": self.time_zone},
        )
        if not answer.get("found") or not answer.get("event"):
            return None
        return self._event(answer["event"])

    async def create_event(
        self,
        *,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        description: str,
        properties: dict[str, str],
        authority: WriteAuthority,
        time_zone: str | None = None,
    ) -> CalendarEvent:
        """Insert an event, stamped with `properties`, and return it as stored."""
        zone = time_zone or self.time_zone
        _zone(zone)
        answer = await self._write(
            "calendar_create_event",
            {
                "calendar_id": self.calendar_id,
                "summary": title[:998] or "Event",
                "start": as_utc(starts_at).isoformat(),
                "end": as_utc(ends_at).isoformat(),
                "description": description[:8000],
                "private_properties": dict(properties),
                "time_zone": zone,
            },
            authority,
        )
        return self._written(answer)

    async def update_event(
        self,
        event_id: str,
        *,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        properties: dict[str, str],
        authority: WriteAuthority,
        time_zone: str | None = None,
    ) -> CalendarEvent:
        """Move an event, merging `properties` into its own."""
        zone = time_zone or self.time_zone
        _zone(zone)
        arguments: dict[str, Any] = {
            "calendar_id": self.calendar_id,
            "event_id": event_id,
            "start": as_utc(starts_at).isoformat(),
            "end": as_utc(ends_at).isoformat(),
            "private_properties": dict(properties),
            "time_zone": zone,
        }
        if title:
            arguments["summary"] = title[:998]
        answer = await self._write("calendar_update_event", arguments, authority)
        return self._written(answer)

    async def cancel_event(self, event_id: str, *, authority: WriteAuthority) -> bool:
        """Delete an event. False when Google already had it deleted."""
        answer = await self._write(
            "calendar_delete_event",
            {"calendar_id": self.calendar_id, "event_id": event_id},
            authority,
        )
        return not answer.get("already_deleted", False)

    def _written(self, answer: dict[str, Any]) -> CalendarEvent:
        event = self._event(answer.get("event") or {})
        if event is None:
            raise ValidationFailed("the calendar write returned an event without times")
        return event


#: Publishes an event to the outbox; returns whatever the outbox returns.
Publish = Callable[[EventEnvelope], Any]


class CalendarChangeScanner:
    """Finds interview events moved outside this system and publishes `calendar.event_changed`.

    Detection is `changed_interviews`: an interview this system stamped whose
    times no longer match the stamps its last approved write left, or whose
    prep blocks were planned around a different time. Only stamped events are
    read, so the candidate's own events are never reported.

    Each change is published once per observed version of the event (its
    etag), keyed in the outbox by `calendar_change_dedupe_key`, so a sweep
    that sees the same move again publishes nothing new.
    """

    def __init__(
        self,
        calendar: GoogleCalendar | Any,
        publish: Publish,
        *,
        source: str = "google_calendar",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    ):
        """Take the calendar to read and where to publish what changed."""
        self.calendar = calendar
        self.publish = publish
        self.source = source
        self.clock = clock

    async def scan(self, user_id: UUID) -> list[EventEnvelope]:
        """Publish one `calendar.event_changed` per moved interview; return them."""
        now = self.clock()
        interviews = await self.calendar.find_by_property(
            PROP_ROLE, CalendarEventRole.INTERVIEW.value
        )
        preps = await self.calendar.find_by_property(PROP_ROLE, CalendarEventRole.PREP.value)
        envelopes: list[EventEnvelope] = []
        for event in changed_interviews([*interviews, *preps], now=now):
            envelope = calendar_event_changed_envelope(
                user_id, event, source=self.source, observed_at=now
            )
            self.publish(envelope)
            envelopes.append(envelope)
        if envelopes:
            logger.info(
                "calendar scan for user %s found %s moved interview(s)", user_id, len(envelopes)
            )
        return envelopes


def calendar_event_changed_envelope(
    user_id: UUID,
    event: CalendarEvent,
    *,
    source: str = "google_calendar",
    observed_at: datetime | None = None,
) -> EventEnvelope:
    """The `calendar.event_changed` envelope for one moved interview event."""
    observed = observed_at or datetime.now(timezone.utc).replace(tzinfo=None)
    return EventEnvelope(
        type=EventType.CALENDAR_EVENT_CHANGED,
        source=source,
        occurred_at=observed,
        received_at=observed,
        subject_ref=f"application:{event.application_id}" if event.application_id else None,
        dedupe_key=calendar_change_dedupe_key(event),
        payload={"user_id": str(user_id), "event": event.model_dump(mode="json")},
        raw_ref=f"gcal:{event.event_id}",
    )


__all__ = [
    "DEFAULT_REQUESTED_BY",
    "GoogleCalendar",
    "CalendarChangeScanner",
    "calendar_event_changed_envelope",
    "to_utc_naive",
    "as_utc",
]
