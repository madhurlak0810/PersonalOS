"""The Calendar subgraph's values: requests, resolved times, availability, typed outcomes.

Three things live here, all pure -- no clock, no calendar, no zone but the one
passed in.

**Time, explicitly zoned.** A person says "next Tuesday afternoon"; the
calendar stores instants. `resolve_when` turns the phrase into absolute times
using the person's configured IANA zone and an explicit `now`, and says how
it read the phrase (`ResolvedTime.notes`): which Tuesday, which offset, and
whether a wall-clock time fell into a DST gap (it does not exist, and is
moved forward) or a DST overlap (it happens twice, and the first is taken).
Every write's approval payload carries the resolved local and UTC times and
those notes, so an ambiguity is on the reviewer's screen before anything is
written. The server's local zone is never consulted.

**Availability from free/busy.** `local_free_windows` cuts free/busy gaps to
the person's working day *in their zone*, per local date, so a DST change
inside the range moves the window with the clock instead of by an hour.
`propose_slots` turns windows into concrete offers.

**Typed outcomes.** A planned change is a `CalendarChangePlan` whose
`outcome` is a closed enum. A double-booking is `CalendarOutcome.CONFLICT`
with a `CalendarConflict` naming what is in the way and the alternatives --
never a write, and never prose a caller has to parse.

Writes are `ActionIntent`s built by `calendar_write_intent`. Their payload is
the approval preview (title, attendees, start and end with the IANA zone,
what changes for a move) and is hashed into the approval, so what the
reviewer read is what executes.
"""

import re
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict

from personalos.domain.errors import ValidationFailed
from personalos.domain.interview_scheduling import (
    PAYLOAD_ENDS_AT,
    PAYLOAD_EVENT_ID,
    PAYLOAD_INTERVIEW_STARTS_AT,
    PAYLOAD_LOGICAL_KEY,
    PAYLOAD_PROPERTIES,
    PAYLOAD_ROLE,
    PAYLOAD_STARTS_AT,
    PAYLOAD_TIME_ZONE,
    PAYLOAD_TITLE,
    PROP_KEY,
    PROP_ROLE,
    AvailabilityWindow,
    BusyInterval,
    CalendarEvent,
    CalendarEventRole,
    ScheduleConflict,
    availability_windows,
)
from personalos.domain.job_search import (
    PAYLOAD_BODY,
    PAYLOAD_PREVIOUS_BODY,
    PAYLOAD_RECIPIENT,
    PAYLOAD_SUBJECT,
    ActionIntent,
    ActionKind,
)


class CalendarContractError(ValidationFailed, ValueError):
    """A Calendar value broke one of its invariants."""


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Payload keys of the approval preview ---------------------------------------

#: Attendee addresses of the event a write creates (or already has, for a move).
PAYLOAD_ATTENDEES = "attendees"
#: The start and end in the person's zone, ISO 8601 with the offset.
PAYLOAD_LOCAL_STARTS_AT = "local_starts_at"
PAYLOAD_LOCAL_ENDS_AT = "local_ends_at"
#: Where the event is now, for a move: UTC and local.
PAYLOAD_PREVIOUS_STARTS_AT = "previous_starts_at"
PAYLOAD_PREVIOUS_ENDS_AT = "previous_ends_at"
PAYLOAD_PREVIOUS_LOCAL_STARTS_AT = "previous_local_starts_at"
PAYLOAD_PREVIOUS_LOCAL_ENDS_AT = "previous_local_ends_at"
#: For a move: one `{"field", "from", "to"}` per thing that changes.
PAYLOAD_CHANGES = "changes"
#: The phrase the time was read from, and how it was read.
PAYLOAD_RESOLVED_FROM = "resolved_from"
PAYLOAD_RESOLUTION_NOTES = "resolution_notes"
PAYLOAD_APPLICATION_ID = "application_id"

DEFAULT_EVENT_DURATION = timedelta(minutes=60)
DEFAULT_DAY_START = time(9, 0)
DEFAULT_DAY_END = time(17, 0)
#: How far ahead an unqualified availability question, or an event search, looks.
DEFAULT_LOOKAHEAD = timedelta(days=5)
EVENT_SEARCH_HORIZON = timedelta(days=60)
#: Bounds on one run's work.
MAX_PROPOSALS = 5
MAX_ALTERNATIVES = 3
MAX_WRITES_PER_RUN = 8
#: Proposed starts land on this grid, in local time.
SLOT_GRANULARITY = timedelta(minutes=30)


# --- Zones -------------------------------------------------------------------


def iana_zone(name: str) -> ZoneInfo:
    """An IANA Area/Location zone (or UTC). Abbreviations such as `EST` are refused."""
    if not name or ("/" not in name and name != "UTC"):
        raise CalendarContractError(f"'{name}' is not an IANA Area/Location time zone")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CalendarContractError(f"'{name}' is not an IANA time zone") from exc


def to_local(moment: datetime, zone: ZoneInfo) -> datetime:
    """A naive-UTC instant as wall-clock time in `zone` (aware)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(zone)


def _utc_naive(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def localize(wall: datetime, zone: ZoneInfo) -> tuple[datetime, str | None]:
    """A naive wall-clock time in `zone` as an aware local time, and any DST note.

    A time in a spring-forward gap does not exist; it is read with the offset
    in force before the gap, which lands it the same distance past the jump
    (02:30 becomes 03:30). A time in a fall-back overlap happens twice; the
    first occurrence is taken. Either way the note says so.
    """
    first = wall.replace(tzinfo=zone, fold=0)
    actual = first.astimezone(timezone.utc).astimezone(zone)
    if actual.replace(tzinfo=None) != wall:
        return actual, (
            f"{wall:%H:%M} on {wall:%a %d %b %Y} does not exist in {zone.key} "
            f"(clocks spring forward); read as {actual:%H:%M %Z}"
        )
    second = wall.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        return first, (
            f"{wall:%H:%M} on {wall:%a %d %b %Y} happens twice in {zone.key} "
            f"(clocks fall back); read as the first, {first:%H:%M %Z}"
        )
    return first, None


def format_local(start: datetime, end: datetime, zone: ZoneInfo) -> str:
    """A range as a reviewer reads it: local wall-clock, abbreviation, IANA zone, UTC."""
    lo, hi = to_local(start, zone), to_local(end, zone)
    same_day = lo.date() == hi.date()
    end_text = f"{hi:%H:%M %Z}" if same_day else f"{hi:%a %d %b %Y %H:%M %Z}"
    utc_lo, utc_hi = to_local(start, ZoneInfo("UTC")), to_local(end, ZoneInfo("UTC"))
    return (
        f"{lo:%a %d %b %Y %H:%M}-{end_text} ({zone.key}; "
        f"{utc_lo:%Y-%m-%d %H:%M}-{utc_hi:%H:%M} UTC)"
    )


# --- Resolved times ------------------------------------------------------------


class ResolvedTime(_Value):
    """A phrase read as absolute times in one zone.

    `exact` is true when the phrase named a start time ("at 2pm"); otherwise
    it named a window ("Thursday", "next Tuesday afternoon") and
    `starts_at`/`ends_at` bound it.
    """

    phrase: str
    time_zone: str
    exact: bool
    #: Wall-clock start and end in `time_zone`, with their offsets.
    local_start: datetime
    local_end: datetime
    #: The same instants as this system stores them: naive UTC.
    starts_at: datetime
    ends_at: datetime
    #: How the phrase was read: which date, which offset, DST adjustments.
    notes: tuple[str, ...] = ()

    @property
    def duration(self) -> timedelta:
        return self.ends_at - self.starts_at


class TimeSlot(_Value):
    """A stretch of time, in UTC and in the person's zone."""

    starts_at: datetime
    ends_at: datetime
    time_zone: str
    local_start: datetime
    local_end: datetime

    @classmethod
    def of(cls, start: datetime, end: datetime, zone: ZoneInfo) -> "TimeSlot":
        return cls(
            starts_at=start,
            ends_at=end,
            time_zone=zone.key,
            local_start=to_local(start, zone),
            local_end=to_local(end, zone),
        )

    @property
    def duration(self) -> timedelta:
        return self.ends_at - self.starts_at

    def describe(self) -> str:
        return format_local(self.starts_at, self.ends_at, iana_zone(self.time_zone))


_WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_WEEKDAY_RE = re.compile(
    r"\b(?:(next|this|coming)\s+)?(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b"
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY_RE = re.compile(r"\b" + _MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b")
_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_RE + r"(?:,?\s+(\d{4}))?\b")
_RELATIVE_DAY_RE = re.compile(r"\b(today|tonight|tomorrow)\b")
_WEEK_RE = re.compile(r"\b(this|next)\s+week\b")
_TIME_12H_RE = re.compile(r"\b(\d{1,2})(?::([0-5]\d))?\s*(am|pm|a\.m\.|p\.m\.)")
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_NOON_RE = re.compile(r"\b(noon|midday)\b")
_PART_OF_DAY = {
    "morning": (time(8, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), time(21, 0)),
    "tonight": (time(17, 0), time(21, 0)),
}
_PART_RE = re.compile(r"\b(morning|afternoon|evening|tonight)\b")
_DURATION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)[\s-]?(minutes?|mins?|hours?|hrs?|h)\b(?!\s*(?:ago|before|after))"
)


def parse_duration(text: str) -> timedelta | None:
    """'for 45 minutes', 'a 30-minute call', '1.5 hours'. `None` when none is named."""
    match = _DURATION_RE.search((text or "").lower())
    if match is None:
        return None
    amount, unit = float(match.group(1)), match.group(2)
    minutes = amount if unit.startswith("m") else amount * 60
    if minutes <= 0 or minutes > 24 * 60:
        return None
    return timedelta(minutes=round(minutes))


_DayReading = tuple[date | None, date | None, str | None]
_NO_DAY: _DayReading = (None, None, None)


def _calendar_date(text: str, today: date) -> _DayReading | None:
    """An explicit date: ISO, 'March 9', '9 March'. `None` when the text names none."""
    if match := _ISO_DATE_RE.search(text):
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return _NO_DAY
        return day, day, None
    for regex, month_group, day_group in ((_MONTH_DAY_RE, 1, 2), (_DAY_MONTH_RE, 2, 1)):
        if match := regex.search(text):
            month = _MONTHS[match.group(month_group)[:3]]
            year = int(match.group(3)) if match.group(3) else today.year
            try:
                day = date(year, month, int(match.group(day_group)))
            except ValueError:
                return _NO_DAY
            if match.group(3) or day >= today:
                return day, day, None
            day = date(year + 1, month, day.day)
            return day, day, f"no year given; read as the next {day:%d %B}, {day.year}"
    return None


def _relative_day(text: str, today: date) -> _DayReading:
    """'today', 'tomorrow', 'this/next week', '(next) Tuesday'."""
    if match := _RELATIVE_DAY_RE.search(text):
        day = today + timedelta(days=1 if match.group(1) == "tomorrow" else 0)
        return day, day, None
    if match := _WEEK_RE.search(text):
        monday = today - timedelta(days=today.weekday())
        if match.group(1) == "next":
            monday += timedelta(days=7)
            return monday, monday + timedelta(days=4), None
        return today, max(today, monday + timedelta(days=4)), None
    if match := _WEEKDAY_RE.search(text):
        qualifier, name = match.group(1), match.group(2)
        ahead = (_WEEKDAYS[name] - today.weekday()) % 7
        if qualifier == "next" and ahead == 0:
            # "next Thursday", said on a Thursday, is not today.
            ahead = 7
        day = today + timedelta(days=ahead)
        spoken = f"{qualifier} {name}" if qualifier else name
        reading = "today" if ahead == 0 else "the first one after today"
        return day, day, f"'{spoken}' read as {day:%a %d %b %Y} ({reading})"
    return _NO_DAY


def _day_from(text: str, today: date) -> _DayReading:
    """The day (or, for a week, the first and last day) a phrase names, and how it was read."""
    explicit = _calendar_date(text, today)
    return explicit if explicit is not None else _relative_day(text, today)


def _time_from(text: str) -> tuple[time | None, str | None]:
    """The clock time a phrase names, and a note when it had to guess."""
    if match := _TIME_12H_RE.search(text):
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12:
            return None, None
        pm = match.group(3).startswith("p")
        hour = (hour % 12) + (12 if pm else 0)
        return time(hour, minute), None
    if match := _TIME_24H_RE.search(text):
        return time(int(match.group(1)), int(match.group(2))), None
    if _NOON_RE.search(text):
        return time(12, 0), None
    return None, None


def resolve_when(
    text: str,
    *,
    now: datetime,
    zone: ZoneInfo,
    duration: timedelta | None = None,
) -> ResolvedTime | None:
    """Read a time phrase as absolute times in `zone`. `None` when it names no day or time.

    `now` is naive UTC. A phrase with a clock time but no day is today's (or
    tomorrow's, if that time has passed). A day with no time is the whole
    local day, or the part of it named ('afternoon'). 'next <weekday>' is the
    first such day strictly after today; a bare weekday may be today. The
    notes record each of those readings.
    """
    lowered = (text or "").lower()
    local_now = to_local(now, zone)
    today = local_now.date()
    first_day, last_day, day_note = _day_from(lowered, today)
    clock, clock_note = _time_from(lowered)
    part = _PART_RE.search(lowered)
    notes = [n for n in (day_note, clock_note) if n]

    if first_day is None and clock is None and part is None:
        return None

    if clock is not None:
        day = first_day
        if day is None:
            day = today
            if datetime.combine(today, clock) <= local_now.replace(tzinfo=None):
                day = today + timedelta(days=1)
                notes.append(f"no day given and {clock:%H:%M} has passed; read as tomorrow")
            else:
                notes.append("no day given; read as today")
        if last_day is not None and last_day != first_day:
            notes.append(f"a time was given for a range of days; read as {day:%a %d %b}")
        start_local, gap_note = localize(datetime.combine(day, clock), zone)
        length = duration or DEFAULT_EVENT_DURATION
        end_local = (start_local.astimezone(timezone.utc) + length).astimezone(zone)
        notes.extend(n for n in (gap_note,) if n)
        exact = True
    else:
        first = first_day or today
        last = last_day or first
        if part is not None:
            lo, hi = _PART_OF_DAY[part.group(1)]
        else:
            lo, hi = time(0, 0), None
        start_local, start_note = localize(datetime.combine(first, lo), zone)
        if hi is None:
            end_local, end_note = localize(datetime.combine(last + timedelta(days=1), lo), zone)
        else:
            end_local, end_note = localize(datetime.combine(last, hi), zone)
        notes.extend(n for n in (start_note, end_note) if n)
        exact = False

    notes.append(f"UTC offset {start_local:%z} ({start_local:%Z}) in {zone.key}")
    starts_at, ends_at = _utc_naive(start_local), _utc_naive(end_local)
    if not exact and starts_at < now < ends_at:
        # "today" and "this week" start now, not at midnight already gone.
        starts_at = now.replace(second=0, microsecond=0)
        start_local = to_local(starts_at, zone)
    return ResolvedTime(
        phrase=(text or "").strip()[:200],
        time_zone=zone.key,
        exact=exact,
        local_start=start_local,
        local_end=end_local,
        starts_at=starts_at,
        ends_at=ends_at,
        notes=tuple(notes),
    )


def at_local_time_of(
    resolved: ResolvedTime, event: CalendarEvent, zone: ZoneInfo
) -> ResolvedTime | None:
    """A move to a single day with no time keeps the event's local time and length.

    `None` when `resolved` spans more than one day: there is no single answer
    to "move it to next week".
    """
    first = to_local(resolved.starts_at, zone).date()
    last = to_local(resolved.ends_at - timedelta(microseconds=1), zone).date()
    if first != last:
        return None
    clock = to_local(event.starts_at, zone).time().replace(tzinfo=None)
    start_local, note = localize(datetime.combine(first, clock), zone)
    length = event.ends_at - event.starts_at
    end_local = (start_local.astimezone(timezone.utc) + length).astimezone(zone)
    notes = [
        *resolved.notes,
        f"no time given; kept the event's {clock:%H:%M} start and "
        f"{int(length.total_seconds() // 60)}-minute length",
    ]
    if note:
        notes.append(note)
    return resolved.model_copy(
        update={
            "exact": True,
            "local_start": start_local,
            "local_end": end_local,
            "starts_at": _utc_naive(start_local),
            "ends_at": _utc_naive(end_local),
            "notes": tuple(notes),
        }
    )


# --- Availability ----------------------------------------------------------------


def local_free_windows(
    busy: list[BusyInterval] | tuple[BusyInterval, ...],
    *,
    start: datetime,
    end: datetime,
    zone: ZoneInfo,
    day_start: time = DEFAULT_DAY_START,
    day_end: time = DEFAULT_DAY_END,
    min_duration: timedelta = timedelta(minutes=30),
) -> tuple[TimeSlot, ...]:
    """Free windows between `start` and `end` (naive UTC) inside the local working day.

    The working day is drawn per local date in `zone`, so the day either side
    of a DST change is each the right length in UTC.
    """
    gaps = availability_windows(busy, start=start, end=end, min_duration=timedelta(0))
    first = to_local(start, zone).date()
    last = to_local(end, zone).date()
    windows: list[TimeSlot] = []
    day = first
    while day <= last:
        lo = _utc_naive(localize(datetime.combine(day, day_start), zone)[0])
        hi = _utc_naive(localize(datetime.combine(day, day_end), zone)[0])
        for gap in gaps:
            piece_lo, piece_hi = max(gap.starts_at, lo), min(gap.ends_at, hi)
            if piece_hi - piece_lo >= min_duration and piece_hi > piece_lo:
                windows.append(TimeSlot.of(piece_lo, piece_hi, zone))
        day += timedelta(days=1)
    return tuple(windows)


def _align_up(moment: datetime, zone: ZoneInfo) -> datetime:
    """The first local grid point at or after `moment` (naive UTC in, naive UTC out)."""
    local = to_local(moment, zone)
    step = int(SLOT_GRANULARITY.total_seconds() // 60)
    minute = local.hour * 60 + local.minute
    extra = (-minute) % step
    if extra == 0 and local.second == 0 and local.microsecond == 0:
        return moment
    aligned = local.replace(second=0, microsecond=0) + timedelta(minutes=extra or step)
    return _utc_naive(aligned)


def propose_slots(
    windows: tuple[TimeSlot, ...] | list[TimeSlot],
    *,
    duration: timedelta,
    zone: ZoneInfo,
    limit: int = MAX_PROPOSALS,
    per_day: int = 1,
    near: datetime | None = None,
) -> tuple[TimeSlot, ...]:
    """Concrete slots of `duration` from free windows, on the local half-hour grid.

    By default the earliest slot of each local day, so offers spread across
    the range. With `near`, the slots closest to that instant come first.
    """
    candidates: list[TimeSlot] = []
    for window in windows:
        start = _align_up(window.starts_at, zone)
        while start + duration <= window.ends_at:
            candidates.append(TimeSlot.of(start, start + duration, zone))
            if near is None:
                break
            start += SLOT_GRANULARITY
    if near is not None:
        candidates.sort(key=lambda s: (abs(s.starts_at - near), s.starts_at))
    chosen: list[TimeSlot] = []
    days: dict[date, int] = {}
    for slot in candidates:
        day = slot.local_start.date()
        if days.get(day, 0) >= per_day:
            continue
        days[day] = days.get(day, 0) + 1
        chosen.append(slot)
        if len(chosen) >= limit:
            break
    return tuple(sorted(chosen, key=lambda s: s.starts_at))


# --- Requests ----------------------------------------------------------------------


class CalendarTask(str, Enum):
    """What a person asked the Calendar subgraph to do. A closed set."""

    AVAILABILITY = "availability"
    PROPOSE_TIMES = "propose_times"
    CREATE_EVENT = "create_event"
    MOVE_EVENT = "move_event"
    CANCEL_EVENT = "cancel_event"


#: Tasks that answer from free/busy and write nothing.
READ_TASKS = frozenset({CalendarTask.AVAILABILITY, CalendarTask.PROPOSE_TIMES})


class CalendarRequest(_Value):
    """A request, parsed. Deterministic: the text decides, no model."""

    task: CalendarTask
    time_zone: str
    #: The time the request is about: the window asked about, the time to
    #: book, or -- for a move -- where to move the event to.
    when: ResolvedTime | None = None
    #: For a move or cancel: the time the event is at now, if the request says.
    current_when: ResolvedTime | None = None
    duration: timedelta | None = None
    title: str | None = None
    attendees: tuple[str, ...] = ()
    #: Words naming the event to move or cancel ("the Acme interview"), or --
    #: for a create -- what it is about, which links an interview to its
    #: application by company name.
    event_hint: str | None = None
    event_id: str | None = None
    application_id: UUID | None = None
    #: An interview, as opposed to any other event.
    interview: bool = False

    @property
    def is_write(self) -> bool:
        return self.task not in READ_TASKS


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_EVENT_ID_RE = re.compile(r"\bevent(?:\s+id)?\s+([A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*)\b", re.I)
_QUOTED_RE = re.compile(r"[\"“']([^\"”']{2,120})[\"”']")
_TITLED_RE = re.compile(
    r"\b(?:titled|called|named)\s+(.+?)(?=\s+(?:on|at|for|with|next|this|tomorrow|today)\b|[.?!]|$)",
    re.I,
)
_CANCEL_RE = re.compile(r"\b(cancel|delete|remove|drop)\b")
_MOVE_RE = re.compile(r"\b(move|reschedule|push|shift|bring forward|change the time)\b")
_CREATE_RE = re.compile(
    r"\b(schedule|book|add|create|set up|put|block( out)?|arrange)\b(?!.*\bavailab)"
)
_PROPOSE_RE = re.compile(
    r"\b(propose|suggest|offer|options for|find (?:a |some )?(?:time|slot)s?|"
    r"interview times|times? for)\b"
)
#: Words that say which event, but not by its title.
_HINT_STOPWORDS = frozenset(
    {
        "my",
        "the",
        "a",
        "an",
        "our",
        "this",
        "that",
        "next",
        "event",
        "meeting",
        "appointment",
        "call",
        "on",
        "at",
        "from",
        "with",
        "calendar",
        "today",
        "tomorrow",
        "tonight",
        "morning",
        "afternoon",
        "evening",
        "back",
        "forward",
        *(_WEEKDAYS.keys()),
    }
)


def _split_move(text: str) -> tuple[str, str]:
    """For a move, the part naming the event and the part naming where it goes."""
    lowered = text.lower()
    index = lowered.rfind(" to ")
    if index < 0:
        return text, ""
    return text[:index], text[index + 4 :]


def _event_hint(source: str) -> str | None:
    lowered = source.lower()
    quoted = _QUOTED_RE.search(source)
    if quoted:
        return quoted.group(1).strip()
    match = re.search(
        r"\b(?:move|reschedule|push|shift|cancel|delete|remove|drop)\b\s+(.*)", lowered
    )
    rest = match.group(1) if match else lowered
    rest = _EVENT_ID_RE.sub(" ", rest)
    rest = _UUID_RE.sub(" ", rest)
    for regex in (_ISO_DATE_RE, _MONTH_DAY_RE, _DAY_MONTH_RE, _TIME_12H_RE, _TIME_24H_RE):
        rest = regex.sub(" ", rest)
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9&\-]*", rest) if w not in _HINT_STOPWORDS]
    return " ".join(words) or None


def _title(text: str) -> str | None:
    if match := _TITLED_RE.search(text):
        return match.group(1).strip(" '\"")[:200] or None
    if match := _QUOTED_RE.search(text):
        return match.group(1).strip()[:200]
    return None


#: Checked in order: a request that says "cancel" is a cancel even if it also
#: names a time; anything that matches nothing is an availability question.
_TASK_PATTERNS: tuple[tuple[re.Pattern[str], CalendarTask], ...] = (
    (_CANCEL_RE, CalendarTask.CANCEL_EVENT),
    (_MOVE_RE, CalendarTask.MOVE_EVENT),
    (_PROPOSE_RE, CalendarTask.PROPOSE_TIMES),
    (_CREATE_RE, CalendarTask.CREATE_EVENT),
)


def _task_of(lowered: str) -> CalendarTask:
    for pattern, task in _TASK_PATTERNS:
        if pattern.search(lowered):
            return task
    return CalendarTask.AVAILABILITY


def parse_calendar_request(
    text: str,
    *,
    now: datetime,
    time_zone: str,
    application_id: UUID | None = None,
    event_id: str | None = None,
) -> CalendarRequest:
    """The task, times and event a request names. Never fails: an unclear request asks availability.

    `application_id` and `event_id`, when the caller already knows them,
    win over anything in the text.
    """
    zone = iana_zone(time_zone)
    raw = text or ""
    lowered = raw.lower()
    task = _task_of(lowered)

    named_event = event_id
    if named_event is None and (match := _EVENT_ID_RE.search(raw)):
        named_event = match.group(1)
    named_application = application_id
    if named_application is None and (match := _UUID_RE.search(raw)):
        named_application = UUID(match.group(0))

    duration = parse_duration(lowered)
    when: ResolvedTime | None
    current: ResolvedTime | None = None
    hint: str | None = None
    if task is CalendarTask.MOVE_EVENT:
        source, destination = _split_move(raw)
        when = resolve_when(destination, now=now, zone=zone, duration=duration)
        current = resolve_when(source, now=now, zone=zone)
        hint = _event_hint(source)
    elif task is CalendarTask.CANCEL_EVENT:
        when = None
        current = resolve_when(raw, now=now, zone=zone)
        hint = _event_hint(raw)
    else:
        when = resolve_when(raw, now=now, zone=zone, duration=duration)
        if task is CalendarTask.CREATE_EVENT:
            # What the event is about ("the Acme interview"), for linking it.
            hint = _event_hint(raw)

    return CalendarRequest(
        task=task,
        time_zone=zone.key,
        when=when,
        current_when=current,
        duration=duration,
        title=_title(raw) if task is CalendarTask.CREATE_EVENT else None,
        attendees=tuple(dict.fromkeys(m.lower() for m in _EMAIL_RE.findall(raw)))[:50],
        event_hint=hint,
        event_id=named_event,
        application_id=named_application,
        interview="interview" in lowered,
    )


# --- Outcomes ------------------------------------------------------------------------


class CalendarOutcome(str, Enum):
    """How a Calendar request ended. A closed set; callers branch on it, not on text."""

    #: Free windows, answered from free/busy.
    AVAILABILITY = "availability"
    #: Concrete slots offered from free/busy.
    PROPOSALS = "proposals"
    #: Writes proposed and waiting for a person's approval.
    PROPOSED = "proposed"
    #: The requested time is taken. Nothing is written; alternatives offered.
    CONFLICT = "conflict"
    #: The request names no usable time; it is asked back.
    NEEDS_CLARIFICATION = "needs_clarification"
    EVENT_NOT_FOUND = "event_not_found"
    #: More than one event fits the description; it is asked back.
    AMBIGUOUS_EVENT = "ambiguous_event"
    #: The event was not created by PersonalOS. It is never modified or deleted.
    NOT_MANAGED = "not_managed"
    #: The event is already where it was asked to be.
    NO_CHANGE = "no_change"
    IN_PAST = "in_past"
    #: The application named is not one of this person's.
    APPLICATION_NOT_FOUND = "application_not_found"


class BusyEvent(_Value):
    """An event in the way, or a candidate for the one meant. Naive UTC."""

    event_id: str | None = None
    title: str = ""
    starts_at: datetime
    ends_at: datetime
    #: Whether PersonalOS created it (it carries the idempotency stamp).
    managed: bool = False

    @classmethod
    def of(cls, event: CalendarEvent) -> "BusyEvent":
        return cls(
            event_id=event.event_id,
            title=event.title,
            starts_at=event.starts_at,
            ends_at=event.ends_at,
            managed=event.stamped,
        )


class CalendarConflict(_Value):
    """A requested time that is already taken, what takes it, and what is free instead."""

    requested: TimeSlot
    #: Titled events in the way. Free/busy intervals without a title are
    #: listed with no `event_id`.
    conflicts_with: tuple[BusyEvent, ...]
    alternatives: tuple[TimeSlot, ...] = ()


class CalendarChangePlan(_Value):
    """What a create, move or cancel request came to."""

    outcome: CalendarOutcome
    detail: str
    task: CalendarTask
    #: The time the write puts the event at (create, move).
    slot: TimeSlot | None = None
    conflict: CalendarConflict | None = None
    #: The event being moved or cancelled, as it is now.
    event: BusyEvent | None = None
    #: For an ambiguous description: the events it could mean.
    candidates: tuple[BusyEvent, ...] = ()
    application_id: UUID | None = None
    #: Prep blocks of a linked interview that could not be placed. The
    #: interview is still proposed; these are for the person to see.
    prep_conflicts: tuple[ScheduleConflict, ...] = ()
    #: How the times were read, for the person to check.
    resolution_notes: tuple[str, ...] = ()


def overlapping(
    slot_start: datetime,
    slot_end: datetime,
    events: list[CalendarEvent] | tuple[CalendarEvent, ...],
    busy: list[BusyInterval] | tuple[BusyInterval, ...] = (),
    *,
    exclude: frozenset[str] = frozenset(),
    own_intervals: tuple[tuple[datetime, datetime], ...] = (),
) -> tuple[BusyEvent, ...]:
    """What is booked over `[slot_start, slot_end)`.

    Busy events (not `exclude`d), plus free/busy intervals no listed event
    explains -- another calendar's booking, say. `own_intervals` are the
    excluded events' own times, which free/busy reports as busy too.
    """
    hits: list[BusyEvent] = []
    explained: list[tuple[datetime, datetime]] = list(own_intervals)
    for event in events:
        if event.event_id in exclude:
            continue
        explained.append((event.starts_at, event.ends_at))
        if event.busy and event.starts_at < slot_end and event.ends_at > slot_start:
            hits.append(BusyEvent.of(event))
    for interval in busy:
        if not (interval.starts_at < slot_end and interval.ends_at > slot_start):
            continue
        covered = any(lo <= interval.starts_at and interval.ends_at <= hi for lo, hi in explained)
        if not covered:
            hits.append(BusyEvent(starts_at=interval.starts_at, ends_at=interval.ends_at))
    return tuple(sorted(hits, key=lambda h: h.starts_at))


def without_intervals(
    busy: list[BusyInterval] | tuple[BusyInterval, ...],
    own: tuple[tuple[datetime, datetime], ...],
) -> tuple[BusyInterval, ...]:
    """Free/busy less the intervals that are exactly an event being moved."""
    return tuple(b for b in busy if (b.starts_at, b.ends_at) not in set(own))


def match_events(
    events: list[CalendarEvent] | tuple[CalendarEvent, ...], hint: str | None
) -> tuple[CalendarEvent, ...]:
    """The events whose titles best fit `hint` (most hint words in the title)."""
    words = [w for w in (hint or "").lower().split() if w]
    if not words:
        return ()
    scored = []
    for event in events:
        title = event.title.lower()
        score = sum(1 for word in words if word in title)
        if score:
            scored.append((score, event))
    if not scored:
        return ()
    best = max(score for score, _ in scored)
    return tuple(
        sorted((e for s, e in scored if s == best), key=lambda e: (e.starts_at, e.event_id))
    )


# --- Write intents -------------------------------------------------------------------


def general_event_key(user_id: UUID, title: str, starts_at: datetime) -> str:
    """The logical key of an event the Calendar subgraph creates: stable for one request."""
    seed = f"{user_id}|{title.strip().lower()}|{starts_at:%Y%m%dT%H%M}"
    return f"event:{uuid5(NAMESPACE_URL, seed)}"


def general_event_properties(logical_key: str) -> dict[str, str]:
    return {PROP_KEY: logical_key, PROP_ROLE: CalendarEventRole.EVENT.value}


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def calendar_write_intent(
    *,
    kind: ActionKind,
    title: str,
    starts_at: datetime,
    ends_at: datetime,
    zone: ZoneInfo,
    requested_by: str,
    idempotency_key: str,
    logical_key: str | None,
    role: CalendarEventRole | None,
    now: datetime,
    attendees: tuple[str, ...] = (),
    event_id: str | None = None,
    previous: tuple[datetime, datetime] | None = None,
    application_id: UUID | None = None,
    interview_starts_at: datetime | None = None,
    properties: dict[str, str] | None = None,
    resolved: ResolvedTime | None = None,
) -> ActionIntent:
    """One approval-gated calendar write, with its preview in the hashed payload.

    The payload names the title, attendees, start and end (UTC and local,
    with the IANA zone), and -- for a move -- where the event is now and what
    changes; plus the phrase the time was read from and how it was read.
    """
    when = format_local(starts_at, ends_at, zone)
    who = ", ".join(attendees) if attendees else "your calendar"
    attendee_line = f"\nAttendees: {who}" if attendees else ""
    optional = {
        PAYLOAD_LOGICAL_KEY: logical_key or None,
        PAYLOAD_ROLE: role.value if role is not None else None,
        PAYLOAD_APPLICATION_ID: str(application_id) if application_id is not None else None,
        PAYLOAD_INTERVIEW_STARTS_AT: _iso(interview_starts_at) if interview_starts_at else None,
        PAYLOAD_RESOLVED_FROM: resolved.phrase if resolved is not None else None,
        PAYLOAD_RESOLUTION_NOTES: list(resolved.notes) if resolved is not None else None,
    }
    payload: dict[str, Any] = {
        PAYLOAD_TITLE: title,
        PAYLOAD_SUBJECT: title,
        PAYLOAD_RECIPIENT: who,
        PAYLOAD_ATTENDEES: list(attendees),
        PAYLOAD_STARTS_AT: _iso(starts_at),
        PAYLOAD_ENDS_AT: _iso(ends_at),
        PAYLOAD_TIME_ZONE: zone.key,
        PAYLOAD_LOCAL_STARTS_AT: _iso(to_local(starts_at, zone)),
        PAYLOAD_LOCAL_ENDS_AT: _iso(to_local(ends_at, zone)),
        **{key: value for key, value in optional.items() if value is not None},
    }

    def intent(target: str, summary: str) -> ActionIntent:
        return ActionIntent(
            kind=kind,
            target=target,
            summary=summary[:500],
            payload=payload,
            idempotency_key=idempotency_key,
            requested_by=requested_by,
            created_at=now,
        )

    if kind is ActionKind.CREATE_CALENDAR_EVENT:
        payload[PAYLOAD_PROPERTIES] = dict(properties or {})
        payload[PAYLOAD_BODY] = f"{title}\n{when}{attendee_line}"
        payload[PAYLOAD_PREVIOUS_BODY] = ""
        return intent(f"calendar: new event '{title}'", f"Add '{title}' to the calendar, {when}")

    if not event_id:
        raise CalendarContractError(f"a calendar {kind.value} must name the event it changes")
    payload[PAYLOAD_EVENT_ID] = event_id
    target = f"calendar: event {event_id} ('{title}')"

    if kind is ActionKind.UPDATE_CALENDAR_EVENT:
        if previous is None:
            raise CalendarContractError("a calendar move must say where the event is now")
        was = format_local(*previous, zone)
        payload.update(_move_fields(previous, (starts_at, ends_at), zone))
        payload[PAYLOAD_PREVIOUS_BODY] = f"{title}\n{was}{attendee_line}"
        payload[PAYLOAD_BODY] = f"{title}\n{when}{attendee_line}"
        summary = f"Move '{title}' from {was} to {when}"
        return intent(target, summary)

    if kind is ActionKind.CANCEL_CALENDAR_EVENT:
        payload[PAYLOAD_PREVIOUS_BODY] = f"{title}\n{when}{attendee_line}"
        payload[PAYLOAD_BODY] = ""
        summary = f"Cancel '{title}', {when}"
        return intent(target, summary)

    raise CalendarContractError(f"'{kind.value}' is not a calendar write")


def _move_fields(
    previous: tuple[datetime, datetime], new: tuple[datetime, datetime], zone: ZoneInfo
) -> dict[str, Any]:
    """Where a moved event is now, in UTC and local time, and each field that changes."""
    (was_start, was_end), (starts_at, ends_at) = previous, new
    changes = [
        {"field": name, "from": _iso(to_local(old, zone)), "to": _iso(to_local(value, zone))}
        for name, old, value in (("start", was_start, starts_at), ("end", was_end, ends_at))
        if old != value
    ]
    return {
        PAYLOAD_PREVIOUS_STARTS_AT: _iso(was_start),
        PAYLOAD_PREVIOUS_ENDS_AT: _iso(was_end),
        PAYLOAD_PREVIOUS_LOCAL_STARTS_AT: _iso(to_local(was_start, zone)),
        PAYLOAD_PREVIOUS_LOCAL_ENDS_AT: _iso(to_local(was_end, zone)),
        PAYLOAD_CHANGES: changes,
    }


def calendar_cancel_key(event: CalendarEvent) -> str:
    """Idempotency key for cancelling one event as it is now."""
    version = event.etag or f"{event.starts_at:%Y%m%dT%H%M}"
    return f"cal-cancel:{event.event_id}:{version}"[:255]


__all__ = [
    "PAYLOAD_ATTENDEES",
    "PAYLOAD_LOCAL_STARTS_AT",
    "PAYLOAD_LOCAL_ENDS_AT",
    "PAYLOAD_PREVIOUS_STARTS_AT",
    "PAYLOAD_PREVIOUS_ENDS_AT",
    "PAYLOAD_PREVIOUS_LOCAL_STARTS_AT",
    "PAYLOAD_PREVIOUS_LOCAL_ENDS_AT",
    "PAYLOAD_CHANGES",
    "PAYLOAD_RESOLVED_FROM",
    "PAYLOAD_RESOLUTION_NOTES",
    "PAYLOAD_APPLICATION_ID",
    "DEFAULT_EVENT_DURATION",
    "DEFAULT_DAY_START",
    "DEFAULT_DAY_END",
    "DEFAULT_LOOKAHEAD",
    "EVENT_SEARCH_HORIZON",
    "MAX_PROPOSALS",
    "MAX_ALTERNATIVES",
    "MAX_WRITES_PER_RUN",
    "SLOT_GRANULARITY",
    "AvailabilityWindow",
    "CalendarContractError",
    "iana_zone",
    "to_local",
    "localize",
    "format_local",
    "ResolvedTime",
    "TimeSlot",
    "parse_duration",
    "resolve_when",
    "at_local_time_of",
    "local_free_windows",
    "propose_slots",
    "CalendarTask",
    "READ_TASKS",
    "CalendarRequest",
    "parse_calendar_request",
    "CalendarOutcome",
    "BusyEvent",
    "CalendarConflict",
    "CalendarChangePlan",
    "overlapping",
    "without_intervals",
    "match_events",
    "general_event_key",
    "general_event_properties",
    "calendar_write_intent",
    "calendar_cancel_key",
]
