"""Interview scheduling: what goes on the calendar for an interview, and where.

An interview invite with a time in it calls for three calendar entries: the
interview itself and the prep blocks that lead up to it. This module decides
what those entries are and how they differ from what the calendar already
holds. It decides nothing about *doing* any of it: the result is an
`InterviewSchedulePlan`, which the Job Search graph turns into
`CREATE_CALENDAR_EVENT` / `UPDATE_CALENDAR_EVENT` intents, and those reach the
calendar only through the approval triple.

Two properties are the reason it is a module of its own.

**The calendar is the record of the schedule.** Every event this system
creates carries private extended properties naming the application it belongs
to and a stable *logical key* (`interview:<application>`,
`prep:<application>:<slot>`). Nothing else remembers which prep block goes
with which interview. So `plan_interview_schedule` is a reconciliation: given
the interview time and the events currently on the calendar, it reports for
each logical event whether to create it, move it or leave it. That one
function covers the first invite, a recruiter moving the interview, and the
candidate dragging the interview to another slot in the calendar UI -- the
prep blocks follow the interview because they are re-planned against it.

**Prep blocks go around what is already there.** A block is placed in the
latest free slot before the interview that fits its `PrepBlockSpec`, and an
existing block is left where it is while it is still usable. A block that no
longer fits (the interview moved, or something else was booked over it) is
moved; one that cannot be placed at all is reported as a `ScheduleConflict`
rather than dropped silently.

Pure: no clock, no calendar client. Every function takes the events and `now`
it should reason about.
"""

import re
from datetime import datetime, time, timedelta
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personalos.domain.job_search import JobSearchContractError
from personalos.domain.recruiter_events import Commitment, InterviewInviteReceived

#: Private extended properties written on every calendar event this system
#: creates. They are how an event is found again: by application, by logical
#: key, and -- on a retry -- by the idempotency key of the write that made it.
PROP_APPLICATION_ID = "personalos_application_id"
PROP_KEY = "personalos_key"
PROP_ROLE = "personalos_role"
#: On a prep block: the logical key of the interview it prepares for.
PROP_INTERVIEW_KEY = "personalos_interview_key"
#: The idempotency key of the last write applied to the event.
PROP_IDEMPOTENCY_KEY = "personalos_idempotency_key"

#: `ActionIntent.payload` keys of the two calendar action kinds. Part of the
#: payload, and so of the hash an approval is bound to.
PAYLOAD_EVENT_ID = "event_id"
PAYLOAD_LOGICAL_KEY = "logical_key"
PAYLOAD_ROLE = "role"
PAYLOAD_TITLE = "title"
PAYLOAD_STARTS_AT = "starts_at"
PAYLOAD_ENDS_AT = "ends_at"
PAYLOAD_PROPERTIES = "properties"

DEFAULT_INTERVIEW_DURATION = timedelta(hours=1)

#: How long before the interview the reminder fires when no prep block could
#: be placed to hang it on.
DEFAULT_REMINDER_LEAD = timedelta(days=1)

#: Bound on the slot search, so a calendar of back-to-back events ends it.
_MAX_PLACEMENT_STEPS = 500

#: A commitment whose action reads like the interview itself, as opposed to
#: "send your availability by Friday".
_INTERVIEW_ACTION_RE = re.compile(
    r"\b(interview|phone screen|screening|onsite|on-site|technical screen|"
    r"video call|call with|meet with|meeting with|chat with)\b",
    re.IGNORECASE,
)
#: ...unless it is about arranging one.
_ARRANGING_RE = re.compile(
    r"\b(availability|available times|let us know|confirm by|reply by|respond by|"
    r"send (us )?(your )?times)\b",
    re.IGNORECASE,
)


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalendarEventRole(str, Enum):
    INTERVIEW = "interview"
    PREP = "prep"


class CalendarEvent(_Value):
    """One event as the calendar reports it. Times are naive UTC."""

    event_id: str
    title: str = ""
    starts_at: datetime
    ends_at: datetime
    #: Private extended properties. Empty for events this system did not make.
    properties: dict[str, str] = Field(default_factory=dict)
    #: False for an event marked free/transparent, which blocks nothing.
    busy: bool = True
    etag: str | None = None

    @property
    def application_id(self) -> str | None:
        return self.properties.get(PROP_APPLICATION_ID)

    @property
    def logical_key(self) -> str | None:
        return self.properties.get(PROP_KEY)

    @property
    def role(self) -> CalendarEventRole | None:
        raw = self.properties.get(PROP_ROLE)
        return CalendarEventRole(raw) if raw in {r.value for r in CalendarEventRole} else None


class InterviewRequest(_Value):
    """An interview at a known time, to be put on (or reconciled with) the calendar."""

    application_id: UUID
    starts_at: datetime
    duration: timedelta = DEFAULT_INTERVIEW_DURATION
    title: str = "Interview"
    #: The message the time was read from; `None` when it came from the
    #: calendar itself.
    source_message_id: str | None = None

    @property
    def ends_at(self) -> datetime:
        return self.starts_at + self.duration


class WorkingHours(_Value):
    """The part of the day a prep block may be placed in.

    Times in this system are naive UTC, so the candidate's day is expressed as
    an offset from it rather than as a zone name.
    """

    start: time = time(8, 0)
    end: time = time(20, 0)
    utc_offset: timedelta = timedelta(0)

    @model_validator(mode="after")
    def _start_before_end(self) -> "WorkingHours":
        if self.start >= self.end:
            raise JobSearchContractError("working hours must start before they end")
        return self

    @property
    def length(self) -> timedelta:
        day = datetime(2000, 1, 1)
        return datetime.combine(day, self.end) - datetime.combine(day, self.start)

    def contains(self, start: datetime, end: datetime) -> bool:
        local_start, local_end = start + self.utc_offset, end + self.utc_offset
        return (
            local_start.date() == local_end.date()
            and local_start.time() >= self.start
            and local_end >= local_start
            and local_end <= datetime.combine(local_end.date(), self.end)
        )

    def latest_end(self, end: datetime, duration: timedelta) -> datetime:
        """The latest moment at or before `end` a block of `duration` can end at."""
        local = end + self.utc_offset
        day_end = datetime.combine(local.date(), self.end)
        if local > day_end:
            local = day_end
        if local - duration < datetime.combine(local.date(), self.start):
            local = datetime.combine(local.date() - timedelta(days=1), self.end)
        return local - self.utc_offset


class PrepBlockSpec(_Value):
    """One kind of prep block: how long, and where before the interview it may sit."""

    slot: str
    title: str
    duration: timedelta
    #: The block must end at least this long before the interview starts.
    lead: timedelta
    #: The block may start no earlier than this long before the interview.
    horizon: timedelta
    working_hours_only: bool = True


#: A long block the day before, and a short one right before the interview.
DEFAULT_PREP_BLOCKS: tuple[PrepBlockSpec, ...] = (
    PrepBlockSpec(
        slot="deep_prep",
        title="Interview prep",
        duration=timedelta(minutes=90),
        lead=timedelta(hours=12),
        horizon=timedelta(hours=72),
    ),
    PrepBlockSpec(
        slot="warm_up",
        title="Interview warm-up",
        duration=timedelta(minutes=30),
        lead=timedelta(minutes=15),
        horizon=timedelta(hours=2),
        # Tied to the interview, whatever hour that is.
        working_hours_only=False,
    ),
)


class ScheduleOp(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    #: Already on the calendar where it should be. Nothing to write.
    KEEP = "keep"


class ScheduleChange(_Value):
    """One logical event of the schedule, and what the calendar needs for it."""

    op: ScheduleOp
    logical_key: str
    role: CalendarEventRole
    title: str
    starts_at: datetime
    ends_at: datetime
    #: Set for UPDATE and KEEP: the event already on the calendar.
    event_id: str | None = None
    etag: str | None = None
    previous_starts_at: datetime | None = None
    previous_ends_at: datetime | None = None


class ConflictKind(str, Enum):
    #: Something else is booked over the interview.
    INTERVIEW_OVERLAP = "interview_overlap"
    #: No free slot fits this prep block before the interview.
    PREP_UNPLACEABLE = "prep_unplaceable"
    #: The interview time has already passed; nothing is scheduled.
    INTERVIEW_IN_PAST = "interview_in_past"


class ScheduleConflict(_Value):
    kind: ConflictKind
    logical_key: str
    detail: str
    #: The event in the way, for an overlap.
    event_id: str | None = None


class InterviewSchedulePlan(_Value):
    """The interview and its prep blocks, reconciled against the calendar."""

    application_id: UUID
    interview_key: str
    interview_starts_at: datetime
    interview_ends_at: datetime
    changes: tuple[ScheduleChange, ...] = ()
    conflicts: tuple[ScheduleConflict, ...] = ()

    @property
    def writes(self) -> tuple[ScheduleChange, ...]:
        """The changes that need a calendar write."""
        return tuple(change for change in self.changes if change.op is not ScheduleOp.KEEP)

    @property
    def prep_blocks(self) -> tuple[ScheduleChange, ...]:
        return tuple(c for c in self.changes if c.role is CalendarEventRole.PREP)

    def reminder_at(self, now: datetime) -> datetime | None:
        """When to remind the candidate to prepare, or `None` if it is too late to.

        The start of the first prep block still ahead; with none, a day before
        the interview.
        """
        upcoming = [c.starts_at for c in self.prep_blocks if c.starts_at > now]
        at = min(upcoming) if upcoming else self.interview_starts_at - DEFAULT_REMINDER_LEAD
        at = max(at, now)
        return at if at < self.interview_starts_at else None


def interview_event_key(application_id: UUID) -> str:
    """The logical key of an application's interview event."""
    return f"interview:{application_id}"


def prep_block_key(application_id: UUID, slot: str) -> str:
    """The logical key of one of an application's prep blocks."""
    return f"prep:{application_id}:{slot}"


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M")


def calendar_create_key(change: ScheduleChange) -> str:
    """Idempotency key for creating one logical event at one time.

    Stable across re-proposals, so a create that is retried, or proposed again
    by a replayed step, is one create.
    """
    return f"cal-create:{change.logical_key}:{_stamp(change.starts_at)}"[:255]


def calendar_update_key(change: ScheduleChange) -> str:
    """Idempotency key for moving one event from where it is to where it should be."""
    was = change.etag or (_stamp(change.previous_starts_at) if change.previous_starts_at else "")
    return f"cal-update:{change.event_id}:{was}:{_stamp(change.starts_at)}"[:255]


def interview_reminder_dedupe_key(application_id: UUID, starts_at: datetime) -> str:
    """Identity of the prep reminder for one interview at one time.

    The time is part of it: a moved interview gets a new reminder, and the old
    one is cancelled rather than re-dated.
    """
    return f"interview_reminder:{application_id}:{_stamp(starts_at)}"


def interview_time_from(commitments: tuple[Commitment, ...]) -> datetime | None:
    """The interview's start time, if one of the commitments states it.

    A commitment is taken as the interview itself when its action reads like
    one ("interview on Tuesday at 2pm") and not like arranging one ("send your
    availability by Friday"). With several, the most confident wins, then the
    earliest. `None` means the invite did not fix a time, and nothing should
    be put on a calendar for it.
    """
    stated = [
        c
        for c in commitments
        if c.due_at is not None
        and _INTERVIEW_ACTION_RE.search(c.action)
        and not _ARRANGING_RE.search(c.action)
    ]
    if not stated:
        return None
    return min(stated, key=lambda c: (-c.confidence, c.due_at)).due_at


def interview_request_from(invite: InterviewInviteReceived) -> InterviewRequest | None:
    """The interview an invite asks for, or `None` when it names no time."""
    starts_at = interview_time_from(invite.commitments)
    if starts_at is None:
        return None
    title = (
        f"Interview: {invite.subject.strip()}" if (invite.subject or "").strip() else "Interview"
    )
    return InterviewRequest(
        application_id=invite.application_id,
        starts_at=starts_at,
        title=title[:200],
        source_message_id=invite.source_message_id,
    )


def interview_request_from_event(event: CalendarEvent) -> InterviewRequest | None:
    """The interview a changed calendar event now describes, if it is one of ours."""
    if event.role is not CalendarEventRole.INTERVIEW or not event.application_id:
        return None
    try:
        application_id = UUID(event.application_id)
    except ValueError:
        return None
    if event.ends_at <= event.starts_at:
        return None
    return InterviewRequest(
        application_id=application_id,
        starts_at=event.starts_at,
        duration=event.ends_at - event.starts_at,
        title=event.title or "Interview",
    )


def _overlaps(start: datetime, end: datetime, other_start: datetime, other_end: datetime) -> bool:
    return other_start < end and other_end > start


def _latest_free_slot(
    *,
    duration: timedelta,
    earliest_start: datetime,
    latest_end: datetime,
    busy: list[tuple[datetime, datetime]],
    hours: WorkingHours | None,
) -> tuple[datetime, datetime] | None:
    """The latest gap of `duration` inside the window that nothing occupies."""
    end = latest_end
    for _ in range(_MAX_PLACEMENT_STEPS):
        if hours is not None:
            end = hours.latest_end(end, duration)
        start = end - duration
        if start < earliest_start:
            return None
        blockers = [b for b in busy if _overlaps(start, end, *b)]
        if not blockers:
            return start, end
        # Try to end where the earliest thing in the way begins.
        end = min(b[0] for b in blockers)
    return None


def _change(
    *,
    logical_key: str,
    role: CalendarEventRole,
    title: str,
    starts_at: datetime,
    ends_at: datetime,
    existing: CalendarEvent | None,
) -> ScheduleChange:
    if existing is None:
        return ScheduleChange(
            op=ScheduleOp.CREATE,
            logical_key=logical_key,
            role=role,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
        )
    moved = (existing.starts_at, existing.ends_at) != (starts_at, ends_at)
    return ScheduleChange(
        op=ScheduleOp.UPDATE if moved else ScheduleOp.KEEP,
        logical_key=logical_key,
        role=role,
        # A title the candidate edited is theirs; only the time is reconciled.
        title=existing.title or title,
        starts_at=starts_at,
        ends_at=ends_at,
        event_id=existing.event_id,
        etag=existing.etag,
        previous_starts_at=existing.starts_at,
        previous_ends_at=existing.ends_at,
    )


def plan_interview_schedule(
    request: InterviewRequest,
    calendar_events: list[CalendarEvent] | tuple[CalendarEvent, ...],
    *,
    now: datetime,
    prep_blocks: tuple[PrepBlockSpec, ...] = DEFAULT_PREP_BLOCKS,
    working_hours: WorkingHours | None = None,
) -> InterviewSchedulePlan:
    """Reconcile an interview and its prep blocks against the calendar.

    `calendar_events` must include every event already linked to this
    application (wherever on the calendar it sits) and everything else booked
    between `now` and the interview. Linked events are matched by logical key
    and moved rather than duplicated; everything else is something to place
    prep blocks around.
    """
    hours = working_hours or WorkingHours()
    application = str(request.application_id)
    interview_key = interview_event_key(request.application_id)

    own: dict[str, CalendarEvent] = {}
    foreign: list[CalendarEvent] = []
    for event in calendar_events:
        if event.application_id == application and event.logical_key:
            own.setdefault(event.logical_key, event)
        elif event.busy:
            foreign.append(event)

    plan = {
        "application_id": request.application_id,
        "interview_key": interview_key,
        "interview_starts_at": request.starts_at,
        "interview_ends_at": request.ends_at,
    }
    if request.starts_at <= now:
        return InterviewSchedulePlan(
            **plan,
            conflicts=(
                ScheduleConflict(
                    kind=ConflictKind.INTERVIEW_IN_PAST,
                    logical_key=interview_key,
                    detail=f"the interview time {request.starts_at.isoformat()} has passed",
                ),
            ),
        )

    changes = [
        _change(
            logical_key=interview_key,
            role=CalendarEventRole.INTERVIEW,
            title=request.title,
            starts_at=request.starts_at,
            ends_at=request.ends_at,
            existing=own.get(interview_key),
        )
    ]
    conflicts = [
        ScheduleConflict(
            kind=ConflictKind.INTERVIEW_OVERLAP,
            logical_key=interview_key,
            event_id=event.event_id,
            detail=(
                f"'{event.title or event.event_id}' "
                f"({event.starts_at.isoformat()} to {event.ends_at.isoformat()}) "
                f"overlaps the interview"
            ),
        )
        for event in foreign
        if _overlaps(request.starts_at, request.ends_at, event.starts_at, event.ends_at)
    ]

    busy = [(event.starts_at, event.ends_at) for event in foreign]
    busy.append((request.starts_at, request.ends_at))

    for spec in prep_blocks:
        key = prep_block_key(request.application_id, spec.slot)
        existing = own.get(key)
        spec_hours = hours if spec.working_hours_only else None
        if spec_hours is not None and spec.duration > spec_hours.length:
            raise JobSearchContractError(
                f"prep block '{spec.slot}' is longer than the working day it must fit in"
            )
        earliest_start = max(request.starts_at - spec.horizon, now)
        latest_end = request.starts_at - spec.lead

        slot: tuple[datetime, datetime] | None = None
        if existing is not None and (
            # Already over: history, not something to move.
            existing.ends_at <= now
            or (
                existing.starts_at >= earliest_start
                and existing.ends_at <= latest_end
                and (
                    spec_hours is None or spec_hours.contains(existing.starts_at, existing.ends_at)
                )
                and not any(_overlaps(existing.starts_at, existing.ends_at, *b) for b in busy)
            )
        ):
            slot = (existing.starts_at, existing.ends_at)
        else:
            slot = _latest_free_slot(
                duration=spec.duration,
                earliest_start=earliest_start,
                latest_end=latest_end,
                busy=busy,
                hours=spec_hours,
            )

        if slot is None:
            conflicts.append(
                ScheduleConflict(
                    kind=ConflictKind.PREP_UNPLACEABLE,
                    logical_key=key,
                    event_id=existing.event_id if existing else None,
                    detail=(
                        f"no free {int(spec.duration.total_seconds() // 60)}-minute slot for "
                        f"'{spec.title}' between {earliest_start.isoformat()} and "
                        f"{latest_end.isoformat()}"
                    ),
                )
            )
            continue

        busy.append(slot)
        changes.append(
            _change(
                logical_key=key,
                role=CalendarEventRole.PREP,
                title=spec.title,
                starts_at=slot[0],
                ends_at=slot[1],
                existing=existing,
            )
        )

    return InterviewSchedulePlan(**plan, changes=tuple(changes), conflicts=tuple(conflicts))


def event_properties(application_id: UUID, change: ScheduleChange) -> dict[str, str]:
    """The private extended properties a created event is stamped with."""
    properties = {
        PROP_APPLICATION_ID: str(application_id),
        PROP_KEY: change.logical_key,
        PROP_ROLE: change.role.value,
    }
    if change.role is CalendarEventRole.PREP:
        properties[PROP_INTERVIEW_KEY] = interview_event_key(application_id)
    return properties


__all__ = [
    "PROP_APPLICATION_ID",
    "PROP_KEY",
    "PROP_ROLE",
    "PROP_INTERVIEW_KEY",
    "PROP_IDEMPOTENCY_KEY",
    "PAYLOAD_EVENT_ID",
    "PAYLOAD_LOGICAL_KEY",
    "PAYLOAD_ROLE",
    "PAYLOAD_TITLE",
    "PAYLOAD_STARTS_AT",
    "PAYLOAD_ENDS_AT",
    "PAYLOAD_PROPERTIES",
    "DEFAULT_INTERVIEW_DURATION",
    "DEFAULT_REMINDER_LEAD",
    "DEFAULT_PREP_BLOCKS",
    "CalendarEventRole",
    "CalendarEvent",
    "InterviewRequest",
    "WorkingHours",
    "PrepBlockSpec",
    "ScheduleOp",
    "ScheduleChange",
    "ConflictKind",
    "ScheduleConflict",
    "InterviewSchedulePlan",
    "interview_event_key",
    "prep_block_key",
    "calendar_create_key",
    "calendar_update_key",
    "interview_reminder_dedupe_key",
    "interview_time_from",
    "interview_request_from",
    "interview_request_from_event",
    "plan_interview_schedule",
    "event_properties",
]
