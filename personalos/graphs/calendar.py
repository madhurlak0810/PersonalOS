"""The Calendar subgraph: availability, proposed times, and approved event changes.

Pipeline:

    parse_request -> check_availability -> END              (when am I free, propose times)
    parse_request -> plan_calendar_change -> END            (conflict, not found, not ours, ...)
                  -> plan_calendar_change -> request_calendar_approval
                  -> calendar_approval_checkpoint (interrupts)
                  -> execute_calendar_changes -> END        (create, move, cancel)

It builds on J4 -- `GoogleCalendar` as the reader and, inside the shared
action executor, as the `CalendarClient` -- and adds no provider code.

| Effect | Class | Where |
| --- | --- | --- |
| Free/busy, event listings and reads | `READ_EXTERNAL`, no approval | `check_availability`, `plan_calendar_change` |
| Create or move an event | `WRITE_EXTERNAL`, `require_approval` | `execute_calendar_changes`, only after approval |
| Cancel an event | `DESTRUCTIVE`, `require_approval` | `execute_calendar_changes`, only after approval |

**Time is zoned explicitly.** The graph is built with the person's IANA zone
(`CALENDAR_TIME_ZONE`). `parse_request` reads relative phrases against it and
an explicit clock, and every write's approval payload carries the resolved
local and UTC times, the zone, and how the phrase was read -- including a DST
gap or overlap -- so an ambiguity is visible before anything executes.

**No double-booking.** `plan_calendar_change` checks the slot against both
the events on the calendar and free/busy. A taken slot ends the run as a
typed `CalendarOutcome.CONFLICT` with the alternatives free/busy offers, and
no write is proposed.

**Only PersonalOS events are changed.** A move or cancel whose target lacks
the PersonalOS idempotency stamp ends as `CalendarOutcome.NOT_MANAGED`; and
should such an intent ever reach execution, `CalendarActionExecutor` refuses
it again before anything is sent.

**Exactly once.** Every create carries a stable idempotency key, which the
executor stamps on the event (`extendedProperties.private`). Under
`ToolExecutor`, an attempt whose response was lost is reconciled against the
calendar by that stamp before any retry (`CalendarReconciler`), so a retry
after a timeout finds the event instead of inserting another.

**Applications and prep blocks.** A create or move of an interview linked to
an application goes through the Phase E planner
(`plan_interview_schedule`): the interview and its prep blocks are proposed
together, so a changed interview time moves the prep blocks in the same
approval. Cancelling an interview cancels its upcoming prep blocks with it.
The application must be one of the person's (`ApplicationDirectory`).
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol, TypedDict
from uuid import UUID
from zoneinfo import ZoneInfo

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from personalos.domain.calendar import (
    DEFAULT_DAY_END,
    DEFAULT_DAY_START,
    DEFAULT_EVENT_DURATION,
    DEFAULT_LOOKAHEAD,
    EVENT_SEARCH_HORIZON,
    MAX_ALTERNATIVES,
    MAX_WRITES_PER_RUN,
    BusyEvent,
    CalendarChangePlan,
    CalendarConflict,
    CalendarContractError,
    CalendarOutcome,
    CalendarRequest,
    CalendarTask,
    ResolvedTime,
    TimeSlot,
    at_local_time_of,
    calendar_cancel_key,
    calendar_write_intent,
    general_event_key,
    general_event_properties,
    iana_zone,
    local_free_windows,
    localize,
    match_events,
    overlapping,
    parse_calendar_request,
    propose_slots,
    to_local,
    without_intervals,
)
from personalos.domain.errors import PersonalOSError
from personalos.domain.interview_scheduling import (
    DEFAULT_PREP_BLOCKS,
    PROP_INTERVIEW_KEY,
    BusyInterval,
    CalendarEvent,
    CalendarEventRole,
    ConflictKind,
    InterviewRequest,
    PrepBlockSpec,
    ScheduleChange,
    ScheduleOp,
    WorkingHours,
    calendar_create_key,
    calendar_update_key,
    event_properties,
    plan_interview_schedule,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRequest,
    JobSearchContractError,
    authorize_execution,
)
from personalos.domain.redaction import redact_text
from personalos.domain.workflow import calendar_thread_id
from personalos.graphs.job_search import ActionExecutor, ApplicationDirectory
from personalos.observability.structured_logging import LogEvent
from personalos.observability.tracing import traced_node
from personalos.policy import PolicyError
from personalos.policy.rules import CALENDAR_REQUESTER_PREFIX

logger = logging.getLogger(__name__)

GRAPH_NAME = "calendar"

PARSE_REQUEST = "parse_request"
CHECK_AVAILABILITY = "check_availability"
PLAN_CALENDAR_CHANGE = "plan_calendar_change"
REQUEST_CALENDAR_APPROVAL = "request_calendar_approval"
CALENDAR_APPROVAL_CHECKPOINT = "calendar_approval_checkpoint"
EXECUTE_CALENDAR_CHANGES = "execute_calendar_changes"

#: `approval_stage` / the interrupt's `stage`.
STAGE_CALENDAR_CHANGE = "calendar_change"

#: The `requested_by` of this subgraph's calendar reads: the identity the
#: composition root builds its `GoogleCalendar` reader with. Not a node; the
#: policy lets it read and nothing else.
READ_CALENDAR = "read_calendar"


def requester(node: str) -> str:
    """The `requested_by` a node stamps on its intents; policy scopes tools by it."""
    return f"{CALENDAR_REQUESTER_PREFIX}{node}"


# --- Ports ------------------------------------------------------------------------


class CalendarAvailabilityReader(Protocol):
    """Reads the calendar. Cannot write to it.

    `personalos.executor.google_calendar.GoogleCalendar`, built with
    `requested_by=requester(READ_CALENDAR)`.
    """

    async def free_busy(
        self, start: datetime, end: datetime, *, calendar_ids: Sequence[str] | None = None
    ) -> tuple[BusyInterval, ...]: ...

    async def events_between(
        self, user_id: UUID, start: datetime, end: datetime
    ) -> Sequence[CalendarEvent]: ...

    async def events_for_application(
        self, user_id: UUID, application_id: UUID
    ) -> Sequence[CalendarEvent]: ...

    async def get_event(self, event_id: str) -> CalendarEvent | None: ...


class CalendarState(TypedDict, total=False):
    """Graph state. JSON only."""

    request: str
    user_id: str | None
    #: An application or event the caller already knows, bypassing the text.
    application_id: str | None
    event_id: str | None
    calendar_request: dict[str, Any] | None
    availability: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    change_plan: dict[str, Any] | None
    approval_stage: str | None
    pending_actions: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    approval_refusals: list[dict[str, Any]]
    action_receipts: list[dict[str, Any]]
    errors: list[str]


def _dump(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [value.model_dump(mode="json") for value in values]


def _failure(exc: Exception) -> str:
    return redact_text(f"{type(exc).__name__}: {exc}")[:500]


def _user_id(state: CalendarState) -> UUID:
    raw = state.get("user_id")
    if not raw:
        raise JobSearchContractError("a Calendar task needs the candidate's user_id")
    return UUID(str(raw))


def _load_request(state: CalendarState) -> CalendarRequest | None:
    raw = state.get("calendar_request")
    return CalendarRequest.model_validate(raw) if raw else None


def _plan(
    request: CalendarRequest, outcome: CalendarOutcome, detail: str, **fields: Any
) -> CalendarChangePlan:
    notes = fields.pop("resolution_notes", None)
    if notes is None:
        notes = request.when.notes if request.when is not None else ()
    return CalendarChangePlan(
        outcome=outcome, detail=detail, task=request.task, resolution_notes=tuple(notes), **fields
    )


# --- Graph ------------------------------------------------------------------------


class CalendarGraph:
    """Builds and compiles the Calendar subgraph."""

    def __init__(
        self,
        calendar: CalendarAvailabilityReader,
        *,
        time_zone: str,
        action_executor: ActionExecutor | None = None,
        application_directory: ApplicationDirectory | None = None,
        day_start=DEFAULT_DAY_START,
        day_end=DEFAULT_DAY_END,
        prep_blocks: tuple[PrepBlockSpec, ...] = DEFAULT_PREP_BLOCKS,
        approval_ttl: timedelta | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        """Wire the ports.

        `calendar` and `time_zone` (an IANA zone) are required. Without an
        `action_executor`, a write is planned and its approval asked, but
        executing it fails. Without an `application_directory`, a request
        naming an application is refused rather than trusted.
        """
        if calendar is None:
            raise ValueError("CalendarGraph requires a calendar reader")
        self.calendar = calendar
        self.zone: ZoneInfo = iana_zone(time_zone)
        self.time_zone = self.zone.key
        self.action_executor = action_executor
        self.application_directory = application_directory
        self.day_start = day_start
        self.day_end = day_end
        self.prep_blocks = prep_blocks
        self.approval_ttl = approval_ttl
        self.clock = clock
        self.checkpointer = checkpointer or InMemorySaver()

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(CalendarState)
        for name, method in (
            (PARSE_REQUEST, self.parse_request),
            (CHECK_AVAILABILITY, self.check_availability),
            (PLAN_CALENDAR_CHANGE, self.plan_calendar_change),
            (REQUEST_CALENDAR_APPROVAL, self.request_calendar_approval),
            (CALENDAR_APPROVAL_CHECKPOINT, self.calendar_approval_checkpoint),
            (EXECUTE_CALENDAR_CHANGES, self.execute_calendar_changes),
        ):
            graph.add_node(name, traced_node(name, method, graph=GRAPH_NAME))

        graph.add_edge(START, PARSE_REQUEST)
        graph.add_conditional_edges(
            PARSE_REQUEST,
            self.route_after_parse,
            {CHECK_AVAILABILITY: CHECK_AVAILABILITY, PLAN_CALENDAR_CHANGE: PLAN_CALENDAR_CHANGE},
        )
        graph.add_edge(CHECK_AVAILABILITY, END)
        graph.add_conditional_edges(
            PLAN_CALENDAR_CHANGE,
            self.route_after_plan,
            {REQUEST_CALENDAR_APPROVAL: REQUEST_CALENDAR_APPROVAL, END: END},
        )
        graph.add_edge(REQUEST_CALENDAR_APPROVAL, CALENDAR_APPROVAL_CHECKPOINT)
        graph.add_edge(CALENDAR_APPROVAL_CHECKPOINT, EXECUTE_CALENDAR_CHANGES)
        graph.add_edge(EXECUTE_CALENDAR_CHANGES, END)
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def parse_request(self, state: CalendarState) -> dict[str, Any]:
        """Parse the request and resolve its times in the configured zone. Deterministic."""
        raw_application = state.get("application_id")
        request = parse_calendar_request(
            state.get("request", ""),
            now=self.clock(),
            time_zone=self.time_zone,
            application_id=UUID(str(raw_application)) if raw_application else None,
            event_id=state.get("event_id") or None,
        )
        logger.info(
            "calendar request: %s%s",
            request.task.value,
            f" at {request.when.starts_at.isoformat()} UTC" if request.when else "",
        )
        return {
            "calendar_request": request.model_dump(mode="json"),
            "errors": [],
            # A previous run's settled writes are not this run's.
            "pending_actions": [],
            "change_plan": None,
        }

    async def check_availability(self, state: CalendarState) -> dict[str, Any]:
        """Answer from free/busy: the free windows, or concrete slots to offer.

        Reads only. Windows are cut to the working day in the configured
        zone, per local date.
        """
        request = _load_request(state)
        if request is None:  # pragma: no cover - parse_request always runs first
            raise JobSearchContractError("check_availability reached without a parsed request")
        errors = list(state.get("errors") or [])
        now = self.clock()
        start, end = self._read_window(request, now)
        try:
            busy = await self.calendar.free_busy(start, end)
        except (PolicyError, PersonalOSError) as exc:
            errors.append(_failure(exc))
            return {"availability": [], "proposals": [], "errors": errors}
        windows = local_free_windows(
            busy,
            start=start,
            end=end,
            zone=self.zone,
            day_start=self.day_start,
            day_end=self.day_end,
            min_duration=timedelta(minutes=30),
        )
        proposals: tuple[TimeSlot, ...] = ()
        if request.task is CalendarTask.PROPOSE_TIMES:
            proposals = propose_slots(
                windows, duration=request.duration or DEFAULT_EVENT_DURATION, zone=self.zone
            )
        logger.info(
            "availability %s to %s UTC: %d window(s), %d proposal(s)",
            start.isoformat(),
            end.isoformat(),
            len(windows),
            len(proposals),
        )
        return {"availability": _dump(windows), "proposals": _dump(proposals), "errors": errors}

    def _read_window(self, request: CalendarRequest, now: datetime) -> tuple[datetime, datetime]:
        if request.when is not None:
            start, end = request.when.starts_at, request.when.ends_at
            if request.when.exact:
                # "Am I free at 3pm Thursday": the slot itself.
                return start, end
            return max(start, now), max(end, now)
        if request.task is CalendarTask.PROPOSE_TIMES:
            tomorrow = to_local(now, self.zone).date() + timedelta(days=1)
            start = datetime.combine(tomorrow, self.day_start)
            begins = _naive_utc_of(start, self.zone)
            return begins, begins + DEFAULT_LOOKAHEAD
        return now, now + DEFAULT_LOOKAHEAD

    async def plan_calendar_change(self, state: CalendarState) -> dict[str, Any]:
        """Turn a create, move or cancel into approval-gated intents -- or a typed refusal.

        Reads the calendar. Writes nothing and holds no executor.
        """
        request = _load_request(state)
        if request is None:  # pragma: no cover - parse_request always runs first
            raise JobSearchContractError("plan_calendar_change reached without a parsed request")
        errors = list(state.get("errors") or [])
        user_id = _user_id(state)
        now = self.clock()
        try:
            if request.task is CalendarTask.CREATE_EVENT:
                plan, intents = await self._plan_create(request, user_id, now)
            elif request.task is CalendarTask.MOVE_EVENT:
                plan, intents = await self._plan_move(request, user_id, now)
            else:
                plan, intents = await self._plan_cancel(request, user_id, now)
        except (PolicyError, PersonalOSError) as exc:
            errors.append(_failure(exc))
            return {"change_plan": None, "pending_actions": [], "errors": errors}
        if len(intents) > MAX_WRITES_PER_RUN:
            errors.append(
                f"the change needs {len(intents)} calendar writes; at most "
                f"{MAX_WRITES_PER_RUN} are proposed in one run"
            )
            return {
                "change_plan": plan.model_dump(mode="json"),
                "pending_actions": [],
                "errors": errors,
            }
        logger.info(
            "calendar %s: %s (%d write(s))", request.task.value, plan.outcome.value, len(intents)
        )
        update: dict[str, Any] = {
            "change_plan": plan.model_dump(mode="json"),
            "pending_actions": _dump(intents),
            "errors": errors,
        }
        if intents:
            update["approval_stage"] = STAGE_CALENDAR_CHANGE
        return update

    # --- planning helpers --------------------------------------------------

    async def _application(
        self, request: CalendarRequest, user_id: UUID
    ) -> tuple[UUID | None, str | None, CalendarChangePlan | None]:
        """The application the request is about, checked to be this person's."""
        if self.application_directory is None:
            if request.application_id is not None:
                return (
                    None,
                    None,
                    _plan(
                        request,
                        CalendarOutcome.APPLICATION_NOT_FOUND,
                        "no application directory is wired, so the application could not be "
                        "checked as yours; nothing was planned",
                    ),
                )
            return None, None, None
        candidates = list(await self.application_directory.candidates(user_id))
        if request.application_id is not None:
            match = next(
                (c for c in candidates if c.application_id == request.application_id), None
            )
            if match is None:
                return (
                    None,
                    None,
                    _plan(
                        request,
                        CalendarOutcome.APPLICATION_NOT_FOUND,
                        f"application {request.application_id} is not one of yours",
                    ),
                )
            return match.application_id, match.company, None
        if request.interview:
            text = " ".join(part for part in (request.title, request.event_hint) if part).lower()
            named = [c for c in candidates if c.company and c.company.lower() in text]
            if len(named) == 1:
                return named[0].application_id, named[0].company, None
        return None, None, None

    def _hours_at(self, moment: datetime) -> WorkingHours:
        """Working hours with the zone's UTC offset on the day of `moment`."""
        offset = to_local(moment, self.zone).utcoffset() or timedelta(0)
        return WorkingHours(start=self.day_start, end=self.day_end, utc_offset=offset)

    async def _conflict(
        self,
        user_id: UUID,
        start: datetime,
        end: datetime,
        *,
        exclude: Sequence[CalendarEvent] = (),
    ) -> CalendarConflict | None:
        """What is booked over the slot, from events and free/busy, with alternatives.

        `exclude` are the events the change itself moves: they are not in its way.
        """
        events = list(await self.calendar.events_between(user_id, start, end))
        busy = await self.calendar.free_busy(start, end)
        own = tuple((e.starts_at, e.ends_at) for e in exclude)
        hits = overlapping(
            start,
            end,
            events,
            busy,
            exclude=frozenset(e.event_id for e in exclude),
            own_intervals=own,
        )
        if not hits:
            return None
        local_day = to_local(start, self.zone).date()
        search_from = max(
            _naive_utc_of(datetime.combine(local_day, self.day_start), self.zone), self.clock()
        )
        search_to = _naive_utc_of(
            datetime.combine(local_day + timedelta(days=3), self.day_end), self.zone
        )
        around = without_intervals(await self.calendar.free_busy(search_from, search_to), own)
        windows = local_free_windows(
            around,
            start=search_from,
            end=search_to,
            zone=self.zone,
            day_start=self.day_start,
            day_end=self.day_end,
            min_duration=end - start,
        )
        alternatives = propose_slots(
            windows,
            duration=end - start,
            zone=self.zone,
            limit=MAX_ALTERNATIVES,
            per_day=MAX_ALTERNATIVES,
            near=start,
        )
        return CalendarConflict(
            requested=TimeSlot.of(start, end, self.zone),
            conflicts_with=hits,
            alternatives=alternatives,
        )

    async def _first_free_in(
        self, user_id: UUID, when: ResolvedTime, duration: timedelta, now: datetime
    ) -> TimeSlot | None:
        start, end = max(when.starts_at, now), when.ends_at
        windows = local_free_windows(
            await self.calendar.free_busy(start, end),
            start=start,
            end=end,
            zone=self.zone,
            day_start=self.day_start,
            day_end=self.day_end,
            min_duration=duration,
        )
        slots = propose_slots(windows, duration=duration, zone=self.zone, limit=1)
        return slots[0] if slots else None

    async def _slot_for_create(
        self, request: CalendarRequest, user_id: UUID, now: datetime
    ) -> tuple[ResolvedTime | None, CalendarChangePlan | None]:
        when = request.when
        if when is None:
            return None, _plan(
                request,
                CalendarOutcome.NEEDS_CLARIFICATION,
                "say when: a day and a time, such as 'Tuesday at 2pm'",
            )
        if when.exact:
            return when, None
        duration = request.duration or DEFAULT_EVENT_DURATION
        slot = await self._first_free_in(user_id, when, duration, now)
        if slot is None:
            return None, _plan(
                request,
                CalendarOutcome.CONFLICT,
                f"no free {int(duration.total_seconds() // 60)}-minute slot in '{when.phrase}'",
                conflict=CalendarConflict(
                    requested=TimeSlot.of(max(when.starts_at, now), when.ends_at, self.zone),
                    conflicts_with=(),
                ),
            )
        picked = when.model_copy(
            update={
                "exact": True,
                "local_start": slot.local_start,
                "local_end": slot.local_end,
                "starts_at": slot.starts_at,
                "ends_at": slot.ends_at,
                "notes": (
                    *when.notes,
                    f"no time given; picked the earliest free "
                    f"{int(duration.total_seconds() // 60)}-minute slot in '{when.phrase}'",
                ),
            }
        )
        return picked, None

    async def _plan_create(
        self, request: CalendarRequest, user_id: UUID, now: datetime
    ) -> tuple[CalendarChangePlan, list[ActionIntent]]:
        application_id, company, refused = await self._application(request, user_id)
        if refused is not None:
            return refused, []
        when, refused = await self._slot_for_create(request, user_id, now)
        if refused is not None:
            return refused, []
        assert when is not None
        if when.starts_at <= now:
            return _plan(
                request,
                CalendarOutcome.IN_PAST,
                f"{when.local_start:%a %d %b %Y %H:%M %Z} has already passed",
                resolution_notes=when.notes,
            ), []

        if application_id is not None:
            title = request.title or (f"Interview: {company}" if company else "Interview")
            return await self._plan_interview(
                request,
                user_id,
                now,
                application_id=application_id,
                starts_at=when.starts_at,
                duration=when.duration,
                title=title,
                resolved=when,
            )

        conflict = await self._conflict(user_id, when.starts_at, when.ends_at)
        slot = TimeSlot.of(when.starts_at, when.ends_at, self.zone)
        if conflict is not None:
            return _plan(
                request,
                CalendarOutcome.CONFLICT,
                f"{slot.describe()} is already booked; nothing was written",
                slot=slot,
                conflict=conflict,
                resolution_notes=when.notes,
            ), []
        title = request.title or (
            f"Meeting with {', '.join(request.attendees)}" if request.attendees else "Event"
        )
        key = general_event_key(user_id, title, when.starts_at)
        change = ScheduleChange(
            op=ScheduleOp.CREATE,
            logical_key=key,
            role=CalendarEventRole.EVENT,
            title=title,
            starts_at=when.starts_at,
            ends_at=when.ends_at,
        )
        intent = calendar_write_intent(
            kind=ActionKind.CREATE_CALENDAR_EVENT,
            title=title,
            starts_at=when.starts_at,
            ends_at=when.ends_at,
            zone=self.zone,
            requested_by=requester(PLAN_CALENDAR_CHANGE),
            idempotency_key=calendar_create_key(change),
            logical_key=key,
            role=CalendarEventRole.EVENT,
            now=now,
            attendees=request.attendees,
            properties=general_event_properties(key),
            resolved=when,
        )
        return _plan(
            request,
            CalendarOutcome.PROPOSED,
            f"add '{title}' at {slot.describe()}, once approved",
            slot=slot,
            resolution_notes=when.notes,
        ), [intent]

    async def _plan_interview(
        self,
        request: CalendarRequest,
        user_id: UUID,
        now: datetime,
        *,
        application_id: UUID,
        starts_at: datetime,
        duration: timedelta,
        title: str,
        resolved: ResolvedTime | None,
        current: CalendarEvent | None = None,
    ) -> tuple[CalendarChangePlan, list[ActionIntent]]:
        """The interview and its prep blocks, through the Phase E planner."""
        interview = InterviewRequest(
            application_id=application_id, starts_at=starts_at, duration=duration, title=title
        )
        linked = await self.calendar.events_for_application(user_id, application_id)
        horizon = max((spec.horizon for spec in self.prep_blocks), default=timedelta(0))
        booked = await self.calendar.events_between(
            user_id, min(now, starts_at - horizon), interview.ends_at
        )
        by_id = {event.event_id: event for event in (*booked, *linked)}
        # An interview event the candidate made is not ours to move: plan
        # around it as something booked, not as the interview.
        events = [
            e
            if e.stamped or e.application_id != str(application_id)
            else e.model_copy(update={"properties": {}})
            for e in by_id.values()
        ]
        schedule = plan_interview_schedule(
            interview,
            events,
            now=now,
            prep_blocks=self.prep_blocks,
            working_hours=self._hours_at(starts_at),
        )
        slot = TimeSlot.of(interview.starts_at, interview.ends_at, self.zone)
        notes = resolved.notes if resolved is not None else ()
        overlaps = [c for c in schedule.conflicts if c.kind is ConflictKind.INTERVIEW_OVERLAP]
        if any(c.kind is ConflictKind.INTERVIEW_IN_PAST for c in schedule.conflicts):
            return _plan(
                request,
                CalendarOutcome.IN_PAST,
                f"{slot.describe()} has already passed",
                application_id=application_id,
                resolution_notes=notes,
            ), []
        # Free/busy may know of bookings the event listing does not (another calendar).
        own = [e for e in by_id.values() if e.stamped and e.application_id == str(application_id)]
        conflict = await self._conflict(
            user_id, interview.starts_at, interview.ends_at, exclude=own
        )
        if overlaps or conflict is not None:
            if conflict is None:  # pragma: no cover - the listing and free/busy agree in practice
                conflict = CalendarConflict(
                    requested=slot,
                    conflicts_with=tuple(
                        BusyEvent.of(by_id[c.event_id]) for c in overlaps if c.event_id in by_id
                    ),
                )
            return _plan(
                request,
                CalendarOutcome.CONFLICT,
                f"{slot.describe()} is already booked; nothing was written",
                slot=slot,
                conflict=conflict,
                application_id=application_id,
                resolution_notes=notes,
            ), []

        prep_conflicts = tuple(
            c for c in schedule.conflicts if c.kind is ConflictKind.PREP_UNPLACEABLE
        )
        intents: list[ActionIntent] = []
        for change in schedule.writes:
            existing = by_id.get(change.event_id or "")
            if change.op is ScheduleOp.UPDATE and (existing is None or not existing.stamped):
                # Never move what PersonalOS did not make.
                continue
            intents.append(
                self._schedule_intent(
                    change,
                    application_id=application_id,
                    interview_starts_at=schedule.interview_starts_at,
                    now=now,
                    resolved=resolved if change.role is CalendarEventRole.INTERVIEW else None,
                    attendees=existing.attendees if existing is not None else request.attendees,
                )
            )
        if not intents:
            return _plan(
                request,
                CalendarOutcome.NO_CHANGE,
                f"the interview and its prep blocks are already at {slot.describe()}",
                slot=slot,
                application_id=application_id,
                prep_conflicts=prep_conflicts,
                resolution_notes=notes,
            ), []
        moved_prep = sum(
            1 for i in intents if i.payload.get("role") == CalendarEventRole.PREP.value
        )
        return _plan(
            request,
            CalendarOutcome.PROPOSED,
            f"put the interview at {slot.describe()} and {moved_prep} prep block(s) "
            f"around it, once approved",
            slot=slot,
            application_id=application_id,
            event=BusyEvent.of(current) if current is not None else None,
            prep_conflicts=prep_conflicts,
            resolution_notes=notes,
        ), intents

    def _schedule_intent(
        self,
        change: ScheduleChange,
        *,
        application_id: UUID,
        interview_starts_at: datetime,
        now: datetime,
        resolved: ResolvedTime | None,
        attendees: tuple[str, ...],
    ) -> ActionIntent:
        prep = change.role is CalendarEventRole.PREP
        if change.op is ScheduleOp.CREATE:
            return calendar_write_intent(
                kind=ActionKind.CREATE_CALENDAR_EVENT,
                title=change.title,
                starts_at=change.starts_at,
                ends_at=change.ends_at,
                zone=self.zone,
                requested_by=requester(PLAN_CALENDAR_CHANGE),
                idempotency_key=calendar_create_key(change),
                logical_key=change.logical_key,
                role=change.role,
                now=now,
                attendees=() if prep else attendees,
                application_id=application_id,
                interview_starts_at=interview_starts_at if prep else None,
                properties=event_properties(application_id, change),
                resolved=resolved,
            )
        assert change.previous_starts_at is not None and change.previous_ends_at is not None
        return calendar_write_intent(
            kind=ActionKind.UPDATE_CALENDAR_EVENT,
            title=change.title,
            starts_at=change.starts_at,
            ends_at=change.ends_at,
            zone=self.zone,
            requested_by=requester(PLAN_CALENDAR_CHANGE),
            idempotency_key=calendar_update_key(change),
            logical_key=change.logical_key,
            role=change.role,
            now=now,
            attendees=() if prep else attendees,
            event_id=change.event_id,
            previous=(change.previous_starts_at, change.previous_ends_at),
            application_id=application_id,
            interview_starts_at=interview_starts_at if prep else None,
            resolved=resolved,
        )

    async def _find_event(
        self, request: CalendarRequest, user_id: UUID, now: datetime
    ) -> tuple[CalendarEvent | None, CalendarChangePlan | None]:
        """The one event a move or cancel is about, or why it could not be settled.

        By id if the request names one; else the application's one upcoming
        interview; else by title among upcoming events (on the day the
        request says it is, if it says).
        """
        if request.event_id:
            event = await self.calendar.get_event(request.event_id)
            if event is None:
                return None, _plan(
                    request,
                    CalendarOutcome.EVENT_NOT_FOUND,
                    f"there is no calendar event {request.event_id}",
                )
            return event, None
        if request.application_id is not None:
            event, refused = await self._linked_interview(request, user_id, now)
            if event is not None or refused is not None:
                return event, refused
        lo, hi = self._search_window(request, now)
        events = [e for e in await self.calendar.events_between(user_id, lo, hi) if e.ends_at > now]
        matches = match_events(events, request.event_hint)
        if not matches and request.current_when is not None and not request.event_hint:
            matches = tuple(events)
        if len(matches) == 1:
            return matches[0], None
        if matches:
            return None, _plan(
                request,
                CalendarOutcome.AMBIGUOUS_EVENT,
                f"{len(matches)} events match; say which one (by its event id)",
                candidates=tuple(BusyEvent.of(e) for e in matches[:10]),
            )
        described = request.event_hint or (
            request.current_when.phrase if request.current_when else ""
        )
        return None, _plan(
            request,
            CalendarOutcome.EVENT_NOT_FOUND,
            f"no upcoming event matches '{described}'",
        )

    async def _linked_interview(
        self, request: CalendarRequest, user_id: UUID, now: datetime
    ) -> tuple[CalendarEvent | None, CalendarChangePlan | None]:
        """The named application's one upcoming interview event, if it has exactly one."""
        assert request.application_id is not None
        _, _, refused = await self._application(request, user_id)
        if refused is not None:
            return None, refused
        linked = [
            e
            for e in await self.calendar.events_for_application(user_id, request.application_id)
            if e.role is CalendarEventRole.INTERVIEW and e.ends_at > now
        ]
        return (linked[0] if len(linked) == 1 else None), None

    @staticmethod
    def _search_window(request: CalendarRequest, now: datetime) -> tuple[datetime, datetime]:
        current = request.current_when
        if current is None:
            return now, now + EVENT_SEARCH_HORIZON
        if current.exact:
            return current.starts_at - timedelta(minutes=1), current.ends_at
        return current.starts_at, current.ends_at

    def _move_time(
        self, request: CalendarRequest, event: CalendarEvent, now: datetime
    ) -> tuple[ResolvedTime | None, CalendarChangePlan | None]:
        """Where a move puts the event, or why it cannot be moved there."""
        current = BusyEvent.of(event)
        when = request.when
        if when is not None and not when.exact:
            when = at_local_time_of(when, event, self.zone)
        if when is None:
            return None, _plan(
                request,
                CalendarOutcome.NEEDS_CLARIFICATION,
                "say where to move it: a day and a time, such as 'Friday at 3pm'",
                event=current,
            )
        duration = request.duration or (event.ends_at - event.starts_at)
        when = when.model_copy(
            update={
                "ends_at": when.starts_at + duration,
                "local_end": to_local(when.starts_at + duration, self.zone),
            }
        )
        if when.starts_at <= now:
            return None, _plan(
                request,
                CalendarOutcome.IN_PAST,
                f"{when.local_start:%a %d %b %Y %H:%M %Z} has already passed",
                event=current,
                resolution_notes=when.notes,
            )
        if (event.starts_at, event.ends_at) == (when.starts_at, when.ends_at):
            return None, _plan(
                request,
                CalendarOutcome.NO_CHANGE,
                "the event is already at that time",
                event=current,
                resolution_notes=when.notes,
            )
        return when, None

    async def _move_interview(
        self,
        request: CalendarRequest,
        user_id: UUID,
        now: datetime,
        event: CalendarEvent,
        when: ResolvedTime,
    ) -> tuple[CalendarChangePlan, list[ActionIntent]] | None:
        """A linked interview moves with its prep blocks. `None` for any other event."""
        if event.role is not CalendarEventRole.INTERVIEW or not event.application_id:
            return None
        try:
            application_id = UUID(event.application_id)
        except ValueError:
            return None
        if self.application_directory is not None:
            mine = {c.application_id for c in await self.application_directory.candidates(user_id)}
            if application_id not in mine:
                return _plan(
                    request,
                    CalendarOutcome.APPLICATION_NOT_FOUND,
                    f"the interview belongs to application {application_id}, which is not "
                    f"one of yours",
                    event=BusyEvent.of(event),
                ), []
        return await self._plan_interview(
            request,
            user_id,
            now,
            application_id=application_id,
            starts_at=when.starts_at,
            duration=when.duration,
            title=event.title or "Interview",
            resolved=when,
            current=event,
        )

    async def _plan_move(
        self, request: CalendarRequest, user_id: UUID, now: datetime
    ) -> tuple[CalendarChangePlan, list[ActionIntent]]:
        event, refused = await self._find_event(request, user_id, now)
        if refused is not None:
            return refused, []
        assert event is not None
        current = BusyEvent.of(event)
        if not event.stamped:
            return _plan(
                request,
                CalendarOutcome.NOT_MANAGED,
                f"'{event.title or event.event_id}' was not created by PersonalOS; it is "
                f"never moved or deleted here",
                event=current,
            ), []
        when, refused = self._move_time(request, event, now)
        if refused is not None:
            return refused, []
        assert when is not None
        interview = await self._move_interview(request, user_id, now, event, when)
        if interview is not None:
            return interview
        starts_at, ends_at = when.starts_at, when.ends_at

        slot = TimeSlot.of(starts_at, ends_at, self.zone)
        conflict = await self._conflict(user_id, starts_at, ends_at, exclude=[event])
        if conflict is not None:
            return _plan(
                request,
                CalendarOutcome.CONFLICT,
                f"{slot.describe()} is already booked; '{event.title}' was not moved",
                slot=slot,
                conflict=conflict,
                event=current,
                resolution_notes=when.notes,
            ), []
        role = event.role or CalendarEventRole.EVENT
        change = ScheduleChange(
            op=ScheduleOp.UPDATE,
            logical_key=event.logical_key or f"event:{event.event_id}",
            role=role,
            title=event.title,
            starts_at=starts_at,
            ends_at=ends_at,
            event_id=event.event_id,
            etag=event.etag,
            previous_starts_at=event.starts_at,
            previous_ends_at=event.ends_at,
        )
        intent = calendar_write_intent(
            kind=ActionKind.UPDATE_CALENDAR_EVENT,
            title=event.title or "Event",
            starts_at=starts_at,
            ends_at=ends_at,
            zone=self.zone,
            requested_by=requester(PLAN_CALENDAR_CHANGE),
            idempotency_key=calendar_update_key(change),
            logical_key=change.logical_key,
            role=role,
            now=now,
            attendees=event.attendees,
            event_id=event.event_id,
            previous=(event.starts_at, event.ends_at),
            resolved=when,
        )
        return _plan(
            request,
            CalendarOutcome.PROPOSED,
            f"move '{event.title}' to {slot.describe()}, once approved",
            slot=slot,
            event=current,
            resolution_notes=when.notes,
        ), [intent]

    async def _plan_cancel(
        self, request: CalendarRequest, user_id: UUID, now: datetime
    ) -> tuple[CalendarChangePlan, list[ActionIntent]]:
        event, refused = await self._find_event(request, user_id, now)
        if refused is not None:
            return refused, []
        assert event is not None
        current = BusyEvent.of(event)
        if not event.stamped:
            return _plan(
                request,
                CalendarOutcome.NOT_MANAGED,
                f"'{event.title or event.event_id}' was not created by PersonalOS; it is "
                f"never moved or deleted here",
                event=current,
            ), []
        targets = [event]
        application_id: UUID | None = None
        if event.role is CalendarEventRole.INTERVIEW and event.application_id:
            try:
                application_id = UUID(event.application_id)
            except ValueError:
                application_id = None
            if application_id is not None:
                targets.extend(
                    e
                    for e in await self.calendar.events_for_application(user_id, application_id)
                    if e.stamped
                    and e.role is CalendarEventRole.PREP
                    and e.properties.get(PROP_INTERVIEW_KEY) == event.logical_key
                    and e.ends_at > now
                )
        intents = [
            calendar_write_intent(
                kind=ActionKind.CANCEL_CALENDAR_EVENT,
                title=target.title or "Event",
                starts_at=target.starts_at,
                ends_at=target.ends_at,
                zone=self.zone,
                requested_by=requester(PLAN_CALENDAR_CHANGE),
                idempotency_key=calendar_cancel_key(target),
                logical_key=target.logical_key,
                role=target.role,
                now=now,
                attendees=target.attendees,
                event_id=target.event_id,
                application_id=application_id,
            )
            for target in targets
        ]
        extra = f" and {len(targets) - 1} prep block(s)" if len(targets) > 1 else ""
        return _plan(
            request,
            CalendarOutcome.PROPOSED,
            f"cancel '{event.title}'{extra}, once approved",
            slot=TimeSlot.of(event.starts_at, event.ends_at, self.zone),
            event=current,
            application_id=application_id,
            resolution_notes=(),
        ), intents

    # --- approval ------------------------------------------------------------

    def request_calendar_approval(self, state: CalendarState) -> dict[str, Any]:
        """Mint one `ApprovalRequest` per write, before the interrupt checkpoints them.

        Each request carries the intent's payload: title, attendees, start and
        end with the IANA zone, the resolution notes, and for a move the
        previous times and what changes.
        """
        now = self.clock()
        requests = [
            ApprovalRequest.for_intent(intent, now=now, ttl=self.approval_ttl)
            for intent in (
                ActionIntent.model_validate(raw) for raw in state.get("pending_actions") or ()
            )
        ]
        for request in requests:
            logger.info(
                "approval requested for %s (%s); expires %s",
                request.action_id,
                request.kind.value,
                request.expires_at.isoformat(),
            )
        return {"approval_requests": [*(state.get("approval_requests") or []), *_dump(requests)]}

    def calendar_approval_checkpoint(self, state: CalendarState) -> dict[str, Any]:
        """Park the run until a person answers. No calendar write happens without that answer."""
        intents = [ActionIntent.model_validate(raw) for raw in state.get("pending_actions") or ()]
        if not intents:
            return {}
        requests = {
            request.action_id: request
            for request in (
                ApprovalRequest.model_validate(raw) for raw in state.get("approval_requests") or ()
            )
        }
        outstanding = [requests[i.action_id] for i in intents if i.action_id in requests]
        if len(outstanding) != len(intents):
            raise JobSearchContractError(
                "a calendar write reached the approval checkpoint with no approval request on file"
            )
        for request in outstanding:
            logger.info(
                "calendar write %s awaits approval",
                request.action_id,
                extra={
                    "event": LogEvent.APPROVAL_REQUESTED,
                    "action_id": request.action_id,
                    "approval_id": request.request_id,
                },
            )
        resumed = interrupt(
            {
                "workflow": GRAPH_NAME,
                "stage": state.get("approval_stage") or STAGE_CALENDAR_CHANGE,
                "requests": [request.model_dump(mode="json") for request in outstanding],
            }
        )
        decisions = _decisions_from_resume(resumed)
        return {"approvals": [*(state.get("approvals") or []), *_dump(decisions)]}

    async def execute_calendar_changes(self, state: CalendarState) -> dict[str, Any]:
        """Make the approved writes, and only those.

        `authorize_execution` checks the request, the decision, its expiry and
        the recomputed hash; anything else is an `ApprovalRefusal` and the
        executor is not called. The executor (under `ToolExecutor`) reconciles
        an attempt with no recorded outcome against the calendar before it
        writes again, and refuses any event without the PersonalOS stamp.
        """
        requests = {
            r.action_id: r
            for r in (
                ApprovalRequest.model_validate(raw) for raw in state.get("approval_requests") or ()
            )
        }
        decisions = {
            d.action_id: d
            for d in (ApprovalDecision.model_validate(raw) for raw in state.get("approvals") or ())
        }
        refusals = list(state.get("approval_refusals") or [])
        receipts: list[ActionReceipt] = []
        for raw in state.get("pending_actions") or ():
            intent = ActionIntent.model_validate(raw)
            refusal = authorize_execution(
                intent=intent,
                request=requests.get(intent.action_id),
                decision=decisions.get(intent.action_id),
                now=self.clock(),
            )
            if refusal is not None:
                (logger.warning if refusal.suspicious else logger.info)(
                    "not writing %s to the calendar: %s", intent.action_id, refusal.detail
                )
                refusals.append(refusal.model_dump(mode="json"))
                receipts.append(refusal.to_receipt())
                continue
            if self.action_executor is None:
                raise JobSearchContractError(
                    "execute_calendar_changes reached without an action executor wired"
                )
            receipts.append(await self.action_executor.execute(intent, decisions[intent.action_id]))
        return {"action_receipts": _dump(receipts), "approval_refusals": refusals}

    # ------------------------------------------------------------------
    # Routing: reads the parsed request and the nodes' conclusions
    # ------------------------------------------------------------------

    def route_after_parse(self, state: CalendarState) -> str:
        request = _load_request(state)
        return (
            PLAN_CALENDAR_CHANGE if request is not None and request.is_write else CHECK_AVAILABILITY
        )

    def route_after_plan(self, state: CalendarState) -> str:
        return REQUEST_CALENDAR_APPROVAL if state.get("pending_actions") else END


def _naive_utc_of(wall: datetime, zone: ZoneInfo) -> datetime:
    local, _ = localize(wall, zone)
    return local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def _decisions_from_resume(resumed: Any) -> list[ApprovalDecision]:
    """The reviewer's decisions, from whatever the interrupt was resumed with."""
    if resumed is None:
        return []
    if isinstance(resumed, ApprovalDecision):
        return [resumed]
    if isinstance(resumed, dict):
        if "action_id" in resumed:
            return [ApprovalDecision.model_validate(resumed)]
        return _decisions_from_resume(list(resumed.values()))
    if isinstance(resumed, Sequence) and not isinstance(resumed, str | bytes):
        return [d for item in resumed for d in _decisions_from_resume(item)]
    raise CalendarContractError(
        f"cannot read an approval decision out of a resume value of type {type(resumed).__name__}"
    )


# --- Supervisor adapter --------------------------------------------------------------


class CalendarSubgraphRunner:
    """Satisfies the Supervisor's `CalendarSubgraphRunner` port with this subgraph.

    Runs on its own stable thread (`calendar_thread_id`), continued rather than
    restarted while it has steps left -- a write parked at its approval, say.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        user_id: UUID | None = None,
        thread_id: str | None = None,
        workflow_id: UUID | None = None,
    ):
        self.graph = graph
        self.user_id = user_id
        self.thread_id = thread_id or calendar_thread_id(user_id or "anonymous")
        self.workflow_id = workflow_id

    async def __call__(self, task_dag: dict[str, Any], state: Any) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": self.thread_id}
        if self.workflow_id is not None:
            configurable["workflow_id"] = str(self.workflow_id)
        config = {"configurable": configurable}
        stored = await self.graph.aget_state(config)
        initial: CalendarState | None = None
        if not stored.next:
            initial = {
                "request": (state or {}).get("message", "") if hasattr(state, "get") else "",
                "user_id": str(self.user_id) if self.user_id else None,
            }
        else:
            logger.info(
                "calendar thread '%s' has %s still to run; continuing it",
                self.thread_id,
                ", ".join(stored.next),
            )
        final = await self.graph.ainvoke(initial, config=config)
        return calendar_result(task_dag, final)


def calendar_result(task_dag: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """What the Supervisor records of a Calendar task."""
    request = final.get("calendar_request") or {}
    return {
        "goal": task_dag.get("goal"),
        "domain": "calendar",
        "task": request.get("task"),
        "time_zone": request.get("time_zone"),
        "when": request.get("when"),
        "availability": final.get("availability") or [],
        "proposals": final.get("proposals") or [],
        "change_plan": final.get("change_plan"),
        "awaiting_approval": bool(final.get("__interrupt__")),
        "action_receipts": final.get("action_receipts") or [],
        "errors": final.get("errors") or [],
    }


__all__ = [
    "CALENDAR_APPROVAL_CHECKPOINT",
    "CHECK_AVAILABILITY",
    "EXECUTE_CALENDAR_CHANGES",
    "GRAPH_NAME",
    "PARSE_REQUEST",
    "PLAN_CALENDAR_CHANGE",
    "READ_CALENDAR",
    "REQUEST_CALENDAR_APPROVAL",
    "STAGE_CALENDAR_CHANGE",
    "CalendarAvailabilityReader",
    "CalendarContractError",
    "CalendarGraph",
    "CalendarState",
    "CalendarSubgraphRunner",
    "calendar_result",
    "requester",
]
