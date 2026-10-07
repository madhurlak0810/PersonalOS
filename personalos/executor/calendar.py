"""Calendar writes that cannot duplicate an event on a retry.

A calendar insert is not idempotent at the provider: sent twice, it makes two
events. And the case that sends it twice is an ordinary one -- the insert
lands, the response is lost to a timeout, and whoever is retrying has no event
id to show for it.

So every event this module creates is stamped, in its private extended
properties, with the idempotency key of the write that made it
(`PROP_IDEMPOTENCY_KEY`), and both halves below look for that stamp before
anything is written:

- `CalendarActionExecutor` is the provider adapter for
  `CREATE_CALENDAR_EVENT` / `UPDATE_CALENDAR_EVENT`. A create first asks the
  calendar for an event carrying this intent's key and returns it if there is
  one. An update reads the event by id and writes only if it is not already
  where the intent puts it.
- `CalendarReconciler` is the `ProviderReconciler` that
  `personalos.executor.tool_executor.ToolExecutor` consults when a previous
  attempt left no recorded outcome. It answers `APPLIED` (with the event id)
  or `NOT_APPLIED` from the same lookups, and it is that answer -- not the
  absence of a receipt -- that permits a second call.

`CalendarClient` is the port to the provider. Nothing here imports one.
"""

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from personalos.domain.interview_scheduling import (
    PAYLOAD_ENDS_AT,
    PAYLOAD_EVENT_ID,
    PAYLOAD_PROPERTIES,
    PAYLOAD_STARTS_AT,
    PAYLOAD_TITLE,
    PROP_IDEMPOTENCY_KEY,
    CalendarEvent,
)
from personalos.domain.job_search import (
    PAYLOAD_BODY,
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    JobSearchContractError,
)
from personalos.executor.tool_executor import (
    ActionExecutorPort,
    ReconcileOutcome,
    Reconciliation,
)

logger = logging.getLogger(__name__)

CALENDAR_ACTION_KINDS = frozenset(
    {ActionKind.CREATE_CALENDAR_EVENT, ActionKind.UPDATE_CALENDAR_EVENT}
)


class CalendarClient(Protocol):
    """The calendar provider, as far as this module needs it."""

    async def find_by_property(self, name: str, value: str) -> Sequence[CalendarEvent]:
        """Events whose private extended property `name` equals `value`."""
        ...

    async def get_event(self, event_id: str) -> CalendarEvent | None:
        """One event by id, or `None` if the calendar has no such event."""
        ...

    async def create_event(
        self,
        *,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        description: str,
        properties: dict[str, str],
    ) -> CalendarEvent:
        """Insert an event and return it as stored."""
        ...

    async def update_event(
        self,
        event_id: str,
        *,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        properties: dict[str, str],
    ) -> CalendarEvent:
        """Move an existing event, merging `properties` into its own."""
        ...


def _times(intent: ActionIntent) -> tuple[datetime, datetime]:
    try:
        return (
            datetime.fromisoformat(intent.payload[PAYLOAD_STARTS_AT]),
            datetime.fromisoformat(intent.payload[PAYLOAD_ENDS_AT]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JobSearchContractError(
            f"calendar action {intent.action_id} does not carry a start and end time"
        ) from exc


def _event_id(intent: ActionIntent) -> str:
    event_id = intent.payload.get(PAYLOAD_EVENT_ID)
    if not event_id:
        raise JobSearchContractError(
            f"calendar update {intent.action_id} does not name the event it moves"
        )
    return str(event_id)


def _receipt(intent: ActionIntent, event: CalendarEvent, detail: str) -> ActionReceipt:
    return ActionReceipt(
        action_id=intent.action_id, ok=True, external_reference=event.event_id, detail=detail
    )


async def _created_by(client: CalendarClient, intent: ActionIntent) -> CalendarEvent | None:
    """The event an earlier attempt at this create left behind, if any."""
    found = list(await client.find_by_property(PROP_IDEMPOTENCY_KEY, intent.idempotency_key))
    if len(found) > 1:
        logger.warning(
            "%s calendar events carry idempotency key '%s'; using %s",
            len(found),
            intent.idempotency_key,
            found[0].event_id,
        )
    return found[0] if found else None


def _already_at(event: CalendarEvent, intent: ActionIntent) -> bool:
    return (event.starts_at, event.ends_at) == _times(intent)


class CalendarActionExecutor:
    """Performs approved calendar writes, looking before each one."""

    def __init__(self, client: CalendarClient, *, fallback: ActionExecutorPort | None = None):
        """Wrap the calendar client. Other action kinds go to `fallback`."""
        if client is None:
            raise ValueError("CalendarActionExecutor requires a CalendarClient")
        self.client = client
        self.fallback = fallback

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Create or move the event the intent describes, unless it is already so."""
        if intent.kind is ActionKind.CREATE_CALENDAR_EVENT:
            return await self._create(intent)
        if intent.kind is ActionKind.UPDATE_CALENDAR_EVENT:
            return await self._update(intent)
        if self.fallback is None:
            raise JobSearchContractError(
                f"CalendarActionExecutor cannot execute '{intent.kind.value}' and has no "
                f"fallback executor"
            )
        return await self.fallback.execute(intent, decision)

    async def _create(self, intent: ActionIntent) -> ActionReceipt:
        existing = await _created_by(self.client, intent)
        if existing is not None:
            logger.info(
                "calendar event %s already exists for idempotency key '%s'; not creating "
                "another",
                existing.event_id,
                intent.idempotency_key,
            )
            return _receipt(intent, existing, "already on the calendar; not created again")

        starts_at, ends_at = _times(intent)
        event = await self.client.create_event(
            title=str(intent.payload.get(PAYLOAD_TITLE) or intent.summary),
            starts_at=starts_at,
            ends_at=ends_at,
            description=str(intent.payload.get(PAYLOAD_BODY) or ""),
            properties={
                **dict(intent.payload.get(PAYLOAD_PROPERTIES) or {}),
                PROP_IDEMPOTENCY_KEY: intent.idempotency_key,
            },
        )
        return _receipt(intent, event, "created")

    async def _update(self, intent: ActionIntent) -> ActionReceipt:
        event_id = _event_id(intent)
        current = await self.client.get_event(event_id)
        if current is None:
            return ActionReceipt(
                action_id=intent.action_id,
                ok=False,
                detail=f"calendar event {event_id} no longer exists; nothing was moved",
            )
        if _already_at(current, intent):
            return _receipt(intent, current, "already at the requested time; not updated again")

        starts_at, ends_at = _times(intent)
        event = await self.client.update_event(
            event_id,
            title=str(intent.payload.get(PAYLOAD_TITLE) or current.title),
            starts_at=starts_at,
            ends_at=ends_at,
            properties={PROP_IDEMPOTENCY_KEY: intent.idempotency_key},
        )
        return _receipt(intent, event, "moved")


class CalendarReconciler:
    """Tells `ToolExecutor` whether an unrecorded calendar write took effect.

    A lookup that raises is left to raise: `ToolExecutor` treats that as "the
    provider could not be asked" and executes nothing.
    """

    def __init__(self, client: CalendarClient):
        """Wrap the same client the executor writes through."""
        self.client = client

    async def reconcile(self, intent: ActionIntent) -> Reconciliation:
        """Find the event by idempotency key (create) or by id (update)."""
        if intent.kind is ActionKind.CREATE_CALENDAR_EVENT:
            event = await _created_by(self.client, intent)
            if event is None:
                return Reconciliation(ReconcileOutcome.NOT_APPLIED)
            return Reconciliation(
                ReconcileOutcome.APPLIED,
                _receipt(intent, event, "found on the calendar; not created again"),
            )

        if intent.kind is ActionKind.UPDATE_CALENDAR_EVENT:
            event = await self.client.get_event(_event_id(intent))
            if event is not None and _already_at(event, intent):
                return Reconciliation(
                    ReconcileOutcome.APPLIED,
                    _receipt(intent, event, "found at the requested time; not updated again"),
                )
            # Not there, or still where it was: the executor settles both.
            return Reconciliation(ReconcileOutcome.NOT_APPLIED)

        return Reconciliation(ReconcileOutcome.INDETERMINATE)


__all__ = [
    "CALENDAR_ACTION_KINDS",
    "CalendarClient",
    "CalendarActionExecutor",
    "CalendarReconciler",
]
