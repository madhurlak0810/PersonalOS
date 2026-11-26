"""Reads the operational state the dashboards and alerts are built on, from the database.

`OperationalStateReader.snapshot` answers, as of one instant: how many workflow
commands are queued or running, how much of the outbox is undelivered and how
old its oldest event is, how many events are parked as dead letters or
quarantined, how many workflows are active, which of them have **stalled**, and
the stored health of each integration.

It reads rows only, so any process with the database can take a snapshot. The
outbox is read through the modules that own it -- `EventOutbox.backlog` and
`WorkflowCommandQueue.depth`/`unfinished_activity` -- so a second transport
(docs/EVENT_TRANSPORT.md) only has to answer those; the rest is
`workflow_runs`, `event_log`, `pending_checkpoints` and `integration_health`. The API does (`personalos.observability.state_metrics`):
a stall alert must still fire when the worker is the thing that stopped.

**What "stalled" means.** A workflow is *active* while one of its threads is
`pending` or `running`, or a command for it is queued or claimed. An active
workflow is then:

- **waiting on an approval** when its newest progress event is
  `approval_requested` and no command for it is queued -- nobody has
  answered yet, which is not a stall. It becomes **stalled
  (`approval_overdue`)** once the earliest expiry among the requests passed
  more than `approval_grace` ago: the worker's expiry sweep should have
  closed it, so that sweep is not running;
- **waiting on a follow-up** when it has a pending checkpoint that has not
  come due -- not a stall either;
- otherwise **stalled (`no_progress`)** when the newest of its progress
  events, its threads' status changes and its commands is older than
  `stall_after`. A command queued while no worker runs is this case: the
  command is the newest thing that happened, and nothing follows it.

A workflow that moves again -- an event, a status change, a fresh command --
is no longer stalled in the next snapshot, so an alert built on it resolves by
itself.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from personalos.domain.operational_state import (
    OperationalSnapshot,
    StalledWorkflow,
    StallReason,
)
from personalos.domain.progress import WORKFLOW_AGGREGATE, ProgressEventType
from personalos.persistence.event_outbox import EventOutbox
from personalos.persistence.models import (
    EventLogModel,
    IntegrationHealthModel,
    PendingCheckpointModel,
    WorkflowRunModel,
)
from personalos.persistence.workflow_commands import WorkflowCommandQueue

#: Thread statuses of a run someone believes is still going.
_ACTIVE_RUN_STATUSES = ("pending", "running")

#: At most this many active workflows are examined per snapshot. A
#: single-user deployment has a handful; the cap keeps a runaway bounded.
DEFAULT_MAX_WORKFLOWS = 500


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = (parsed - parsed.utcoffset()).replace(tzinfo=None)
    return parsed


def _earliest_expiry(payload: Any) -> datetime | None:
    """The earliest `expires_at` among the requests of an `approval_requested` event."""
    data = (payload or {}).get("data") if isinstance(payload, dict) else None
    requests = (data or {}).get("requests") if isinstance(data, dict) else None
    expiries = [
        parsed
        for request in requests or ()
        if isinstance(request, dict) and (parsed := _parse_time(request.get("expires_at")))
    ]
    return min(expiries) if expiries else None


class OperationalStateReader:
    """Takes `OperationalSnapshot`s of one database."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        outbox: EventOutbox,
        commands: WorkflowCommandQueue,
        stall_after: timedelta,
        approval_grace: timedelta,
        clock: Callable[[], datetime] = datetime.utcnow,
        max_workflows: int = DEFAULT_MAX_WORKFLOWS,
    ):
        """Wire the database, the outbox and command queue over it, and the stall thresholds.

        `outbox` should exclude workflow commands (`build_event_outbox` does),
        so its backlog is events only; `commands` counts those.
        """
        if stall_after <= timedelta(0):
            raise ValueError("stall_after must be positive")
        if approval_grace < timedelta(0):
            raise ValueError("approval_grace must not be negative")
        self.session_factory = session_factory
        self.outbox = outbox
        self.commands = commands
        self.stall_after = stall_after
        self.approval_grace = approval_grace
        self.clock = clock
        self.max_workflows = max_workflows

    def snapshot(self, *, now: datetime | None = None) -> OperationalSnapshot:
        """The state of the database as of `now`."""
        now = now or self.clock()
        queue_depth = self.commands.depth()
        backlog = self.outbox.backlog()
        commands = self.commands.unfinished_activity(limit=self.max_workflows)
        oldest = backlog.oldest_created_at
        session = self.session_factory()
        try:
            active = self._active_workflows(session, commands)
            stalled = self._stalled(session, active, commands, now)
            integrations = dict(
                session.query(IntegrationHealthModel.provider, IntegrationHealthModel.status).all()
            )
        finally:
            session.close()
        return OperationalSnapshot(
            taken_at=now,
            queue_depth=queue_depth,
            outbox_backlog=backlog.waiting,
            oldest_pending_age_seconds=max((now - oldest).total_seconds(), 0.0) if oldest else 0.0,
            parked=backlog.parked,
            active_workflows=len(active),
            stalled=tuple(stalled),
            integrations=integrations,
        )

    # ------------------------------------------------------------------
    # Stalls
    # ------------------------------------------------------------------

    def _active_workflows(
        self, session: Session, commands: dict[UUID, datetime]
    ) -> dict[UUID, datetime | None]:
        """Every active workflow, with its active threads' newest status change (or `None`).

        Active: a thread `pending` or `running`, or a command in `commands`.
        """
        active: dict[UUID, datetime | None] = {}
        for workflow_id, updated in (
            session.query(WorkflowRunModel.workflow_id, func.max(WorkflowRunModel.updated_at))
            .filter(WorkflowRunModel.status.in_(_ACTIVE_RUN_STATUSES))
            .group_by(WorkflowRunModel.workflow_id)
            .limit(self.max_workflows)
        ):
            active[workflow_id] = updated
        for workflow_id in commands:
            active.setdefault(workflow_id, None)
        return active

    def _stalled(
        self,
        session: Session,
        active: dict[UUID, datetime | None],
        commands: dict[UUID, datetime],
        now: datetime,
    ) -> list[StalledWorkflow]:
        if not active:
            return []
        ids = list(active)
        waiting_on_follow_up = {
            workflow_id
            for (workflow_id,) in session.query(PendingCheckpointModel.workflow_id)
            .filter(
                PendingCheckpointModel.workflow_id.in_(ids),
                PendingCheckpointModel.status == "pending",
                PendingCheckpointModel.trigger_at > now,
            )
            .distinct()
        }
        stalled: list[StalledWorkflow] = []
        for workflow_id in ids:
            latest = (
                session.query(
                    EventLogModel.event_type,
                    EventLogModel.occurred_at,
                    EventLogModel.payload_json,
                )
                .filter(
                    EventLogModel.aggregate_type == WORKFLOW_AGGREGATE,
                    EventLogModel.aggregate_id == workflow_id,
                )
                .order_by(EventLogModel.occurred_at.desc(), EventLogModel.id.desc())
                .first()
            )
            queued_at = commands.get(workflow_id)
            if (
                latest is not None
                and latest.event_type == ProgressEventType.APPROVAL_REQUESTED.stored
                and queued_at is None
            ):
                expiry = _earliest_expiry(latest.payload_json)
                if expiry is not None and now - expiry > self.approval_grace:
                    stalled.append(
                        StalledWorkflow(
                            workflow_id=workflow_id,
                            reason=StallReason.APPROVAL_OVERDUE,
                            stalled_for_seconds=(now - expiry).total_seconds(),
                        )
                    )
                continue
            if workflow_id in waiting_on_follow_up:
                continue
            moments = [
                t
                for t in (latest.occurred_at if latest else None, active[workflow_id], queued_at)
                if t is not None
            ]
            if not moments:
                continue
            idle = now - max(moments)
            if idle > self.stall_after:
                stalled.append(
                    StalledWorkflow(
                        workflow_id=workflow_id,
                        reason=StallReason.NO_PROGRESS,
                        stalled_for_seconds=idle.total_seconds(),
                    )
                )
        return stalled


__all__ = ["DEFAULT_MAX_WORKFLOWS", "OperationalStateReader"]
