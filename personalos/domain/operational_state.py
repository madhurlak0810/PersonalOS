"""Operational state: what is queued, parked and stuck, as of one moment.

The counters and histograms in `personalos.observability.metrics` say what
*happened*. Some questions an operator asks are about what *is*: how deep the
command queue is, how many events are dead-lettered, which workflows have
stopped moving. Those are answered from the database, by
`personalos.persistence.operational_state.OperationalStateReader`, as one
`OperationalSnapshot`; `personalos.observability.state_metrics` publishes the
latest snapshot as gauges.

Only ids, counts, durations and code-defined names are in a snapshot -- never
a payload -- so nothing here can put user content on a metric.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from uuid import UUID

_EMPTY: Mapping[str, int] = MappingProxyType({})


class StallReason(str, Enum):
    """Why a workflow counts as stalled."""

    #: Unfinished, not waiting on anything, and nothing has been recorded for
    #: it for longer than the stall threshold (`WORKFLOW_STALL_SECONDS`).
    NO_PROGRESS = "no_progress"
    #: Parked at an approval whose expiry passed more than the grace period
    #: ago (`APPROVAL_OVERDUE_GRACE_SECONDS`): the expiry sweep should have
    #: closed it, so it is not running.
    APPROVAL_OVERDUE = "approval_overdue"


@dataclass(frozen=True)
class StalledWorkflow:
    """One workflow that has stopped moving."""

    workflow_id: UUID
    reason: StallReason
    #: Seconds since its last recorded activity (no progress), or since its
    #: approval expired (approval overdue).
    stalled_for_seconds: float


@dataclass(frozen=True)
class OperationalSnapshot:
    """Queue depth, backlog, parked events, stalled workflows and integration health."""

    taken_at: datetime
    #: Workflow commands not yet settled, by outbox status (`pending`, `in_progress`).
    queue_depth: Mapping[str, int] = field(default_factory=lambda: _EMPTY)
    #: Other outbox events not yet delivered, by status (`pending`, `in_progress`).
    outbox_backlog: Mapping[str, int] = field(default_factory=lambda: _EMPTY)
    #: Age in seconds of the oldest undelivered (non-command) outbox event; 0 if none.
    oldest_pending_age_seconds: float = 0.0
    #: Outbox rows parked for an operator, by status (`dead_letter`, `quarantined`).
    parked: Mapping[str, int] = field(default_factory=lambda: _EMPTY)
    #: Workflows with a thread running or a command queued.
    active_workflows: int = 0
    stalled: tuple[StalledWorkflow, ...] = ()
    #: Stored health status of each integration (`healthy`, `degraded`, `down`).
    integrations: Mapping[str, str] = field(default_factory=dict)

    def stalled_count(self, reason: StallReason) -> int:
        """How many stalled workflows have `reason`."""
        return sum(1 for item in self.stalled if item.reason is reason)


__all__ = ["OperationalSnapshot", "StallReason", "StalledWorkflow"]
