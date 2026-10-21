"""Workflow progress as a watcher sees it: what just happened, in order.

`GET /v1/workflows/{id}/events` streams these. They are not a second record
of the run kept beside the first: each one is a row of `event_log` -- the
durable, append-only domain event history -- under the workflow's own
aggregate, written by the worker that ran the step. A client watching live
and a client that connects a day later read the same rows, which is what
makes a reconnect with `Last-Event-ID` exact: the id *is* the row, and
"everything after it" is a query, not a buffer some process had to keep.

Six kinds, matching the moments a long run (discovery, scoring, drafting) is
worth watching at:

- `step_started` / `step_completed` -- a graph node began / finished;
- `approval_requested` -- the run parked on an interrupt, waiting on a human;
- `approval_resolved` -- the answer to that interrupt was picked up;
- `failure` -- a step raised; its checkpoint is intact and
  `GET /v1/workflows/{id}/failures` has the rest;
- `workflow_completed` -- a thread of the workflow ran to its end.

**Payloads are minimal on purpose.** An event says *that* something happened
and where, never the state it happened to: ids, step names, verdicts, a failure
class and a redacted one-line message. An approval request's target, summary
and preview stay in the checkpoint, behind the approvals endpoints. What does
go in passes the same redactor the audit trail does, once on the way into the
log and again on the way out.

Pure values, like the rest of `domain`; nothing here reads or writes.
"""

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from personalos.domain.errors import ValidationFailed

#: `event_log.aggregate_type` for a workflow's progress events.
WORKFLOW_AGGREGATE = "workflow"

#: Prefix of `event_log.event_type` for a progress event, after the
#: `application.` convention the board uses: `workflow.step_started`, ...
_EVENT_TYPE_PREFIX = "workflow."

#: How many events one read of the log returns at most. A reconnecting client
#: that missed more than this gets them over successive reads, never truncated.
DEFAULT_PROGRESS_BATCH = 500


class ProgressEventType(str, Enum):
    """What happened. Wire-visible: it is the SSE `event:` field."""

    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    FAILURE = "failure"
    WORKFLOW_COMPLETED = "workflow_completed"

    @property
    def stored(self) -> str:
        """The `event_log.event_type` this kind is recorded under."""
        return f"{_EVENT_TYPE_PREFIX}{self.value}"

    @classmethod
    def from_stored(cls, event_type: str) -> "ProgressEventType | None":
        """The kind an `event_log.event_type` names, or `None` for any other event."""
        if not event_type.startswith(_EVENT_TYPE_PREFIX):
            return None
        try:
            return cls(event_type[len(_EVENT_TYPE_PREFIX) :])
        except ValueError:
            return None


#: Every `event_log.event_type` a progress stream reads.
STORED_PROGRESS_TYPES: tuple[str, ...] = tuple(kind.stored for kind in ProgressEventType)


class InvalidEventCursor(ValidationFailed):
    """A `Last-Event-ID` that does not name an event of this workflow.

    Refused rather than read as "from the beginning": replaying everything to
    a client that believes it is resuming is exactly the duplication the
    cursor exists to prevent.
    """


class WorkflowProgressEvent(BaseModel):
    """One thing that happened in a workflow, as one `event_log` row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The `event_log` row id; the SSE `id:` a client resumes from.
    event_id: UUID
    type: ProgressEventType
    workflow_id: UUID
    #: The request that queued the run this happened in.
    correlation_id: UUID | None = None
    thread_id: str | None = None
    at: datetime
    #: The kind-specific, minimal, redacted details.
    data: dict[str, Any] = Field(default_factory=dict)


# --- Payload builders -----------------------------------------------------------
#
# What each kind carries beyond the envelope. Built from whatever the runner
# has in hand -- an interrupt's raw value, a resume value -- and keeping only
# the fields a watcher needs to tell one request or decision from another.

#: The fields of an `ApprovalRequest` a progress event may carry. Not
#: `target`, `summary` or `preview`: those are about the action's content, and
#: are read through the approvals endpoints.
_REQUEST_FIELDS = ("request_id", "action_id", "kind", "risk", "expires_at")

#: The fields of an `ApprovalDecision` a progress event may carry.
_DECISION_FIELDS = ("request_id", "action_id", "verdict")


def approval_requested_data(
    *, step: str | None, interrupt_id: str | None, value: Any
) -> dict[str, Any]:
    """What an `approval_requested` event says about one interrupt."""
    data: dict[str, Any] = {"step": step, "interrupt_id": interrupt_id}
    if isinstance(value, Mapping):
        if isinstance(value.get("stage"), str):
            data["stage"] = value["stage"]
        requests = value.get("requests")
        if isinstance(requests, list | tuple):
            data["requests"] = [_pick(request, _REQUEST_FIELDS) for request in requests]
    return data


def approval_resolved_data(
    *,
    steps: tuple[str, ...],
    interrupt_ids: tuple[str, ...],
    resume_value: Any,
    actor_id: str | None,
) -> dict[str, Any]:
    """What an `approval_resolved` event says about the answer being picked up."""
    data: dict[str, Any] = {
        "steps": list(steps),
        "interrupt_ids": list(interrupt_ids),
        "resolved_by": actor_id,
    }
    if isinstance(resume_value, list | tuple) and all(
        isinstance(item, Mapping) and "verdict" in item for item in resume_value
    ):
        data["decisions"] = [_pick(item, _DECISION_FIELDS) for item in resume_value]
    return data


def _pick(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {field: _scalar(value[field]) for field in fields if field in value}


def _scalar(value: Any) -> Any:
    """A JSON-safe scalar: enums by value, everything not already plain as text."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = [
    "DEFAULT_PROGRESS_BATCH",
    "STORED_PROGRESS_TYPES",
    "WORKFLOW_AGGREGATE",
    "InvalidEventCursor",
    "ProgressEventType",
    "WorkflowProgressEvent",
    "approval_requested_data",
    "approval_resolved_data",
]
