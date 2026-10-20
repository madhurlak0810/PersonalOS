"""The audit trail as a reader sees it: what happened, and what authorized it.

`GET /v1/audit` and `GET /v1/workflows/{id}/audit` answer with these values,
built by `personalos.persistence.audit_trail` from four tables that each hold
one part of an action's story:

- `policy_decisions` -- the action as it was proposed (tool, principal, the
  hash of its arguments, the scopes it asked for) and the verdict on it;
- `approvals` -- the human sign-off, bound to the same action hash;
- `tool_executions` -- the call itself, its status and the provider's receipt;
- `audit_events` -- the append-only record tying them together.

An `AuditTrailEntry` is one step of that story with the three rows it belongs
to attached under `links`, so a reader can walk from any step to the others
without a second request. Every value here has been through the redactor on
its way out of the database: the trail is append-only, so a credential that
got into a row cannot be taken out of it, but it can be kept out of a response.

Pure values, like the rest of `domain`; nothing here reads or writes.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personalos.domain.errors import ValidationFailed

DEFAULT_AUDIT_PAGE_SIZE = 50
MAX_AUDIT_PAGE_SIZE = 200


class AuditStage(str, Enum):
    """Where in an action's life an entry sits. Declared in the order they happen."""

    #: The action as first proposed, from the first policy decision on its hash.
    INTENT = "intent"
    #: A verdict the policy engine reached on the action.
    POLICY_DECISION = "policy_decision"
    #: A human decision on the action, or a refused attempt at one.
    APPROVAL = "approval"
    #: The call, with the provider's receipt on its `tool_execution` link.
    EXECUTION = "execution"
    #: Any other audit event recorded for the workflow.
    EVENT = "event"


#: Tie-break for entries recorded at the same instant: the order they happen in.
STAGE_ORDER: dict[AuditStage, int] = {stage: rank for rank, stage in enumerate(AuditStage)}


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LinkedPolicyDecision(_Value):
    """The `policy_decisions` row an entry was decided under."""

    id: UUID
    principal: str
    tool: str
    args_hash: str
    decision: str
    requested_scopes: list[str] = Field(default_factory=list)
    decided_at: datetime


class LinkedToolExecution(_Value):
    """The `tool_executions` row an entry belongs to, receipt included."""

    operation_id: UUID
    tool_name: str
    idempotency_key: str
    status: str
    request_fingerprint: str | None = None
    policy_decision_id: UUID | None = None
    approval_ref: str | None = None
    approved_by: str | None = None
    attempts: int
    #: The provider's receipt, redacted.
    receipt: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class LinkedApproval(_Value):
    """The `approvals` row an entry belongs to."""

    id: UUID
    status: str
    action_hash: str
    #: From the audit trail, the only place the decider is kept.
    decided_by: str | None = None
    requested_at: datetime
    decided_at: datetime | None = None


class AuditLinks(_Value):
    """The rest of the chain an entry belongs to; any of them may not exist (yet)."""

    policy_decision: LinkedPolicyDecision | None = None
    tool_execution: LinkedToolExecution | None = None
    approval: LinkedApproval | None = None


class AuditTrailEntry(_Value):
    """One step of what happened, with the rows that explain it."""

    stage: AuditStage
    at: datetime
    actor: str
    action: str
    target_ref: str | None = None
    #: `success` / `failure` for a recorded audit event; absent for the intent
    #: and policy-decision steps, which are read from `policy_decisions`.
    result: str | None = None
    #: The verdict in force for this step (allow / deny / require_approval).
    policy_decision: str | None = None
    #: The fingerprint of the action, shared by every step of its chain.
    action_hash: str | None = None
    workflow_id: UUID | None = None
    #: The `audit_events` row this entry is, when it is one.
    audit_event_id: UUID | None = None
    links: AuditLinks = Field(default_factory=AuditLinks)


class AuditQuery(_Value):
    """Which audit events `GET /v1/audit` returns, and which page of them."""

    workflow_id: UUID | None = None
    #: Events of the workflows run on the application's own thread.
    application_id: UUID | None = None
    actor: str | None = None
    #: Exact `audit_events.action`, e.g. `approval.APPROVED` or a tool name.
    action: str | None = None
    #: Inclusive bounds on the event's timestamp.
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=DEFAULT_AUDIT_PAGE_SIZE, ge=1, le=MAX_AUDIT_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "AuditQuery":
        if self.since and self.until and self.since > self.until:
            raise ValidationFailed("the time range ends before it starts")
        return self


class AuditPage(_Value):
    """One page of audit events, newest first."""

    entries: tuple[AuditTrailEntry, ...] = ()
    total: int
    limit: int
    offset: int

    @property
    def next_offset(self) -> int | None:
        """Where the next page starts; `None` on the last one."""
        following = self.offset + len(self.entries)
        return following if following < self.total else None


class WorkflowAuditTrail(_Value):
    """Everything recorded for one workflow, oldest first."""

    workflow_id: UUID
    entries: tuple[AuditTrailEntry, ...] = ()


__all__ = [
    "DEFAULT_AUDIT_PAGE_SIZE",
    "MAX_AUDIT_PAGE_SIZE",
    "STAGE_ORDER",
    "AuditLinks",
    "AuditPage",
    "AuditQuery",
    "AuditStage",
    "AuditTrailEntry",
    "LinkedApproval",
    "LinkedPolicyDecision",
    "LinkedToolExecution",
    "WorkflowAuditTrail",
]
