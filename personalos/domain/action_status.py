"""What the system suggested, and what it actually did.

Every outward-facing action a workflow touches is reported with exactly one
`ActionStatus`, and the status is derived here -- once, from the stored facts
-- rather than by whichever surface happens to be showing the action:

- `RECOMMENDED` -- proposed. Nothing has happened outside this system, and
  nobody has been asked about it yet.
- `AWAITING_APPROVAL` -- put to a reviewer and not yet carried out. Still
  nothing has happened outside this system. An approval that has been given
  but not yet redeemed stays here: approving is not doing.
- `EXECUTED` -- carried out, with the receipt that says so.
- `DENIED` -- a reviewer said no. Nothing happened.
- `FAILED` -- it was meant to happen and did not: the executor refused it (an
  expired approval, an action that changed after it was approved) or the call
  itself failed.

**Only a receipt makes an action `EXECUTED`.** Not a draft, not an approval,
not a finished workflow: `WorkflowAction` refuses to be built as `EXECUTED`
without a successful `ActionReceipt`, and refuses to carry a successful
receipt under any other status. A drafted email is therefore `RECOMMENDED` or
`AWAITING_APPROVAL` until the send itself returns, and there is no way to
construct one that reads as sent.

**A shortlist entry is a recommendation.** `ShortlistItem` is `RECOMMENDED`
unless a submission action exists for that posting, in which case it reports
that action's status -- so an entry is only ever shown as submitted when the
submission has a receipt.

Pure values and a pure derivation, like the rest of `domain`: the reads that
feed `derive_thread_actions` live in `personalos.persistence.workflow_actions`.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionPreview,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    ApprovalVerdict,
    RefusalReason,
    ShortlistEntry,
)
from personalos.domain.workflow import ApprovalState


class ActionStatus(str, Enum):
    """Where one action stands, from the outside world's point of view. Wire-visible."""

    RECOMMENDED = "RECOMMENDED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTED = "EXECUTED"
    DENIED = "DENIED"
    FAILED = "FAILED"

    @property
    def happened(self) -> bool:
        """True only for the one status that means the outside world changed."""
        return self is ActionStatus.EXECUTED


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionLinks(_Value):
    """Where to read more about an action: its execution row and its audit trail."""

    #: The `tool_executions` row the call was recorded under, once there is one.
    operation_id: UUID | None = None
    #: That row's status, verbatim (`in_progress`, `completed`, `failed`, ...).
    tool_execution_status: str | None = None
    #: The `audit_events` rows recorded for that call.
    audit_event_ids: tuple[UUID, ...] = ()
    #: The workflow's audit trail, where every step of this action is listed.
    audit_url: str | None = None


class WorkflowAction(_Value):
    """One outward-facing action, labelled with what actually became of it.

    The shape every surface lists an action in. `action_status` is required --
    there is no default -- so an action object that does not say whether it
    happened is not a valid one.
    """

    action_id: UUID
    action_status: ActionStatus
    workflow_id: UUID | None = None
    thread_id: str | None = None
    kind: ActionKind
    target: str
    summary: str
    risk_level: str | None = None
    #: Recipient, subject, body diff, attachments, and reply-or-new-thread.
    preview: ActionPreview | None = None
    #: What carrying the action out would do, one plain sentence each.
    consequences: tuple[str, ...] = ()
    #: The approval request put to a reviewer, once there is one.
    approval_id: UUID | None = None
    approval_state: ApprovalState | None = None
    approval_expires_at: datetime | None = None
    decided_by: str | None = None
    #: The provider's answer. Present on `EXECUTED`; may be present, not ok, on
    #: `FAILED` or `DENIED`.
    receipt: ActionReceipt | None = None
    #: Why a `DENIED` or `FAILED` action did not happen.
    refusal_reason: RefusalReason | None = None
    #: One line on where the action stands.
    detail: str | None = None
    proposed_at: datetime | None = None
    links: ActionLinks = Field(default_factory=ActionLinks)

    @model_validator(mode="after")
    def _executed_means_receipted(self) -> "WorkflowAction":
        succeeded = self.receipt is not None and self.receipt.ok
        if self.action_status is ActionStatus.EXECUTED and not succeeded:
            raise ValueError(
                f"action {self.action_id} cannot be EXECUTED without a successful receipt"
            )
        if succeeded and self.action_status is not ActionStatus.EXECUTED:
            raise ValueError(
                f"action {self.action_id} has a successful receipt but is labelled "
                f"{self.action_status.value}"
            )
        return self


class ShortlistItem(_Value):
    """One posting a run recommended, and whether anything was done about it."""

    rank: int
    #: `RECOMMENDED` unless `action_id` names a submission; then its status.
    action_status: ActionStatus
    dedupe_key: str
    title: str
    company: str
    url: str | None = None
    score: float
    reasons: tuple[str, ...] = ()
    thread_id: str | None = None
    #: The submission proposed for this posting, when there is one.
    action_id: UUID | None = None

    @model_validator(mode="after")
    def _only_an_action_moves_it(self) -> "ShortlistItem":
        if self.action_status is not ActionStatus.RECOMMENDED and self.action_id is None:
            raise ValueError(
                f"shortlist entry '{self.dedupe_key}' is {self.action_status.value} with no "
                f"submission action behind it; a shortlist entry on its own is a recommendation"
            )
        return self


class WorkflowActions(_Value):
    """Every action a workflow touched, and the postings it recommended."""

    workflow_id: UUID
    actions: tuple[WorkflowAction, ...] = ()
    shortlist: tuple[ShortlistItem, ...] = ()

    def with_status(self, status: ActionStatus) -> tuple[WorkflowAction, ...]:
        """The actions in one status, in the order they were proposed."""
        return tuple(action for action in self.actions if action.action_status is status)

    def counts(self) -> dict[ActionStatus, int]:
        """How many actions are in each status, every status listed."""
        return {status: len(self.with_status(status)) for status in ActionStatus}


# --- Derivation -----------------------------------------------------------------


class RecordedDecision(_Value):
    """A decision recorded through the approval queue, which state may not hold yet."""

    state: ApprovalState
    decided_by: str | None = None


def audit_url(workflow_id: UUID) -> str:
    """Where a workflow's audit trail is read."""
    return f"/v1/workflows/{workflow_id}/audit"


def consequences(kind: ActionKind, target: str, preview: ActionPreview | None) -> tuple[str, ...]:
    """What carrying an action out would do, as a reviewer should read it first."""
    recipient = preview.recipient if preview else target
    lines: list[str] = []
    if kind is ActionKind.SEND_RECRUITER_MESSAGE:
        thread = (
            f"as a reply in thread {preview.in_reply_to}"
            if preview and preview.is_reply
            else "as a new thread"
        )
        lines.append(f"Sends an email to {recipient} {thread}.")
        lines.append("It cannot be unsent; a correction would be a second message.")
    elif kind is ActionKind.SUBMIT_APPLICATION:
        lines.append(f"Submits an application to {recipient}.")
        lines.append("It cannot be withdrawn once the company has received it.")
    elif kind is ActionKind.OVERWRITE_DOCUMENT:
        lines.append(f"Replaces {recipient} with the proposed version.")
    elif kind is ActionKind.CREATE_CALENDAR_EVENT:
        lines.append(f"Adds an event to your calendar: {target}.")
    elif kind is ActionKind.UPDATE_CALENDAR_EVENT:
        lines.append(f"Moves an existing event on your calendar: {target}.")
    if preview is not None:
        if preview.subject:
            lines.append(f"Subject: {preview.subject}")
        if preview.attachments:
            names = ", ".join(item.name for item in preview.attachments)
            lines.append(f"Attaches {len(preview.attachments)} file(s): {names}.")
        elif kind in (ActionKind.SEND_RECRUITER_MESSAGE, ActionKind.SUBMIT_APPLICATION):
            lines.append("No attachments.")
    return tuple(lines)


def recommended_action(
    intent: ActionIntent, *, workflow_id: UUID | None = None, thread_id: str | None = None
) -> WorkflowAction:
    """An intent nobody has been asked about yet, labelled as the proposal it is."""
    preview = ActionPreview.for_intent(intent)
    return WorkflowAction(
        action_id=intent.action_id,
        action_status=ActionStatus.RECOMMENDED,
        workflow_id=workflow_id,
        thread_id=thread_id,
        kind=intent.kind,
        target=intent.target,
        summary=intent.summary,
        risk_level=intent.risk_profile().risk.value,
        preview=preview,
        consequences=consequences(intent.kind, intent.target, preview),
        detail="Proposed; nothing has been done.",
        proposed_at=intent.created_at,
        links=ActionLinks(audit_url=audit_url(workflow_id) if workflow_id else None),
    )


def derive_thread_actions(
    values: Mapping[str, Any],
    *,
    workflow_id: UUID,
    thread_id: str,
    recorded: Mapping[UUID, RecordedDecision] | None = None,
    links: Mapping[str, ActionLinks] | None = None,
    now: datetime,
) -> tuple[WorkflowAction, ...]:
    """Label every action one thread's state holds with what became of it.

    `values` is the thread's checkpointed state; only the action keys of
    `JobSearchState` are read. `recorded` is the approval queue's decisions by
    request id -- a decision made through the API is on record before the
    resumed run writes it to state. `links` is execution rows by idempotency
    key.

    The status is decided by the most final fact on record, in this order: a
    receipt, a refusal, a decision, a request -- and, failing all of those, the
    intent alone, which is a recommendation.
    """
    recorded = recorded or {}
    links = links or {}
    intents = {i.action_id: i for i in _load(ActionIntent, values.get("pending_actions"))}
    requests = {r.action_id: r for r in _load(ApprovalRequest, values.get("approval_requests"))}
    decisions = {d.action_id: d for d in _load(ApprovalDecision, values.get("approvals"))}
    refusals = {r.action_id: r for r in _load(ApprovalRefusal, values.get("approval_refusals"))}
    receipts = {r.action_id: r for r in _load(ActionReceipt, values.get("action_receipts"))}

    order: dict[UUID, None] = {}
    for action_id in (*requests, *intents):
        order.setdefault(action_id, None)

    actions: list[WorkflowAction] = []
    for action_id in order:
        intent, request = intents.get(action_id), requests.get(action_id)
        kind = intent.kind if intent else request.kind
        target = intent.target if intent else request.target
        preview = (request.preview if request else None) or (
            ActionPreview.for_intent(intent) if intent else None
        )
        decision = decisions.get(action_id)
        record = recorded.get(request.request_id) if request else None
        key = intent.idempotency_key if intent else request.idempotency_key
        status, detail = _status(
            request=request,
            decision=decision,
            record=record,
            refusal=refusals.get(action_id),
            receipt=receipts.get(action_id),
            now=now,
        )
        found = links.get(key) or ActionLinks()
        actions.append(
            WorkflowAction(
                action_id=action_id,
                action_status=status,
                workflow_id=workflow_id,
                thread_id=thread_id,
                kind=kind,
                target=target,
                summary=intent.summary if intent else request.summary,
                risk_level=(request.risk.value if request else intent.risk_profile().risk.value),
                preview=preview,
                consequences=consequences(kind, target, preview),
                approval_id=request.request_id if request else None,
                approval_state=_approval_state(request, decision, record, now),
                approval_expires_at=request.expires_at if request else None,
                decided_by=(decision.decided_by if decision else None)
                or (record.decided_by if record else None),
                receipt=receipts.get(action_id),
                refusal_reason=(refusals[action_id].reason if action_id in refusals else None),
                detail=detail,
                proposed_at=intent.created_at if intent else request.requested_at,
                links=found.model_copy(update={"audit_url": audit_url(workflow_id)}),
            )
        )
    return tuple(actions)


def derive_shortlist(
    values: Mapping[str, Any], *, thread_id: str, actions: Sequence[WorkflowAction]
) -> tuple[ShortlistItem, ...]:
    """The thread's shortlist, each entry labelled by the submission made for it, if any."""
    submissions = _submissions_by_posting(values, actions)
    items: list[ShortlistItem] = []
    for entry in _load(ShortlistEntry, values.get("shortlist")):
        posting = entry.scored.posting
        action = submissions.get(posting.dedupe_key)
        items.append(
            ShortlistItem(
                rank=entry.rank,
                action_status=action.action_status if action else ActionStatus.RECOMMENDED,
                dedupe_key=posting.dedupe_key,
                title=posting.title,
                company=posting.company,
                url=posting.url,
                score=entry.scored.score,
                reasons=entry.scored.reasons,
                thread_id=thread_id,
                action_id=action.action_id if action else None,
            )
        )
    return tuple(items)


# --- Internals ------------------------------------------------------------------


def _status(
    *,
    request: ApprovalRequest | None,
    decision: ApprovalDecision | None,
    record: RecordedDecision | None,
    refusal: ApprovalRefusal | None,
    receipt: ActionReceipt | None,
    now: datetime,
) -> tuple[ActionStatus, str]:
    if receipt is not None and receipt.ok:
        reference = (
            f" (reference {receipt.external_reference})" if receipt.external_reference else ""
        )
        return ActionStatus.EXECUTED, f"Done{reference}."
    if refusal is not None:
        if refusal.reason is RefusalReason.DENIED_BY_USER:
            return ActionStatus.DENIED, "Denied by the reviewer; nothing was done."
        return ActionStatus.FAILED, f"Not carried out: {refusal.detail}"
    if receipt is not None:
        return ActionStatus.FAILED, f"Attempted and failed: {receipt.detail or 'no detail'}"
    if decision is not None and decision.verdict is ApprovalVerdict.REJECTED:
        return ActionStatus.DENIED, "Denied by the reviewer; nothing was done."
    if record is not None and record.state is ApprovalState.DENIED:
        return ActionStatus.DENIED, "Denied by the reviewer; nothing was done."
    if request is None:
        return ActionStatus.RECOMMENDED, "Proposed; nothing has been done."
    if (decision is not None and decision.verdict is ApprovalVerdict.APPROVED) or (
        record is not None and record.state is ApprovalState.APPROVED
    ):
        return (
            ActionStatus.AWAITING_APPROVAL,
            "Approved; not carried out yet. Nothing has been sent.",
        )
    if request.is_expired(now):
        return (
            ActionStatus.AWAITING_APPROVAL,
            "The approval window has closed; it can only be denied now.",
        )
    return ActionStatus.AWAITING_APPROVAL, "Waiting for your approval; nothing has been sent."


def _approval_state(
    request: ApprovalRequest | None,
    decision: ApprovalDecision | None,
    record: RecordedDecision | None,
    now: datetime,
) -> ApprovalState | None:
    if request is None:
        return None
    if decision is not None and decision.verdict is ApprovalVerdict.APPROVED:
        return ApprovalState.APPROVED
    if decision is not None and decision.verdict is ApprovalVerdict.REJECTED:
        return ApprovalState.DENIED
    if record is not None:
        return record.state
    return ApprovalState.EXPIRED if request.is_expired(now) else ApprovalState.PENDING


def _submissions_by_posting(
    values: Mapping[str, Any], actions: Sequence[WorkflowAction]
) -> dict[str, WorkflowAction]:
    """Submission actions keyed by the posting's dedupe key, from the intent payloads."""
    by_id = {action.action_id: action for action in actions}
    found: dict[str, WorkflowAction] = {}
    for intent in _load(ActionIntent, values.get("pending_actions")):
        dedupe_key = intent.payload.get("dedupe_key")
        action = by_id.get(intent.action_id)
        if intent.kind is ActionKind.SUBMIT_APPLICATION and dedupe_key and action:
            found[str(dedupe_key)] = action
    return found


def _load(model: type[_Value] | type[BaseModel], raws: Iterable[Any] | None) -> list[Any]:
    """Rebuild typed values from state, skipping any that no longer fit the model.

    A status read must not fail on one malformed row: the rest of the
    workflow's actions still have to be reported.
    """
    loaded = []
    for raw in raws or ():
        try:
            loaded.append(model.model_validate(raw))
        except ValidationError:
            continue
    return loaded


__all__ = [
    "ActionLinks",
    "ActionStatus",
    "RecordedDecision",
    "ShortlistItem",
    "WorkflowAction",
    "WorkflowActions",
    "audit_url",
    "consequences",
    "derive_shortlist",
    "derive_thread_actions",
    "recommended_action",
]
