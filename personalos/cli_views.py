"""Plain-text views of a workflow for the CLI: what was suggested, what was done.

Renders the same bodies the API returns (`WorkflowStatusResponse`,
`PendingApprovalItem`), so the terminal and the API can never disagree about
an action's status. The layout is the point:

- **Done** lists `EXECUTED` actions only, each with its receipt and audit link.
- **Waiting for your approval** and **Suggested** say, in their headings, that
  nothing has been sent; an approval shows the target and every material
  consequence -- recipient, subject, reply or new thread, attachments, the
  body diff -- before the reviewer says yes.
- **Denied** and **Failed** say why nothing happened.

Internal graph traffic -- interrupt payloads, node names beyond the plan and
completed steps -- is left out.
"""

from collections.abc import Iterable, Sequence

from apps.api.routes.approvals import PendingApprovalItem
from apps.api.routes.workflows import WorkflowStatusResponse
from personalos.domain.action_status import ActionStatus, ShortlistItem, WorkflowAction
from personalos.domain.job_search import ActionPreview

#: Section order and headings. Done comes first so it cannot be mistaken for
#: anything further down.
_SECTIONS: tuple[tuple[ActionStatus, str], ...] = (
    (ActionStatus.EXECUTED, "DONE -- executed, with receipts"),
    (ActionStatus.AWAITING_APPROVAL, "WAITING FOR YOUR APPROVAL -- nothing has been sent"),
    (ActionStatus.RECOMMENDED, "SUGGESTED -- not acted on"),
    (ActionStatus.DENIED, "DENIED -- nothing was done"),
    (ActionStatus.FAILED, "FAILED -- did not happen"),
)

_MARK = {
    ActionStatus.EXECUTED: "[x]",
    ActionStatus.AWAITING_APPROVAL: "[?]",
    ActionStatus.RECOMMENDED: "[ ]",
    ActionStatus.DENIED: "[-]",
    ActionStatus.FAILED: "[!]",
}

_INDENT = "      "


def render_workflow(view: WorkflowStatusResponse) -> str:
    """The whole status surface for one workflow, as the CLI prints it."""
    lines = [
        f"Workflow {view.workflow_id}" + (f" ({view.name})" if view.name else ""),
        f"Status: {view.status.value}",
    ]
    if view.current_step:
        lines.append(f"Now at: {view.current_step.step}")
    plan = _unique(step.step for step in view.plan)
    lines.append("Plan: " + (" -> ".join(plan) if plan else "nothing further"))
    completed = _unique(step.step for step in view.completed_steps)
    lines.append("Completed steps: " + (", ".join(completed) if completed else "none"))
    counts = ", ".join(
        f"{view.action_counts.get(status, 0)} {status.value}" for status, _ in _SECTIONS
    )
    lines.append(f"Actions: {counts}")

    for status, heading in _SECTIONS:
        actions = [action for action in view.actions if action.action_status is status]
        if not actions and status is not ActionStatus.EXECUTED:
            continue
        lines += ["", f"{heading} ({len(actions)})"]
        if not actions:
            lines.append("  (nothing has been executed)")
        for action in actions:
            lines += _action_lines(action)

    if view.shortlist:
        lines += ["", f"SHORTLIST -- recommendations, not applications ({len(view.shortlist)})"]
        lines += [_shortlist_line(item) for item in view.shortlist]

    if view.recoverable_failures:
        lines += ["", f"STEP FAILURES -- retryable ({len(view.recoverable_failures)})"]
        lines += [
            f"  {failure.step or 'unknown step'}: {failure.message}"
            for failure in view.recoverable_failures
        ]

    lines += ["", f"Audit trail: {view.audit_url}"]
    return "\n".join(lines)


def render_approvals(items: Sequence[PendingApprovalItem]) -> str:
    """The approval queue: every action waiting on a reviewer, consequences first."""
    if not items:
        return "No approvals waiting. Nothing is pending."
    lines = [f"WAITING FOR YOUR APPROVAL -- nothing has been sent ({len(items)})"]
    for item in items:
        lines += [
            "",
            f"{_MARK[item.action_status]} {item.action_status.value}  {item.action_type}",
            f"{_INDENT}target: {item.target}",
            f"{_INDENT}{item.summary}",
            f"{_INDENT}risk: {item.risk_level} · approval {item.id} ({item.status.value}, "
            f"expires {item.expires_at:%Y-%m-%d %H:%M} UTC)",
        ]
        lines += [f"{_INDENT}- {line}" for line in item.consequences]
        lines += _preview_lines(item.preview)
        lines.append(f"{_INDENT}workflow {item.workflow_id} · action_hash {item.action_hash}")
    return "\n".join(lines)


def _action_lines(action: WorkflowAction) -> list[str]:
    status = action.action_status
    lines = [
        f"  {_MARK[status]} {status.value}  {action.kind.value} -> {action.target}",
        f"{_INDENT}{action.summary}",
    ]
    if action.detail:
        lines.append(f"{_INDENT}{action.detail}")
    if status is ActionStatus.EXECUTED and action.receipt is not None:
        receipt = f"{_INDENT}receipt: {action.receipt.external_reference or 'no reference'}"
        if action.links.operation_id:
            receipt += f" · operation {action.links.operation_id}"
        lines.append(receipt)
        if action.links.audit_event_ids:
            ids = ", ".join(str(event_id) for event_id in action.links.audit_event_ids)
            lines.append(f"{_INDENT}audit entries: {ids}")
        if action.links.audit_url:
            lines.append(f"{_INDENT}audit: {action.links.audit_url}")
    if status is ActionStatus.AWAITING_APPROVAL:
        if action.approval_id:
            expires = (
                f", expires {action.approval_expires_at:%Y-%m-%d %H:%M} UTC"
                if action.approval_expires_at
                else ""
            )
            lines.append(f"{_INDENT}approval {action.approval_id}{expires}")
        lines += [f"{_INDENT}- {line}" for line in action.consequences]
        lines += _preview_lines(action.preview)
    if status in (ActionStatus.DENIED, ActionStatus.FAILED) and action.decided_by:
        lines.append(f"{_INDENT}decided by {action.decided_by}")
    return lines


def _preview_lines(preview: ActionPreview | None) -> list[str]:
    if preview is None:
        return []
    lines = [f"{_INDENT}to: {preview.recipient}"]
    lines.append(
        f"{_INDENT}thread: reply to {preview.in_reply_to}"
        if preview.is_reply
        else f"{_INDENT}thread: new"
    )
    if preview.subject:
        lines.append(f"{_INDENT}subject: {preview.subject}")
    for attachment in preview.attachments:
        digest = f" (sha256 {attachment.sha256[:12]})" if attachment.sha256 else ""
        lines.append(f"{_INDENT}attachment: {attachment.name}{digest}")
    if preview.body_diff:
        lines.append(f"{_INDENT}body diff:")
        lines += [f"{_INDENT}  {line}" for line in preview.body_diff.splitlines()]
    return lines


def _shortlist_line(item: ShortlistItem) -> str:
    if item.action_status is ActionStatus.RECOMMENDED:
        note = "recommended -- not applied"
    elif item.action_status is ActionStatus.EXECUTED:
        note = "application submitted (see DONE)"
    else:
        note = f"submission {item.action_status.value} -- not submitted"
    return (
        f"  {_MARK[item.action_status]} #{item.rank} {item.company} -- {item.title} "
        f"(score {item.score:.2f}): {note}"
    )


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


__all__ = ["render_approvals", "render_workflow"]
