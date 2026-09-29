"""The durable journal around an outward-facing action.

A durable checkpointer guarantees that a killed run resumes; it guarantees
nothing about what the killed run had already *done*. If a worker dies between
submitting an application and recording the receipt, the checkpoint says the
submission has not happened, so the resumed run submits again -- and the
candidate has applied twice.

So the side effect is bracketed by two durable writes, and this module is the
bracket:

1. **Before the call**, the intent's `idempotency_key` is claimed in
   `tool_executions`, committed. That row is the durable statement "this action
   is being attempted", and it survives the process that wrote it.
2. **After the call**, the receipt is written to the same row, committed. That
   row is now the durable statement "this action happened, and here is what came
   back".

A resumed run re-reaching the same action finds the claim already there and does
not call out again. What it does instead depends on which of the two writes
landed:

- the receipt is there: the action completed, and its stored receipt is
  replayed. No second side effect, and the run continues with the same outcome
  the first attempt got.
- the claim is there but no receipt: the outcome is *unknown*. The first attempt
  may have submitted and died before recording it. This returns a not-ok receipt
  rather than retrying, because a duplicate application cannot be taken back
  while a missed one can be resubmitted deliberately, and it is the reviewer's
  call, not a retry loop's.

That second rule is the one that costs something -- an action whose call never
actually left the process is left needing a human -- and it is the only rule
consistent with "a crash between those two points cannot produce a duplicate
write on resume".

This wraps the graph's `ActionExecutor` port rather than living inside the
approval node, for two reasons: `personalos.graphs` may not import
`persistence`, and the approval node's own invariant is that reviewing and
redeeming happen together in one place (see `graphs/job_search.py`). Wrapping is
the same shape `PolicyEnforcingToolGateway` uses one level up -- the adapter the
composition root binds is what adds the guarantee, and the caller's contract
does not change.
"""

import logging
from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.job_search import ActionIntent, ActionReceipt, ApprovalDecision
from personalos.domain.models import ToolExecutionStatus
from personalos.persistence.repositories import ToolExecutionRepository

logger = logging.getLogger(__name__)

#: Prefix for the `tool_executions.tool_name` an action is journaled under, so
#: graph-level actions are distinguishable from MCP tool calls sharing the table.
TOOL_NAME_PREFIX = "job_search.action"


class ActionExecutorPort(Protocol):
    """The shape this journal wraps.

    Declared structurally here rather than imported from
    `personalos.graphs.job_search`: `persistence` may not import `graphs`, and a
    `Protocol` is all the coupling either side actually needs.
    """

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Perform the approved action and return its receipt."""
        ...


class JournaledActionExecutor:
    """Runs an `ActionExecutor` at most once per idempotency key, durably.

    Takes a session factory rather than a session for the same reason
    `SqlOperationStore` does: the claim has to be committed and visible to other
    processes *before* the side effect runs, which it cannot be if it is sitting
    in the caller's open transaction.
    """

    def __init__(
        self,
        inner: ActionExecutorPort,
        session_factory: Callable[[], Session],
        *,
        workflow_id: UUID | None = None,
    ):
        """Wrap `inner`, journaling to `tool_executions` through `session_factory`."""
        if inner is None:
            raise ValueError("JournaledActionExecutor requires an inner ActionExecutor")
        self.inner = inner
        self.session_factory = session_factory
        self.workflow_id = workflow_id

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Execute the action unless this key already has an outcome.

        The decision is not re-checked here: the caller
        (`JobSearchGraph.approval_checkpoint`) has already refused to call an
        executor for an intent its decision does not authorize, and a second,
        weaker copy of that check in an adapter would invite the belief that the
        adapter is where the rule lives.
        """
        key = intent.idempotency_key
        tool_name = f"{TOOL_NAME_PREFIX}.{intent.kind.value}"

        record, claimed = self._claim(key, tool_name)
        if not claimed:
            return self._replay(intent, record)

        try:
            receipt = await self.inner.execute(intent, decision)
        except Exception as exc:
            self._fail(key, str(exc))
            raise

        self._complete(key, receipt)
        return receipt

    # ------------------------------------------------------------------
    # Journal operations, each its own committed transaction
    # ------------------------------------------------------------------

    def _claim(self, key: str, tool_name: str) -> tuple[dict, bool]:
        session = self.session_factory()
        try:
            record, claimed = ToolExecutionRepository(session).claim(
                key, tool_name, self.workflow_id
            )
            return record.to_dict(), claimed
        finally:
            session.close()

    def _complete(self, key: str, receipt: ActionReceipt) -> None:
        session = self.session_factory()
        try:
            ToolExecutionRepository(session).complete(key, receipt.model_dump(mode="json"))
        finally:
            session.close()

    def _fail(self, key: str, error: str) -> None:
        session = self.session_factory()
        try:
            ToolExecutionRepository(session).fail(key, error)
        finally:
            session.close()

    def _replay(self, intent: ActionIntent, record: dict) -> ActionReceipt:
        """Turn a pre-existing journal row into this attempt's receipt."""
        status = record.get("status")

        if status == ToolExecutionStatus.COMPLETED.value and record.get("receipt_json"):
            stored = dict(record["receipt_json"])
            logger.info(
                "action %s (%s) already completed under idempotency key '%s'; replaying "
                "its receipt instead of executing again",
                intent.action_id,
                intent.kind.value,
                intent.idempotency_key,
            )
            # `action_id` is re-bound to *this* intent: the stored receipt
            # belongs to the attempt that ran, and a receipt whose action_id
            # named a different attempt would not tie back to the intent the
            # rest of the run is carrying.
            stored["action_id"] = str(intent.action_id)
            return ActionReceipt.model_validate(stored)

        detail = (
            f"a previous attempt at '{intent.idempotency_key}' did not record an "
            f"outcome (status={status}); not retried, because this action may already "
            f"have taken effect outside this system"
        )
        logger.warning(
            "action %s (%s) is in doubt: %s", intent.action_id, intent.kind.value, detail
        )
        return ActionReceipt(action_id=intent.action_id, ok=False, detail=detail)


__all__ = ["TOOL_NAME_PREFIX", "ActionExecutorPort", "JournaledActionExecutor"]
