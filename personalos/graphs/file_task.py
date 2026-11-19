"""The File subgraph: find, read, summarize and organize job-search documents.

Pipeline:

    discover_files -> plan_file_task -> read_documents
        -> [needs approval] request_file_approval -> file_approval_checkpoint (interrupts)
        -> apply_file_changes -> verify_file_changes -> END

**Planner, executor, verifier.** `plan_file_task` turns the request and the
inventory `discover_files` found into a typed `FilePlan` -- explicit paths,
a closed set of operations -- through the `ResilientFilePlanner` (a
schema-constrained model with repair attempts, then a deterministic fallback)
and re-validates whatever comes back against `FILE_PLAN_JSON_SCHEMA` and the
inventory itself. `read_documents` and `apply_file_changes` execute it;
`verify_file_changes` checks every change against the disk.

**All file access is a tool call through the sandboxed files MCP server.**
This module holds no file handle and imports nothing that could open one: its
one way out is the `FileOperations` port, bound by the composition root to
`personalos.executor.file_operations.GatewayFileOperations`, which turns every
call into a `files.*` intent through the policy-enforcing gateway. The server
resolves every path canonically and refuses -- and audits -- anything outside
its allowed roots.

**The plan is fixed before any document is opened, and documents are data.**
The planner sees names, kinds, sizes and times, never content. A document's
text reaches `read_documents` and stops there: it is handed to the summarizer
as quoted data, and what comes back is a `DocumentSummary`, which has no field
a tool call or a new step could ride in. Which tools a node may call is
decided by policy from the node's own `requested_by`
(`personalos.policy.rules.FILE_TASK_NODE_TOOL_ALLOWLISTS`), so the node that
reads documents cannot move or delete one whatever a document says.

**Approval is per plan, and shows the exact paths.** A plan with any delete,
any overwrite, or more mutated files than `FileApprovalPolicy` allows is
proposed as one `APPLY_FILE_PLAN` action whose payload lists every step and
every affected path and pins the plan's hash. It goes through the same
request -> interrupt -> redeem triple as the Job Search subgraph, so the
approvals API and inbox handle it unchanged. At redemption the action's hash
*and* the plan's hash are recomputed; either having drifted fails closed.

**Mutations are idempotent by key and reconciled by hash.** Every mutating
step is sent with `step_idempotency_key(plan, step)` and the hash of the
version discovery saw. Before each one, the apply node looks at the disk: a
file already in the step's goal state -- the run crashed after the write and
before the checkpoint -- is recorded as `RECONCILED` and not written again.

**State is JSON.** As in every graph here, so the run survives an interrupt
in a durable checkpointer -- which redacts credentials from every checkpoint
it writes. Document text is never kept whole in state: only hashes, summaries
and short excerpts, each redacted again on the way in.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol, TypedDict
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from personalos.domain.errors import PersonalOSError, PolicyDeniedError
from personalos.domain.files import (
    FileApprovalPolicy,
    FileContent,
    FileEntry,
    FileInventory,
    FileOp,
    FilePlan,
    FilePlanInvalid,
    FileReceipt,
    FileStat,
    FileStep,
    FileVerification,
    PlanApproval,
    StepOutcome,
    UndoRecord,
    check_plan_against_inventory,
    describe_step,
    expected_sha256,
    parse_file_plan,
    step_idempotency_key,
)
from personalos.domain.job_search import (
    PAYLOAD_BODY,
    ActionIntent,
    ActionKind,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    JobSearchContractError,
    RefusalReason,
    authorize_execution,
)
from personalos.domain.redaction import redact, redact_text
from personalos.domain.workflow import file_task_thread_id
from personalos.models.file_planning import (
    DocumentSummarizer,
    ExtractiveDocumentSummarizer,
    ResilientFilePlanner,
)
from personalos.observability.structured_logging import LogEvent
from personalos.observability.tracing import traced_node
from personalos.policy import PolicyError
from personalos.policy.rules import FILE_TASK_REQUESTER_PREFIX

logger = logging.getLogger(__name__)

GRAPH_NAME = "file_task"

DISCOVER_FILES = "discover_files"
PLAN_FILE_TASK = "plan_file_task"
READ_DOCUMENTS = "read_documents"
REQUEST_FILE_APPROVAL = "request_file_approval"
FILE_APPROVAL_CHECKPOINT = "file_approval_checkpoint"
APPLY_FILE_CHANGES = "apply_file_changes"
VERIFY_FILE_CHANGES = "verify_file_changes"

#: `approval_stage` / the interrupt's `stage`.
STAGE_FILE_PLAN = "file_plan"

#: Longest excerpt a read step keeps of what it read.
EXCERPT_CHARS = 500

#: Payload keys of an `APPLY_FILE_PLAN` action. Part of its hash.
PAYLOAD_PLAN_HASH = "plan_hash"
PAYLOAD_STEPS = "steps"
PAYLOAD_AFFECTED_PATHS = "affected_paths"
PAYLOAD_REASONS = "reasons"


def requester(node: str) -> str:
    """The `requested_by` a node stamps on its tool calls; policy scopes tools by it."""
    return f"{FILE_TASK_REQUESTER_PREFIX}{node}"


# --- Ports ----------------------------------------------------------------------


class FileOperations(Protocol):
    """Every file access this subgraph makes. Each method is one `files.*` tool call.

    Raises a `PolicyError` when policy refuses the call, and a
    `PersonalOSError` when the tool does (a path outside the sandbox is a
    `PolicyDeniedError`).
    """

    async def list_files(self, *, requested_by: str) -> FileInventory: ...

    async def stat_file(self, path: str, *, requested_by: str) -> FileStat: ...

    async def read_file(self, path: str, *, requested_by: str) -> FileContent: ...

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str | None,
        idempotency_key: str,
        requested_by: str,
        approval: PlanApproval | None,
    ) -> dict[str, Any]: ...

    async def move_file(
        self,
        source: str,
        destination: str,
        *,
        expected_sha256: str,
        plan_size: int,
        idempotency_key: str,
        requested_by: str,
        approval: PlanApproval | None,
    ) -> dict[str, Any]: ...

    async def delete_file(
        self,
        path: str,
        *,
        expected_sha256: str,
        idempotency_key: str,
        requested_by: str,
        approval: PlanApproval | None,
    ) -> dict[str, Any]: ...


class FileTaskState(TypedDict, total=False):
    """Graph state. JSON only."""

    request: str
    user_id: str | None
    inventory: dict[str, Any] | None
    plan: dict[str, Any] | None
    plan_source: str | None
    plan_errors: list[str]
    approval_reasons: list[str]
    approval_stage: str | None
    pending_actions: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    approval_refusals: list[dict[str, Any]]
    read_receipts: list[dict[str, Any]]
    change_receipts: list[dict[str, Any]]
    verifications: list[dict[str, Any]]
    verified: bool | None


def _load_plan(state: FileTaskState) -> FilePlan | None:
    raw = state.get("plan")
    return FilePlan.model_validate(raw) if raw else None


def _load_inventory(state: FileTaskState) -> FileInventory:
    raw = state.get("inventory")
    if not raw:
        raise JobSearchContractError("no inventory in state; discover_files must run first")
    return FileInventory.model_validate(raw)


def _dump(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [value.model_dump(mode="json") for value in values]


# --- Graph ----------------------------------------------------------------------


class FileTaskGraph:
    """Builds and compiles the File subgraph."""

    def __init__(
        self,
        files: FileOperations,
        *,
        planner: ResilientFilePlanner | None = None,
        summarizer: DocumentSummarizer | None = None,
        approval_policy: FileApprovalPolicy | None = None,
        approval_ttl: timedelta | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        if files is None:
            raise ValueError("FileTaskGraph requires a FileOperations port")
        self.files = files
        self.planner = planner or ResilientFilePlanner(primary=None)
        self.summarizer = summarizer or ExtractiveDocumentSummarizer()
        self.approval_policy = approval_policy or FileApprovalPolicy()
        self.approval_ttl = approval_ttl
        self.clock = clock
        self.checkpointer = checkpointer or InMemorySaver()

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(FileTaskState)
        for name, method in (
            (DISCOVER_FILES, self.discover_files),
            (PLAN_FILE_TASK, self.plan_file_task),
            (READ_DOCUMENTS, self.read_documents),
            (REQUEST_FILE_APPROVAL, self.request_file_approval),
            (FILE_APPROVAL_CHECKPOINT, self.file_approval_checkpoint),
            (APPLY_FILE_CHANGES, self.apply_file_changes),
            (VERIFY_FILE_CHANGES, self.verify_file_changes),
        ):
            graph.add_node(name, traced_node(name, method, graph=GRAPH_NAME))

        graph.add_edge(START, DISCOVER_FILES)
        graph.add_edge(DISCOVER_FILES, PLAN_FILE_TASK)
        graph.add_conditional_edges(
            PLAN_FILE_TASK, self.route_after_plan, {READ_DOCUMENTS: READ_DOCUMENTS, END: END}
        )
        graph.add_conditional_edges(
            READ_DOCUMENTS,
            self.route_after_reads,
            {
                REQUEST_FILE_APPROVAL: REQUEST_FILE_APPROVAL,
                APPLY_FILE_CHANGES: APPLY_FILE_CHANGES,
                END: END,
            },
        )
        graph.add_edge(REQUEST_FILE_APPROVAL, FILE_APPROVAL_CHECKPOINT)
        graph.add_edge(FILE_APPROVAL_CHECKPOINT, APPLY_FILE_CHANGES)
        graph.add_edge(APPLY_FILE_CHANGES, VERIFY_FILE_CHANGES)
        graph.add_edge(VERIFY_FILE_CHANGES, END)
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def discover_files(self, state: FileTaskState) -> dict[str, Any]:
        """List the job-search documents inside the allowed roots: metadata, no content."""
        inventory = await self.files.list_files(requested_by=requester(DISCOVER_FILES))
        logger.info(
            "discovered %d document(s) in %d root(s)",
            len(inventory.entries),
            len(inventory.roots),
        )
        return {"inventory": inventory.model_dump(mode="json")}

    async def plan_file_task(self, state: FileTaskState) -> dict[str, Any]:
        """Plan the request over the inventory, and decide whether it needs approval.

        Whatever planner produced the plan, it is validated here again --
        schema, then inventory -- so a planner cannot hand the executor a step
        the checks never saw.
        """
        inventory = _load_inventory(state)
        outcome = await self.planner.plan(state.get("request", ""), inventory)
        errors = list(outcome.errors)
        plan: FilePlan | None = None
        if outcome.plan is not None:
            try:
                plan = check_plan_against_inventory(parse_file_plan(outcome.plan), inventory)
            except FilePlanInvalid as exc:
                errors.append(f"graph: {exc}")
        if plan is None:
            logger.info("no valid file plan for the request: %s", "; ".join(errors))
            return {
                "plan": None,
                "plan_source": outcome.source,
                "plan_errors": errors,
                "approval_reasons": [],
                "pending_actions": [],
            }

        reasons = self.approval_policy.reasons(plan, inventory)
        pending = [self._plan_action(plan, reasons)] if reasons else []
        logger.info(
            "planned %d step(s) (%d mutating) from the %s planner%s",
            len(plan.steps),
            len(plan.mutating_steps),
            outcome.source,
            f"; needs approval: {', '.join(reasons)}" if reasons else "",
        )
        update: dict[str, Any] = {
            "plan": plan.model_dump(mode="json"),
            "plan_source": outcome.source,
            "plan_errors": errors,
            "approval_reasons": list(reasons),
            "pending_actions": _dump(pending),
        }
        if pending:
            update["approval_stage"] = STAGE_FILE_PLAN
        return update

    def _plan_action(self, plan: FilePlan, reasons: Sequence[str]) -> ActionIntent:
        """The one action a reviewer approves for the whole plan: exact steps and paths."""
        lines = [describe_step(step) for step in plan.steps]
        affected = list(plan.affected_paths())
        return ActionIntent(
            kind=ActionKind.APPLY_FILE_PLAN,
            target=f"{len(affected)} file(s): {', '.join(affected)}"[:500],
            summary=f"{plan.goal} ({'; '.join(reasons)})"[:500],
            payload={
                PAYLOAD_PLAN_HASH: plan.plan_hash(),
                PAYLOAD_STEPS: lines,
                PAYLOAD_AFFECTED_PATHS: affected,
                PAYLOAD_REASONS: list(reasons),
                PAYLOAD_BODY: "\n".join(lines),
            },
            idempotency_key=f"file-plan-{plan.plan_hash()[:40]}",
            requested_by=requester(PLAN_FILE_TASK),
        )

    async def read_documents(self, state: FileTaskState) -> dict[str, Any]:
        """Run the plan's locate, read and summarize steps.

        The one node that sees document content. Its tool calls are scoped to
        reading; the text goes to the summarizer as data and comes back as a
        `DocumentSummary`, redacted before it becomes state. Nothing here
        changes the plan, the pending actions or the route.
        """
        plan = _load_plan(state)
        if plan is None:
            return {"read_receipts": []}
        inventory = _load_inventory(state)
        entries = inventory.by_path()
        receipts: list[FileReceipt] = []
        for step in plan.steps:
            if step.mutating:
                continue
            entry = entries.get(step.path)
            if step.op == FileOp.LOCATE:
                receipts.append(_receipt(step, StepOutcome.APPLIED, entry=entry))
                continue
            try:
                content = await self.files.read_file(
                    step.path, requested_by=requester(READ_DOCUMENTS)
                )
            except (PolicyError, PersonalOSError) as exc:
                receipts.append(_refused(step, exc))
                continue
            receipts.append(await self._read_receipt(step, entry, content))
        return {"read_receipts": _dump(receipts)}

    async def _read_receipt(
        self, step: FileStep, entry: FileEntry | None, content: FileContent
    ) -> FileReceipt:
        text = content.text
        if text is None:
            return _receipt(
                step,
                StepOutcome.APPLIED,
                entry=entry,
                sha256=content.sha256,
                detail="binary document: no text could be extracted",
            )
        summary = None
        if step.op == FileOp.SUMMARIZE and entry is not None:
            summary = redact(await self.summarizer.summarize(entry, text))
            # The path is the plan's, whatever the summarizer said it was.
            summary = summary.model_copy(update={"path": step.path})
        return _receipt(
            step,
            StepOutcome.APPLIED,
            entry=entry,
            sha256=content.sha256,
            summary=summary,
            excerpt=redact_text(text[:EXCERPT_CHARS]) if step.op == FileOp.READ else None,
        )

    def request_file_approval(self, state: FileTaskState) -> dict[str, Any]:
        """Mint the `ApprovalRequest` for the plan's action, before the interrupt.

        Separate from the interrupt for the reason the Job Search subgraph's
        is: the request's hash must be checkpointed before the run pauses.
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
                "approval requested for file plan action %s on %s; expires %s",
                request.action_id,
                request.target,
                request.expires_at.isoformat(),
            )
        return {"approval_requests": [*(state.get("approval_requests") or []), *_dump(requests)]}

    def file_approval_checkpoint(self, state: FileTaskState) -> dict[str, Any]:
        """Park the run until a person answers the plan's approval request."""
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
                "a file plan action reached the approval checkpoint with no request on file"
            )
        for request in outstanding:
            logger.info(
                "file plan action %s awaits approval",
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
                "stage": state.get("approval_stage") or STAGE_FILE_PLAN,
                "requests": [request.model_dump(mode="json") for request in outstanding],
            }
        )
        decisions = _decisions_from_resume(resumed)
        return {"approvals": [*(state.get("approvals") or []), *_dump(decisions)]}

    async def apply_file_changes(self, state: FileTaskState) -> dict[str, Any]:
        """Run the plan's writes, moves and deletes -- each reconciled against the disk first.

        A plan that needed approval runs only if `authorize_execution` passes
        and the plan in state still hashes to the hash the approval pinned;
        otherwise every mutating step is skipped and the refusal recorded.
        """
        plan = _load_plan(state)
        if plan is None or not plan.mutating_steps:
            return {
                "change_receipts": [],
                "approval_refusals": state.get("approval_refusals") or [],
            }
        inventory = _load_inventory(state)

        approval: PlanApproval | None = None
        refusals = list(state.get("approval_refusals") or [])
        intents = [ActionIntent.model_validate(raw) for raw in state.get("pending_actions") or ()]
        if intents:
            approval, refusal = self._redeem(state, plan, intents[0])
            if refusal is not None:
                refusals.append(refusal.model_dump(mode="json"))
                skipped = [
                    _receipt(step, StepOutcome.SKIPPED, detail=f"not approved: {refusal.detail}")
                    for step in plan.mutating_steps
                ]
                return {"change_receipts": _dump(skipped), "approval_refusals": refusals}

        receipts: list[FileReceipt] = []
        failed = False
        plan_size = len(plan.changed_files())
        for step in plan.mutating_steps:
            if failed:
                receipts.append(
                    _receipt(step, StepOutcome.SKIPPED, detail="an earlier step did not complete")
                )
                continue
            receipt = await self._apply(step, plan, inventory, plan_size, approval)
            receipts.append(receipt)
            failed = not receipt.ok
        return {"change_receipts": _dump(receipts), "approval_refusals": refusals}

    def _redeem(
        self, state: FileTaskState, plan: FilePlan, intent: ActionIntent
    ) -> tuple[PlanApproval | None, ApprovalRefusal | None]:
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
        request = requests.get(intent.action_id)
        refusal = authorize_execution(
            intent=intent,
            request=request,
            decision=decisions.get(intent.action_id),
            now=self.clock(),
        )
        if refusal is None and intent.payload.get(PAYLOAD_PLAN_HASH) != plan.plan_hash():
            refusal = ApprovalRefusal(
                action_id=intent.action_id,
                request_id=request.request_id if request else None,
                reason=RefusalReason.HASH_MISMATCH,
                detail="the plan changed after it was approved; nothing was run",
                approved_hash=str(intent.payload.get(PAYLOAD_PLAN_HASH)),
                recomputed_hash=plan.plan_hash(),
            )
        if refusal is not None:
            (logger.warning if refusal.suspicious else logger.info)(
                "refusing to apply file plan %s: %s", intent.action_id, refusal.detail
            )
            return None, refusal
        return PlanApproval(action=intent, decision=decisions[intent.action_id]), None

    async def _apply(
        self,
        step: FileStep,
        plan: FilePlan,
        inventory: FileInventory,
        plan_size: int,
        approval: PlanApproval | None,
    ) -> FileReceipt:
        key = step_idempotency_key(plan, step)
        expected = expected_sha256(step, inventory)
        who = requester(APPLY_FILE_CHANGES)
        try:
            if step.op == FileOp.WRITE:
                return await self._apply_write(step, key, expected, who, approval)
            if step.op == FileOp.MOVE:
                return await self._apply_move(step, key, expected, plan_size, who, approval)
            return await self._apply_delete(step, key, expected, who, approval)
        except (PolicyError, PersonalOSError) as exc:
            return _refused(step, exc, idempotency_key=key)

    async def _apply_write(self, step, key, expected, who, approval) -> FileReceipt:
        goal = step.content_sha256()
        current = await self.files.stat_file(step.path, requested_by=who)
        if current.exists and current.sha256 == goal:
            return _receipt(
                step,
                StepOutcome.RECONCILED,
                sha256=goal,
                previous_sha256=expected,
                idempotency_key=key,
                detail="already written; not written again",
            )
        if current.exists and current.sha256 != expected:
            return _receipt(
                step,
                StepOutcome.FAILED,
                idempotency_key=key,
                detail="the file changed since it was discovered; nothing was written",
            )
        if not current.exists and expected is not None:
            return _receipt(
                step,
                StepOutcome.FAILED,
                idempotency_key=key,
                detail="the file was removed since it was discovered; nothing was written",
            )
        result = await self.files.write_file(
            step.path,
            step.content or "",
            expected_sha256=expected,
            idempotency_key=key,
            requested_by=who,
            approval=approval,
        )
        return _receipt(
            step,
            StepOutcome.APPLIED,
            sha256=result.get("sha256"),
            previous_sha256=result.get("previous_sha256"),
            idempotency_key=key,
        )

    async def _apply_move(self, step, key, expected, plan_size, who, approval) -> FileReceipt:
        assert step.destination is not None and expected is not None
        source = await self.files.stat_file(step.path, requested_by=who)
        target = await self.files.stat_file(step.destination, requested_by=who)
        undo = UndoRecord(path=step.destination, destination=step.path, expected_sha256=expected)
        if not source.exists and target.exists and target.sha256 == expected:
            return _receipt(
                step,
                StepOutcome.RECONCILED,
                sha256=expected,
                idempotency_key=key,
                undo=undo,
                detail="already moved; not moved again",
            )
        if not source.exists or source.sha256 != expected or target.exists:
            return _receipt(
                step,
                StepOutcome.FAILED,
                idempotency_key=key,
                detail=(
                    "the source changed or the destination exists since discovery; "
                    "nothing was moved"
                ),
            )
        result = await self.files.move_file(
            step.path,
            step.destination,
            expected_sha256=expected,
            plan_size=plan_size,
            idempotency_key=key,
            requested_by=who,
            approval=approval,
        )
        returned = result.get("undo") or {}
        return _receipt(
            step,
            StepOutcome.APPLIED,
            sha256=result.get("sha256"),
            idempotency_key=key,
            undo=UndoRecord.model_validate({**undo.model_dump(mode="json"), **returned})
            if returned
            else undo,
        )

    async def _apply_delete(self, step, key, expected, who, approval) -> FileReceipt:
        assert expected is not None
        current = await self.files.stat_file(step.path, requested_by=who)
        if not current.exists:
            return _receipt(
                step,
                StepOutcome.RECONCILED,
                sha256=expected,
                idempotency_key=key,
                detail="already deleted",
            )
        if current.sha256 != expected:
            return _receipt(
                step,
                StepOutcome.FAILED,
                idempotency_key=key,
                detail="the file changed since it was discovered; nothing was deleted",
            )
        result = await self.files.delete_file(
            step.path,
            expected_sha256=expected,
            idempotency_key=key,
            requested_by=who,
            approval=approval,
        )
        return _receipt(step, StepOutcome.APPLIED, sha256=result.get("sha256"), idempotency_key=key)

    async def verify_file_changes(self, state: FileTaskState) -> dict[str, Any]:
        """Check every applied change against the disk, by hash."""
        plan = _load_plan(state)
        steps = {step.step_id: step for step in (plan.steps if plan else ())}
        verifications: list[FileVerification] = []
        who = requester(VERIFY_FILE_CHANGES)
        for raw in state.get("change_receipts") or ():
            receipt = FileReceipt.model_validate(raw)
            step = steps.get(receipt.step_id)
            if step is None or not receipt.ok:
                continue
            try:
                verifications.append(await self._verify(step, receipt, who))
            except (PolicyError, PersonalOSError) as exc:
                verifications.append(
                    FileVerification(step_id=step.step_id, ok=False, detail=redact_text(str(exc)))
                )
        verified = all(v.ok for v in verifications)
        if not verified:
            logger.warning(
                "file plan verification failed for step(s) %s",
                [v.step_id for v in verifications if not v.ok],
            )
        return {"verifications": _dump(verifications), "verified": verified}

    async def _verify(self, step: FileStep, receipt: FileReceipt, who: str) -> FileVerification:
        if step.op == FileOp.WRITE:
            now = await self.files.stat_file(step.path, requested_by=who)
            ok = now.exists and now.sha256 == step.content_sha256()
            return FileVerification(
                step_id=step.step_id,
                ok=ok,
                detail="content matches" if ok else "the file does not hold what was written",
            )
        if step.op == FileOp.MOVE:
            assert step.destination is not None
            source = await self.files.stat_file(step.path, requested_by=who)
            target = await self.files.stat_file(step.destination, requested_by=who)
            ok = not source.exists and target.exists and target.sha256 == receipt.sha256
            return FileVerification(
                step_id=step.step_id,
                ok=ok,
                detail="moved intact" if ok else "the file is not where the move put it",
            )
        now = await self.files.stat_file(step.path, requested_by=who)
        return FileVerification(
            step_id=step.step_id,
            ok=not now.exists,
            detail="deleted" if not now.exists else "the file still exists",
        )

    # ------------------------------------------------------------------
    # Routing: reads only the plan and the approval reasons, never content
    # ------------------------------------------------------------------

    def route_after_plan(self, state: FileTaskState) -> str:
        return READ_DOCUMENTS if state.get("plan") else END

    def route_after_reads(self, state: FileTaskState) -> str:
        plan = _load_plan(state)
        if plan is None or not plan.mutating_steps:
            return END
        if state.get("pending_actions"):
            return REQUEST_FILE_APPROVAL
        return APPLY_FILE_CHANGES


def _receipt(step: FileStep, outcome: StepOutcome, **fields: Any) -> FileReceipt:
    return FileReceipt(
        step_id=step.step_id,
        op=step.op,
        path=step.path,
        destination=step.destination,
        outcome=outcome,
        **fields,
    )


def _refused(step: FileStep, exc: Exception, **fields: Any) -> FileReceipt:
    """A step policy or the sandbox refused, or the tool failed."""
    denied = isinstance(exc, PolicyError | PolicyDeniedError)
    return _receipt(
        step,
        StepOutcome.DENIED if denied else StepOutcome.FAILED,
        detail=redact_text(f"{type(exc).__name__}: {exc}")[:500],
        **fields,
    )


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
    raise JobSearchContractError(
        f"cannot read an approval decision out of a resume value of type {type(resumed).__name__}"
    )


# --- Supervisor adapter -----------------------------------------------------------


class FileTaskSubgraphRunner:
    """Satisfies the Supervisor's `FileSubgraphRunner` port with this subgraph.

    Runs on its own stable thread (`file_task_thread_id`), continued rather
    than restarted when it already has steps to run -- a task parked at an
    approval, or one whose worker died part way through.
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
        self.thread_id = thread_id or file_task_thread_id(user_id or "anonymous")
        self.workflow_id = workflow_id

    async def __call__(self, task_dag: dict[str, Any], state: Any) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": self.thread_id}
        if self.workflow_id is not None:
            configurable["workflow_id"] = str(self.workflow_id)
        config = {"configurable": configurable}
        stored = await self.graph.aget_state(config)
        initial: FileTaskState | None = None
        if not stored.next:
            initial = {
                "request": (state or {}).get("message", "") if hasattr(state, "get") else "",
                "user_id": str(self.user_id) if self.user_id else None,
            }
        else:
            logger.info(
                "file task thread '%s' has %s still to run; continuing it",
                self.thread_id,
                ", ".join(stored.next),
            )
        final = await self.graph.ainvoke(initial, config=config)
        return file_task_result(task_dag, final)


def file_task_result(task_dag: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """What the Supervisor records of a file task: the outcome, never document text."""
    return {
        "goal": task_dag.get("goal"),
        "domain": "file",
        "plan": final.get("plan"),
        "plan_source": final.get("plan_source"),
        "plan_errors": final.get("plan_errors") or [],
        "approval_reasons": final.get("approval_reasons") or [],
        "awaiting_approval": bool(final.get("__interrupt__")),
        "read_receipts": final.get("read_receipts") or [],
        "change_receipts": final.get("change_receipts") or [],
        "verified": final.get("verified"),
    }


__all__ = [
    "APPLY_FILE_CHANGES",
    "DISCOVER_FILES",
    "FILE_APPROVAL_CHECKPOINT",
    "GRAPH_NAME",
    "PLAN_FILE_TASK",
    "READ_DOCUMENTS",
    "REQUEST_FILE_APPROVAL",
    "STAGE_FILE_PLAN",
    "VERIFY_FILE_CHANGES",
    "FileOperations",
    "FileTaskGraph",
    "FileTaskState",
    "FileTaskSubgraphRunner",
    "PlanApproval",
    "file_task_result",
    "requester",
]
