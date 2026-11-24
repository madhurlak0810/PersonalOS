"""Runs one step of a cross-domain `TaskPlan` through its domain's existing subgraph.

The Supervisor executes a plan step by step (`personalos.graphs.supervisor`);
this module is what a step actually runs on. Each step gets a thread of its
own -- derived from the plan's parent thread and the step id, in the domain's
own namespace (`calendar:...`, `communications:...`) -- so:

- every step's subgraph checkpoints on its own, and a step parked at its
  approval is resumed on its own thread by the ordinary approval path;
- each step's approval is that subgraph's own `ApprovalRequest`, bound to its
  own action hash -- a plan never batches two effects into one approval;
- a re-run of the Supervisor step that started a step finds the step's
  thread and reads where it got to, instead of starting the subgraph again.

**By reference.** `initial_state` maps the refs a step was handed onto the
subgraph's structured fields -- a thread id, an event id, an application id
-- and the step's instruction is the person's own words. No content from an
earlier step is pasted into a later step's request.

**What comes back is a `StepRecord`**, read from the step thread's state, not
from text: its approvals (one per request, with its hash and how it was
answered), its effects (from ok receipts), the refs it hands on, and a status
-- `awaiting_approval` while parked, `rejected` / `expired` when a person
said no or nobody answered, `failed` when the subgraph concluded it could not
do what was asked.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from personalos.domain.calendar import CalendarOutcome
from personalos.domain.files import FilePlan
from personalos.domain.job_search import (
    APPROVAL_EXPIRY_ACTOR,
    ActionIntent,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    ApprovalVerdict,
    RefusalReason,
)
from personalos.domain.routing import RouteDomain
from personalos.domain.task_plan import (
    PlanStep,
    StepApproval,
    StepEffect,
    StepRecord,
    StepRef,
    StepRefKind,
    StepStatus,
)
from personalos.domain.workflow import (
    calendar_thread_id,
    communications_thread_id,
    file_task_thread_id,
    job_search_thread_id,
)

logger = logging.getLogger(__name__)


# --- Ports ------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepContext:
    """Who a step runs for, and under which parent run."""

    plan_id: str
    parent_thread_id: str
    workflow_id: str | None = None
    actor_id: str = "system"
    correlation_id: str | None = None


@dataclass(frozen=True)
class BoundThread:
    """The thread a step runs on, and the candidate it runs for."""

    thread_id: str
    user_id: UUID | None
    configurable: Mapping[str, Any] = field(default_factory=dict)


class StepThreadBinder(Protocol):
    """Decides which thread a step runs on, and records how its runs went.

    The worker's binder registers the thread under the parent's workflow, so
    every step is a thread of that one workflow, owned by the conversation's
    candidate (`apps.worker.main.WorkflowScopedStepThreads`).
    """

    def bind(self, step: PlanStep, context: StepContext) -> BoundThread: ...

    def started(self, thread: BoundThread) -> None: ...

    def settled(self, thread: BoundThread, *, finished: bool, failed: bool = False) -> None: ...


class PlanStepRunner(Protocol):
    """Port the Supervisor runs plan steps through, one per domain."""

    async def run(
        self, step: PlanStep, inputs: Mapping[str, StepRef], context: StepContext
    ) -> StepRecord:
        """Start the step (or pick it up where it got to) and report where it stands."""
        ...

    async def poll(self, step: PlanStep, record: StepRecord, context: StepContext) -> StepRecord:
        """Report where a parked step stands now, without running anything."""
        ...


#: The thread-id recipe for a step of each domain: `(user_id, task_key)`.
STEP_THREAD_IDS: Mapping[RouteDomain, Callable[[Any, str], str]] = {
    RouteDomain.CALENDAR: calendar_thread_id,
    RouteDomain.COMMUNICATIONS: communications_thread_id,
    RouteDomain.FILE: file_task_thread_id,
    RouteDomain.JOB: job_search_thread_id,
}


def step_task_key(context: StepContext, step: PlanStep) -> str:
    """What distinguishes one step's thread: the parent run, the plan, and the step."""
    parent = context.workflow_id or context.parent_thread_id
    return f"{parent}:{context.plan_id}:{step.id}"


class LocalStepThreads:
    """A binder for one candidate, outside the worker: derives the thread, records nothing."""

    def __init__(self, user_id: UUID | None = None):
        self.user_id = user_id

    def bind(self, step: PlanStep, context: StepContext) -> BoundThread:
        derive = STEP_THREAD_IDS[step.domain]
        thread_id = derive(self.user_id or "anonymous", step_task_key(context, step))
        configurable = {"workflow_id": context.workflow_id} if context.workflow_id else {}
        return BoundThread(thread_id=thread_id, user_id=self.user_id, configurable=configurable)

    def started(self, thread: BoundThread) -> None:
        return None

    def settled(self, thread: BoundThread, *, finished: bool, failed: bool = False) -> None:
        return None


# --- Domain adapters --------------------------------------------------------------------


class StepAdapter(Protocol):
    """What one domain's subgraph state means for a plan step."""

    def initial_state(
        self, step: PlanStep, inputs: Mapping[str, StepRef], user_id: UUID | None
    ) -> dict[str, Any]: ...

    def outputs(
        self, values: Mapping[str, Any], effects: list[StepEffect]
    ) -> dict[str, StepRef]: ...

    def failure(self, values: Mapping[str, Any]) -> tuple[str, str] | None:
        """`(outcome, detail)` when the subgraph could not do what it was asked."""
        ...


def _ref(kind: StepRefKind, value: Any, trust: str = "untrusted") -> StepRef | None:
    if not value:
        return None
    try:
        return StepRef(kind=kind, id=str(value), content_trust=trust)  # type: ignore[arg-type]
    except ValueError:
        logger.warning("dropping a %s reference that is not a bare id", kind.value)
        return None


def _compact(refs: Mapping[str, StepRef | None]) -> dict[str, StepRef]:
    return {name: ref for name, ref in refs.items() if ref is not None}


_CALENDAR_FAILURES = frozenset(
    {
        CalendarOutcome.CONFLICT.value,
        CalendarOutcome.NEEDS_CLARIFICATION.value,
        CalendarOutcome.EVENT_NOT_FOUND.value,
        CalendarOutcome.AMBIGUOUS_EVENT.value,
        CalendarOutcome.NOT_MANAGED.value,
        CalendarOutcome.IN_PAST.value,
        CalendarOutcome.APPLICATION_NOT_FOUND.value,
    }
)


class CalendarStepAdapter:
    """A calendar step: an event or application by ref, the time in the person's words."""

    def initial_state(
        self, step: PlanStep, inputs: Mapping[str, StepRef], user_id: UUID | None
    ) -> dict[str, Any]:
        event = inputs.get("event")
        application = inputs.get("application")
        return {
            "request": step.instruction,
            "user_id": str(user_id) if user_id else None,
            "event_id": event.id if event is not None else None,
            "application_id": application.id if application is not None else None,
        }

    def outputs(self, values: Mapping[str, Any], effects: list[StepEffect]) -> dict[str, StepRef]:
        created = next((e.external_reference for e in effects if e.external_reference), None)
        plan = values.get("change_plan") or {}
        event = (plan.get("event") or {}).get("event_id")
        # An event this system wrote (or is about to) -- an id, not its contents.
        return _compact({"event": _ref(StepRefKind.CALENDAR_EVENT, created or event, "trusted")})

    def failure(self, values: Mapping[str, Any]) -> tuple[str, str] | None:
        plan = values.get("change_plan") or {}
        outcome = plan.get("outcome")
        if outcome in _CALENDAR_FAILURES:
            return outcome, plan.get("detail") or outcome
        return None


class CommunicationsStepAdapter:
    """A communications step: a thread by ref, or found from the person's words."""

    def initial_state(
        self, step: PlanStep, inputs: Mapping[str, StepRef], user_id: UUID | None
    ) -> dict[str, Any]:
        thread = inputs.get("thread")
        return {
            "request": step.instruction,
            "user_id": str(user_id) if user_id else None,
            "thread_id": thread.id if thread is not None else None,
        }

    def outputs(self, values: Mapping[str, Any], effects: list[StepEffect]) -> dict[str, StepRef]:
        draft = values.get("reply_draft") or {}
        return _compact(
            {
                # Read from outside: whatever is behind these is data, never instructions.
                "thread": _ref(
                    StepRefKind.EMAIL_THREAD, (values.get("thread") or {}).get("thread_id")
                ),
                "reply_draft": _ref(
                    StepRefKind.REPLY_DRAFT,
                    draft.get("artifact_version_id") or draft.get("gmail_draft_id"),
                ),
                "application": _ref(StepRefKind.APPLICATION, values.get("application_id")),
            }
        )

    def failure(self, values: Mapping[str, Any]) -> tuple[str, str] | None:
        request = values.get("comms_request") or {}
        thread_task = request.get("task") in {
            "summarize_thread",
            "extract_action_items",
            "draft_reply",
        }
        if thread_task and not values.get("thread"):
            errors = values.get("errors") or []
            return "thread_not_found", (errors[-1] if errors else "no in-scope thread matches")
        if request.get("task") == "draft_reply" and not values.get("reply_draft"):
            errors = values.get("errors") or []
            return "no_draft", (errors[-1] if errors else "no reply could be drafted")
        return None


class FileStepAdapter:
    """A file step: the File subgraph plans its own operations from the person's words."""

    def initial_state(
        self, step: PlanStep, inputs: Mapping[str, StepRef], user_id: UUID | None
    ) -> dict[str, Any]:
        return {"request": step.instruction, "user_id": str(user_id) if user_id else None}

    def outputs(self, values: Mapping[str, Any], effects: list[StepEffect]) -> dict[str, StepRef]:
        raw = values.get("plan")
        if not raw:
            return {}
        try:
            digest = FilePlan.model_validate(raw).plan_hash()[:32]
        except ValueError:
            return {}
        return _compact({"file_plan": _ref(StepRefKind.FILE_PLAN, digest, "trusted")})

    def failure(self, values: Mapping[str, Any]) -> tuple[str, str] | None:
        if not values.get("plan"):
            errors = values.get("plan_errors") or []
            return "no_plan", (errors[-1] if errors else "no file plan could be made")
        if values.get("verified") is False:
            return "not_verified", "the file changes could not be verified"
        return None


class JobStepAdapter:
    """A job-search step: the search runs from the person's words."""

    def initial_state(
        self, step: PlanStep, inputs: Mapping[str, StepRef], user_id: UUID | None
    ) -> dict[str, Any]:
        return {
            "user_id": str(user_id) if user_id else None,
            "query": step.instruction,
            "prepare_application": False,
        }

    def outputs(self, values: Mapping[str, Any], effects: list[StepEffect]) -> dict[str, StepRef]:
        application = values.get("application") or {}
        return _compact(
            {
                "application": _ref(
                    StepRefKind.APPLICATION,
                    application.get("application_id") or application.get("id"),
                    "trusted",
                )
            }
        )

    def failure(self, values: Mapping[str, Any]) -> tuple[str, str] | None:
        return None


STEP_ADAPTERS: Mapping[RouteDomain, StepAdapter] = {
    RouteDomain.CALENDAR: CalendarStepAdapter(),
    RouteDomain.COMMUNICATIONS: CommunicationsStepAdapter(),
    RouteDomain.FILE: FileStepAdapter(),
    RouteDomain.JOB: JobStepAdapter(),
}


# --- Reading a step thread's state ---------------------------------------------------------


def _approval_state(decision: ApprovalDecision | None) -> str:
    if decision is None:
        return "pending"
    if decision.decided_by == APPROVAL_EXPIRY_ACTOR:
        return "expired"
    return "approved" if decision.verdict is ApprovalVerdict.APPROVED else "rejected"


def _effects(
    intents: Mapping[UUID, ActionIntent],
    decisions: Mapping[UUID, ApprovalDecision],
    refusals: list[ApprovalRefusal],
    receipts: list[ActionReceipt],
    *,
    settled_by_receipts: bool,
) -> list[StepEffect]:
    """The effects that happened: ok receipts of intents that were not refused."""
    refused = {r.action_id for r in refusals}
    effects = [
        StepEffect(
            action_id=intent.action_id,
            kind=intent.kind,
            target=intent.target,
            external_reference=receipt.external_reference,
            idempotency_key=intent.idempotency_key,
        )
        for receipt in receipts
        if receipt.ok
        and receipt.action_id not in refused
        and (intent := intents.get(receipt.action_id)) is not None
    ]
    if not settled_by_receipts:
        # The File subgraph settles its approved plan with file receipts; the
        # approved, unrefused intent is the effect.
        effects.extend(
            StepEffect(
                action_id=action_id,
                kind=intent.kind,
                target=intent.target,
                idempotency_key=intent.idempotency_key,
            )
            for action_id, intent in intents.items()
            if action_id not in refused
            and (decision := decisions.get(action_id)) is not None
            and decision.authorizes(intent)
        )
    return effects


def _status(
    parked: bool,
    open_requests: int,
    refusals: list[ApprovalRefusal],
    receipts: list[ActionReceipt],
    adapter: StepAdapter,
    values: Mapping[str, Any],
) -> tuple[StepStatus, str, str | None]:
    """`(status, outcome, detail)`: parked, refused, failed, or done -- in that order."""
    if parked:
        return (
            StepStatus.AWAITING_APPROVAL,
            "awaiting_approval",
            f"{open_requests} approval request(s) open",
        )
    if refusals:
        reasons = {r.reason for r in refusals}
        detail = refusals[0].detail[:500]
        if RefusalReason.EXPIRED in reasons:
            return StepStatus.EXPIRED, RefusalReason.EXPIRED.value, detail
        if RefusalReason.DENIED_BY_USER in reasons:
            return StepStatus.REJECTED, RefusalReason.DENIED_BY_USER.value, detail
        return StepStatus.FAILED, refusals[0].reason.value, detail
    failed = next((r for r in receipts if not r.ok), None)
    if failed is not None:
        return StepStatus.FAILED, "action_failed", (failed.detail or "")[:500]
    failure = adapter.failure(values)
    if failure is not None:
        return StepStatus.FAILED, failure[0], (failure[1] or "")[:500]
    return StepStatus.SUCCEEDED, "done", None


def step_record_from_state(
    step: PlanStep,
    values: Mapping[str, Any],
    *,
    parked: bool,
    thread_id: str,
    adapter: StepAdapter,
) -> StepRecord:
    """The `StepRecord` a step thread's state adds up to."""
    intents = {
        i.action_id: i
        for i in (ActionIntent.model_validate(raw) for raw in values.get("pending_actions") or ())
    }
    requests = [ApprovalRequest.model_validate(r) for r in values.get("approval_requests") or ()]
    decisions = {
        d.action_id: d
        for d in (ApprovalDecision.model_validate(r) for r in values.get("approvals") or ())
    }
    refusals = [ApprovalRefusal.model_validate(r) for r in values.get("approval_refusals") or ()]
    receipts = [ActionReceipt.model_validate(r) for r in values.get("action_receipts") or ()]

    approvals = tuple(
        StepApproval(
            request_id=request.request_id,
            action_id=request.action_id,
            action_hash=request.action_hash,
            kind=request.kind,
            summary=request.summary,
            state=_approval_state(decisions.get(request.action_id)),  # type: ignore[arg-type]
            decided_by=(d.decided_by if (d := decisions.get(request.action_id)) else None),
        )
        for request in requests
        if request.action_id in intents or not intents
    )
    effects = _effects(
        intents,
        decisions,
        refusals,
        receipts,
        settled_by_receipts=("action_receipts" in values or parked),
    )
    outputs = adapter.outputs(values, effects)
    status, outcome, detail = _status(parked, len(approvals), refusals, receipts, adapter, values)
    return StepRecord(
        step_id=step.id,
        status=status,
        thread_id=thread_id,
        outcome=outcome,
        detail=detail,
        approvals=approvals,
        effects=tuple(effects),
        outputs=outputs,
    )


# --- The runner ------------------------------------------------------------------------------


class SubgraphStepRunner:
    """Runs plan steps of one domain through that domain's compiled subgraph."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        domain: RouteDomain,
        *,
        binder: StepThreadBinder | None = None,
        adapter: StepAdapter | None = None,
    ):
        if graph is None:
            raise ValueError("SubgraphStepRunner requires a compiled subgraph")
        self.graph = graph
        self.domain = domain
        self.binder = binder or LocalStepThreads()
        self.adapter = adapter or STEP_ADAPTERS[domain]

    def _config(self, thread: BoundThread) -> dict[str, Any]:
        return {"configurable": {**thread.configurable, "thread_id": thread.thread_id}}

    async def run(
        self, step: PlanStep, inputs: Mapping[str, StepRef], context: StepContext
    ) -> StepRecord:
        thread = self.binder.bind(step, context)
        config = self._config(thread)
        stored = await self.graph.aget_state(config)
        self.binder.started(thread)
        try:
            if stored.values and not stored.next:
                # Already ran to its end -- the parent died before recording it.
                logger.info("plan step '%s' already finished on '%s'", step.id, thread.thread_id)
            elif stored.next:
                logger.info(
                    "plan step '%s' has %s still to run on '%s'; continuing it",
                    step.id,
                    ", ".join(stored.next),
                    thread.thread_id,
                )
                await self.graph.ainvoke(None, config=config)
            else:
                initial = self.adapter.initial_state(step, inputs, thread.user_id)
                await self.graph.ainvoke(initial, config=config)
        except Exception:
            self.binder.settled(thread, finished=True, failed=True)
            raise
        return await self._read(step, thread, config)

    async def poll(self, step: PlanStep, record: StepRecord, context: StepContext) -> StepRecord:
        thread = self.binder.bind(step, context)
        return await self._read(step, thread, self._config(thread), settle=False)

    async def _read(
        self, step: PlanStep, thread: BoundThread, config: dict[str, Any], *, settle: bool = True
    ) -> StepRecord:
        snapshot = await self.graph.aget_state(config)
        parked = bool(snapshot.next)
        if settle:
            self.binder.settled(thread, finished=not parked)
        return step_record_from_state(
            step,
            dict(snapshot.values or {}),
            parked=parked,
            thread_id=thread.thread_id,
            adapter=self.adapter,
        )


__all__ = [
    "STEP_ADAPTERS",
    "STEP_THREAD_IDS",
    "BoundThread",
    "CalendarStepAdapter",
    "CommunicationsStepAdapter",
    "FileStepAdapter",
    "JobStepAdapter",
    "LocalStepThreads",
    "PlanStepRunner",
    "StepAdapter",
    "StepContext",
    "StepThreadBinder",
    "SubgraphStepRunner",
    "step_record_from_state",
    "step_task_key",
]
