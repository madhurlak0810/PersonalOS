"""Cross-domain task plans: a typed DAG of steps, each run by one domain subgraph.

A request like "reply to the recruiter and hold Thursday at 2pm" spans two
domains. The Supervisor plans it as a `TaskPlan`: a bounded, acyclic set of
`PlanStep`s, each tagged with the domain that runs it, the inputs it takes
from earlier steps, the outputs it hands on, and what it depends on. Each
step then runs through that domain's existing subgraph, on a thread of its
own under the conversation's one workflow.

What this module fixes, so the graph layer cannot drift from it:

**A plan is validated before anything runs.** `parse_plan_draft` checks the
planner's raw output against `PLAN_DRAFT_JSON_SCHEMA` (what a model is
constrained to) and then rebuilds it as a typed draft; `validate_task_plan`
adds what a schema cannot say -- at most `max_steps` steps, only domains this
build can run, unique ids, dependencies that exist and form no cycle, inputs
drawn only from a step's own dependencies and only of a kind the consuming
domain accepts. Every problem is reported at once (`TaskPlanInvalid.issues`),
so a repair attempt can fix them together.

**Data moves by reference.** A step's outputs are `StepRef`s -- a kind and an
id (a Gmail thread id, a calendar event id), never content. A downstream step
is handed the ids, mapped onto its subgraph's structured state fields; nothing
an earlier step read is pasted into a later step's instruction. A ref whose
source is outside content (mail) is flagged `untrusted`.

**Each effect is approved on its own.** A plan has no approval of its own: a
step's subgraph raises one `ApprovalRequest` per action, bound to that
action's hash, exactly as it does when it runs alone. `StepRecord.approvals`
lists them per step.

**A refusal stops what depends on it, and only that.** `settle_records`
cancels (or, if the step asks, pauses) the steps downstream of a rejected,
expired or failed step; independent steps carry on.

**A partial result is never cleaned up silently.** When a plan ends with
something not done, `plan_compensations` lists every effect that did happen.
One whose step asked for `PROPOSE_RELEASE` and that can be undone (a calendar
hold) gets a compensation step that *proposes* the release -- itself a write
that waits for its own approval. Everything else is left as is, and listed.

Pure: no I/O and nothing internal imported but `domain`.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.job_search import ActionKind
from personalos.domain.routing import RouteDomain

#: The most steps a plan may hold when the deployment does not configure its
#: own cap (`SUPERVISOR_MAX_PLAN_STEPS`).
DEFAULT_MAX_PLAN_STEPS = 6

#: The highest cap a deployment may configure. A plan is a bounded unit of
#: work a person reads before it runs, not a script.
MAX_PLAN_STEPS_CEILING = 12

#: How many times the Supervisor re-asks a planner whose plan did not
#: validate, by default, before it asks the person instead.
DEFAULT_PLAN_REPAIR_ATTEMPTS = 2

MAX_INSTRUCTION_CHARS = 500
MAX_GOAL_CHARS = 200

_STEP_ID = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_REF_ID = re.compile(r"^[A-Za-z0-9_.:@=+/-]{1,256}$")


class TaskPlanInvalid(ValidationFailed, ValueError):
    """A plan failed validation. `issues` lists every problem found, not just the first."""

    def __init__(self, issues: Iterable[str]):
        self.issues: tuple[str, ...] = tuple(issues) or ("plan is invalid",)
        super().__init__("; ".join(self.issues))


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- References -----------------------------------------------------------------------


class StepRefKind(str, Enum):
    """What a reference passed between steps points at."""

    EMAIL_THREAD = "email_thread"
    REPLY_DRAFT = "reply_draft"
    CALENDAR_EVENT = "calendar_event"
    APPLICATION = "application"
    FILE_PLAN = "file_plan"


ContentTrust = Literal["trusted", "untrusted"]


class StepRef(_Value):
    """An id an earlier step produced. Never content.

    `content_trust` says where what it points at came from: `untrusted` for
    anything read from outside (a mail thread, a draft written over one),
    `trusted` for something this system made itself (an event it created).
    A downstream step treats whatever it reads through an untrusted ref as data.
    """

    kind: StepRefKind
    id: str
    content_trust: ContentTrust = "untrusted"

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not _REF_ID.match(value):
            raise ValueError("a step reference id must be a bare identifier, not content")
        return value


#: The outputs a step of each domain can hand on, by name, and what each is.
DOMAIN_OUTPUTS: Mapping[RouteDomain, Mapping[str, StepRefKind]] = {
    RouteDomain.CALENDAR: {"event": StepRefKind.CALENDAR_EVENT},
    RouteDomain.COMMUNICATIONS: {
        "thread": StepRefKind.EMAIL_THREAD,
        "reply_draft": StepRefKind.REPLY_DRAFT,
        "application": StepRefKind.APPLICATION,
    },
    RouteDomain.FILE: {"file_plan": StepRefKind.FILE_PLAN},
    RouteDomain.JOB: {"application": StepRefKind.APPLICATION},
}

#: The inputs a step of each domain accepts, by name, and the kind each must be.
DOMAIN_INPUTS: Mapping[RouteDomain, Mapping[str, StepRefKind]] = {
    RouteDomain.CALENDAR: {
        "event": StepRefKind.CALENDAR_EVENT,
        "application": StepRefKind.APPLICATION,
    },
    RouteDomain.COMMUNICATIONS: {
        "thread": StepRefKind.EMAIL_THREAD,
        "application": StepRefKind.APPLICATION,
    },
    RouteDomain.FILE: {},
    RouteDomain.JOB: {},
}


# --- Plans ---------------------------------------------------------------------------


class Compensation(str, Enum):
    """What happens to a step's effects when the plan they were part of does not finish."""

    #: Keep it, and list it in the plan's failure view.
    LEAVE_AS_IS = "leave_as_is"
    #: Propose undoing it -- a write of its own, which waits for its own approval.
    PROPOSE_RELEASE = "propose_release"


class UpstreamFailure(str, Enum):
    """What a step does when a step it depends on was rejected, expired or failed."""

    #: Do not run; recorded as skipped.
    CANCEL = "cancel"
    #: Do not run yet; recorded as paused until the plan is asked again.
    PAUSE = "pause"


#: The action kinds a `PROPOSE_RELEASE` step can actually release, and the
#: domain that proposes the release.
RELEASABLE_KINDS: Mapping[ActionKind, RouteDomain] = {
    ActionKind.CREATE_CALENDAR_EVENT: RouteDomain.CALENDAR,
}

#: Domains whose steps may ask for `PROPOSE_RELEASE`.
RELEASING_DOMAINS: frozenset[RouteDomain] = frozenset(RELEASABLE_KINDS.values())


class StepInput(_Value):
    """One input a step takes: a named output of a step it depends on."""

    name: str = Field(min_length=1, max_length=32)
    from_step: str = Field(min_length=1, max_length=48)
    output: str = Field(min_length=1, max_length=32)


class DraftStep(_Value):
    """One step, as a planner proposes it."""

    id: str = Field(min_length=1, max_length=48)
    domain: RouteDomain
    #: What the step is to do, in the person's own words. Never mail, never a
    #: document: the planner is shown the request and nothing else.
    instruction: str = Field(min_length=1, max_length=MAX_INSTRUCTION_CHARS)
    inputs: tuple[StepInput, ...] = ()
    outputs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    compensation: Compensation = Compensation.LEAVE_AS_IS
    on_upstream_failure: UpstreamFailure = UpstreamFailure.CANCEL


class PlanDraft(_Value):
    """What a planner returns: a goal and its steps. Validated before anything runs."""

    goal: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)
    steps: tuple[DraftStep, ...] = Field(min_length=1)


#: The JSON schema a raw plan is checked against, and a model is constrained to.
PLAN_DRAFT_JSON_SCHEMA: dict[str, Any] = PlanDraft.model_json_schema()


class PlanStep(DraftStep):
    """A validated step. Compensation steps also name what they compensate."""

    #: The step whose effect this one proposes to release; set only on the
    #: compensation steps `plan_compensations` adds, never by a planner.
    compensates: str | None = None
    #: The effect it releases, by reference.
    target: StepRef | None = None


class TaskPlan(_Value):
    """A validated plan: bounded, acyclic, every reference resolvable."""

    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...]

    def step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    @property
    def domains(self) -> tuple[RouteDomain, ...]:
        return tuple(dict.fromkeys(step.domain for step in self.steps))

    def topological_order(self) -> list[str]:
        order = _topological_order({s.id: set(s.depends_on) for s in self.steps})
        if order is None:  # pragma: no cover - validation already refused a cycle
            raise TaskPlanInvalid(["plan has a dependency cycle"])
        return order

    def dependents(self, step_id: str) -> set[str]:
        """Every step that depends on `step_id`, directly or not."""
        found: set[str] = set()
        frontier = {step_id}
        while frontier:
            frontier = {
                s.id for s in self.steps if set(s.depends_on) & frontier and s.id not in found
            }
            found |= frontier
        return found


def plan_id_for(draft: PlanDraft) -> str:
    """A stable id for a plan: the hash of its draft, so the same plan has the same id."""
    canonical = json.dumps(draft.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return f"plan-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def parse_plan_draft(raw: Any) -> PlanDraft:
    """Check a raw plan against `PLAN_DRAFT_JSON_SCHEMA`, then build the typed draft."""
    if isinstance(raw, PlanDraft):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise TaskPlanInvalid([f"a plan must be an object, not {type(raw).__name__}"])
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a core dependency
        jsonschema = None
    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(PLAN_DRAFT_JSON_SCHEMA)
        errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
        if errors:
            raise TaskPlanInvalid(
                f"schema: {'/'.join(str(p) for p in e.absolute_path) or '<plan>'}: {e.message}"
                for e in errors[:10]
            )
    try:
        return PlanDraft.model_validate(raw)
    except ValidationError as exc:
        raise TaskPlanInvalid(
            f"schema: {'/'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:10]
        ) from exc


def validate_task_plan(
    raw: Any,
    *,
    routable_domains: Iterable[RouteDomain],
    max_steps: int = DEFAULT_MAX_PLAN_STEPS,
) -> TaskPlan:
    """The plan, if it is one this build can run. `TaskPlanInvalid` with every issue if not."""
    if not 1 <= max_steps <= MAX_PLAN_STEPS_CEILING:
        raise ValueError(f"max_steps must be between 1 and {MAX_PLAN_STEPS_CEILING}")
    draft = parse_plan_draft(raw)
    routable = frozenset(routable_domains)
    issues: list[str] = []

    if len(draft.steps) > max_steps:
        issues.append(f"plan has {len(draft.steps)} steps; at most {max_steps} are allowed")

    ids = [step.id for step in draft.steps]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        issues.append(f"step ids must be unique: {duplicates}")
    by_id = {step.id: step for step in draft.steps}

    for step in draft.steps:
        issues.extend(_step_issues(step, by_id, routable))
        issues.extend(_input_issues(step, by_id))

    if not duplicates:
        graph = {s.id: set(s.depends_on) & set(by_id) for s in draft.steps}
        if _topological_order(graph) is None:
            issues.append(f"plan has a dependency cycle among {sorted(_cycle_members(graph))}")

    if issues:
        raise TaskPlanInvalid(issues)
    return TaskPlan(
        plan_id=plan_id_for(draft),
        goal=draft.goal,
        steps=tuple(PlanStep(**step.model_dump()) for step in draft.steps),
    )


def _step_issues(
    step: DraftStep, by_id: Mapping[str, DraftStep], routable: frozenset[RouteDomain]
) -> list[str]:
    """What is wrong with one step on its own: id, domain, dependencies, outputs, release."""
    where = f"step '{step.id}'"
    issues: list[str] = []
    if not _STEP_ID.match(step.id):
        issues.append(f"{where}: id must be lowercase letters, digits and '_'")
    if step.domain not in routable:
        issues.append(
            f"{where}: domain '{step.domain.value}' is not one this build can run; "
            f"use one of {sorted(d.value for d in routable)}"
        )
    if not step.instruction.strip():
        issues.append(f"{where}: instruction must not be blank")
    unknown = sorted(set(step.depends_on) - set(by_id))
    if unknown:
        issues.append(f"{where}: depends on unknown step(s) {unknown}")
    if step.id in step.depends_on:
        issues.append(f"{where}: depends on itself")
    offered = DOMAIN_OUTPUTS.get(step.domain, {})
    issues.extend(
        f"{where}: a {step.domain.value} step cannot output '{output}' "
        f"(it can output {sorted(offered)})"
        for output in step.outputs
        if output not in offered
    )
    if step.compensation is Compensation.PROPOSE_RELEASE and step.domain not in RELEASING_DOMAINS:
        issues.append(
            f"{where}: only {sorted(d.value for d in RELEASING_DOMAINS)} steps can be "
            f"released; a {step.domain.value} step's effects are left as is"
        )
    return issues


def _input_issues(step: DraftStep, by_id: Mapping[str, DraftStep]) -> list[str]:
    """What is wrong with a step's inputs: each must be a declared output of a dependency."""
    where = f"step '{step.id}'"
    accepted = DOMAIN_INPUTS.get(step.domain, {})
    issues: list[str] = []
    for item in step.inputs:
        if item.name not in accepted:
            issues.append(
                f"{where}: a {step.domain.value} step takes no input '{item.name}' "
                f"(it takes {sorted(accepted)})"
            )
            continue
        if item.from_step not in step.depends_on:
            issues.append(
                f"{where}: input '{item.name}' comes from '{item.from_step}', which it "
                f"does not depend on"
            )
        source = by_id.get(item.from_step)
        if source is None:
            continue
        if item.output not in source.outputs:
            issues.append(
                f"{where}: input '{item.name}' names output '{item.output}', which "
                f"step '{source.id}' does not declare"
            )
            continue
        produced = DOMAIN_OUTPUTS.get(source.domain, {}).get(item.output)
        if produced is not None and produced is not accepted[item.name]:
            issues.append(
                f"{where}: input '{item.name}' must be a {accepted[item.name].value}, "
                f"not a {produced.value}"
            )
    return issues


def _topological_order(graph: Mapping[str, set[str]]) -> list[str] | None:
    remaining = {node: set(deps) for node, deps in graph.items()}
    ordered: list[str] = []
    while remaining:
        ready = sorted(node for node, deps in remaining.items() if not deps)
        if not ready:
            return None
        for node in ready:
            del remaining[node]
        for deps in remaining.values():
            deps.difference_update(ready)
        ordered.extend(ready)
    return ordered


def _cycle_members(graph: Mapping[str, set[str]]) -> set[str]:
    remaining = {node: set(deps) for node, deps in graph.items()}
    while True:
        ready = {node for node, deps in remaining.items() if not deps}
        if not ready:
            return set(remaining)
        for node in ready:
            del remaining[node]
        for deps in remaining.values():
            deps.difference_update(ready)


# --- Execution records ------------------------------------------------------------------


class StepStatus(str, Enum):
    """Where one step of a plan stands. Wire-visible."""

    PENDING = "pending"
    #: Its subgraph has raised its approval request(s) and is parked on them.
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    #: A person declined one of its actions.
    REJECTED = "rejected"
    #: Nobody answered its approval before the request expired.
    EXPIRED = "expired"
    #: It ran and could not do what it was asked (a conflict, no such thread).
    FAILED = "failed"
    #: Not run: something it depends on did not succeed.
    SKIPPED = "skipped"
    #: Not run yet: something it depends on did not succeed, and it asked to wait.
    PAUSED = "paused"


#: A step that ended without doing what it was asked.
UNSUCCESSFUL: frozenset[StepStatus] = frozenset(
    {StepStatus.REJECTED, StepStatus.EXPIRED, StepStatus.FAILED, StepStatus.SKIPPED}
)
#: A step that is finished, one way or the other.
TERMINAL: frozenset[StepStatus] = UNSUCCESSFUL | {StepStatus.SUCCEEDED}

ApprovalState = Literal["pending", "approved", "rejected", "expired"]


class StepApproval(_Value):
    """One approval request a step raised: its own, bound to its own action hash."""

    request_id: UUID
    action_id: UUID
    action_hash: str
    kind: ActionKind
    summary: str
    state: ApprovalState = "pending"
    decided_by: str | None = None


class StepEffect(_Value):
    """One external effect a step made, as its receipt reported it."""

    action_id: UUID
    kind: ActionKind
    target: str
    external_reference: str | None = None
    idempotency_key: str | None = None


class StepRecord(_Value):
    """What became of one step: its status, its approvals, its effects, its outputs."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    #: The thread its subgraph runs on.
    thread_id: str | None = None
    #: What its subgraph concluded, as a closed label (`conflict`, `proposed`).
    outcome: str | None = None
    detail: str | None = None
    approvals: tuple[StepApproval, ...] = ()
    effects: tuple[StepEffect, ...] = ()
    outputs: dict[str, StepRef] = Field(default_factory=dict)

    @property
    def awaiting(self) -> bool:
        return self.status is StepStatus.AWAITING_APPROVAL


def initial_records(plan: TaskPlan) -> dict[str, StepRecord]:
    return {step.id: StepRecord(step_id=step.id) for step in plan.steps}


def settle_records(plan: TaskPlan, records: Mapping[str, StepRecord]) -> dict[str, StepRecord]:
    """Cancel or pause every pending step downstream of one that did not succeed.

    Only dependents are touched: a step with no path to the failure keeps
    running. Repeated to a fixed point, so a skip propagates down a chain.
    """
    settled = dict(records)
    changed = True
    while changed:
        changed = False
        for step in plan.steps:
            record = settled.get(step.id) or StepRecord(step_id=step.id)
            if record.status is not StepStatus.PENDING:
                continue
            blocked = [
                dep
                for dep in step.depends_on
                if (settled.get(dep) or StepRecord(step_id=dep)).status
                in UNSUCCESSFUL | {StepStatus.PAUSED}
            ]
            if not blocked:
                continue
            status = (
                StepStatus.PAUSED
                if step.on_upstream_failure is UpstreamFailure.PAUSE
                else StepStatus.SKIPPED
            )
            settled[step.id] = record.model_copy(
                update={
                    "status": status,
                    "detail": f"not run: {', '.join(sorted(blocked))} did not succeed",
                }
            )
            changed = True
    return settled


def ready_steps(plan: TaskPlan, records: Mapping[str, StepRecord]) -> list[str]:
    """Pending steps whose dependencies have all succeeded, in plan order."""
    ready: list[str] = []
    for step in plan.steps:
        record = records.get(step.id)
        if record is None or record.status is not StepStatus.PENDING:
            continue
        if all(
            (records.get(dep) or StepRecord(step_id=dep)).status is StepStatus.SUCCEEDED
            for dep in step.depends_on
        ):
            ready.append(step.id)
    return ready


def awaiting_steps(plan: TaskPlan, records: Mapping[str, StepRecord]) -> list[str]:
    return [s.id for s in plan.steps if (records.get(s.id) or StepRecord(step_id=s.id)).awaiting]


def step_inputs(
    plan: TaskPlan, step: PlanStep, records: Mapping[str, StepRecord]
) -> dict[str, StepRef]:
    """The refs a step is handed: its declared inputs, resolved from its dependencies' outputs."""
    resolved: dict[str, StepRef] = {}
    if step.target is not None and step.compensates is not None:
        resolved["event"] = step.target
    for item in step.inputs:
        record = records.get(item.from_step)
        ref = record.outputs.get(item.output) if record is not None else None
        if ref is not None:
            resolved[item.name] = ref
    return resolved


# --- Partial failure ------------------------------------------------------------------


class DanglingEffect(_Value):
    """An effect that happened in a plan that did not finish, and what is proposed for it."""

    step_id: str
    effect: StepEffect
    compensation: Compensation
    #: The compensation step proposing its release, when there is one.
    release_step_id: str | None = None
    note: str


def _is_primary(step: PlanStep) -> bool:
    return step.compensates is None


def plan_incomplete(plan: TaskPlan, records: Mapping[str, StepRecord]) -> bool:
    """True once some step the person asked for ended without succeeding."""
    return any(
        (records.get(s.id) or StepRecord(step_id=s.id)).status in UNSUCCESSFUL | {StepStatus.PAUSED}
        for s in plan.steps
        if _is_primary(s)
    )


def _primary_settled(plan: TaskPlan, records: Mapping[str, StepRecord]) -> bool:
    return all(
        (records.get(s.id) or StepRecord(step_id=s.id)).status in TERMINAL | {StepStatus.PAUSED}
        for s in plan.steps
        if _is_primary(s)
    )


def _release_step_id(step_id: str, index: int) -> str:
    return f"release_{step_id}"[:44] + (f"_{index}" if index else "")


def plan_compensations(
    plan: TaskPlan, records: Mapping[str, StepRecord]
) -> tuple[tuple[DanglingEffect, ...], tuple[PlanStep, ...]]:
    """The dangling effects of an unfinished plan, and the release steps not yet proposed.

    Nothing until every step the person asked for has settled -- an open
    approval might still complete the plan -- and nothing for a plan that
    finished. A release is only ever a proposal: the step it adds runs through
    its domain's subgraph and waits for its own approval before anything is
    removed.
    """
    if not _primary_settled(plan, records) or not plan_incomplete(plan, records):
        return (), ()
    proposed = {s.compensates: s.id for s in plan.steps if s.compensates is not None}
    dangling: list[DanglingEffect] = []
    new_steps: list[PlanStep] = []
    for step in plan.steps:
        if not _is_primary(step):
            continue
        record = records.get(step.id)
        if record is None or record.status is not StepStatus.SUCCEEDED:
            continue
        for index, effect in enumerate(record.effects):
            releasing = RELEASABLE_KINDS.get(effect.kind)
            if (
                step.compensation is Compensation.PROPOSE_RELEASE
                and releasing is not None
                and effect.external_reference
            ):
                key = f"{step.id}#{effect.external_reference}"
                release_id = proposed.get(key)
                if release_id is None:
                    release_id = _release_step_id(step.id, index)
                    new_steps.append(
                        PlanStep(
                            id=release_id,
                            domain=releasing,
                            instruction=f"cancel the event step '{step.id}' created",
                            depends_on=(step.id,),
                            compensation=Compensation.LEAVE_AS_IS,
                            compensates=key,
                            target=StepRef(
                                kind=StepRefKind.CALENDAR_EVENT,
                                id=effect.external_reference,
                                content_trust="trusted",
                            ),
                        )
                    )
                dangling.append(
                    DanglingEffect(
                        step_id=step.id,
                        effect=effect,
                        compensation=Compensation.PROPOSE_RELEASE,
                        release_step_id=release_id,
                        note=(
                            "the rest of the plan did not complete; releasing this is "
                            "proposed, and waits for your approval"
                        ),
                    )
                )
            else:
                dangling.append(
                    DanglingEffect(
                        step_id=step.id,
                        effect=effect,
                        compensation=Compensation.LEAVE_AS_IS,
                        note=(
                            "the rest of the plan did not complete; this cannot be undone "
                            "by PersonalOS and is left as is"
                            if releasing is None
                            else "the rest of the plan did not complete; left as is"
                        ),
                    )
                )
    return tuple(dangling), tuple(new_steps)


def with_steps(plan: TaskPlan, steps: Iterable[PlanStep]) -> TaskPlan:
    """The plan with compensation steps appended. The plan id does not change."""
    return plan.model_copy(update={"steps": (*plan.steps, *steps)})


# --- The view a person reads ------------------------------------------------------------


class PlanStepView(_Value):
    step_id: str
    domain: RouteDomain
    instruction: str
    depends_on: tuple[str, ...]
    status: StepStatus
    compensates: str | None = None
    detail: str | None = None
    approvals: tuple[StepApproval, ...] = ()
    effects: tuple[StepEffect, ...] = ()


class PlanView(_Value):
    """The whole plan and where every step stands: shown before, during and after it runs."""

    plan_id: str
    goal: str
    steps: tuple[PlanStepView, ...]
    ran: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    awaiting_approval: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    paused: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    dangling: tuple[DanglingEffect, ...] = ()
    complete: bool = False


def plan_view(
    plan: TaskPlan,
    records: Mapping[str, StepRecord],
    dangling: Iterable[DanglingEffect] = (),
) -> PlanView:
    def ids(*statuses: StepStatus) -> tuple[str, ...]:
        return tuple(
            s.id
            for s in plan.steps
            if (records.get(s.id) or StepRecord(step_id=s.id)).status in statuses
        )

    views = []
    for step in plan.steps:
        record = records.get(step.id) or StepRecord(step_id=step.id)
        views.append(
            PlanStepView(
                step_id=step.id,
                domain=step.domain,
                instruction=step.instruction,
                depends_on=step.depends_on,
                status=record.status,
                compensates=step.compensates,
                detail=record.detail,
                approvals=record.approvals,
                effects=record.effects,
            )
        )
    statuses = [(records.get(s.id) or StepRecord(step_id=s.id)).status for s in plan.steps]
    return PlanView(
        plan_id=plan.plan_id,
        goal=plan.goal,
        steps=tuple(views),
        ran=ids(StepStatus.SUCCEEDED),
        pending=ids(StepStatus.PENDING),
        awaiting_approval=ids(StepStatus.AWAITING_APPROVAL),
        skipped=ids(StepStatus.SKIPPED),
        paused=ids(StepStatus.PAUSED),
        rejected=ids(StepStatus.REJECTED, StepStatus.EXPIRED),
        failed=ids(StepStatus.FAILED),
        dangling=tuple(dangling),
        complete=all(status is StepStatus.SUCCEEDED for status in statuses),
    )


# --- Audit ---------------------------------------------------------------------------

AUDIT_PLAN_PROPOSED = "plan.proposed"
AUDIT_PLAN_REJECTED = "plan.rejected"
AUDIT_STEP_PLANNED = "plan.step.planned"
AUDIT_STEP_INTENT = "plan.step.intent"
AUDIT_STEP_POLICY = "plan.step.policy"
#: Followed by the approval's state: `plan.step.approval.approved`, ...
AUDIT_STEP_APPROVAL = "plan.step.approval"
AUDIT_STEP_RECEIPT = "plan.step.receipt"
AUDIT_STEP_NOT_RUN = "plan.step.not_run"
AUDIT_EFFECT_DANGLING = "plan.effect.dangling"
AUDIT_PLAN_FINISHED = "plan.finished"

#: The policy verdict an approval-gated step action is recorded under. Every
#: action a subgraph proposes is `require_approval`; a step with no action
#: only read, which the policy allows.
POLICY_REQUIRE_APPROVAL = "require_approval"
POLICY_ALLOW = "allow"


class PlanAuditEntry(_Value):
    """One audit row a plan writes. `key` makes recording it twice a no-op."""

    key: str
    action: str
    target_ref: str
    result: Literal["success", "failure"] = "success"
    policy_decision: str | None = None
    approval_ref: str | None = None


def plan_proposed_entries(plan: TaskPlan) -> list[PlanAuditEntry]:
    """The plan, as recorded before any step runs: the whole of it, then each step."""
    outline = ", ".join(
        f"{s.id}({s.domain.value}{' after ' + '+'.join(s.depends_on) if s.depends_on else ''})"
        for s in plan.steps
    )
    entries = [
        PlanAuditEntry(
            key=f"{plan.plan_id}/proposed",
            action=AUDIT_PLAN_PROPOSED,
            target_ref=f"{plan.plan_id}: {len(plan.steps)} step(s): {outline}"[:500],
        )
    ]
    entries.extend(_step_planned(plan, step) for step in plan.steps)
    return entries


def _step_planned(plan: TaskPlan, step: PlanStep) -> PlanAuditEntry:
    return PlanAuditEntry(
        key=f"{plan.plan_id}/{step.id}/planned",
        action=AUDIT_STEP_PLANNED,
        target_ref=f"{plan.plan_id}/{step.id} [{step.domain.value}]: {step.instruction}"[:500],
    )


def plan_rejected_entry(plan_key: str, issues: Iterable[str]) -> PlanAuditEntry:
    return PlanAuditEntry(
        key=f"{plan_key}/rejected",
        action=AUDIT_PLAN_REJECTED,
        target_ref=f"{plan_key}: {'; '.join(issues)}"[:500],
        result="failure",
    )


def step_audit_entries(plan: TaskPlan, step: PlanStep, record: StepRecord) -> list[PlanAuditEntry]:
    """A step's story so far, in the order it happens: intent, policy, approval, receipt."""
    base = f"{plan.plan_id}/{step.id}"
    entries: list[PlanAuditEntry] = []
    if step.compensates is not None:
        entries.append(_step_planned(plan, step))
    for approval in record.approvals:
        what = f"{base}: {approval.kind.value}: {approval.summary}"[:500]
        entries.append(
            PlanAuditEntry(
                key=f"{base}/intent/{approval.action_id}",
                action=AUDIT_STEP_INTENT,
                target_ref=what,
            )
        )
        entries.append(
            PlanAuditEntry(
                key=f"{base}/policy/{approval.action_id}",
                action=AUDIT_STEP_POLICY,
                target_ref=f"{base}: {approval.kind.value} hash {approval.action_hash[:16]}",
                policy_decision=POLICY_REQUIRE_APPROVAL,
                approval_ref=str(approval.request_id),
            )
        )
        if approval.state != "pending":
            entries.append(
                PlanAuditEntry(
                    key=f"{base}/approval/{approval.request_id}",
                    action=f"{AUDIT_STEP_APPROVAL}.{approval.state}",
                    target_ref=(
                        f"{base}: {approval.kind.value} {approval.state}"
                        f"{' by ' + approval.decided_by if approval.decided_by else ''}"
                    )[:500],
                    result="success" if approval.state == "approved" else "failure",
                    policy_decision=POLICY_REQUIRE_APPROVAL,
                    approval_ref=str(approval.request_id),
                )
            )
    if (
        record.status in TERMINAL
        and not record.approvals
        and record.status
        in (
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
        )
    ):
        entries.append(
            PlanAuditEntry(
                key=f"{base}/intent/read",
                action=AUDIT_STEP_INTENT,
                target_ref=f"{base}: {step.domain.value} step with no external effect",
            )
        )
        entries.append(
            PlanAuditEntry(
                key=f"{base}/policy/read",
                action=AUDIT_STEP_POLICY,
                target_ref=f"{base}: reads only",
                policy_decision=POLICY_ALLOW,
            )
        )
    if record.status in (StepStatus.SKIPPED, StepStatus.PAUSED):
        entries.append(
            PlanAuditEntry(
                key=f"{base}/not_run/{record.status.value}",
                action=f"{AUDIT_STEP_NOT_RUN}.{record.status.value}",
                target_ref=f"{base}: {record.detail or record.status.value}"[:500],
                result="failure",
            )
        )
    elif record.status in TERMINAL:
        refs = ", ".join(
            f"{e.kind.value}={e.external_reference or e.target}" for e in record.effects
        )
        entries.append(
            PlanAuditEntry(
                key=f"{base}/receipt",
                action=AUDIT_STEP_RECEIPT,
                target_ref=(
                    f"{base}: {record.status.value}"
                    f"{' (' + record.outcome + ')' if record.outcome else ''}"
                    f"{': ' + refs if refs else ''}"
                )[:500],
                result="success" if record.status is StepStatus.SUCCEEDED else "failure",
            )
        )
    return entries


def dangling_entries(plan: TaskPlan, dangling: Iterable[DanglingEffect]) -> list[PlanAuditEntry]:
    return [
        PlanAuditEntry(
            key=f"{plan.plan_id}/{item.step_id}/dangling/{item.effect.action_id}",
            action=AUDIT_EFFECT_DANGLING,
            target_ref=(
                f"{plan.plan_id}/{item.step_id}: {item.effect.kind.value} "
                f"{item.effect.external_reference or item.effect.target}: "
                f"{item.compensation.value}"
                f"{' via ' + item.release_step_id if item.release_step_id else ''}"
            )[:500],
            result="failure",
        )
        for item in dangling
    ]


def plan_finished_entry(plan: TaskPlan, view: "PlanView") -> PlanAuditEntry:
    return PlanAuditEntry(
        key=f"{plan.plan_id}/finished",
        action=AUDIT_PLAN_FINISHED,
        target_ref=(
            f"{plan.plan_id}: ran={list(view.ran)} rejected={list(view.rejected)} "
            f"failed={list(view.failed)} skipped={list(view.skipped)} "
            f"paused={list(view.paused)} dangling={len(view.dangling)}"
        )[:500],
        result="success" if view.complete else "failure",
    )


__all__ = [
    "AUDIT_EFFECT_DANGLING",
    "AUDIT_PLAN_FINISHED",
    "AUDIT_PLAN_PROPOSED",
    "AUDIT_PLAN_REJECTED",
    "AUDIT_STEP_APPROVAL",
    "AUDIT_STEP_INTENT",
    "AUDIT_STEP_NOT_RUN",
    "AUDIT_STEP_PLANNED",
    "AUDIT_STEP_POLICY",
    "AUDIT_STEP_RECEIPT",
    "POLICY_ALLOW",
    "POLICY_REQUIRE_APPROVAL",
    "PlanAuditEntry",
    "dangling_entries",
    "plan_finished_entry",
    "plan_proposed_entries",
    "plan_rejected_entry",
    "step_audit_entries",
    "DEFAULT_MAX_PLAN_STEPS",
    "DEFAULT_PLAN_REPAIR_ATTEMPTS",
    "DOMAIN_INPUTS",
    "DOMAIN_OUTPUTS",
    "MAX_PLAN_STEPS_CEILING",
    "PLAN_DRAFT_JSON_SCHEMA",
    "RELEASABLE_KINDS",
    "TERMINAL",
    "UNSUCCESSFUL",
    "Compensation",
    "DanglingEffect",
    "DraftStep",
    "PlanDraft",
    "PlanStep",
    "PlanStepView",
    "PlanView",
    "StepApproval",
    "StepEffect",
    "StepInput",
    "StepRecord",
    "StepRef",
    "StepRefKind",
    "StepStatus",
    "TaskPlan",
    "TaskPlanInvalid",
    "UpstreamFailure",
    "awaiting_steps",
    "initial_records",
    "parse_plan_draft",
    "plan_compensations",
    "plan_id_for",
    "plan_incomplete",
    "plan_view",
    "ready_steps",
    "settle_records",
    "step_inputs",
    "validate_task_plan",
    "with_steps",
]
