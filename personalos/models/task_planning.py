"""The Supervisor's planning boundary: a request in, a cross-domain `TaskPlan` out.

`TaskPlanner` turns one request into a `PlanDraft` -- or `None`, meaning the
request is one domain's work and goes down the ordinary routing path. The
planner is shown the person's request and the domains it may use, and nothing
else: no mail, no document, no earlier step's output. A step's instruction is
the person's own words for that step, which is why a plan cannot carry content
from one step into another; refs do that (`personalos.domain.task_plan`).

`ResilientTaskPlanner` is what the Supervisor is given. It validates every
draft (`validate_task_plan`: the JSON schema, the step cap, cycles, unknown
domains, refs), and an invalid one is handed back to the planner with every
issue listed, up to `repair_attempts` times. A plan still invalid after that
is reported as invalid -- never trimmed or guessed into shape -- and the
Supervisor asks the person to clarify. A planner that raises (a provider
outage) falls back to the deterministic `RuleBasedTaskPlanner`.

`RuleBasedTaskPlanner` splits a request into clauses at conjunctions, gives
each clause a domain, merges neighbouring clauses of one domain, and returns
a plan only when that leaves two or more steps. "Then" makes a step depend on
the one before it; plain "and" leaves them independent, so each runs (and
raises its own approval) without waiting for the other. A calendar step that
creates something is marked `PROPOSE_RELEASE`: if the rest of the plan does
not complete, releasing it is proposed rather than left to be noticed.
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from personalos.domain.routing import RouteDomain
from personalos.domain.task_plan import (
    DEFAULT_MAX_PLAN_STEPS,
    DEFAULT_PLAN_REPAIR_ATTEMPTS,
    DOMAIN_OUTPUTS,
    MAX_GOAL_CHARS,
    MAX_INSTRUCTION_CHARS,
    Compensation,
    DraftStep,
    PlanDraft,
    StepInput,
    TaskPlan,
    TaskPlanInvalid,
    validate_task_plan,
)
from personalos.models.routing import KeywordIntentClassifier
from personalos.models.structured import DEFAULT_MAX_ATTEMPTS, SCHEMA_ERRORS, ainvoke_structured
from personalos.observability.tracing import model_name, model_span

logger = logging.getLogger(__name__)


class TaskPlanner(Protocol):
    """Port: a request in, a draft plan out -- or `None` for a single-domain request."""

    async def plan(
        self,
        request: str,
        *,
        domains: Iterable[RouteDomain],
        max_steps: int,
        feedback: str | None = None,
    ) -> PlanDraft | dict[str, Any] | None:
        """Plan `request` over `domains`. `feedback` is why the previous draft was refused."""
        ...


class StructuredChatModel(Protocol):
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any: ...


# --- Deterministic planner --------------------------------------------------------------

#: Where a request splits into clauses. "then" (and "after that") also makes
#: the clause depend on the one before it.
_SPLIT = re.compile(
    r"\s*(?:,\s*and\s+then|,?\s+and\s+then|,\s*then|;\s*then|\.\s+then|\s+then|"
    r",?\s+after\s+that|;|,\s*and|\s+and(?:\s+also)?)\s+",
    re.IGNORECASE,
)
_SEQUENTIAL = re.compile(r"then|after\s+that", re.IGNORECASE)

_WEEKDAY = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?"
_TIME = r"(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|\d{1,2}:\d{2})"
_CALENDAR_WRITE = re.compile(
    r"\b(?:hold|block(?:\s+out)?|book|schedule|reserve|pencil\s+in|put|add|create|set\s+up|"
    r"move|reschedule|cancel)\b",
    re.IGNORECASE,
)
_WHEN = re.compile(rf"\b(?:{_WEEKDAY}|today|tomorrow|tonight|next\s+week|{_TIME})\b", re.I)
_CALENDAR_WORDS = re.compile(
    r"\b(?:calendar|availability|am i free|free (?:time|slot)|meeting|appointment|"
    r"reschedule|interview times?)\b",
    re.IGNORECASE,
)
_COMMUNICATIONS_WORDS = re.compile(
    r"\b(?:reply|respond|write back|email|e-mail|inbox|thread|triage|unread|"
    r"action items?|summari[sz]e (?:the|that|this|my) (?:email|thread|message))\b",
    re.IGNORECASE,
)
_CALENDAR_CREATE = re.compile(
    r"\b(?:hold|block(?:\s+out)?|book|schedule|reserve|pencil\s+in|put|add|create|set\s+up)\b",
    re.IGNORECASE,
)

_FALLBACK = KeywordIntentClassifier()
#: Below this, a clause the keyword classifier is unsure of is not planned.
_CLAUSE_CONFIDENCE = 0.6


def clause_domain(clause: str) -> RouteDomain | None:
    """The domain one clause belongs to, or `None` when it names none clearly."""
    if _CALENDAR_WRITE.search(clause) and _WHEN.search(clause):
        return RouteDomain.CALENDAR
    if _CALENDAR_WORDS.search(clause):
        return RouteDomain.CALENDAR
    if _COMMUNICATIONS_WORDS.search(clause):
        return RouteDomain.COMMUNICATIONS
    decision = _FALLBACK.classify(clause)
    return decision.domain if decision.confidence >= _CLAUSE_CONFIDENCE else None


def _clauses(request: str) -> list[tuple[str, bool]]:
    """`(clause, depends_on_previous)` pairs, in order."""
    parts: list[tuple[str, bool]] = []
    position = 0
    sequential_next = False
    for match in _SPLIT.finditer(request):
        text = request[position : match.start()].strip(" ,.;")
        if text:
            parts.append((text, sequential_next))
        sequential_next = bool(_SEQUENTIAL.search(match.group(0)))
        position = match.end()
    tail = request[position:].strip(" ,.;")
    if tail:
        parts.append((tail, sequential_next))
    return parts


_STEP_NAMES: dict[RouteDomain, str] = {
    RouteDomain.CALENDAR: "calendar",
    RouteDomain.COMMUNICATIONS: "reply",
    RouteDomain.FILE: "files",
    RouteDomain.JOB: "job_search",
}


class RuleBasedTaskPlanner:
    """Deterministic planner: clauses, a domain per clause, "then" as a dependency."""

    async def plan(
        self,
        request: str,
        *,
        domains: Iterable[RouteDomain],
        max_steps: int = DEFAULT_MAX_PLAN_STEPS,
        feedback: str | None = None,
    ) -> PlanDraft | None:
        with model_span("plan_task", model="rules"):
            return self._plan(request or "")

    def _plan(self, request: str) -> PlanDraft | None:
        grouped: list[tuple[RouteDomain, list[str], bool]] = []
        for clause, sequential in _clauses(request):
            domain = clause_domain(clause)
            if domain is None:
                if not grouped:
                    return None
                # An unclassifiable clause belongs to the one before it
                # ("reply to Priya and thank her").
                grouped[-1][1].append(clause)
                continue
            if grouped and grouped[-1][0] is domain:
                grouped[-1][1].append(clause)
                continue
            grouped.append((domain, [clause], sequential))
        if len(grouped) < 2:
            return None

        steps: list[DraftStep] = []
        used: dict[str, int] = {}
        for domain, clauses, sequential in grouped:
            base = _STEP_NAMES[domain]
            used[base] = used.get(base, 0) + 1
            step_id = base if used[base] == 1 else f"{base}_{used[base]}"
            instruction = " and ".join(clauses)[:MAX_INSTRUCTION_CHARS]
            depends_on: tuple[str, ...] = (steps[-1].id,) if sequential and steps else ()
            inputs: tuple[StepInput, ...] = ()
            if depends_on and domain in (RouteDomain.CALENDAR, RouteDomain.COMMUNICATIONS):
                previous = steps[-1]
                if "application" in previous.outputs:
                    inputs = (
                        StepInput(name="application", from_step=previous.id, output="application"),
                    )
            creates = domain is RouteDomain.CALENDAR and bool(_CALENDAR_CREATE.search(instruction))
            steps.append(
                DraftStep(
                    id=step_id,
                    domain=domain,
                    instruction=instruction,
                    inputs=inputs,
                    outputs=tuple(DOMAIN_OUTPUTS.get(domain, {})),
                    depends_on=depends_on,
                    compensation=(
                        Compensation.PROPOSE_RELEASE if creates else Compensation.LEAVE_AS_IS
                    ),
                )
            )
        return PlanDraft(goal=request.strip()[:MAX_GOAL_CHARS] or "plan", steps=tuple(steps))


# --- Model-backed planner ----------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You split a person's request into a plan of steps, each handled by one domain.

Domains you may use:
{domains}

Rules:
- Use one step per domain-sized piece of work. If the whole request belongs to \
one domain, return a plan with exactly one step.
- At most {max_steps} steps.
- Each step's `instruction` is the person's own words for that step. Never add \
content, names, addresses or message text the person did not write.
- `depends_on` lists the steps that must finish first. Only add a dependency the \
request states ("then", "after that") or that an input requires. No cycles.
- `inputs` take an earlier step's declared output, by reference. Outputs by domain: \
{outputs}.
- Set `compensation` to `propose_release` only on a calendar step that creates an \
event the rest of the plan depends on; otherwise `leave_as_is`.
"""


def _render_prompt(domains: Iterable[RouteDomain], max_steps: int) -> str:
    listed = sorted(domains, key=lambda d: d.value)
    return PLANNER_SYSTEM_PROMPT.format(
        domains="\n".join(f"- {d.value}" for d in listed),
        max_steps=max_steps,
        outputs="; ".join(f"{d.value}: {sorted(DOMAIN_OUTPUTS.get(d, {}))}" for d in listed),
    )


class StructuredLLMTaskPlanner:
    """Plans by constraining a chat model to the `PlanDraft` schema."""

    def __init__(self, model: StructuredChatModel, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        if model is None:
            raise ValueError("StructuredLLMTaskPlanner requires a chat model")
        self.max_attempts = max_attempts
        self._structured = model.with_structured_output(PlanDraft)
        self._model_name = model_name(model)

    async def plan(
        self,
        request: str,
        *,
        domains: Iterable[RouteDomain],
        max_steps: int = DEFAULT_MAX_PLAN_STEPS,
        feedback: str | None = None,
    ) -> PlanDraft | None:
        domains = tuple(domains)
        human = f"Request: {request}"
        if feedback:
            human += (
                f"\n\nYour previous plan was refused for these reasons: {feedback}\n"
                "Return a corrected plan."
            )
        draft = await ainvoke_structured(
            self._structured,
            [("system", _render_prompt(domains, max_steps)), ("human", human)],
            PlanDraft,
            span="plan_task",
            model=self._model_name,
            max_attempts=self.max_attempts,
        )
        return draft if len(draft.steps) > 1 else None


# --- Resilient planner -------------------------------------------------------------------


@dataclass(frozen=True)
class TaskPlanningOutcome:
    """What planning came to: a single-domain request, a valid plan, or an invalid one."""

    kind: Literal["single", "plan", "invalid"]
    plan: TaskPlan | None = None
    source: str = "none"
    #: How many drafts were asked for.
    attempts: int = 0
    #: Every refusal on the way, in order.
    errors: tuple[str, ...] = field(default=())


class ResilientTaskPlanner:
    """The planner, validated and repaired; then clarification, never a guess."""

    def __init__(
        self,
        primary: TaskPlanner | None = None,
        *,
        fallback: TaskPlanner | None = None,
        repair_attempts: int = DEFAULT_PLAN_REPAIR_ATTEMPTS,
    ):
        if repair_attempts < 0:
            raise ValueError("repair_attempts must not be negative")
        self.fallback = fallback or RuleBasedTaskPlanner()
        self.primary = primary or self.fallback
        self.repair_attempts = repair_attempts

    async def plan(
        self, request: str, *, domains: Iterable[RouteDomain], max_steps: int
    ) -> TaskPlanningOutcome:
        domains = frozenset(domains)
        errors: list[str] = []
        feedback: str | None = None
        planner, source = self.primary, "primary"
        attempts = 0
        while attempts < 1 + self.repair_attempts:
            attempts += 1
            try:
                raw = await planner.plan(
                    request, domains=domains, max_steps=max_steps, feedback=feedback
                )
                if raw is None:
                    return TaskPlanningOutcome("single", None, source, attempts, tuple(errors))
                plan = validate_task_plan(raw, routable_domains=domains, max_steps=max_steps)
                return TaskPlanningOutcome("plan", plan, source, attempts, tuple(errors))
            except TaskPlanInvalid as exc:
                issues = exc.issues
            except SCHEMA_ERRORS as exc:
                issues = (f"output did not match the plan schema ({exc})"[:500],)
            except Exception as exc:  # noqa: BLE001 - a provider failure falls back
                if planner is self.fallback:
                    raise
                logger.warning("task planner failed; falling back to rules: %s", exc)
                errors.append(f"{source}: {type(exc).__name__}")
                planner, source = self.fallback, "fallback"
                attempts -= 1
                continue
            errors.extend(f"{source}: {issue}" for issue in issues)
            feedback = "; ".join(issues)[:2000]
            logger.info("plan attempt %d refused: %s", attempts, feedback)
        return TaskPlanningOutcome("invalid", None, source, attempts, tuple(errors))


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "ResilientTaskPlanner",
    "RuleBasedTaskPlanner",
    "StructuredLLMTaskPlanner",
    "TaskPlanner",
    "TaskPlanningOutcome",
    "clause_domain",
]
