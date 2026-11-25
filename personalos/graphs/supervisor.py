"""Top-level Supervisor graph: loads context, classifies intent, routes.

The Supervisor is the entry point for every request: it loads cross-thread
context, classifies the request into a typed `RouteDecision`, and either asks
for clarification or plans and hands off a bounded unit of work to a domain
subgraph. It never touches a tool or a repository directly -- see
`docs/ARCHITECTURE_BOUNDARIES.md` -- it only classifies, plans, and delegates.

Two things keep this graph honest to that boundary:

- `classifier`, `job_subgraph`, `file_subgraph`, `communications_subgraph`
  and `calendar_subgraph` are injected ports (`IntentClassifier`,
  `JobSubgraphRunner`, `FileSubgraphRunner`, `CommunicationsSubgraphRunner`,
  `CalendarSubgraphRunner`), not concrete dependencies this module
  constructs. The composition root wires a real classifier and adapters that
  run the Job Search subgraph
  (`personalos.graphs.job_search.JobSearchSubgraphRunner`), the File subgraph
  (`personalos.graphs.file_task.FileTaskSubgraphRunner`), the Communications
  subgraph (`personalos.graphs.communications.CommunicationsSubgraphRunner`)
  and the Calendar subgraph
  (`personalos.graphs.calendar.CalendarSubgraphRunner`). A
  Supervisor built without one of the optional subgraphs treats its domain as
  unsupported and asks for clarification, exactly as it does for a reserved
  domain.
- Graph state is kept JSON-compatible (`dict`, not live `RouteDecision` /
  `TaskDAG` instances) so it round-trips cleanly through any
  `BaseCheckpointSaver`, including the durable, database-backed one in
  `personalos.persistence.checkpointer`.
"""

import hashlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from personalos.domain.routing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    SUPPORTED_ROUTE_DOMAINS,
    RouteDecision,
    RouteDomain,
    RouteReason,
    RouteResolution,
    UnsupportedRouteDomain,
    ensure_supported_domain,
    resolve_route,
    route_decision_from_mapping,
    screen_request,
)
from personalos.domain.task_plan import (
    DEFAULT_MAX_PLAN_STEPS,
    MAX_PLAN_STEPS_CEILING,
    PlanAuditEntry,
    StepRecord,
    TaskPlan,
    awaiting_steps,
    dangling_entries,
    initial_records,
    plan_compensations,
    plan_finished_entry,
    plan_proposed_entries,
    plan_rejected_entry,
    plan_view,
    ready_steps,
    settle_records,
    step_audit_entries,
    step_inputs,
    with_steps,
)
from personalos.domain.tasks import InvalidTaskDAG, TaskDAG, TaskNode
from personalos.graphs.plan_steps import PlanStepRunner, StepContext
from personalos.models.routing import IntentClassifier
from personalos.models.task_planning import ResilientTaskPlanner
from personalos.observability.tracing import traced_node

logger = logging.getLogger(__name__)

#: Long-term store namespace root for cross-thread facts. The full namespace
#: for a given actor is `(_FACTS_NAMESPACE, actor_id)`.
#:
#: Namespacing by actor rather than thread is the whole point of the store:
#: a checkpointer already persists one thread's state, so facts keyed by
#: `thread_id` would be scoped to exactly what the checkpointer covers and
#: nothing would ever be carried *across* threads. Keyed by actor, a fact
#: learned in one conversation is visible in the next one.
_FACTS_NAMESPACE = "supervisor_facts"

#: Matches `ExecutionContext.actor_id`'s default, so a run started without an
#: explicit actor reads and writes the same namespace either way.
_DEFAULT_ACTOR_ID = "system"

#: langgraph node/edge names, so they aren't inline string literals in two
#: places.
LOAD_CONTEXT = "load_context"
CLASSIFY_INTENT = "classify_intent"
PLAN_WORK = "plan_work"
RUN_JOB_SUBGRAPH = "run_job_subgraph"
RUN_FILE_SUBGRAPH = "run_file_subgraph"
RUN_COMMUNICATIONS_SUBGRAPH = "run_communications_subgraph"
RUN_CALENDAR_SUBGRAPH = "run_calendar_subgraph"
CLARIFY = "clarify"
#: The cross-domain plan path (K4), present only when a planner and step
#: runners are wired.
PLAN_TASK = "plan_task"
PUBLISH_PLAN = "publish_plan"
RUN_PLAN_STEP = "run_plan_step"
AWAIT_PLAN_STEPS = "await_plan_steps"
COLLECT_PLAN_STEPS = "collect_plan_steps"
SETTLE_PLAN = "settle_plan"

#: The interrupt `stage` a Supervisor parked on its plan's steps raises. Not an
#: approval: it carries no `requests`, so the approval queue lists nothing for
#: it -- the approvals are the steps' own, on the steps' own threads.
STAGE_PLAN_STEPS = "plan_steps"

#: What each routable domain is called in a clarification.
_DOMAIN_NAMES: dict[RouteDomain, str] = {
    RouteDomain.JOB: "job search",
    RouteDomain.FILE: "your job-search documents",
    RouteDomain.COMMUNICATIONS: "your job-search email",
    RouteDomain.CALENDAR: "your calendar",
}


class JobSubgraphRunner(Protocol):
    """Port the Supervisor uses to hand a planned task DAG to the Job Search subgraph.

    Kept as an injected port rather than a direct call into the Job Search
    subgraph so this module never has to import `persistence` or `tools` to
    build the stores and `ToolGateway` that subgraph needs -- the composition
    root binds an adapter that already has them.
    """

    async def __call__(self, task_dag: dict[str, Any], state: "SupervisorState") -> dict[str, Any]:
        """Run the planned work and return a JSON-compatible result."""
        ...


class FileSubgraphRunner(Protocol):
    """Port the Supervisor uses to hand a planned file task to the File subgraph.

    The File subgraph does its own planning -- a typed `FilePlan` over the
    files it discovers -- so what it is handed is the request, carried in the
    DAG's goal and in `state["message"]`, not a plan.
    """

    async def __call__(self, task_dag: dict[str, Any], state: "SupervisorState") -> dict[str, Any]:
        """Run the file task and return a JSON-compatible result."""
        ...


class CommunicationsSubgraphRunner(Protocol):
    """Port the Supervisor uses to hand a request to the Communications subgraph.

    Like the File subgraph, it parses and plans its own work -- which thread,
    which task -- from the request in `state["message"]`.
    """

    async def __call__(self, task_dag: dict[str, Any], state: "SupervisorState") -> dict[str, Any]:
        """Run the communications task and return a JSON-compatible result."""
        ...


class CalendarSubgraphRunner(Protocol):
    """Port the Supervisor uses to hand a request to the Calendar subgraph.

    It parses its own request -- which times, in the person's zone, and which
    event -- from `state["message"]`; every write waits for approval inside it.
    """

    async def __call__(self, task_dag: dict[str, Any], state: "SupervisorState") -> dict[str, Any]:
        """Run the calendar task and return a JSON-compatible result."""
        ...


class PlanAuditRecorder(Protocol):
    """Port the Supervisor records a plan's story through, into the audit trail.

    `entries` are recorded in order, and an entry whose `key` was recorded
    before is not recorded again, so a re-run step cannot duplicate the trail.
    """

    def record(
        self,
        entries: Sequence[PlanAuditEntry],
        *,
        workflow_id: str | None,
        actor_id: str,
    ) -> None: ...


class SupervisorState(TypedDict, total=False):
    """Graph state carried between Supervisor nodes.

    Every field is JSON-compatible so the state round-trips through any
    `BaseCheckpointSaver` without registering custom types; `RouteDecision`
    and `TaskDAG` are stored as `.model_dump()` and rebuilt on demand via
    `route_decision_from_mapping` / `TaskDAG.model_validate`.
    """

    message: str
    actor_facts: dict[str, Any]
    route_decision: dict[str, Any] | None
    task_dag: dict[str, Any] | None
    clarification: str | None
    result: dict[str, Any] | None
    #: The plan path: `single` (one domain, the routing path), `plan`, or `invalid`.
    plan_kind: str | None
    #: The validated `TaskPlan`, and one `StepRecord` per step, by step id.
    task_plan: dict[str, Any] | None
    plan_records: dict[str, dict[str, Any]]
    #: Why the plan was refused, every attempt.
    plan_errors: list[str]
    #: Audit entry keys already recorded, so nothing is recorded twice.
    plan_audit_keys: list[str]


def _build_job_search_dag(message: str) -> TaskDAG:
    """The bounded plan for a job-domain request.

    A fixed, linear, four-step DAG (prepare -> search -> scrape -> filter),
    not an open-ended loop the model could extend indefinitely.
    """
    goal = f"job_search: {message.strip()}" if message.strip() else "job_search"
    return TaskDAG(
        goal=goal[:200],
        tasks=(
            TaskNode(id="prepare", name="Prepare search parameters"),
            TaskNode(id="search", name="Search job listings", depends_on=("prepare",)),
            TaskNode(id="enrich", name="Scrape job details", depends_on=("search",)),
            TaskNode(id="filter", name="Filter and rank matches", depends_on=("enrich",)),
        ),
    )


def _build_file_task_dag(message: str) -> TaskDAG:
    """The bounded plan for a file-domain request.

    Fixed and linear, like the job search's: discover, plan, execute, verify.
    The File subgraph's own planner decides *which* files and operations; this
    DAG only says the task runs as one bounded unit.
    """
    goal = f"file_task: {message.strip()}" if message.strip() else "file_task"
    return TaskDAG(
        goal=goal[:200],
        tasks=(
            TaskNode(id="discover", name="Discover job-search documents"),
            TaskNode(id="plan", name="Plan file operations", depends_on=("discover",)),
            TaskNode(id="execute", name="Execute the approved plan", depends_on=("plan",)),
            TaskNode(id="verify", name="Verify the changes", depends_on=("execute",)),
        ),
    )


def _build_communications_dag(message: str) -> TaskDAG:
    """The bounded plan for a communications request: read, analyze, draft, send.

    Fixed and linear. Which thread and which of those steps actually run is
    the Communications subgraph's own parse of the request; a send always
    waits for approval inside it.
    """
    goal = f"communications: {message.strip()}" if message.strip() else "communications"
    return TaskDAG(
        goal=goal[:200],
        tasks=(
            TaskNode(id="read", name="Read in-scope mail"),
            TaskNode(id="analyze", name="Triage or summarize it", depends_on=("read",)),
            TaskNode(id="draft", name="Draft a reply if asked", depends_on=("analyze",)),
            TaskNode(id="send", name="Send only once approved", depends_on=("draft",)),
        ),
    )


def _build_calendar_dag(message: str) -> TaskDAG:
    """The bounded plan for a calendar request: read, plan, write only once approved.

    Fixed and linear. Whether the request is a question (answered from
    free/busy, nothing written) or a change is the Calendar subgraph's own
    parse; a write always waits for approval inside it.
    """
    goal = f"calendar: {message.strip()}" if message.strip() else "calendar"
    return TaskDAG(
        goal=goal[:200],
        tasks=(
            TaskNode(id="resolve", name="Resolve the times in your time zone"),
            TaskNode(id="read", name="Read free/busy and events", depends_on=("resolve",)),
            TaskNode(id="plan", name="Answer, or plan the change", depends_on=("read",)),
            TaskNode(id="write", name="Write only once approved", depends_on=("plan",)),
        ),
    )


def _join(names: Sequence[str], conjunction: str = "and") -> str:
    """`a`, `a and b`, `a, b and c`."""
    if len(names) < 3:
        return f" {conjunction} ".join(names)
    return f"{', '.join(names[:-1])} {conjunction} {names[-1]}"


def _request_key(message: str) -> str:
    """A stable name for a request whose plan was refused: there is no plan id to use."""
    return f"plan-request-{hashlib.sha256(message.encode('utf-8')).hexdigest()[:16]}"


def _load_plan(state: SupervisorState) -> TaskPlan:
    raw = state.get("task_plan")
    if not raw:  # pragma: no cover - the plan path only runs with a plan
        raise InvalidTaskDAG("a plan step was reached without a validated plan")
    return TaskPlan.model_validate(raw)


def _load_records(state: SupervisorState) -> dict[str, StepRecord]:
    return {
        step_id: StepRecord.model_validate(raw)
        for step_id, raw in (state.get("plan_records") or {}).items()
    }


def _dump_records(records: Mapping[str, StepRecord]) -> dict[str, dict[str, Any]]:
    return {step_id: record.model_dump(mode="json") for step_id, record in records.items()}


def _plan_as_dag(plan: TaskPlan) -> TaskDAG:
    """The plan as the bounded `TaskDAG` every Supervisor run records."""
    return TaskDAG(
        goal=plan.goal,
        tasks=tuple(
            TaskNode(
                id=step.id,
                name=f"[{step.domain.value}] {step.instruction}"[:200],
                depends_on=step.depends_on,
            )
            for step in plan.steps
        ),
    )


def _plan_result(
    plan: TaskPlan,
    records: Mapping[str, StepRecord],
    dangling: Sequence[Any] = (),
) -> dict[str, Any]:
    """What the Supervisor records of a plan: every step and where it stands. No content."""
    view = plan_view(plan, records, dangling)
    return {
        "goal": plan.goal,
        "domain": "plan",
        "plan": view.model_dump(mode="json"),
        "awaiting_approval": bool(view.awaiting_approval),
    }


class SupervisorGraph:
    """Builds and compiles the top-level Supervisor LangGraph graph."""

    def __init__(
        self,
        classifier: IntentClassifier,
        job_subgraph: JobSubgraphRunner,
        *,
        file_subgraph: FileSubgraphRunner | None = None,
        communications_subgraph: CommunicationsSubgraphRunner | None = None,
        calendar_subgraph: CalendarSubgraphRunner | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        task_planner: ResilientTaskPlanner | None = None,
        step_runners: Mapping[RouteDomain, PlanStepRunner] | None = None,
        plan_audit: PlanAuditRecorder | None = None,
        max_plan_steps: int = DEFAULT_MAX_PLAN_STEPS,
    ):
        """Initialize with the ports this graph delegates to.

        `classifier` and `job_subgraph` are required: a Supervisor with no
        classifier would have to reach for a global default, which is exactly
        the kind of implicit wiring the composition root exists to own instead.

        `checkpointer` / `store` default to in-memory implementations so the
        graph compiles and is usable out of the box. In-memory checkpointing is
        for tests only: it loses every thread when the process ends. A real
        deployment is handed
        `personalos.persistence.checkpointer.SqlAlchemyCheckpointSaver` by the
        composition root (`personalos.bootstrap.build_durable_checkpointer`),
        and this module does not change for it -- which is the point of taking
        the port rather than constructing a saver here. The thread this graph
        runs on must be registered first; see
        `personalos.bootstrap.register_supervisor_thread`.
        """
        if classifier is None:
            raise ValueError("SupervisorGraph requires an IntentClassifier")
        if job_subgraph is None:
            raise ValueError("SupervisorGraph requires a JobSubgraphRunner")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.classifier = classifier
        self.job_subgraph = job_subgraph
        self.file_subgraph = file_subgraph
        self.communications_subgraph = communications_subgraph
        self.calendar_subgraph = calendar_subgraph
        optional = {
            RouteDomain.FILE: file_subgraph,
            RouteDomain.COMMUNICATIONS: communications_subgraph,
            RouteDomain.CALENDAR: calendar_subgraph,
        }
        #: The domains this instance can actually hand work to: the build's
        #: supported set, less any whose subgraph was not wired.
        self.routable_domains: frozenset[RouteDomain] = frozenset(
            domain
            for domain in SUPPORTED_ROUTE_DOMAINS
            if domain not in optional or optional[domain] is not None
        )
        self.confidence_threshold = confidence_threshold
        self.checkpointer = checkpointer or InMemorySaver()
        self.store = store or InMemoryStore()

        if not 1 <= max_plan_steps <= MAX_PLAN_STEPS_CEILING:
            raise ValueError(f"max_plan_steps must be between 1 and {MAX_PLAN_STEPS_CEILING}")
        self.task_planner = task_planner
        self.step_runners: dict[RouteDomain, PlanStepRunner] = {
            domain: runner
            for domain, runner in (step_runners or {}).items()
            if domain in SUPPORTED_ROUTE_DOMAINS and runner is not None
        }
        self.plan_audit = plan_audit
        self.max_plan_steps = max_plan_steps

    @property
    def plans_enabled(self) -> bool:
        """True when a request may be planned across domains: a planner, and two domains to run."""
        return self.task_planner is not None and len(self.step_runners) >= 2

    def build(self) -> CompiledStateGraph:
        """Assemble and compile the graph with this instance's checkpointer and store."""
        graph = StateGraph(SupervisorState)
        graph.add_node(
            LOAD_CONTEXT, traced_node(LOAD_CONTEXT, self._load_context, graph="supervisor")
        )
        graph.add_node(
            CLASSIFY_INTENT,
            traced_node(CLASSIFY_INTENT, self._classify_intent, graph="supervisor"),
        )
        graph.add_node(PLAN_WORK, traced_node(PLAN_WORK, self._plan_work, graph="supervisor"))
        graph.add_node(
            RUN_JOB_SUBGRAPH,
            traced_node(RUN_JOB_SUBGRAPH, self._run_job_subgraph, graph="supervisor"),
        )
        graph.add_node(
            RUN_FILE_SUBGRAPH,
            traced_node(RUN_FILE_SUBGRAPH, self._run_file_subgraph, graph="supervisor"),
        )
        graph.add_node(
            RUN_COMMUNICATIONS_SUBGRAPH,
            traced_node(
                RUN_COMMUNICATIONS_SUBGRAPH, self._run_communications_subgraph, graph="supervisor"
            ),
        )
        graph.add_node(
            RUN_CALENDAR_SUBGRAPH,
            traced_node(RUN_CALENDAR_SUBGRAPH, self._run_calendar_subgraph, graph="supervisor"),
        )
        graph.add_node(CLARIFY, traced_node(CLARIFY, self._clarify, graph="supervisor"))

        graph.add_edge(START, LOAD_CONTEXT)
        if self.plans_enabled:
            self._add_plan_nodes(graph)
        else:
            graph.add_edge(LOAD_CONTEXT, CLASSIFY_INTENT)
        graph.add_conditional_edges(
            CLASSIFY_INTENT,
            self._route_after_classification,
            {PLAN_WORK: PLAN_WORK, CLARIFY: CLARIFY},
        )
        graph.add_conditional_edges(
            PLAN_WORK,
            self._route_after_planning,
            {
                RUN_JOB_SUBGRAPH: RUN_JOB_SUBGRAPH,
                RUN_FILE_SUBGRAPH: RUN_FILE_SUBGRAPH,
                RUN_COMMUNICATIONS_SUBGRAPH: RUN_COMMUNICATIONS_SUBGRAPH,
                RUN_CALENDAR_SUBGRAPH: RUN_CALENDAR_SUBGRAPH,
            },
        )
        graph.add_edge(RUN_JOB_SUBGRAPH, END)
        graph.add_edge(RUN_FILE_SUBGRAPH, END)
        graph.add_edge(RUN_COMMUNICATIONS_SUBGRAPH, END)
        graph.add_edge(RUN_CALENDAR_SUBGRAPH, END)
        graph.add_edge(CLARIFY, END)

        return graph.compile(checkpointer=self.checkpointer, store=self.store)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _load_context(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        """Load this actor's long-term facts, carried across all of their threads.

        Keyed by `actor_id` (see `personalos.domain.context.ExecutionContext`),
        not `thread_id`, so what was learned in an earlier conversation is
        available in this one.
        """
        configurable = config.get("configurable") or {}
        actor_id = configurable.get("actor_id") or _DEFAULT_ACTOR_ID
        facts = {item.key: item.value for item in self.store.search((_FACTS_NAMESPACE, actor_id))}
        return {"actor_facts": facts}

    def _classify_intent(self, state: SupervisorState) -> dict[str, Any]:
        """Classify the message into a typed `RouteDecision`.

        A classifier failure is treated as a zero-confidence decision rather
        than propagated, so a flaky classifier degrades to "ask for
        clarification" instead of failing the whole graph run.

        The request is also screened (`screen_request`) for asking to skip an
        approval or widen tool access. The screen is deterministic and is
        applied over whatever the classifier said, so no model -- however it
        was persuaded -- can route such a request.
        """
        message = state.get("message", "")
        try:
            decision = self.classifier.classify(message)
        except Exception:
            logger.exception("intent classification failed; routing to clarification")
            decision = RouteDecision(
                domain=RouteDomain.JOB,
                confidence=0.0,
                reasoning="classification failed",
            )
        escalation = screen_request(message)
        if escalation is not None:
            decision = decision.model_copy(
                update={
                    "access_escalation": True,
                    "reasoning": f"request asks to {escalation}",
                }
            )
        return {"route_decision": decision.model_dump(mode="json")}

    def resolve(self, raw_decision: Mapping[str, Any] | None) -> RouteResolution:
        """Where this instance sends a recorded decision, and why."""
        decision = route_decision_from_mapping(dict(raw_decision)) if raw_decision else None
        return resolve_route(
            decision, threshold=self.confidence_threshold, routable=self.routable_domains
        )

    def _route_after_classification(self, state: SupervisorState) -> str:
        """Anything short of one confident, unambiguous, routable domain: clarify."""
        return PLAN_WORK if self.resolve(state.get("route_decision")).routed else CLARIFY

    def _ensure_routable(self, decision: RouteDecision) -> RouteDecision:
        """`ensure_supported_domain`, narrowed to the subgraphs this instance has."""
        ensure_supported_domain(decision)
        if decision.domain not in self.routable_domains:
            raise UnsupportedRouteDomain(
                f"domain '{decision.domain.value}' is supported but no subgraph is wired for it"
            )
        return decision

    def _route_after_planning(self, state: SupervisorState) -> str:
        """Hand the DAG to the subgraph for the decided domain."""
        decision = route_decision_from_mapping(state["route_decision"])
        if decision.domain == RouteDomain.FILE:
            return RUN_FILE_SUBGRAPH
        if decision.domain == RouteDomain.COMMUNICATIONS:
            return RUN_COMMUNICATIONS_SUBGRAPH
        if decision.domain == RouteDomain.CALENDAR:
            return RUN_CALENDAR_SUBGRAPH
        return RUN_JOB_SUBGRAPH

    def _plan_work(self, state: SupervisorState) -> dict[str, Any]:
        """Produce the bounded task DAG for a confidently-routed, supported domain."""
        raw_decision = state.get("route_decision")
        if raw_decision is None:  # pragma: no cover - the router only reaches here with one
            raise UnsupportedRouteDomain("plan_work reached without a route decision")
        decision = route_decision_from_mapping(raw_decision)
        self._ensure_routable(decision)  # re-checked: plan_work must never plan the unsupported

        if decision.domain == RouteDomain.JOB:
            dag = _build_job_search_dag(state.get("message", ""))
        elif decision.domain == RouteDomain.FILE:
            dag = _build_file_task_dag(state.get("message", ""))
        elif decision.domain == RouteDomain.COMMUNICATIONS:
            dag = _build_communications_dag(state.get("message", ""))
        elif decision.domain == RouteDomain.CALENDAR:
            dag = _build_calendar_dag(state.get("message", ""))
        else:  # pragma: no cover - unreachable while every supported domain has a planner
            raise UnsupportedRouteDomain(f"no planner registered for domain '{decision.domain}'")

        return {"task_dag": dag.model_dump(mode="json")}

    async def _run_job_subgraph(self, state: SupervisorState) -> dict[str, Any]:
        """Hand the planned DAG to the Job Search subgraph and record its result."""
        task_dag = state.get("task_dag")
        if task_dag is None:  # pragma: no cover - plan_work always sets one before this node
            raise InvalidTaskDAG("job subgraph reached without a planned task DAG")
        result = await self.job_subgraph(task_dag, state)
        return {"result": result}

    async def _run_file_subgraph(self, state: SupervisorState) -> dict[str, Any]:
        """Hand the planned file task to the File subgraph and record its result."""
        task_dag = state.get("task_dag")
        if task_dag is None:  # pragma: no cover - plan_work always sets one before this node
            raise InvalidTaskDAG("file subgraph reached without a planned task DAG")
        if self.file_subgraph is None:  # pragma: no cover - routing never reaches here without one
            raise UnsupportedRouteDomain("no file subgraph is wired")
        result = await self.file_subgraph(task_dag, state)
        return {"result": result}

    async def _run_communications_subgraph(self, state: SupervisorState) -> dict[str, Any]:
        """Hand the request to the Communications subgraph and record its result."""
        task_dag = state.get("task_dag")
        if task_dag is None:  # pragma: no cover - plan_work always sets one before this node
            raise InvalidTaskDAG("communications subgraph reached without a planned task DAG")
        if self.communications_subgraph is None:  # pragma: no cover - routing guards this
            raise UnsupportedRouteDomain("no communications subgraph is wired")
        result = await self.communications_subgraph(task_dag, state)
        return {"result": result}

    async def _run_calendar_subgraph(self, state: SupervisorState) -> dict[str, Any]:
        """Hand the request to the Calendar subgraph and record its result."""
        task_dag = state.get("task_dag")
        if task_dag is None:  # pragma: no cover - plan_work always sets one before this node
            raise InvalidTaskDAG("calendar subgraph reached without a planned task DAG")
        if self.calendar_subgraph is None:  # pragma: no cover - routing guards this
            raise UnsupportedRouteDomain("no calendar subgraph is wired")
        result = await self.calendar_subgraph(task_dag, state)
        return {"result": result}

    # ------------------------------------------------------------------
    # The cross-domain plan path
    # ------------------------------------------------------------------
    #
    #   load_context -> plan_task -> classify_intent ...           (one domain: as before)
    #                   plan_task -> clarify                       (no valid plan after repairs)
    #                   plan_task -> publish_plan -> run_plan_step (one step per superstep)
    #                                   run_plan_step -> await_plan_steps (interrupts)
    #                                   -> collect_plan_steps -> ...
    #                                   -> settle_plan -> run_plan_step (a proposed release)
    #                                   -> settle_plan -> END
    #
    # Each step is its own superstep, so its record is checkpointed before the
    # next one starts: a crash after a step resumes at the step after it.

    def _add_plan_nodes(self, graph: StateGraph) -> None:
        for name, node in (
            (PLAN_TASK, self._plan_task),
            (PUBLISH_PLAN, self._publish_plan),
            (RUN_PLAN_STEP, self._run_plan_step),
            (AWAIT_PLAN_STEPS, self._await_plan_steps),
            (COLLECT_PLAN_STEPS, self._collect_plan_steps),
            (SETTLE_PLAN, self._settle_plan),
        ):
            graph.add_node(name, traced_node(name, node, graph="supervisor"))
        graph.add_edge(LOAD_CONTEXT, PLAN_TASK)
        graph.add_conditional_edges(
            PLAN_TASK,
            self._route_after_plan_task,
            {CLASSIFY_INTENT: CLASSIFY_INTENT, PUBLISH_PLAN: PUBLISH_PLAN, CLARIFY: CLARIFY},
        )
        graph.add_edge(PUBLISH_PLAN, RUN_PLAN_STEP)
        routes = {
            RUN_PLAN_STEP: RUN_PLAN_STEP,
            AWAIT_PLAN_STEPS: AWAIT_PLAN_STEPS,
            SETTLE_PLAN: SETTLE_PLAN,
        }
        graph.add_conditional_edges(RUN_PLAN_STEP, self._route_plan, routes)
        graph.add_edge(AWAIT_PLAN_STEPS, COLLECT_PLAN_STEPS)
        graph.add_conditional_edges(COLLECT_PLAN_STEPS, self._route_plan, routes)
        graph.add_conditional_edges(
            SETTLE_PLAN, self._route_after_settle, {RUN_PLAN_STEP: RUN_PLAN_STEP, END: END}
        )

    async def _plan_task(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        """Plan the request across domains, or say it is one domain's work.

        A plan that is still invalid after the planner's repair attempts is
        not run, trimmed or guessed at: the person is asked to clarify, and
        the refusal is recorded.
        """
        assert self.task_planner is not None
        message = state.get("message", "")
        if screen_request(message) is not None:
            # Not planned at all: classification marks it and clarify refuses it.
            return {"plan_kind": "single"}
        try:
            outcome = await self.task_planner.plan(
                message, domains=self.step_runners.keys(), max_steps=self.max_plan_steps
            )
        except Exception:
            logger.exception("task planning failed; routing the request to one domain")
            return {"plan_kind": "single"}
        if outcome.kind == "single":
            return {"plan_kind": "single"}
        if outcome.kind == "invalid" or outcome.plan is None:
            logger.info("plan refused after %d attempt(s): %s", outcome.attempts, outcome.errors)
            keys = self._audit(
                state,
                config,
                [plan_rejected_entry(_request_key(message), outcome.errors)],
            )
            return {
                "plan_kind": "invalid",
                "plan_errors": list(outcome.errors),
                "plan_audit_keys": keys,
            }
        plan = outcome.plan
        return {
            "plan_kind": "plan",
            "task_plan": plan.model_dump(mode="json"),
            "plan_records": _dump_records(initial_records(plan)),
            "plan_errors": list(outcome.errors),
        }

    def _route_after_plan_task(self, state: SupervisorState) -> str:
        kind = state.get("plan_kind")
        if kind == "plan":
            return PUBLISH_PLAN
        if kind == "invalid":
            return CLARIFY
        return CLASSIFY_INTENT

    def _publish_plan(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        """Record the whole plan, and return it, before any step runs.

        No step with an external effect has run yet, and none can without its
        own approval; the plan the person is shown is the plan that runs.
        """
        plan = _load_plan(state)
        records = _load_records(state)
        keys = self._audit(state, config, plan_proposed_entries(plan))
        return {
            "task_dag": _plan_as_dag(plan).model_dump(mode="json"),
            "result": _plan_result(plan, records),
            "plan_audit_keys": keys,
        }

    async def _run_plan_step(
        self, state: SupervisorState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run the next ready step through its domain's subgraph. One step, then a checkpoint."""
        plan = _load_plan(state)
        records = settle_records(plan, _load_records(state))
        ready = ready_steps(plan, records)
        if ready:
            step = plan.step(ready[0])
            runner = self.step_runners.get(step.domain)
            if runner is None:  # pragma: no cover - validation only admits wired domains
                raise UnsupportedRouteDomain(f"no step runner is wired for '{step.domain.value}'")
            logger.info("running plan step '%s' (%s)", step.id, step.domain.value)
            record = await runner.run(
                step, step_inputs(plan, step, records), self._step_context(plan, config)
            )
            records[step.id] = record
            records = settle_records(plan, records)
        return self._plan_update(state, config, plan, records)

    def _await_plan_steps(self, state: SupervisorState) -> dict[str, Any]:
        """Park until a step's approval has been answered and its thread has moved.

        The interrupt carries no approval request -- each step's own request
        is on its own thread -- only which steps this run is waiting on.
        """
        plan = _load_plan(state)
        records = _load_records(state)
        interrupt(
            {
                "workflow": "supervisor",
                "stage": STAGE_PLAN_STEPS,
                "plan_id": plan.plan_id,
                "waiting_on": [
                    {"step_id": step_id, "thread_id": records[step_id].thread_id}
                    for step_id in awaiting_steps(plan, records)
                ],
            }
        )
        return {}

    async def _collect_plan_steps(
        self, state: SupervisorState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Read where each parked step stands now. Runs nothing."""
        plan = _load_plan(state)
        records = _load_records(state)
        context = self._step_context(plan, config)
        for step_id in awaiting_steps(plan, records):
            step = plan.step(step_id)
            runner = self.step_runners[step.domain]
            records[step_id] = await runner.poll(step, records[step_id], context)
        return self._plan_update(state, config, plan, settle_records(plan, records))

    def _route_plan(self, state: SupervisorState) -> str:
        plan = _load_plan(state)
        records = settle_records(plan, _load_records(state))
        if ready_steps(plan, records):
            return RUN_PLAN_STEP
        if awaiting_steps(plan, records):
            return AWAIT_PLAN_STEPS
        return SETTLE_PLAN

    def _settle_plan(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        """Every step has settled. Flag what dangles, propose its release, or finish.

        A release is a step like any other: it runs through its domain's
        subgraph and waits for its own approval. Nothing is removed here.
        """
        plan = _load_plan(state)
        records = settle_records(plan, _load_records(state))
        dangling, releases = plan_compensations(plan, records)
        entries = dangling_entries(plan, dangling)
        if releases:
            plan = with_steps(plan, releases)
            for release in releases:
                records[release.id] = StepRecord(step_id=release.id)
        update = self._plan_update(state, config, plan, records, extra_entries=entries)
        if not releases:
            view = plan_view(plan, records, dangling)
            keys = self._audit(
                {**state, "plan_audit_keys": update["plan_audit_keys"]},
                config,
                [plan_finished_entry(plan, view)],
            )
            update["plan_audit_keys"] = keys
        return update

    def _route_after_settle(self, state: SupervisorState) -> str:
        plan = _load_plan(state)
        records = settle_records(plan, _load_records(state))
        return RUN_PLAN_STEP if ready_steps(plan, records) else END

    # --- helpers ------------------------------------------------------------

    def _step_context(self, plan: TaskPlan, config: RunnableConfig) -> StepContext:
        configurable = config.get("configurable") or {}
        return StepContext(
            plan_id=plan.plan_id,
            parent_thread_id=str(configurable.get("thread_id") or "supervisor"),
            workflow_id=str(configurable["workflow_id"]) if configurable.get("workflow_id") else None,
            actor_id=configurable.get("actor_id") or _DEFAULT_ACTOR_ID,
            correlation_id=(
                str(configurable["correlation_id"]) if configurable.get("correlation_id") else None
            ),
        )

    def _plan_update(
        self,
        state: SupervisorState,
        config: RunnableConfig,
        plan: TaskPlan,
        records: dict[str, StepRecord],
        *,
        extra_entries: Sequence[PlanAuditEntry] = (),
    ) -> dict[str, Any]:
        entries = [
            entry
            for step in plan.steps
            if step.id in records
            for entry in step_audit_entries(plan, step, records[step.id])
        ]
        keys = self._audit(state, config, [*entries, *extra_entries])
        dangling, _ = plan_compensations(plan, records)
        return {
            "task_plan": plan.model_dump(mode="json"),
            "plan_records": _dump_records(records),
            "plan_audit_keys": keys,
            "result": _plan_result(plan, records, dangling),
        }

    def _audit(
        self,
        state: SupervisorState,
        config: RunnableConfig,
        entries: Sequence[PlanAuditEntry],
    ) -> list[str]:
        """Record the entries not yet recorded; the keys recorded so far."""
        keys = list(state.get("plan_audit_keys") or [])
        seen = set(keys)
        fresh = [entry for entry in entries if entry.key not in seen]
        if not fresh:
            return keys
        if self.plan_audit is not None:
            configurable = config.get("configurable") or {}
            workflow_id = configurable.get("workflow_id")
            self.plan_audit.record(
                fresh,
                workflow_id=str(workflow_id) if workflow_id else None,
                actor_id=configurable.get("actor_id") or _DEFAULT_ACTOR_ID,
            )
        return [*keys, *(entry.key for entry in fresh)]

    def _clarify(self, state: SupervisorState) -> dict[str, Any]:
        """Ask for clarification instead of guessing at a low-confidence or unsupported route."""
        if state.get("plan_kind") == "invalid":
            issues = state.get("plan_errors") or []
            shown = "; ".join(issues[-3:]) if issues else "the steps did not fit together"
            return {
                "clarification": (
                    "I couldn't turn that into a plan I can run safely "
                    f"({shown}). Could you split it up, or say which part to do first?"
                )
            }
        resolution = self.resolve(state.get("route_decision"))
        if resolution.reason is RouteReason.ACCESS_ESCALATION:
            message = (
                "I can't do that: every send, submission, calendar change and file change "
                "waits for your approval, and I can only use the tools a task needs. "
                "I'm happy to prepare it for you to review -- what would you like me to draft?"
            )
        elif resolution.reason in (RouteReason.AMBIGUOUS, RouteReason.MULTI_DOMAIN):
            names = [_DOMAIN_NAMES[d] for d in resolution.candidates]
            if resolution.reason is RouteReason.AMBIGUOUS:
                message = f"That could be about {_join(names, 'or')}. Which did you mean?"
            else:
                message = (
                    f"That touches {_join(names)}. I'd rather not do only part of it -- "
                    "could you tell me which part to start with?"
                )
        elif resolution.reason is RouteReason.UNSUPPORTED and resolution.domain is not None:
            handled = _join([_DOMAIN_NAMES[d] for d in RouteDomain if d in self.routable_domains])
            message = (
                f"I can't help with '{resolution.domain.value}' yet -- I currently only "
                f"handle {handled}. Could you rephrase your request?"
            )
        else:
            message = (
                "I'm not confident I understood that. Could you clarify what you'd like help with?"
            )
        return {"clarification": message}


__all__ = [
    "AWAIT_PLAN_STEPS",
    "COLLECT_PLAN_STEPS",
    "PLAN_TASK",
    "PUBLISH_PLAN",
    "RUN_PLAN_STEP",
    "SETTLE_PLAN",
    "STAGE_PLAN_STEPS",
    "PlanAuditRecorder",
    "SupervisorGraph",
    "SupervisorState",
    "JobSubgraphRunner",
    "FileSubgraphRunner",
    "CommunicationsSubgraphRunner",
    "CalendarSubgraphRunner",
]
