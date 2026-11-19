"""Top-level Supervisor graph: loads context, classifies intent, routes.

The Supervisor is the entry point for every request: it loads cross-thread
context, classifies the request into a typed `RouteDecision`, and either asks
for clarification or plans and hands off a bounded unit of work to a domain
subgraph. It never touches a tool or a repository directly -- see
`docs/ARCHITECTURE_BOUNDARIES.md` -- it only classifies, plans, and delegates.

Two things keep this graph honest to that boundary:

- `classifier`, `job_subgraph`, `file_subgraph` and `communications_subgraph`
  are injected ports (`IntentClassifier`, `JobSubgraphRunner`,
  `FileSubgraphRunner`, `CommunicationsSubgraphRunner`), not concrete
  dependencies this module constructs. The composition root wires a real
  classifier and adapters that run the Job Search subgraph
  (`personalos.graphs.job_search.JobSearchSubgraphRunner`), the File subgraph
  (`personalos.graphs.file_task.FileTaskSubgraphRunner`) and the
  Communications subgraph
  (`personalos.graphs.communications.CommunicationsSubgraphRunner`). A
  Supervisor built without one of the optional subgraphs treats its domain as
  unsupported and asks for clarification, exactly as it does for a reserved
  domain.
- Graph state is kept JSON-compatible (`dict`, not live `RouteDecision` /
  `TaskDAG` instances) so it round-trips cleanly through any
  `BaseCheckpointSaver`, including the durable, database-backed one in
  `personalos.persistence.checkpointer`.
"""

import logging
from typing import Any, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from personalos.domain.routing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    SUPPORTED_ROUTE_DOMAINS,
    RouteDecision,
    RouteDomain,
    UnsupportedRouteDomain,
    ensure_supported_domain,
    route_decision_from_mapping,
)
from personalos.domain.tasks import InvalidTaskDAG, TaskDAG, TaskNode
from personalos.models.routing import IntentClassifier
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
CLARIFY = "clarify"

#: What each routable domain is called in a clarification.
_DOMAIN_NAMES: dict[RouteDomain, str] = {
    RouteDomain.JOB: "job search",
    RouteDomain.FILE: "your job-search documents",
    RouteDomain.COMMUNICATIONS: "your job-search email",
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


class SupervisorGraph:
    """Builds and compiles the top-level Supervisor LangGraph graph."""

    def __init__(
        self,
        classifier: IntentClassifier,
        job_subgraph: JobSubgraphRunner,
        *,
        file_subgraph: FileSubgraphRunner | None = None,
        communications_subgraph: CommunicationsSubgraphRunner | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
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
        optional = {
            RouteDomain.FILE: file_subgraph,
            RouteDomain.COMMUNICATIONS: communications_subgraph,
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
        graph.add_node(CLARIFY, traced_node(CLARIFY, self._clarify, graph="supervisor"))

        graph.add_edge(START, LOAD_CONTEXT)
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
            },
        )
        graph.add_edge(RUN_JOB_SUBGRAPH, END)
        graph.add_edge(RUN_FILE_SUBGRAPH, END)
        graph.add_edge(RUN_COMMUNICATIONS_SUBGRAPH, END)
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
        return {"route_decision": decision.model_dump(mode="json")}

    def _route_after_classification(self, state: SupervisorState) -> str:
        """Below the confidence threshold, or outside the supported domains: clarify."""
        raw_decision = state.get("route_decision")
        if raw_decision is None:
            return CLARIFY
        decision = route_decision_from_mapping(raw_decision)
        if decision.confidence < self.confidence_threshold:
            return CLARIFY
        try:
            self._ensure_routable(decision)
        except UnsupportedRouteDomain:
            return CLARIFY
        return PLAN_WORK

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

    def _clarify(self, state: SupervisorState) -> dict[str, Any]:
        """Ask for clarification instead of guessing at a low-confidence or unsupported route."""
        raw_decision = state.get("route_decision")
        decision = route_decision_from_mapping(raw_decision) if raw_decision else None

        if decision is not None and decision.confidence >= self.confidence_threshold:
            names = [_DOMAIN_NAMES[d] for d in RouteDomain if d in self.routable_domains]
            handled = (
                " and ".join(names)
                if len(names) < 3
                else f"{', '.join(names[:-1])} and {names[-1]}"
            )
            message = (
                f"I can't help with '{decision.domain.value}' yet -- I currently only "
                f"handle {handled}. Could you rephrase your request?"
            )
        else:
            message = (
                "I'm not confident I understood that. Could you clarify what you'd like help with?"
            )
        return {"clarification": message}


__all__ = [
    "SupervisorGraph",
    "SupervisorState",
    "JobSubgraphRunner",
    "FileSubgraphRunner",
    "CommunicationsSubgraphRunner",
]
