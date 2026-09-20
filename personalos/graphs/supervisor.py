"""Top-level Supervisor graph: loads context, classifies intent, routes.

The Supervisor is the entry point for every request: it loads cross-thread
context, classifies the request into a typed `RouteDecision`, and either asks
for clarification or plans and hands off a bounded unit of work to a domain
subgraph. It never touches a tool or a repository directly -- see
`docs/ARCHITECTURE_BOUNDARIES.md` -- it only classifies, plans, and delegates.

Two things keep this graph honest to that boundary:

- `classifier` and `job_subgraph` are injected ports (`IntentClassifier` and
  `JobSubgraphRunner`), not concrete dependencies this module constructs. The
  composition root wires a real classifier and an adapter that knows how to
  run `personalos.executor.job_search.JobSearchExecutor`.
- Graph state is kept JSON-compatible (`dict`, not live `RouteDecision` /
  `TaskDAG` instances) so it round-trips cleanly through any
  `BaseCheckpointSaver`, including a durable one added in a follow-up issue.
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
    RouteDecision,
    RouteDomain,
    UnsupportedRouteDomain,
    ensure_supported_domain,
    route_decision_from_mapping,
)
from personalos.domain.tasks import TaskDAG, TaskNode
from personalos.models.routing import IntentClassifier

logger = logging.getLogger(__name__)

#: Long-term store namespace root for cross-thread facts. The full namespace
#: for a given thread is `(_FACTS_NAMESPACE, thread_id)`.
_FACTS_NAMESPACE = "supervisor_facts"

#: langgraph node/edge names, so they aren't inline string literals in two
#: places.
_LOAD_CONTEXT = "load_context"
_CLASSIFY_INTENT = "classify_intent"
_PLAN_WORK = "plan_work"
_RUN_JOB_SUBGRAPH = "run_job_subgraph"
_CLARIFY = "clarify"


class JobSubgraphRunner(Protocol):
    """Port the Supervisor uses to hand a planned task DAG to the Job Search subgraph.

    Kept as an injected port rather than a direct call into
    `personalos.executor.job_search.JobSearchExecutor` so this module never has
    to import `persistence` or `tools` to build the `Job` and `ToolGateway` an
    executor needs -- the composition root binds an adapter that already has
    them, matching how `JobSearchExecutor` itself receives its `ToolGateway`.
    """

    async def __call__(self, task_dag: dict[str, Any], state: "SupervisorState") -> dict[str, Any]:
        """Run the planned work and return a JSON-compatible result."""
        ...


class SupervisorState(TypedDict, total=False):
    """Graph state carried between Supervisor nodes.

    Every field is JSON-compatible so the state round-trips through any
    `BaseCheckpointSaver` without registering custom types; `RouteDecision`
    and `TaskDAG` are stored as `.model_dump()` and rebuilt on demand via
    `route_decision_from_mapping` / `TaskDAG.model_validate`.
    """

    message: str
    thread_facts: dict[str, Any]
    route_decision: dict[str, Any] | None
    task_dag: dict[str, Any] | None
    clarification: str | None
    result: dict[str, Any] | None


def _build_job_search_dag(message: str) -> TaskDAG:
    """The bounded plan for a job-domain request.

    Mirrors the fixed step sequence `JobSearchExecutor.run_job_search` already
    runs (prepare -> search -> scrape -> filter): a fixed, linear, four-step
    DAG, not an open-ended loop the model could extend indefinitely.
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


class SupervisorGraph:
    """Builds and compiles the top-level Supervisor LangGraph graph."""

    def __init__(
        self,
        classifier: IntentClassifier,
        job_subgraph: JobSubgraphRunner,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
    ):
        """Initialize with the ports this graph delegates to.

        `classifier` and `job_subgraph` are required, mirroring
        `JobSearchExecutor` requiring a `ToolGateway`: a Supervisor with no
        classifier would have to reach for a global default, which is exactly
        the kind of implicit wiring the composition root exists to own instead.

        `checkpointer` / `store` default to in-memory implementations so the
        graph compiles and is usable out of the box; a durable, Postgres-backed
        checkpointer and store are wired in by the composition root once the
        follow-up persistence issue lands, without this module changing.
        """
        if classifier is None:
            raise ValueError("SupervisorGraph requires an IntentClassifier")
        if job_subgraph is None:
            raise ValueError("SupervisorGraph requires a JobSubgraphRunner")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.classifier = classifier
        self.job_subgraph = job_subgraph
        self.confidence_threshold = confidence_threshold
        self.checkpointer = checkpointer or InMemorySaver()
        self.store = store or InMemoryStore()

    def build(self) -> CompiledStateGraph:
        """Assemble and compile the graph with this instance's checkpointer and store."""
        graph = StateGraph(SupervisorState)
        graph.add_node(_LOAD_CONTEXT, self._load_context)
        graph.add_node(_CLASSIFY_INTENT, self._classify_intent)
        graph.add_node(_PLAN_WORK, self._plan_work)
        graph.add_node(_RUN_JOB_SUBGRAPH, self._run_job_subgraph)
        graph.add_node(_CLARIFY, self._clarify)

        graph.add_edge(START, _LOAD_CONTEXT)
        graph.add_edge(_LOAD_CONTEXT, _CLASSIFY_INTENT)
        graph.add_conditional_edges(
            _CLASSIFY_INTENT,
            self._route_after_classification,
            {_PLAN_WORK: _PLAN_WORK, _CLARIFY: _CLARIFY},
        )
        graph.add_edge(_PLAN_WORK, _RUN_JOB_SUBGRAPH)
        graph.add_edge(_RUN_JOB_SUBGRAPH, END)
        graph.add_edge(_CLARIFY, END)

        return graph.compile(checkpointer=self.checkpointer, store=self.store)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _load_context(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        """Load cross-thread facts for this thread from the long-term store."""
        thread_id = (config.get("configurable") or {}).get("thread_id", "default")
        facts = {item.key: item.value for item in self.store.search((_FACTS_NAMESPACE, thread_id))}
        return {"thread_facts": facts}

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
            return _CLARIFY
        decision = route_decision_from_mapping(raw_decision)
        if decision.confidence < self.confidence_threshold:
            return _CLARIFY
        try:
            ensure_supported_domain(decision)
        except UnsupportedRouteDomain:
            return _CLARIFY
        return _PLAN_WORK

    def _plan_work(self, state: SupervisorState) -> dict[str, Any]:
        """Produce the bounded task DAG for a confidently-routed, supported domain."""
        decision = route_decision_from_mapping(state["route_decision"])
        ensure_supported_domain(decision)  # re-checked: plan_work must never plan the unsupported

        if decision.domain == RouteDomain.JOB:
            dag = _build_job_search_dag(state.get("message", ""))
        else:  # pragma: no cover - unreachable while SUPPORTED_ROUTE_DOMAINS == {JOB}
            raise UnsupportedRouteDomain(f"no planner registered for domain '{decision.domain}'")

        return {"task_dag": dag.model_dump(mode="json")}

    async def _run_job_subgraph(self, state: SupervisorState) -> dict[str, Any]:
        """Hand the planned DAG to the Job Search subgraph and record its result."""
        task_dag = state["task_dag"]
        result = await self.job_subgraph(task_dag, state)
        return {"result": result}

    def _clarify(self, state: SupervisorState) -> dict[str, Any]:
        """Ask for clarification instead of guessing at a low-confidence or unsupported route."""
        raw_decision = state.get("route_decision")
        decision = route_decision_from_mapping(raw_decision) if raw_decision else None

        if decision is not None and decision.confidence >= self.confidence_threshold:
            message = (
                f"I can't help with '{decision.domain.value}' yet -- I currently only "
                "handle job search. Could you rephrase your request?"
            )
        else:
            message = (
                "I'm not confident I understood that. Could you clarify what you'd like help with?"
            )
        return {"clarification": message}


__all__ = ["SupervisorGraph", "SupervisorState", "JobSubgraphRunner"]
