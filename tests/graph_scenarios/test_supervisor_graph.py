"""Scenario tests for the Supervisor graph: typed routing, clarification, delegation.

Covers the issue's acceptance criteria directly:
- valid job-domain routing reaches the Job Search subgraph,
- low-confidence routing goes to clarification instead of guessing,
- a RouteDecision naming a domain outside the supported enum is rejected
  (routed to clarification, never planned or delegated).
"""

from typing import Any

import pytest

from personalos.domain.routing import RouteDecision, RouteDomain
from personalos.domain.tasks import TaskDAG
from personalos.graphs.supervisor import SupervisorGraph, SupervisorState


class StubClassifier:
    """Returns a fixed RouteDecision regardless of input message."""

    def __init__(self, decision: RouteDecision):
        self.decision = decision
        self.calls: list[str] = []

    def classify(self, message: str) -> RouteDecision:
        self.calls.append(message)
        return self.decision


class RaisingClassifier:
    """Always raises, to exercise the graph's classifier-failure fallback."""

    def classify(self, message: str) -> RouteDecision:
        raise RuntimeError("boom")


class RecordingJobSubgraph:
    """Fake Job Search subgraph runner that records whether it was invoked."""

    def __init__(self, result: dict[str, Any] | None = None):
        self.result = result or {"matches": []}
        self.calls: list[tuple[dict[str, Any], SupervisorState]] = []

    async def __call__(self, task_dag: dict[str, Any], state: SupervisorState) -> dict[str, Any]:
        self.calls.append((task_dag, state))
        return self.result


def build_graph(classifier, job_subgraph, **kwargs):
    return SupervisorGraph(classifier, job_subgraph, **kwargs).build()


async def test_valid_job_domain_routing_reaches_job_subgraph():
    decision = RouteDecision(domain=RouteDomain.JOB, confidence=0.9, reasoning="clear job intent")
    classifier = StubClassifier(decision)
    job_subgraph = RecordingJobSubgraph(result={"matches": [{"id": "job_1"}]})
    graph = build_graph(classifier, job_subgraph)

    final = await graph.ainvoke(
        {"message": "find me a python job"},
        config={"configurable": {"thread_id": "t-job"}},
    )

    assert final["route_decision"]["domain"] == "job"
    assert not final.get("clarification")
    assert final["result"] == {"matches": [{"id": "job_1"}]}
    assert len(job_subgraph.calls) == 1

    task_dag_payload, _state = job_subgraph.calls[0]
    task_dag = TaskDAG.model_validate(task_dag_payload)
    assert task_dag.topological_order() == ["prepare", "search", "enrich", "filter"]


async def test_low_confidence_routes_to_clarification_not_job_subgraph():
    decision = RouteDecision(domain=RouteDomain.JOB, confidence=0.2, reasoning="unsure")
    classifier = StubClassifier(decision)
    job_subgraph = RecordingJobSubgraph()
    graph = build_graph(classifier, job_subgraph, confidence_threshold=0.6)

    final = await graph.ainvoke(
        {"message": "maybe help with something?"},
        config={"configurable": {"thread_id": "t-lowconf"}},
    )

    assert final.get("clarification")
    assert "result" not in final or final["result"] is None
    assert job_subgraph.calls == []


@pytest.mark.parametrize(
    "domain", [RouteDomain.FILE, RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR]
)
async def test_unsupported_domain_is_rejected_and_routes_to_clarification(domain):
    """A RouteDecision naming a domain outside the supported enum is rejected."""
    decision = RouteDecision(domain=domain, confidence=0.95, reasoning="confident but unsupported")
    classifier = StubClassifier(decision)
    job_subgraph = RecordingJobSubgraph()
    graph = build_graph(classifier, job_subgraph)

    final = await graph.ainvoke(
        {"message": "do something in an unsupported domain"},
        config={"configurable": {"thread_id": "t-unsupported"}},
    )

    assert final.get("clarification")
    assert domain.value in final["clarification"]
    assert job_subgraph.calls == []
    assert "task_dag" not in final or final["task_dag"] is None


async def test_classifier_failure_falls_back_to_clarification():
    graph = build_graph(RaisingClassifier(), RecordingJobSubgraph())

    final = await graph.ainvoke(
        {"message": "anything"},
        config={"configurable": {"thread_id": "t-fail"}},
    )

    assert final.get("clarification")


async def test_missing_classifier_or_job_subgraph_is_rejected_at_construction():
    with pytest.raises(ValueError):
        SupervisorGraph(None, RecordingJobSubgraph())
    with pytest.raises(ValueError):
        SupervisorGraph(StubClassifier(RouteDecision(domain=RouteDomain.JOB, confidence=0.9)), None)


async def test_graph_compiles_with_a_checkpointer_and_store_and_persists_state():
    """Supervisor graph compiles with a durable checkpointer and a long-term store.

    In-memory implementations stand in here for the durable, Postgres-backed
    ones wired in by the follow-up persistence issue; what this asserts is
    that the graph is checkpointer/store-shaped at all, and that state for a
    thread is retrievable after a run.
    """
    decision = RouteDecision(domain=RouteDomain.JOB, confidence=0.9)
    graph = build_graph(StubClassifier(decision), RecordingJobSubgraph())

    config = {"configurable": {"thread_id": "t-persist"}}
    await graph.ainvoke({"message": "find a job"}, config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.values["message"] == "find a job"
    assert snapshot.values["route_decision"]["domain"] == "job"


async def test_cross_thread_facts_are_loaded_from_the_long_term_store():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    store.put(("supervisor_facts", "t-facts"), "prefers_remote", {"value": True})

    decision = RouteDecision(domain=RouteDomain.JOB, confidence=0.9)
    graph = build_graph(StubClassifier(decision), RecordingJobSubgraph(), store=store)

    final = await graph.ainvoke(
        {"message": "find a job"},
        config={"configurable": {"thread_id": "t-facts"}},
    )

    assert final["thread_facts"] == {"prefers_remote": {"value": True}}
