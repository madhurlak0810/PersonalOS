"""Per-node input/output contract tests for the Supervisor graph.

The counterpart of `tests/unit/test_job_search_nodes.py` for the top-level
graph: one class per registered node, each calling the node method directly on
an instance whose node methods check their output against
`personalos.graphs.contracts` (see `tests.fixtures.node_contracts`).

`classify_intent` is the one model-backed node here, so its class wires the
production `StructuredLLMIntentClassifier` over `ReplayChatModel` and replays
the outputs recorded in `tests/fixtures/recorded_model_outputs/
intent_classifier.json`: a valid answer, one repaired on the second try, one
that never validates and falls back to clarification, a low-confidence answer
and a confident one naming a domain this build does not support. Nothing here
reaches a provider.

End-to-end routing through the compiled graph is in
`tests/graph_scenarios/test_supervisor_graph.py`.
"""

from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from personalos.domain.routing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    RouteDecision,
    RouteDomain,
    UnsupportedRouteDomain,
)
from personalos.domain.tasks import TaskDAG
from personalos.graphs import supervisor as sup
from personalos.graphs.contracts import SUPERVISOR_GRAPH, contracts_for
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.routing import KeywordIntentClassifier, StructuredLLMIntentClassifier
from personalos.models.structured import REPAIR_PROMPT
from tests.fixtures.node_contracts import ReplayChatModel, covers_node, enforce_contracts

CONTRACTS = contracts_for(SUPERVISOR_GRAPH)


class StubClassifier:
    """Returns one fixed decision, or raises it."""

    def __init__(self, decision: RouteDecision | Exception):
        self.decision = decision
        self.calls: list[str] = []

    def classify(self, message: str) -> RouteDecision:
        self.calls.append(message)
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


class RecordingJobSubgraph:
    """A `JobSubgraphRunner` that records what it was handed."""

    def __init__(self, result: dict[str, Any] | None = None):
        self.result = result if result is not None else {"shortlist": []}
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def __call__(self, task_dag: dict[str, Any], state: Any) -> dict[str, Any]:
        self.calls.append((task_dag, dict(state)))
        return self.result


def supervisor(classifier: Any = None, job_subgraph: Any = None, **kwargs: Any) -> SupervisorGraph:
    """An uncompiled Supervisor whose node methods check their output contract."""
    graph = SupervisorGraph(
        classifier or StubClassifier(RouteDecision(domain=RouteDomain.JOB, confidence=0.9)),
        job_subgraph or RecordingJobSubgraph(),
        **kwargs,
    )
    return enforce_contracts(graph, SUPERVISOR_GRAPH)


def replaying(case: str) -> tuple[StructuredLLMIntentClassifier, ReplayChatModel]:
    model = ReplayChatModel.from_recording("intent_classifier", case)
    return StructuredLLMIntentClassifier(model), model


def decided(domain: RouteDomain = RouteDomain.JOB, confidence: float = 0.9) -> dict[str, Any]:
    """`route_decision` as the router and the nodes after it find it in state."""
    return RouteDecision(domain=domain, confidence=confidence).model_dump(mode="json")


# --- load_context --------------------------------------------------------------


@covers_node(SUPERVISOR_GRAPH, sup.LOAD_CONTEXT)
class TestLoadContext:
    def test_returns_only_this_actors_facts(self):
        store = InMemoryStore()
        store.put(("supervisor_facts", "actor-a"), "preferred_location", {"value": "Remote"})
        store.put(("supervisor_facts", "actor-b"), "preferred_location", {"value": "Berlin"})
        graph = supervisor(store=store)

        update = graph._load_context({}, {"configurable": {"actor_id": "actor-a"}})

        assert update == {"actor_facts": {"preferred_location": {"value": "Remote"}}}

    def test_no_actor_reads_the_system_namespace(self):
        store = InMemoryStore()
        store.put(("supervisor_facts", "system"), "k", {"value": 1})

        update = supervisor(store=store)._load_context({}, {})

        assert update == {"actor_facts": {"k": {"value": 1}}}

    def test_an_actor_with_nothing_on_file_gets_an_empty_mapping(self):
        update = supervisor()._load_context({}, {"configurable": {"actor_id": "new"}})

        assert update == {"actor_facts": {}}


# --- classify_intent -----------------------------------------------------------


@covers_node(SUPERVISOR_GRAPH, sup.CLASSIFY_INTENT)
class TestClassifyIntent:
    def test_a_valid_recorded_answer_becomes_a_typed_decision(self):
        classifier, model = replaying("valid_job")
        graph = supervisor(classifier)

        update = graph._classify_intent({"message": "find me backend roles"})

        assert set(update) == {"route_decision"}
        decision = RouteDecision.model_validate(update["route_decision"])
        assert decision == RouteDecision(
            domain=RouteDomain.JOB, confidence=0.93, reasoning="asks to find backend roles"
        )
        assert model.bound_schema is RouteDecision
        assert len(model.calls) == 1
        assert graph._route_after_classification(update) == sup.PLAN_WORK

    def test_schema_invalid_output_is_repaired_on_the_second_attempt(self):
        classifier, model = replaying("repaired")
        graph = supervisor(classifier)

        update = graph._classify_intent({"message": "find me backend roles"})

        assert RouteDecision.model_validate(update["route_decision"]).confidence == 0.88
        assert model.exhausted and len(model.calls) == 2
        # The repair turn repeats the request and says what was wrong with the answer.
        repair = model.calls[1]
        assert repair[:2] == model.calls[0][:2]
        assert repair[2] == ("human", REPAIR_PROMPT.format(error="ValidationError"))
        assert graph._route_after_classification(update) == sup.PLAN_WORK

    def test_output_that_never_validates_falls_back_to_clarification(self):
        classifier, model = replaying("invalid_twice")
        graph = supervisor(classifier)

        update = graph._classify_intent({"message": "buy me some bitcoin"})

        assert model.exhausted
        decision = RouteDecision.model_validate(update["route_decision"])
        assert decision.confidence == 0.0
        assert decision.reasoning == "classification failed"
        assert graph._route_after_classification(update) == sup.CLARIFY

    def test_a_provider_error_is_not_retried_and_falls_back(self):
        model = ReplayChatModel(RuntimeError("503 from provider"))
        graph = supervisor(StructuredLLMIntentClassifier(model))

        update = graph._classify_intent({"message": "find me a job"})

        assert len(model.calls) == 1
        assert RouteDecision.model_validate(update["route_decision"]).confidence == 0.0
        assert graph._route_after_classification(update) == sup.CLARIFY

    def test_a_low_confidence_answer_is_recorded_as_given_and_routes_to_clarify(self):
        classifier, _model = replaying("low_confidence")
        graph = supervisor(classifier)

        update = graph._classify_intent({"message": "help?"})

        decision = RouteDecision.model_validate(update["route_decision"])
        assert decision.confidence == 0.31
        assert decision.confidence < DEFAULT_CONFIDENCE_THRESHOLD
        assert graph._route_after_classification(update) == sup.CLARIFY

    def test_a_confident_unsupported_domain_is_recorded_and_routes_to_clarify(self):
        classifier, _model = replaying("unsupported_domain")
        graph = supervisor(classifier)

        update = graph._classify_intent({"message": "move my 3pm to tomorrow"})

        assert RouteDecision.model_validate(update["route_decision"]).domain == (
            RouteDomain.CALENDAR
        )
        assert graph._route_after_classification(update) == sup.CLARIFY

    def test_a_missing_message_is_classified_as_empty_text(self):
        classifier = StubClassifier(RouteDecision(domain=RouteDomain.JOB, confidence=0.1))

        supervisor(classifier)._classify_intent({})

        assert classifier.calls == [""]

    def test_the_keyword_fallback_is_deterministic(self):
        graph = supervisor(KeywordIntentClassifier())

        first = graph._classify_intent({"message": "update my resume and apply"})
        second = graph._classify_intent({"message": "update my resume and apply"})

        assert first == second
        assert first["route_decision"]["domain"] == RouteDomain.JOB.value


# --- plan_work -----------------------------------------------------------------


@covers_node(SUPERVISOR_GRAPH, sup.PLAN_WORK)
class TestPlanWork:
    def test_a_job_decision_plans_the_bounded_four_step_dag(self):
        update = supervisor()._plan_work({"message": "python jobs", "route_decision": decided()})

        assert set(update) == {"task_dag"}
        dag = TaskDAG.model_validate(update["task_dag"])
        assert dag.topological_order() == ["prepare", "search", "enrich", "filter"]
        assert dag.goal == "job_search: python jobs"

    def test_the_goal_is_bounded_whatever_the_message_length(self):
        update = supervisor()._plan_work({"message": "x" * 1000, "route_decision": decided()})

        assert len(TaskDAG.model_validate(update["task_dag"]).goal) <= 200

    @pytest.mark.parametrize(
        "domain", [RouteDomain.FILE, RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR]
    )
    def test_an_unsupported_domain_is_never_planned(self, domain):
        with pytest.raises(UnsupportedRouteDomain):
            supervisor()._plan_work({"route_decision": decided(domain)})

    def test_its_input_contract_requires_a_route_decision(self):
        with pytest.raises(ValidationError):
            CONTRACTS[sup.PLAN_WORK].validate_input({"message": "python jobs"})


# --- run_job_subgraph ----------------------------------------------------------


@covers_node(SUPERVISOR_GRAPH, sup.RUN_JOB_SUBGRAPH)
class TestRunJobSubgraph:
    async def test_hands_the_planned_dag_over_and_records_only_the_result(self):
        job_subgraph = RecordingJobSubgraph(result={"shortlist": [{"rank": 1}]})
        graph = supervisor(job_subgraph=job_subgraph)
        dag = graph._plan_work({"message": "python jobs", "route_decision": decided()})["task_dag"]

        update = await graph._run_job_subgraph({"message": "python jobs", "task_dag": dag})

        assert update == {"result": {"shortlist": [{"rank": 1}]}}
        ((handed, state),) = job_subgraph.calls
        assert handed == dag
        assert state["message"] == "python jobs"

    def test_its_input_contract_requires_a_task_dag(self):
        with pytest.raises(ValidationError):
            CONTRACTS[sup.RUN_JOB_SUBGRAPH].validate_input({})


# --- clarify -------------------------------------------------------------------


@covers_node(SUPERVISOR_GRAPH, sup.CLARIFY)
class TestClarify:
    def test_low_confidence_asks_what_was_meant(self):
        update = supervisor()._clarify({"route_decision": decided(confidence=0.2)})

        assert set(update) == {"clarification"}
        assert "not confident" in update["clarification"]

    def test_a_confident_unsupported_domain_says_which_domain_is_missing(self):
        update = supervisor()._clarify({"route_decision": decided(RouteDomain.CALENDAR, 0.95)})

        assert "'calendar'" in update["clarification"]
        assert "only" in update["clarification"]

    def test_no_decision_at_all_still_asks_for_clarification(self):
        assert "not confident" in supervisor()._clarify({})["clarification"]


# --- Router (pure logic) -------------------------------------------------------


class TestRouteAfterClassification:
    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [(0.0, sup.CLARIFY), (0.59, sup.CLARIFY), (0.6, sup.PLAN_WORK), (1.0, sup.PLAN_WORK)],
    )
    def test_the_threshold_is_inclusive(self, confidence, expected):
        graph = supervisor(confidence_threshold=0.6)

        assert graph._route_after_classification(
            {"route_decision": decided(confidence=confidence)}
        ) == (expected)

    def test_no_decision_routes_to_clarify(self):
        assert supervisor()._route_after_classification({}) == sup.CLARIFY

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_a_threshold_outside_the_unit_interval_is_refused(self, threshold):
        with pytest.raises(ValueError):
            SupervisorGraph(
                KeywordIntentClassifier(), RecordingJobSubgraph(), confidence_threshold=threshold
            )
