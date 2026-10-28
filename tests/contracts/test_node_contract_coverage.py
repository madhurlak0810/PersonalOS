"""CI gate: every node registered in a graph has a typed contract and a contract test.

Three checks, each read off the compiled graphs themselves rather than a list
kept by hand, so adding a node is what makes them fail:

1. **Every registered node has a `NodeContract`** in
   `personalos.graphs.contracts`, and every contract names a node that is
   still registered.
2. **Every registered node has a contract test**: a class marked
   `@covers_node(graph, node)` somewhere under `tests/`, with at least one
   test in it.
3. **The contracts agree with the graphs' real data flow.** Each graph is run
   end to end, through every entry point, with every node's input *and* output
   validated against its contract. The node tests check outputs against states
   written by hand; this checks that the state each node actually receives
   from the nodes before it is the state its input model says it reads -- and
   that the runs between them reach every node.

Offline and deterministic: every port is a fake from `tests/fixtures`, the
intent classifier is the keyword one, and the graph's clock is pinned.
"""

import importlib
import pkgutil
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

import tests
from personalos.domain.checkpoints import PendingCheckpoint
from personalos.domain.job_search import FollowUpKind
from personalos.domain.recruiter_events import CommitmentActor, ExtractedCommitment
from personalos.graphs import job_search as jsg
from personalos.graphs.contracts import (
    JOB_SEARCH_GRAPH,
    NODE_CONTRACTS,
    SUPERVISOR_GRAPH,
    NodeContract,
    NodeOutput,
    contracts_for,
)
from personalos.graphs.job_search import JobSearchGraph
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.routing import KeywordIntentClassifier
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.node_contracts import CONTRACT_TESTS, enforce_contracts


class _JobSubgraph:
    async def __call__(self, task_dag, state):
        return {"shortlist": []}


def _job_search(**overrides) -> JobSearchGraph:
    """A Job Search graph with every optional capability wired, over fakes."""
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [fakes.FakeProvider()],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": fakes.FakeApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
        "posting_catalog": fakes.FakePostingCatalog(),
        "recruiter_inbox": fakes.FakeRecruiterInbox([fakes.recruiter_message()]),
        "recruiter_classifier": fakes.FakeRecruiterClassifier(),
        "checkpoint_scheduler": fakes.FakePendingCheckpointScheduler(),
        "recruiter_event_extractor": fakes.FakeRecruiterEventExtractor(
            default=fakes.extraction(
                requires_reply=True,
                commitments=[
                    ExtractedCommitment(
                        actor=CommitmentActor.EXTERNAL_PERSON,
                        action="Technical interview with the team",
                        due_at=datetime(2026, 10, 1, 14, 0),
                        confidence=0.9,
                    )
                ],
            )
        ),
        "application_directory": fakes.FakeApplicationDirectory(),
        "recruiter_event_recorder": fakes.FakeRecruiterEventRecorder(),
        "calendar_reader": fakes.FakeCalendar(),
        "application_transitioner": fakes.FakeApplicationTransitioner(),
        "clock": lambda: fakes.NOW,
    }
    ports.update(overrides)
    return JobSearchGraph(**ports)


def _supervisor() -> SupervisorGraph:
    return SupervisorGraph(KeywordIntentClassifier(), _JobSubgraph())


def _registered_nodes(compiled) -> set[str]:
    return {name for name in compiled.get_graph().nodes if name not in ("__start__", "__end__")}


#: Every graph this build compiles, by the name its contracts are filed under.
GRAPHS = {
    SUPERVISOR_GRAPH: _supervisor,
    JOB_SEARCH_GRAPH: _job_search,
}


def _import_all_test_modules() -> None:
    """Import every test module, so every `@covers_node` has run."""
    for module in pkgutil.walk_packages(tests.__path__, prefix="tests."):
        if module.name.rsplit(".", 1)[-1].startswith("test_"):
            importlib.import_module(module.name)


# --- 1 and 2: every registered node is contracted and tested --------------------


@pytest.mark.parametrize("graph", sorted(GRAPHS))
def test_every_registered_node_has_a_typed_contract(graph):
    registered = _registered_nodes(GRAPHS[graph]().build())
    contracted = set(contracts_for(graph))

    assert registered == contracted, (
        f"{graph}: node(s) without a NodeContract: {sorted(registered - contracted)}; "
        f"contract(s) for nodes no longer registered: {sorted(contracted - registered)}"
    )


@pytest.mark.parametrize("graph", sorted(GRAPHS))
def test_every_registered_node_has_a_contract_test(graph):
    _import_all_test_modules()
    registered = _registered_nodes(GRAPHS[graph]().build())
    covered = {node for (name, node) in CONTRACT_TESTS if name == graph}

    assert (
        registered <= covered
    ), f"{graph}: node(s) with no @covers_node contract test: {sorted(registered - covered)}"
    for node in registered:
        cls = CONTRACT_TESTS[(graph, node)]
        tests_in_class = [name for name in dir(cls) if name.startswith("test_")]
        assert tests_in_class, f"{graph}:{node} is covered by {cls.__name__}, which has no tests"


def test_every_contract_names_a_real_method_and_forbids_undeclared_keys():
    for (graph, node), contract in NODE_CONTRACTS.items():
        instance = GRAPHS[graph]()
        assert callable(
            getattr(instance, contract.method, None)
        ), f"{graph}:{node} names method {contract.method!r}, which does not exist"
        assert contract.output.model_config.get("extra") == "forbid", f"{graph}:{node}"


def test_the_stage_vocabulary_matches_the_graph():
    """`ApprovalStage` restates the graph's stage constants; this keeps them equal."""
    from typing import get_args

    from personalos.graphs.contracts import ApprovalStage

    assert set(get_args(ApprovalStage)) == {
        jsg.STAGE_SUBMISSION,
        jsg.STAGE_RECRUITER_OUTREACH,
        jsg.STAGE_INTERVIEW_SCHEDULING,
        jsg.STAGE_LIFECYCLE_TRANSITION,
    }


def test_the_output_check_rejects_a_key_the_contract_does_not_declare():
    """The wrapper the node tests rely on actually fails on drift."""

    class Probe:
        def rank(self, state):
            return {"ranked_postings": [], "shortlist": []}

    for name, other in contracts_for(JOB_SEARCH_GRAPH).items():
        if name != jsg.RANK:
            setattr(Probe, other.method, lambda self, *a, **k: {})
    probe = enforce_contracts(Probe(), JOB_SEARCH_GRAPH)

    with pytest.raises(ValidationError):
        probe.rank({})


def test_a_consumed_channel_must_be_cleared():
    contract: NodeContract = NODE_CONTRACTS[(JOB_SEARCH_GRAPH, jsg.APPLY_REQUESTED_TRANSITIONS)]
    leftover = fakes.requested_withdrawal().model_dump(mode="json")

    with pytest.raises(ValidationError):
        contract.validate_output({"transition_outcomes": [], "requested_transitions": [leftover]})
    assert isinstance(
        contract.validate_output({"transition_outcomes": [], "requested_transitions": []}),
        NodeOutput,
    )


# --- 3: the contracts hold across real graph runs -------------------------------


async def test_supervisor_runs_satisfy_every_node_contract():
    visited: set[str] = set()
    graph = enforce_contracts(_supervisor(), SUPERVISOR_GRAPH, inputs=True, visited=visited)
    compiled = graph.build()

    for message in ("find me a python job", "hmm, not sure"):
        await compiled.ainvoke(
            {"message": message}, config={"configurable": {"thread_id": f"t-{uuid4()}"}}
        )

    assert visited == set(contracts_for(SUPERVISOR_GRAPH))


async def test_job_search_runs_satisfy_every_node_contract():
    """Every entry point, on two threads, with every node's input and output checked."""
    visited: set[str] = set()
    graph = enforce_contracts(_job_search(), JOB_SEARCH_GRAPH, inputs=True, visited=visited)
    compiled = graph.build()
    search = {"configurable": {"thread_id": f"search-{uuid4()}"}}
    inbox = {"configurable": {"thread_id": f"inbox-{uuid4()}"}}

    # Discovery -> application -> approval -> persist -> recruiter reply ->
    # follow-up checkpoint -> the reply's approval.
    final = await compiled.ainvoke(
        {"user_id": str(fakes.USER_ID), "prepare_application": True}, config=search
    )
    assert final["application"] is not None

    # A durable wait coming due on the same thread.
    fired = PendingCheckpoint.for_follow_up(
        application_id=fakes.APPLICATION_ID,
        kind=FollowUpKind.NO_RESPONSE,
        due_at=fakes.NOW + timedelta(days=7),
        reason="no recruiter response received yet",
        thread_id=search["configurable"]["thread_id"],
        created_at=fakes.NOW,
    )
    await compiled.ainvoke({"fired_checkpoints": [fired.model_dump(mode="json")]}, config=search)

    # A board move with an external effect.
    await compiled.ainvoke(
        {
            "user_id": str(fakes.USER_ID),
            "requested_transitions": [fakes.requested_withdrawal().model_dump(mode="json")],
        },
        config=search,
    )

    # Inbound mail with a timed invite -> interview scheduling.
    await compiled.ainvoke(
        {
            "user_id": str(fakes.USER_ID),
            "inbound_messages": [fakes.inbound_message().model_dump(mode="json")],
        },
        config=inbox,
    )

    assert visited == set(
        contracts_for(JOB_SEARCH_GRAPH)
    ), f"not reached by any run: {sorted(set(contracts_for(JOB_SEARCH_GRAPH)) - visited)}"
