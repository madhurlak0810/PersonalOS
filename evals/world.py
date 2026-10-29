"""The compiled graphs an eval case runs against, built from a `ModelGateway`.

Every model-backed port comes from the gateway; every other port is a fake
from `tests.fixtures.job_search_fakes`, the same ones the graph scenario tests
use. Those fakes enforce the real lifecycle rules (an application cannot jump
from applied to offer, a redelivered message is recorded once), so an outcome
measured here is an outcome production would reach, and the only thing that
differs between two runs of one case is the model configuration.

A world is built fresh for every case: nothing a case does is visible to the
next.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from personalos.domain.job_search import (
    EvidenceRecord,
    RawPosting,
    RecruiterMessage,
    SearchProfile,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.recruiter_events import ApplicationCandidate
from personalos.graphs.job_search import JobSearchGraph
from personalos.graphs.supervisor import SupervisorGraph, SupervisorState
from personalos.models.gateway import ModelGateway
from personalos.retrieval.job_matching import HybridJobMatcher
from tests.fixtures import job_search_fakes as fakes

#: Every case runs as this user, and an inbound-mail case's one application
#: on file has this id.
USER_ID = fakes.USER_ID
APPLICATION_ID = fakes.APPLICATION_ID

#: The instant every run believes it is. Golden mail is dated after it, so
#: nothing a node computes from the clock depends on when the harness runs.
NOW = fakes.NOW


class StaticEvidenceSource:
    """An `EvidenceSource` over one candidate's golden evidence records."""

    def __init__(self, records: Sequence[EvidenceRecord]):
        self.records = tuple(records)

    async def load(self, user_id: Any) -> Sequence[EvidenceRecord]:
        return self.records


class EchoJobSubgraph:
    """A `JobSubgraphRunner` that records the plan it was handed and runs nothing.

    Routing cases measure where the Supervisor sends a request, not what the
    Job Search subgraph then does with it; the scenarios cover that.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, task_dag: dict[str, Any], state: SupervisorState) -> dict[str, Any]:
        self.calls.append(task_dag)
        return {"handed_off": True}


def supervisor(gateway: ModelGateway) -> tuple[Any, EchoJobSubgraph]:
    """The compiled Supervisor over the gateway's intent classifier."""
    job_subgraph = EchoJobSubgraph()
    return SupervisorGraph(gateway.intent_classifier(), job_subgraph).build(), job_subgraph


@dataclass
class JobSearchWorld:
    """A compiled Job Search graph and the fakes a case reads its outcome from."""

    graph: Any
    application_store: fakes.FakeApplicationStore
    action_executor: fakes.FakeActionExecutor
    recorder: fakes.FakeRecruiterEventRecorder
    event_emitter: fakes.FakeEventEmitter
    approval_gate: Any
    extra: dict[str, Any] = field(default_factory=dict)


def job_search(
    gateway: ModelGateway,
    *,
    profile: SearchProfile | None = None,
    evidence: Sequence[EvidenceRecord] = (),
    postings: Sequence[RawPosting] = (),
    application: ApplicationCandidate | None = None,
    approve: bool = False,
) -> JobSearchWorld:
    """The compiled Job Search graph over fake ports and the gateway's models.

    `approve` stands a reviewer's approval on file for every action, so the
    run carries on through execution; otherwise it pauses at the interrupt.
    """
    matcher = HybridJobMatcher(
        evidence_source=StaticEvidenceSource(evidence), assessor=gateway.semantic_assessor()
    )
    application_store = fakes.FakeApplicationStore()
    action_executor = fakes.FakeActionExecutor()
    event_emitter = fakes.FakeEventEmitter()
    approval_gate = fakes.FakeApprovalGate() if approve else fakes.NoStandingApprovalGate()
    statuses = {APPLICATION_ID: application.status} if application else None
    recorder = fakes.FakeRecruiterEventRecorder(statuses)
    graph = JobSearchGraph(
        profile_store=fakes.FakeProfileStore(profile or fakes.profile()),
        providers=[fakes.FakeProvider(results=list(postings))],
        scorer=matcher,
        evidence_checker=matcher,
        packet_builder=fakes.FakePacketBuilder(),
        approval_gate=approval_gate,
        action_executor=action_executor,
        application_store=application_store,
        event_emitter=event_emitter,
        recruiter_event_extractor=gateway.recruiter_event_extractor(),
        application_directory=fakes.FakeApplicationDirectory([application] if application else []),
        recruiter_event_recorder=recorder,
        clock=lambda: NOW,
    ).build()
    return JobSearchWorld(
        graph=graph,
        application_store=application_store,
        action_executor=action_executor,
        recorder=recorder,
        event_emitter=event_emitter,
        approval_gate=approval_gate,
    )


def final_status(world: JobSearchWorld, *, inbound: bool) -> ApplicationStatus | None:
    """The application's status once the run stopped, or None if none exists."""
    if inbound:
        return world.recorder.statuses.get(APPLICATION_ID)
    created = world.application_store.created
    return created[-1].status if created else None


def thread(case_id: str) -> dict[str, Any]:
    """A case's run config: its thread, and a workflow id derived from the case.

    The workflow id is what puts a node's span in a trace with its thread id
    on it, and deriving it keeps both the same on every run.
    """
    return {
        "configurable": {
            "thread_id": f"eval-{case_id}",
            "workflow_id": str(uuid5(NAMESPACE_URL, f"personalos-evals/{case_id}")),
        }
    }


def inbound_input(*messages: RecruiterMessage) -> dict[str, Any]:
    return {
        "user_id": str(USER_ID),
        "inbound_messages": [message.model_dump(mode="json") for message in messages],
    }


def discovery_input(*, prepare_application: bool) -> dict[str, Any]:
    return {"user_id": str(USER_ID), "prepare_application": prepare_application}


__all__ = [
    "USER_ID",
    "APPLICATION_ID",
    "NOW",
    "StaticEvidenceSource",
    "EchoJobSubgraph",
    "supervisor",
    "JobSearchWorld",
    "job_search",
    "final_status",
    "thread",
    "inbound_input",
    "discovery_input",
]
