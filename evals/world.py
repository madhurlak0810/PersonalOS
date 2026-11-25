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

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
from personalos.domain.routing import RouteDomain
from personalos.domain.task_plan import PlanStep, StepRecord, StepRef, StepStatus
from personalos.graphs.job_search import JobSearchGraph
from personalos.graphs.plan_steps import StepContext
from personalos.graphs.supervisor import SupervisorGraph, SupervisorState
from personalos.models.gateway import ModelGateway
from personalos.retrieval.job_matching import HybridJobMatcher
from tests.fixtures import calendar as calendar_fixtures
from tests.fixtures import communications as communications_fixtures
from tests.fixtures import file_task as file_fixtures
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


class EchoFileSubgraph(EchoJobSubgraph):
    """Records the file tasks the Supervisor hands off, as `EchoJobSubgraph` does jobs."""


class EchoCommunicationsSubgraph(EchoJobSubgraph):
    """Records the communications tasks the Supervisor hands off."""


class EchoCalendarSubgraph(EchoJobSubgraph):
    """Records the calendar tasks the Supervisor hands off."""


class EchoStepRunner:
    """A `PlanStepRunner` that records the step it was handed and reports it done.

    The routing suite measures whether a cross-domain request was planned,
    and into which domains -- not what each step's subgraph then did.
    """

    def __init__(self, domain: RouteDomain) -> None:
        self.domain = domain
        self.steps: list[PlanStep] = []

    async def run(
        self, step: PlanStep, inputs: Mapping[str, StepRef], context: StepContext
    ) -> StepRecord:
        self.steps.append(step)
        return StepRecord(step_id=step.id, status=StepStatus.SUCCEEDED, outcome="done")

    async def poll(self, step: PlanStep, record: StepRecord, context: StepContext) -> StepRecord:
        return record


@dataclass
class SupervisorWorld:
    """The compiled Supervisor and the echoes a routing case reads its route from."""

    graph: Any
    #: The uncompiled `SupervisorGraph`, for `resolve`: why a decision clarified.
    graph_owner: Any
    job: EchoJobSubgraph
    file: EchoFileSubgraph
    communications: EchoCommunicationsSubgraph
    calendar: EchoCalendarSubgraph
    steps: dict[RouteDomain, EchoStepRunner]

    def route(self, final: dict[str, Any]) -> str:
        """Where the request went: a subgraph, a cross-domain `plan`, or `clarify`."""
        if final.get("plan_kind") == "plan":
            return "plan"
        for name, echo in (
            ("job_search", self.job),
            ("file_task", self.file),
            ("communications", self.communications),
            ("calendar", self.calendar),
        ):
            if echo.calls:
                return name
        return "clarify"


def supervisor(gateway: ModelGateway) -> SupervisorWorld:
    """The compiled Supervisor as the worker boots it, over the gateway's models.

    Intent classifier and cross-domain task planner both come from the
    gateway, with every domain's subgraph and plan-step runner wired, so a
    routing case takes the same path a real request does: planned across
    domains first, classified to one otherwise.
    """
    steps = {domain: EchoStepRunner(domain) for domain in RouteDomain}
    world = SupervisorWorld(
        graph=None,
        graph_owner=None,
        job=EchoJobSubgraph(),
        file=EchoFileSubgraph(),
        communications=EchoCommunicationsSubgraph(),
        calendar=EchoCalendarSubgraph(),
        steps=steps,
    )
    world.graph_owner = SupervisorGraph(
        gateway.intent_classifier(),
        world.job,
        file_subgraph=world.file,
        communications_subgraph=world.communications,
        calendar_subgraph=world.calendar,
        task_planner=gateway.task_planner(),
        step_runners=steps,
    )
    world.graph = world.graph_owner.build()
    return world


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


# --- The new domains' subgraphs ------------------------------------------------------


@dataclass
class DomainWorld:
    """One compiled domain subgraph, its first input, and how to read its effects.

    `executor_owner` is the object whose `action_executor` (or `files`) a
    recovery case wraps to inject a fault; `effects()` reads what reached the
    fake outside world, keyed by what makes two effects the same one.
    """

    domain: str
    graph: Any
    graph_input: dict[str, Any]
    fixtures: Any
    approve: Any
    effects: Any
    executor_owner: Any
    extra: dict[str, Any] = field(default_factory=dict)


def _calendar_events(entries: Sequence[dict[str, Any]]):
    cf = calendar_fixtures
    events = []
    for entry in entries:
        if entry["kind"] == "interview_with_prep":
            events += cf.interview_with_prep(datetime.fromisoformat(entry["starts_at"]))
            continue
        make = cf.managed if entry["kind"] == "managed" else cf.external
        events.append(
            make(
                entry["event_id"],
                entry["title"],
                datetime.fromisoformat(entry["starts_at"]),
                datetime.fromisoformat(entry["ends_at"]),
            )
        )
    return events


def calendar(gateway: ModelGateway, case: dict[str, Any]) -> DomainWorld:
    """The Calendar subgraph over the in-memory calendar. It has no model-backed port."""
    cf = calendar_fixtures
    built = cf.build_world(_calendar_events(case.get("events", ())))
    fake = built.calendar

    def effects() -> list[tuple[str, Any]]:
        from personalos.domain.interview_scheduling import PROP_IDEMPOTENCY_KEY

        created = [
            (
                "calendar.create",
                fake.events[event_id].properties.get(PROP_IDEMPOTENCY_KEY, event_id),
            )
            for event_id in fake.created
            if event_id in fake.events
        ]
        return (
            created
            + [("calendar.update", event_id) for event_id in fake.updated]
            + [("calendar.cancel", event_id) for event_id in fake.cancelled]
        )

    return DomainWorld(
        domain="calendar",
        graph=built.graph.build(),
        graph_input=cf.start(case["request"]),
        fixtures=cf,
        approve=cf.approve,
        effects=effects,
        executor_owner=built.graph,
    )


def communications(gateway: ModelGateway, case: dict[str, Any]) -> DomainWorld:
    """The Communications subgraph over the Gmail tool path and a fake Google API.

    Triage, thread analysis and the reply writer come from the gateway; under
    `rules` the reply is the fixture's scripted one, as the template would be.
    """
    cf = communications_fixtures
    writer = gateway.thread_reply_writer()
    built = cf.build_world_in_process(
        triager=gateway.message_triager(),
        analyzer=gateway.thread_analyzer(),
        **({"writer": writer} if writer is not None else {}),
    )
    user_id, _ = cf.applied_application(built.factory)
    for index, mail in enumerate(case.get("mail", ())):
        built.add_mail(
            mail.get("message_id", f"m-{index + 1}"),
            **{key: value for key, value in mail.items() if key != "message_id"},
        )

    def effects() -> list[tuple[str, Any]]:
        return [("gmail.send", (send.json or {}).get("raw")) for send in built.sends()]

    return DomainWorld(
        domain="communications",
        graph=built.compile(),
        graph_input={"request": case["request"], "user_id": str(user_id)},
        fixtures=cf,
        approve=cf.approve,
        effects=effects,
        executor_owner=built.graph,
        extra={"world": built},
    )


def file_task(gateway: ModelGateway, case: dict[str, Any]) -> DomainWorld:
    """The File subgraph over a real files server confined to a fresh temporary root."""
    ff = file_fixtures
    root = Path(tempfile.mkdtemp(prefix="personalos-eval-"))
    layout = ff.seed_documents(root)
    built = ff.build_world(
        root,
        planner=gateway.file_planner(),
        summarizer=gateway.document_summarizer(),
        max_unattended=case.get("max_unattended", 3),
    )

    def effects() -> list[tuple[str, Any]]:
        # What is on disk now, against the seeded layout: a file gone from
        # where it was seeded was moved or deleted, once.
        return [("file.change", name) for name, path in layout.items() if not path.exists()]

    return DomainWorld(
        domain="file",
        graph=built.compile(),
        graph_input={"request": case["request"]},
        fixtures=ff,
        approve=ff.approve,
        effects=effects,
        executor_owner=built.graph,
        extra={"root": root, "layout": layout},
    )


DOMAIN_WORLDS = {"calendar": calendar, "communications": communications, "file": file_task}


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
    "EchoStepRunner",
    "SupervisorWorld",
    "supervisor",
    "DomainWorld",
    "DOMAIN_WORLDS",
    "calendar",
    "communications",
    "file_task",
    "JobSearchWorld",
    "job_search",
    "final_status",
    "thread",
    "inbound_input",
    "discovery_input",
]
