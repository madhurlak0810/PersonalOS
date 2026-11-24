"""A cross-domain plan through the worker: durable threads, the approval queue, the wake-up.

The in-graph behaviour of plans is in `tests/adversarial/test_cross_domain_plans.py`.
This file runs the same request the way production does: the Supervisor, the
Calendar subgraph and the Communications subgraph on the durable SQLite
checkpointer, each driven by a `DurableWorkflowRunner` under the workflow
lease; the request queued as a `start` command; approvals listed and decided
through the `ApprovalQueue`; and the worker draining the command queue,
including the plan resume `PlanStepResumer` queues when a step thread moves.
"""

from typing import Any
from uuid import uuid4

import pytest

from apps.worker.main import Worker
from apps.worker.plan_steps import WorkflowScopedStepThreads
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    build_approval_queue,
    build_durable_checkpointer,
    build_worker_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_supervisor_thread,
)
from personalos.domain.auth import HUMAN_SCOPES, AuthMethod, Principal
from personalos.domain.job_search import ActionKind
from personalos.domain.ownership import SYSTEM_ACCESS, ActorScope
from personalos.domain.routing import RouteDomain
from personalos.domain.search_profiles import actor_user_id
from personalos.domain.task_plan import PlanView, StepStatus
from personalos.domain.workflow import (
    CALENDAR_THREAD_NAMESPACE,
    COMMUNICATIONS_THREAD_NAMESPACE,
    SUPERVISOR_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
    thread_namespace,
)
from personalos.graphs.plan_steps import SubgraphStepRunner
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.routing import KeywordIntentClassifier
from personalos.models.task_planning import ResilientTaskPlanner
from personalos.persistence.audit_trail import AuditTrailStore
from personalos.persistence.models import UserModel
from personalos.persistence.plan_audit import SqlPlanAuditRecorder
from tests.fixtures import calendar as calf
from tests.fixtures import communications as cf
from tests.fixtures.durable_workflow import session_factory

ACTOR = "sam"
REQUEST = "reply to the recruiter and hold Thursday at 2pm"


class NoJobSearch:
    async def __call__(self, task_dag, state):
        raise AssertionError("a plan never reaches the single-domain job port")


@pytest.fixture
async def plant(tmp_path):
    factory = session_factory(tmp_path / "plans.db")
    user_id = actor_user_id(ACTOR)
    session = factory()
    session.add(UserModel(id=user_id, email="sam@example.com"))
    session.commit()
    session.close()

    comms = await cf.build_world(factory=factory)
    try:
        comms.add_mail("m-1")
        yield Plant(factory, comms, user_id)
    finally:
        await comms.google.stop()


class Plant:
    def __init__(self, factory, comms: cf.CommsWorld, user_id):
        self.factory = factory
        self.comms = comms
        self.user_id = user_id
        registry = build_workflow_thread_registry(factory)
        checkpointer = build_durable_checkpointer(factory, registry)
        self.registry = registry
        self.calendar = calf.build_world(clock_at=calf.NOW)
        self.calendar.graph.checkpointer = checkpointer
        comms.graph.checkpointer = checkpointer
        comms.graph.clock = lambda: calf.NOW
        calendar_graph = self.calendar.graph.build()
        comms_graph = comms.graph.build()
        binder = WorkflowScopedStepThreads(registry)
        supervisor = SupervisorGraph(
            KeywordIntentClassifier(),
            NoJobSearch(),
            task_planner=ResilientTaskPlanner(),
            step_runners={
                RouteDomain.CALENDAR: SubgraphStepRunner(
                    calendar_graph, RouteDomain.CALENDAR, binder=binder
                ),
                RouteDomain.COMMUNICATIONS: SubgraphStepRunner(
                    comms_graph, RouteDomain.COMMUNICATIONS, binder=binder
                ),
            },
            plan_audit=SqlPlanAuditRecorder(factory, clock=lambda: calf.NOW),
            checkpointer=checkpointer,
        ).build()
        leases = build_workflow_lease_store(factory)

        def runner(graph) -> DurableWorkflowRunner:
            return DurableWorkflowRunner(graph, registry=registry, leases=leases, owner="w-1")

        self.runners = {
            SUPERVISOR_THREAD_NAMESPACE: runner(supervisor),
            CALENDAR_THREAD_NAMESPACE: runner(calendar_graph),
            COMMUNICATIONS_THREAD_NAMESPACE: runner(comms_graph),
        }
        self.supervisor = supervisor
        self.commands = build_worker_command_queue(factory)
        self.approvals = build_approval_queue(factory, registry)
        self.approvals.clock = lambda: calf.NOW
        self.worker = Worker(commands=self.commands, runners=self.runners, idle_seconds=0)
        self.thread = register_supervisor_thread(
            conversation_key=f"chat-{uuid4()}",
            registry=registry,
            user_id=user_id,
            actor_id=ACTOR,
        )
        self.principal = Principal(ACTOR, AuthMethod.LOCAL_TOKEN, scopes=HUMAN_SCOPES)

    def chat(self, message: str = REQUEST) -> None:
        self.commands.enqueue(
            WorkflowCommand(
                command_id=uuid4(),
                kind=WorkflowCommandKind.START,
                workflow_id=self.thread.workflow_id,
                thread_id=self.thread.thread_id,
                actor_id=ACTOR,
                correlation_id=uuid4(),
                graph_input={"message": message},
            )
        )

    def pending(self) -> dict[ActionKind, Any]:
        return {
            ActionKind(p.action_type): p
            for p in self.approvals.list_pending(access=ActorScope.of(ACTOR))
        }

    def decide(self, approval, approved: bool) -> None:
        self.approvals.decide(
            approval.approval_id,
            approved=approved,
            action_hash=approval.action_hash,
            principal=self.principal,
        )

    async def view(self) -> PlanView:
        snapshot = await self.supervisor.aget_state(self.thread.config())
        return PlanView.model_validate(snapshot.values["result"]["plan"])


async def test_a_plan_runs_through_the_worker_with_one_approval_per_effect(plant):
    plant.chat()
    await plant.worker.drain()

    pending = plant.pending()
    assert set(pending) == {ActionKind.SEND_EMAIL, ActionKind.CREATE_CALENDAR_EVENT}
    hold, reply = pending[ActionKind.CREATE_CALENDAR_EVENT], pending[ActionKind.SEND_EMAIL]
    # Two approvals, each on its own step thread, each bound to its own hash.
    assert hold.thread_id != reply.thread_id
    assert hold.action_hash != reply.action_hash
    # Every step thread is a thread of the one workflow, owned by the candidate.
    threads = plant.registry.threads_for_workflow(plant.thread.workflow_id)
    assert {thread_namespace(t.thread_id) for t in threads} == {
        SUPERVISOR_THREAD_NAMESPACE,
        CALENDAR_THREAD_NAMESPACE,
        COMMUNICATIONS_THREAD_NAMESPACE,
    }
    assert {hold.workflow_id, reply.workflow_id} == {plant.thread.workflow_id}
    assert all(plant.registry.user_id_for(t.thread_id) == plant.user_id for t in threads)
    # Parked on approvals, the plan is not an active workflow at all; while
    # running it is one, however many steps it has.
    assert plant.commands.active_workflow_ids(ACTOR) == set()

    # The hold is approved; the reply is rejected.
    plant.decide(hold, approved=True)
    plant.decide(reply, approved=False)
    assert plant.commands.active_workflow_ids(ACTOR) == {plant.thread.workflow_id}
    await plant.worker.drain()

    view = await plant.view()
    steps = {s.step_id: s for s in view.steps}
    assert steps["calendar"].status is StepStatus.SUCCEEDED
    assert steps["reply"].status is StepStatus.REJECTED
    (dangling,) = view.dangling
    (event_id,) = plant.calendar.calendar.created
    assert dangling.effect.external_reference == event_id
    # The release is proposed as an approval of its own -- and nothing is gone yet.
    (release,) = plant.pending().values()
    assert release.action_type == ActionKind.CANCEL_CALENDAR_EVENT.value
    assert event_id in plant.calendar.calendar.events
    assert plant.comms.sends() == []

    plant.decide(release, approved=True)
    await plant.worker.drain()

    assert plant.calendar.calendar.cancelled == [event_id]
    view = await plant.view()
    assert {s.step_id: s.status for s in view.steps}[dangling.release_step_id] is (
        StepStatus.SUCCEEDED
    )
    trail = AuditTrailStore(plant.factory).workflow_trail(
        plant.thread.workflow_id, access=SYSTEM_ACCESS
    )
    actions = [e.action for e in trail.entries]
    assert actions[0] == "plan.proposed" and actions[-1] == "plan.finished"
    # The plan's approval entries link to the decisions people recorded.
    approved = [e for e in trail.entries if e.action == "plan.step.approval.approved"]
    assert approved and all(e.links.approval is not None for e in approved)
