"""Cross-domain task plans (K4): one request, a DAG of steps, each run by its own subgraph.

The Supervisor plans "reply to the recruiter and hold Thursday at 2pm" as two
steps -- a Communications reply and a Calendar hold -- and runs each through
that domain's real subgraph: the Communications subgraph over the Gmail tool
path against a fake Google API (`tests.fixtures.communications`), the Calendar
subgraph over the in-memory calendar (`tests.fixtures.calendar`). Each step is
on a thread of its own, and each effect is approved on that thread.

The acceptance criteria this file exists for:

1. 'reply to the recruiter and hold Thursday at 2pm' produces a two-domain DAG
   with the calendar hold and the reply as separate approval requests.
2. A cyclic or over-length plan is rejected and does not execute.
3. A crash after step 1 completes resumes at step 2 without repeating step 1's
   side effect.
4. Rejecting the reply approval leaves the hold flagged as dangling with a
   proposed release, and nothing is deleted without approval.
5. The audit trail shows the plan, each step's intent, policy decision,
   approval, and receipt in order.

Plus: data moves between steps by reference; a rejection stops only the steps
that depend on it; every step thread is a thread of the one workflow.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from personalos.domain.job_search import ActionKind, ApprovalDecision, ApprovalVerdict
from personalos.domain.ownership import SYSTEM_ACCESS
from personalos.domain.routing import RouteDomain
from personalos.domain.task_plan import (
    AUDIT_PLAN_PROPOSED,
    AUDIT_PLAN_REJECTED,
    AUDIT_STEP_INTENT,
    AUDIT_STEP_POLICY,
    AUDIT_STEP_RECEIPT,
    Compensation,
    PlanView,
    StepStatus,
    TaskPlan,
    plan_proposed_entries,
)
from personalos.graphs.plan_steps import LocalStepThreads, StepContext, SubgraphStepRunner
from personalos.graphs.supervisor import (
    CLASSIFY_INTENT,
    RUN_PLAN_STEP,
    STAGE_PLAN_STEPS,
    SupervisorGraph,
)
from personalos.models.routing import KeywordIntentClassifier
from personalos.models.task_planning import ResilientTaskPlanner
from personalos.persistence.audit_trail import AuditTrailStore
from personalos.persistence.models import WorkflowModel
from personalos.persistence.plan_audit import SqlPlanAuditRecorder
from tests.fixtures import calendar as calf
from tests.fixtures import communications as cf

REQUEST = "reply to the recruiter and hold Thursday at 2pm"
SEQUENTIAL = "hold Thursday at 2pm, then reply to the recruiter to confirm"


class RecordingJobSubgraph:
    def __init__(self):
        self.calls: list[Any] = []

    async def __call__(self, task_dag, state):
        self.calls.append(task_dag)
        return {}


class CountingRunner:
    """Wraps a step runner; counts runs, and can crash the next run of a step."""

    def __init__(self, inner: SubgraphStepRunner):
        self.inner = inner
        self.runs: list[str] = []
        self.polls: list[str] = []
        self.crash_next = False

    async def run(self, step, inputs, context):
        self.runs.append(step.id)
        if self.crash_next:
            self.crash_next = False
            raise RuntimeError("worker died running this step")
        self.last_inputs = dict(inputs)
        return await self.inner.run(step, inputs, context)

    async def poll(self, step, record, context):
        self.polls.append(step.id)
        return await self.inner.poll(step, record, context)


class ScriptedPlanner:
    """Returns the drafts it was given, one per call, and records the feedback it got."""

    def __init__(self, *drafts: dict[str, Any]):
        self.drafts = list(drafts)
        self.feedback: list[str | None] = []

    async def plan(self, request, *, domains, max_steps, feedback=None):
        self.feedback.append(feedback)
        return self.drafts[min(len(self.feedback), len(self.drafts)) - 1]


class World:
    def __init__(self, comms: cf.CommsWorld, *, planner=None, max_plan_steps: int = 6):
        self.comms = comms
        self.user_id, self.application_id = cf.applied_application(comms.factory)
        comms.add_mail("m-1")
        self.calendar = calf.build_world(clock_at=calf.NOW)
        binder = LocalStepThreads(self.user_id)
        self.calendar_graph = self.calendar.graph.build()
        self.comms_graph = comms.graph.build()
        self.calendar_runner = CountingRunner(
            SubgraphStepRunner(self.calendar_graph, RouteDomain.CALENDAR, binder=binder)
        )
        self.comms_runner = CountingRunner(
            SubgraphStepRunner(self.comms_graph, RouteDomain.COMMUNICATIONS, binder=binder)
        )
        self.workflow_id = uuid4()
        session = comms.factory()
        session.add(WorkflowModel(id=self.workflow_id, name="supervisor"))
        session.commit()
        session.close()
        self.audit = SqlPlanAuditRecorder(comms.factory, clock=lambda: datetime(2026, 11, 23, 14))
        self.job = RecordingJobSubgraph()
        self.planner = ResilientTaskPlanner(planner) if planner else ResilientTaskPlanner()
        self.supervisor = SupervisorGraph(
            KeywordIntentClassifier(),
            self.job,
            task_planner=self.planner,
            step_runners={
                RouteDomain.CALENDAR: self.calendar_runner,
                RouteDomain.COMMUNICATIONS: self.comms_runner,
            },
            plan_audit=self.audit,
            max_plan_steps=max_plan_steps,
            checkpointer=InMemorySaver(),
        ).build()
        self.config = {
            "configurable": {
                "thread_id": f"supervisor:{uuid4().hex}",
                "workflow_id": str(self.workflow_id),
                "actor_id": "sam",
            }
        }

    async def start(self, message: str = REQUEST) -> dict[str, Any]:
        return await self.supervisor.ainvoke({"message": message}, self.config)

    async def resume(self) -> dict[str, Any]:
        return await self.supervisor.ainvoke(Command(resume={"plan_steps": "moved"}), self.config)

    def view(self, final: dict[str, Any]) -> PlanView:
        return PlanView.model_validate(final["result"]["plan"])

    def step_config(self, final: dict[str, Any], step_id: str) -> dict[str, Any]:
        record = final["plan_records"][step_id]
        return {"configurable": {"thread_id": record["thread_id"]}}

    async def requests(self, graph, config) -> list[dict[str, Any]]:
        snapshot = await graph.aget_state(config)
        (task,) = [t for t in snapshot.tasks if t.interrupts]
        return task.interrupts[0].value["requests"]

    async def answer(self, final, step_id: str, verdict: ApprovalVerdict) -> dict[str, Any]:
        graph = (
            self.calendar_graph
            if final["plan_records"][step_id]["thread_id"].startswith("calendar:")
            else self.comms_graph
        )
        config = self.step_config(final, step_id)
        decisions = [
            ApprovalDecision(
                action_id=r["action_id"],
                action_fingerprint=r["action_hash"],
                verdict=verdict,
                decided_by="sam",
                decided_at=cf.NOW if graph is self.comms_graph else calf.NOW,
                request_id=r["request_id"],
            ).model_dump(mode="json")
            for r in await self.requests(graph, config)
        ]
        return await graph.ainvoke(Command(resume=decisions), config)

    def trail(self) -> list[Any]:
        trail = AuditTrailStore(self.comms.factory).workflow_trail(
            self.workflow_id, access=SYSTEM_ACCESS
        )
        assert trail is not None
        return list(trail.entries)


@pytest.fixture
async def comms():
    w = await cf.build_world()
    try:
        yield w
    finally:
        await w.google.stop()


@pytest.fixture
def world(comms):
    return World(comms)


def steps_by_id(view: PlanView) -> dict[str, Any]:
    return {step.step_id: step for step in view.steps}


# --- 1. Two domains, two approvals -------------------------------------------------------


async def test_reply_and_hold_is_a_two_domain_dag_with_separate_approvals(world):
    final = await world.start()

    assert world.job.calls == []
    view = world.view(final)
    steps = steps_by_id(view)
    assert {s.domain for s in view.steps} == {RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR}
    assert set(view.awaiting_approval) == {"reply", "calendar"}
    # Independent: neither waits for the other to be approved.
    assert steps["reply"].depends_on == () and steps["calendar"].depends_on == ()

    (reply,) = steps["reply"].approvals
    (hold,) = steps["calendar"].approvals
    assert reply.kind is ActionKind.SEND_EMAIL
    assert hold.kind is ActionKind.CREATE_CALENDAR_EVENT
    assert reply.request_id != hold.request_id and reply.action_hash != hold.action_hash

    # Each is its own request, on its own step thread -- not one batched approval.
    reply_requests = await world.requests(world.comms_graph, world.step_config(final, "reply"))
    hold_requests = await world.requests(world.calendar_graph, world.step_config(final, "calendar"))
    assert [r["request_id"] for r in reply_requests] == [str(reply.request_id)]
    assert [r["request_id"] for r in hold_requests] == [str(hold.request_id)]

    # The Supervisor parks on the steps, with no approval of its own.
    snapshot = await world.supervisor.aget_state(world.config)
    (parked,) = [t for t in snapshot.tasks if t.interrupts]
    value = parked.interrupts[0].value
    assert value["stage"] == STAGE_PLAN_STEPS and "requests" not in value
    # Nothing has been written anywhere yet.
    assert world.calendar.calendar.created == [] and world.comms.sends() == []


async def test_the_whole_plan_is_shown_and_audited_before_any_step_runs(world):
    final = await world.start()

    assert final["task_dag"]["tasks"][0]["id"] == "reply"
    trail = world.trail()
    first_intent = next(i for i, e in enumerate(trail) if e.action == AUDIT_STEP_INTENT)
    proposed = [i for i, e in enumerate(trail) if e.action == AUDIT_PLAN_PROPOSED]
    assert proposed and proposed[0] < first_intent
    planned = [e.target_ref for e in trail[:first_intent] if e.action == "plan.step.planned"]
    assert any("[communications]: reply to the recruiter" in ref for ref in planned)
    assert any("[calendar]: hold Thursday at 2pm" in ref for ref in planned)


async def test_approving_both_completes_the_plan(world):
    final = await world.start()
    await world.answer(final, "calendar", ApprovalVerdict.APPROVED)
    await world.answer(final, "reply", ApprovalVerdict.APPROVED)

    final = await world.resume()

    view = world.view(final)
    assert view.complete and set(view.ran) == {"reply", "calendar"}
    assert view.dangling == ()
    assert len(world.calendar.calendar.created) == 1 and len(world.comms.sends()) == 1


# --- 2. A cyclic or over-length plan does not run ----------------------------------------


def _step(step_id: str, domain: str, *deps: str) -> dict[str, Any]:
    return {
        "id": step_id,
        "domain": domain,
        "instruction": f"do {step_id}",
        "depends_on": list(deps),
    }


CYCLIC = {
    "goal": "cyclic",
    "steps": [_step("a", "calendar", "b"), _step("b", "communications", "a")],
}
OVER_LENGTH = {
    "goal": "long",
    "steps": [_step(f"s{i}", "calendar" if i % 2 else "communications") for i in range(7)],
}


@pytest.mark.parametrize("draft", [CYCLIC, OVER_LENGTH], ids=["cyclic", "over_length"])
async def test_an_invalid_plan_is_repaired_then_refused_and_never_runs(comms, draft):
    planner = ScriptedPlanner(draft)
    world = World(comms, planner=planner, max_plan_steps=6)

    final = await world.start()

    # Asked once, then twice more with every issue listed -- then not run.
    assert len(planner.feedback) == 3
    assert planner.feedback[0] is None and all(planner.feedback[1:])
    issue = "cycle" if draft is CYCLIC else "at most 6"
    assert issue in planner.feedback[1]
    assert final["plan_kind"] == "invalid"
    assert final.get("task_plan") is None and final.get("result") is None
    assert "couldn't turn that into a plan" in final["clarification"]
    assert world.calendar_runner.runs == [] and world.comms_runner.runs == []
    assert world.job.calls == []
    assert [e.action for e in world.trail()] == [AUDIT_PLAN_REJECTED]


async def test_a_plan_with_an_unknown_domain_is_refused(comms):
    draft = {"goal": "x", "steps": [_step("a", "calendar"), _step("b", "weather")]}
    world = World(comms, planner=ScriptedPlanner(draft))

    final = await world.start()

    assert final["plan_kind"] == "invalid"
    assert any("schema" in e for e in final["plan_errors"])
    assert world.calendar_runner.runs == []


async def test_a_plan_naming_an_unwired_domain_is_refused(comms):
    draft = {"goal": "x", "steps": [_step("a", "calendar"), _step("b", "file")]}
    world = World(comms, planner=ScriptedPlanner(draft))

    final = await world.start()

    assert final["plan_kind"] == "invalid"
    assert any("'file' is not one this build can run" in e for e in final["plan_errors"])
    assert world.calendar_runner.runs == []


async def test_a_repaired_plan_runs(comms):
    fixed = {
        "goal": "fixed",
        "steps": [
            {**_step("hold", "calendar"), "instruction": "hold Thursday at 2pm"},
            {**_step("reply", "communications"), "instruction": "reply to the recruiter"},
        ],
    }
    planner = ScriptedPlanner(CYCLIC, fixed)
    world = World(comms, planner=planner)

    final = await world.start()

    assert len(planner.feedback) == 2
    assert set(world.view(final).awaiting_approval) == {"hold", "reply"}


async def test_a_single_domain_request_takes_the_ordinary_route(world):
    final = await world.start("when am I free Thursday")

    assert final["plan_kind"] == "single"
    assert final["route_decision"]["domain"] == RouteDomain.CALENDAR.value
    assert world.calendar_runner.runs == []


# --- 3. A crash after step 1 resumes at step 2 ---------------------------------------------


async def test_a_crash_after_step_one_resumes_at_step_two_without_repeating_it(world):
    final = await world.start(SEQUENTIAL)
    view = world.view(final)
    assert steps_by_id(view)["reply"].depends_on == ("calendar",)
    # The reply waits for the hold; only the hold has run.
    assert view.awaiting_approval == ("calendar",) and view.pending == ("reply",)
    assert world.comms_runner.runs == []

    await world.answer(final, "calendar", ApprovalVerdict.APPROVED)
    assert len(world.calendar.calendar.created) == 1

    world.comms_runner.crash_next = True
    with pytest.raises(RuntimeError, match="worker died"):
        await world.resume()

    # The hold's completion was checkpointed; the run stopped at step 2.
    snapshot = await world.supervisor.aget_state(world.config)
    assert snapshot.next == (RUN_PLAN_STEP,)
    assert snapshot.values["plan_records"]["calendar"]["status"] == StepStatus.SUCCEEDED.value

    final = await world.supervisor.ainvoke(None, world.config)

    assert world.calendar_runner.runs == ["calendar"]
    assert world.comms_runner.runs == ["reply", "reply"]
    assert len(world.calendar.calendar.created) == 1
    assert world.view(final).awaiting_approval == ("reply",)


async def test_a_rerun_step_reads_its_thread_instead_of_starting_again(world):
    final = await world.start(SEQUENTIAL)
    calendar_thread = final["plan_records"]["calendar"]["thread_id"]
    record = await world.calendar_runner.inner.run(
        _plan_step(final, "calendar"), {}, _context(world, final)
    )
    assert record.thread_id == calendar_thread
    assert record.status is StepStatus.AWAITING_APPROVAL
    # One proposal on that thread, not two.
    assert (
        len(await world.requests(world.calendar_graph, world.step_config(final, "calendar"))) == 1
    )


def _plan_step(final, step_id):
    return _plan(final).step(step_id)


def _context(world, final) -> StepContext:
    return StepContext(
        plan_id=final["task_plan"]["plan_id"],
        parent_thread_id=world.config["configurable"]["thread_id"],
        workflow_id=str(world.workflow_id),
    )


# --- 4. A rejected reply leaves the hold dangling, with a proposed release ---------------


async def test_rejecting_the_reply_flags_the_hold_and_proposes_its_release(world):
    final = await world.start()
    await world.answer(final, "calendar", ApprovalVerdict.APPROVED)
    (event_id,) = world.calendar.calendar.created
    await world.answer(final, "reply", ApprovalVerdict.REJECTED)

    final = await world.resume()

    view = world.view(final)
    steps = steps_by_id(view)
    assert steps["reply"].status is StepStatus.REJECTED
    assert steps["calendar"].status is StepStatus.SUCCEEDED
    (dangling,) = view.dangling
    assert dangling.step_id == "calendar"
    assert dangling.effect.external_reference == event_id
    assert dangling.compensation is Compensation.PROPOSE_RELEASE
    release = steps[dangling.release_step_id]
    assert release.compensates and release.status is StepStatus.AWAITING_APPROVAL
    (cancel,) = release.approvals
    assert cancel.kind is ActionKind.CANCEL_CALENDAR_EVENT and cancel.state == "pending"
    # The release targets the hold by reference, not by a re-parsed description.
    assert world.calendar_runner.last_inputs["event"].id == event_id

    # Proposed, not done: the hold is still on the calendar.
    assert event_id in world.calendar.calendar.events
    assert world.calendar.calendar.cancelled == []
    assert world.comms.sends() == []


async def test_declining_the_release_leaves_the_hold_in_place(world):
    final = await world.start()
    await world.answer(final, "calendar", ApprovalVerdict.APPROVED)
    (event_id,) = world.calendar.calendar.created
    await world.answer(final, "reply", ApprovalVerdict.REJECTED)
    final = await world.resume()
    release_id = world.view(final).dangling[0].release_step_id

    await world.answer(final, release_id, ApprovalVerdict.REJECTED)
    final = await world.resume()

    assert event_id in world.calendar.calendar.events
    assert world.calendar.calendar.cancelled == []
    view = world.view(final)
    assert steps_by_id(view)[release_id].status is StepStatus.REJECTED
    assert view.dangling[0].effect.external_reference == event_id
    # Nothing more is proposed for it.
    assert [s.step_id for s in view.steps if s.compensates] == [release_id]
    assert snapshot_done(await world.supervisor.aget_state(world.config))


async def test_approving_the_release_cancels_the_hold(world):
    final = await world.start()
    await world.answer(final, "calendar", ApprovalVerdict.APPROVED)
    (event_id,) = world.calendar.calendar.created
    await world.answer(final, "reply", ApprovalVerdict.REJECTED)
    final = await world.resume()
    release_id = world.view(final).dangling[0].release_step_id

    await world.answer(final, release_id, ApprovalVerdict.APPROVED)
    final = await world.resume()

    assert world.calendar.calendar.cancelled == [event_id]
    assert steps_by_id(world.view(final))[release_id].status is StepStatus.SUCCEEDED


async def test_a_rejection_stops_only_the_steps_that_depend_on_it(world):
    final = await world.start(SEQUENTIAL)

    await world.answer(final, "calendar", ApprovalVerdict.REJECTED)
    final = await world.resume()

    view = world.view(final)
    assert view.rejected == ("calendar",) and view.skipped == ("reply",)
    assert world.comms_runner.runs == []
    # Nothing happened, so nothing dangles.
    assert view.dangling == () and world.calendar.calendar.created == []


def snapshot_done(snapshot) -> bool:
    return not snapshot.next


# --- 5. The audit trail, in order --------------------------------------------------------


async def test_the_audit_trail_shows_plan_intent_policy_approval_and_receipt_in_order(world):
    final = await world.start()
    await world.answer(final, "calendar", ApprovalVerdict.APPROVED)
    await world.answer(final, "reply", ApprovalVerdict.REJECTED)
    await world.resume()

    trail = world.trail()
    actions = [e.action for e in trail]
    assert actions[0] == AUDIT_PLAN_PROPOSED

    def story(step_id: str) -> list[tuple[str, Any]]:
        prefix = f"/{step_id}"
        return [
            (e.action, e)
            for e in trail
            if prefix in e.target_ref.split(":")[0] or f"{prefix}/" in e.target_ref
        ]

    for step_id, verdict, result in (
        ("calendar", "plan.step.approval.approved", "success"),
        ("reply", "plan.step.approval.rejected", "failure"),
    ):
        names = [name for name, _ in story(step_id) if name != "plan.step.planned"]
        assert names[:4] == [AUDIT_STEP_INTENT, AUDIT_STEP_POLICY, verdict, AUDIT_STEP_RECEIPT]
        by_name = dict(story(step_id))
        assert by_name[AUDIT_STEP_POLICY].policy_decision == "require_approval"
        assert by_name[verdict].result == result
        assert by_name[AUDIT_STEP_RECEIPT].result == result
    # The hold's receipt names the event it made.
    (event_id,) = world.calendar.calendar.created
    assert event_id in dict(story("calendar"))[AUDIT_STEP_RECEIPT].target_ref
    assert "plan.effect.dangling" in actions
    # Strictly ordered: nothing shares a timestamp to be reordered on.
    times = [e.at for e in trail]
    assert times == sorted(times) and len(set(times)) == len(times)


async def test_recording_the_same_story_twice_adds_nothing(world):
    final = await world.start()
    before = len(world.trail())

    # A re-run of the Supervisor's last step re-records what it had recorded.
    world.audit.record(
        plan_proposed_entries(_plan(final)),
        workflow_id=str(world.workflow_id),
        actor_id="sam",
    )

    assert len(world.trail()) == before


def _plan(final) -> TaskPlan:
    return TaskPlan.model_validate(final["task_plan"])


# --- Scope -------------------------------------------------------------------------------


async def test_every_step_runs_on_its_own_thread_for_the_conversations_candidate(world):
    final = await world.start()

    threads = {step_id: r["thread_id"] for step_id, r in final["plan_records"].items()}
    assert threads["reply"].startswith("communications:")
    assert threads["calendar"].startswith("calendar:")
    assert len(set(threads.values())) == 2
    calendar_state = await world.calendar_graph.aget_state(world.step_config(final, "calendar"))
    comms_state = await world.comms_graph.aget_state(world.step_config(final, "reply"))
    assert calendar_state.values["user_id"] == str(world.user_id)
    assert comms_state.values["user_id"] == str(world.user_id)
    # Each step was handed its own words, not the whole request or anything read.
    assert calendar_state.values["request"] == "hold Thursday at 2pm"
    assert comms_state.values["request"] == "reply to the recruiter"


def test_planning_is_off_without_two_step_runners():
    graph = SupervisorGraph(
        KeywordIntentClassifier(), RecordingJobSubgraph(), task_planner=ResilientTaskPlanner()
    )
    assert not graph.plans_enabled
    compiled = graph.build()
    assert "plan_task" not in compiled.get_graph().nodes
    assert CLASSIFY_INTENT in compiled.get_graph().nodes


def test_user_id_is_a_uuid_in_the_world(world):
    assert isinstance(world.user_id, UUID)
