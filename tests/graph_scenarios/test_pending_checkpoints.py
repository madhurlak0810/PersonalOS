"""Durable conditional waits, end to end: schedule, wait, re-evaluate, act.

The three acceptance criteria this file exists for, each as one test:

1. **A wait whose condition resolves early is closed without a follow-up.**
   `test_a_wait_whose_condition_resolved_during_the_wait_is_closed_silently`
   schedules a real wait, lets the recruiter reply during the seven days, and
   sweeps at the trigger. Nothing is drafted, nothing is emitted, and the row
   says `resolved` -- not `expired`, which is the other silence and means the
   opposite thing.
2. **A wait whose condition is still unmet at its trigger fires and resumes the
   right graph path.**
   `test_a_wait_still_unmet_at_its_trigger_fires_and_resumes_the_follow_up_path`
   sweeps with the condition unmet and asserts the thread comes back at the
   follow-up draft -- not at discovery -- and parks at the approval checkpoint
   with a reviewable request.
3. **A wait that never fires expires rather than staying pending forever.**
   `test_a_wait_nobody_swept_in_time_is_marked_expired_not_left_pending` skips
   the sweep entirely until after the expiry and asserts the checkpoint is
   written off, with nothing sent.

Everything runs against a file-backed SQLite database through the real durable
checkpointer, the real store and the real monitor, because the property under
test is precisely that a wait survives the run that created it: the checkpoint
is scheduled by a run that has *finished* by the time the sweep picks it up, and
the only thing connecting the two is a row.

Time is moved rather than waited for. Every decision in
`personalos.domain.checkpoints` takes an explicit `now`, and every sweep here
passes one, so a seven-day wait is a seven-day test only in the fiction.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.worker.checkpoint_monitor import PendingCheckpointMonitor
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.domain.checkpoints import (
    ConditionKind,
    PendingCheckpointStatus,
)
from personalos.domain.job_search import (
    ActionKind,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    FollowUpKind,
    JobSearchEventType,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.workflow import job_search_thread_id
from personalos.graphs.job_search import (
    APPROVAL_CHECKPOINT,
    LOAD_SEARCH_PROFILE,
    JobSearchGraph,
)
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import WorkflowLeaseStore
from personalos.persistence.models import Base
from personalos.persistence.pending_checkpoints import (
    PendingCheckpointStore,
    StorePendingCheckpointScheduler,
)
from tests.fixtures import job_search_fakes as fakes

#: The moment the first run happens. Everything else is expressed relative to
#: it, so a reader can see at a glance which side of the trigger a sweep is on.
NOW = datetime(2026, 9, 29, 12, 0, 0)
#: The follow-up branch schedules a no-response nudge seven days out; see
#: `personalos.graphs.job_search.NO_RESPONSE_FOLLOW_UP_DAYS`.
AT_TRIGGER = NOW + timedelta(days=7)
BEFORE_TRIGGER = NOW + timedelta(days=2)


class ApproveSubmissionsOnlyGate:
    """A standing approval for submissions, and nothing on file for messages.

    Lets one deployment stand in for two moments: the original application goes
    through without a human (so the test reaches the state a follow-up hangs
    off) while the follow-up message it later prompts genuinely parks at the
    interrupt -- which is the property being checked. A gate that approved both
    would prove the monitor can start a run, but not that a run the monitor
    started still cannot write to the outside world unattended.
    """

    def __init__(self):
        self.reviewed = []

    async def review(self, intent) -> ApprovalDecision:
        self.reviewed.append(intent)
        verdict = (
            ApprovalVerdict.APPROVED
            if intent.kind == ActionKind.SUBMIT_APPLICATION
            else ApprovalVerdict.PENDING
        )
        return ApprovalDecision(
            action_id=intent.action_id,
            action_fingerprint=intent.fingerprint(),
            verdict=verdict,
            decided_by="reviewer@example.test",
        )


class Deployment:
    """One wired-up deployment: durable graph, durable store, and a monitor over both.

    A class rather than a pile of fixtures because the interesting thing is that
    all three talk to the *same database* and to nothing else. The graph
    schedules a wait; the monitor finds it; the runner starts the thread the wait
    named. Nothing is handed between them in memory.
    """

    def __init__(self, tmp_path, *, evaluator, clock=lambda: NOW):
        engine = create_engine(f"sqlite:///{tmp_path / 'waits.db'}")
        Base.metadata.create_all(engine)
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        self.registry = WorkflowThreadRegistry(self.factory)
        self.store = PendingCheckpointStore(self.factory, clock=clock)
        self.gate = ApproveSubmissionsOnlyGate()
        self.ports = {
            "profile_store": fakes.FakeProfileStore(),
            "providers": [fakes.FakeProvider()],
            "scorer": fakes.FakeScorer(),
            "evidence_checker": fakes.FakeEvidenceChecker(),
            "packet_builder": fakes.FakePacketBuilder(),
            "approval_gate": self.gate,
            "action_executor": fakes.FakeActionExecutor(),
            "application_store": fakes.FakeApplicationStore(),
            "event_emitter": fakes.FakeEventEmitter(),
            # An inbox with nothing in it is the case the feature is named for:
            # "seven days after applying, if no recruiter response exists".
            "recruiter_inbox": fakes.FakeRecruiterInbox(),
            "recruiter_classifier": fakes.FakeRecruiterClassifier(),
            "checkpoint_scheduler": StorePendingCheckpointScheduler(self.store),
            "checkpointer": SqlAlchemyCheckpointSaver(self.factory, self.registry),
            "clock": clock,
        }
        self.graph = JobSearchGraph(**self.ports).build()
        self.runner = DurableWorkflowRunner(
            self.graph,
            registry=self.registry,
            leases=WorkflowLeaseStore(self.factory),
            owner="worker-1",
        )
        self.evaluator = evaluator
        self.monitor = PendingCheckpointMonitor(
            store=self.store,
            evaluator=evaluator,
            runner=self.runner,
            registry=self.registry,
            clock=clock,
        )
        self.thread = self.registry.register(
            thread_id=job_search_thread_id(fakes.USER_ID, "waits"),
            workflow_name="job_search",
            user_id=fakes.USER_ID,
        )

    async def apply(self):
        """Run the pipeline through to the follow-up branch, as a real search does."""
        return await self.runner.start(
            self.thread, {"user_id": str(fakes.USER_ID), "prepare_application": True}
        )

    @property
    def events(self) -> list[str]:
        return self.ports["event_emitter"].types()

    @property
    def searches(self) -> int:
        return len(self.ports["providers"][0].calls)


async def applied(tmp_path, *, evaluator):
    """A deployment that has applied once and scheduled its no-response wait."""
    deployment = Deployment(tmp_path, evaluator=evaluator)
    final = await deployment.apply()
    assert final["application"]["status"] == ApplicationStatus.APPLIED.value
    return deployment


# --- What the run leaves behind ----------------------------------------------


async def test_applying_schedules_a_durable_wait_that_outlives_the_run(tmp_path):
    """The wait is a row, and it carries everything a later process needs.

    Nothing in it refers to the run that made it: by the time it matters, that
    run has finished, its worker has exited and its graph object is gone. What
    is left is a condition to re-ask, a moment to ask it at, an expiry, and the
    thread id to come back to.
    """
    evaluator = fakes.ScriptedConditionEvaluator()
    deployment = await applied(tmp_path, evaluator=evaluator)

    waits = deployment.store.open_for_application(fakes.APPLICATION_ID)

    assert len(waits) == 1
    wait = waits[0]
    assert wait.kind == FollowUpKind.NO_RESPONSE
    assert wait.condition.kind == ConditionKind.RECRUITER_RESPONSE_RECEIVED
    assert wait.condition.subject_id == fakes.APPLICATION_ID
    assert wait.trigger_at == AT_TRIGGER
    assert wait.expires_at > wait.trigger_at
    # The thread the wait will come back to is the one this run ran on.
    assert wait.thread_id == deployment.thread.thread_id
    assert wait.workflow_id == deployment.thread.workflow_id
    # The condition has not been asked. Asking it now would answer a question
    # about today, and the question is about the 6th of October.
    assert evaluator.asked == []


async def test_a_wait_is_not_actionable_before_its_trigger(tmp_path):
    """A sweep two days in does nothing, and asks nothing."""
    evaluator = fakes.ScriptedConditionEvaluator()
    deployment = await applied(tmp_path, evaluator=evaluator)

    report = await deployment.monitor.sweep(now=BEFORE_TRIGGER)

    assert report.considered == 0
    assert evaluator.asked == []
    assert deployment.store.open_for_application(fakes.APPLICATION_ID)


# --- Acceptance 1: resolved early, closed silently ---------------------------


async def test_a_wait_whose_condition_resolved_during_the_wait_is_closed_silently(tmp_path):
    """A recruiter who replies during the seven days cancels the follow-up.

    The first acceptance criterion. The condition is re-asked at the trigger --
    not at the moment the wait was created, when the answer was still "no" --
    and because it now holds, the checkpoint closes with no draft, no event and
    no run. The row says `resolved`, which is how an operator tells this silence
    from the expired one.
    """
    # By the trigger, a reply has landed. The evaluator is what knows that; the
    # checkpoint only knows which question to ask.
    evaluator = fakes.ScriptedConditionEvaluator(
        {ConditionKind.RECRUITER_RESPONSE_RECEIVED: True}
    )
    deployment = await applied(tmp_path, evaluator=evaluator)
    wait = deployment.store.open_for_application(fakes.APPLICATION_ID)[0]
    events_before = list(deployment.events)

    report = await deployment.monitor.sweep(now=AT_TRIGGER)

    assert report.resolved == (wait.checkpoint_id,)
    assert report.fired == ()
    # Asked once, at the trigger, with the condition that was stored a week ago.
    assert [condition.kind for condition in evaluator.asked] == [
        ConditionKind.RECRUITER_RESPONSE_RECEIVED
    ]

    # Silently: no follow-up event, and no follow-up action anywhere.
    assert deployment.events == events_before
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value not in deployment.events
    assert not any(
        intent.kind == ActionKind.SEND_RECRUITER_MESSAGE for intent in deployment.gate.reviewed
    )

    closed = deployment.store.get(wait.checkpoint_id)
    assert closed.status == PendingCheckpointStatus.RESOLVED
    assert "condition met" in closed.closed_reason
    # And the thread was never started again: still finished, still at no step.
    state = await deployment.runner.inspect(deployment.thread)
    assert state.next == ()


async def test_a_resolution_noticed_as_it_happens_closes_the_wait_before_the_trigger(tmp_path):
    """The other way a wait resolves early: the event closes it when it lands.

    An optimization over waiting for the trigger, not a second mechanism --
    nothing depends on it, as the previous test shows. What it buys is a table
    that is honest *now* about what is still outstanding, rather than five days
    from now.
    """
    evaluator = fakes.ScriptedConditionEvaluator()
    deployment = await applied(tmp_path, evaluator=evaluator)
    wait = deployment.store.open_for_application(fakes.APPLICATION_ID)[0]

    closed = deployment.store.resolve_matching(
        condition_kind=ConditionKind.RECRUITER_RESPONSE_RECEIVED,
        subject_id=fakes.APPLICATION_ID,
        reason="recruiter replied on day two",
        occurred_at=BEFORE_TRIGGER,
        now=BEFORE_TRIGGER,
    )

    assert closed == [wait.checkpoint_id]
    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.RESOLVED

    # And the sweep at the trigger finds nothing left to do.
    report = await deployment.monitor.sweep(now=AT_TRIGGER)
    assert report.considered == 0
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value not in deployment.events


# --- Acceptance 2: still unmet at the trigger, fires --------------------------


async def test_a_wait_still_unmet_at_its_trigger_fires_and_resumes_the_follow_up_path(tmp_path):
    """Seven days on with no reply, the wait starts the follow-up path. Not discovery.

    The second acceptance criterion, and the "correct workflow" in it is doing
    real work: the thread resumes on top of the state it already holds, so the
    draft is about the application this thread has been about all along, and the
    job boards are not searched again.

    It parks at the approval checkpoint rather than sending anything. A run
    started by a monitor at 3am is still a run that cannot write to the outside
    world without a human, and that is structural -- the draft node has no
    executor and no edge to one.
    """
    evaluator = fakes.ScriptedConditionEvaluator(
        {ConditionKind.RECRUITER_RESPONSE_RECEIVED: False}
    )
    deployment = await applied(tmp_path, evaluator=evaluator)
    wait = deployment.store.open_for_application(fakes.APPLICATION_ID)[0]
    searches_before = deployment.searches

    report = await deployment.monitor.sweep(now=AT_TRIGGER)

    assert report.fired == (wait.checkpoint_id,)
    assert report.resolved == () and report.expired == ()
    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.FIRED

    # The follow-up was drafted and announced...
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value in deployment.events

    # ...and the run is parked at the approval checkpoint, holding a reviewable
    # request for the message rather than having sent one.
    state = await deployment.runner.inspect(deployment.thread)
    assert state.next == (APPROVAL_CHECKPOINT,)
    assert LOAD_SEARCH_PROFILE not in state.next

    requests = [
        ApprovalRequest.model_validate(raw) for raw in state.values["approval_requests"]
    ]
    follow_ups = [r for r in requests if r.kind == ActionKind.SEND_RECRUITER_MESSAGE]
    assert len(follow_ups) == 1
    assert str(wait.checkpoint_id) in state.values["pending_actions"][0]["payload"][
        "checkpoint_id"
    ]
    # Nothing was sent: the only action ever executed is the original submission.
    assert [
        intent.kind for intent, _ in deployment.ports["action_executor"].executed
    ] == [ActionKind.SUBMIT_APPLICATION]

    # And the run picked up where the thread was, rather than starting over.
    assert deployment.searches == searches_before
    assert state.values["application"]["application_id"] == str(fakes.APPLICATION_ID)


async def test_a_fired_wait_is_not_fired_again_by_the_next_sweep(tmp_path):
    """Claimed once. A second sweep has nothing to claim.

    The guarded close is what makes this true for two monitors running at the
    same time; here it is checked in the shape a single monitor on a schedule
    actually runs -- sweep, sweep again a minute later.
    """
    evaluator = fakes.ScriptedConditionEvaluator(
        {ConditionKind.RECRUITER_RESPONSE_RECEIVED: False}
    )
    deployment = await applied(tmp_path, evaluator=evaluator)

    first = await deployment.monitor.sweep(now=AT_TRIGGER)
    second = await deployment.monitor.sweep(now=AT_TRIGGER + timedelta(minutes=1))

    assert len(first.fired) == 1
    assert second.considered == 0
    assert deployment.events.count(JobSearchEventType.FOLLOW_UP_TRIGGERED.value) == 1


async def test_a_wait_whose_thread_is_parked_on_an_approval_is_deferred(tmp_path):
    """A second request stacked on an unanswered one helps nobody.

    The thread is parked at the follow-up's own approval interrupt from the
    first sweep. A second wait coming due while a human still has not answered
    is held back rather than fired -- and because it stays pending, its own
    expiry is what eventually settles it if the answer never comes.
    """
    evaluator = fakes.ScriptedConditionEvaluator(
        {ConditionKind.RECRUITER_RESPONSE_RECEIVED: False}
    )
    deployment = await applied(tmp_path, evaluator=evaluator)
    await deployment.monitor.sweep(now=AT_TRIGGER)
    assert (await deployment.runner.inspect(deployment.thread)).next == (APPROVAL_CHECKPOINT,)

    # A second wait on the same thread, due now.
    second = deployment.store.schedule(
        deployment.store.get(
            deployment.store.history_for_application(fakes.APPLICATION_ID)[0].checkpoint_id
        ).model_copy(
            update={
                "checkpoint_id": uuid4(),
                "kind": FollowUpKind.AWAITING_CANDIDATE_REPLY,
                "dedupe_key": "follow_up:second",
                "status": PendingCheckpointStatus.PENDING,
                "closed_at": None,
                "closed_reason": None,
            }
        )
    )

    report = await deployment.monitor.sweep(now=AT_TRIGGER + timedelta(minutes=5))

    assert report.deferred == (second.checkpoint_id,)
    assert report.fired == ()
    assert deployment.store.get(second.checkpoint_id).status == PendingCheckpointStatus.PENDING


# --- Acceptance 3: never swept in time, expired ------------------------------


async def test_a_wait_nobody_swept_in_time_is_marked_expired_not_left_pending(tmp_path):
    """A monitor that was down through the whole window writes the wait off.

    The third acceptance criterion. Nothing swept this checkpoint at its
    trigger -- the process was not running -- and by the time one does, the
    follow-up is stale. It is recorded as `expired`, with the reason, rather
    than sent late or left `pending` for a sweep that will now never help it.
    """
    evaluator = fakes.ScriptedConditionEvaluator(
        {ConditionKind.RECRUITER_RESPONSE_RECEIVED: False}
    )
    deployment = await applied(tmp_path, evaluator=evaluator)
    wait = deployment.store.open_for_application(fakes.APPLICATION_ID)[0]

    # The first sweep in a fortnight, long past the expiry.
    report = await deployment.monitor.sweep(now=wait.expires_at + timedelta(days=7))

    assert report.expired == (wait.checkpoint_id,)
    assert report.fired == ()

    closed = deployment.store.get(wait.checkpoint_id)
    assert closed.status == PendingCheckpointStatus.EXPIRED
    assert "without firing" in closed.closed_reason
    # Nothing was drafted and nothing was sent.
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value not in deployment.events
    assert (await deployment.runner.inspect(deployment.thread)).next == ()
    # And it is off the pending list for good, rather than coming back next sweep.
    assert deployment.store.open_for_application(fakes.APPLICATION_ID) == []
    assert (await deployment.monitor.sweep(now=wait.expires_at + timedelta(days=8))).considered == 0


async def test_an_expired_wait_whose_condition_came_true_is_resolved_not_expired(tmp_path):
    """Two silences, opposite meanings, and the table must not confuse them.

    A checkpoint that is both past its expiry and no longer needed is recorded
    as `resolved`: the follow-up was not missed, it stopped being wanted.
    Reporting it as `expired` would send an operator looking for an outage.
    """
    evaluator = fakes.ScriptedConditionEvaluator(
        {ConditionKind.RECRUITER_RESPONSE_RECEIVED: True}
    )
    deployment = await applied(tmp_path, evaluator=evaluator)
    wait = deployment.store.open_for_application(fakes.APPLICATION_ID)[0]

    report = await deployment.monitor.sweep(now=wait.expires_at + timedelta(days=7))

    assert report.resolved == (wait.checkpoint_id,)
    assert report.expired == ()
    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.RESOLVED


# --- Degenerate wiring --------------------------------------------------------


async def test_a_wait_naming_a_thread_that_never_ran_is_cancelled_not_fired(tmp_path):
    """There is no state to follow up on, so firing would draft from nothing."""
    evaluator = fakes.ScriptedConditionEvaluator()
    deployment = await applied(tmp_path, evaluator=evaluator)
    orphan_thread = deployment.registry.register(
        thread_id=job_search_thread_id(fakes.USER_ID, "never-ran"),
        workflow_name="job_search",
        user_id=fakes.USER_ID,
    )
    orphan = deployment.store.schedule(
        deployment.store.open_for_application(fakes.APPLICATION_ID)[0].model_copy(
            update={
                "checkpoint_id": uuid4(),
                "thread_id": orphan_thread.thread_id,
                "dedupe_key": "follow_up:orphan",
            }
        )
    )

    report = await deployment.monitor.sweep(now=AT_TRIGGER)

    assert orphan.checkpoint_id in report.cancelled
    closed = deployment.store.get(orphan.checkpoint_id)
    assert closed.status == PendingCheckpointStatus.CANCELLED
    assert "no stored state" in closed.closed_reason


def test_a_graph_with_no_scheduler_still_runs_but_stores_no_wait(tmp_path):
    """The port is optional, and leaving it out turns the durable half off cleanly.

    Worth pinning: a deployment with no monitor process must not write waits
    nothing will ever sweep, because a row that reads as "a follow-up is coming"
    and never produces one is worse than no row.
    """
    graph = JobSearchGraph(
        profile_store=fakes.FakeProfileStore(),
        providers=[fakes.FakeProvider()],
        scorer=fakes.FakeScorer(),
        evidence_checker=fakes.FakeEvidenceChecker(),
        packet_builder=fakes.FakePacketBuilder(),
        approval_gate=fakes.FakeApprovalGate(),
        action_executor=fakes.FakeActionExecutor(),
        application_store=fakes.FakeApplicationStore(),
        event_emitter=fakes.FakeEventEmitter(),
        recruiter_inbox=fakes.FakeRecruiterInbox(),
        recruiter_classifier=fakes.FakeRecruiterClassifier(),
    )

    assert graph.build() is not None


async def test_a_fired_checkpoint_does_not_reroute_the_next_ordinary_run(tmp_path):
    """`fired_checkpoints` is consumed, not left lying in the thread's state.

    Thread state outlives the run that wrote it, and this one channel decides
    which way a run enters the graph. A checkpoint left in state after it was
    acted on would silently divert the candidate's next real job search into the
    follow-up path -- a search that returns no postings, for no visible reason.
    """
    deployment = Deployment(tmp_path, evaluator=fakes.ScriptedConditionEvaluator())
    # A gate that answers everything, so the follow-up run completes instead of
    # parking: this test is about what the run leaves behind, not about the
    # interrupt.
    deployment.graph = JobSearchGraph(
        **{**deployment.ports, "approval_gate": fakes.FakeApprovalGate()}
    ).build()
    deployment.runner.graph = deployment.graph
    await deployment.apply()

    wait = deployment.store.open_for_application(fakes.APPLICATION_ID)[0]
    await deployment.runner.start(
        deployment.thread, {"fired_checkpoints": [wait.model_dump(mode="json")]}
    )
    assert (await deployment.runner.inspect(deployment.thread)).values[
        "fired_checkpoints"
    ] == []

    searches_before = deployment.searches
    await deployment.runner.start(
        deployment.thread, {"user_id": str(fakes.USER_ID), "prepare_application": False}
    )

    # Discovery ran, which it would not have done had the run been rerouted.
    assert deployment.searches == searches_before + 1
