"""A conditional follow-up that cancels itself, with nothing scripted.

`test_pending_checkpoints.py` proves the wait's mechanics against an evaluator
whose answers are written into the test. This file closes the loop: the
condition is answered by `SqlCheckpointConditionEvaluator` from rows the rest
of the system wrote, so "the recruiter replied" is a real recruiter message
going through the real inbound-mail path, on a different thread, days after
the wait was scheduled.

The acceptance criterion:

**A follow-up whose condition resolves (recruiter replied) before `trigger_at`
closes without drafting anything.**
`test_a_recruiter_reply_before_the_trigger_closes_the_follow_up_silently`.

Its counterpart, `test_with_no_reply_the_same_follow_up_is_drafted`, is what
makes the first one mean something: same deployment, same sweep, no reply --
and the follow-up is drafted and parked for approval.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.worker.checkpoint_monitor import PendingCheckpointMonitor
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import build_checkpoint_condition_evaluator
from personalos.domain.checkpoints import ConditionKind, PendingCheckpointStatus
from personalos.domain.job_search import (
    ActionKind,
    ApprovalDecision,
    ApprovalVerdict,
    FollowUpKind,
    JobSearchEventType,
    RecruiterMessage,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.workflow import job_search_thread_id, recruiter_inbox_thread_id
from personalos.graphs.job_search import APPROVAL_CHECKPOINT, JobSearchGraph
from personalos.models.recruiter_events import RuleBasedRecruiterEventExtractor
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import WorkflowLeaseStore
from personalos.persistence.models import Base, CommunicationEventModel, UserModel
from personalos.persistence.pending_checkpoints import (
    PendingCheckpointStore,
    StorePendingCheckpointScheduler,
)
from personalos.persistence.recruiter_events import SqlRecruiterEventStore
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository
from tests.fixtures import job_search_fakes as fakes

#: The moment of applying. The no-response follow-up is due seven days on.
APPLIED_AT = datetime(2026, 9, 29, 12, 0, 0)
TRIGGER_AT = APPLIED_AT + timedelta(days=7)
REPLIED_AT = APPLIED_AT + timedelta(days=2)


class SubmissionsOnlyGate:
    """Standing approval for the application itself; nothing on file for messages."""

    def __init__(self):
        self.reviewed = []

    async def review(self, intent) -> ApprovalDecision:
        self.reviewed.append(intent)
        approved = intent.kind == ActionKind.SUBMIT_APPLICATION
        return ApprovalDecision(
            action_id=intent.action_id,
            action_fingerprint=intent.fingerprint(),
            verdict=ApprovalVerdict.APPROVED if approved else ApprovalVerdict.PENDING,
            decided_by="reviewer@example.test",
        )


class Deployment:
    """Graph, inbox, store and monitor over one database, and nothing in memory."""

    def __init__(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path / 'follow_ups.db'}")
        Base.metadata.create_all(engine)
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.now = APPLIED_AT
        self.user_id, self.application_id = self._application_on_file()

        self.registry = WorkflowThreadRegistry(self.factory)
        self.store = PendingCheckpointStore(self.factory, clock=self.clock)
        self.recruiter_events = SqlRecruiterEventStore(self.factory, clock=self.clock)
        self.gate = SubmissionsOnlyGate()
        self.emitter = fakes.FakeEventEmitter()
        self.executor = fakes.FakeActionExecutor()
        self.graph = JobSearchGraph(
            profile_store=fakes.FakeProfileStore(),
            providers=[fakes.FakeProvider()],
            scorer=fakes.FakeScorer(),
            evidence_checker=fakes.FakeEvidenceChecker(),
            packet_builder=fakes.FakePacketBuilder(),
            approval_gate=self.gate,
            action_executor=self.executor,
            # The application the search "creates" is the one on file, so the
            # wait it schedules and the reply recorded later are about one row.
            application_store=fakes.FakeApplicationStore(self.application_id),
            event_emitter=self.emitter,
            recruiter_inbox=fakes.FakeRecruiterInbox(),
            recruiter_classifier=fakes.FakeRecruiterClassifier(),
            checkpoint_scheduler=StorePendingCheckpointScheduler(self.store),
            recruiter_event_extractor=RuleBasedRecruiterEventExtractor(),
            application_directory=self.recruiter_events,
            recruiter_event_recorder=self.recruiter_events,
            checkpointer=SqlAlchemyCheckpointSaver(self.factory, self.registry),
            clock=self.clock,
        ).build()
        self.runner = DurableWorkflowRunner(
            self.graph,
            registry=self.registry,
            leases=WorkflowLeaseStore(self.factory),
            owner="worker-1",
        )
        self.monitor = PendingCheckpointMonitor(
            store=self.store,
            # The real one: it knows nothing this test told it.
            evaluator=build_checkpoint_condition_evaluator(self.factory),
            runner=self.runner,
            registry=self.registry,
            clock=self.clock,
        )
        self.search_thread = self.registry.register(
            thread_id=job_search_thread_id(self.user_id, "follow-ups"),
            workflow_name="job_search",
            user_id=self.user_id,
        )
        self.inbox_thread = self.registry.register(
            thread_id=recruiter_inbox_thread_id(self.user_id),
            workflow_name="job_search",
            user_id=self.user_id,
        )

    def clock(self) -> datetime:
        return self.now

    def _application_on_file(self):
        session = self.factory()
        try:
            user = UserModel(email=f"candidate-{uuid4()}@example.com")
            session.add(user)
            session.commit()
            posting = JobPostingRepository(session).create(
                source="greenhouse",
                source_job_id="REQ-48213",
                title="Senior Backend Engineer",
                company="Northwind",
                description_hash="a" * 64,
                dedupe_key=f"posting:{uuid4()}",
            )
            applications = ApplicationRepository(session)
            application = applications.create(job_posting_id=posting.id, user_id=user.id)
            for status in (
                ApplicationStatus.SAVED,
                ApplicationStatus.PREPARING,
                ApplicationStatus.READY_TO_APPLY,
                ApplicationStatus.APPLIED,
            ):
                applications.update_status(application.id, status, now=APPLIED_AT)
            return user.id, application.id
        finally:
            session.close()

    async def apply(self):
        """Search, apply, and schedule the seven-day no-response follow-up."""
        await self.runner.start(
            self.search_thread, {"user_id": str(self.user_id), "prepare_application": True}
        )
        (wait,) = self.store.open_for_application(self.application_id)
        assert wait.kind == FollowUpKind.NO_RESPONSE
        assert wait.condition.kind == ConditionKind.RECRUITER_RESPONSE_RECEIVED
        assert wait.trigger_at == TRIGGER_AT
        return wait

    async def recruiter_replies(self, at: datetime):
        """A recruiter's reply arrives and is processed on the inbox thread."""
        self.now = at
        message = RecruiterMessage(
            provider_message_id="<reply-1@northwind.test>",
            received_at=at,
            subject="Re: your application for REQ-48213 at Northwind",
            from_address="priya@northwind.test",
            body=(
                "Hi Sam,\n\nThanks for applying. We have received your application and "
                "the team is reviewing it now.\n\nPriya"
            ),
            thread_id="thread-northwind-1",
        )
        await self.runner.start(
            self.inbox_thread,
            {
                "user_id": str(self.user_id),
                "inbound_messages": [message.model_dump(mode="json")],
            },
        )

    def replies_on_file(self) -> int:
        session = self.factory()
        try:
            return (
                session.query(CommunicationEventModel)
                .filter(CommunicationEventModel.application_id == self.application_id)
                .count()
            )
        finally:
            session.close()


async def test_a_recruiter_reply_before_the_trigger_closes_the_follow_up_silently(tmp_path):
    """Day 0: apply. Day 2: the recruiter replies. Day 7: nothing is drafted.

    Nothing told the wait about the reply. It was scheduled with a question --
    "has a recruiter response landed since?" -- and the sweep on day seven asks
    it of the database, where the reply has been sitting for five days.
    """
    deployment = Deployment(tmp_path)
    wait = await deployment.apply()

    await deployment.recruiter_replies(REPLIED_AT)
    assert deployment.replies_on_file() == 1
    # The reply did not close the wait on arrival; the re-evaluation does.
    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.PENDING
    events_before = deployment.emitter.types()
    reviewed_before = len(deployment.gate.reviewed)

    report = await deployment.monitor.sweep(now=TRIGGER_AT)

    assert report.resolved == (wait.checkpoint_id,)
    assert report.fired == ()
    closed = deployment.store.get(wait.checkpoint_id)
    assert closed.status == PendingCheckpointStatus.RESOLVED
    assert "condition met" in closed.closed_reason

    # Silently: no draft, no approval request, no event, no run.
    assert deployment.emitter.types() == events_before
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value not in deployment.emitter.types()
    assert len(deployment.gate.reviewed) == reviewed_before
    assert [intent.kind for intent, _ in deployment.executor.executed] == [
        ActionKind.SUBMIT_APPLICATION
    ]
    state = await deployment.runner.inspect(deployment.search_thread)
    assert state.next == ()
    assert not any(
        raw["kind"] == ActionKind.SEND_RECRUITER_MESSAGE.value
        for raw in state.values["pending_actions"]
    )


async def test_with_no_reply_the_same_follow_up_is_drafted(tmp_path):
    """The control: same sweep, no reply on file, and the follow-up is drafted."""
    deployment = Deployment(tmp_path)
    wait = await deployment.apply()

    report = await deployment.monitor.sweep(now=TRIGGER_AT)

    assert report.fired == (wait.checkpoint_id,)
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value in deployment.emitter.types()
    state = await deployment.runner.inspect(deployment.search_thread)
    # Drafted, and parked: still nothing sent without a person.
    assert state.next == (APPROVAL_CHECKPOINT,)
    assert state.values["pending_actions"][0]["kind"] == ActionKind.SEND_RECRUITER_MESSAGE.value


async def test_a_reply_from_before_the_wait_existed_does_not_cancel_it(tmp_path):
    """`condition.since` bounds the question: an old message is not an answer."""
    deployment = Deployment(tmp_path)
    await deployment.recruiter_replies(APPLIED_AT - timedelta(days=1))
    deployment.now = APPLIED_AT
    wait = await deployment.apply()

    report = await deployment.monitor.sweep(now=TRIGGER_AT)

    assert report.fired == (wait.checkpoint_id,)
