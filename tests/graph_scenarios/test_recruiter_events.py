"""Inbound recruiter mail through the compiled graph, against a real database.

The acceptance criteria this file exists for:

1. **A duplicate email event produces exactly one application transition and
   one `communication_events` row.** `test_a_duplicate_email_...` delivers the
   same message in two separate runs and counts rows.
2. **An INTERVIEW_INVITE email creates the right event, transitions the
   application, and emits an event consumable by the calendar step.**
   `test_an_interview_invite_...` reads the outbox row back and parses it the
   way a calendar step would.

Everything outward-facing is a fake except the part under test: the
`SqlRecruiterEventStore` is real, over SQLite, and the extractor is the
deterministic rule-based one, so nothing here depends on a model. Per-node
contracts are in `tests/unit/test_job_search_nodes.py`.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from personalos.domain.job_search import (
    ActionKind,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    JobSearchEventType,
    RecruiterMessage,
)
from personalos.domain.models import APPLICATION_STATUS_CHANGED_EVENT, ApplicationStatus
from personalos.domain.recruiter_events import CommitmentActor, InterviewInviteReceived
from personalos.domain.workflow import recruiter_inbox_thread_id
from personalos.graphs.job_search import JobSearchGraph
from personalos.models.recruiter_events import RuleBasedRecruiterEventExtractor
from personalos.persistence.models import (
    ApplicationModel,
    Base,
    CommitmentModel,
    CommunicationEventModel,
    EventLogModel,
    OutboxEventModel,
    UserModel,
)
from personalos.persistence.recruiter_events import SqlRecruiterEventStore
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository
from tests.fixtures import job_search_fakes as fakes

#: A Wednesday.
T0 = datetime(2026, 10, 7, 9, 0, 0)


@pytest.fixture
def factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    engine.dispose()


def applied_application(factory, *, company="Northwind", reference="REQ-48213", user_id=None):
    """An APPLIED application. Returns `(user_id, application_id)`."""
    session = factory()
    try:
        if user_id is None:
            user = UserModel(email=f"candidate-{uuid4()}@example.com")
            session.add(user)
            session.commit()
            user_id = user.id
        posting = JobPostingRepository(session).create(
            source="greenhouse",
            source_job_id=reference,
            title="Senior Backend Engineer",
            company=company,
            description_hash="a" * 64,
            dedupe_key=f"posting:{uuid4()}",
        )
        applications = ApplicationRepository(session)
        application = applications.create(job_posting_id=posting.id, user_id=user_id)
        for status in (
            ApplicationStatus.SAVED,
            ApplicationStatus.PREPARING,
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.APPLIED,
        ):
            applications.update_status(application.id, status, now=T0 - timedelta(days=3))
        return user_id, application.id
    finally:
        session.close()


def build(factory, **overrides):
    """A compiled graph whose inbound-mail ports are the real store."""
    store = SqlRecruiterEventStore(factory, clock=lambda: T0)
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [fakes.FakeProvider()],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": fakes.NoStandingApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
        "recruiter_event_extractor": RuleBasedRecruiterEventExtractor(),
        "application_directory": store,
        "recruiter_event_recorder": store,
        "clock": lambda: T0,
    }
    ports.update(overrides)
    return JobSearchGraph(**ports).build(), ports


def invite(message_id: str = "<invite-1@northwind.test>", **overrides) -> RecruiterMessage:
    fields = {
        "provider_message_id": message_id,
        "received_at": T0,
        "subject": "Interview for REQ-48213 at Northwind",
        "from_address": "priya@northwind.test",
        "body": (
            "Hi Sam,\n\nWe would like to schedule a technical interview with the team. "
            "Please send your availability by Friday.\n\nPriya"
        ),
        "thread_id": "thread-northwind-1",
    }
    fields.update(overrides)
    return RecruiterMessage(**fields)


def run_input(user_id, *messages, **extra) -> dict:
    return {
        "user_id": str(user_id),
        "inbound_messages": [m.model_dump(mode="json") for m in messages],
        **extra,
    }


def thread(user_id) -> dict:
    return {"configurable": {"thread_id": recruiter_inbox_thread_id(user_id)}}


def rows(factory, model, **filters):
    session = factory()
    try:
        query = session.query(model)
        for column, value in filters.items():
            query = query.filter(getattr(model, column) == value)
        return [row.to_dict() for row in query.all()]
    finally:
        session.close()


def status_changes(factory, application_id):
    return [
        (row["payload_json"]["from"], row["payload_json"]["to"])
        for row in rows(
            factory,
            EventLogModel,
            aggregate_id=application_id,
            event_type=APPLICATION_STATUS_CHANGED_EVENT,
        )
        if row["payload_json"]["actor"] == "recruiter_event"
    ]


def status_of(factory, application_id) -> str:
    return rows(factory, ApplicationModel, id=application_id)[0]["status"]


async def test_an_interview_invite_is_recorded_transitions_and_reaches_the_calendar(factory):
    user_id, application_id = applied_application(factory)
    graph, ports = build(factory)

    final = await graph.ainvoke(run_input(user_id, invite()), thread(user_id))

    # The right communication event, on the right application.
    (row,) = rows(factory, CommunicationEventModel)
    assert row["classification"] == "interview_invite"
    assert row["application_id"] == str(application_id)
    assert row["provider_message_id"] == "<invite-1@northwind.test>"
    assert row["metadata_json"]["extraction_source"] == "fallback"

    # The application moved, through the lifecycle, with the event that says why.
    assert status_of(factory, application_id) == "interviewing"
    assert status_changes(factory, application_id) == [("applied", "interviewing")]

    # The event a calendar step consumes is in the outbox, and parses.
    by_type = {row["type"]: row for row in rows(factory, OutboxEventModel)}
    assert set(by_type) == {
        JobSearchEventType.RECRUITER_RESPONSE_RECORDED.value,
        JobSearchEventType.INTERVIEW_INVITE_RECEIVED.value,
    }
    calendar_input = InterviewInviteReceived.model_validate(
        by_type[JobSearchEventType.INTERVIEW_INVITE_RECEIVED.value]["payload_json"]
    )
    assert calendar_input.application_id == application_id
    assert calendar_input.thread_id == "thread-northwind-1"
    assert calendar_input.from_address == "priya@northwind.test"
    (commitment,) = calendar_input.commitments
    assert commitment.actor == CommitmentActor.USER
    assert commitment.due_at == datetime(2026, 10, 9, 17, 0)  # "by Friday"
    assert commitment.source_message_id == "<invite-1@northwind.test>"

    # The same commitment is a row of its own.
    (stored,) = rows(factory, CommitmentModel)
    assert stored["communication_event_id"] == row["id"]
    assert stored["due_at"] == "2026-10-09T17:00:00"

    # State carries the same story, and the input channel was consumed.
    assert [e["type"] for e in final["emitted_events"]] == [
        JobSearchEventType.RECRUITER_RESPONSE_RECORDED.value,
        JobSearchEventType.INTERVIEW_INVITE_RECEIVED.value,
    ]
    assert final["inbound_messages"] == []
    assert ports["event_emitter"].events == []


async def test_a_duplicate_email_produces_one_transition_and_one_row(factory):
    user_id, application_id = applied_application(factory)
    graph, _ports = build(factory, approval_gate=fakes.FakeApprovalGate(ApprovalVerdict.REJECTED))

    await graph.ainvoke(run_input(user_id, invite()), thread(user_id))
    # Redelivered: once more on the same thread, and once on a fresh one, as a
    # second worker picking up the same webhook would.
    second = await graph.ainvoke(run_input(user_id, invite()), thread(user_id))
    third = await graph.ainvoke(
        run_input(user_id, invite()), {"configurable": {"thread_id": f"t-{uuid4()}"}}
    )

    assert len(rows(factory, CommunicationEventModel)) == 1
    assert status_changes(factory, application_id) == [("applied", "interviewing")]
    assert len(rows(factory, CommitmentModel)) == 1
    invites = rows(
        factory, OutboxEventModel, type=JobSearchEventType.INTERVIEW_INVITE_RECEIVED.value
    )
    assert len(invites) == 1
    for rerun in (second, third):
        assert [r["created"] for r in rerun["recruiter_event_records"]] == [False]
        assert rerun["pending_actions"] == []


async def test_a_reply_arriving_on_a_known_thread_is_matched_by_the_thread_alone(factory):
    user_id, application_id = applied_application(factory)
    graph, _ports = build(factory, approval_gate=fakes.FakeApprovalGate(ApprovalVerdict.REJECTED))
    await graph.ainvoke(run_input(user_id, invite()), thread(user_id))

    rejection = invite(
        "<later@mail.test>",
        subject="Re: next steps",
        from_address="noreply@ats-mail.test",
        body="Unfortunately we will not be moving forward with your application.",
    )
    final = await graph.ainvoke(run_input(user_id, rejection), thread(user_id))

    assert final["recruiter_events"][0]["correlation"]["signals"] == ["thread"]
    assert status_of(factory, application_id) == "rejected"


async def test_a_low_confidence_match_goes_to_review_and_transitions_nothing(factory):
    user_id, application_id = applied_application(factory)
    graph, ports = build(factory)
    # Names the company and nothing else: no thread, no requisition, a webmail sender.
    vague = invite(
        "<vague@gmail.test>",
        subject="Northwind interview",
        from_address="someone@gmail.test",
        thread_id=None,
    )

    final = await graph.ainvoke(run_input(user_id, vague), thread(user_id))

    assert rows(factory, CommunicationEventModel) == []
    assert rows(factory, OutboxEventModel) == []
    assert status_of(factory, application_id) == "applied"
    (review,) = final["recruiter_event_reviews"]
    assert review["application_id"] is None
    assert [c["application_id"] for c in review["candidates"]] == [str(application_id)]
    assert ports["event_emitter"].types() == [
        JobSearchEventType.RECRUITER_EVENT_REVIEW_REQUIRED.value
    ]
    assert final.get("__interrupt__") is None

    # The reviewer says which application it was; now it is acted on.
    confirmed = await graph.ainvoke(
        run_input(
            user_id,
            vague,
            confirmed_correlations={vague.provider_message_id: str(application_id)},
        ),
        thread(user_id),
    )
    assert confirmed["recruiter_events"][0]["correlation"]["signals"] == ["reviewer_confirmed"]
    assert status_of(factory, application_id) == "interviewing"


async def test_another_users_application_is_never_a_candidate(factory):
    _owner, application_id = applied_application(factory)
    stranger, _ = applied_application(factory, company="Globex", reference="REQ-90000")
    graph, _ports = build(factory)

    final = await graph.ainvoke(run_input(stranger, invite()), thread(stranger))

    assert final["recruiter_event_records"] == []
    assert status_of(factory, application_id) == "applied"


async def test_a_reply_is_drafted_automatically_but_sent_only_on_approval(factory):
    user_id, application_id = applied_application(factory)
    graph, ports = build(factory)
    cfg = thread(user_id)

    parked = await graph.ainvoke(run_input(user_id, invite()), cfg)

    # Drafted without anyone asking: the text exists and is what the reviewer sees.
    (request,) = [ApprovalRequest.model_validate(r) for r in parked["approval_requests"]]
    assert request.kind == ActionKind.SEND_RECRUITER_MESSAGE
    (pending,) = parked["pending_actions"]
    assert pending["payload"]["recipient"] == "priya@northwind.test"
    assert pending["payload"]["subject"] == "Re: Interview for REQ-48213 at Northwind"
    assert "October 09" in pending["payload"]["body"]
    # Not sent: the run is parked at the interrupt and the executor is untouched.
    assert parked.get("__interrupt__")
    assert ports["action_executor"].executed == []
    # Recording did not wait on the reply.
    assert status_of(factory, application_id) == "interviewing"

    decision = ApprovalDecision(
        action_id=request.action_id,
        action_fingerprint=request.action_hash,
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
        request_id=request.request_id,
    )
    resumed = await graph.ainvoke(Command(resume=[decision.model_dump(mode="json")]), cfg)

    assert resumed.get("__interrupt__") is None
    assert [intent.kind for intent, _decision in ports["action_executor"].executed] == [
        ActionKind.SEND_RECRUITER_MESSAGE
    ]
    # Sending the reply recorded nothing new.
    assert len(rows(factory, CommunicationEventModel)) == 1
