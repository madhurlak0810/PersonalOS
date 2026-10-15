"""Scenario tests for the job-search graph's requested-transition entry point.

A board move that asks for an external effect is queued by
`POST /v1/applications/{id}/transition` as a run with `requested_transitions`
in its input. These tests drive that run through the compiled graph, with the
lifecycle store on a real database behind the `ApplicationTransitioner` port,
and check the one property the entry point exists for: the status changes only
after the effect has been approved *and* has run.
"""

from uuid import UUID, uuid4

import pytest
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from personalos.domain.applications import (
    RequestedTransition,
    TransitionEffect,
    transition_action_intent,
)
from personalos.domain.job_search import (
    ActionKind,
    ApprovalRequest,
    ApprovalVerdict,
    JobSearchContractError,
)
from personalos.domain.models import APPLICATION_STATUS_CHANGED_EVENT, ApplicationStatus
from personalos.graphs.job_search import (
    APPROVAL_CHECKPOINT,
    STAGE_LIFECYCLE_TRANSITION,
    JobSearchGraph,
)
from personalos.persistence.application_lifecycle import (
    ApplicationLifecycleStore,
    StoreApplicationTransitioner,
)
from personalos.persistence.models import Base, UserModel
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository
from tests.fixtures import job_search_fakes as fakes

S = ApplicationStatus


@pytest.fixture
def lifecycle():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    try:
        yield ApplicationLifecycleStore(
            sessionmaker(autocommit=False, autoflush=False, bind=engine)
        )
    finally:
        engine.dispose()


def applied_application(lifecycle: ApplicationLifecycleStore) -> UUID:
    session = lifecycle.session_factory()
    try:
        user = UserModel(email=f"candidate-{uuid4()}@example.com")
        session.add(user)
        session.commit()
        posting = JobPostingRepository(session).create(
            source="greenhouse",
            title="Staff Engineer",
            company="Acme",
            description_hash="a" * 64,
            dedupe_key=f"acme:staff:{uuid4()}",
        )
        application_id = (
            ApplicationRepository(session).create(job_posting_id=posting.id, user_id=user.id).id
        )
    finally:
        session.close()
    for status in (S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED):
        lifecycle.transition(application_id, status, actor="setup")
    return application_id


def withdrawal(application_id: UUID) -> RequestedTransition:
    return RequestedTransition(
        application_id=application_id,
        from_status=S.APPLIED,
        to_status=S.WITHDRAWN,
        effect=TransitionEffect.NOTIFY_RECRUITER,
        reason="took another offer",
        requested_by="maddy",
        company="Acme",
        title="Staff Engineer",
        recipient="recruiter@acme.example",
        message="Thank you -- I'm withdrawing my application.",
    )


def build(lifecycle, *, transitioner=True, executor=None):
    executor = executor or fakes.FakeActionExecutor()
    graph = JobSearchGraph(
        profile_store=fakes.FakeProfileStore(),
        providers=[fakes.FakeProvider()],
        scorer=fakes.FakeScorer(),
        evidence_checker=fakes.FakeEvidenceChecker(),
        packet_builder=fakes.FakePacketBuilder(),
        approval_gate=fakes.NoStandingApprovalGate(),
        action_executor=executor,
        application_store=fakes.FakeApplicationStore(),
        event_emitter=fakes.FakeEventEmitter(),
        application_transitioner=(
            StoreApplicationTransitioner(lifecycle) if transitioner else None
        ),
    ).build()
    return graph, executor


async def start(graph, request: RequestedTransition):
    config = {"configurable": {"thread_id": f"job_search:{uuid4()}"}}
    await graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "requested_transitions": [request.model_dump(mode="json")]},
        config=config,
    )
    return config


async def answer(graph, config, verdict: ApprovalVerdict):
    snapshot = await graph.aget_state(config)
    (request,) = [ApprovalRequest.model_validate(r) for r in snapshot.values["approval_requests"]]
    return await graph.ainvoke(
        Command(resume=[fakes.approval_decision(request, verdict).model_dump(mode="json")]),
        config=config,
    )


async def test_the_effect_parks_at_approval_and_the_status_waits(lifecycle):
    application_id = applied_application(lifecycle)
    graph, executor = build(lifecycle)
    request = withdrawal(application_id)

    config = await start(graph, request)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == (APPROVAL_CHECKPOINT,)
    assert snapshot.values["approval_stage"] == STAGE_LIFECYCLE_TRANSITION
    (action,) = snapshot.values["pending_actions"]
    assert action["kind"] == ActionKind.SEND_RECRUITER_MESSAGE.value
    assert action["action_id"] == str(request.request_id)
    (approval,) = snapshot.values["approval_requests"]
    assert approval["action_hash"] == transition_action_intent(request).fingerprint()
    assert executor.executed == []
    assert lifecycle.get(application_id).status is S.APPLIED


async def test_an_approved_effect_runs_and_then_the_status_is_written(lifecycle):
    application_id = applied_application(lifecycle)
    graph, executor = build(lifecycle)
    request = withdrawal(application_id)
    config = await start(graph, request)

    final = await answer(graph, config, ApprovalVerdict.APPROVED)

    executed, _ = executor.executed[0]
    assert executed.payload["recipient"] == "recruiter@acme.example"
    assert lifecycle.get(application_id).status is S.WITHDRAWN
    event = lifecycle.history(application_id)[-1]
    assert event["event_type"] == APPLICATION_STATUS_CHANGED_EVENT
    assert event["payload_json"]["to"] == "withdrawn"
    assert event["payload_json"]["actor"] == "maddy"
    (outcome,) = final["transition_outcomes"]
    assert outcome["applied"] is True
    assert final["requested_transitions"] == []


async def test_a_rejected_effect_leaves_the_status_alone(lifecycle):
    application_id = applied_application(lifecycle)
    graph, executor = build(lifecycle)
    config = await start(graph, withdrawal(application_id))

    final = await answer(graph, config, ApprovalVerdict.REJECTED)

    assert executor.executed == []
    assert lifecycle.get(application_id).status is S.APPLIED
    (outcome,) = final["transition_outcomes"]
    assert outcome["applied"] is False
    assert "denied" in outcome["detail"]


async def test_a_failed_effect_leaves_the_status_alone(lifecycle):
    application_id = applied_application(lifecycle)
    graph, _ = build(lifecycle, executor=fakes.FakeActionExecutor(ok=False))
    config = await start(graph, withdrawal(application_id))

    final = await answer(graph, config, ApprovalVerdict.APPROVED)

    assert lifecycle.get(application_id).status is S.APPLIED
    assert final["transition_outcomes"][0]["detail"].startswith("effect failed")


async def test_a_move_made_illegal_while_parked_is_refused_not_forced(lifecycle):
    application_id = applied_application(lifecycle)
    graph, _ = build(lifecycle)
    config = await start(graph, withdrawal(application_id))
    # While the run waits for a reviewer, the application is rejected.
    lifecycle.transition(application_id, S.REJECTED, actor="recruiter_events")

    final = await answer(graph, config, ApprovalVerdict.APPROVED)

    assert lifecycle.get(application_id).status is S.REJECTED
    (outcome,) = final["transition_outcomes"]
    assert outcome["applied"] is False
    assert "status not changed" in outcome["detail"]


async def test_without_a_transitioner_nothing_is_proposed(lifecycle):
    application_id = applied_application(lifecycle)
    graph, executor = build(lifecycle, transitioner=False)

    with pytest.raises(JobSearchContractError):
        await start(graph, withdrawal(application_id))
    assert executor.executed == []
