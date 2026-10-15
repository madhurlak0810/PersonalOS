"""Tests for the v1 application board and transition endpoints.

The acceptance criteria this file exists for:

1. **An invalid transition is rejected with the valid next states** -- a 422
   whose body lists them, with nothing written --
   `test_an_invalid_transition_is_rejected_with_the_valid_next_states`.
2. **The board reflects a transition immediately after it is applied** --
   `test_the_board_reflects_a_transition_immediately`.

Each request goes to a fresh app built by `create_app()` over a database file,
as in `test_approval_routes`, so nothing a request sees can have been left in
memory by the one before it.
"""

from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes.applications import (
    ApplicationServices,
    build_application_services,
    get_application_services,
)
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from personalos.domain.applications import (
    APPLICATION_TRANSITION_REQUESTED_EVENT,
    RequestedTransition,
    TransitionEffect,
    transition_action_intent,
)
from personalos.domain.checkpoints import (
    CheckpointCondition,
    ConditionKind,
    PendingCheckpoint,
)
from personalos.domain.job_search import FollowUpKind, JobMatch
from personalos.domain.models import APPLICATION_STATUS_CHANGED_EVENT, ApplicationStatus
from personalos.domain.workflow import application_transition_thread_id
from personalos.persistence.application_board import ApplicationBoardStore
from personalos.persistence.application_lifecycle import ApplicationLifecycleStore
from personalos.persistence.models import (
    ApplicationStatusViewModel,
    ArtifactVersionModel,
    Base,
    CommitmentModel,
    CommunicationEventModel,
    EventLogModel,
    OutboxEventModel,
    PolicyDecisionModel,
    UserModel,
)
from personalos.persistence.pending_checkpoints import PendingCheckpointStore
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository
from personalos.policy.engine import PolicyEngine
from personalos.policy.intents import Decision
from personalos.policy.permissions import PermissionClass

S = ApplicationStatus
T0 = datetime(2026, 10, 1, 9, 0, 0)
ACTOR = "maddy"


# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "applications.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def session_factory(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    finally:
        engine.dispose()


def api_process(db_path: Path, *, policy: PolicyEngine | None = None) -> TestClient:
    """A fresh API instance over the database: new app, new engine, new services."""
    engine = create_engine(f"sqlite:///{db_path}")
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def services() -> ApplicationServices:
        built = build_application_services(factory)
        if policy is None:
            return built
        return ApplicationServices(
            board=built.board,
            lifecycle=built.lifecycle,
            policy=policy,
            workflows=built.workflows,
        )

    app = create_app()
    app.dependency_overrides[get_application_services] = services
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    return TestClient(app)


def new_application(
    factory,
    *,
    company: str = "Acme",
    title: str = "Staff Engineer",
    created_at: datetime = T0,
    user_id: UUID | None = None,
) -> UUID:
    """A DISCOVERED application for a fresh posting."""
    session = factory()
    try:
        if user_id is None:
            user = UserModel(email=f"candidate-{uuid4()}@example.com")
            session.add(user)
            session.commit()
            user_id = user.id
        posting = JobPostingRepository(session).create(
            source="greenhouse",
            title=title,
            company=company,
            description_hash="a" * 64,
            dedupe_key=f"{company}:{title}:{uuid4()}".lower(),
            url=f"https://jobs.example.com/{uuid4()}",
        )
        application = ApplicationRepository(session).create(
            job_posting_id=posting.id, user_id=user_id
        )
        application.created_at = created_at
        application.last_activity_at = created_at
        session.commit()
        return application.id
    finally:
        session.close()


def move(factory, application_id: UUID, *statuses: ApplicationStatus) -> None:
    lifecycle = ApplicationLifecycleStore(factory)
    for status in statuses:
        lifecycle.transition(application_id, status, actor="setup")


def add_artifact(factory, application_id: UUID, artifact_type: str = "resume") -> UUID:
    session = factory()
    try:
        row = ArtifactVersionModel(
            application_id=application_id,
            artifact_type=artifact_type,
            version=1,
            content=f"tailored {artifact_type}",
            evidence=[{"type": "resume_section", "ref": "exp-1"}],
            generated_by="test",
        )
        session.add(row)
        session.commit()
        return row.id
    finally:
        session.close()


def events(factory, application_id: UUID, event_type: str) -> list[dict]:
    session = factory()
    try:
        rows = (
            session.query(EventLogModel)
            .filter(
                EventLogModel.aggregate_id == application_id,
                EventLogModel.event_type == event_type,
            )
            .order_by(EventLogModel.occurred_at)
            .all()
        )
        return [row.payload_json for row in rows]
    finally:
        session.close()


def column(board: dict, status: ApplicationStatus) -> dict:
    return next(c for c in board["columns"] if c["status"] == status.value)


def card_ids(board: dict, status: ApplicationStatus) -> list[str]:
    return [card["application_id"] for card in column(board, status)["applications"]]


def a_match(score: float = 0.82) -> JobMatch:
    return JobMatch(
        dedupe_key="acme:staff-engineer",
        score=score,
        weighted_score=score,
        recommendation="APPLY",
        matched_requirements=[
            {
                "requirement": "python",
                "evidence_id": "exp-1",
                "evidence_type": "resume",
                "strength": "strong",
            }
        ],
        missing_requirements=[{"requirement": "kubernetes", "severity": "minor"}],
    )


# --- Acceptance criteria --------------------------------------------------------


def test_an_invalid_transition_is_rejected_with_the_valid_next_states(db_path, session_factory):
    application_id = new_application(session_factory)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "offer"},
        headers={"X-Actor-Id": ACTOR},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation"
    assert body["allowed_next_states"] == ["saved", "skipped"]
    assert "context_id" in body
    # Refused before anything was written.
    assert events(session_factory, application_id, APPLICATION_STATUS_CHANGED_EVENT) == []
    board = api_process(db_path).get("/v1/applications").json()
    assert card_ids(board, S.DISCOVERED) == [str(application_id)]


def test_the_board_reflects_a_transition_immediately(db_path, session_factory):
    application_id = new_application(session_factory)
    before = api_process(db_path).get("/v1/applications").json()
    assert card_ids(before, S.DISCOVERED) == [str(application_id)]
    assert card_ids(before, S.SAVED) == []

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "saved", "reason": "looks promising"},
        headers={"X-Actor-Id": ACTOR},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "applied"
    assert (body["from_status"], body["to_status"]) == ("discovered", "saved")
    assert body["application"]["status"] == "saved"

    # A different API process, straight after: the projection already moved.
    after = api_process(db_path).get("/v1/applications").json()
    assert card_ids(after, S.DISCOVERED) == []
    assert card_ids(after, S.SAVED) == [str(application_id)]
    assert column(after, S.SAVED)["total"] == 1
    assert column(after, S.SAVED)["applications"][0]["last_event_id"] == body["event_id"]

    # Through the state machine, with its immutable event, attributed to the caller.
    (event,) = events(session_factory, application_id, APPLICATION_STATUS_CHANGED_EVENT)
    assert event["from"] == "discovered" and event["to"] == "saved"
    assert event["actor"] == ACTOR and event["reason"] == "looks promising"
    session = session_factory()
    try:
        view = session.get(ApplicationStatusViewModel, application_id)
        assert view.status == "saved" and str(view.last_event_id) == body["event_id"]
    finally:
        session.close()


# --- Transitions ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ((S.SAVED,), S.SKIPPED),
        ((S.SAVED, S.PREPARING, S.READY_TO_APPLY), S.APPLIED),
        ((S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED), S.WITHDRAWN),
    ],
)
def test_bookkeeping_transitions_are_applied_at_once(db_path, session_factory, path, target):
    application_id = new_application(session_factory)
    move(session_factory, application_id, *path)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition", json={"to": target.value}
    )

    assert response.status_code == 200
    assert response.json()["application"]["status"] == target.value
    assert (
        events(session_factory, application_id, APPLICATION_STATUS_CHANGED_EVENT)[-1]["actor"]
        == "api"
    )


def test_stalled_cannot_be_requested(db_path, session_factory):
    application_id = new_application(session_factory)
    move(session_factory, application_id, S.SAVED)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition", json={"to": "stalled"}
    )

    assert response.status_code == 422
    assert "stalled" not in response.json()["allowed_next_states"]
    assert response.json()["allowed_next_states"] == ["preparing", "skipped"]


def test_a_terminal_application_lists_no_next_states(db_path, session_factory):
    application_id = new_application(session_factory)
    move(session_factory, application_id, S.SKIPPED)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition", json={"to": "saved"}
    )

    assert response.status_code == 422
    assert response.json()["allowed_next_states"] == []


def test_an_unknown_status_is_a_validation_error(db_path, session_factory):
    application_id = new_application(session_factory)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition", json={"to": "hired"}
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "validation"


def test_transitioning_an_unknown_application_is_a_404(db_path):
    response = api_process(db_path).post(
        f"/v1/applications/{uuid4()}/transition", json={"to": "saved"}
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"


def test_an_effect_that_does_not_fit_the_target_is_refused(db_path, session_factory):
    application_id = new_application(session_factory)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "skipped", "effect": "submit_application"},
    )

    assert response.status_code == 422
    assert "submit_application" in response.json()["message"]


# --- Transitions with an external effect ----------------------------------------


def _ready(factory) -> UUID:
    application_id = new_application(factory)
    move(factory, application_id, S.SAVED, S.PREPARING, S.READY_TO_APPLY)
    add_artifact(factory, application_id, "resume")
    add_artifact(factory, application_id, "cover_letter")
    return application_id


def test_an_effectful_transition_is_queued_for_approval_not_applied(db_path, session_factory):
    application_id = _ready(session_factory)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "applied", "effect": "submit_application"},
        headers={"X-Actor-Id": ACTOR},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["outcome"] == "queued"
    assert body["policy_decision"] == Decision.REQUIRE_APPROVAL.value
    # Not applied: the board still shows where it was.
    assert body["application"]["status"] == "ready_to_apply"
    board = api_process(db_path).get("/v1/applications").json()
    assert card_ids(board, S.READY_TO_APPLY) == [str(application_id)]
    assert card_ids(board, S.APPLIED) == []
    assert (
        events(session_factory, application_id, APPLICATION_STATUS_CHANGED_EVENT)[-1]["to"]
        == "ready_to_apply"
    )

    # The request is an immutable event of its own.
    (requested,) = events(session_factory, application_id, APPLICATION_TRANSITION_REQUESTED_EVENT)
    transition = RequestedTransition.model_validate(requested)
    assert transition.to_status is S.APPLIED
    assert transition.effect is TransitionEffect.SUBMIT_APPLICATION
    assert transition.requested_by == ACTOR
    assert {item["artifact_type"] for item in transition.attachments} == {
        "resume",
        "cover_letter",
    }
    assert transition.message == "tailored cover_letter"

    session = session_factory()
    try:
        # The policy engine saw it, and recorded its verdict on the submission tool.
        (decision,) = session.query(PolicyDecisionModel).all()
        assert decision.tool == "jobs.submit_application"
        assert decision.decision == "require_approval"
        assert decision.principal == ACTOR
        assert decision.args_hash == transition_action_intent(transition).fingerprint()
        # And the run was handed to the worker on the application's own thread.
        (outbox,) = session.query(OutboxEventModel).all()
        command = outbox.payload_json
        assert command["kind"] == "start"
        assert command["thread_id"] == body["thread_id"]
        assert command["graph_input"]["requested_transitions"][0]["request_id"] == str(
            transition.request_id
        )
    finally:
        session.close()
    detail = api_process(db_path).get(f"/v1/applications/{application_id}").json()
    assert body["thread_id"] == application_transition_thread_id(
        detail["application"]["user_id"], application_id
    )


def test_a_second_effectful_transition_while_one_is_in_flight_is_a_409(db_path, session_factory):
    application_id = _ready(session_factory)
    client = api_process(db_path)
    first = client.post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "applied", "effect": "submit_application"},
    )
    assert first.status_code == 202

    second = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "applied", "effect": "submit_application"},
    )

    assert second.status_code == 409
    assert len(events(session_factory, application_id, APPLICATION_TRANSITION_REQUESTED_EVENT)) == 1


def test_a_policy_denial_is_a_403_and_queues_nothing(db_path, session_factory):
    application_id = new_application(session_factory)
    move(session_factory, application_id, S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED)
    strict = PolicyEngine(class_outcomes={PermissionClass.WRITE_EXTERNAL: Decision.DENY})

    response = api_process(db_path, policy=strict).post(
        f"/v1/applications/{application_id}/transition",
        json={
            "to": "withdrawn",
            "effect": "notify_recruiter",
            "recipient": "recruiter@acme.example",
            "message": "I'm withdrawing my application.",
        },
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "policy_denied"
    assert events(session_factory, application_id, APPLICATION_TRANSITION_REQUESTED_EVENT) == []
    session = session_factory()
    try:
        assert session.query(OutboxEventModel).count() == 0
    finally:
        session.close()


def test_a_submission_with_nothing_to_submit_is_refused(db_path, session_factory):
    application_id = new_application(session_factory)
    move(session_factory, application_id, S.SAVED, S.PREPARING, S.READY_TO_APPLY)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "applied", "effect": "submit_application"},
    )

    assert response.status_code == 422
    assert "nothing to submit" in response.json()["message"]


def test_notifying_the_recruiter_needs_a_recipient_and_message(db_path, session_factory):
    application_id = new_application(session_factory)
    move(session_factory, application_id, S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED)

    response = api_process(db_path).post(
        f"/v1/applications/{application_id}/transition",
        json={"to": "withdrawn", "effect": "notify_recruiter"},
    )

    assert response.status_code == 422


# --- Board ----------------------------------------------------------------------


def test_the_board_has_every_column_in_lifecycle_order(db_path, session_factory):
    new_application(session_factory)

    board = api_process(db_path).get("/v1/applications").json()

    assert [c["status"] for c in board["columns"]] == [status.value for status in S]
    assert board["total"] == 1 and board["next_offset"] is None


def test_the_board_filters_by_state_company_and_date(db_path, session_factory):
    acme_old = new_application(session_factory, company="Acme", created_at=T0)
    acme_new = new_application(
        session_factory, company="Acme Robotics", created_at=T0 + timedelta(days=10)
    )
    globex = new_application(session_factory, company="Globex", created_at=T0 + timedelta(days=10))
    move(session_factory, acme_new, S.SAVED)
    client = api_process(db_path)

    by_state = client.get("/v1/applications", params={"state": ["saved", "skipped"]}).json()
    assert [c["status"] for c in by_state["columns"]] == ["saved", "skipped"]
    assert card_ids(by_state, S.SAVED) == [str(acme_new)]
    assert by_state["total"] == 1

    by_company = client.get("/v1/applications", params={"company": "acme"}).json()
    assert by_company["total"] == 2
    assert card_ids(by_company, S.DISCOVERED) == [str(acme_old)]

    by_date = client.get(
        "/v1/applications",
        params={
            "created_from": (T0 + timedelta(days=5)).isoformat(),
            "created_to": (T0 + timedelta(days=15)).isoformat(),
        },
    ).json()
    assert {card for status in S for card in card_ids(by_date, status)} == {
        str(acme_new),
        str(globex),
    }


def test_an_inverted_date_range_is_a_validation_error(db_path):
    response = api_process(db_path).get(
        "/v1/applications",
        params={"created_from": T0.isoformat(), "created_to": (T0 - timedelta(days=1)).isoformat()},
    )

    assert response.status_code == 422


def test_the_board_paginates_in_column_order(db_path, session_factory):
    ids = [new_application(session_factory, created_at=T0 + timedelta(hours=n)) for n in range(5)]
    move(session_factory, ids[0], S.SAVED)
    client = api_process(db_path)

    first = client.get("/v1/applications", params={"limit": 3}).json()
    assert first["total"] == 5 and first["next_offset"] == 3
    # Column totals count every match, not just this page's.
    assert column(first, S.DISCOVERED)["total"] == 4
    assert len(column(first, S.DISCOVERED)["applications"]) == 3
    assert column(first, S.SAVED)["applications"] == []

    second = client.get("/v1/applications", params={"limit": 3, "offset": 3}).json()
    assert second["next_offset"] is None
    assert len(column(second, S.DISCOVERED)["applications"]) == 1
    assert card_ids(second, S.SAVED) == [str(ids[0])]

    seen = [card for page in (first, second) for status in S for card in card_ids(page, status)]
    assert sorted(seen) == sorted(map(str, ids))


def test_the_page_size_is_bounded(db_path):
    assert api_process(db_path).get("/v1/applications", params={"limit": 1000}).status_code == 422


# --- Detail ---------------------------------------------------------------------


def test_the_detail_carries_fit_evidence_artifacts_recruiter_events_and_checkpoints(
    db_path, session_factory
):
    application_id = new_application(session_factory)
    move(session_factory, application_id, S.SAVED, S.PREPARING, S.READY_TO_APPLY, S.APPLIED)
    board = ApplicationBoardStore(session_factory)
    board.record_match(application_id, a_match(0.6), recorded_by="old-run")
    board.record_match(application_id, a_match(0.82), recorded_by="job_search")
    artifact_id = add_artifact(session_factory, application_id)
    session = session_factory()
    try:
        message = CommunicationEventModel(
            application_id=application_id,
            classification="interview_invite",
            provider_message_id="msg-1",
            occurred_at=T0 + timedelta(days=3),
        )
        session.add(message)
        session.flush()
        session.add(
            CommitmentModel(
                communication_event_id=message.id,
                application_id=application_id,
                actor="user",
                action="send availability",
                due_at=T0 + timedelta(days=5),
                confidence=0.9,
                source_message_id="msg-1",
            )
        )
        session.commit()
    finally:
        session.close()
    checkpoint = PendingCheckpointStore(session_factory).schedule(
        PendingCheckpoint(
            application_id=application_id,
            kind=FollowUpKind.NO_RESPONSE,
            condition=CheckpointCondition(
                kind=ConditionKind.RECRUITER_RESPONSE_RECEIVED, subject_id=application_id, since=T0
            ),
            reason="follow up if no reply",
            thread_id="job_search:test",
            created_at=T0,
            trigger_at=T0 + timedelta(days=7),
            expires_at=T0 + timedelta(days=14),
            dedupe_key=f"follow-up:{application_id}",
        )
    )

    response = api_process(db_path).get(f"/v1/applications/{application_id}")

    assert response.status_code == 200
    detail = response.json()
    assert detail["application"]["status"] == "applied"
    assert detail["application"]["company"] == "Acme"
    assert detail["fit_score"] == pytest.approx(0.82)
    assert detail["match"]["matched_requirements"][0]["evidence_id"] == "exp-1"
    assert detail["match"]["missing_requirements"][0]["requirement"] == "kubernetes"
    assert detail["allowed_next_states"] == [
        "response",
        "interviewing",
        "rejected",
        "withdrawn",
        "follow_up_pending",
    ]
    (artifact,) = detail["artifacts"]
    assert artifact["artifact_version_id"] == str(artifact_id)
    assert "content" not in artifact
    (recruiter_event,) = detail["recruiter_events"]
    assert recruiter_event["classification"] == "interview_invite"
    assert recruiter_event["commitments"][0]["action"] == "send availability"
    (open_wait,) = detail["pending_checkpoints"]
    assert open_wait["checkpoint_id"] == str(checkpoint.checkpoint_id)
    assert [(h["from_status"], h["to_status"]) for h in detail["history"]] == [
        ("discovered", "saved"),
        ("saved", "preparing"),
        ("preparing", "ready_to_apply"),
        ("ready_to_apply", "applied"),
    ]
    assert detail["pending_approvals"] == []


def test_a_detail_with_no_match_has_no_fit_score(db_path, session_factory):
    application_id = new_application(session_factory)

    detail = api_process(db_path).get(f"/v1/applications/{application_id}").json()

    assert detail["fit_score"] is None and detail["match"] is None
    assert detail["allowed_next_states"] == ["saved", "skipped"]


def test_an_unknown_application_detail_is_a_404(db_path):
    response = api_process(db_path).get(f"/v1/applications/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"
