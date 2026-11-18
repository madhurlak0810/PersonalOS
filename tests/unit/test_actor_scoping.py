"""Phase L: actor-scoped authorization, with two actors.

Everything here runs against two seeded actors, `alice` and `bob`, sharing one
database. Alice owns a workflow parked at an approval, an application with a
tailored document, a parked outbox event, a credential reference and audit
records; Bob owns a workflow of his own. The acceptance criteria:

1. **Bob gets 404 for every one of Alice's objects, and cannot list them** --
   `test_bob_gets_404_for_every_one_of_alices_objects`,
   `test_bob_cannot_list_alices_objects`.
2. **An actor id in a request body changes nothing** --
   `test_an_actor_id_in_a_body_is_refused_and_stores_nothing`,
   `test_every_request_body_refuses_identity_fields`.
3. **Deciding needs `approve`; the worker's system identity is denied, and
   audited** -- `test_a_token_without_approve_gets_403_on_the_decision`,
   `test_the_worker_system_identity_is_denied_and_audited`,
   `test_the_approval_queue_refuses_a_system_principal_when_called_directly`.
4. **A stream ticket opens only its workflow, and only until it expires** --
   `test_a_ticket_for_one_workflow_is_rejected_for_another`,
   `test_a_ticket_is_rejected_after_it_expires`.
5. **Repositories scope by themselves, outside any route** --
   `test_scoped_stores_return_nothing_for_another_actor`.
6. **Every denial is audited, and no audit row carries token material** --
   `test_every_denial_is_audited_without_token_material`.
"""

import asyncio
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from apps.api.auth import ApiAuth, StreamTicketSigner, get_audit_session_factory
from apps.api.main import create_app
from apps.api.routes.applications import build_application_services, get_application_services
from apps.api.routes.audit import build_audit_store, get_audit_store
from apps.api.routes.events import build_progress_log, get_progress_log
from apps.api.routes.failures import build_failure_services, get_failure_services
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    build_application_board_store,
    build_application_lifecycle_store,
    build_audit_trail_store,
    build_durable_checkpointer,
    build_event_failure_store,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_progress_log,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.domain.auth import (
    HUMAN_SCOPES,
    WORKER_ACTOR,
    AuthMethod,
    Principal,
    Scope,
    SystemIdentityCannotApprove,
    system_principal,
)
from personalos.domain.credentials import CredentialKind, CredentialRef
from personalos.domain.errors import Forbidden
from personalos.domain.job_search import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    RefusalReason,
    authorize_execution,
)
from personalos.domain.models import ApplicationStatus, OutboxEventStatus
from personalos.domain.ownership import SYSTEM_ACCESS, ActorScope
from personalos.domain.workflow import (
    JOB_SEARCH_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
)
from personalos.persistence.models import (
    ApprovalModel,
    ArtifactVersionModel,
    AuditEventModel,
    OutboxEventModel,
    UserModel,
    WorkflowRunModel,
)
from personalos.persistence.ownership import AUDIT_DENIED_PREFIX
from personalos.persistence.repositories import (
    ApplicationRepository,
    CredentialRepository,
    JobPostingRepository,
)
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.auth import bearer, install_test_auth, owner_id
from tests.fixtures.durable_workflow import session_factory as file_session_factory

ALICE = "alice"
BOB = "bob"
ALICE_ACCESS = ActorScope.of(ALICE)
BOB_ACCESS = ActorScope.of(BOB)

#: The prefix of every bearer token the test verifier accepts: token material.
TOKEN_MARKER = "test-token:"


# --- A world with two actors ------------------------------------------------------


class _JobState(TypedDict, total=False):
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]


def _approval_checkpoint(state: _JobState) -> dict[str, Any]:
    decisions = interrupt(
        {"workflow": "job_search", "stage": "submission", "requests": state["approval_requests"]}
    )
    return {"approvals": decisions}


class World:
    """Two actors' data in one database, and an API over it."""

    def __init__(self, db_path: Path, *, clock: list[float] | None = None):
        self.db_path = db_path
        self.factory = file_session_factory(db_path)
        self.registry = build_workflow_thread_registry(self.factory)
        self.checkpointer = build_durable_checkpointer(self.factory, self.registry)
        self.queue = build_workflow_command_queue(self.factory)
        self.clock = clock if clock is not None else [datetime.utcnow().timestamp()]
        graph = StateGraph(_JobState)
        graph.add_node("approval_checkpoint", _approval_checkpoint)
        graph.add_edge(START, "approval_checkpoint")
        graph.add_edge("approval_checkpoint", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)

        for actor in (ALICE, BOB):
            self._user(actor)
        self.alice_workflow, self.alice_request = self._parked_workflow(ALICE)
        self.alice_second_workflow, _ = self._parked_workflow(ALICE)
        self.bob_workflow, self.bob_request = self._parked_workflow(BOB)
        self.alice_application = self._application(ALICE)
        self.alice_audit_event = self._audit_event(ALICE, self.alice_workflow)
        self.alice_event_failure = self._parked_event(self.alice_workflow)
        self.alice_credential = self._credential(ALICE)

    # --- Seeding -------------------------------------------------------------

    def _user(self, actor: str) -> None:
        session = self.factory()
        try:
            session.add(UserModel(id=owner_id(actor), display_name=actor))
            session.commit()
        finally:
            session.close()

    def _parked_workflow(self, actor: str) -> tuple[UUID, ApprovalRequest]:
        intent = fakes.submit_intent(fakes.posting(company=f"{actor}-corp"))
        request = ApprovalRequest.for_intent(intent, now=datetime.utcnow())
        workflow_id = self.registry.create_workflow(JOB_SEARCH_WORKFLOW)
        thread = register_job_search_thread(
            user_id=owner_id(actor),
            registry=self.registry,
            workflow_id=workflow_id,
            search_key=str(workflow_id),
            actor_id=actor,
        )
        self.queue.enqueue(
            WorkflowCommand(
                command_id=uuid4(),
                kind=WorkflowCommandKind.START,
                workflow_id=workflow_id,
                thread_id=thread.thread_id,
                actor_id=actor,
                correlation_id=uuid4(),
                graph_input={"approval_requests": [request.model_dump(mode="json")]},
            )
        )
        runner = DurableWorkflowRunner(
            self.graph, registry=self.registry, leases=build_workflow_lease_store(self.factory)
        )
        assert asyncio.run(process_next(self.queue, {JOB_SEARCH_THREAD_NAMESPACE: runner}))
        return workflow_id, request

    def _application(self, actor: str) -> UUID:
        session = self.factory()
        try:
            posting = JobPostingRepository(session).create(
                source="greenhouse",
                title="Staff Engineer",
                company=f"{actor}-corp",
                description_hash="a" * 64,
                dedupe_key=f"{actor}:{uuid4()}",
                url=f"https://jobs.example.com/{uuid4()}",
            )
            application = ApplicationRepository(session).create(
                job_posting_id=posting.id, user_id=owner_id(actor)
            )
            session.add(
                ArtifactVersionModel(
                    application_id=application.id,
                    artifact_type="cover_letter",
                    version=1,
                    content=f"{actor}'s private cover letter",
                    evidence=[{"type": "resume_section", "ref": "exp-1"}],
                    generated_by="test",
                )
            )
            session.commit()
            return application.id
        finally:
            session.close()

    def _audit_event(self, actor: str, workflow_id: UUID) -> UUID:
        session = self.factory()
        try:
            event = AuditEventModel(
                actor=actor,
                workflow_id=workflow_id,
                action="files.overwrite_document",
                target_ref=f"{actor}-notes.md",
                result="success",
            )
            session.add(event)
            session.commit()
            return event.id
        finally:
            session.close()

    def _parked_event(self, workflow_id: UUID) -> UUID:
        session = self.factory()
        try:
            row = OutboxEventModel(
                type="test.happened",
                dedupe_key=f"test:{uuid4()}",
                payload_json={},
                status=OutboxEventStatus.DEAD_LETTER.value,
                workflow_id=workflow_id,
                last_error="gave up",
                failed_at=datetime.utcnow(),
            )
            session.add(row)
            session.commit()
            return row.id
        finally:
            session.close()

    def _credential(self, actor: str) -> UUID:
        session = self.factory()
        try:
            row = CredentialRepository(session).create(
                ref=CredentialRef("google", f"{actor}@example.com"),
                kind=CredentialKind.OAUTH_REFRESH_TOKEN,
                user_id=owner_id(actor),
            )
            return row.id
        finally:
            session.close()

    # --- The API -------------------------------------------------------------

    def signer(self, ttl_seconds: float = 60) -> StreamTicketSigner:
        return StreamTicketSigner(
            lambda: b"test stream ticket key, 32 bytes",
            ttl_seconds=ttl_seconds,
            clock=lambda: self.clock[0],
        )

    def api(self, actor: str | None = None, *, scopes: str | None = None) -> TestClient:
        """An API instance over this database, as `actor` (anonymous when `None`)."""
        factory = file_session_factory(self.db_path, create=False)
        app = install_test_auth(create_app())
        app.state.api_auth = ApiAuth(verifier=app.state.api_auth.verifier, tickets=self.signer())
        overrides = {
            get_workflow_services: lambda: build_workflow_services(factory),
            get_application_services: lambda: build_application_services(factory),
            get_audit_store: lambda: build_audit_store(factory),
            get_failure_services: lambda: build_failure_services(factory),
            get_progress_log: lambda: build_progress_log(factory),
            get_audit_session_factory: lambda: factory,
        }
        app.dependency_overrides.update(overrides)
        headers = bearer(actor, scopes=scopes) if actor is not None else {}
        return TestClient(app, headers=headers)

    # --- Reading back --------------------------------------------------------

    def rows(self, model) -> list[Any]:
        session = self.factory()
        try:
            rows = session.query(model).all()
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()

    def denials(self, actor: str | None = None) -> list[AuditEventModel]:
        return [
            row
            for row in self.rows(AuditEventModel)
            if row.action.startswith(AUDIT_DENIED_PREFIX) and (actor is None or row.actor == actor)
        ]


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path / "scoping.db")


def _alices_get_paths(world: World) -> list[str]:
    workflow, application = world.alice_workflow, world.alice_application
    return [
        f"/v1/workflows/{workflow}",
        f"/v1/workflows/{workflow}/audit",
        f"/v1/workflows/{workflow}/failures",
        f"/v1/workflows/{workflow}/events?follow=false",
        f"/v1/applications/{application}",
    ]


# --- 1. Bob cannot see Alice's objects -----------------------------------------------


def test_alice_can_see_her_own_objects(world):
    """The control: everything Bob is refused below, Alice is shown."""
    alice = world.api(ALICE)
    for path in _alices_get_paths(world):
        assert alice.get(path).status_code == 200, path
    approvals = {a["id"] for a in alice.get("/v1/approvals").json()["approvals"]}
    assert str(world.alice_request.request_id) in approvals
    assert str(world.bob_request.request_id) not in approvals
    assert alice.get("/v1/applications").json()["total"] == 1


def test_bob_gets_404_for_every_one_of_alices_objects(world):
    bob = world.api(BOB)

    for path in _alices_get_paths(world):
        response = bob.get(path)
        assert response.status_code == 404, f"{path}: {response.status_code}"
        assert response.json()["error_code"] == "not_found"

    # Her approval, through either way of deciding it, and her workflow's
    # stream ticket: 404, as for an id that does not exist.
    decide = bob.post(
        f"/v1/approvals/{world.alice_request.request_id}/decision",
        json={"approved": True, "action_hash": world.alice_request.action_hash},
    )
    assert decide.status_code == 404
    resume = bob.post(
        f"/v1/workflows/{world.alice_workflow}/resume",
        json={
            "decisions": [
                {
                    "action_id": str(world.alice_request.action_id),
                    "action_fingerprint": world.alice_request.action_hash,
                    "verdict": "approved",
                }
            ]
        },
    )
    assert resume.status_code == 404
    ticket = bob.post("/v1/auth/stream-tickets", json={"workflow_id": str(world.alice_workflow)})
    assert ticket.status_code == 404
    move = bob.post(f"/v1/applications/{world.alice_application}/transition", json={"to": "saved"})
    assert move.status_code == 404
    retry = bob.post(f"/v1/failures/events/{world.alice_event_failure}/retry")
    assert retry.status_code == 404

    # A 404 for Alice's object is the same 404 as for nothing at all.
    unknown = bob.get(f"/v1/workflows/{uuid4()}")
    foreign = bob.get(f"/v1/workflows/{world.alice_workflow}")
    assert unknown.json()["message"].replace(str(unknown.url).rsplit("/", 1)[-1], "X") == (
        foreign.json()["message"].replace(str(world.alice_workflow), "X")
    )

    # Nothing Bob did was recorded as a decision.
    assert world.rows(ApprovalModel) == []


def test_bob_cannot_list_alices_objects(world):
    bob = world.api(BOB)

    [approval] = bob.get("/v1/approvals").json()["approvals"]
    assert approval["id"] == str(world.bob_request.request_id)

    board = bob.get("/v1/applications").json()
    assert board["total"] == 0
    assert all(column["applications"] == [] for column in board["columns"])

    search = bob.get("/v1/audit").json()
    assert str(world.alice_audit_event) not in {e["audit_event_id"] for e in search["entries"]}
    assert all(e.get("workflow_id") != str(world.alice_workflow) for e in search["entries"])
    # Naming her workflow or application as a filter narrows to nothing.
    assert (
        bob.get("/v1/audit", params={"workflow_id": str(world.alice_workflow)}).json()["total"] == 0
    )
    by_application = bob.get("/v1/audit", params={"application_id": str(world.alice_application)})
    assert by_application.json()["total"] == 0
    # Paging past the first page reaches nothing of hers either.
    assert bob.get("/v1/audit", params={"offset": 50}).json()["entries"] == []

    assert bob.get("/v1/failures/events").json()["failures"] == []

    # Her event cursor does not open Bob's own stream onto her events.
    alice_events = world.api(ALICE).get(f"/v1/workflows/{world.alice_workflow}/events?follow=false")
    alice_ids = [line[4:] for line in alice_events.text.splitlines() if line.startswith("id: ")]
    assert alice_ids
    cursor = bob.get(
        f"/v1/workflows/{world.bob_workflow}/events?follow=false",
        headers={"Last-Event-ID": alice_ids[0]},
    )
    assert cursor.status_code == 422


# --- 2. Identity in a body -----------------------------------------------------------


def test_an_actor_id_in_a_body_is_refused_and_stores_nothing(world):
    bob = world.api(BOB)
    runs_before = len(world.rows(WorkflowRunModel))

    for claim in (
        {"actor_id": ALICE},
        {"user_id": str(owner_id(ALICE))},
        {"actor_id": ALICE, "user_id": str(owner_id(ALICE))},
    ):
        response = bob.post("/v1/chat", json={"message": "find me jobs", **claim})
        assert response.status_code == 422, claim
    assert len(world.rows(WorkflowRunModel)) == runs_before

    decision = bob.post(
        f"/v1/approvals/{world.bob_request.request_id}/decision",
        json={
            "approved": True,
            "action_hash": world.bob_request.action_hash,
            "decided_by": ALICE,
        },
    )
    assert decision.status_code == 422
    event = bob.post(
        f"/v1/workflows/{world.bob_workflow}/resume", json={"event": {"actor_id": ALICE}}
    )
    assert event.status_code == 422
    ticket = bob.post(
        "/v1/auth/stream-tickets",
        json={"workflow_id": str(world.bob_workflow), "actor_id": ALICE},
    )
    assert ticket.status_code == 422

    # Without the claim, what is stored is the caller's, whoever it named.
    started = bob.post("/v1/chat", json={"message": "find me jobs"})
    assert started.status_code == 202, started.text
    [run] = [r for r in world.rows(WorkflowRunModel) if r.thread_id == started.json()["thread_id"]]
    assert run.user_id == owner_id(BOB)
    assert run.actor_id == BOB
    assert world.api(ALICE).get(f"/v1/workflows/{started.json()['workflow_id']}").status_code == (
        404
    )


def test_every_request_body_refuses_identity_fields():
    """No body model has an identity field, and every one refuses unknown fields."""
    import importlib
    import pkgutil

    import apps.api.routes as route_package
    from apps.api.routes.workflows import IDENTITY_FIELDS

    checked = 0
    for module_info in pkgutil.iter_modules(route_package.__path__):
        module = importlib.import_module(f"{route_package.__name__}.{module_info.name}")
        routers = [getattr(module, name, None) for name in ("router", "callback_router")]
        for route in (r for router in routers if router is not None for r in router.routes):
            body = getattr(route, "body_field", None)
            if body is None:
                continue
            for candidate in _models(body.field_info.annotation):
                for model in _nested(candidate):
                    assert not IDENTITY_FIELDS & set(model.model_fields), (
                        f"{route.path}: {model.__name__} has an identity field"
                    )
                    assert model.model_config.get("extra") == "forbid", (
                        f"{route.path}: {model.__name__} accepts unknown fields"
                    )
                    checked += 1
    assert checked >= 10


def _nested(model: type[BaseModel]) -> list[type[BaseModel]]:
    """`model` and every model its fields hold (a resume's decisions, say)."""
    found = [model]
    for field in model.model_fields.values():
        for inner in _models(field.annotation):
            found.extend(_nested(inner))
    return found


def _models(annotation) -> list[type[BaseModel]]:
    """The pydantic models an annotation names (unwrapping `X | None`)."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return [annotation]
    return [m for arg in getattr(annotation, "__args__", ()) for m in _models(arg)]


def test_no_query_parameter_names_an_identity():
    """`user_id`, `actor_id` and their like are not accepted as query parameters either."""
    from apps.api.routes.workflows import IDENTITY_FIELDS

    for path, operations in create_app().openapi()["paths"].items():
        for operation in operations.values():
            names = {p["name"] for p in operation.get("parameters", []) if p["in"] == "query"}
            # `actor` on `/v1/audit` filters the caller's own trail; it is not an identity.
            assert not (IDENTITY_FIELDS - {"actor"}) & names, path


# --- 3. Approve scope and system identities --------------------------------------------


def test_a_token_without_approve_gets_403_on_the_decision(world):
    reader = world.api(BOB, scopes="read,write")

    response = reader.post(
        f"/v1/approvals/{world.bob_request.request_id}/decision",
        json={"approved": True, "action_hash": world.bob_request.action_hash},
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "forbidden"
    assert world.rows(ApprovalModel) == []
    # Resuming with decisions is deciding too.
    resume = reader.post(
        f"/v1/workflows/{world.bob_workflow}/resume",
        json={
            "decisions": [
                {
                    "action_id": str(world.bob_request.action_id),
                    "action_fingerprint": world.bob_request.action_hash,
                    "verdict": "approved",
                }
            ]
        },
    )
    assert resume.status_code == 403
    # And `approve` alone is enough to decide; reading still needs `read`.
    approver = world.api(BOB, scopes="approve")
    decided = approver.post(
        f"/v1/approvals/{world.bob_request.request_id}/decision",
        json={"approved": True, "action_hash": world.bob_request.action_hash},
    )
    assert decided.status_code == 200, decided.text
    assert approver.get("/v1/approvals").status_code == 403


def test_a_read_only_token_cannot_write(world):
    reader = world.api(BOB, scopes="read")
    assert reader.get(f"/v1/workflows/{world.bob_workflow}").status_code == 200
    assert reader.post("/v1/chat", json={"message": "find me jobs"}).status_code == 403


def test_the_worker_system_identity_is_denied_and_audited(world):
    worker = world.api(WORKER_ACTOR)

    response = worker.post(
        f"/v1/approvals/{world.bob_request.request_id}/decision",
        json={"approved": True, "action_hash": world.bob_request.action_hash},
    )

    assert response.status_code == 403
    assert world.rows(ApprovalModel) == []
    [denial] = world.denials(WORKER_ACTOR)
    assert denial.action == f"{AUDIT_DENIED_PREFIX}system_identity"
    assert denial.result == "failure"
    assert "/v1/approvals/" in denial.target_ref


def test_the_approval_queue_refuses_a_system_principal_when_called_directly(world):
    """The gate is in the queue itself, not only in the route in front of it."""
    queue = build_workflow_services(world.factory).approvals
    scheduler = system_principal("system:scheduler")

    with pytest.raises(Forbidden):
        queue.decide(
            world.bob_request.request_id,
            approved=True,
            action_hash=world.bob_request.action_hash,
            principal=scheduler,
        )
    without_approve = Principal(BOB, AuthMethod.LOCAL_TOKEN, scopes=frozenset({Scope.WRITE}))
    with pytest.raises(Forbidden):
        queue.decide(
            world.bob_request.request_id,
            approved=True,
            action_hash=world.bob_request.action_hash,
            principal=without_approve,
        )

    assert world.rows(ApprovalModel) == []
    assert [d.action for d in world.denials()] == [
        f"{AUDIT_DENIED_PREFIX}system_identity",
        f"{AUDIT_DENIED_PREFIX}missing_scope",
    ]


def test_a_system_identity_can_never_hold_approve():
    with pytest.raises(SystemIdentityCannotApprove):
        Principal(WORKER_ACTOR, AuthMethod.SYSTEM, scopes=HUMAN_SCOPES)
    with pytest.raises(SystemIdentityCannotApprove):
        system_principal("model:planner", scopes=HUMAN_SCOPES)
    # Kind follows the actor id's shape, whatever the caller claims.
    assert Principal("event:gmail", AuthMethod.LOCAL_TOKEN).is_system
    assert not Principal("event:gmail", AuthMethod.LOCAL_TOKEN).has(Scope.APPROVE)


def test_an_approval_decided_by_a_system_identity_is_never_executed():
    """Even a decision written straight into graph state cannot spend a system approval."""
    intent = fakes.submit_intent()
    request = ApprovalRequest.for_intent(intent, now=datetime.utcnow())
    decision = ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=request.action_hash,
        verdict=ApprovalVerdict.APPROVED,
        decided_by=WORKER_ACTOR,
        request_id=request.request_id,
    )

    refusal = authorize_execution(
        intent=intent, request=request, decision=decision, now=datetime.utcnow()
    )

    assert refusal is not None
    assert refusal.reason is RefusalReason.SYSTEM_APPROVER


def test_the_approval_record_names_the_decider_and_the_hash(world):
    decided = world.api(BOB).post(
        f"/v1/approvals/{world.bob_request.request_id}/decision",
        json={"approved": True, "action_hash": world.bob_request.action_hash},
    )
    assert decided.status_code == 200, decided.text

    [row] = world.rows(ApprovalModel)
    assert row.decided_by == BOB
    assert row.action_hash == world.bob_request.action_hash
    assert decided.json()["decided_by"] == BOB


# --- 4. Stream tickets ----------------------------------------------------------------


def _ticket(client: TestClient, workflow_id: UUID) -> str:
    response = client.post("/v1/auth/stream-tickets", json={"workflow_id": str(workflow_id)})
    assert response.status_code == 201, response.text
    return response.json()["ticket"]


def test_a_ticket_for_one_workflow_is_rejected_for_another(world):
    ticket = _ticket(world.api(ALICE), world.alice_workflow)
    anonymous = world.api()

    own = anonymous.get(f"/v1/workflows/{world.alice_workflow}/events?follow=false&ticket={ticket}")
    assert own.status_code == 200
    # Another of her own workflows, and someone else's.
    for other in (world.alice_second_workflow, world.bob_workflow):
        response = anonymous.get(f"/v1/workflows/{other}/events?follow=false&ticket={ticket}")
        assert response.status_code == 401, other
    denials = world.denials(ALICE)
    assert [d.action for d in denials] == [f"{AUDIT_DENIED_PREFIX}ticket_wrong_workflow"] * 2
    # A ticket is not a token: it opens nothing but the stream.
    assert (
        anonymous.get(
            f"/v1/workflows/{world.alice_workflow}", headers={"Authorization": f"Bearer {ticket}"}
        ).status_code
        == 401
    )


def test_a_ticket_is_rejected_after_it_expires(world):
    ticket = _ticket(world.api(ALICE), world.alice_workflow)
    path = f"/v1/workflows/{world.alice_workflow}/events?follow=false&ticket={ticket}"

    assert world.api().get(path).status_code == 200
    world.clock[0] += 61
    assert world.api().get(path).status_code == 401
    [denial] = world.denials(ALICE)
    assert denial.action == f"{AUDIT_DENIED_PREFIX}ticket_expired"


def test_a_ticket_carries_read_only(world):
    ticket = _ticket(world.api(ALICE), world.alice_workflow)
    principal = world.signer().verify(ticket, world.alice_workflow)
    assert principal is not None
    assert principal.actor_id == ALICE
    assert principal.scopes == frozenset({Scope.READ})


# --- 5. Repositories, called directly ----------------------------------------------------


def test_scoped_stores_return_nothing_for_another_actor(world):
    factory = world.factory
    services = build_workflow_services(factory)
    board = build_application_board_store(factory)
    audit = build_audit_trail_store(factory)
    progress = build_workflow_progress_log(factory)
    events = build_event_failure_store(factory)
    from personalos.domain.applications import BoardQuery
    from personalos.domain.audit import AuditQuery

    # Alice sees each of hers...
    assert services.reader.read(world.alice_workflow, access=ALICE_ACCESS) is not None
    assert services.actions.read(world.alice_workflow, access=ALICE_ACCESS) is not None
    assert board.card(world.alice_application, access=ALICE_ACCESS) is not None
    assert board.cover_note(world.alice_application, access=ALICE_ACCESS)
    assert audit.workflow_trail(world.alice_workflow, access=ALICE_ACCESS) is not None
    assert progress.workflow_exists(world.alice_workflow, access=ALICE_ACCESS)
    assert progress.events_after(world.alice_workflow, access=ALICE_ACCESS)

    # ...Bob sees none of them, though he names each by id.
    assert services.reader.read(world.alice_workflow, access=BOB_ACCESS) is None
    assert world.alice_workflow not in services.reader.interrupted_workflow_ids(access=BOB_ACCESS)
    assert services.actions.read(world.alice_workflow, access=BOB_ACCESS) is None
    assert {a.approval_id for a in services.approvals.list_pending(access=BOB_ACCESS)} == {
        world.bob_request.request_id
    }
    assert board.card(world.alice_application, access=BOB_ACCESS) is None
    assert board.detail(world.alice_application, access=BOB_ACCESS) is None
    assert board.board(BoardQuery(), access=BOB_ACCESS).total == 0
    assert board.cover_note(world.alice_application, access=BOB_ACCESS) == ""
    assert board.submission_attachments(world.alice_application, access=BOB_ACCESS) == []
    assert audit.workflow_trail(world.alice_workflow, access=BOB_ACCESS) is None
    assert all(
        entry.workflow_id != world.alice_workflow
        for entry in audit.search(AuditQuery(), access=BOB_ACCESS).entries
    )
    assert not progress.workflow_exists(world.alice_workflow, access=BOB_ACCESS)
    assert progress.events_after(world.alice_workflow, access=BOB_ACCESS) == []
    assert events.failures(access=BOB_ACCESS) == []
    assert events.retry(world.alice_event_failure, actor=BOB, access=BOB_ACCESS) is None
    session = factory()
    try:
        credentials = CredentialRepository(session).list_by_provider("google", access=BOB_ACCESS)
        assert credentials == []
        mine = CredentialRepository(session).list_by_provider("google", access=ALICE_ACCESS)
        assert [c.id for c in mine] == [world.alice_credential]
    finally:
        session.close()

    # Writes are scoped as well.
    with pytest.raises(ValueError):
        build_application_lifecycle_store(factory).transition(
            world.alice_application, ApplicationStatus.SAVED, access=BOB_ACCESS
        )
    from personalos.persistence.checkpointer import WorkflowOwnerMismatch

    with pytest.raises(WorkflowOwnerMismatch):
        world.registry.register(
            thread_id=f"job_search:{uuid4().hex}",
            workflow_name=JOB_SEARCH_WORKFLOW,
            workflow_id=world.alice_workflow,
            user_id=owner_id(BOB),
        )

    # The worker's own access still sees everything.
    assert services.reader.read(world.alice_workflow, access=SYSTEM_ACCESS) is not None
    assert len(services.approvals.list_pending(access=SYSTEM_ACCESS)) == 3


def test_scoped_store_methods_cannot_be_called_without_an_access():
    """`access` is keyword-only with no default: a caller has to say whose data it wants."""
    from personalos.executor.google_connect import GoogleConnectionReader
    from personalos.persistence.application_board import ApplicationBoardStore
    from personalos.persistence.application_lifecycle import ApplicationLifecycleStore
    from personalos.persistence.approvals import ApprovalQueue
    from personalos.persistence.audit_trail import AuditTrailStore
    from personalos.persistence.event_failures import EventFailureStore
    from personalos.persistence.workflow_actions import WorkflowActionReader
    from personalos.persistence.workflow_failures import WorkflowFailureStore
    from personalos.persistence.workflow_progress import WorkflowProgressLog
    from personalos.persistence.workflow_status import WorkflowStatusReader

    methods = [
        WorkflowStatusReader.read,
        WorkflowStatusReader.interrupted_workflow_ids,
        WorkflowActionReader.read,
        WorkflowFailureStore.snapshot,
        ApprovalQueue.list_pending,
        ApplicationBoardStore.board,
        ApplicationBoardStore.card,
        ApplicationBoardStore.detail,
        ApplicationBoardStore.submission_attachments,
        ApplicationBoardStore.cover_note,
        ApplicationBoardStore.record_transition_request,
        ApplicationLifecycleStore.transition,
        AuditTrailStore.search,
        AuditTrailStore.workflow_trail,
        WorkflowProgressLog.workflow_exists,
        WorkflowProgressLog.events_after,
        EventFailureStore.failures,
        EventFailureStore.exists,
        EventFailureStore.retry,
        CredentialRepository.list_by_provider,
        GoogleConnectionReader.connections,
        GoogleConnectionReader.status,
    ]
    for method in methods:
        parameter = inspect.signature(method).parameters.get("access")
        assert parameter is not None, method.__qualname__
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, method.__qualname__
        assert parameter.default is inspect.Parameter.empty, method.__qualname__
    # And the decision takes a verified principal, not an actor string.
    assert "principal" in inspect.signature(ApprovalQueue.decide).parameters
    assert "actor_id" not in inspect.signature(ApprovalQueue.decide).parameters


def test_routes_reach_data_only_through_scoped_stores():
    """No route imports the ORM models, the raw repositories or SQLAlchemy itself."""
    import ast

    routes = Path(__file__).resolve().parents[2] / "apps" / "api" / "routes"
    forbidden = (
        "personalos.persistence.models",
        "personalos.persistence.repositories",
        "sqlalchemy",
    )
    for path in routes.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"


# --- 6. Denials are audited, without tokens ----------------------------------------------


def test_every_denial_is_audited_without_token_material(world):
    bob = world.api(BOB)
    reader = world.api(BOB, scopes="read")
    worker = world.api(WORKER_ACTOR)
    ticket = _ticket(world.api(ALICE), world.alice_workflow)

    refused = [
        # Not visible: each of Alice's objects, by id.
        bob.get(f"/v1/workflows/{world.alice_workflow}"),
        bob.get(f"/v1/workflows/{world.alice_workflow}/audit"),
        bob.get(f"/v1/workflows/{world.alice_workflow}/events?follow=false"),
        bob.get(f"/v1/applications/{world.alice_application}"),
        bob.post(
            f"/v1/approvals/{world.alice_request.request_id}/decision",
            json={"approved": True, "action_hash": world.alice_request.action_hash},
        ),
        # Missing scope, and a system identity deciding.
        reader.post("/v1/chat", json={"message": "find me jobs"}),
        worker.post(
            f"/v1/approvals/{world.bob_request.request_id}/decision",
            json={"approved": True, "action_hash": world.bob_request.action_hash},
        ),
        # A ticket for the wrong workflow, and one past its expiry.
        world.api().get(f"/v1/workflows/{world.bob_workflow}/events?follow=false&ticket={ticket}"),
    ]
    world.clock[0] += 61
    refused.append(
        world.api().get(f"/v1/workflows/{world.alice_workflow}/events?follow=false&ticket={ticket}")
    )
    assert [r.status_code for r in refused] == [404] * 5 + [403, 403, 401, 401]

    denials = world.denials()
    assert len(denials) == len(refused)
    by_actor = {}
    for denial in denials:
        by_actor.setdefault(denial.actor, []).append(
            denial.action.removeprefix(AUDIT_DENIED_PREFIX)
        )
    assert by_actor == {
        BOB: ["not_visible"] * 5 + ["missing_scope"],
        WORKER_ACTOR: ["system_identity"],
        ALICE: ["ticket_wrong_workflow", "ticket_expired"],
    }
    for denial in denials:
        assert denial.result == "failure"
        assert denial.workflow_id is None  # the denied actor's record, nobody else's
        assert " /v1/" in denial.target_ref, denial.target_ref

    # No audit row of any kind carries a bearer token or a ticket.
    every_row = json.dumps([row.to_dict() for row in world.rows(AuditEventModel)])
    assert TOKEN_MARKER not in every_row
    assert ticket not in every_row
    assert "pos_st_" not in every_row
    assert "Bearer" not in every_row


def test_the_denied_actor_can_read_its_own_denials_and_nobody_else_can(world):
    world.api(BOB).get(f"/v1/workflows/{world.alice_workflow}")

    bob_trail = world.api(BOB).get("/v1/audit", params={"action": "authz.denied.not_visible"})
    alice_trail = world.api(ALICE).get("/v1/audit", params={"action": "authz.denied.not_visible"})

    assert bob_trail.json()["total"] == 1
    assert alice_trail.json()["total"] == 0
