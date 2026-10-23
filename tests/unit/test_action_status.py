"""Tests for telling apart what the system suggested from what it actually did.

The acceptance criteria this file exists for:

1. **API contract: the action status enum is required on every action
   object**, and a response missing it fails schema validation --
   `test_every_action_object_in_the_api_schema_requires_action_status` and
   `test_a_response_missing_the_action_status_fails_validation`.
2. **A drafted-but-unsent email is RECOMMENDED or AWAITING_APPROVAL, never
   EXECUTED** -- `test_a_drafted_unsent_email_is_never_executed` (the
   derivation, at every stage before the send returns) and
   `test_a_mixed_workflow_through_the_api` (the same through the API, approved
   and waiting on the worker).
3. **CLI output for a mixed workflow separates what was suggested from what
   was done** -- `test_cli_output_for_a_mixed_workflow_separates_suggested_from_done`.

The end-to-end tests run a real API over a database file and a worker whose
graph is a stand-in with the real shapes at its edges, as in
`test_audit_routes`: it parks on real `ApprovalRequest`s, spends the answer
through `authorize_execution`, and sends through the real `ToolExecutor`, so
the receipt an `EXECUTED` action links to is a real `tool_executions` row.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import apps.api.routes.workflows as workflow_routes
from apps.api.main import create_app
from apps.api.routes.approvals import PendingApprovalItem
from apps.api.routes.audit import build_audit_store, get_audit_store
from apps.api.routes.workflows import (
    WorkflowStatusResponse,
    build_workflow_services,
    get_workflow_services,
)
from apps.worker.workflow_commands import process_next
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    build_durable_checkpointer,
    build_tool_executor,
    build_workflow_command_queue,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_job_search_thread,
)
from personalos.cli import cli
from personalos.cli_views import render_workflow
from personalos.domain.action_status import (
    ActionLinks,
    ActionStatus,
    ShortlistItem,
    WorkflowAction,
    derive_shortlist,
    derive_thread_actions,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    ApprovalVerdict,
    ShortlistEntry,
    authorize_execution,
)
from personalos.domain.workflow import (
    JOB_SEARCH_THREAD_NAMESPACE,
    WorkflowCommand,
    WorkflowCommandKind,
)
from personalos.persistence.models import Base
from tests.fixtures import job_search_fakes as fakes

USER_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000bb")
WORKFLOW_ID = UUID("6f1e7b3a-0000-4000-8000-0000000000cc")
THREAD = "job_search:test"
ACTOR = "maddy"


# --- Builders -------------------------------------------------------------------


def reply_draft(message_id: str = "msg-in-1") -> ActionIntent:
    """A drafted reply to a recruiter, as `record_recruiter_events` proposes one."""
    return ActionIntent(
        kind=ActionKind.SEND_RECRUITER_MESSAGE,
        target=f"recruiter@acme.example (thread {message_id})",
        summary="Reply to the recruiter's interview invite: Interview at Acme",
        payload={
            "application_id": str(fakes.APPLICATION_ID),
            "in_reply_to": message_id,
            "classification": "interview_invite",
            "recipient": "recruiter@acme.example",
            "subject": "Re: Interview at Acme",
            "body": "Thursday at 10 works for me.",
        },
        idempotency_key=f"reply-{fakes.APPLICATION_ID}-{message_id}",
    )


def follow_up_draft() -> ActionIntent:
    """A follow-up email drafted to start a new thread, never put to a reviewer."""
    return ActionIntent(
        kind=ActionKind.SEND_RECRUITER_MESSAGE,
        target="recruiter@globex.example",
        summary="Follow up with Globex about the platform role",
        payload={
            "recipient": "recruiter@globex.example",
            "subject": "Following up",
            "body": "Checking in on my application.",
        },
        idempotency_key=f"follow-up-{uuid4()}",
    )


def shortlist(*companies: str) -> list[ShortlistEntry]:
    entries = []
    for rank, company in enumerate(companies, start=1):
        target = fakes.posting(company=company)
        entries.append(
            ShortlistEntry(
                rank=rank,
                scored=fakes.scored(target, score=0.9 - rank / 10),
                evidence=fakes.evidence_check(target),
            )
        )
    return entries


def dump(values) -> list[dict[str, Any]]:
    return [value.model_dump(mode="json") for value in values]


def derive(values: dict[str, Any], **kwargs: Any) -> list[WorkflowAction]:
    return list(
        derive_thread_actions(
            values, workflow_id=WORKFLOW_ID, thread_id=THREAD, now=fakes.NOW, **kwargs
        )
    )


# --- 1. The contract ------------------------------------------------------------


def _schema_refs(schema: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            found.add(ref.rsplit("/", 1)[-1])
        for value in schema.values():
            found |= _schema_refs(value)
    elif isinstance(schema, list):
        for value in schema:
            found |= _schema_refs(value)
    return found


def _response_schemas(openapi: dict[str, Any]) -> set[str]:
    """Every component schema reachable from a response body."""
    components = openapi["components"]["schemas"]
    pending: set[str] = set()
    for path in openapi["paths"].values():
        for operation in path.values():
            for response in (operation.get("responses") or {}).values():
                pending |= _schema_refs(response)
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable or name not in components:
            continue
        reachable.add(name)
        pending |= _schema_refs(components[name])
    return reachable


def test_every_action_object_in_the_api_schema_requires_action_status():
    openapi = create_app().openapi()
    components = openapi["components"]["schemas"]
    assert components["ActionStatus"]["enum"] == [
        "RECOMMENDED",
        "AWAITING_APPROVAL",
        "EXECUTED",
        "DENIED",
        "FAILED",
    ]

    # A receipt names the action it settles but is not an action: it only ever
    # appears inside one, under `WorkflowAction.receipt`.
    not_actions = {"ActionReceipt"}
    action_objects = [
        name
        for name in sorted(_response_schemas(openapi) - not_actions)
        if "action_id" in (components[name].get("properties") or {})
    ]
    # The surfaces that list actions are all covered...
    assert {"WorkflowAction", "PendingApprovalItem", "ShortlistItem"} <= set(action_objects)
    # ...and every one of them requires the status, typed as the enum.
    for name in action_objects:
        schema = components[name]
        assert "action_status" in schema.get("required", []), name
        assert _schema_refs(schema["properties"]["action_status"]) == {"ActionStatus"}, name


@pytest.mark.parametrize("model", [WorkflowAction, PendingApprovalItem, ShortlistItem])
def test_an_action_object_without_a_status_is_not_valid(model):
    with pytest.raises(ValidationError, match="action_status"):
        model.model_validate({"action_id": str(uuid4())})


def test_a_response_missing_the_action_status_fails_validation():
    action = derive(
        {"pending_actions": dump([reply_draft()])},
    )[0].model_dump(mode="json")
    body = {
        "workflow_id": str(WORKFLOW_ID),
        "status": "waiting",
        "audit_url": f"/v1/workflows/{WORKFLOW_ID}/audit",
        "actions": [action],
    }
    assert WorkflowStatusResponse.model_validate(body).actions[0].action_status is (
        ActionStatus.RECOMMENDED
    )

    del action["action_status"]
    with pytest.raises(ValidationError, match="action_status"):
        WorkflowStatusResponse.model_validate(body)


def test_executed_cannot_be_claimed_without_a_successful_receipt():
    base = derive({"pending_actions": dump([reply_draft()])})[0]
    with pytest.raises(ValidationError, match="without a successful receipt"):
        base.model_validate({**base.model_dump(), "action_status": ActionStatus.EXECUTED})
    failed_receipt = ActionReceipt(action_id=base.action_id, ok=False, detail="smtp 550")
    with pytest.raises(ValidationError, match="without a successful receipt"):
        base.model_validate(
            {**base.model_dump(), "action_status": "EXECUTED", "receipt": failed_receipt}
        )


def test_a_successful_receipt_cannot_be_labelled_anything_but_executed():
    base = derive({"pending_actions": dump([reply_draft()])})[0]
    receipt = ActionReceipt(action_id=base.action_id, ok=True, external_reference="gmail-1")
    with pytest.raises(ValidationError, match="successful receipt but is labelled"):
        base.model_validate({**base.model_dump(), "receipt": receipt})


def test_a_shortlist_entry_alone_is_only_ever_a_recommendation():
    with pytest.raises(ValidationError, match="a shortlist entry on its own is a recommendation"):
        ShortlistItem(
            rank=1,
            action_status=ActionStatus.EXECUTED,
            dedupe_key="acme:backend",
            title="Backend",
            company="Acme",
            score=0.9,
        )


# --- 2. Derivation --------------------------------------------------------------


def test_a_drafted_unsent_email_is_never_executed():
    draft = reply_draft()
    request = ApprovalRequest.for_intent(draft, now=fakes.NOW)
    approval = fakes.approval_decision(request)

    # Drafted, not yet put to anyone.
    (action,) = derive({"pending_actions": dump([draft])})
    assert action.action_status is ActionStatus.RECOMMENDED

    # Put to a reviewer.
    state = {"pending_actions": dump([draft]), "approval_requests": dump([request])}
    (action,) = derive(state)
    assert action.action_status is ActionStatus.AWAITING_APPROVAL

    # Approved, and a Gmail *draft* exists for it -- but nothing was sent.
    state["approvals"] = dump([approval])
    draft_row = ActionLinks(operation_id=uuid4(), tool_execution_status="completed")
    (action,) = derive(state, links={draft.idempotency_key: draft_row})
    assert action.action_status is ActionStatus.AWAITING_APPROVAL
    assert action.approval_state.value == "approved"
    assert "Nothing has been sent" in action.detail
    assert action.receipt is None

    # The send itself failed.
    failed = ActionReceipt(action_id=draft.action_id, ok=False, detail="mailbox unavailable")
    (action,) = derive({**state, "action_receipts": dump([failed])})
    assert action.action_status is ActionStatus.FAILED

    # Only the send's own receipt makes it EXECUTED.
    sent = ActionReceipt(action_id=draft.action_id, ok=True, external_reference="gmail-1")
    (action,) = derive({**state, "action_receipts": dump([sent])})
    assert action.action_status is ActionStatus.EXECUTED
    assert action.receipt.external_reference == "gmail-1"


def test_a_denied_or_refused_action_says_why_nothing_happened():
    draft, submit = reply_draft(), fakes.submit_intent()
    requests = [ApprovalRequest.for_intent(i, now=fakes.NOW) for i in (draft, submit)]
    denied = fakes.approval_decision(requests[0], ApprovalVerdict.REJECTED)
    state = {
        "pending_actions": dump([draft, submit]),
        "approval_requests": dump(requests),
        "approvals": dump([denied, fakes.approval_decision(requests[1])]),
    }
    refusals = [
        authorize_execution(
            intent=i,
            request=r,
            decision=ApprovalDecision.model_validate(d),
            now=fakes.NOW + timedelta(days=30),
        )
        for i, r, d in zip((draft, submit), requests, state["approvals"], strict=True)
    ]
    state["approval_refusals"] = dump(refusals)

    by_kind = {action.kind: action for action in derive(state)}
    assert by_kind[ActionKind.SEND_RECRUITER_MESSAGE].action_status is ActionStatus.DENIED
    assert by_kind[ActionKind.SUBMIT_APPLICATION].action_status is ActionStatus.FAILED
    assert by_kind[ActionKind.SUBMIT_APPLICATION].refusal_reason.value == "approval_expired"


def test_the_approval_preview_names_target_and_consequences():
    draft = reply_draft()
    state = {
        "pending_actions": dump([draft]),
        "approval_requests": dump([ApprovalRequest.for_intent(draft, now=fakes.NOW)]),
    }
    (action,) = derive(state)
    assert action.target == draft.target
    assert action.preview.recipient == "recruiter@acme.example"
    assert action.preview.subject == "Re: Interview at Acme"
    assert action.preview.is_reply and action.preview.in_reply_to == "msg-in-1"
    assert "+Thursday at 10 works for me." in action.preview.body_diff
    assert action.consequences[0] == (
        "Sends an email to recruiter@acme.example as a reply in thread msg-in-1."
    )
    assert "No attachments." in action.consequences

    (fresh,) = derive({"pending_actions": dump([follow_up_draft()])})
    assert not fresh.preview.is_reply
    assert fresh.consequences[0].endswith("as a new thread.")


def test_shortlist_entries_take_the_status_of_their_submission_only():
    entries = shortlist("Acme", "Globex")
    submit = fakes.submit_intent(entries[0].scored.posting)
    state = {
        "shortlist": dump(entries),
        "pending_actions": dump([submit]),
        "approval_requests": dump([ApprovalRequest.for_intent(submit, now=fakes.NOW)]),
    }
    actions = derive(state)
    acme, globex = derive_shortlist(state, thread_id=THREAD, actions=actions)
    assert acme.action_status is ActionStatus.AWAITING_APPROVAL
    assert acme.action_id == submit.action_id
    assert globex.action_status is ActionStatus.RECOMMENDED
    assert globex.action_id is None


# --- 3. End to end: API, worker, CLI --------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "actions.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


def factory_for(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def services_for(db_path: Path):
    return build_workflow_services(factory_for(db_path))


def api_process(db_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: services_for(db_path)
    app.dependency_overrides[get_audit_store] = lambda: build_audit_store(factory_for(db_path))
    return TestClient(app)


class Mailer:
    """The provider: accepts a send and hands back a receipt."""

    def __init__(self):
        self.sent: list[ActionIntent] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.sent.append(intent)
        return ActionReceipt(
            action_id=intent.action_id, ok=True, external_reference=f"gmail-{len(self.sent)}"
        )


class _State(TypedDict, total=False):
    shortlist: list[dict[str, Any]]
    pending_actions: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    approval_refusals: list[dict[str, Any]]
    action_receipts: list[dict[str, Any]]


class Worker:
    """Parks on the approval requests it is given, then executes what was approved."""

    def __init__(self, db_path: Path):
        engine = create_engine(f"sqlite:///{db_path}")
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.registry = build_workflow_thread_registry(self.factory)
        self.checkpointer = build_durable_checkpointer(self.factory, self.registry)
        self.leases = build_workflow_lease_store(self.factory)
        self.queue = build_workflow_command_queue(self.factory)
        self.mailer = Mailer()

        graph = StateGraph(_State)
        graph.add_node("approval_checkpoint", self._approval_checkpoint)
        graph.add_node("execute_approved_actions", self._execute)
        graph.add_edge(START, "approval_checkpoint")
        graph.add_edge("approval_checkpoint", "execute_approved_actions")
        graph.add_edge("execute_approved_actions", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _approval_checkpoint(state: _State) -> dict[str, Any]:
        decisions = interrupt(
            {"workflow": "job_search", "stage": "mixed", "requests": state["approval_requests"]}
        )
        return {"approvals": decisions}

    async def _execute(self, state: _State, config) -> dict[str, Any]:
        thread = self.registry.resolve(config["configurable"]["thread_id"])
        executor = build_tool_executor(self.mailer, self.factory, workflow_id=thread.workflow_id)
        requests = {
            r.action_id: r for r in map(ApprovalRequest.model_validate, state["approval_requests"])
        }
        decisions = {
            d.action_id: d for d in map(ApprovalDecision.model_validate, state["approvals"])
        }
        receipts: list[ActionReceipt] = []
        refusals: list[ApprovalRefusal] = []
        for intent in map(ActionIntent.model_validate, state["pending_actions"]):
            if intent.action_id not in requests:
                continue  # only ever suggested; nobody was asked
            refusal = authorize_execution(
                intent=intent,
                request=requests[intent.action_id],
                decision=decisions.get(intent.action_id),
                now=datetime.utcnow(),
            )
            if refusal is not None:
                refusals.append(refusal)
            else:
                receipts.append(await executor.execute(intent, decisions[intent.action_id]))
        return {"action_receipts": dump(receipts), "approval_refusals": dump(refusals)}

    def run_next(self) -> bool:
        runner = DurableWorkflowRunner(self.graph, registry=self.registry, leases=self.leases)
        return asyncio.run(process_next(self.queue, {JOB_SEARCH_THREAD_NAMESPACE: runner}))


def park_mixed(db_path: Path) -> tuple[Worker, UUID, dict[str, ActionIntent]]:
    """A run that drafted a reply, proposed a submission, and drafted a follow-up.

    The reply and the submission are put to a reviewer; the follow-up is only
    suggested. Two postings are shortlisted, one of them the submission's.
    """
    worker = Worker(db_path)
    entries = shortlist("Acme", "Globex")
    intents = {
        "reply": reply_draft(),
        "submit": fakes.submit_intent(entries[0].scored.posting),
        "follow_up": follow_up_draft(),
    }
    requests = [
        ApprovalRequest.for_intent(intents[name], now=datetime.utcnow())
        for name in ("reply", "submit")
    ]
    workflow_id = worker.registry.create_workflow(JOB_SEARCH_WORKFLOW)
    thread = register_job_search_thread(
        user_id=USER_ID, registry=worker.registry, workflow_id=workflow_id, search_key="mixed"
    )
    worker.queue.enqueue(
        WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.START,
            workflow_id=workflow_id,
            thread_id=thread.thread_id,
            actor_id=ACTOR,
            correlation_id=uuid4(),
            graph_input={
                "shortlist": dump(entries),
                "pending_actions": dump(intents.values()),
                "approval_requests": dump(requests),
            },
        )
    )
    assert worker.run_next()
    return worker, workflow_id, intents


def by_action(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {action["action_id"]: action for action in body["actions"]}


def run_mixed(db_path: Path) -> tuple[UUID, dict[str, ActionIntent]]:
    """Park the mixed run, approve the reply, deny the submission, let the worker act."""
    worker, workflow_id, intents = park_mixed(db_path)
    api = api_process(db_path)
    pending = {item["action_id"]: item for item in api.get("/v1/approvals").json()["approvals"]}
    for name, approved in (("reply", True), ("submit", False)):
        item = pending[str(intents[name].action_id)]
        decided = api.post(
            f"/v1/approvals/{item['id']}/decision",
            json={"approved": approved, "action_hash": item["action_hash"]},
            headers={"X-Actor-Id": ACTOR},
        )
        assert decided.status_code == 200, decided.text
    assert worker.run_next()
    assert [intent.action_id for intent in worker.mailer.sent] == [intents["reply"].action_id]
    return workflow_id, intents


def test_a_mixed_workflow_through_the_api(db_path):
    worker, workflow_id, intents = park_mixed(db_path)
    reply, submit, follow_up = (str(intents[k].action_id) for k in ("reply", "submit", "follow_up"))
    api = api_process(db_path)

    # Parked: nothing has happened, and every surface says so.
    body = api.get(f"/v1/workflows/{workflow_id}").json()
    actions = by_action(body)
    assert actions[reply]["action_status"] == "AWAITING_APPROVAL"
    assert actions[submit]["action_status"] == "AWAITING_APPROVAL"
    assert actions[follow_up]["action_status"] == "RECOMMENDED"
    assert body["action_counts"]["EXECUTED"] == 0
    assert [item["action_status"] for item in body["shortlist"]] == [
        "AWAITING_APPROVAL",
        "RECOMMENDED",
    ]
    assert body["audit_url"] == f"/v1/workflows/{workflow_id}/audit"

    approvals = {item["action_id"]: item for item in api.get("/v1/approvals").json()["approvals"]}
    assert set(approvals) == {reply, submit}
    assert {item["action_status"] for item in approvals.values()} == {"AWAITING_APPROVAL"}
    shown = approvals[reply]
    assert shown["target"] == "recruiter@acme.example (thread msg-in-1)"
    assert shown["preview"]["recipient"] == "recruiter@acme.example"
    assert shown["preview"]["subject"] == "Re: Interview at Acme"
    assert shown["preview"]["in_reply_to"] == "msg-in-1"
    assert "+Thursday at 10 works for me." in shown["preview"]["body_diff"]
    assert "as a reply in thread msg-in-1" in shown["consequences"][0]

    # Decided, but the worker has not run: approving is not sending.
    for item, approved in ((approvals[reply], True), (approvals[submit], False)):
        assert (
            api.post(
                f"/v1/approvals/{item['id']}/decision",
                json={"approved": approved, "action_hash": item["action_hash"]},
                headers={"X-Actor-Id": ACTOR},
            ).status_code
            == 200
        )
    actions = by_action(api.get(f"/v1/workflows/{workflow_id}").json())
    assert actions[reply]["action_status"] == "AWAITING_APPROVAL"
    assert actions[reply]["approval_state"] == "approved"
    assert actions[reply]["receipt"] is None
    assert actions[submit]["action_status"] == "DENIED"
    assert actions[submit]["decided_by"] == ACTOR

    # The worker sends: only now is the reply EXECUTED, receipt and audit linked.
    assert worker.run_next()
    body = api.get(f"/v1/workflows/{workflow_id}").json()
    actions = by_action(body)
    executed = actions[reply]
    assert executed["action_status"] == "EXECUTED"
    assert executed["receipt"]["external_reference"] == "gmail-1"
    assert executed["links"]["operation_id"] is not None
    assert executed["links"]["tool_execution_status"] == "completed"
    assert executed["links"]["audit_event_ids"]
    assert executed["links"]["audit_url"] == f"/v1/workflows/{workflow_id}/audit"
    audit = api.get(executed["links"]["audit_url"]).json()["entries"]
    assert set(executed["links"]["audit_event_ids"]) <= {e["audit_event_id"] for e in audit}

    assert actions[submit]["action_status"] == "DENIED"
    assert actions[follow_up]["action_status"] == "RECOMMENDED"
    assert body["action_counts"] == {
        "RECOMMENDED": 1,
        "AWAITING_APPROVAL": 0,
        "EXECUTED": 1,
        "DENIED": 1,
        "FAILED": 0,
    }
    assert [item["action_status"] for item in body["shortlist"]] == ["DENIED", "RECOMMENDED"]


def test_cli_output_for_a_mixed_workflow_separates_suggested_from_done(db_path, monkeypatch):
    workflow_id, intents = run_mixed(db_path)
    monkeypatch.setattr(workflow_routes, "get_workflow_services", lambda: services_for(db_path))

    result = CliRunner().invoke(cli, ["workflow-status", str(workflow_id)])
    assert result.exit_code == 0, result.output
    output = result.output
    print(output)  # the manual UX check reads this: pytest -s -k cli_output

    done = output.index("DONE -- executed, with receipts (1)")
    denied = output.index("DENIED -- nothing was done (1)")
    suggested = output.index("SUGGESTED -- not acted on (1)")
    shortlisted = output.index("SHORTLIST -- recommendations, not applications (2)")
    assert done < suggested < denied < shortlisted

    done_section = output[done : output.index("\n\n", done)]
    assert intents["reply"].summary in done_section
    assert "receipt: gmail-1 · operation " in done_section
    assert f"audit: /v1/workflows/{workflow_id}/audit" in done_section
    assert intents["follow_up"].summary not in done_section
    assert intents["submit"].summary not in done_section

    suggested_section = output[suggested : output.index("\n\n", suggested)]
    assert "[ ] RECOMMENDED" in suggested_section
    assert intents["follow_up"].summary in suggested_section

    assert "#1 Acme" in output and "submission DENIED -- not submitted" in output
    assert "#2 Globex" in output and "recommended -- not applied" in output

    as_json = CliRunner().invoke(cli, ["workflow-status", str(workflow_id), "--json"])
    assert as_json.exit_code == 0, as_json.output
    assert (
        WorkflowStatusResponse.model_validate_json(as_json.output).action_counts[
            ActionStatus.EXECUTED
        ]
        == 1
    )


def test_cli_approvals_shows_the_action_before_anything_is_sent(db_path, monkeypatch):
    park_mixed(db_path)
    monkeypatch.setattr(workflow_routes, "get_workflow_services", lambda: services_for(db_path))

    result = CliRunner().invoke(cli, ["approvals"])
    assert result.exit_code == 0, result.output
    output = result.output
    assert output.startswith("WAITING FOR YOUR APPROVAL -- nothing has been sent (2)")
    assert "EXECUTED" not in output
    assert "target: recruiter@acme.example (thread msg-in-1)" in output
    assert "to: recruiter@acme.example" in output
    assert "thread: reply to msg-in-1" in output
    assert "subject: Re: Interview at Acme" in output
    assert "+Thursday at 10 works for me." in output


def test_render_workflow_always_shows_the_done_section():
    view = WorkflowStatusResponse.model_validate(
        {
            "workflow_id": str(WORKFLOW_ID),
            "status": "waiting",
            "audit_url": f"/v1/workflows/{WORKFLOW_ID}/audit",
            "actions": [
                derive({"pending_actions": dump([reply_draft()])})[0].model_dump(mode="json")
            ],
        }
    )
    output = render_workflow(view)
    assert "DONE -- executed, with receipts (0)\n  (nothing has been executed)" in output
    assert "SUGGESTED -- not acted on (1)" in output
