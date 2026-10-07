"""Nothing a draft is used for leaves this system, or lands on a file, unapproved.

Storing a draft is automatic (`test_artifact_prep.py`). Everything downstream
of that is not, and the two acceptance criteria this file exists for are:

- `test_sending_or_submitting_a_draft_without_an_approval_is_denied_by_policy`
- `test_overwriting_an_existing_resume_requires_approval_and_keeps_a_backup`

Both go through the real policy engine, and the overwrite through the real
files server on a temporary directory, so "denied" means the provider or the
disk was never touched rather than that a flag was set.
"""

import hashlib
from pathlib import Path

import pytest

from mcp_servers.files.sandbox import PathSandbox
from mcp_servers.files.server import FilesMCPServer
from personalos.bootstrap import (
    build_document_overwrite_executor,
    build_tool_executor,
    build_tool_gateway,
)
from personalos.domain.artifacts import content_sha256, document_overwrite_intent
from personalos.domain.job_search import (
    PAYLOAD_ATTACHMENTS,
    PAYLOAD_BODY,
    PAYLOAD_RECIPIENT,
    PAYLOAD_SUBJECT,
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    RiskLevel,
    authorize_execution,
)
from personalos.domain.models import ActionTarget, ToolCallRequest
from personalos.executor.tool_executor import ACTION_TOOLS
from personalos.mcp.manager import MCPServerManager
from personalos.persistence.models import (
    AuditEventModel,
    PolicyDecisionModel,
    ToolExecutionModel,
)
from personalos.policy import (
    ApprovalGrant,
    ApprovalRequired,
    Decision,
    IntentOrigin,
    PolicyDenied,
    PolicyEngine,
    PolicyViolation,
    Provenance,
    ToolIntent,
    default_policy_engine,
)
from personalos.policy.rules import (
    FILE_TOOL_ARGUMENTS,
    DocumentOverwriteRule,
    MutatingToolRule,
    RequireProvenanceRule,
    ToolAllowlistRule,
)
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.durable_workflow import session_factory

OLD_RESUME = "# Resume\n\n- Built Python APIs at Acme Corp\n- Operated Kafka pipelines\n"
NEW_RESUME = "# Resume\n\n- Built Python APIs on Postgres at Acme Corp\n- Operated Kafka pipelines\n"


class SpyProvider:
    """Stands in for the submission/mail adapter and counts what reached it."""

    def __init__(self):
        self.calls: list[ActionIntent] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.calls.append(intent)
        return ActionReceipt(action_id=intent.action_id, ok=True, external_reference="ext-1")


def _rows(factory, model):
    session = factory()
    try:
        return [row.to_dict() for row in session.query(model).all()]
    finally:
        session.close()


def _approve(intent: ActionIntent, verdict=ApprovalVerdict.APPROVED) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=verdict,
        decided_by="reviewer@example.test",
    )


def _draft_action(kind: ActionKind) -> ActionIntent:
    """Sending or submitting a stored draft, as the graph would propose it."""
    cover_letter = "Dear Hiring Manager,\nI built Python APIs at Acme Corp."
    return ActionIntent(
        kind=kind,
        target="recruiter@initech.test",
        summary="Send the tailored cover letter to Initech",
        payload={
            PAYLOAD_RECIPIENT: "recruiter@initech.test",
            PAYLOAD_SUBJECT: "Application for Senior Backend Engineer at Initech",
            PAYLOAD_BODY: cover_letter,
            PAYLOAD_ATTACHMENTS: [
                {
                    "name": "resume",
                    "artifact_type": "resume",
                    "sha256": content_sha256("Tailored resume"),
                    "artifact_version_id": "7c0f8a52-0000-4000-8000-000000000001",
                    "evidence_ids": ["ev-acme"],
                }
            ],
        },
        idempotency_key=f"{kind.value}-initech-backend",
    )


# --- Send / submit -----------------------------------------------------------


@pytest.mark.parametrize(
    "kind", [ActionKind.SUBMIT_APPLICATION, ActionKind.SEND_RECRUITER_MESSAGE]
)
@pytest.mark.parametrize("answer", ["none", "pending", "rejected"])
async def test_sending_or_submitting_a_draft_without_an_approval_is_denied_by_policy(
    tmp_path, kind, answer
):
    factory = session_factory(tmp_path / "gate.db")
    provider = SpyProvider()
    intent = _draft_action(kind)
    decision = {
        "none": None,
        "pending": _approve(intent, ApprovalVerdict.PENDING),
        "rejected": _approve(intent, ApprovalVerdict.REJECTED),
    }[answer]

    with pytest.raises(ApprovalRequired) as excinfo:
        await build_tool_executor(provider, factory).execute(intent, decision)

    # The policy engine's own verdict, on record, and nothing past it.
    assert excinfo.value.decision.rule == "permission_class"
    (verdict,) = _rows(factory, PolicyDecisionModel)
    assert verdict["tool"] == ACTION_TOOLS[kind]
    assert verdict["decision"] == Decision.REQUIRE_APPROVAL.value
    assert provider.calls == []
    assert _rows(factory, ToolExecutionModel) == []
    assert _rows(factory, AuditEventModel) == []


@pytest.mark.parametrize(
    "tool", ["jobs.submit_application", "google.gmail_send_message", "files.overwrite_document"]
)
def test_no_origin_gets_an_external_write_or_overwrite_unattended(tool):
    engine = default_policy_engine()
    scopes = sorted(engine.tool_permissions[tool].scopes)

    for origin in IntentOrigin:
        verdict = engine.evaluate(
            "graph:job_search", None, tool, "0" * 64, scopes, Provenance(origin=origin, requested_by="t")
        )
        assert verdict == Decision.REQUIRE_APPROVAL


async def test_an_approval_of_one_draft_does_not_send_a_different_one(tmp_path):
    """The body and the attachment hashes are in the action's hash."""
    factory = session_factory(tmp_path / "gate.db")
    provider = SpyProvider()
    approved = _draft_action(ActionKind.SEND_RECRUITER_MESSAGE)
    swapped = approved.model_copy(
        update={"payload": {**approved.payload, PAYLOAD_BODY: "I built Python APIs at Google."}}
    )

    with pytest.raises(ApprovalRequired):
        await build_tool_executor(provider, factory).execute(swapped, _approve(approved))

    assert provider.calls == []


async def test_an_approved_send_goes_through_once(tmp_path):
    factory = session_factory(tmp_path / "gate.db")
    provider = SpyProvider()
    intent = _draft_action(ActionKind.SEND_RECRUITER_MESSAGE)

    receipt = await build_tool_executor(provider, factory).execute(intent, _approve(intent))

    assert receipt.ok and len(provider.calls) == 1


# --- What the reviewer is shown ----------------------------------------------


def test_the_approval_request_shows_recipient_subject_body_diff_and_attachments():
    intent = _draft_action(ActionKind.SEND_RECRUITER_MESSAGE)

    request = ApprovalRequest.for_intent(intent, now=fakes.NOW)

    preview = request.preview
    assert preview.recipient == "recruiter@initech.test"
    assert preview.subject == "Application for Senior Backend Engineer at Initech"
    # A new message diffs against nothing: every line is an addition.
    assert preview.body_diff.splitlines() == [
        "--- current",
        "+++ proposed",
        "@@ -0,0 +1,2 @@",
        "+Dear Hiring Manager,",
        "+I built Python APIs at Acme Corp.",
    ]
    (attachment,) = preview.attachments
    assert attachment.artifact_type == "resume"
    assert attachment.sha256 == content_sha256("Tailored resume")
    assert attachment.evidence_ids == ("ev-acme",)
    # And it survives the trip into the interrupt payload and back.
    assert ApprovalRequest.model_validate(request.model_dump(mode="json")) == request


def test_the_overwrite_request_shows_the_diff_against_the_current_file():
    intent = document_overwrite_intent(
        path="resume.md",
        current_content=OLD_RESUME,
        new_content=NEW_RESUME,
        requested_by="graph:job_search#export_resume",
    )

    request = ApprovalRequest.for_intent(intent, now=fakes.NOW)

    assert request.kind == ActionKind.OVERWRITE_DOCUMENT
    assert request.risk == RiskLevel.MEDIUM
    assert request.requested_scopes == ("artifacts:write",)
    assert request.preview.recipient == "resume.md"
    assert "-- Built Python APIs at Acme Corp" in request.preview.body_diff
    assert "+- Built Python APIs on Postgres at Acme Corp" in request.preview.body_diff
    assert intent.payload["expected_sha256"] == content_sha256(OLD_RESUME)
    # With no request and no decision, the graph's own last gate refuses it too.
    assert authorize_execution(intent=intent, request=None, decision=None, now=fakes.NOW)


# --- Overwriting an existing document ---------------------------------------


@pytest.fixture
def resume(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    path = root / "resume.md"
    path.write_text(OLD_RESUME)
    return path


@pytest.fixture
def gateway(resume: Path):
    """The production gateway over a files server confined to the resume's directory."""
    manager = MCPServerManager()
    manager.register_server(FilesMCPServer(PathSandbox([resume.parent])))
    return build_tool_gateway(manager=manager, policy=default_policy_engine())


def _backups(resume: Path) -> list[Path]:
    return sorted(resume.parent.glob("resume.md.bak-*"))


def _overwrite_tool_intent(**arguments) -> ToolIntent:
    defaults = {
        "path": "resume.md",
        "content": NEW_RESUME,
        "expected_sha256": content_sha256(OLD_RESUME),
        "idempotency_key": "overwrite-resume-1",
    }
    defaults.update(arguments)
    return ToolIntent(
        server="files",
        tool="overwrite_document",
        arguments={k: v for k, v in defaults.items() if v is not None},
        mutating=True,
        requested_by="test:overwrite",
    )


def _overwrite_action() -> ActionIntent:
    return document_overwrite_intent(
        path="resume.md",
        current_content=OLD_RESUME,
        new_content=NEW_RESUME,
        requested_by="graph:job_search#export_resume",
    )


async def test_overwriting_an_existing_resume_requires_approval_and_keeps_a_backup(
    tmp_path, resume, gateway
):
    factory = session_factory(tmp_path / "gate.db")
    executor = build_tool_executor(build_document_overwrite_executor(gateway), factory)
    intent = _overwrite_action()

    # Without an approval: refused by policy, and the file is exactly as it was.
    with pytest.raises(ApprovalRequired):
        await executor.execute(intent, _approve(intent, ApprovalVerdict.PENDING))
    assert resume.read_text() == OLD_RESUME
    assert _backups(resume) == []

    # With one: the new version lands, and the old one is beside it.
    receipt = await executor.execute(intent, _approve(intent))

    assert receipt.ok is True
    assert resume.read_text() == NEW_RESUME
    (backup,) = _backups(resume)
    assert backup.read_text() == OLD_RESUME
    assert receipt.external_reference == str(backup)
    assert backup.name == f"resume.md.bak-{content_sha256(OLD_RESUME)[:12]}"

    # Audited as the destructive tool it is, against the approval it ran under.
    (execution,) = _rows(factory, ToolExecutionModel)
    assert execution["tool_name"] == "files.overwrite_document"
    assert execution["approved_by"] == "reviewer@example.test"
    assert [row["decision"] for row in _rows(factory, PolicyDecisionModel)] == [
        Decision.REQUIRE_APPROVAL.value,
        Decision.REQUIRE_APPROVAL.value,
    ]


async def test_an_overwrite_approved_against_a_version_that_has_since_changed_is_refused(
    tmp_path, resume, gateway
):
    """The precondition: the approval was of a diff against one version of the file."""
    factory = session_factory(tmp_path / "gate.db")
    executor = build_tool_executor(build_document_overwrite_executor(gateway), factory)
    intent = _overwrite_action()
    edited_by_hand = OLD_RESUME + "- Added by the candidate after the request went out\n"
    resume.write_text(edited_by_hand)

    receipt = await executor.execute(intent, _approve(intent))

    assert receipt.ok is False
    assert "changed since it was read" in receipt.detail
    assert resume.read_text() == edited_by_hand
    assert _backups(resume) == []


async def test_the_overwrite_tool_cannot_be_dispatched_without_a_grant(resume, gateway):
    intent = _overwrite_tool_intent()

    with pytest.raises(ApprovalRequired):
        await gateway.dispatch(intent)
    # A grant for some other content does not clear it either.
    other = _overwrite_tool_intent(content="something else entirely")
    stale = ApprovalGrant(
        intent_id=intent.intent_id, intent_fingerprint=other.fingerprint(), approved_by="r"
    )
    with pytest.raises(ApprovalRequired):
        await gateway.dispatch(intent, stale)

    assert resume.read_text() == OLD_RESUME
    assert _backups(resume) == []


async def test_an_overwrite_with_no_precondition_is_denied_outright(resume, gateway):
    intent = _overwrite_tool_intent(expected_sha256=None)
    grant = ApprovalGrant(
        intent_id=intent.intent_id, intent_fingerprint=intent.fingerprint(), approved_by="r"
    )

    with pytest.raises(PolicyDenied) as excinfo:
        await gateway.dispatch(intent, grant)

    assert excinfo.value.decision.rule == "document_overwrite"
    assert resume.read_text() == OLD_RESUME


def test_auto_approving_draft_writes_never_extends_to_replacing_a_file():
    """`files.write_file` may be trusted to create; with a hash, it is an overwrite."""
    engine = PolicyEngine(
        [
            RequireProvenanceRule(),
            ToolAllowlistRule(FILE_TOOL_ARGUMENTS),
            MutatingToolRule(auto_approved={"files.write_file"}),
            DocumentOverwriteRule(),
        ]
    )

    def write(**extra) -> ToolIntent:
        return ToolIntent(
            server="files",
            tool="write_file",
            arguments={"path": "draft.md", "content": "x", "idempotency_key": "write-1234", **extra},
            mutating=True,
            requested_by="test:write",
        )

    assert engine.evaluate_intent(write()).decision == Decision.ALLOW
    replacing = engine.evaluate_intent(write(expected_sha256="a" * 64))
    assert replacing.decision == Decision.REQUIRE_APPROVAL
    assert replacing.rule == "document_overwrite"


async def test_the_overwrite_executor_refuses_an_action_no_decision_authorizes(gateway, resume):
    """Belt and braces behind `ToolExecutor`: no approval, no grant, no call."""
    intent = _overwrite_action()

    with pytest.raises(PolicyViolation):
        await build_document_overwrite_executor(gateway).execute(
            intent, _approve(intent, ApprovalVerdict.REJECTED)
        )

    assert resume.read_text() == OLD_RESUME


# --- The files tool itself ---------------------------------------------------


def _call(tool: str, **params) -> ToolCallRequest:
    return ToolCallRequest(target=ActionTarget(server="files", tool=tool), params=params)


async def test_overwrite_document_only_replaces_and_always_needs_the_current_hash(resume):
    server = FilesMCPServer(PathSandbox([resume.parent]))
    sha = hashlib.sha256(OLD_RESUME.encode()).hexdigest()

    missing_hash = await server.execute(
        _call("overwrite_document", path="resume.md", content="x", idempotency_key="k-000001")
    )
    not_there = await server.execute(
        _call(
            "overwrite_document",
            path="cover_letter.md",
            content="x",
            expected_sha256=sha,
            idempotency_key="k-000002",
        )
    )
    stale = await server.execute(
        _call(
            "overwrite_document",
            path="resume.md",
            content="x",
            expected_sha256="0" * 64,
            idempotency_key="k-000003",
        )
    )

    assert not missing_hash.ok and not not_there.ok and not stale.ok
    assert "does not exist" in not_there.error.message
    assert resume.read_text() == OLD_RESUME
    assert not (resume.parent / "cover_letter.md").exists()
    assert _backups(resume) == []


async def test_a_retried_overwrite_replays_and_leaves_one_backup(resume):
    server = FilesMCPServer(PathSandbox([resume.parent]))
    request = _call(
        "overwrite_document",
        path="resume.md",
        content=NEW_RESUME,
        expected_sha256=content_sha256(OLD_RESUME),
        idempotency_key="k-retry-01",
    )

    first = await server.execute(request)
    second = await server.execute(request)

    assert first.ok and second.ok and second.replayed
    assert first.result["previous_sha256"] == content_sha256(OLD_RESUME)
    assert resume.read_text() == NEW_RESUME
    assert [path.read_text() for path in _backups(resume)] == [OLD_RESUME]
