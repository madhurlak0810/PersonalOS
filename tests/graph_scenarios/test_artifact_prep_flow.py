"""Artifact prep through the compiled graph: drafts first, approval before anything else.

The real `TailoredPacketBuilder` (over a database, with a bag-of-words embedder
and a scripted writer) is wired in as the graph's `packet_builder`, so these
runs show the order the phase promises: evidence is selected and drafts are
stored on the way *to* the approval interrupt, and nothing is submitted until
a reviewer has answered a request that shows them what would be sent.
"""

import pytest
from langgraph.types import Command

from personalos.bootstrap import build_artifact_packet_builder
from personalos.domain.artifacts import UntraceableClaim, content_sha256
from personalos.domain.job_search import ApprovalDecision, ApprovalRequest, ApprovalVerdict
from personalos.domain.models import ArtifactType
from personalos.graphs.job_search import JobSearchGraph
from personalos.persistence.models import ArtifactVersionModel
from tests.fixtures import artifact_prep as prep
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.durable_workflow import session_factory

POSTING_URL = "https://example.test/initech"


def _build(tmp_path, proposals=None):
    factory = session_factory(tmp_path / "flow.db")
    embedder = prep.HashingEmbedder()
    ids = prep.seed_evidence(factory, embedder)
    writer = prep.ScriptedDraftWriter(
        proposals(ids)
        if proposals
        else {
            ArtifactType.RESUME: prep.resume_proposal(ids),
            ArtifactType.COVER_LETTER: prep.cover_letter_proposal(ids),
        }
    )
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [
            fakes.FakeProvider(results=[fakes.raw_posting(company="Initech", url=POSTING_URL)])
        ],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": build_artifact_packet_builder(writer, embedder, factory, top_k=3),
        "approval_gate": fakes.NoStandingApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
    }
    return JobSearchGraph(**ports).build(), ports, factory, ids


def _versions(factory) -> list[dict]:
    session = factory()
    try:
        return [row.to_dict() for row in session.query(ArtifactVersionModel).all()]
    finally:
        session.close()


def _config(name: str) -> dict:
    return {"configurable": {"thread_id": f"artifact-prep-{name}"}}


async def _run_to_interrupt(graph, cfg):
    final = await graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "prepare_application": True}, config=cfg
    )
    (interrupt,) = final["__interrupt__"]
    return final, interrupt.value


async def test_drafts_are_stored_before_the_interrupt_and_nothing_is_submitted(tmp_path):
    graph, ports, factory, ids = _build(tmp_path)

    final, payload = await _run_to_interrupt(graph, _config("park"))

    # Drafts exist as rows already: creating them needed no approval.
    stored = {row["artifact_type"]: row for row in _versions(factory)}
    assert set(stored) == {"resume", "cover_letter"}
    assert [e["ref"] for e in stored["resume"]["evidence"]] == [ids[prep.ACME], ids[prep.PGSYNC]]
    # Submitting them did, and has not happened.
    assert ports["action_executor"].executed == []
    assert ports["application_store"].created == []

    # The approval payload shows what would go out, to whom.
    (request,) = payload["requests"]
    preview = request["preview"]
    assert preview["recipient"] == POSTING_URL
    assert preview["subject"] == "Application for Senior Backend Engineer at Initech"
    assert "+Dear Hiring Manager," in preview["body_diff"]
    assert "+At Acme Corp I built Python APIs" in preview["body_diff"]
    attachments = {item["artifact_type"]: item for item in preview["attachments"]}
    assert set(attachments) == {"resume", "cover_letter"}
    assert attachments["resume"]["sha256"] == content_sha256(stored["resume"]["content"])
    assert attachments["resume"]["artifact_version_id"] == stored["resume"]["id"]
    assert attachments["resume"]["evidence_ids"] == [ids[prep.ACME], ids[prep.PGSYNC]]
    # The packet in state points at the same rows.
    assert {a["artifact_version_id"] for a in final["application_packet"]["artifacts"]} == {
        row["id"] for row in stored.values()
    }


@pytest.mark.parametrize(
    ("verdict", "submitted"),
    [(ApprovalVerdict.APPROVED, True), (ApprovalVerdict.REJECTED, False)],
)
async def test_the_packet_is_submitted_only_on_an_approval(tmp_path, verdict, submitted):
    graph, ports, factory, _ = _build(tmp_path)
    cfg = _config(verdict.value)
    _, payload = await _run_to_interrupt(graph, cfg)
    request = ApprovalRequest.model_validate(payload["requests"][0])
    decision = ApprovalDecision(
        action_id=request.action_id,
        action_fingerprint=request.action_hash,
        verdict=verdict,
        decided_by="reviewer@example.test",
        request_id=request.request_id,
    )

    final = await graph.ainvoke(Command(resume=[decision.model_dump(mode="json")]), config=cfg)

    assert len(ports["action_executor"].executed) == (1 if submitted else 0)
    assert final["application"]["submitted"] is submitted
    # Either way the drafts are still there: a rejection discards nothing.
    assert len(_versions(factory)) == 2


async def test_a_fabricated_draft_stops_the_run_before_any_approval_is_requested(tmp_path):
    graph, ports, factory, _ = _build(
        tmp_path,
        proposals=lambda ids: {
            ArtifactType.RESUME: prep.fabricated_resume_proposal(ids),
            ArtifactType.COVER_LETTER: prep.cover_letter_proposal(ids),
        },
    )
    cfg = _config("fabricated")

    with pytest.raises(UntraceableClaim):
        await graph.ainvoke(
            {"user_id": str(fakes.USER_ID), "prepare_application": True}, config=cfg
        )

    state = (await graph.aget_state(cfg)).values
    assert not state.get("approval_requests")
    assert not state.get("pending_actions")
    assert _versions(factory) == []
    assert ports["approval_gate"].reviewed == []
    assert ports["action_executor"].executed == []
