"""A job search over evidence loaded by `ingest-resume`, and what its matches may cite.

The fixture resume is ingested through the production path into SQLite, and
the Job Search subgraph is run with the matcher a real deployment builds
(`build_job_matcher` over `evidence_chunks`, the configured -- here rules --
semantic assessor). Every `JobMatch` it produces must cite only evidence ids
that are in the index, and a claimed match with no evidence behind it must
not survive as a match.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from personalos.bootstrap import build_job_matcher, build_model_gateway
from personalos.domain.job_search import (
    ClaimedMatch,
    JobSearchContractError,
    MatchedRequirement,
    MatchStrength,
    ScoredPosting,
    SemanticAssessment,
    ShortlistEntry,
)
from personalos.domain.models import EvidenceSourceType
from personalos.graphs.job_search import JobSearchGraph
from personalos.persistence.evidence import SqlEvidenceSource
from tests.fixtures import evidence_ingestion as ingestion
from tests.fixtures import job_matching as fx
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.job_discovery import session_factory


@pytest.fixture
async def indexed(tmp_path):
    """A database holding the ingested fixture resume, and the ids it was given."""
    factory = session_factory(tmp_path / "db")
    root = ingestion.allowed_root(tmp_path)
    result = await ingestion.ingestor(root, factory).ingest(fakes.USER_ID, "resume.md")
    return factory, {str(item.evidence_id) for item in result.document.evidence}, root


async def run(matcher):
    graph = JobSearchGraph(
        profile_store=fakes.FakeProfileStore(fx.profile()),
        providers=[fakes.FakeProvider(results=fx.raw_postings())],
        scorer=matcher,
        evidence_checker=matcher,
        packet_builder=fakes.FakePacketBuilder(),
        approval_gate=fakes.FakeApprovalGate(),
        action_executor=fakes.FakeActionExecutor(),
        application_store=fakes.FakeApplicationStore(),
        event_emitter=fakes.FakeEventEmitter(),
    ).build()
    final = await graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "prepare_application": False},
        config={"configurable": {"thread_id": f"t-{uuid4()}"}},
    )
    ranked = [ScoredPosting.model_validate(raw) for raw in final["ranked_postings"]]
    shortlist = [ShortlistEntry.model_validate(raw) for raw in final["shortlist"]]
    return ranked, shortlist


class OverclaimingAssessor:
    """Claims one real match, one citing an id not on record, and one citing nothing."""

    def __init__(self, real_id: str):
        self.real_id = real_id

    async def assess(self, posting, profile, evidence) -> SemanticAssessment:
        return SemanticAssessment(
            matched_requirements=(
                ClaimedMatch(requirement="python", evidence_id=self.real_id),
                ClaimedMatch(requirement="pytorch", evidence_id=str(uuid4())),
                ClaimedMatch(requirement="go", evidence_id=None, strength=MatchStrength.STRONG),
            )
        )


async def test_every_job_match_cites_only_evidence_in_the_index(indexed):
    factory, ids, _ = indexed
    matcher = build_job_matcher(build_model_gateway().semantic_assessor(), factory)

    ranked, shortlist = await run(matcher)

    cited = [item for scored in ranked for item in scored.match.matched_requirements]
    assert cited, "the ingested resume should ground at least one requirement"
    assert {item.evidence_id for item in cited} <= ids
    for scored in ranked:
        scored.match.ensure_grounded_in(ids)
    northwind = next(scored for scored in ranked if scored.posting.company == "Northwind")
    assert {item.requirement for item in northwind.match.matched_requirements} >= {
        "python",
        "postgres",
        "kafka",
    }
    assert any(entry.scored.posting.company == "Northwind" for entry in shortlist)
    for entry in shortlist:
        assert entry.evidence.grounded
        assert {ref.ref for ref in entry.evidence.citations} <= ids


async def test_a_claimed_match_with_no_evidence_does_not_survive_as_a_match(indexed):
    factory, ids, _ = indexed
    real_id = sorted(ids)[0]
    matcher = build_job_matcher(OverclaimingAssessor(real_id), factory)

    ranked, _ = await run(matcher)

    for scored in ranked:
        match = scored.match
        assert [item.evidence_id for item in match.matched_requirements] == [real_id]
        assert set(match.ungrounded_claims) == {"pytorch", "go"}
        assert {"pytorch", "go"} <= {gap.requirement for gap in match.missing_requirements}


async def test_a_matched_requirement_without_evidence_fails_validation(indexed):
    factory, ids, _ = indexed
    matcher = build_job_matcher(build_model_gateway().semantic_assessor(), factory)
    ranked, _ = await run(matcher)
    match = next(scored.match for scored in ranked if scored.match.matched_requirements)

    with pytest.raises(ValidationError, match="cite an evidence_id"):
        MatchedRequirement(
            requirement="python",
            evidence_id=" ",
            evidence_type=EvidenceSourceType.RESUME,
            strength=MatchStrength.STRONG,
        )

    fabricated = match.model_copy(
        update={
            "matched_requirements": (
                *match.matched_requirements,
                MatchedRequirement(
                    requirement="pytorch",
                    evidence_id=str(uuid4()),
                    evidence_type=EvidenceSourceType.RESUME,
                    strength=MatchStrength.STRONG,
                ),
            )
        }
    )
    with pytest.raises(JobSearchContractError, match="not on record"):
        fabricated.ensure_grounded_in(ids)


async def test_a_retired_chunk_is_no_longer_citable_by_a_new_match(indexed):
    """Retirement keeps an old artifact's citation resolvable, not usable as fresh evidence."""
    factory, ids, root = indexed
    loaded = await SqlEvidenceSource(factory).load(fakes.USER_ID)
    globex = next(record for record in loaded if "globex" in (record.source_ref or ""))
    ingestion.cite(factory, fakes.USER_ID, [globex.evidence_id])
    resume = root / "resume.md"
    resume.write_text(
        resume.read_text().replace("- Wrote the on-call runbook for the ingestion platform.\n", "")
    )
    await ingestion.ingestor(root, factory).ingest(fakes.USER_ID, "resume.md")
    active = {record.evidence_id for record in await SqlEvidenceSource(factory).load(fakes.USER_ID)}

    matcher = build_job_matcher(OverclaimingAssessor(globex.evidence_id), factory)
    ranked, _ = await run(matcher)

    assert globex.evidence_id not in active
    assert globex.evidence_id in ingestion.snapshot(factory)["chunks"]
    for scored in ranked:
        assert globex.evidence_id not in {
            item.evidence_id for item in scored.match.matched_requirements
        }
        scored.match.ensure_grounded_in(active)
