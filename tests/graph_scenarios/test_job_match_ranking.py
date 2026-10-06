"""Golden ranking for the Job Search subgraph over `HybridJobMatcher`.

A fixed synthetic profile, seven fixed postings and a scripted model answer for
each go in; the ranked order, scores, recommendations and shortlist that come
out are compared to `tests/fixtures/golden/job_match_ranking.json`. The model
is mocked and nothing else in the path is random, so any diff here is a real
change in scoring behaviour -- update the golden file deliberately, with the
weight or rule change that caused it.
"""

from uuid import uuid4

import pytest

from personalos.domain.job_search import ScoredPosting, ShortlistEntry
from personalos.graphs.job_search import JobSearchGraph
from personalos.retrieval.job_matching import HybridJobMatcher
from tests.fixtures import job_matching as fx
from tests.fixtures import job_search_fakes as fakes


async def run(raw_postings):
    """One full run on a fresh graph, matcher and thread."""
    matcher = HybridJobMatcher(
        evidence_source=fx.StaticEvidenceSource(), assessor=fx.ScriptedAssessor()
    )
    graph = JobSearchGraph(
        profile_store=fakes.FakeProfileStore(fx.profile()),
        providers=[fakes.FakeProvider(results=raw_postings)],
        scorer=matcher,
        evidence_checker=matcher,
        packet_builder=fakes.FakePacketBuilder(),
        approval_gate=fakes.FakeApprovalGate(),
        action_executor=fakes.FakeActionExecutor(),
        application_store=fakes.FakeApplicationStore(),
        event_emitter=fakes.FakeEventEmitter(),
    ).build()
    return await graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "prepare_application": False},
        config={"configurable": {"thread_id": f"t-{uuid4()}"}},
    )


def summarize(final) -> dict:
    """Reduce a final state to the shape the golden file records."""
    ranked = [ScoredPosting.model_validate(raw) for raw in final["ranked_postings"]]
    shortlist = [ShortlistEntry.model_validate(raw) for raw in final["shortlist"]]
    return {
        "ranked": [
            {
                "company": item.posting.company,
                "score": item.score,
                "recommendation": item.match.recommendation.value,
            }
            for item in ranked
        ],
        "shortlist": [entry.scored.posting.company for entry in shortlist],
    }


async def test_ranking_matches_the_golden_fixture():
    final = await run(fx.raw_postings())

    assert final["filter_rejections"] == []
    assert summarize(final) == fx.golden_ranking()


@pytest.mark.parametrize("order", ["declared", "reversed", "rotated"])
async def test_ranked_order_is_stable_across_runs_and_input_orders(order):
    postings = fx.raw_postings()
    if order == "reversed":
        postings.reverse()
    elif order == "rotated":
        postings = postings[3:] + postings[:3]

    first = summarize(await run(postings))
    second = summarize(await run(postings))

    assert first == second == fx.golden_ranking()


async def test_the_shortlist_never_recommends_a_skipped_or_ungrounded_posting():
    final = await run(fx.raw_postings())

    known = {record.evidence_id for record in fx.EVIDENCE}
    for raw in final["shortlist"]:
        entry = ShortlistEntry.model_validate(raw)
        assert entry.scored.match.recommendation.value != "SKIP"
        assert entry.evidence.grounded
        entry.scored.match.ensure_grounded_in(known)

    companies = [raw["scored"]["posting"]["company"] for raw in final["shortlist"]]
    assert not {"Hooli", "Vandelay"} & set(companies)
