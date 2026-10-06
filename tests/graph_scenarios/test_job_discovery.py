"""Scenario tests for the discovery half of the Job Search subgraph.

Providers are real `JobProvider` adapters reached through the policy gateway,
and postings land in a real `job_postings` table, so what is asserted here is
the row count a deployment would actually see.
"""

from uuid import uuid4

from personalos.policy import Decision
from personalos.providers import FakeJobProvider
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.job_discovery import (
    build_discovery_graph,
    posting_payload,
    session_factory,
    stored_postings,
)


async def discover(graph):
    """Run a discovery-only search on a fresh thread."""
    return await graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "prepare_application": False},
        config={"configurable": {"thread_id": f"t-{uuid4()}"}},
    )


def two_boards_one_posting() -> list[FakeJobProvider]:
    """The same opening on two boards, each with its own id and formatting."""
    return [
        FakeJobProvider("board_a", [posting_payload(id="a-17")]),
        FakeJobProvider(
            "board_b",
            [
                posting_payload(
                    id="b-90210",
                    company="ACME, Inc.",
                    title="Senior  Backend Engineer",
                    description="We need Python and Postgres experience.\n",
                    url="https://board-b.test/jobs/90210",
                )
            ],
        ),
    ]


async def test_same_posting_from_two_adapters_collapses_into_one_row(tmp_path):
    """The acceptance criterion: two adapters, one opening, one `job_postings` row."""
    factory = session_factory(tmp_path)
    adapters = two_boards_one_posting()
    graph, _, _ = build_discovery_graph(adapters, factory)

    final = await discover(graph)

    assert [len(adapter.search_calls) for adapter in adapters] == [1, 1]
    assert len(final["raw_postings"]) == 2
    assert len(final["normalized_postings"]) == 2
    assert len(final["deduplicated_postings"]) == 1
    assert len(final["duplicate_dedupe_keys"]) == 1

    rows = stored_postings(factory)
    assert len(rows) == 1
    # First provider to return it keeps the attribution.
    assert rows[0].source == "board_a"
    assert rows[0].source_job_id == "a-17"
    assert rows[0].dedupe_key == final["duplicate_dedupe_keys"][0]
    assert len(rows[0].description_hash) == 64
    assert final["persisted_postings"] == [
        {"job_posting_id": str(rows[0].id), "dedupe_key": rows[0].dedupe_key, "created": True}
    ]


async def test_repeating_a_search_does_not_add_rows(tmp_path):
    """A second run of the same search resolves to the rows the first created."""
    factory = session_factory(tmp_path)
    graph, _, _ = build_discovery_graph(two_boards_one_posting(), factory)

    first = await discover(graph)
    second = await discover(graph)

    assert len(stored_postings(factory)) == 1
    assert second["persisted_postings"] == [{**first["persisted_postings"][0], "created": False}]


async def test_a_posting_seen_first_on_another_board_in_a_later_run_collapses(tmp_path):
    """Cross-provider dedupe holds across runs, not only within one."""
    factory = session_factory(tmp_path)
    board_a, board_b = two_boards_one_posting()

    graph_a, _, _ = build_discovery_graph([board_a], factory)
    await discover(graph_a)
    graph_b, _, _ = build_discovery_graph([board_b], factory)
    later = await discover(graph_b)

    rows = stored_postings(factory)
    assert len(rows) == 1
    assert rows[0].source == "board_a"
    assert later["persisted_postings"][0]["created"] is False


async def test_distinct_postings_are_kept_apart(tmp_path):
    """Dedupe collapses the same opening, not merely the same employer."""
    factory = session_factory(tmp_path)
    adapters = [
        FakeJobProvider(
            "board_a",
            [
                posting_payload(id="a-1"),
                posting_payload(id="a-2", title="Staff Data Engineer"),
                posting_payload(id="a-3", description="A different python team entirely."),
            ],
        )
    ]
    graph, _, _ = build_discovery_graph(adapters, factory)

    await discover(graph)

    assert len(stored_postings(factory)) == 3


async def test_a_filtered_out_posting_is_still_recorded(tmp_path):
    """Recording precedes `hard_filter`, so a rejected posting is not rediscovered."""
    factory = session_factory(tmp_path)
    adapters = [FakeJobProvider("board_a", [posting_payload(company="Stealth Co")])]
    graph, _, _ = build_discovery_graph(adapters, factory)

    final = await discover(graph)

    assert final["filtered_postings"] == []
    assert [row.company for row in stored_postings(factory)] == ["Stealth Co"]


async def test_every_provider_call_is_an_allowed_read_external_decision(tmp_path):
    """Discovery reaches providers only as recorded, non-mutating policy decisions."""
    factory = session_factory(tmp_path)
    graph, _, decisions = build_discovery_graph(two_boards_one_posting(), factory)

    await discover(graph)

    assert [intent.tool_ref for intent, _ in decisions] == ["job_providers.search"] * 2
    assert [intent.arguments["provider"] for intent, _ in decisions] == ["board_a", "board_b"]
    assert all(not intent.mutating for intent, _ in decisions)
    assert all(decision.decision is Decision.ALLOW for _, decision in decisions)


async def test_one_provider_failing_does_not_lose_the_others(tmp_path):
    """A dead board is a recorded failure; the rest of the search still lands."""

    class DownProvider(FakeJobProvider):
        async def search(self, profile):
            raise RuntimeError("upstream 503")

    factory = session_factory(tmp_path)
    adapters = [DownProvider("deadboard"), FakeJobProvider("board_a", [posting_payload()])]
    graph, _, _ = build_discovery_graph(adapters, factory)

    final = await discover(graph)

    assert [failure["provider"] for failure in final["provider_failures"]] == ["deadboard"]
    assert len(stored_postings(factory)) == 1
