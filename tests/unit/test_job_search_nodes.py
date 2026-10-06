"""Per-node input/output contract tests for the Job Search subgraph.

One test class per node, each calling the node method directly with the
narrowest state that node is allowed to read, and asserting three things:

1. **Which keys it returns.** A node returns a *partial* state update, so the
   exact key set is part of its contract -- a node that quietly also wrote a
   neighbour's key would let the graph work by accident.
2. **The shape of the values.** Every returned value is rebuilt into its
   `personalos.domain.job_search` type, so a node that emitted a dict the next
   node cannot validate fails here rather than three steps downstream.
3. **Which ports it touched.** A node's reach is part of its contract too:
   `rank` must call nothing, and `prepare_application_packet` must not call the
   executor.

`test_every_graph_node_has_a_contract_test` at the bottom fails if a node is
added to the graph without a class here, so this file cannot silently fall
behind the pipeline.

End-to-end ordering and the conditional edges are covered in
`tests/graph_scenarios/test_job_search_graph.py`.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from personalos.domain.checkpoints import (
    ConditionKind,
    PendingCheckpoint,
    PendingCheckpointStatus,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApplicationPacket,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    ApprovalVerdict,
    EmittedEvent,
    EvidenceCheck,
    FilterRejection,
    FollowUpCheckpoint,
    FollowUpKind,
    JobSearchContractError,
    JobSearchEventType,
    NormalizedPosting,
    PersistedApplication,
    ProviderFailure,
    RawPosting,
    RecruiterMessage,
    RecruiterResponse,
    RefusalReason,
    RiskLevel,
    ScoredPosting,
    SearchProfile,
    ShortlistEntry,
)
from personalos.domain.models import ApplicationStatus, CommunicationEventClassification
from personalos.graphs import job_search as jsg
from personalos.graphs.job_search import (
    END,
    HANDLE_RECRUITER_RESPONSE,
    PERSIST_APPLICATION,
    PREPARE_APPLICATION_PACKET,
    REQUEST_APPROVAL,
    STAGE_RECRUITER_OUTREACH,
    STAGE_SUBMISSION,
    JobSearchGraph,
)
from tests.fixtures import job_search_fakes as fakes


def graph(**overrides) -> tuple[JobSearchGraph, dict]:
    """A `JobSearchGraph` instance over fake ports, uncompiled.

    Uncompiled on purpose: these tests call node methods directly, so the
    compiled graph's edges are deliberately not in the way.
    """
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [fakes.FakeProvider()],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": fakes.FakeApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
        # Pinned to the instant the fixtures are dated at. Requests built by
        # `fakes.approval_request` are raised at `fakes.NOW`, so a node reading
        # the wall clock would see them expire once the calendar caught up.
        "clock": lambda: fakes.NOW,
    }
    ports.update(overrides)
    return JobSearchGraph(**ports), ports


def dumped(*values) -> list[dict]:
    """JSON-compatible dict form, as the node would find it in state."""
    return [value.model_dump(mode="json") for value in values]


PROFILE_STATE = {"search_profile": fakes.profile().model_dump(mode="json")}


def _awaiting(intent, request=None) -> dict:
    """State as `approval_checkpoint` finds it: an action with its request raised."""
    request = request or fakes.approval_request(intent)
    return {
        "pending_actions": dumped(intent),
        "approval_requests": dumped(request),
        "approval_stage": STAGE_SUBMISSION,
    }


def _answered(intent, request, decision) -> dict:
    """State as `execute_approved_actions` finds it, after the interrupt resumed.

    `intent` is passed separately from `request` on purpose: the two disagreeing
    is exactly the situation the executor's hash recomputation exists to catch.
    """
    return {
        "pending_actions": dumped(intent),
        "approval_requests": dumped(request),
        "approvals": dumped(decision),
        "approval_stage": STAGE_SUBMISSION,
    }


# --- load_search_profile ------------------------------------------------------


class TestLoadSearchProfile:
    async def test_returns_only_the_profile_it_loaded(self):
        subgraph, ports = graph()

        update = await subgraph.load_search_profile({"user_id": str(fakes.USER_ID)})

        assert set(update) == {"search_profile"}
        profile = SearchProfile.model_validate(update["search_profile"])
        assert profile.user_id == fakes.USER_ID
        assert profile.profile_version == 3
        assert ports["profile_store"].calls == [fakes.USER_ID]

    async def test_an_inline_query_is_folded_into_the_profile_keywords(self):
        """One source of criteria: a query becomes a keyword, not a parallel input."""
        subgraph, _ports = graph()

        update = await subgraph.load_search_profile(
            {"user_id": str(fakes.USER_ID), "query": "kubernetes"}
        )

        profile = SearchProfile.model_validate(update["search_profile"])
        assert profile.keywords == ("python", "postgres", "kubernetes")

    async def test_a_query_already_in_the_profile_is_not_duplicated(self):
        subgraph, _ports = graph()

        update = await subgraph.load_search_profile(
            {"user_id": str(fakes.USER_ID), "query": "python"}
        )

        assert SearchProfile.model_validate(update["search_profile"]).keywords == (
            "python",
            "postgres",
        )

    @pytest.mark.parametrize("bad", [None, "", "not-a-uuid"])
    async def test_a_missing_or_malformed_user_id_is_a_typed_contract_error(self, bad):
        subgraph, ports = graph()

        with pytest.raises(JobSearchContractError):
            await subgraph.load_search_profile({"user_id": bad})
        assert ports["profile_store"].calls == []


# --- search_providers ---------------------------------------------------------


class TestSearchProviders:
    async def test_returns_raw_postings_and_an_empty_failure_list(self):
        subgraph, ports = graph()

        update = await subgraph.search_providers(PROFILE_STATE)

        assert set(update) == {"raw_postings", "provider_failures"}
        raws = [RawPosting.model_validate(row) for row in update["raw_postings"]]
        assert [raw.provider for raw in raws] == ["fakeboard"]
        assert update["provider_failures"] == []
        # The provider is handed the profile, not the raw request.
        assert isinstance(ports["providers"][0].calls[0], SearchProfile)

    async def test_a_raising_provider_becomes_a_recorded_failure_not_an_exception(self):
        subgraph, _ports = graph(providers=[fakes.FailingProvider(), fakes.FakeProvider()])

        update = await subgraph.search_providers(PROFILE_STATE)

        failures = [ProviderFailure.model_validate(row) for row in update["provider_failures"]]
        assert [failure.provider for failure in failures] == ["deadboard"]
        assert "upstream 503" in failures[0].error
        # The surviving provider's results are still there.
        assert len(update["raw_postings"]) == 1

    async def test_the_fan_out_is_bounded_per_run(self):
        """A provider returning an unbounded page cannot unbound the pipeline."""
        flood = [fakes.raw_posting(id=f"job-{i}") for i in range(jsg.MAX_POSTINGS_PER_RUN + 25)]
        subgraph, _ports = graph(providers=[fakes.FakeProvider("flood", flood)])

        update = await subgraph.search_providers(PROFILE_STATE)

        assert len(update["raw_postings"]) == jsg.MAX_POSTINGS_PER_RUN

    async def test_running_without_a_profile_is_a_typed_contract_error(self):
        subgraph, _ports = graph()

        with pytest.raises(JobSearchContractError, match="load_search_profile must run first"):
            await subgraph.search_providers({})


# --- normalize_jobs -----------------------------------------------------------


class TestNormalizeJobs:
    def test_returns_normalized_postings_only(self):
        subgraph, _ports = graph()

        update = subgraph.normalize_jobs({"raw_postings": dumped(fakes.raw_posting())})

        assert set(update) == {"normalized_postings"}
        postings = [NormalizedPosting.model_validate(row) for row in update["normalized_postings"]]
        assert postings[0].title == "Senior Backend Engineer"
        assert postings[0].company == "Acme"
        assert postings[0].source == "fakeboard"
        # Provenance survives normalization.
        assert postings[0].raw["id"] == "fb-1"

    def test_a_payload_missing_a_title_or_company_is_dropped_not_raised(self):
        """One bad row from a provider must not lose the other ninety-nine."""
        subgraph, _ports = graph()

        update = subgraph.normalize_jobs(
            {
                "raw_postings": dumped(
                    RawPosting(provider="fakeboard", payload={"description": "no title"}),
                    fakes.raw_posting(),
                )
            }
        )

        assert len(update["normalized_postings"]) == 1

    def test_provider_field_aliases_are_accepted(self):
        subgraph, _ports = graph()

        update = subgraph.normalize_jobs(
            {
                "raw_postings": dumped(
                    RawPosting(
                        provider="aliasboard",
                        payload={
                            "job_title": "Platform Engineer",
                            "employer": "Globex",
                            "job_url": "https://example.test/globex",
                            "min_salary": "130000",
                            "is_remote": True,
                        },
                    )
                )
            }
        )

        posting = NormalizedPosting.model_validate(update["normalized_postings"][0])
        assert (posting.title, posting.company) == ("Platform Engineer", "Globex")
        assert posting.salary_min == 130_000
        assert posting.remote is True

    def test_an_empty_input_yields_an_empty_output(self):
        subgraph, _ports = graph()

        assert subgraph.normalize_jobs({}) == {"normalized_postings": []}


# --- deduplicate --------------------------------------------------------------


class TestDeduplicate:
    def test_returns_survivors_and_the_keys_that_collided(self):
        subgraph, _ports = graph()
        first = fakes.posting(source="board_a")
        # Same company, title and description: the same opening, cross-posted.
        second = fakes.posting(source="board_b")

        update = subgraph.deduplicate({"normalized_postings": dumped(first, second)})

        assert set(update) == {"deduplicated_postings", "duplicate_dedupe_keys"}
        assert len(update["deduplicated_postings"]) == 1
        assert update["duplicate_dedupe_keys"] == [first.dedupe_key]
        # The first one seen is the one kept.
        kept = NormalizedPosting.model_validate(update["deduplicated_postings"][0])
        assert kept.source == "board_a"

    def test_different_openings_at_the_same_company_both_survive(self):
        subgraph, _ports = graph()

        update = subgraph.deduplicate(
            {
                "normalized_postings": dumped(
                    fakes.posting(title="Backend Engineer"),
                    fakes.posting(title="Frontend Engineer"),
                )
            }
        )

        assert len(update["deduplicated_postings"]) == 2
        assert update["duplicate_dedupe_keys"] == []


# --- persist_postings ---------------------------------------------------------


class TestPersistPostings:
    async def test_records_each_distinct_posting_and_returns_its_row(self):
        catalog = fakes.FakePostingCatalog()
        subgraph, _ports = graph(posting_catalog=catalog)
        first = fakes.posting(title="Backend Engineer")
        second = fakes.posting(title="Frontend Engineer")

        update = await subgraph.persist_postings({"deduplicated_postings": dumped(first, second)})

        assert set(update) == {"persisted_postings"}
        assert catalog.calls == [[first, second]]
        assert [row["dedupe_key"] for row in update["persisted_postings"]] == [
            first.dedupe_key,
            second.dedupe_key,
        ]
        assert all(row["created"] for row in update["persisted_postings"])

    async def test_a_posting_already_in_the_catalog_resolves_to_its_existing_row(self):
        catalog = fakes.FakePostingCatalog()
        subgraph, _ports = graph(posting_catalog=catalog)
        state = {"deduplicated_postings": dumped(fakes.posting())}

        first = await subgraph.persist_postings(state)
        again = await subgraph.persist_postings(state)

        assert again["persisted_postings"] == [{**first["persisted_postings"][0], "created": False}]

    async def test_without_a_catalog_nothing_is_recorded(self):
        subgraph, _ports = graph()

        update = await subgraph.persist_postings({"deduplicated_postings": dumped(fakes.posting())})

        assert update == {"persisted_postings": []}


# --- hard_filter --------------------------------------------------------------


class TestHardFilter:
    def test_returns_survivors_and_an_explained_rejection_for_each_drop(self):
        subgraph, _ports = graph()
        keep = fakes.posting(company="Acme")
        excluded = fakes.posting(company="Stealth Co")

        update = subgraph.hard_filter(
            {**PROFILE_STATE, "deduplicated_postings": dumped(keep, excluded)}
        )

        assert set(update) == {"filtered_postings", "filter_rejections"}
        assert len(update["filtered_postings"]) == 1
        rejection = FilterRejection.model_validate(update["filter_rejections"][0])
        assert rejection.company == "Stealth Co"
        assert rejection.dedupe_key == excluded.dedupe_key
        assert "exclusion list" in rejection.reason

    @pytest.mark.parametrize(
        ("posting_kwargs", "expected"),
        [
            ({"company": "Stealth Co"}, "exclusion list"),
            ({"remote": False}, "requires remote"),
            ({"salary_max": 60_000}, "below the floor"),
            ({"description": "java only", "skills": ("java",)}, "required skill"),
        ],
    )
    def test_each_stated_constraint_produces_its_own_reason(self, posting_kwargs, expected):
        subgraph, _ports = graph()

        update = subgraph.hard_filter(
            {**PROFILE_STATE, "deduplicated_postings": dumped(fakes.posting(**posting_kwargs))}
        )

        assert update["filtered_postings"] == []
        assert expected in update["filter_rejections"][0]["reason"]

    def test_an_unstated_salary_is_not_treated_as_a_violation(self):
        """Most postings omit a salary; treating silence as failure filters out the market."""
        subgraph, _ports = graph()

        update = subgraph.hard_filter(
            {
                **PROFILE_STATE,
                "deduplicated_postings": dumped(fakes.posting(salary_min=None, salary_max=None)),
            }
        )

        assert len(update["filtered_postings"]) == 1
        assert update["filter_rejections"] == []


# --- score_candidates ---------------------------------------------------------


class TestScoreCandidates:
    async def test_returns_one_scored_posting_per_survivor_with_components(self):
        subgraph, ports = graph()
        target = fakes.posting()

        update = await subgraph.score_candidates(
            {**PROFILE_STATE, "filtered_postings": dumped(target)}
        )

        assert set(update) == {"scored_postings"}
        scored = ScoredPosting.model_validate(update["scored_postings"][0])
        assert scored.score == 1.0
        assert scored.components
        assert scored.reasons
        assert scored.posting.dedupe_key == target.dedupe_key
        # The scorer receives the profile, so scoring criteria have one source.
        assert ports["scorer"].calls[0][1].user_id == fakes.USER_ID

    async def test_a_score_with_no_components_cannot_cross_the_boundary(self):
        """An unexplainable ranking is the failure mode the contract prevents."""
        with pytest.raises(ValidationError, match="scoring component"):
            ScoredPosting(posting=fakes.posting(), score=0.9, components={})

    async def test_scoring_nothing_returns_an_empty_list(self):
        subgraph, ports = graph()

        update = await subgraph.score_candidates({**PROFILE_STATE, "filtered_postings": []})

        assert update == {"scored_postings": []}
        assert ports["scorer"].calls == []


# --- evidence_check -----------------------------------------------------------


class TestEvidenceCheck:
    async def test_returns_one_check_per_scored_posting_keyed_by_dedupe_key(self):
        subgraph, ports = graph()
        target = fakes.posting()

        update = await subgraph.evidence_check(
            {**PROFILE_STATE, "scored_postings": dumped(fakes.scored(target))}
        )

        assert set(update) == {"evidence_checks"}
        check = EvidenceCheck.model_validate(update["evidence_checks"][0])
        assert check.dedupe_key == target.dedupe_key
        assert check.grounded is True
        assert check.citations
        assert len(ports["evidence_checker"].calls) == 1

    async def test_an_ungrounded_check_reports_what_was_unsupported(self):
        subgraph, _ports = graph(evidence_checker=fakes.FakeEvidenceChecker(always_grounded=False))

        update = await subgraph.evidence_check(
            {**PROFILE_STATE, "scored_postings": dumped(fakes.scored())}
        )

        check = EvidenceCheck.model_validate(update["evidence_checks"][0])
        assert check.grounded is False
        assert check.citations == ()
        assert check.unsupported_claims

    async def test_grounded_with_no_citations_cannot_cross_the_boundary(self):
        with pytest.raises(ValidationError, match="grounded=True with no citations"):
            EvidenceCheck(dedupe_key="k", grounded=True, citations=())


# --- rank ---------------------------------------------------------------------


class TestRank:
    def test_orders_best_first_and_calls_no_port(self):
        subgraph, ports = graph()
        low = fakes.scored(fakes.posting(title="Low"), score=0.4)
        high = fakes.scored(fakes.posting(title="High"), score=0.95)

        update = subgraph.rank({"scored_postings": dumped(low, high)})

        assert set(update) == {"ranked_postings"}
        assert [row["score"] for row in update["ranked_postings"]] == [0.95, 0.4]
        # A pure transformer: nothing was called.
        assert ports["scorer"].calls == []
        assert ports["evidence_checker"].calls == []

    def test_ties_break_on_the_content_derived_key_so_ranking_is_reproducible(self):
        subgraph, _ports = graph()
        a = fakes.scored(fakes.posting(company="Zeta"), score=0.7)
        b = fakes.scored(fakes.posting(company="Alpha"), score=0.7)

        forwards = subgraph.rank({"scored_postings": dumped(a, b)})
        backwards = subgraph.rank({"scored_postings": dumped(b, a)})

        assert forwards == backwards

    def test_ranking_nothing_returns_an_empty_list(self):
        subgraph, _ports = graph()

        assert subgraph.rank({}) == {"ranked_postings": []}


# --- shortlist ----------------------------------------------------------------


class TestShortlist:
    def test_returns_entries_ranked_from_one(self):
        subgraph, _ports = graph()
        first, second = fakes.posting(title="A"), fakes.posting(title="B")

        update = subgraph.shortlist(
            {
                **PROFILE_STATE,
                "ranked_postings": dumped(
                    fakes.scored(first, score=0.9), fakes.scored(second, score=0.8)
                ),
                "evidence_checks": dumped(
                    fakes.evidence_check(first), fakes.evidence_check(second)
                ),
            }
        )

        assert set(update) == {"shortlist"}
        entries = [ShortlistEntry.model_validate(row) for row in update["shortlist"]]
        assert [entry.rank for entry in entries] == [1, 2]
        assert entries[0].scored.posting.title == "A"
        assert entries[0].evidence.grounded is True

    def test_a_posting_below_the_score_floor_is_excluded(self):
        subgraph, _ports = graph()
        target = fakes.posting()

        update = subgraph.shortlist(
            {
                "search_profile": fakes.profile(min_score=0.8).model_dump(mode="json"),
                "ranked_postings": dumped(fakes.scored(target, score=0.7)),
                "evidence_checks": dumped(fakes.evidence_check(target)),
            }
        )

        assert update == {"shortlist": []}

    def test_an_ungrounded_posting_is_excluded(self):
        subgraph, _ports = graph()
        target = fakes.posting()

        update = subgraph.shortlist(
            {
                **PROFILE_STATE,
                "ranked_postings": dumped(fakes.scored(target, score=0.99)),
                "evidence_checks": dumped(fakes.evidence_check(target, grounded=False)),
            }
        )

        assert update == {"shortlist": []}

    def test_a_posting_with_no_evidence_check_at_all_is_excluded(self):
        """Absence of a verdict is not a pass."""
        subgraph, _ports = graph()

        update = subgraph.shortlist(
            {
                **PROFILE_STATE,
                "ranked_postings": dumped(fakes.scored(score=0.99)),
                "evidence_checks": [],
            }
        )

        assert update == {"shortlist": []}

    def test_the_shortlist_is_capped_by_the_profile(self):
        subgraph, _ports = graph()
        postings = [fakes.posting(title=f"Role {i}") for i in range(5)]

        update = subgraph.shortlist(
            {
                "search_profile": fakes.profile(max_shortlist=2).model_dump(mode="json"),
                "ranked_postings": dumped(*(fakes.scored(p, score=0.9) for p in postings)),
                "evidence_checks": dumped(*(fakes.evidence_check(p) for p in postings)),
            }
        )

        assert len(update["shortlist"]) == 2


# --- prepare_application_packet ----------------------------------------------


class TestPrepareApplicationPacket:
    async def test_returns_a_packet_and_a_proposed_submission_never_a_submission(self):
        """The node's whole contract: build locally, propose outwardly."""
        subgraph, ports = graph()

        update = await subgraph.prepare_application_packet(
            {**PROFILE_STATE, "shortlist": dumped(fakes.shortlist_entry())}
        )

        assert set(update) == {"application_packet", "pending_actions", "approval_stage"}
        packet = ApplicationPacket.model_validate(update["application_packet"])
        assert packet.posting.company == "Acme"

        intent = ActionIntent.model_validate(update["pending_actions"][0])
        assert intent.kind == ActionKind.SUBMIT_APPLICATION
        assert intent.summary
        assert intent.idempotency_key == f"submit-{packet.dedupe_key}"[:255]
        assert update["approval_stage"] == STAGE_SUBMISSION

        # Nothing was reviewed and nothing was executed by this node.
        assert ports["approval_gate"].reviewed == []
        assert ports["action_executor"].executed == []

    async def test_only_the_top_entry_is_prepared(self):
        subgraph, ports = graph()

        await subgraph.prepare_application_packet(
            {
                **PROFILE_STATE,
                "shortlist": dumped(
                    fakes.shortlist_entry(rank=1, score=0.9),
                    fakes.shortlist_entry(rank=2, score=0.8),
                ),
            }
        )

        assert len(ports["packet_builder"].calls) == 1
        assert ports["packet_builder"].calls[0].rank == 1

    async def test_the_proposed_idempotency_key_is_content_derived(self):
        """Retrying the run proposes the same submission, not a second one."""
        subgraph, _ports = graph()
        state = {**PROFILE_STATE, "shortlist": dumped(fakes.shortlist_entry())}

        first = await subgraph.prepare_application_packet(state)
        second = await subgraph.prepare_application_packet(state)

        assert (
            first["pending_actions"][0]["idempotency_key"]
            == second["pending_actions"][0]["idempotency_key"]
        )

    async def test_a_packet_without_a_resume_cannot_cross_the_boundary(self):
        with pytest.raises(ValidationError, match="must include a resume draft"):
            ApplicationPacket(dedupe_key="k", posting=fakes.posting(), artifacts=())


# --- request_approval ---------------------------------------------------------


class TestRequestApproval:
    def test_it_raises_a_reviewable_request_for_every_pending_action(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()

        update = subgraph.request_approval({"pending_actions": dumped(intent)})

        assert set(update) == {"approval_requests"}
        request = ApprovalRequest.model_validate(update["approval_requests"][0])
        assert request.action_id == intent.action_id
        assert request.action_hash == intent.fingerprint()
        assert request.target == intent.target
        assert request.summary == intent.summary
        assert request.risk == RiskLevel.HIGH
        assert request.requested_scopes == ("applications:submit", "artifacts:read")
        assert request.expires_at > request.requested_at
        # It only describes the action; nothing has been reviewed or executed.
        assert ports["approval_gate"].reviewed == []
        assert ports["action_executor"].executed == []

    def test_the_expiry_comes_from_the_action_kind_unless_overridden(self):
        intent = fakes.submit_intent()
        subgraph, _ = graph(clock=lambda: fakes.NOW)

        update = subgraph.request_approval({"pending_actions": dumped(intent)})
        request = ApprovalRequest.model_validate(update["approval_requests"][0])
        assert request.expires_at == fakes.NOW + timedelta(days=7)

        strict, _ = graph(clock=lambda: fakes.NOW, approval_ttl=timedelta(minutes=30))
        update = strict.request_approval({"pending_actions": dumped(intent)})
        request = ApprovalRequest.model_validate(update["approval_requests"][0])
        assert request.expires_at == fakes.NOW + timedelta(minutes=30)

    def test_requests_from_an_earlier_stage_are_kept(self):
        """The node runs twice per recruiter-handling run; requests accumulate."""
        subgraph, _ports = graph()
        earlier = fakes.approval_request()

        update = subgraph.request_approval(
            {
                "pending_actions": dumped(fakes.submit_intent()),
                "approval_requests": dumped(earlier),
            }
        )

        assert len(update["approval_requests"]) == 2
        assert update["approval_requests"][0]["request_id"] == str(earlier.request_id)

    def test_nothing_pending_raises_nothing(self):
        subgraph, _ports = graph()

        assert subgraph.request_approval({"pending_actions": []}) == {"approval_requests": []}


# --- approval_checkpoint ------------------------------------------------------


class TestApprovalCheckpoint:
    """Contract tests for the node that parks the run.

    The interrupt itself is not asserted here: `interrupt()` only works inside
    a runnable context, so calling this node directly can only cover the paths
    that *do not* pause. What the run looks like when it does pause, and what a
    reviewer is shown, is covered end-to-end in
    `tests/graph_scenarios/test_approval_interrupts.py`.
    """

    async def test_a_standing_approval_is_recorded_without_interrupting(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()

        update = await subgraph.approval_checkpoint(_awaiting(intent))

        assert set(update) == {"approvals"}
        decision = ApprovalDecision.model_validate(update["approvals"][0])
        assert decision.verdict == ApprovalVerdict.APPROVED
        assert decision.action_id == intent.action_id
        assert [i.action_id for i in ports["approval_gate"].reviewed] == [intent.action_id]
        # The checkpoint decides; it never acts.
        assert ports["action_executor"].executed == []

    @pytest.mark.parametrize("verdict", [ApprovalVerdict.REJECTED, ApprovalVerdict.APPROVED])
    async def test_a_binding_verdict_on_file_is_not_put_to_a_human(self, verdict):
        subgraph, _ports = graph(approval_gate=fakes.FakeApprovalGate(verdict))

        update = await subgraph.approval_checkpoint(_awaiting(fakes.submit_intent()))

        assert update["approvals"][0]["verdict"] == verdict.value

    async def test_earlier_stages_decisions_are_kept_not_overwritten(self):
        """The node runs twice per recruiter-handling run; the audit trail accumulates."""
        subgraph, _ports = graph()
        prior = ApprovalDecision(
            action_id=uuid4(),
            action_fingerprint="earlier",
            verdict=ApprovalVerdict.APPROVED,
            decided_by="reviewer@example.test",
        )

        update = await subgraph.approval_checkpoint(
            {**_awaiting(fakes.submit_intent()), "approvals": dumped(prior)}
        )

        assert len(update["approvals"]) == 2

    async def test_nothing_pending_means_nothing_reviewed(self):
        subgraph, ports = graph()

        update = await subgraph.approval_checkpoint({"pending_actions": []})

        assert update == {}
        assert ports["approval_gate"].reviewed == []

    async def test_an_action_with_no_request_on_file_is_a_contract_error(self):
        """`request_approval` must have run; without it there is no hash to check."""
        subgraph, _ports = graph()

        with pytest.raises(JobSearchContractError, match="no request on file"):
            await subgraph.approval_checkpoint(
                {"pending_actions": dumped(fakes.submit_intent()), "approval_requests": []}
            )

    async def test_an_action_intent_without_a_summary_cannot_be_built(self):
        """The summary is what a reviewer approves, so it is not optional."""
        with pytest.raises(ValidationError, match="human-readable summary"):
            ActionIntent(
                kind=ActionKind.SUBMIT_APPLICATION,
                target="https://example.test/acme",
                summary="   ",
                idempotency_key="key-that-is-long-enough",
            )

    async def test_an_action_intent_without_a_target_cannot_be_built(self):
        """A reviewer checks where the write lands separately from what it says."""
        with pytest.raises(ValidationError, match="human-readable summary"):
            ActionIntent(
                kind=ActionKind.SUBMIT_APPLICATION,
                target="  ",
                summary="Submit an application to Acme",
                idempotency_key="key-that-is-long-enough",
            )


# --- execute_approved_actions -------------------------------------------------


class TestExecuteApprovedActions:
    async def test_an_action_whose_approval_still_fits_it_is_redeemed(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()
        request = fakes.approval_request(intent)

        update = await subgraph.execute_approved_actions(
            _answered(intent, request, fakes.approval_decision(request))
        )

        assert set(update) == {"action_receipts", "approval_refusals"}
        assert update["approval_refusals"] == []
        receipt = ActionReceipt.model_validate(update["action_receipts"][0])
        assert receipt.ok is True
        assert [i.action_id for i, _ in ports["action_executor"].executed] == [intent.action_id]

    async def test_an_action_mutated_after_approval_is_refused(self):
        """The check the whole interrupt design exists for."""
        subgraph, ports = graph()
        approved = fakes.submit_intent()
        request = fakes.approval_request(approved)
        decision = fakes.approval_decision(request)
        # The reviewer said yes to `approved`; state now holds something else
        # under the same action id.
        mutated = approved.model_copy(update={"payload": {"dedupe_key": "somewhere:else"}})

        update = await subgraph.execute_approved_actions(_answered(mutated, request, decision))

        assert update["action_receipts"] == []
        assert ports["action_executor"].executed == []
        refusal = ApprovalRefusal.model_validate(update["approval_refusals"][0])
        assert refusal.reason == RefusalReason.HASH_MISMATCH
        assert refusal.approved_hash == request.action_hash
        assert refusal.recomputed_hash == mutated.fingerprint()
        assert refusal.suspicious is True

    async def test_redirecting_an_action_at_a_new_target_is_refused(self):
        """The target is part of the hash, so re-pointing an action voids its approval."""
        subgraph, ports = graph()
        approved = fakes.submit_intent()
        request = fakes.approval_request(approved)
        decision = fakes.approval_decision(request)
        redirected = approved.model_copy(update={"target": "https://evil.test/collect"})

        update = await subgraph.execute_approved_actions(_answered(redirected, request, decision))

        assert ports["action_executor"].executed == []
        assert (
            ApprovalRefusal.model_validate(update["approval_refusals"][0]).reason
            == RefusalReason.HASH_MISMATCH
        )

    async def test_an_approval_that_expired_while_the_run_was_parked_is_refused(self):
        intent = fakes.submit_intent()
        request = fakes.approval_request(intent)
        subgraph, ports = graph(clock=lambda: request.expires_at + timedelta(seconds=1))

        update = await subgraph.execute_approved_actions(
            _answered(intent, request, fakes.approval_decision(request))
        )

        assert ports["action_executor"].executed == []
        refusal = ApprovalRefusal.model_validate(update["approval_refusals"][0])
        assert refusal.reason == RefusalReason.EXPIRED
        assert refusal.suspicious is False

    @pytest.mark.parametrize(
        "verdict", [ApprovalVerdict.REJECTED, ApprovalVerdict.PENDING]
    )
    async def test_a_verdict_that_is_not_an_approval_yields_no_receipt(self, verdict):
        subgraph, ports = graph()
        intent = fakes.submit_intent()
        request = fakes.approval_request(intent)

        update = await subgraph.execute_approved_actions(
            _answered(intent, request, fakes.approval_decision(request, verdict))
        )

        assert update["action_receipts"] == []
        assert ports["action_executor"].executed == []
        assert (
            ApprovalRefusal.model_validate(update["approval_refusals"][0]).reason
            == RefusalReason.NOT_APPROVED
        )

    async def test_an_approval_bound_to_another_payload_does_not_authorize(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()
        request = fakes.approval_request(intent)
        decision = fakes.approval_decision(request, action_fingerprint="stale")

        update = await subgraph.execute_approved_actions(_answered(intent, request, decision))

        assert ports["action_executor"].executed == []
        assert (
            ApprovalRefusal.model_validate(update["approval_refusals"][0]).reason
            == RefusalReason.HASH_MISMATCH
        )

    async def test_a_decision_answering_a_different_request_is_refused(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()
        request = fakes.approval_request(intent)
        decision = fakes.approval_decision(request, request_id=uuid4())

        update = await subgraph.execute_approved_actions(_answered(intent, request, decision))

        assert ports["action_executor"].executed == []
        assert (
            ApprovalRefusal.model_validate(update["approval_refusals"][0]).reason
            == RefusalReason.MISDIRECTED_DECISION
        )

    async def test_an_action_that_never_reached_a_reviewer_is_refused(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()

        update = await subgraph.execute_approved_actions(
            {"pending_actions": dumped(intent), "approval_requests": [], "approvals": []}
        )

        assert ports["action_executor"].executed == []
        assert (
            ApprovalRefusal.model_validate(update["approval_refusals"][0]).reason
            == RefusalReason.NO_REQUEST
        )

    async def test_earlier_stages_receipts_are_kept_not_overwritten(self):
        subgraph, _ports = graph()
        intent = fakes.submit_intent()
        request = fakes.approval_request(intent)
        prior = ActionReceipt(action_id=uuid4(), ok=True)

        update = await subgraph.execute_approved_actions(
            {
                **_answered(intent, request, fakes.approval_decision(request)),
                "action_receipts": dumped(prior),
            }
        )

        assert len(update["action_receipts"]) == 2

    async def test_nothing_pending_means_nothing_executed(self):
        subgraph, ports = graph()

        assert await subgraph.execute_approved_actions({"pending_actions": []}) == {}
        assert ports["action_executor"].executed == []


# --- persist_application ------------------------------------------------------


class TestPersistApplication:
    async def test_an_executed_submission_persists_as_applied(self):
        subgraph, ports = graph()
        intent = fakes.submit_intent()
        packet = fakes.packet()

        update = await subgraph.persist_application(
            {
                **PROFILE_STATE,
                "application_packet": packet.model_dump(mode="json"),
                "pending_actions": dumped(intent),
                "action_receipts": dumped(
                    ActionReceipt(action_id=intent.action_id, ok=True, external_reference="ext-1")
                ),
            }
        )

        assert set(update) == {"application"}
        application = PersistedApplication.model_validate(update["application"])
        assert application.status == ApplicationStatus.APPLIED
        assert application.submitted is True
        assert application.external_reference == "ext-1"
        assert application.dedupe_key == packet.dedupe_key
        assert ports["application_store"].created[0].submitted is True

    async def test_an_unapproved_submission_persists_as_ready_to_apply(self):
        subgraph, _ports = graph()

        update = await subgraph.persist_application(
            {
                **PROFILE_STATE,
                "application_packet": fakes.packet().model_dump(mode="json"),
                "pending_actions": dumped(fakes.submit_intent()),
                "action_receipts": [],
            }
        )

        application = PersistedApplication.model_validate(update["application"])
        assert application.status == ApplicationStatus.READY_TO_APPLY
        assert application.submitted is False
        assert application.external_reference is None

    async def test_a_failed_submission_receipt_is_not_treated_as_submitted(self):
        subgraph, _ports = graph()
        intent = fakes.submit_intent()

        update = await subgraph.persist_application(
            {
                **PROFILE_STATE,
                "application_packet": fakes.packet().model_dump(mode="json"),
                "pending_actions": dumped(intent),
                "action_receipts": dumped(
                    ActionReceipt(action_id=intent.action_id, ok=False, detail="provider refused")
                ),
            }
        )

        assert update["application"]["submitted"] is False
        assert update["application"]["status"] == ApplicationStatus.READY_TO_APPLY.value

    async def test_a_receipt_for_a_different_kind_of_action_does_not_count_as_a_submission(self):
        """A redeemed recruiter reply must not make an application look submitted."""
        subgraph, _ports = graph()
        reply = ActionIntent(
            kind=ActionKind.SEND_RECRUITER_MESSAGE,
            target="recruiter@acme.test (thread msg-1)",
            summary="Reply to the recruiter",
            idempotency_key="reply-key-long-enough",
        )

        update = await subgraph.persist_application(
            {
                **PROFILE_STATE,
                "application_packet": fakes.packet().model_dump(mode="json"),
                "pending_actions": dumped(reply),
                "action_receipts": dumped(ActionReceipt(action_id=reply.action_id, ok=True)),
            }
        )

        assert update["application"]["submitted"] is False

    async def test_running_without_a_packet_is_a_typed_contract_error(self):
        subgraph, _ports = graph()

        with pytest.raises(
            JobSearchContractError, match="prepare_application_packet must run first"
        ):
            await subgraph.persist_application(PROFILE_STATE)


# --- emit_application_created -------------------------------------------------


def _application(*, submitted: bool = True) -> PersistedApplication:
    return PersistedApplication(
        application_id=fakes.APPLICATION_ID,
        job_posting_id=fakes.POSTING_ID,
        user_id=fakes.USER_ID,
        dedupe_key="acme|senior backend engineer|hash",
        status=ApplicationStatus.APPLIED if submitted else ApplicationStatus.READY_TO_APPLY,
        submitted=submitted,
        external_reference="ext-1" if submitted else None,
    )


class TestEmitApplicationCreated:
    async def test_emits_application_created_with_a_dedupe_key(self):
        subgraph, ports = graph()

        update = await subgraph.emit_application_created(
            {"application": _application().model_dump(mode="json")}
        )

        assert set(update) == {"emitted_events"}
        events = [EmittedEvent.model_validate(row) for row in update["emitted_events"]]
        assert [event.type for event in events] == [JobSearchEventType.APPLICATION_CREATED]
        assert events[0].aggregate_id == fakes.APPLICATION_ID
        assert events[0].dedupe_key == f"application.created:{fakes.APPLICATION_ID}"
        assert events[0].payload["submitted"] is True
        # Emitted through the outbox port, not published directly.
        assert ports["event_emitter"].types() == [JobSearchEventType.APPLICATION_CREATED.value]

    async def test_an_unsubmitted_application_also_emits_a_rejection_event(self):
        subgraph, ports = graph()

        update = await subgraph.emit_application_created(
            {"application": _application(submitted=False).model_dump(mode="json")}
        )

        assert [row["type"] for row in update["emitted_events"]] == [
            JobSearchEventType.APPLICATION_CREATED.value,
            JobSearchEventType.APPLICATION_SUBMISSION_REJECTED.value,
        ]
        assert len(ports["event_emitter"].events) == 2

    async def test_running_without_an_application_is_a_typed_contract_error(self):
        subgraph, _ports = graph()

        with pytest.raises(JobSearchContractError, match="persist_application must run first"):
            await subgraph.emit_application_created({})


# --- handle_recruiter_response ------------------------------------------------


def recruiter_graph(messages=None, **overrides):
    """A graph with the recruiter branch wired on."""
    return graph(
        recruiter_inbox=fakes.FakeRecruiterInbox(
            messages if messages is not None else [fakes.recruiter_message()]
        ),
        recruiter_classifier=fakes.FakeRecruiterClassifier(),
        **overrides,
    )


APPLIED_STATE = {"application": _application().model_dump(mode="json")}

#: The config a run is invoked with. `thread_id` is not state -- it is the
#: identity of the thread state is stored under -- so a node that schedules a
#: durable wait has to read it from here.
THREAD_CONFIG = {"configurable": {"thread_id": "thread-under-test"}}


def _pending_wait(**overrides) -> PendingCheckpoint:
    """A durable wait as the monitor hands one back to the graph."""
    wait = PendingCheckpoint.for_follow_up(
        application_id=fakes.APPLICATION_ID,
        kind=FollowUpKind.NO_RESPONSE,
        due_at=fakes.NOW + timedelta(days=7),
        reason="no recruiter response received yet",
        thread_id="thread-under-test",
        created_at=fakes.NOW,
    )
    return wait.model_copy(update=overrides) if overrides else wait


class TestHandleRecruiterResponse:
    async def test_classifies_records_and_proposes_a_reply(self):
        subgraph, ports = recruiter_graph()

        update = await subgraph.handle_recruiter_response(APPLIED_STATE)

        assert set(update) == {
            "recruiter_messages",
            "recruiter_responses",
            "emitted_events",
            "pending_actions",
            "approval_stage",
        }
        message = RecruiterMessage.model_validate(update["recruiter_messages"][0])
        assert message.provider_message_id == "msg-1"
        response = RecruiterResponse.model_validate(update["recruiter_responses"][0])
        assert response.classification == CommunicationEventClassification.INTERVIEW_INVITE
        assert response.implied_status == ApplicationStatus.INTERVIEWING
        assert response.requires_reply is True

        # Recorded against the application, in this subgraph.
        assert ports["application_store"].recruiter_responses == [(fakes.APPLICATION_ID, response)]

        # The reply is proposed, not sent.
        intent = ActionIntent.model_validate(update["pending_actions"][0])
        assert intent.kind == ActionKind.SEND_RECRUITER_MESSAGE
        assert intent.payload["in_reply_to"] == "msg-1"
        assert update["approval_stage"] == STAGE_RECRUITER_OUTREACH
        assert ports["action_executor"].executed == []

    async def test_a_response_needing_no_reply_proposes_no_action(self):
        subgraph, _ports = recruiter_graph(
            [fakes.recruiter_message("msg-2", "Unfortunately we are not moving forward.")]
        )

        update = await subgraph.handle_recruiter_response(APPLIED_STATE)

        assert update["pending_actions"] == []
        response = RecruiterResponse.model_validate(update["recruiter_responses"][0])
        assert response.classification == CommunicationEventClassification.REJECTION

    async def test_an_empty_inbox_records_nothing_and_proposes_nothing(self):
        subgraph, ports = recruiter_graph([])

        update = await subgraph.handle_recruiter_response(APPLIED_STATE)

        assert update["recruiter_responses"] == []
        assert update["pending_actions"] == []
        assert ports["application_store"].recruiter_responses == []

    async def test_earlier_events_are_preserved_alongside_the_new_ones(self):
        subgraph, _ports = recruiter_graph()
        earlier = EmittedEvent(
            type=JobSearchEventType.APPLICATION_CREATED, aggregate_id=fakes.APPLICATION_ID
        )

        update = await subgraph.handle_recruiter_response(
            {**APPLIED_STATE, "emitted_events": dumped(earlier)}
        )

        assert [row["type"] for row in update["emitted_events"]] == [
            JobSearchEventType.APPLICATION_CREATED.value,
            JobSearchEventType.RECRUITER_RESPONSE_RECORDED.value,
        ]

    async def test_running_without_an_application_is_a_typed_contract_error(self):
        subgraph, _ports = recruiter_graph()

        with pytest.raises(JobSearchContractError, match="persist_application must run first"):
            await subgraph.handle_recruiter_response({})


# --- create_follow_up_checkpoint ---------------------------------------------


class TestCreateFollowUpCheckpoint:
    async def test_an_interview_invite_schedules_prep_and_a_reply_reminder(self):
        subgraph, ports = recruiter_graph()
        response = RecruiterResponse(
            provider_message_id="msg-1",
            classification=CommunicationEventClassification.INTERVIEW_INVITE,
            occurred_at=fakes.NOW,
            implied_status=ApplicationStatus.INTERVIEWING,
            requires_reply=True,
        )

        update = await subgraph.create_follow_up_checkpoint(
            {**APPLIED_STATE, "recruiter_responses": dumped(response)}
        )

        assert set(update) == {
            "follow_up_checkpoints",
            "pending_checkpoints",
            "emitted_events",
        }
        checkpoints = [
            FollowUpCheckpoint.model_validate(row) for row in update["follow_up_checkpoints"]
        ]
        assert {c.kind for c in checkpoints} == {
            FollowUpKind.INTERVIEW_PREP,
            FollowUpKind.AWAITING_CANDIDATE_REPLY,
        }
        assert all(c.application_id == fakes.APPLICATION_ID for c in checkpoints)
        assert all(c.reason for c in checkpoints)
        # Recorded here, not in a separate Calendar subgraph.
        assert len(ports["application_store"].follow_ups) == 2
        assert [row["type"] for row in update["emitted_events"]] == [
            JobSearchEventType.FOLLOW_UP_SCHEDULED.value
        ] * 2

    async def test_no_responses_at_all_schedules_the_quiet_period_check_in(self):
        subgraph, ports = recruiter_graph()

        update = await subgraph.create_follow_up_checkpoint(
            {**APPLIED_STATE, "recruiter_responses": []}
        )

        checkpoint = FollowUpCheckpoint.model_validate(update["follow_up_checkpoints"][0])
        assert checkpoint.kind == FollowUpKind.NO_RESPONSE
        assert ports["application_store"].follow_ups == [checkpoint]

    async def test_a_rejection_schedules_nothing(self):
        subgraph, ports = recruiter_graph()
        response = RecruiterResponse(
            provider_message_id="msg-2",
            classification=CommunicationEventClassification.REJECTION,
            occurred_at=fakes.NOW,
        )

        update = await subgraph.create_follow_up_checkpoint(
            {**APPLIED_STATE, "recruiter_responses": dumped(response)}
        )

        assert update["follow_up_checkpoints"] == []
        assert ports["application_store"].follow_ups == []

    async def test_repeated_signals_do_not_pile_up_duplicate_reminders(self):
        subgraph, _ports = recruiter_graph()
        responses = [
            RecruiterResponse(
                provider_message_id=f"msg-{i}",
                classification=CommunicationEventClassification.ACTION_REQUIRED,
                occurred_at=fakes.NOW,
                requires_reply=True,
            )
            for i in range(3)
        ]

        update = await subgraph.create_follow_up_checkpoint(
            {**APPLIED_STATE, "recruiter_responses": dumped(*responses)}
        )

        assert [row["kind"] for row in update["follow_up_checkpoints"]] == [
            FollowUpKind.AWAITING_CANDIDATE_REPLY.value
        ]

    async def test_a_checkpoint_without_a_reason_cannot_be_built(self):
        with pytest.raises(ValidationError, match="record why it exists"):
            FollowUpCheckpoint(
                application_id=fakes.APPLICATION_ID,
                kind=FollowUpKind.NO_RESPONSE,
                due_at=datetime.utcnow() + timedelta(days=1),
                reason="  ",
            )

    async def test_a_scheduled_reminder_is_also_stored_as_a_durable_wait(self):
        """The in-run checkpoint and the stored one are written together.

        They are different objects because they have different lifetimes: the
        `FollowUpCheckpoint` is part of this run's story, and the
        `PendingCheckpoint` is what makes something come back in seven days
        after the run, the worker and the process are gone.
        """
        scheduler = fakes.FakePendingCheckpointScheduler()
        subgraph, _ports = recruiter_graph(checkpoint_scheduler=scheduler)

        update = await subgraph.create_follow_up_checkpoint(
            {**APPLIED_STATE, "recruiter_responses": []}, THREAD_CONFIG
        )

        in_run = FollowUpCheckpoint.model_validate(update["follow_up_checkpoints"][0])
        stored = PendingCheckpoint.model_validate(update["pending_checkpoints"][0])
        assert scheduler.waits() == [stored]
        assert stored.status == PendingCheckpointStatus.PENDING
        assert stored.trigger_at == in_run.due_at
        # The three things the state copy has no use for and the row cannot
        # work without.
        assert stored.condition.kind == ConditionKind.RECRUITER_RESPONSE_RECEIVED
        assert stored.expires_at > stored.trigger_at
        assert stored.thread_id == "thread-under-test"

    async def test_a_replayed_super_step_rejoins_the_existing_wait(self):
        """Re-entering the branch must not stack a second reminder.

        The node is re-run whenever the graph replays that super-step, and each
        pass proposes the same wait. The scheduler deduplicates on
        `dedupe_key`, so the second pass gets the first wait back -- including
        its original trigger date, because re-dating it on every replay is the
        subtle way of never firing.
        """
        scheduler = fakes.FakePendingCheckpointScheduler()
        subgraph, _ports = recruiter_graph(checkpoint_scheduler=scheduler)
        state = {**APPLIED_STATE, "recruiter_responses": []}

        first = await subgraph.create_follow_up_checkpoint(state, THREAD_CONFIG)
        second = await subgraph.create_follow_up_checkpoint(state, THREAD_CONFIG)

        assert len(scheduler.waits()) == 1
        assert second["pending_checkpoints"] == first["pending_checkpoints"]

    async def test_scheduling_a_wait_on_a_run_with_no_thread_is_refused(self):
        """A wait that cannot name a thread has nowhere to come back to.

        Refused where it is built rather than stored and never fired: an
        unactionable row reads, in SQL, exactly like a follow-up that is coming.
        """
        subgraph, _ports = recruiter_graph(
            checkpoint_scheduler=fakes.FakePendingCheckpointScheduler()
        )

        with pytest.raises(JobSearchContractError, match="nowhere to resume"):
            await subgraph.create_follow_up_checkpoint(
                {**APPLIED_STATE, "recruiter_responses": []}, {"configurable": {}}
            )

    async def test_no_scheduler_wired_means_no_durable_wait_and_no_failure(self):
        """The durable half is optional, and switching it off changes nothing else."""
        subgraph, ports = recruiter_graph()

        update = await subgraph.create_follow_up_checkpoint(
            {**APPLIED_STATE, "recruiter_responses": []}, THREAD_CONFIG
        )

        assert update["pending_checkpoints"] == []
        assert len(update["follow_up_checkpoints"]) == 1
        assert len(ports["application_store"].follow_ups) == 1


class TestDraftFollowUp:
    """The node a triggered durable wait lands on."""

    async def test_a_fired_wait_becomes_a_proposed_message_and_nothing_else(self):
        """It proposes. It does not send -- it holds no executor and has no edge to one.

        Which matters more here than anywhere else in this graph: this run was
        started by a monitor on a schedule, with no human anywhere near it.
        """
        subgraph, ports = recruiter_graph()
        wait = _pending_wait()

        update = await subgraph.draft_follow_up(
            {**APPLIED_STATE, "fired_checkpoints": dumped(wait)}
        )

        assert set(update) == {
            "pending_actions",
            "approval_stage",
            "emitted_events",
            # Cleared, because it is an input channel and thread state outlives
            # the run: a fired checkpoint left in state would route the next
            # ordinary search on this thread into the follow-up path.
            "fired_checkpoints",
        }
        assert update["fired_checkpoints"] == []
        intent = ActionIntent.model_validate(update["pending_actions"][0])
        assert intent.kind == ActionKind.SEND_RECRUITER_MESSAGE
        assert intent.payload["checkpoint_id"] == str(wait.checkpoint_id)
        assert intent.payload["unmet_condition"] == wait.condition.kind.value
        # Routed back through the approval triple by stage, like every other
        # outward-facing action this graph proposes.
        assert update["approval_stage"] == STAGE_RECRUITER_OUTREACH
        assert ports["action_executor"].executed == []

    async def test_it_announces_the_trigger_so_a_fired_wait_is_distinguishable(self):
        """`follow_up_triggered` is emitted only by the waits that actually came due.

        Most scheduled follow-ups never reach this node: the recruiter replies
        and the wait closes silently. That is why this is a separate event type
        from `follow_up_scheduled` rather than a second copy of it.
        """
        subgraph, ports = recruiter_graph()
        wait = _pending_wait()

        update = await subgraph.draft_follow_up(
            {**APPLIED_STATE, "fired_checkpoints": dumped(wait)}
        )

        event = EmittedEvent.model_validate(update["emitted_events"][-1])
        assert event.type == JobSearchEventType.FOLLOW_UP_TRIGGERED
        assert event.dedupe_key == f"follow_up_triggered:{wait.checkpoint_id}"
        assert ports["event_emitter"].types() == [
            JobSearchEventType.FOLLOW_UP_TRIGGERED.value
        ]

    async def test_a_wait_for_another_application_is_refused(self):
        """The draft is written from this thread's state, so the two must agree.

        A mismatch means a message about the wrong application, which is the
        one outcome worse than no follow-up at all.
        """
        subgraph, _ports = recruiter_graph()
        stranger = _pending_wait().model_copy(update={"application_id": uuid4()})

        with pytest.raises(JobSearchContractError, match="but thread state holds"):
            await subgraph.draft_follow_up(
                {**APPLIED_STATE, "fired_checkpoints": dumped(stranger)}
            )

    async def test_it_requires_an_application_to_follow_up_on(self):
        """Ordered like every other node that reads a value it did not compute."""
        subgraph, _ports = recruiter_graph()

        with pytest.raises(JobSearchContractError, match="no application in state"):
            await subgraph.draft_follow_up({"fired_checkpoints": dumped(_pending_wait())})


# --- Routers ------------------------------------------------------------------


class TestRouters:
    def test_shortlist_router_skips_the_optional_packet_step_unless_asked(self):
        subgraph, _ports = graph()
        shortlisted = {"shortlist": dumped(fakes.shortlist_entry())}

        assert subgraph.route_after_shortlist({**shortlisted, "prepare_application": True}) == (
            PREPARE_APPLICATION_PACKET
        )
        assert subgraph.route_after_shortlist({**shortlisted}) == END
        assert subgraph.route_after_shortlist({"prepare_application": True, "shortlist": []}) == END

    def test_approval_router_returns_to_the_stage_that_proposed_the_action(self):
        subgraph, _ports = graph()
        intent = fakes.submit_intent()
        approved = dumped(
            ApprovalDecision(
                action_id=intent.action_id,
                action_fingerprint=intent.fingerprint(),
                verdict=ApprovalVerdict.APPROVED,
                decided_by="reviewer@example.test",
            )
        )

        assert (
            subgraph.route_after_approval(
                {
                    "approval_stage": STAGE_SUBMISSION,
                    "pending_actions": dumped(intent),
                    "approvals": approved,
                }
            )
            == PERSIST_APPLICATION
        )
        assert (
            subgraph.route_after_approval(
                {
                    "approval_stage": STAGE_RECRUITER_OUTREACH,
                    "pending_actions": dumped(intent),
                    "approvals": approved,
                }
            )
            == END
        )

    def test_an_unanswered_request_ends_the_run_whatever_the_stage(self):
        subgraph, _ports = graph()
        intent = fakes.submit_intent()
        pending = dumped(
            ApprovalDecision(
                action_id=intent.action_id,
                action_fingerprint=intent.fingerprint(),
                verdict=ApprovalVerdict.PENDING,
                decided_by="reviewer@example.test",
            )
        )

        assert (
            subgraph.route_after_approval(
                {
                    "approval_stage": STAGE_SUBMISSION,
                    "pending_actions": dumped(intent),
                    "approvals": pending,
                }
            )
            == END
        )

    def test_a_pending_verdict_from_an_earlier_stage_does_not_stall_a_later_one(self):
        subgraph, _ports = graph()
        stale = ApprovalDecision(
            action_id=uuid4(),
            action_fingerprint="earlier",
            verdict=ApprovalVerdict.PENDING,
            decided_by="reviewer@example.test",
        )
        current = fakes.submit_intent()

        assert (
            subgraph.route_after_approval(
                {
                    "approval_stage": STAGE_SUBMISSION,
                    "pending_actions": dumped(current),
                    "approvals": dumped(
                        stale,
                        ApprovalDecision(
                            action_id=current.action_id,
                            action_fingerprint=current.fingerprint(),
                            verdict=ApprovalVerdict.APPROVED,
                            decided_by="reviewer@example.test",
                        ),
                    ),
                }
            )
            == PERSIST_APPLICATION
        )

    def test_emit_router_enters_the_recruiter_branch_only_when_an_inbox_is_wired(self):
        without, _ = graph()
        with_inbox, _ = recruiter_graph()

        assert without.route_after_emit(APPLIED_STATE) == END
        assert with_inbox.route_after_emit(APPLIED_STATE) == HANDLE_RECRUITER_RESPONSE

    def test_follow_up_router_sends_an_owed_reply_back_through_the_approval_node(self):
        subgraph, _ports = recruiter_graph()
        reply = ActionIntent(
            kind=ActionKind.SEND_RECRUITER_MESSAGE,
            target="recruiter@acme.test (thread msg-1)",
            summary="Reply to the recruiter",
            idempotency_key="reply-key-long-enough",
        )

        assert (
            subgraph.route_after_follow_up({"pending_actions": dumped(reply)})
            == REQUEST_APPROVAL
        )
        assert subgraph.route_after_follow_up({"pending_actions": []}) == END


# --- Coverage guard -----------------------------------------------------------


#: Which contract test class covers which graph node. Explicit rather than
#: derived from the node name: `approval_checkpoint_for_external_submission` is
#: deliberately named for what it gates, not for its method, and a naming
#: convention that happened to work for the other fourteen would break on it.
NODE_TEST_CLASSES = {
    jsg.LOAD_SEARCH_PROFILE: TestLoadSearchProfile,
    jsg.SEARCH_PROVIDERS: TestSearchProviders,
    jsg.NORMALIZE_JOBS: TestNormalizeJobs,
    jsg.DEDUPLICATE: TestDeduplicate,
    jsg.PERSIST_POSTINGS: TestPersistPostings,
    jsg.HARD_FILTER: TestHardFilter,
    jsg.SCORE_CANDIDATES: TestScoreCandidates,
    jsg.EVIDENCE_CHECK: TestEvidenceCheck,
    jsg.RANK: TestRank,
    jsg.SHORTLIST: TestShortlist,
    jsg.PREPARE_APPLICATION_PACKET: TestPrepareApplicationPacket,
    jsg.REQUEST_APPROVAL: TestRequestApproval,
    jsg.APPROVAL_CHECKPOINT: TestApprovalCheckpoint,
    jsg.EXECUTE_APPROVED_ACTIONS: TestExecuteApprovedActions,
    jsg.PERSIST_APPLICATION: TestPersistApplication,
    jsg.EMIT_APPLICATION_CREATED: TestEmitApplicationCreated,
    jsg.HANDLE_RECRUITER_RESPONSE: TestHandleRecruiterResponse,
    jsg.CREATE_FOLLOW_UP_CHECKPOINT: TestCreateFollowUpCheckpoint,
    jsg.DRAFT_FOLLOW_UP: TestDraftFollowUp,
}


async def test_every_graph_node_has_a_contract_test():
    """A node added to the graph without a contract test fails here.

    The acceptance criterion is "a node unit test exists for each node", which
    is only durable if adding a node breaks something. This reads the compiled
    graph's own node list rather than a hand-maintained one, so the map above
    is what has to be kept honest, and forgetting to is a test failure.
    """
    subgraph, _ports = recruiter_graph()
    compiled = subgraph.build()

    graph_nodes = {
        name for name in compiled.get_graph().nodes if name not in ("__start__", "__end__")
    }

    assert graph_nodes == set(NODE_TEST_CLASSES), (
        f"untested node(s): {sorted(graph_nodes - set(NODE_TEST_CLASSES))}; "
        f"mapped but not in the graph: {sorted(set(NODE_TEST_CLASSES) - graph_nodes)}"
    )
