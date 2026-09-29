"""Scenario tests for the Job Search subgraph, end to end against fake providers.

The acceptance criterion this file exists for is the first test: a full happy
path (discover -> score -> shortlist -> prepare -> approve -> persist) asserting
the final application state. The rest cover the branches that decide whether
that path is taken at all -- an unapproved submission, a pending reviewer, an
empty shortlist -- plus the recruiter-response and follow-up branches, which
live in this subgraph rather than a Communications or Calendar one.

Every run goes through the compiled graph, not through node methods directly:
the ordering, the conditional edges and the state round-trip are as much of the
contract as the node bodies are, and calling nodes by hand would test none of
them. Per-node input/output contracts are in
`tests/unit/test_job_search_nodes.py`.
"""

from uuid import uuid4

import pytest

from personalos.domain.job_search import (
    ActionKind,
    ApprovalVerdict,
    FollowUpKind,
    JobSearchEventType,
    NormalizedPosting,
)
from personalos.domain.models import ApplicationStatus, CommunicationEventClassification
from personalos.graphs.job_search import (
    STAGE_RECRUITER_OUTREACH,
    STAGE_SUBMISSION,
    JobSearchGraph,
)
from tests.fixtures import job_search_fakes as fakes


def build(**overrides):
    """Compile a subgraph over fake ports, returning the graph and its fakes.

    The fakes are returned alongside the graph because most assertions here are
    about what the graph *did* to its ports -- which action was redeemed, which
    events were enqueued -- not only about the state it ended in.
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
    }
    ports.update(overrides)
    return JobSearchGraph(**ports).build(), ports


async def run(graph, **state):
    """Invoke the graph on a fresh thread with the given input state."""
    initial = {"user_id": str(fakes.USER_ID), "prepare_application": True}
    initial.update(state)
    return await graph.ainvoke(initial, config={"configurable": {"thread_id": f"t-{uuid4()}"}})


async def test_happy_path_discovers_scores_shortlists_prepares_approves_and_persists():
    """The acceptance path, asserted on the final application state.

    One posting is discovered, scores 1.0 on keyword overlap, is grounded
    against the candidate's record, is shortlisted at rank 1, has a packet
    built, is approved, is submitted, and lands as an APPLIED application
    carrying the external reference the submission returned.
    """
    graph, ports = build()

    final = await run(graph)

    # Discovery.
    assert ports["profile_store"].calls == [fakes.USER_ID]
    assert len(final["raw_postings"]) == 1
    assert final["provider_failures"] == []
    assert len(final["normalized_postings"]) == 1

    # Selection.
    assert final["filter_rejections"] == []
    assert final["scored_postings"][0]["score"] == 1.0
    assert final["evidence_checks"][0]["grounded"] is True
    assert [entry["rank"] for entry in final["shortlist"]] == [1]

    # Preparation and approval.
    # `dedupe_key` is derived, not stored, so it is rebuilt to compare.
    shortlisted = NormalizedPosting.model_validate(final["shortlist"][0]["scored"]["posting"])
    assert final["application_packet"]["dedupe_key"] == shortlisted.dedupe_key
    assert [intent["kind"] for intent in final["pending_actions"]] == [
        ActionKind.SUBMIT_APPLICATION.value
    ]
    assert [decision["verdict"] for decision in final["approvals"]] == [
        ApprovalVerdict.APPROVED.value
    ]
    assert len(ports["action_executor"].executed) == 1
    submitted_intent, _decision = ports["action_executor"].executed[0]
    assert submitted_intent.kind == ActionKind.SUBMIT_APPLICATION

    # The final application state: the assertion this test exists for.
    application = final["application"]
    assert application["application_id"] == str(fakes.APPLICATION_ID)
    assert application["user_id"] == str(fakes.USER_ID)
    assert application["status"] == ApplicationStatus.APPLIED.value
    assert application["submitted"] is True
    assert application["external_reference"] == "ext-submit_application"
    assert len(application["artifact_version_ids"]) == 1

    # And the event that says so.
    assert ports["event_emitter"].types() == [JobSearchEventType.APPLICATION_CREATED.value]
    created = ports["event_emitter"].events[0]
    assert created.aggregate_id == fakes.APPLICATION_ID
    assert created.dedupe_key == f"application.created:{fakes.APPLICATION_ID}"
    assert created.payload["submitted"] is True


async def test_no_node_executes_a_submission_without_going_through_the_approval_gate():
    """Every redeemed action was reviewed first, and only reviewed ones were redeemed.

    The structural invariant of this subgraph, asserted on the fakes rather than
    on the graph's shape: the executor's call log must be a subset of the gate's.
    """
    graph, ports = build()

    await run(graph)

    reviewed = {intent.action_id for intent in ports["approval_gate"].reviewed}
    executed = {intent.action_id for intent, _ in ports["action_executor"].executed}
    assert executed
    assert executed <= reviewed


async def test_a_rejected_submission_is_persisted_as_prepared_but_not_sent():
    """A rejected submission still persists the packet's work, unsent.

    `READY_TO_APPLY` is the lifecycle state for "prepared, not sent"; discarding
    the packet instead would mean rebuilding it on the next run.
    """
    graph, ports = build(approval_gate=fakes.FakeApprovalGate(ApprovalVerdict.REJECTED))

    final = await run(graph)

    assert ports["action_executor"].executed == []
    application = final["application"]
    assert application["status"] == ApplicationStatus.READY_TO_APPLY.value
    assert application["submitted"] is False
    assert application["external_reference"] is None
    assert ports["event_emitter"].types() == [
        JobSearchEventType.APPLICATION_CREATED.value,
        JobSearchEventType.APPLICATION_SUBMISSION_REJECTED.value,
    ]


async def test_an_unanswered_approval_pauses_the_run_at_the_interrupt():
    """With no decision on file, the run parks instead of proceeding.

    Nothing is executed and nothing is persisted: the checkpointed thread is
    what a later human answer resumes from. The interrupt itself, and what
    happens on the way back, are covered in
    `tests/graph_scenarios/test_approval_interrupts.py`.
    """
    graph, ports = build(approval_gate=fakes.NoStandingApprovalGate())

    final = await run(graph)

    assert ports["action_executor"].executed == []
    assert ports["application_store"].created == []
    assert ports["event_emitter"].events == []
    assert final.get("application") is None
    assert final.get("approvals") is None
    assert [interrupt.value["stage"] for interrupt in final["__interrupt__"]] == [
        STAGE_SUBMISSION
    ]


async def test_an_approval_bound_to_a_different_payload_does_not_authorize_the_action():
    """A verdict whose fingerprint does not match the intent is not an approval.

    This is what stops an approval of "apply to the backend role at Acme" from
    clearing a mutated payload that applies somewhere else.
    """
    graph, ports = build(
        approval_gate=fakes.FakeApprovalGate(
            ApprovalVerdict.APPROVED, fingerprint_override="not-the-right-fingerprint"
        )
    )

    final = await run(graph)

    assert ports["action_executor"].executed == []
    assert final["application"]["submitted"] is False


async def test_a_discovery_only_run_stops_at_the_shortlist():
    """Without `prepare_application`, the optional packet step is skipped entirely."""
    graph, ports = build()

    final = await run(graph, prepare_application=False)

    assert len(final["shortlist"]) == 1
    assert final.get("application_packet") is None
    assert ports["packet_builder"].calls == []
    assert ports["approval_gate"].reviewed == []
    assert final.get("application") is None


async def test_an_ungrounded_match_is_not_shortlisted_however_well_it_scored():
    """Scoring well is not enough: an ungrounded match is a coincidence in the text."""
    graph, ports = build(evidence_checker=fakes.FakeEvidenceChecker(always_grounded=False))

    final = await run(graph)

    assert final["scored_postings"][0]["score"] == 1.0
    assert final["evidence_checks"][0]["grounded"] is False
    assert final["shortlist"] == []
    assert ports["packet_builder"].calls == []
    assert final.get("application") is None


async def test_a_dead_provider_is_recorded_and_the_run_continues_on_the_others():
    """One failing job board must not fail the search.

    `provider_failures` is what makes a thin result set distinguishable from a
    genuinely empty one.
    """
    graph, _ports = build(providers=[fakes.FailingProvider(), fakes.FakeProvider()])

    final = await run(graph)

    assert [failure["provider"] for failure in final["provider_failures"]] == ["deadboard"]
    assert "upstream 503" in final["provider_failures"][0]["error"]
    assert len(final["shortlist"]) == 1
    assert final["application"]["submitted"] is True


async def test_the_same_role_cross_posted_to_two_boards_is_shortlisted_once():
    """Dedup keys off content, so the same opening on two boards collapses to one."""
    graph, _ports = build(
        providers=[
            fakes.FakeProvider("board_a", [fakes.raw_posting("board_a", id="a-1")]),
            fakes.FakeProvider("board_b", [fakes.raw_posting("board_b", id="b-9")]),
        ]
    )

    final = await run(graph)

    assert len(final["normalized_postings"]) == 2
    assert len(final["deduplicated_postings"]) == 1
    assert len(final["duplicate_dedupe_keys"]) == 1
    assert len(final["shortlist"]) == 1


async def test_hard_filters_run_before_scoring_and_record_why_each_posting_was_dropped():
    """A non-negotiable cannot be out-scored, and a rejection is always explained."""
    graph, ports = build(
        providers=[
            fakes.FakeProvider(
                "fakeboard",
                [
                    fakes.raw_posting(id="ok", company="Acme"),
                    fakes.raw_posting(id="excluded", company="Stealth Co"),
                    fakes.raw_posting(id="onsite", company="Onsite Inc", remote=False),
                    fakes.raw_posting(id="lowpay", company="Cheap Ltd", salary_max=60_000),
                ],
            )
        ]
    )

    final = await run(graph)

    assert len(final["filtered_postings"]) == 1
    reasons = {row["company"]: row["reason"] for row in final["filter_rejections"]}
    assert "exclusion list" in reasons["Stealth Co"]
    assert "remote" in reasons["Onsite Inc"]
    assert "below the floor" in reasons["Cheap Ltd"]
    # Filtered postings never reach the scorer at all.
    assert [call[0].company for call in ports["scorer"].calls] == ["Acme"]


async def test_ranking_is_best_first_and_the_shortlist_is_capped_by_the_profile():
    """Ranking orders on score; the shortlist is bounded by `max_shortlist`."""
    graph, _ports = build(
        profile_store=fakes.FakeProfileStore(fakes.profile(max_shortlist=2)),
        providers=[
            fakes.FakeProvider(
                "fakeboard",
                [
                    # Scores 1.0: mentions both keywords.
                    fakes.raw_posting(
                        id="both", company="Both Co", description="python and postgres"
                    ),
                    # Scores 0.5: mentions python only, and so does its skills list.
                    fakes.raw_posting(
                        id="one", company="One Co", description="python only", skills=["python"]
                    ),
                    # Also 0.5, but a different company, so it is the capped-out third.
                    fakes.raw_posting(
                        id="one-b", company="Alt Co", description="python only", skills=["python"]
                    ),
                ],
            )
        ],
    )

    final = await run(graph)

    ranked = [(row["posting"]["company"], row["score"]) for row in final["ranked_postings"]]
    assert [score for _company, score in ranked] == sorted(
        (score for _company, score in ranked), reverse=True
    )
    assert ranked[0] == ("Both Co", 1.0)
    assert [entry["rank"] for entry in final["shortlist"]] == [1, 2]


async def test_a_posting_below_the_score_floor_is_never_shortlisted():
    """A run that finds nothing good shortlists nothing, not its least-bad option."""
    graph, _ports = build(
        profile_store=fakes.FakeProfileStore(fakes.profile(min_score=0.75)),
        providers=[
            fakes.FakeProvider(
                "fakeboard",
                [fakes.raw_posting(description="python only", skills=["python"])],
            )
        ],
    )

    final = await run(graph)

    assert final["ranked_postings"][0]["score"] == 0.5
    assert final["shortlist"] == []


# --- Recruiter lifecycle branch ----------------------------------------------


async def test_a_recruiter_interview_invite_is_recorded_and_a_reply_is_routed_for_approval():
    """The recruiter branch: classify, record, schedule, and propose a reply.

    The reply is proposed, not sent, and it reaches the executor only by going
    back through the approval checkpoint -- the same chokepoint the submission
    went through.
    """
    inbox = fakes.FakeRecruiterInbox([fakes.recruiter_message()])
    graph, ports = build(
        recruiter_inbox=inbox,
        recruiter_classifier=fakes.FakeRecruiterClassifier(),
    )

    final = await run(graph)

    # The message was classified and recorded against the application.
    assert inbox.calls == [fakes.APPLICATION_ID]
    recorded = ports["application_store"].recruiter_responses
    assert [app_id for app_id, _ in recorded] == [fakes.APPLICATION_ID]
    assert recorded[0][1].classification == CommunicationEventClassification.INTERVIEW_INVITE

    # Follow-up checkpoints were created in this subgraph, not a Calendar one.
    kinds = {checkpoint.kind for checkpoint in ports["application_store"].follow_ups}
    assert kinds == {FollowUpKind.INTERVIEW_PREP, FollowUpKind.AWAITING_CANDIDATE_REPLY}

    # The owed reply went back through the approval node and was then redeemed.
    reviewed_kinds = [intent.kind for intent in ports["approval_gate"].reviewed]
    assert reviewed_kinds == [ActionKind.SUBMIT_APPLICATION, ActionKind.SEND_RECRUITER_MESSAGE]
    executed_kinds = [intent.kind for intent, _ in ports["action_executor"].executed]
    assert executed_kinds == [ActionKind.SUBMIT_APPLICATION, ActionKind.SEND_RECRUITER_MESSAGE]

    # Both stages' decisions survive in the final state.
    assert len(final["approvals"]) == 2
    assert final["approval_stage"] == STAGE_RECRUITER_OUTREACH
    assert ports["event_emitter"].types() == [
        JobSearchEventType.APPLICATION_CREATED.value,
        JobSearchEventType.RECRUITER_RESPONSE_RECORDED.value,
        JobSearchEventType.FOLLOW_UP_SCHEDULED.value,
        JobSearchEventType.FOLLOW_UP_SCHEDULED.value,
    ]


async def test_a_rejection_needs_no_reply_and_schedules_no_follow_up():
    """A rejection closes the loop: nothing is owed, so nothing is proposed."""
    graph, ports = build(
        recruiter_inbox=fakes.FakeRecruiterInbox(
            [fakes.recruiter_message("msg-2", "Unfortunately we are not moving forward.")]
        ),
        recruiter_classifier=fakes.FakeRecruiterClassifier(),
    )

    final = await run(graph)

    response = ports["application_store"].recruiter_responses[0][1]
    assert response.classification == CommunicationEventClassification.REJECTION
    assert ports["application_store"].follow_ups == []
    assert final["pending_actions"] == []
    assert [intent.kind for intent, _ in ports["action_executor"].executed] == [
        ActionKind.SUBMIT_APPLICATION
    ]


async def test_silence_from_the_recruiter_schedules_a_check_in():
    """An empty inbox is itself a signal: schedule the quiet-period check-in."""
    graph, ports = build(
        recruiter_inbox=fakes.FakeRecruiterInbox([]),
        recruiter_classifier=fakes.FakeRecruiterClassifier(),
    )

    final = await run(graph)

    assert final["recruiter_responses"] == []
    assert [c.kind for c in ports["application_store"].follow_ups] == [FollowUpKind.NO_RESPONSE]
    assert final["follow_up_checkpoints"][0]["reason"] == "no recruiter response received yet"


async def test_without_a_recruiter_inbox_the_run_ends_after_the_created_event():
    """The recruiter branch is opt-in: no inbox wired, no branch."""
    graph, ports = build()

    final = await run(graph)

    assert final.get("recruiter_responses") is None
    assert ports["application_store"].follow_ups == []


# --- Construction and checkpointing ------------------------------------------


async def test_construction_requires_every_port_and_at_least_one_provider():
    with pytest.raises(ValueError, match="requires at least one JobBoardProvider"):
        JobSearchGraph(
            profile_store=fakes.FakeProfileStore(),
            providers=[],
            scorer=fakes.FakeScorer(),
            evidence_checker=fakes.FakeEvidenceChecker(),
            packet_builder=fakes.FakePacketBuilder(),
            approval_gate=fakes.FakeApprovalGate(),
            action_executor=fakes.FakeActionExecutor(),
            application_store=fakes.FakeApplicationStore(),
            event_emitter=fakes.FakeEventEmitter(),
        )

    with pytest.raises(ValueError, match="approval_gate"):
        JobSearchGraph(
            profile_store=fakes.FakeProfileStore(),
            providers=[fakes.FakeProvider()],
            scorer=fakes.FakeScorer(),
            evidence_checker=fakes.FakeEvidenceChecker(),
            packet_builder=fakes.FakePacketBuilder(),
            approval_gate=None,
            action_executor=fakes.FakeActionExecutor(),
            application_store=fakes.FakeApplicationStore(),
            event_emitter=fakes.FakeEventEmitter(),
        )


async def test_a_recruiter_inbox_without_a_classifier_is_rejected_at_construction():
    """Half the recruiter branch is not a usable configuration."""
    with pytest.raises(ValueError, match="must be provided together"):
        JobSearchGraph(
            profile_store=fakes.FakeProfileStore(),
            providers=[fakes.FakeProvider()],
            scorer=fakes.FakeScorer(),
            evidence_checker=fakes.FakeEvidenceChecker(),
            packet_builder=fakes.FakePacketBuilder(),
            approval_gate=fakes.FakeApprovalGate(),
            action_executor=fakes.FakeActionExecutor(),
            application_store=fakes.FakeApplicationStore(),
            event_emitter=fakes.FakeEventEmitter(),
            recruiter_inbox=fakes.FakeRecruiterInbox(),
        )


async def test_the_run_is_retrievable_from_the_checkpointer_afterwards():
    """State round-trips through the checkpointer, which is what a paused
    approval later resumes from."""
    graph, _ports = build()
    config = {"configurable": {"thread_id": "t-checkpointed"}}

    await graph.ainvoke({"user_id": str(fakes.USER_ID), "prepare_application": True}, config=config)
    snapshot = await graph.aget_state(config)

    assert snapshot.values["application"]["status"] == ApplicationStatus.APPLIED.value
    assert snapshot.values["search_profile"]["user_id"] == str(fakes.USER_ID)


async def test_a_missing_user_id_fails_before_any_provider_is_called():
    from personalos.domain.job_search import JobSearchContractError

    graph, ports = build()

    with pytest.raises(JobSearchContractError, match="user_id is required"):
        await graph.ainvoke(
            {"prepare_application": True}, config={"configurable": {"thread_id": "t-nouser"}}
        )
    assert ports["providers"][0].calls == []


# --- Supervisor seam ----------------------------------------------------------


async def test_the_supervisor_delegates_to_this_subgraph_through_the_runner_adapter():
    """`JobSearchSubgraphRunner` satisfies the port the Supervisor delegates through.

    This is the seam that makes the Job Search subgraph reachable as *the*
    domain subgraph: the Supervisor classifies, plans a bounded DAG, and hands
    it to a `JobSubgraphRunner` -- and this adapter turns that hand-off into an
    initial `JobSearchState` without the Supervisor knowing any of this graph's
    shape.
    """
    from personalos.domain.routing import RouteDecision, RouteDomain
    from personalos.graphs.job_search import JobSearchSubgraphRunner
    from personalos.graphs.supervisor import SupervisorGraph

    class StubClassifier:
        def classify(self, message):
            return RouteDecision(domain=RouteDomain.JOB, confidence=0.95)

    subgraph, ports = build()
    runner = JobSearchSubgraphRunner(subgraph, user_id=fakes.USER_ID, prepare_application=True)
    supervisor = SupervisorGraph(StubClassifier(), runner).build()

    final = await supervisor.ainvoke(
        {"message": "find me a python job"},
        config={"configurable": {"thread_id": "t-supervisor-seam"}},
    )

    result = final["result"]
    assert result["goal"].startswith("job_search:")
    assert len(result["shortlist"]) == 1
    assert result["application"]["status"] == ApplicationStatus.APPLIED.value
    assert result["emitted_events"][0]["type"] == JobSearchEventType.APPLICATION_CREATED.value
    # The Supervisor's message reached this graph as a search keyword, which is
    # the whole of the adapter's translation job.
    assert ports["providers"][0].calls[0].keywords[-1] == "find me a python job"
