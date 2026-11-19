"""Per-node input/output contract tests for the Communications subgraph.

One class per registered node, each calling the node on a graph whose node
methods check their output against `personalos.graphs.contracts`. The graph
runs over the real Gmail tool path (`tests.fixtures.communications`): policy
gateway with the production per-node allowlists, MCP invoker, the Google MCP
server, and a fake Google API -- plus the real recruiter-event store on SQLite.

The acceptance criteria -- routing, approval and hash binding, the claim
validator, prompt injection, recruiter events exactly once -- are in
`tests/adversarial/test_communications_boundaries.py`.
"""

from typing import Any

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from personalos.domain.communications import (
    TEMPLATE_REPLY_BODY,
    CommunicationsTask,
    InboxItem,
    TriageCategory,
    TriageSource,
    parse_communications_request,
)
from personalos.domain.job_search import ActionIntent, ActionKind, RefusalReason
from personalos.graphs import communications as csg
from personalos.graphs.contracts import COMMUNICATIONS_GRAPH, contracts_for
from personalos.models.communications import ResilientTriager
from tests.fixtures import communications as cf
from tests.fixtures.node_contracts import covers_node, enforce_contracts

CONTRACTS = contracts_for(COMMUNICATIONS_GRAPH)


@pytest.fixture
def world():
    w = cf.build_world_in_process()
    w.graph = enforce_contracts(w.graph, COMMUNICATIONS_GRAPH)
    return w


@pytest.fixture
def candidate(world):
    """An APPLIED application at Acme, and its candidate."""
    return cf.applied_application(world.factory)


def parsed(world, request: str, **state: Any) -> dict[str, Any]:
    return {"request": request, **state, **world.graph.parse_request({"request": request})}


async def loaded(world, request: str, user_id) -> dict[str, Any]:
    state = parsed(world, request, user_id=str(user_id))
    state.update(await world.graph.load_thread(state))
    return state


async def linked(world, request: str, user_id) -> dict[str, Any]:
    state = await loaded(world, request, user_id)
    state.update(await world.graph.link_application(state))
    return state


async def drafted(world, user_id) -> dict[str, Any]:
    state = await linked(world, "draft a reply to thread t-acme", user_id)
    state.update(await world.graph.draft_reply(state))
    return state


@covers_node(COMMUNICATIONS_GRAPH, csg.PARSE_REQUEST)
class TestParseRequest:
    @pytest.mark.parametrize(
        ("request_text", "task"),
        [
            ("triage my inbox", CommunicationsTask.TRIAGE),
            ("summarize the email thread from Acme", CommunicationsTask.SUMMARIZE_THREAD),
            ("what are the action items in thread t-1?", CommunicationsTask.EXTRACT_ACTION_ITEMS),
            ("draft a reply to the recruiter email", CommunicationsTask.DRAFT_REPLY),
            ("hmm", CommunicationsTask.TRIAGE),
        ],
    )
    def test_names_one_task_from_a_closed_set(self, world, request_text, task):
        update = world.graph.parse_request({"request": request_text})

        assert update["comms_request"]["task"] == task.value
        assert update["errors"] == []

    def test_reads_a_thread_id_and_a_sender_but_not_a_time_as_a_sender(self):
        assert parse_communications_request("summarize thread:t-acme").thread_id == "t-acme"
        assert parse_communications_request("reply to the email from Acme").sender_hint == "Acme"
        hint = parse_communications_request("summarize the email from this morning").sender_hint
        assert hint is None

    def test_a_thread_id_the_caller_passes_wins_over_the_text(self, world):
        update = world.graph.parse_request(
            {"request": "summarize thread t-other", "thread_id": "t-acme"}
        )

        assert update["comms_request"]["thread_id"] == "t-acme"

    def test_a_malformed_thread_id_is_dropped_not_passed_to_gmail(self):
        request = parse_communications_request("summarize", thread_id="../../etc")

        assert request.thread_id is None


@covers_node(COMMUNICATIONS_GRAPH, csg.TRIAGE_INBOX)
class TestTriageInbox:
    async def test_triages_only_in_scope_mail_into_typed_results(self, world):
        world.add_mail("m-1")
        world.add_mail("m-out", thread_id="t-out", labelled=False, subject="Lunch?")
        world.add_mail("m-mine", thread_id="t-acme", labels=("SENT",), subject="Re: Interview")

        update = await world.graph.triage_inbox(parsed(world, "triage my inbox"))

        (result,) = update["triage"]
        assert result["message_id"] == "m-1"
        assert result["category"] == TriageCategory.INTERVIEW.value
        assert result["source"] == TriageSource.RULES.value
        # Headers and snippets only: no message body was read.
        assert not any(
            "/messages/m-1" in c.url and c.params.get("format") == "full"
            for c in world.google.api.calls
        )

    async def test_a_triager_that_raises_yields_needs_review(self, world):
        class Broken:
            async def triage(self, item):
                raise RuntimeError("provider down")

        world.graph.triager = Broken()
        world.add_mail("m-1")

        (result,) = (await world.graph.triage_inbox(parsed(world, "triage")))["triage"]

        assert result["category"] == TriageCategory.NEEDS_REVIEW.value
        assert result["suggested_action"] == "review"

    async def test_model_output_outside_the_schema_falls_back_to_needs_review(self):
        class Proposer:
            async def propose(self, item):
                return {
                    "category": "recruiter",
                    "urgency": "high",
                    "suggested_action": "reply",
                    "confidence": 0.9,
                    "send_to": "attacker@evil.example",
                }

        item = InboxItem(message_id="m-1", subject="hi")
        result = await ResilientTriager(Proposer()).triage(item)

        assert result.category is TriageCategory.NEEDS_REVIEW
        assert result.source is TriageSource.FALLBACK


@covers_node(COMMUNICATIONS_GRAPH, csg.LOAD_THREAD)
class TestLoadThread:
    async def test_summarizes_with_message_ids_and_keeps_no_body(self, world, candidate):
        user_id, _ = candidate
        world.add_mail("m-1")
        world.add_mail(
            "m-2", subject="Re: Interview", body="Thursday works. Please confirm.", minute=5
        )

        update = await loaded(world, "summarize thread t-acme", user_id)

        summary = update["digest"]["summary"]
        assert summary["message_ids"] == ["m-1", "m-2"]
        assert all(set(p["message_ids"]) <= {"m-1", "m-2"} for p in summary["key_points"])
        assert all(m["body"] == "" for m in update["thread"]["messages"])
        assert all(e["message"]["body"] == "" for e in update["recruiter_events"])

    async def test_links_are_recorded_inert_and_never_fetched(self, world, candidate):
        world.add_mail("m-1")

        update = await loaded(world, "summarize thread t-acme", candidate[0])

        (message,) = update["thread"]["messages"]
        assert message["links"] == ["https://acme.example/careers/interview-prep"]
        assert not any("acme.example" in call.url for call in world.google.api.calls)

    async def test_a_thread_outside_the_mailbox_scope_is_not_read(self, world, candidate):
        world.add_mail("m-out", thread_id="t-out", labelled=False)

        update = await loaded(world, "summarize thread t-out", candidate[0])

        assert update["thread"] is None
        assert "ThreadNotInScope" in update["errors"][0]
        assert not any(
            c.url.endswith("/messages/m-out") and c.params.get("format") == "full"
            for c in world.google.api.calls
        )

    async def test_finds_the_thread_by_sender_when_none_is_named(self, world, candidate):
        world.add_mail("m-1")

        update = await loaded(world, "summarize the email from priya", candidate[0])

        assert update["thread"]["thread_id"] == "t-acme"


@covers_node(COMMUNICATIONS_GRAPH, csg.LINK_APPLICATION)
class TestLinkApplication:
    async def test_records_the_event_against_its_application_and_consumes_it(
        self, world, candidate
    ):
        user_id, application_id = candidate
        world.add_mail("m-1")

        update = await linked(world, "summarize thread t-acme", user_id)

        (record,) = update["recruiter_event_records"]
        assert record["application_id"] == str(application_id)
        assert record["created"] is True and record["transitioned"] is True
        assert update["application_id"] == str(application_id)
        assert update["recruiter_events"] == []

    async def test_with_no_events_it_records_nothing(self, world):
        update = await world.graph.link_application({"recruiter_events": []})

        assert update["recruiter_event_records"] == []
        assert update["application_id"] is None


@covers_node(COMMUNICATIONS_GRAPH, csg.DRAFT_REPLY)
class TestDraftReply:
    async def test_proposes_a_send_to_the_sender_and_keeps_a_draft(self, world, candidate):
        user_id, application_id = candidate
        world.add_mail("m-1")

        update = await drafted(world, user_id)

        (intent,) = update["pending_actions"]
        assert intent["kind"] == ActionKind.SEND_EMAIL.value
        assert intent["requested_by"] == csg.requester(csg.DRAFT_REPLY)
        assert intent["payload"]["recipient"] == cf.RECRUITER
        assert intent["payload"]["in_reply_to"] == "m-1"
        draft = update["reply_draft"]
        assert draft["gmail_draft_id"] in world.google.api.drafts
        assert draft["artifact_version_id"] is not None
        assert draft["application_id"] == str(application_id)
        assert update["approval_stage"] == csg.STAGE_SEND_REPLY

    async def test_without_a_drafter_it_says_so_and_proposes_nothing(self, world, candidate):
        world.graph.reply_drafter = None
        world.add_mail("m-1")
        state = await linked(world, "draft a reply to thread t-acme", candidate[0])

        update = await world.graph.draft_reply(state)

        assert update["pending_actions"] == [] and update["reply_draft"] is None
        assert update["errors"]


@covers_node(COMMUNICATIONS_GRAPH, csg.REQUEST_SEND_APPROVAL)
class TestRequestSendApproval:
    async def test_the_preview_shows_recipient_subject_body_diff_attachments_and_reply(
        self, world, candidate
    ):
        world.add_mail("m-1")
        state = await drafted(world, candidate[0])

        (request,) = world.graph.request_send_approval(state)["approval_requests"]

        preview = request["preview"]
        assert preview["recipient"] == cf.RECRUITER
        assert preview["subject"] == "Re: Interview for REQ-7781 at Acme"
        assert preview["body_diff"].startswith("--- current\n+++ proposed")
        assert preview["attachments"] == []
        assert preview["in_reply_to"] == "m-1"
        intent = ActionIntent.model_validate(state["pending_actions"][0])
        assert request["action_hash"] == intent.fingerprint()

    def test_its_input_contract_requires_a_pending_action(self):
        with pytest.raises(ValidationError):
            CONTRACTS[csg.REQUEST_SEND_APPROVAL].validate_input({"pending_actions": []})


@covers_node(COMMUNICATIONS_GRAPH, csg.SEND_APPROVAL_CHECKPOINT)
class TestSendApprovalCheckpoint:
    async def test_the_run_parks_with_the_request_and_sends_nothing(self, world, candidate):
        world.add_mail("m-1")
        compiled = world.compile()

        parked = await compiled.ainvoke(
            {"request": "draft a reply to thread t-acme", "user_id": str(candidate[0])},
            cf.thread(),
        )

        (request,) = cf.interrupt_requests(parked)
        assert request["kind"] == ActionKind.SEND_EMAIL.value
        assert world.sends() == [] and world.executor.calls == []

    def test_with_nothing_pending_it_does_not_interrupt(self, world):
        assert world.graph.send_approval_checkpoint({"pending_actions": []}) == {}


@covers_node(COMMUNICATIONS_GRAPH, csg.EXECUTE_SEND)
class TestExecuteSend:
    async def test_without_an_approval_nothing_is_sent(self, world, candidate):
        world.add_mail("m-1")
        state = await drafted(world, candidate[0])

        update = await world.graph.execute_send(state)

        (refusal,) = update["approval_refusals"]
        assert refusal["reason"] == RefusalReason.NO_REQUEST.value
        assert update["action_receipts"][0]["ok"] is False
        assert world.executor.calls == [] and world.sends() == []

    async def test_an_approved_send_goes_out_once_as_a_reply(self, world, candidate):
        world.add_mail("m-1")
        compiled = world.compile()
        config = cf.thread()
        parked = await compiled.ainvoke(
            {"request": "draft a reply to thread t-acme", "user_id": str(candidate[0])}, config
        )
        decisions = [cf.approve(r).model_dump(mode="json") for r in cf.interrupt_requests(parked)]

        final = await compiled.ainvoke(Command(resume=decisions), config)

        (receipt,) = final["action_receipts"]
        assert receipt["ok"] is True and receipt["detail"] == "sent as a reply"
        assert len(world.sends()) == 1
        assert final["reply_draft"]["body"] != TEMPLATE_REPLY_BODY
