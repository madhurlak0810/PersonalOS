"""Contract tests for the Google MCP server's Gmail and Calendar tools.

Every test runs twice -- once with `GoogleMCPServer` registered in-process,
once with the same server behind the stdio MCP transport -- through the same
gateway and the same policy engine, against `tests/fixtures/fake_google.py`.
A behaviour that differs between the two variants fails here.
"""

import pytest

from mcp_servers.google.tools import MAX_RESULT_BODY_CHARS
from personalos.domain.integrations import (
    CALENDAR_EVENTS_SCOPE,
    CALENDAR_READONLY_SCOPE,
    GMAIL_COMPOSE_SCOPE,
    GMAIL_READONLY_SCOPE,
    GMAIL_SEND_SCOPE,
)
from personalos.domain.models import ActionTarget, ToolCallErrorCode, ToolCallRequest
from personalos.policy import ApprovalRequired, PolicyDenied
from personalos.policy.permissions import DEFAULT_TOOL_PERMISSIONS, PermissionClass
from tests.fixtures.fake_google import (
    CREDENTIAL,
    REFRESH_TOKEN,
    VARIANTS,
    GoogleHarness,
    approval_for,
    gmail_message,
)

GOOGLE_TOOLS = [
    "calendar_create_event",
    "calendar_delete_event",
    "calendar_freebusy",
    "calendar_get_event",
    "calendar_list_events",
    "calendar_update_event",
    "gmail_create_draft",
    "gmail_get_profile",
    "gmail_list_history",
    "gmail_list_labels",
    "gmail_read_message",
    "gmail_search",
    "gmail_send_message",
]

MESSAGE = {"to": ["recruiter@acme.example"], "subject": "Re: Interview", "body": "Thursday works."}
EVENT = {
    "summary": "Interview: Acme",
    "start": "2026-11-12T15:00:00+00:00",
    "end": "2026-11-12T16:00:00+00:00",
    "attendees": ["recruiter@acme.example"],
    "private_properties": {"personalos_idempotency_key": "abc12345"},
}


@pytest.fixture(params=VARIANTS)
async def google(request):
    harness = await GoogleHarness(request.param).start()
    try:
        yield harness
    finally:
        await harness.stop()


# --- Surface ------------------------------------------------------------------


async def test_both_variants_expose_the_same_thirteen_tools(google):
    assert google.tool_names() == GOOGLE_TOOLS


@pytest.mark.parametrize(
    "tool, permission_class",
    [
        ("gmail_search", PermissionClass.READ_EXTERNAL),
        ("gmail_read_message", PermissionClass.READ_EXTERNAL),
        ("gmail_list_history", PermissionClass.READ_EXTERNAL),
        ("gmail_get_profile", PermissionClass.READ_EXTERNAL),
        ("gmail_list_labels", PermissionClass.READ_EXTERNAL),
        ("calendar_list_events", PermissionClass.READ_EXTERNAL),
        ("calendar_get_event", PermissionClass.READ_EXTERNAL),
        ("calendar_freebusy", PermissionClass.READ_EXTERNAL),
        ("gmail_create_draft", PermissionClass.WRITE_REVERSIBLE),
        ("gmail_send_message", PermissionClass.WRITE_EXTERNAL),
        ("calendar_create_event", PermissionClass.WRITE_EXTERNAL),
        ("calendar_update_event", PermissionClass.WRITE_EXTERNAL),
        ("calendar_delete_event", PermissionClass.DESTRUCTIVE),
    ],
)
def test_every_tool_has_its_permission_class(tool, permission_class):
    assert DEFAULT_TOOL_PERMISSIONS[f"google.{tool}"].permission_class is permission_class


# --- Gmail --------------------------------------------------------------------


async def test_gmail_search_returns_untrusted_summaries(google):
    google.api.add_message(gmail_message("m1"))
    google.api.add_message(gmail_message("m2", subject="Offer"))

    result = (
        await google.dispatch("gmail_search", {"query": "from:acme", "max_results": 5})
    ).unwrap()

    assert result["content_trust"] == "untrusted"
    assert [m["message_id"] for m in result["messages"]] == ["m1", "m2"]
    assert result["messages"][1]["subject"] == "Offer"
    assert result["messages"][0]["sender"] == "Recruiter <recruiter@acme.example>"
    assert google.api.calls[0].params["q"] == "from:acme"
    assert {call.scopes for call in google.api.calls} == {(GMAIL_READONLY_SCOPE,)}


async def test_gmail_read_message_returns_the_plain_text_body(google):
    google.api.add_message(gmail_message("m1", body="Are you free Thursday?", html="<b>x</b>"))

    result = (await google.dispatch("gmail_read_message", {"message_id": "m1"})).unwrap()

    assert result["body_text"] == "Are you free Thursday?"
    assert result["subject"] == "Interview at Acme"
    assert result["rfc822_message_id"] == "<m1@mail.example>"
    assert result["body_truncated"] is False
    assert result["content_trust"] == "untrusted"


async def test_gmail_read_message_falls_back_to_de_tagged_html(google):
    message = gmail_message("m1", html="<p>Hello <b>there</b> &amp; welcome</p>")
    message["payload"]["parts"] = message["payload"]["parts"][1:]
    google.api.add_message(message)

    result = (await google.dispatch("gmail_read_message", {"message_id": "m1"})).unwrap()

    assert " ".join(result["body_text"].split()) == "Hello there & welcome"


async def test_gmail_read_of_a_missing_message_fails_cleanly(google):
    result = await google.dispatch("gmail_read_message", {"message_id": "nope"})
    assert result.success is False
    assert "404" in result.error


async def test_gmail_read_with_missing_ok_answers_not_found(google):
    result = (
        await google.dispatch("gmail_read_message", {"message_id": "gone", "missing_ok": True})
    ).unwrap()
    assert (result["found"], result["message_id"], result["body_text"]) == (False, "gone", "")


async def test_gmail_history_lists_ids_since_a_cursor_and_reports_one_expired(google):
    label = google.api.add_label("Recruiting")
    start = str(google.api.history_id)
    google.api.add_message(gmail_message("m1", labels=("INBOX",)))
    google.api.label_message("m1", label)

    result = (await google.dispatch("gmail_list_history", {"start_history_id": start})).unwrap()

    assert [(c["change"], c["message_id"]) for c in result["changes"]] == [
        ("message_added", "m1"),
        ("labels_added", "m1"),
    ]
    assert result["changes"][1]["label_ids"] == ["INBOX", label]
    assert result["history_id"] == str(google.api.history_id)
    assert result["cursor_expired"] is False
    (call,) = google.api.calls
    assert call.scopes == (GMAIL_READONLY_SCOPE,)
    assert call.params["historyTypes"] == ["messageAdded", "labelAdded"]

    google.api.expire_history()
    expired = (await google.dispatch("gmail_list_history", {"start_history_id": start})).unwrap()
    assert (expired["cursor_expired"], expired["changes"]) == (True, [])


async def test_gmail_profile_and_labels_are_read_only(google):
    google.api.add_label("Recruiting", "Label_9")

    profile = (await google.dispatch("gmail_get_profile", {})).unwrap()
    labels = (await google.dispatch("gmail_list_labels", {})).unwrap()

    assert profile == {"content_trust": "untrusted", "history_id": str(google.api.history_id)}
    assert {"label_id": "Label_9", "name": "Recruiting", "type": "user"} in labels["labels"]
    assert {call.scopes for call in google.api.calls} == {(GMAIL_READONLY_SCOPE,)}


@pytest.mark.parametrize("bad", ["abc", "-1", "1" * 21])
async def test_gmail_history_refuses_a_cursor_that_is_not_a_history_id(google, bad):
    result = await google.dispatch("gmail_list_history", {"start_history_id": bad})
    assert result.success is False
    assert google.api.calls == []


async def test_gmail_create_draft_saves_and_sends_nothing(google):
    result = (
        await google.dispatch("gmail_create_draft", {**MESSAGE, "thread_id": "t1"}, approve=True)
    ).unwrap()

    assert result["draft_id"].startswith("r-") and result["thread_id"] == "t1"
    assert google.api.sent == []
    (draft,) = google.api.drafts.values()
    assert draft["raw"]["Subject"] == "Re: Interview"
    assert draft["raw"]["To"] == "recruiter@acme.example"
    assert google.api.calls[-1].scopes == (GMAIL_COMPOSE_SCOPE,)


async def test_gmail_send_requires_approval(google):
    with pytest.raises(ApprovalRequired):
        await google.dispatch("gmail_send_message", MESSAGE)
    assert google.api.sent == []


async def test_gmail_send_with_approval_sends_once_per_idempotency_key(google):
    intent = google.intent("gmail_send_message", {**MESSAGE, "in_reply_to": "<m1@mail.example>"})
    first = (await google.gateway.dispatch(intent, approval_for(intent))).unwrap()
    again = await google.gateway.dispatch(intent, approval_for(intent))

    assert first["message_id"].startswith("m-sent-")
    assert again.replayed is True and again.result == first
    assert len(google.api.sent) == 1
    sent = google.api.sent[0]["raw"]
    assert sent["In-Reply-To"] == "<m1@mail.example>"
    assert sent.get_content().strip() == "Thursday works."
    assert google.api.calls[-1].scopes == (GMAIL_SEND_SCOPE,)


async def test_a_header_cannot_be_injected_through_the_subject(google):
    result = await google.dispatch(
        "gmail_send_message",
        {**MESSAGE, "subject": "Hello\r\nBcc: attacker@evil.example"},
        approve=True,
    )
    assert result.success is False
    assert google.api.sent == []


async def test_an_address_list_cannot_be_smuggled_into_one_recipient(google):
    result = await google.dispatch(
        "gmail_send_message",
        {**MESSAGE, "to": ["a@acme.example, attacker@evil.example"]},
        approve=True,
    )
    assert result.success is False
    assert google.api.sent == []


# --- Calendar -----------------------------------------------------------------


async def test_calendar_list_events_in_start_order(google):
    google.api.add_event(
        "primary",
        {
            "id": "e2",
            "summary": "Later",
            "start": {"dateTime": "2026-11-13T10:00:00Z"},
            "end": {"dateTime": "2026-11-13T11:00:00Z"},
        },
    )
    google.api.add_event(
        "primary",
        {
            "id": "e1",
            "summary": "Sooner",
            "start": {"dateTime": "2026-11-12T10:00:00Z"},
            "end": {"dateTime": "2026-11-12T11:00:00Z"},
            "extendedProperties": {"private": {"k": "v"}},
        },
    )

    result = (
        await google.dispatch(
            "calendar_list_events",
            {"time_min": "2026-11-10T00:00:00+00:00", "time_max": "2026-11-20T00:00:00+00:00"},
        )
    ).unwrap()

    assert [e["event_id"] for e in result["events"]] == ["e1", "e2"]
    assert result["events"][0]["private_properties"] == {"k": "v"}
    assert result["content_trust"] == "untrusted"
    call = google.api.calls[-1]
    assert call.params["singleEvents"] == "true" and call.params["timeMin"].startswith("2026-11-10")
    assert call.scopes == (CALENDAR_READONLY_SCOPE,)


async def test_calendar_create_requires_approval_then_creates(google):
    with pytest.raises(ApprovalRequired):
        await google.dispatch("calendar_create_event", EVENT)
    assert google.api.events.get("primary", {}) == {}

    result = (await google.dispatch("calendar_create_event", EVENT, approve=True)).unwrap()

    event = result["event"]
    assert event["summary"] == "Interview: Acme"
    assert event["private_properties"] == {"personalos_idempotency_key": "abc12345"}
    assert google.api.events["primary"][event["event_id"]]["attendees"] == [
        {"email": "recruiter@acme.example"}
    ]
    assert google.api.calls[-1].params["sendUpdates"] == "none"
    assert google.api.calls[-1].scopes == (CALENDAR_EVENTS_SCOPE,)


async def test_calendar_create_rejects_an_end_before_its_start(google):
    result = await google.dispatch(
        "calendar_create_event", {**EVENT, "end": "2026-11-12T14:00:00+00:00"}, approve=True
    )
    assert result.success is False
    assert google.api.calls == []


async def test_calendar_update_requires_approval_then_patches(google):
    created = (await google.dispatch("calendar_create_event", EVENT, approve=True)).unwrap()
    event_id = created["event"]["event_id"]
    patch = {
        "event_id": event_id,
        "start": "2026-11-12T17:00:00+00:00",
        "end": "2026-11-12T18:00:00+00:00",
    }
    with pytest.raises(ApprovalRequired):
        await google.dispatch("calendar_update_event", patch)

    result = (await google.dispatch("calendar_update_event", patch, approve=True)).unwrap()

    assert result["event"]["start"] == "2026-11-12T17:00:00+00:00"
    assert result["event"]["summary"] == "Interview: Acme"
    assert google.api.calls[-1].method == "PATCH"


async def test_calendar_delete_is_destructive_and_tolerates_already_gone(google):
    created = (await google.dispatch("calendar_create_event", EVENT, approve=True)).unwrap()
    event_id = created["event"]["event_id"]
    with pytest.raises(ApprovalRequired):
        await google.dispatch("calendar_delete_event", {"event_id": event_id})

    first = (
        await google.dispatch("calendar_delete_event", {"event_id": event_id}, approve=True)
    ).unwrap()
    second = (
        await google.dispatch("calendar_delete_event", {"event_id": event_id}, approve=True)
    ).unwrap()

    assert first == {**first, "deleted": True, "already_deleted": False}
    assert second["already_deleted"] is True
    assert event_id not in google.api.events["primary"]


async def test_a_path_cannot_be_traversed_through_an_event_id(google):
    result = await google.dispatch(
        "calendar_delete_event", {"event_id": "../../users/me/settings"}, approve=True
    )
    assert result.success is False
    assert google.api.calls == []


# --- Credentials and authorization context ------------------------------------


async def test_each_call_brokers_its_own_token_and_never_sees_the_refresh_token(google):
    google.api.add_message(gmail_message("m1"))

    await google.dispatch("gmail_read_message", {"message_id": "m1"})
    await google.dispatch("gmail_read_message", {"message_id": "m1"})
    result = await google.dispatch("gmail_send_message", MESSAGE, approve=True)

    tokens = [call.token for call in google.api.calls]
    assert len(set(tokens)) == 3, "a token is minted per tool call, never reused"
    assert REFRESH_TOKEN not in tokens
    assert REFRESH_TOKEN not in repr(result.result) and REFRESH_TOKEN not in repr(google.api.sent)
    # Only the broker's exchanger ever held it.
    assert google.exchanger.refresh_tokens_seen == [REFRESH_TOKEN] * 3


async def test_a_credential_cannot_be_chosen_through_the_arguments(google):
    with pytest.raises(PolicyDenied) as denied:
        await google.dispatch(
            "gmail_search",
            {"query": "x", "credential_ref": "cred://google/someone-else@example.com"},
        )
    assert denied.value.decision.rule == "argument_allowlist"

    # And past policy, the server's typed contract refuses it too.
    result = await google.manager.execute_tool(
        ToolCallRequest(
            target=ActionTarget(server="google", tool="gmail_search"),
            params={"query": "x", "authorization": {"credential_ref": "cred://google/x@y.z"}},
        )
    )
    assert result.ok is False and result.error.code is ToolCallErrorCode.VALIDATION_ERROR
    assert google.api.calls == []


@pytest.mark.parametrize("variant", VARIANTS)
async def test_a_call_without_gateway_authorization_is_refused(variant):
    harness = await GoogleHarness(variant, authorize=False).start()
    try:
        result = await harness.dispatch("gmail_search", {"query": "x"})
    finally:
        await harness.stop()
    assert result.success is False
    assert result.error_code == "internal"
    assert "authorization" in result.error
    assert harness.api.calls == [] and harness.exchanger.refresh_tokens_seen == []


async def test_a_revoked_credential_fails_without_calling_google(google):
    google.store.delete(CREDENTIAL)

    result = await google.dispatch("gmail_search", {"query": "x"})

    assert result.success is False and "reconnect" in result.error
    assert google.api.calls == []


async def test_long_bodies_are_truncated(google):
    google.api.add_message(gmail_message("m1", body="x" * (MAX_RESULT_BODY_CHARS + 50)))
    result = (await google.dispatch("gmail_read_message", {"message_id": "m1"})).unwrap()
    assert len(result["body_text"]) == MAX_RESULT_BODY_CHARS and result["body_truncated"] is True
