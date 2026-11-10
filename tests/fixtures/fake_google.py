"""A fake Google API, a fake token exchanger, and a harness wiring both to the gateway.

`FakeGoogleApi` implements `mcp_servers.google.api.GoogleHttp` and answers
with the JSON shapes the real Gmail and Calendar APIs return, so the tools'
response validation is exercised for real. It only accepts access tokens the
`FakeGoogleExchanger` minted, records every call (with the token it carried),
and lets a test override any response to feed the tools malformed or hostile
payloads.

`GoogleHarness` builds the whole path -- credential broker, tools, server,
policy engine, gateway -- in one of two variants:

``in_process``
    `GoogleMCPServer` registered directly on the `MCPServerManager`.
``mcp``
    The same server behind the stdio JSON-RPC transport, reached through a
    `RemoteMCPServer` proxy over real OS pipes.

The contract tests run every scenario against both.
"""

import asyncio
import base64
import itertools
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import message_from_bytes, policy
from typing import Any
from uuid import uuid4

from mcp_servers.google.api import CALENDAR_API_BASE, GMAIL_API_BASE, raise_for_status
from mcp_servers.google.server import GoogleMCPServer
from mcp_servers.google.tools import GoogleWorkspaceTools
from personalos.domain.credentials import AccessToken, CredentialRef, SecretValue
from personalos.domain.models import ToolAuthorization
from personalos.executor.credentials import CredentialBroker
from personalos.mcp.adapter import MCPToolInvoker
from personalos.mcp.manager import MCPServerManager
from personalos.mcp.stdio import MCPStdioClient, MCPStdioServer, RemoteMCPServer, open_pipe_streams
from personalos.persistence.idempotency import InMemoryOperationStore
from personalos.policy import (
    ApprovalGrant,
    IntentOrigin,
    PolicyEngine,
    ToolIntent,
)
from personalos.policy.rules import GOOGLE_TOOL_ARGUMENTS, default_rules
from personalos.secrets.store import InMemorySecretStore
from personalos.tools.gateway import PolicyEnforcingToolGateway, ToolResult

ACCOUNT = "candidate@example.com"
CREDENTIAL = CredentialRef("google", ACCOUNT)
REFRESH_TOKEN = "1//refresh-token-that-must-never-leave-the-broker"

#: A node allowed every Google tool, for the contract tests.
ALL_TOOLS_NODE = "test:google_contract"
VARIANTS = ("in_process", "mcp")

NOW = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)


def b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def gmail_message(
    message_id: str,
    *,
    sender: str = "Recruiter <recruiter@acme.example>",
    to: str = ACCOUNT,
    subject: str = "Interview at Acme",
    body: str = "Hi, are you free Thursday?",
    html: str | None = None,
    thread_id: str | None = None,
    labels: Iterable[str] = ("INBOX",),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A `users.messages.get?format=full` response: multipart/alternative."""
    parts = [{"partId": "0", "mimeType": "text/plain", "body": {"data": b64url(body)}}]
    if html is not None:
        parts.append({"partId": "1", "mimeType": "text/html", "body": {"data": b64url(html)}})
    return {
        "id": message_id,
        "threadId": thread_id or f"t-{message_id}",
        "labelIds": list(labels),
        "snippet": body[:100],
        "internalDate": "1762765200000",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "To", "value": to},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Mon, 10 Nov 2026 09:00:00 +0000"},
                {"name": "Message-ID", "value": f"<{message_id}@mail.example>"},
            ],
            "body": {"size": 0},
            "parts": parts,
        },
        **(extra or {}),
    }


@dataclass
class Call:
    """One request the fake received."""

    method: str
    url: str
    token: str
    scopes: tuple[str, ...]
    params: dict[str, Any]
    json: Any


class FakeGoogleExchanger:
    """Mints opaque fake access tokens; remembers what it minted and what it was given."""

    def __init__(self, ttl: timedelta = timedelta(minutes=30)):
        self.ttl = ttl
        self.issued: dict[str, tuple[str, ...]] = {}
        self.refresh_tokens_seen: list[str] = []
        self._counter = itertools.count(1)

    async def exchange(self, ref, secret: SecretValue, scopes) -> AccessToken:
        self.refresh_tokens_seen.append(secret.reveal())
        token = f"ya29.fake-access-{next(self._counter)}"
        self.issued[token] = tuple(scopes)
        return AccessToken(
            credential_ref=ref,
            secret=SecretValue(token),
            expires_at=datetime.now(timezone.utc) + self.ttl,
            scopes=tuple(scopes),
        )


@dataclass
class FakeGoogleApi:
    """Gmail and Calendar, in memory, speaking Google's JSON."""

    exchanger: FakeGoogleExchanger
    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    drafts: dict[str, dict[str, Any]] = field(default_factory=dict)
    sent: list[dict[str, Any]] = field(default_factory=list)
    events: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    deleted_events: set[str] = field(default_factory=set)
    calls: list[Call] = field(default_factory=list)
    #: `(method, url regex) -> payload or callable(call) -> payload`, consulted first.
    overrides: dict[tuple[str, str], Any] = field(default_factory=dict)
    #: Make the next event insert land, then answer 504 -- the lost response.
    fail_next_insert_after_commit: bool = False
    #: Make the next `messages.send` land, then answer 504 -- the lost response.
    fail_next_send_after_commit: bool = False
    #: Label id -> name. System labels are their own names, as at Google.
    labels: dict[str, str] = field(
        default_factory=lambda: {name: name for name in ("INBOX", "SENT", "SPAM", "UNREAD")}
    )
    #: `users.history.list` records, oldest first.
    history: list[dict[str, Any]] = field(default_factory=list)
    history_id: int = 1000
    #: History ids at or below this are no longer kept: a cursor there gets a 404.
    history_floor: int = 0
    #: Evaluate `messages.list` queries (`_matches`) instead of listing every
    #: message. Off by default, for the tests written before it existed.
    evaluate_queries: bool = False

    def __post_init__(self):
        self._ids = itertools.count(1)

    def override(self, method: str, url_pattern: str, response: Any) -> None:
        self.overrides[(method, url_pattern)] = response

    def add_message(self, message: dict[str, Any]) -> None:
        """Put a message in the mailbox, and in its history as `messagesAdded`."""
        self.messages[message["id"]] = message
        self._record("messagesAdded", {"message": self._history_ref(message)})

    def add_label(self, name: str, label_id: str | None = None) -> str:
        """Create a user label; return its id."""
        label_id = label_id or f"Label_{len(self.labels) + 1}"
        self.labels[label_id] = name
        return label_id

    def label_message(self, message_id: str, label_id: str) -> None:
        """Label an existing message, as a filter or the candidate would."""
        message = self.messages[message_id]
        message["labelIds"] = [*message["labelIds"], label_id]
        self._record("labelsAdded", {"message": self._history_ref(message), "labelIds": [label_id]})

    def expire_history(self) -> None:
        """Forget all history so far, as Google does after about a week."""
        self.history_floor = self.history_id
        self.history = []

    @staticmethod
    def _history_ref(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": message["id"],
            "threadId": message["threadId"],
            "labelIds": list(message["labelIds"]),
        }

    def _record(self, kind: str, entry: dict[str, Any]) -> None:
        self.history_id += 1
        self.history.append({"id": str(self.history_id), kind: [entry]})

    def _matches(self, message: dict[str, Any], query: str) -> bool:
        """A small subset of Gmail search: `label:`, `from:`, `subject:`, bare words.

        Time bounds (`newer_than:`, `after:`) are accepted and ignored, and an
        `OR` between parenthesised groups is honoured -- enough to evaluate the
        queries the inbox poller builds.
        """
        groups = [g.strip() for g in re.split(r"\)\s+OR\s+\(", query.strip())]
        return any(self._matches_all(message, g.strip("()")) for g in groups)

    def _matches_all(self, message: dict[str, Any], query: str) -> bool:
        headers = {h["name"].lower(): h["value"] for h in message["payload"]["headers"]}
        fields = {
            "label": " ".join(self.labels.get(i, i).lower() for i in message["labelIds"]),
            "from": headers.get("from", "").lower(),
            "subject": headers.get("subject", "").lower(),
            "": f"{headers.get('subject', '')} {message.get('snippet', '')}".lower(),
        }
        for term in re.findall(r'(?:\w+:)?"[^"]*"|\S+', query):
            key, _, value = term.strip("()").rpartition(":")
            if key in ("newer_than", "after", "older_than", "before") or not value:
                continue
            if value.strip('"').lower() not in fields.get(key, fields[""]):
                return False
        return True

    def add_event(self, calendar_id: str, event: dict[str, Any]) -> None:
        self.events.setdefault(calendar_id, {})[event["id"]] = event

    async def request(self, method, url, *, token: AccessToken, params=None, json=None):
        secret = token.secret.reveal()
        if secret not in self.exchanger.issued or token.is_expired():
            raise_for_status(401)
        call = Call(method, url, secret, self.exchanger.issued[secret], dict(params or {}), json)
        self.calls.append(call)

        for (o_method, pattern), response in self.overrides.items():
            if o_method == method and re.fullmatch(pattern, url):
                return response(call) if callable(response) else response

        if url.startswith(GMAIL_API_BASE):
            return self._gmail(method, url[len(GMAIL_API_BASE) :], call)
        if url.startswith(CALENDAR_API_BASE):
            return self._calendar(method, url[len(CALENDAR_API_BASE) :], call)
        raise_for_status(404)

    # Gmail ---------------------------------------------------------------

    def _gmail(self, method: str, path: str, call: Call):
        if method == "GET" and path == "/messages":
            return self._list_messages(call)
        if method == "GET" and path == "/profile":
            return {"emailAddress": ACCOUNT, "historyId": str(self.history_id)}
        if method == "GET" and path == "/labels":
            return {
                "labels": [
                    {"id": i, "name": n, "type": "system" if i == n else "user"}
                    for i, n in self.labels.items()
                ]
            }
        if method == "GET" and path == "/history":
            return self._history_page(call)
        match = re.fullmatch(r"/messages/([^/]+)", path)
        if method == "GET" and match:
            message = self.messages.get(match.group(1)) or self._sent_message(match.group(1))
            if message is None:
                raise_for_status(404)
            return message
        if method == "POST" and path == "/drafts":
            draft_id = f"r-{next(self._ids)}"
            message_id = f"m-draft-{next(self._ids)}"
            self.drafts[draft_id] = {**call.json, "raw": decode_raw(call.json["message"]["raw"])}
            return {
                "id": draft_id,
                "message": {
                    "id": message_id,
                    "threadId": call.json["message"].get("threadId", f"t-{message_id}"),
                },
            }
        if method == "POST" and path == "/messages/send":
            message_id = f"m-sent-{next(self._ids)}"
            self.sent.append({**call.json, "id": message_id, "raw": decode_raw(call.json["raw"])})
            if self.fail_next_send_after_commit:
                # The message went out; the caller sees a gateway timeout.
                self.fail_next_send_after_commit = False
                raise_for_status(504)
            return {
                "id": message_id,
                "threadId": call.json.get("threadId", f"t-{message_id}"),
                "labelIds": ["SENT"],
            }
        raise_for_status(404)

    def _list_messages(self, call: Call) -> dict[str, Any]:
        query = call.params.get("q")
        if "in:sent" in (query or ""):
            return self._sent_search(call)
        ids = [
            i
            for i, m in self.messages.items()
            if not (query and self.evaluate_queries) or self._matches(m, query)
        ]
        size = int(call.params.get("maxResults", 100))
        offset = int(call.params.get("pageToken") or 0)
        body: dict[str, Any] = {
            "messages": [
                {"id": i, "threadId": self.messages[i]["threadId"]}
                for i in ids[offset : offset + size]
            ],
            "resultSizeEstimate": len(ids),
        }
        if offset + size < len(ids):
            body["nextPageToken"] = str(offset + size)
        return body

    def _sent_message(self, message_id: str) -> dict[str, Any] | None:
        """A sent message as `users.messages.get` returns it."""
        for sent in self.sent:
            if sent["id"] == message_id:
                raw = sent["raw"]
                return {
                    "id": message_id,
                    "threadId": sent.get("threadId", f"t-{message_id}"),
                    "labelIds": ["SENT"],
                    "snippet": "",
                    "internalDate": "1762765200000",
                    "payload": {
                        "mimeType": "text/plain",
                        "headers": [{"name": k, "value": str(v)} for k, v in raw.items()],
                        "body": {"data": b64url(raw.get_content())},
                    },
                }
        return None

    def _sent_search(self, call: Call) -> dict[str, Any]:
        """`in:sent`, optionally narrowed by `rfc822msgid:<...>`: the reconciler's lookup."""
        wanted = re.search(r"rfc822msgid:(\S+)", call.params.get("q") or "")
        hits = [
            sent
            for sent in self.sent
            if wanted is None or str(sent["raw"].get("Message-ID", "")) == wanted.group(1)
        ]
        return {
            "messages": [
                {"id": sent["id"], "threadId": sent.get("threadId", f"t-{sent['id']}")}
                for sent in hits
            ],
            "resultSizeEstimate": len(hits),
        }

    def _history_page(self, call: Call) -> dict[str, Any]:
        start = int(call.params["startHistoryId"])
        if start < self.history_floor:
            raise_for_status(404)
        kinds = {"messageAdded": "messagesAdded", "labelAdded": "labelsAdded"}
        wanted = {kinds[t] for t in call.params.get("historyTypes") or kinds}
        records = [
            {"id": r["id"], **{k: v for k, v in r.items() if k in wanted}}
            for r in self.history
            if int(r["id"]) > start and any(k in r for k in wanted)
        ]
        size = int(call.params.get("maxResults", 100))
        offset = int(call.params.get("pageToken") or 0)
        body: dict[str, Any] = {
            "history": records[offset : offset + size],
            "historyId": str(self.history_id),
        }
        if offset + size < len(records):
            body["nextPageToken"] = str(offset + size)
        return body

    # Calendar ------------------------------------------------------------

    def _calendar(self, method: str, path: str, call: Call):
        if method == "POST" and path == "/freeBusy":
            return self._freebusy(call)
        match = re.fullmatch(r"/calendars/([^/]+)/events(?:/([^/]+))?", path)
        if not match:
            raise_for_status(404)
        calendar_id, event_id = match.group(1), match.group(2)
        calendar = self.events.setdefault(calendar_id, {})
        if method == "GET" and event_id is None:
            return self._list(calendar, call)
        if method == "POST" and event_id is None:
            return self._insert(calendar, call.json)
        if event_id in self.deleted_events:
            raise_for_status(410)
        if event_id not in calendar:
            raise_for_status(404)
        if method == "GET":
            return calendar[event_id]
        if method == "PATCH":
            calendar[event_id] = self._patched(calendar[event_id], call.json)
            return calendar[event_id]
        if method == "DELETE":
            del calendar[event_id]
            self.deleted_events.add(event_id)
            return None
        raise_for_status(404)

    def _list(self, calendar: dict[str, Any], call: Call) -> dict[str, Any]:
        items = sorted(calendar.values(), key=lambda e: e["start"].get("dateTime", ""))
        wanted = call.params.get("privateExtendedProperty") or []
        for pair in [wanted] if isinstance(wanted, str) else wanted:
            key, _, value = pair.partition("=")
            items = [e for e in items if _private(e).get(key) == value]
        time_min, time_max = call.params.get("timeMin"), call.params.get("timeMax")
        if time_min or time_max:
            lo = datetime.fromisoformat(time_min) if time_min else None
            hi = datetime.fromisoformat(time_max) if time_max else None
            items = [e for e in items if _overlaps(e, lo, hi)]
        size = int(call.params["maxResults"])
        offset = int(call.params.get("pageToken") or 0)
        body: dict[str, Any] = {"kind": "calendar#events", "items": items[offset : offset + size]}
        if offset + size < len(items):
            body["nextPageToken"] = str(offset + size)
        return body

    def _patched(self, current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        patch = dict(patch)
        if "extendedProperties" in patch:
            # PATCH merges nested objects: named keys change, others stay.
            private = {**_private(current), **patch.pop("extendedProperties").get("private", {})}
            patch["extendedProperties"] = {"private": private}
        return {**current, **patch, "etag": self._etag()}

    def _etag(self) -> str:
        return f'"{next(self._ids)}"'

    def _insert(self, calendar: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        lose_response = self.fail_next_insert_after_commit
        self.fail_next_insert_after_commit = False
        new_id = f"evt{next(self._ids)}"
        event = {
            "id": new_id,
            "status": "confirmed",
            "etag": self._etag(),
            "htmlLink": f"https://calendar.google.com/event?eid={new_id}",
            **body,
        }
        calendar[new_id] = event
        if lose_response:
            # The insert landed; the caller sees a gateway timeout.
            raise_for_status(504)
        return event

    def _freebusy(self, call: Call) -> dict[str, Any]:
        lo = datetime.fromisoformat(call.json["timeMin"])
        hi = datetime.fromisoformat(call.json["timeMax"])
        calendars: dict[str, Any] = {}
        for item in call.json["items"]:
            calendar_id = item["id"]
            if calendar_id not in self.events:
                calendars[calendar_id] = {"busy": [], "errors": [{"reason": "notFound"}]}
                continue
            busy = []
            for event in self.events[calendar_id].values():
                if event.get("transparency") == "transparent" or "dateTime" not in event["start"]:
                    continue
                start = datetime.fromisoformat(event["start"]["dateTime"])
                end = datetime.fromisoformat(event["end"]["dateTime"])
                if start < hi and end > lo:
                    busy.append(
                        {"start": max(start, lo).isoformat(), "end": min(end, hi).isoformat()}
                    )
            calendars[calendar_id] = {"busy": sorted(busy, key=lambda b: b["start"])}
        return {"kind": "calendar#freeBusy", "calendars": calendars}

    def move_event(self, calendar_id: str, event_id: str, start: datetime, end: datetime) -> None:
        """Move an event the way the candidate would, in the calendar UI."""
        event = self.events[calendar_id][event_id]
        event["start"] = {**event["start"], "dateTime": start.isoformat()}
        event["end"] = {**event["end"], "dateTime": end.isoformat()}
        event["etag"] = self._etag()


def _private(event: dict[str, Any]) -> dict[str, str]:
    return (event.get("extendedProperties") or {}).get("private", {})


def _overlaps(event: dict[str, Any], lo: datetime | None, hi: datetime | None) -> bool:
    if "dateTime" not in event["start"]:
        return False
    start = datetime.fromisoformat(event["start"]["dateTime"])
    end = datetime.fromisoformat(event["end"]["dateTime"])
    return (hi is None or start < hi) and (lo is None or end > lo)


def decode_raw(raw: str):
    """A Gmail `raw` field, parsed back into an email message."""
    return message_from_bytes(base64.urlsafe_b64decode(raw), policy=policy.default)


def approval_for(intent: ToolIntent) -> ApprovalGrant:
    return ApprovalGrant(
        intent_id=intent.intent_id,
        intent_fingerprint=intent.fingerprint(),
        approved_by="human:candidate",
    )


class GoogleHarness:
    """The full Google tool path, in-process or over MCP stdio."""

    def __init__(
        self,
        variant: str,
        *,
        node_allowlists: Mapping[str, Iterable[str]] | None = None,
        authorize: bool = True,
    ):
        assert variant in VARIANTS, variant
        self.variant = variant
        self.exchanger = FakeGoogleExchanger()
        self.api = FakeGoogleApi(self.exchanger)
        self.store = InMemorySecretStore()
        self.store.put(CREDENTIAL, SecretValue(REFRESH_TOKEN))
        self.broker = CredentialBroker(self.store, {"google": self.exchanger})
        self.tools = GoogleWorkspaceTools(self.broker, self.api)
        self.server = GoogleMCPServer(self.tools, operation_store=InMemoryOperationStore())
        self.manager = MCPServerManager()
        allowlists = (
            {ALL_TOOLS_NODE: set(GOOGLE_TOOL_ARGUMENTS)}
            if node_allowlists is None
            else node_allowlists
        )
        self.policy = PolicyEngine(default_rules(node_tool_allowlists=allowlists))
        authorizers = {"google": self._authorize} if authorize else {}
        self.gateway = PolicyEnforcingToolGateway(
            self.policy, MCPToolInvoker(self.manager, authorizers=authorizers)
        )
        self._client: MCPStdioClient | None = None
        self._serving: asyncio.Task | None = None
        self.remote: RemoteMCPServer | None = None

    @staticmethod
    def _authorize(approved) -> ToolAuthorization:
        return ToolAuthorization(
            credential_ref=str(CREDENTIAL),
            principal=approved.intent.requested_by,
            policy_decision_id=approved.decision.record_id,
        )

    async def start(self) -> "GoogleHarness":
        if self.variant == "in_process":
            self.manager.register_server(self.server)
            return self
        # Two OS pipes, exactly as a parent and its stdio child would have.
        c2s_read, c2s_write = os.pipe()
        s2c_read, s2c_write = os.pipe()
        server_reader, server_writer = await open_pipe_streams(
            os.fdopen(c2s_read, "rb", 0), os.fdopen(s2c_write, "wb", 0)
        )
        client_reader, client_writer = await open_pipe_streams(
            os.fdopen(s2c_read, "rb", 0), os.fdopen(c2s_write, "wb", 0)
        )
        self._server_writer = server_writer
        self._serving = asyncio.create_task(
            MCPStdioServer(self.server).serve(server_reader, server_writer)
        )
        self._client = MCPStdioClient(client_reader, client_writer, timeout=10)
        self.remote = await RemoteMCPServer.connect("google", self._client)
        self.manager.register_server(self.remote)
        return self

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._serving is not None:
            await asyncio.wait_for(self._serving, 5)
            self._server_writer.close()

    def intent(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        node: str = ALL_TOOLS_NODE,
        origin: IntentOrigin = IntentOrigin.SYSTEM,
        mutating: bool | None = None,
    ) -> ToolIntent:
        args = dict(arguments)
        is_mutating = self.server.get_tool_schema(tool).mutating if mutating is None else mutating
        if is_mutating and "idempotency_key" not in args:
            args["idempotency_key"] = f"key-{uuid4()}"
        return ToolIntent(
            server="google",
            tool=tool,
            arguments=args,
            origin=origin,
            mutating=is_mutating,
            requested_by=node,
        )

    async def dispatch(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        approve: bool = False,
        node: str = ALL_TOOLS_NODE,
        origin: IntentOrigin = IntentOrigin.SYSTEM,
    ) -> ToolResult:
        intent = self.intent(tool, arguments, node=node, origin=origin)
        return await self.gateway.dispatch(intent, approval_for(intent) if approve else None)

    def tool_names(self) -> list[str]:
        return sorted(tool.name for tool in self.manager.get_server("google").get_tools())


Factory = Callable[..., GoogleHarness]
