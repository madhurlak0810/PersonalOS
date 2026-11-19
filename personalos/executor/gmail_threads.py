"""The Communications subgraph's mailbox: scoped inbox listings and thread reads.

`GmailThreadReader` satisfies the subgraph's `MailboxReader` port over the
Google MCP tools that already exist (`gmail_search`, `gmail_read_message`);
there is no new provider code behind it. Every call is a `ToolIntent` through
the policy gateway, stamped with the `requested_by` of the node making it, so
`COMMUNICATIONS_NODE_TOOL_ALLOWLISTS` decides what each node can reach.

**Scope.** What this reader can see is the `MailboxScope` the Gmail inbox
poller uses (`GMAIL_RECRUITER_LABELS`, `GMAIL_RECRUITER_QUERIES`): every
search is one of those rules, optionally narrowed. A thread is read only if
at least one of its messages matches a rule, and only its matching messages
are read. An empty scope reads nothing. Spam, trash and drafts are never in
view; the candidate's own sent mail is in view inside a thread (so a summary
has both sides) and never in a triage listing.

**Content is untrusted.** The MCP server strips control and bidi characters
and stamps results `content_trust: "untrusted"`; this module keeps that stamp
on every value it returns, records URLs as inert strings, and never fetches
one. No Gmail tool here downloads an attachment.
"""

import logging
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from personalos.domain.communications import (
    MAX_MESSAGE_BODY_CHARS,
    MAX_THREAD_MESSAGES,
    MAX_TRIAGE_ITEMS,
    EmailMessage,
    EmailThread,
    InboxItem,
    inert_links,
)
from personalos.domain.errors import NotFound
from personalos.domain.mailbox import EXCLUDED_LABEL_IDS, MailboxScope
from personalos.policy import IntentOrigin, ToolIntent
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

GOOGLE_SERVER = "google"

#: Labels never read into a thread: not mail anyone sent.
THREAD_EXCLUDED_LABEL_IDS = frozenset({"DRAFT", "SPAM", "TRASH"})

SEARCH_PAGE_SIZE = 25
MAX_SEARCH_PAGES = 4


class ThreadNotInScope(NotFound):
    """The thread has no message inside the mailbox scope, or does not exist."""


def _quoted(term: str) -> str:
    return '"' + term.replace('"', " ").strip() + '"'


class GmailThreadReader:
    """`MailboxReader` over `google.gmail_search` and `google.gmail_read_message`."""

    def __init__(self, gateway: ToolGateway, *, scope: MailboxScope):
        if gateway is None:
            raise ValueError("GmailThreadReader requires a ToolGateway")
        self.gateway = gateway
        self.scope = scope

    async def _call(self, tool: str, arguments: dict[str, Any], requested_by: str) -> dict:
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=False,
            requested_by=requested_by,
        )
        return (await self.gateway.dispatch(intent)).unwrap() or {}

    async def _search(self, query: str, requested_by: str, *, limit: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(MAX_SEARCH_PAGES):
            arguments: dict[str, Any] = {"query": query, "max_results": SEARCH_PAGE_SIZE}
            if page_token:
                arguments["page_token"] = page_token
            page = await self._call("gmail_search", arguments, requested_by)
            hits.extend(page.get("messages") or [])
            page_token = page.get("next_page_token")
            if not page_token or len(hits) >= limit:
                break
        return hits[:limit]

    # --- triage ---------------------------------------------------------

    async def list_inbox(
        self, *, requested_by: str, limit: int = MAX_TRIAGE_ITEMS
    ) -> list[InboxItem]:
        """The newest in-scope messages, headers and snippets only, deduplicated."""
        if self.scope.is_empty:
            logger.warning("the mailbox scope is empty; there is nothing to triage")
            return []
        items: dict[str, InboxItem] = {}
        for rule in self.scope.rules():
            for hit in await self._search(f"({rule})", requested_by, limit=limit):
                labels = tuple(str(label) for label in hit.get("label_ids") or ())
                if set(labels) & EXCLUDED_LABEL_IDS:
                    continue
                message_id = str(hit["message_id"])
                existing = items.get(message_id)
                if existing is not None:
                    items[message_id] = existing.model_copy(
                        update={"matched_rules": (*existing.matched_rules, rule)}
                    )
                    continue
                items[message_id] = InboxItem(
                    message_id=message_id,
                    thread_id=hit.get("thread_id") or None,
                    sender=str(hit.get("sender") or ""),
                    subject=str(hit.get("subject") or ""),
                    snippet=str(hit.get("snippet") or ""),
                    date=str(hit.get("date") or ""),
                    label_ids=labels,
                    matched_rules=(rule,),
                )
        return list(items.values())[:limit]

    # --- threads --------------------------------------------------------

    async def find_thread(self, *, sender_hint: str | None, requested_by: str) -> str | None:
        """The newest in-scope thread, from `sender_hint` when one is given."""
        if self.scope.is_empty:
            return None
        narrow = f" from:{_quoted(sender_hint)}" if sender_hint else ""
        for rule in self.scope.rules():
            for hit in await self._search(f"({rule}){narrow}", requested_by, limit=5):
                if set(hit.get("label_ids") or ()) & EXCLUDED_LABEL_IDS:
                    continue
                if hit.get("thread_id"):
                    return str(hit["thread_id"])
        return None

    async def read_thread(self, thread_id: str, *, requested_by: str) -> EmailThread:
        """The in-scope messages of `thread_id`, read, oldest first.

        Raises `ThreadNotInScope` when no message of the thread matches a
        scope rule: a thread outside the scope is not read, whoever names it.
        """
        ids = await self._in_scope_ids(thread_id, requested_by)
        if not ids:
            raise ThreadNotInScope(f"thread '{thread_id}' has no message inside the mailbox scope")

        messages: list[EmailMessage] = []
        for message_id in ids:
            result = await self._call(
                "gmail_read_message", {"message_id": message_id, "missing_ok": True}, requested_by
            )
            if not result.get("found", True):
                continue
            if str(result.get("thread_id") or thread_id) != thread_id:
                continue
            messages.append(_message(result, thread_id))
        if not messages:
            raise ThreadNotInScope(f"thread '{thread_id}' has no readable message left")
        messages.sort(key=lambda m: m.received_at or datetime.min)
        messages = messages[-MAX_THREAD_MESSAGES:]
        subject = next((m.subject for m in messages if m.subject), "")
        return EmailThread(thread_id=thread_id, subject=subject, messages=tuple(messages))

    async def _in_scope_ids(self, thread_id: str, requested_by: str) -> list[str]:
        """Ids of the thread's messages that some scope rule matches, in search order."""
        ids: list[str] = []
        for rule in self.scope.rules():
            for hit in await self._search(f"({rule})", requested_by, limit=100):
                if str(hit.get("thread_id") or "") != thread_id:
                    continue
                if set(hit.get("label_ids") or ()) & THREAD_EXCLUDED_LABEL_IDS:
                    continue
                message_id = str(hit["message_id"])
                if message_id not in ids:
                    ids.append(message_id)
        return ids


def _message(result: dict[str, Any], thread_id: str) -> EmailMessage:
    sender = str(result.get("sender") or "")
    address = parseaddr(sender)[1] or None
    body = str(result.get("body_text") or "")[:MAX_MESSAGE_BODY_CHARS]
    header = str(result.get("rfc822_message_id") or "") or None
    if header and not (header.startswith("<") and header.endswith(">")):
        header = None
    labels = tuple(str(label) for label in result.get("label_ids") or ())
    return EmailMessage(
        message_id=str(result["message_id"]),
        thread_id=thread_id,
        sender=sender,
        from_address=address if address and "@" in address else None,
        to=str(result.get("to") or ""),
        subject=str(result.get("subject") or ""),
        received_at=_received_at(result),
        body=body,
        label_ids=labels,
        rfc822_message_id=header,
        from_user="SENT" in labels,
        links=inert_links(body),
    )


def _received_at(result: dict[str, Any]) -> datetime | None:
    for value, parse in (
        (result.get("received_at"), datetime.fromisoformat),
        (result.get("date"), parsedate_to_datetime),
    ):
        if not value:
            continue
        try:
            when = parse(str(value))
        except (TypeError, ValueError, IndexError):
            continue
        if when.tzinfo is None:
            continue
        return when.astimezone(timezone.utc).replace(tzinfo=None)
    return None


__all__ = ["GmailThreadReader", "THREAD_EXCLUDED_LABEL_IDS", "ThreadNotInScope"]
