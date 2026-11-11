"""The candidate's recruiter mail, from Gmail, over the Google MCP tools.

`GmailRecruiterInbox` has two halves, run by different parts of the system and
allowed different tools:

**`poll`** is a worker sweep. It reads the mailbox's *history* -- Gmail's log
of which message ids arrived or gained labels since a stored history id --
decides which of those are in the user's `MailboxScope`, and writes one
`email.received` envelope per in-scope message to the outbox, together with
the new cursor (`SqlMailboxCursorStore.advance`). It sees ids, label ids and,
for query rules, Gmail's own search hits; it never reads a message body. Its
intents are `requested_by` `GMAIL_INBOX_POLLER`, whose allowlist has no
content read and no write.

- *Label rules* are matched from the label ids each history record carries,
  so mail outside the scope is never fetched at all.
- *Query rules* are evaluated by Gmail: one search per rule, bounded to mail
  newer than the previous poll (less `query_lookback`), intersected with the
  ids the history reported. A search hit the history did not report is not
  new and is skipped.
- The candidate's own mail (`SENT`, `DRAFT`) and spam and trash are never in
  scope, whatever labels a thread carries.

No cursor yet, a cursor read under different rules, or one Google no longer
keeps history for (about a week) means a *resync*: the current history id is
taken first, then each rule is searched over the last `backfill_days`, and
both are committed together. Anything a resync finds that was already
published is deduplicated by the outbox on the message id.

**`read`** is fetch-on-demand for the classifier (see
`personalos.domain.mailbox` for the retention decision). It reads one message
through `gmail_read_message`, `requested_by` the `classify_recruiter_events`
node, and returns it as a `RecruiterMessage`. The caller holds it in memory;
nothing here stores it.

Message content is untrusted throughout: the MCP server strips control and
bidi characters and stamps every result `content_trust: "untrusted"`, the
extractor's prompt says to treat the email as data, and no tool the classifier
node lacks becomes reachable because an email asks for it -- the node
allowlist is keyed on code-set provenance, not on anything a message says.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any
from uuid import UUID

from personalos.domain.events import EventEnvelope
from personalos.domain.health import GMAIL, HealthCheckResult, run_check
from personalos.domain.job_search import RecruiterMessage
from personalos.domain.mailbox import (
    EXCLUDED_LABEL_IDS,
    GMAIL_SOURCE,
    MailboxCursor,
    MailboxScope,
    email_received_envelope,
    label_query,
    parse_gmail_raw_ref,
)
from personalos.persistence.mailbox_cursors import SqlMailboxCursorStore
from personalos.policy import IntentOrigin, ToolIntent
from personalos.policy.rules import GMAIL_INBOX_POLLER
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

GOOGLE_SERVER = "google"

#: The node whose allowlist the on-demand read is made under.
CLASSIFIER_NODE = "graph:job_search#classify_recruiter_events"

#: History pages read in one poll; the rest wait for the next one.
MAX_HISTORY_PAGES = 20
HISTORY_PAGE_SIZE = 100
SEARCH_PAGE_SIZE = 25
#: Search pages followed per rule (so at most 200 messages a rule).
MAX_SEARCH_PAGES = 8

DEFAULT_BACKFILL_DAYS = 14
DEFAULT_QUERY_LOOKBACK = timedelta(days=2)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _later(a: str, b: str) -> str:
    return a if int(a) >= int(b) else b


def _page_position(page: dict[str, Any], changes: list[dict[str, Any]], position: str) -> str:
    """Where the cursor stands once `page` is committed.

    After the last page, the mailbox's own history id. Part-way, the last
    record this page held: the next page starts after it.
    """
    if not page.get("next_page_token"):
        return _later(page.get("history_id") or position, position)
    for change in changes:
        position = _later(str(change["history_id"]), position)
    return position


@dataclass
class _Candidate:
    """A message the history says is new, and what it says about it."""

    thread_id: str | None
    history_id: str
    label_ids: set[str] = field(default_factory=set)
    rules: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PollResult:
    """What one poll did."""

    account: str
    #: Where the cursor stands now.
    history_id: str
    #: `email.received` events staged by this poll (redeliveries not counted).
    published: int
    #: Ids of the in-scope messages this poll saw, new or not.
    matched: tuple[str, ...] = ()
    #: Messages the history reported, in scope or not.
    considered: int = 0
    #: Why the cursor was rebuilt, if it was.
    resynced: str | None = None
    #: Configured label names the mailbox has no label for.
    unknown_labels: tuple[str, ...] = ()


class GmailRecruiterInbox:
    """Polls one Gmail account for in-scope mail, and reads a message on demand."""

    #: The integration `health_check` reports on.
    integration = GMAIL

    def __init__(
        self,
        gateway: ToolGateway,
        cursors: SqlMailboxCursorStore,
        *,
        scope: MailboxScope,
        user_id: UUID,
        account: str,
        backfill_days: int = DEFAULT_BACKFILL_DAYS,
        query_lookback: timedelta = DEFAULT_QUERY_LOOKBACK,
        clock: Callable[[], datetime] = _utcnow,
        poller_requested_by: str = GMAIL_INBOX_POLLER,
        reader_requested_by: str = CLASSIFIER_NODE,
    ):
        """Take the gateway, the cursor store, the scope, and whose mail this is."""
        if gateway is None:
            raise ValueError("GmailRecruiterInbox requires a ToolGateway")
        if backfill_days < 0:
            raise ValueError("backfill_days must not be negative")
        self.gateway = gateway
        self.cursors = cursors
        self.scope = scope
        self.user_id = user_id
        self.account = account.lower()
        self.backfill_days = backfill_days
        self.query_lookback = query_lookback
        self.clock = clock
        self.poller_requested_by = poller_requested_by
        self.reader_requested_by = reader_requested_by

    # --- dispatch -------------------------------------------------------

    async def _call(self, tool: str, arguments: dict[str, Any], requested_by: str) -> dict:
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=False,
            requested_by=requested_by,
        )
        result = await self.gateway.dispatch(intent)
        return result.unwrap() or {}

    async def _poller(self, tool: str, arguments: dict[str, Any]) -> dict:
        return await self._call(tool, arguments, self.poller_requested_by)

    # --- health ---------------------------------------------------------

    async def health_check(self) -> HealthCheckResult:
        """Read the mailbox's profile: one cheap call that needs the same grant a poll does."""
        return await run_check(
            self.integration, lambda: self._poller("gmail_get_profile", {}), checked_at=self.clock()
        )

    # --- poll -----------------------------------------------------------

    async def poll(self) -> PollResult:
        """Publish an `email.received` for each new in-scope message; move the cursor."""
        now = self.clock()
        cursor = self.cursors.get(GMAIL_SOURCE, self.account)
        if self.scope.is_empty:
            logger.warning("the Gmail inbox scope has no labels or queries; nothing is read")
            return PollResult(self.account, cursor.history_id if cursor else "0", 0)
        if cursor is None:
            return await self._resync(now, "first poll")
        if cursor.scope_fingerprint != self.scope.fingerprint():
            return await self._resync(now, "the scope rules changed")
        result = await self._follow_history(cursor, now)
        if result.matched or result.published:
            logger.info(
                "Gmail poll for %s: %s in scope of %s new, %s published",
                self.account,
                len(result.matched),
                result.considered,
                result.published,
            )
        return result

    async def _follow_history(self, cursor: MailboxCursor, now: datetime) -> PollResult:
        """Read the history after `cursor` page by page, committing each page as it is read.

        All pages answer the one request made from `cursor`; a crash part-way
        leaves the cursor after the last page committed, and the next poll
        asks again from there.
        """
        labels, unknown = await self._label_ids()
        position = cursor.history_id
        published, considered = 0, 0
        matched: list[str] = []
        page_token: str | None = None
        for _ in range(MAX_HISTORY_PAGES):
            page = await self._history_page(cursor.history_id, page_token)
            if page.get("cursor_expired"):
                return await self._resync(now, "Gmail no longer keeps history that far back")

            changes = page.get("changes") or []
            candidates = self._candidates(changes)
            considered += len(candidates)
            await self._match(candidates, labels, since=cursor.polled_at)
            in_scope = {i: c for i, c in candidates.items() if c.rules}

            page_token = page.get("next_page_token")
            position = _page_position(page, changes, position)
            published += self.cursors.advance(
                self._cursor(position, now), self._envelopes(in_scope, now)
            )
            matched.extend(in_scope)
            if not page_token:
                break
        else:
            logger.info(
                "Gmail history for %s has more than %s pages; the rest is read next poll",
                self.account,
                MAX_HISTORY_PAGES,
            )
        return PollResult(
            self.account,
            position,
            published,
            matched=tuple(matched),
            considered=considered,
            unknown_labels=unknown,
        )

    async def _history_page(self, start: str, page_token: str | None) -> dict[str, Any]:
        arguments: dict[str, Any] = {"start_history_id": start, "max_results": HISTORY_PAGE_SIZE}
        if page_token:
            arguments["page_token"] = page_token
        return await self._poller("gmail_list_history", arguments)

    async def _resync(self, now: datetime, reason: str) -> PollResult:
        """Start the cursor at the mailbox's current position, after a bounded backfill.

        The position is taken *before* the search: anything that arrives in
        between is in both, and the outbox keeps one copy.
        """
        logger.info("resyncing the Gmail cursor for %s: %s", self.account, reason)
        profile = await self._poller("gmail_get_profile", {})
        position = str(profile["history_id"])
        _labels, unknown = await self._label_ids()

        candidates: dict[str, _Candidate] = {}
        if self.backfill_days > 0:
            for rule in self.scope.rules():
                for message_id, thread_id in await self._search(
                    f"({rule}) newer_than:{self.backfill_days}d"
                ):
                    candidate = candidates.setdefault(
                        message_id, _Candidate(thread_id=thread_id, history_id=position)
                    )
                    candidate.rules.add(rule)

        published = self.cursors.advance(
            self._cursor(position, now), self._envelopes(candidates, now)
        )
        return PollResult(
            self.account,
            position,
            published,
            matched=tuple(candidates),
            considered=len(candidates),
            resynced=reason,
            unknown_labels=unknown,
        )

    # --- scope ----------------------------------------------------------

    async def _label_ids(self) -> tuple[dict[str, str], tuple[str, ...]]:
        """Configured label name -> the mailbox's label id, and the names with none."""
        if not self.scope.labels:
            return {}, ()
        listing = await self._poller("gmail_list_labels", {})
        by_name = {
            str(label["name"]).casefold(): str(label["label_id"])
            for label in listing.get("labels") or []
        }
        found: dict[str, str] = {}
        unknown: list[str] = []
        for name in self.scope.labels:
            label_id = by_name.get(name.casefold())
            if label_id is None:
                unknown.append(name)
            else:
                found[name] = label_id
        if unknown:
            logger.warning(
                "Gmail has no label named %s; those rules match nothing",
                ", ".join(repr(n) for n in unknown),
            )
        return found, tuple(unknown)

    @staticmethod
    def _candidates(changes: list[dict[str, Any]]) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}
        for change in changes:
            message_id = str(change["message_id"])
            candidate = candidates.setdefault(
                message_id,
                _Candidate(thread_id=change.get("thread_id"), history_id=change["history_id"]),
            )
            candidate.label_ids.update(change.get("label_ids") or ())
            candidate.history_id = _later(change["history_id"], candidate.history_id)
        return {
            message_id: candidate
            for message_id, candidate in candidates.items()
            if not candidate.label_ids & EXCLUDED_LABEL_IDS
        }

    async def _match(
        self, candidates: dict[str, _Candidate], labels: dict[str, str], *, since: datetime
    ) -> None:
        """Mark each candidate with the rules it matches. Unmatched ones stay unmarked."""
        if not candidates:
            return
        for name, label_id in labels.items():
            for candidate in candidates.values():
                if label_id in candidate.label_ids:
                    candidate.rules.add(label_query(name))
        if not self.scope.queries:
            return
        bound = since - self.query_lookback
        after = int(bound.replace(tzinfo=timezone.utc).timestamp())
        for query in self.scope.queries:
            for message_id, _thread in await self._search(f"({query}) after:{after}"):
                if message_id in candidates:
                    candidates[message_id].rules.add(query)

    async def _search(self, query: str) -> list[tuple[str, str | None]]:
        """Message and thread ids Gmail finds for `query`, excluding the candidate's own mail."""
        hits: list[tuple[str, str | None]] = []
        page_token: str | None = None
        for _ in range(MAX_SEARCH_PAGES):
            arguments: dict[str, Any] = {"query": query, "max_results": SEARCH_PAGE_SIZE}
            if page_token:
                arguments["page_token"] = page_token
            page = await self._poller("gmail_search", arguments)
            for summary in page.get("messages") or []:
                if set(summary.get("label_ids") or ()) & EXCLUDED_LABEL_IDS:
                    continue
                hits.append((str(summary["message_id"]), summary.get("thread_id")))
            page_token = page.get("next_page_token")
            if not page_token:
                break
        return hits

    # --- writing --------------------------------------------------------

    def _cursor(self, position: str, now: datetime) -> MailboxCursor:
        return MailboxCursor(
            provider=GMAIL_SOURCE,
            account=self.account,
            history_id=position,
            scope_fingerprint=self.scope.fingerprint(),
            polled_at=now,
        )

    def _envelopes(self, candidates: dict[str, _Candidate], now: datetime) -> list[EventEnvelope]:
        return [
            email_received_envelope(
                user_id=self.user_id,
                message_id=message_id,
                thread_id=candidate.thread_id,
                matched_rules=tuple(candidate.rules),
                observed_at=now,
                history_id=candidate.history_id,
            )
            for message_id, candidate in candidates.items()
        ]

    # --- read -----------------------------------------------------------

    async def read(self, ref: str) -> RecruiterMessage | None:
        """The message `ref` (`gmail:<id>`) points at, read now; `None` if Gmail no longer has it."""
        message_id = parse_gmail_raw_ref(ref)
        result = await self._call(
            "gmail_read_message",
            {"message_id": message_id, "missing_ok": True},
            self.reader_requested_by,
        )
        if not result.get("found", True):
            logger.info("gmail message %s is gone; it is not classified", message_id)
            return None
        sender = str(result.get("sender") or "")
        address = parseaddr(sender)[1] or None
        return RecruiterMessage(
            provider_message_id=str(result.get("message_id") or message_id),
            received_at=self._received_at(result),
            body=str(result.get("body_text") or ""),
            subject=str(result.get("subject") or "") or None,
            from_address=address if address and "@" in address else None,
            thread_id=result.get("thread_id") or None,
        )

    def _received_at(self, result: dict[str, Any]) -> datetime:
        """Gmail's receipt time; else the `Date` header; else now. Naive UTC."""
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
        return self.clock()


__all__ = [
    "CLASSIFIER_NODE",
    "DEFAULT_BACKFILL_DAYS",
    "DEFAULT_QUERY_LOOKBACK",
    "GmailRecruiterInbox",
    "PollResult",
]
