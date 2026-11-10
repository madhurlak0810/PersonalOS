"""Which mail this system may look at, and the event that says some arrived.

**Scope.** `MailboxScope` is the whole of the mailbox this system sees: the
Gmail labels and Gmail search queries the user configured
(`GMAIL_RECRUITER_LABELS`, `GMAIL_RECRUITER_QUERIES`), shown back to them by
`personalos.cli inbox status`. A message matching none of them is ignored --
not fetched, not stored, not logged by content. An empty scope matches
nothing; there is no "whole inbox" default.

**Retention: provider ids, and fetch on demand.** The decision this module
records (issue #113): PersonalOS does not keep a copy of a message. What it
stores about one is a pointer -- the Gmail message id in an `email.received`
envelope's `raw_ref` and payload -- and, once a message is classified against
an application, the `communication_events` row (classification, subject,
sender, thread id) and its commitments. The body is read from Gmail at the
moment classification needs it (`GmailRecruiterInbox.read`), held in memory
for that one step, and left out of everything that step persists: the
outbox, the workflow command, and the graph checkpoint.

Why not keep bodies locally: a local copy is a second, unencrypted mailbox
with its own retention, backup and breach story, for a benefit (offline
re-classification) that refetching gives anyway. The costs accepted: a
message deleted at Gmail before it is classified is never classified, and
re-running classification needs the Google connection. Revisit if the
offline case starts to matter; the place to add a store is behind `read`.

Pure values, like the rest of `domain`; nothing here reads or writes.
"""

import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.events import EventEnvelope, EventType

#: `EventEnvelope.source` and the `raw_ref` scheme of Gmail-sourced events.
GMAIL_SOURCE = "gmail"

#: Gmail labels that are never recruiter mail *to* the candidate, whatever
#: else a message is labelled with: their own sent mail and drafts (a
#: labelled thread labels the candidate's replies too), and spam and trash.
EXCLUDED_LABEL_IDS = frozenset({"SENT", "DRAFT", "SPAM", "TRASH"})

MAX_SCOPE_RULES = 20
MAX_LABEL_NAME_CHARS = 225
MAX_QUERY_CHARS = 500


class MailboxScopeInvalid(ValidationFailed):
    """A configured mailbox rule cannot be used as written."""


def _rule_text(value: str, *, limit: int, what: str) -> str:
    text = " ".join(value.split())
    if not text:
        raise MailboxScopeInvalid(f"a {what} must not be blank")
    if len(text) > limit:
        raise MailboxScopeInvalid(f"a {what} is {len(text)} characters; the limit is {limit}")
    return text


class MailboxScope(BaseModel):
    """The labels and queries that put a message in view. Everything else is ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Gmail label names, as the user sees them (`Recruiting`, `Jobs/Acme`).
    #: Matched case-insensitively against the mailbox's labels.
    labels: tuple[str, ...] = ()
    #: Gmail search queries (`from:greenhouse.io`, `subject:interview`).
    queries: tuple[str, ...] = ()

    @field_validator("labels", mode="before")
    @classmethod
    def _labels(cls, value: Any) -> tuple[str, ...]:
        labels = [
            _rule_text(str(v), limit=MAX_LABEL_NAME_CHARS, what="label name") for v in value or ()
        ]
        for label in labels:
            if '"' in label:
                raise MailboxScopeInvalid(f"label name {label!r} contains a double quote")
        return _unique(labels, "labels")

    @field_validator("queries", mode="before")
    @classmethod
    def _queries(cls, value: Any) -> tuple[str, ...]:
        queries = [_rule_text(str(v), limit=MAX_QUERY_CHARS, what="query") for v in value or ()]
        return _unique(queries, "queries")

    @property
    def is_empty(self) -> bool:
        """True when nothing is in view."""
        return not self.labels and not self.queries

    def rules(self) -> tuple[str, ...]:
        """Every rule, as one Gmail search query each: `label:"<name>"` or the query."""
        return (*(label_query(label) for label in self.labels), *self.queries)

    def fingerprint(self) -> str:
        """Identity of the rule set. A cursor kept under another fingerprint is resynced."""
        canonical = json.dumps(
            {
                "labels": sorted(label.casefold() for label in self.labels),
                "queries": sorted(self.queries),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        """The rules as the user configured them, for display."""
        return {"labels": list(self.labels), "queries": list(self.queries)}


def _unique(values: list[str], what: str) -> tuple[str, ...]:
    if len(values) > MAX_SCOPE_RULES:
        raise MailboxScopeInvalid(f"at most {MAX_SCOPE_RULES} {what} may be configured")
    seen: dict[str, str] = {}
    for value in values:
        seen.setdefault(value.casefold(), value)
    return tuple(seen.values())


def label_query(name: str) -> str:
    """The Gmail search term for one label name."""
    return f'label:"{name}"'


class MailboxCursor(BaseModel):
    """How far through one mailbox's history the poller has read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = GMAIL_SOURCE
    account: str
    #: Gmail's history id: every change at or below it has been considered.
    history_id: str = Field(pattern=r"^\d{1,20}$")
    #: `MailboxScope.fingerprint` of the rules the history was read under.
    scope_fingerprint: str
    #: When the cursor last moved (naive UTC). Bounds the search a query rule
    #: is evaluated with.
    polled_at: datetime

    def is_behind(self, history_id: str) -> bool:
        """True when `history_id` is later than this cursor."""
        return int(history_id) > int(self.history_id)


# --- The event ---------------------------------------------------------------


def gmail_raw_ref(message_id: str) -> str:
    """The pointer an event carries in place of the message."""
    return f"{GMAIL_SOURCE}:{message_id}"


def parse_gmail_raw_ref(raw_ref: str | None) -> str:
    """The Gmail message id a `raw_ref` points at."""
    scheme, separator, message_id = (raw_ref or "").partition(":")
    if scheme != GMAIL_SOURCE or not separator or not message_id:
        raise ValidationFailed(f"'{raw_ref}' is not a gmail message reference")
    return message_id


def email_received_dedupe_key(message_id: str) -> str:
    """The outbox key of one message's arrival: the provider's message id, and nothing else.

    Not the history id, the rule that matched, or when it was seen: a message
    seen twice -- redelivered by a replayed history page, labelled a second
    time, found again by a resync -- is the same fact.
    """
    message_id = message_id.strip()
    if not message_id:
        raise ValidationFailed("a message with no provider id cannot be deduplicated")
    return f"{EventType.EMAIL_RECEIVED}:{GMAIL_SOURCE}:{message_id}"


def email_received_envelope(
    *,
    user_id: UUID,
    message_id: str,
    thread_id: str | None,
    matched_rules: tuple[str, ...] | list[str],
    observed_at: datetime,
    history_id: str | None = None,
) -> EventEnvelope:
    """The `email.received` envelope for one in-scope message.

    Ids and the names of the rules that matched. No subject, sender, snippet
    or body: the envelope sits in the outbox and the worker's logs, and
    `raw_ref` is how a consumer gets the message, from Gmail, when it needs it.
    `occurred_at` is when the poller saw it arrive; the message's own receipt
    time is read with the message.
    """
    return EventEnvelope(
        type=EventType.EMAIL_RECEIVED,
        source=GMAIL_SOURCE,
        occurred_at=observed_at,
        received_at=observed_at,
        dedupe_key=email_received_dedupe_key(message_id),
        payload={
            "user_id": str(user_id),
            "provider_message_id": message_id,
            "thread_id": thread_id,
            "matched_rules": sorted(matched_rules),
            "history_id": history_id,
        },
        raw_ref=gmail_raw_ref(message_id),
    )


__all__ = [
    "EXCLUDED_LABEL_IDS",
    "GMAIL_SOURCE",
    "MAX_QUERY_CHARS",
    "MAX_SCOPE_RULES",
    "MailboxCursor",
    "MailboxScope",
    "MailboxScopeInvalid",
    "email_received_dedupe_key",
    "email_received_envelope",
    "gmail_raw_ref",
    "label_query",
    "parse_gmail_raw_ref",
]
