"""The Communications domain: triage, thread summaries, action items and replies.

Pure values, like the rest of `domain`. The Communications subgraph
(`personalos.graphs.communications`) moves them between its nodes; nothing
here reads a mailbox or calls a model.

**Mail is untrusted data.** Every field read out of an email -- sender,
subject, body, snippet -- was written by a third party. It is carried in
`EmailMessage` / `InboxItem`, stamped `content_trust="untrusted"`, and the
values a model may return about it are closed schemas (`TriageProposal`,
`ThreadDigestProposal`, `ReplyProposal`) with no field for a tool, a recipient
or an instruction. Links found in a body are kept as inert strings
(`EmailMessage.links`); nothing follows them, and no attachment is ever
opened: the Gmail tools this subgraph may call cannot download one.

**Triage is typed, and fails to "needs review".** A triage is a
`TriageResult`: a category, an urgency and a suggested action from closed
enums. A model's answer that will not validate as `TriageProposal` becomes
`needs_review(...)`, deterministically, rather than a guess.

**A summary cites its messages.** Every key point and action item names the
message ids it was read from, and `digest_from_proposal` refuses an id that
is not in the thread, so a summary cannot attribute something to mail that
was never there.
"""

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.recruiter_events import CommitmentActor

#: Longest body kept per message once it is read. The MCP server truncates too;
#: this bounds what a node holds in memory and hands to a model.
MAX_MESSAGE_BODY_CHARS = 8000
#: Most messages one thread read keeps (the most recent ones).
MAX_THREAD_MESSAGES = 25
#: Most inbox items one triage run looks at.
MAX_TRIAGE_ITEMS = 25
MAX_KEY_POINTS = 10
MAX_ACTION_ITEMS = 20
MAX_LINKS_PER_MESSAGE = 20
MAX_REASON_CHARS = 300

#: Stamped on every value read out of a mailbox.
ContentTrust = Literal["untrusted"]


class CommunicationsContractError(ValidationFailed, ValueError):
    """A Communications value broke one of its invariants."""


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _naive_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


# --- The request -----------------------------------------------------------------


class CommunicationsTask(str, Enum):
    """What a person asked the Communications subgraph to do. A closed set."""

    TRIAGE = "triage"
    SUMMARIZE_THREAD = "summarize_thread"
    EXTRACT_ACTION_ITEMS = "extract_action_items"
    DRAFT_REPLY = "draft_reply"


#: Tasks that work on one thread rather than on the inbox.
THREAD_TASKS = frozenset(
    {
        CommunicationsTask.SUMMARIZE_THREAD,
        CommunicationsTask.EXTRACT_ACTION_ITEMS,
        CommunicationsTask.DRAFT_REPLY,
    }
)

#: A Gmail id as the Google MCP server accepts one.
_ID = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


def _check_id(value: str | None, what: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if not _ID.match(value):
        raise CommunicationsContractError(f"'{value}' is not a valid {what}")
    return value


class CommunicationsRequest(_Value):
    """A request, parsed. Which thread, and what to do with it.

    Parsed by `parse_communications_request` from the person's own words,
    deterministically: the request is not mail, but it is not a place for a
    model to decide which mailbox call to make either.
    """

    task: CommunicationsTask
    #: The Gmail thread to work on, when the request named one.
    thread_id: str | None = None
    #: A sender to find the thread by ("reply to the email from Acme"), when
    #: no thread id was named. Letters, digits and `.@_-` only.
    sender_hint: str | None = Field(default=None, max_length=64)

    @field_validator("thread_id")
    @classmethod
    def _thread_id(cls, value: str | None) -> str | None:
        return _check_id(value, "thread id")

    @field_validator("sender_hint")
    @classmethod
    def _sender_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"[^\w.@-]+", " ", value).strip()
        return cleaned[:64] or None

    @property
    def is_thread_task(self) -> bool:
        return self.task in THREAD_TASKS


_TASK_KEYWORDS: tuple[tuple[CommunicationsTask, tuple[str, ...]], ...] = (
    (
        CommunicationsTask.DRAFT_REPLY,
        ("draft a reply", "draft reply", "reply", "respond", "write back", "answer the"),
    ),
    (
        CommunicationsTask.EXTRACT_ACTION_ITEMS,
        ("action item", "to-do", "todo", "to do", "next step", "what do i need", "what do i owe"),
    ),
    (
        CommunicationsTask.SUMMARIZE_THREAD,
        ("summar", "recap", "tl;dr", "tldr", "catch me up", "what did"),
    ),
    (
        CommunicationsTask.TRIAGE,
        ("triage", "inbox", "unread", "prioriti", "what's new", "whats new", "sort my"),
    ),
)

#: `thread:<id>`, `thread <id>`, `thread id <id>`. An id with a digit, `-`
#: or `_` in it, so "the thread from Acme" does not read "from" as an id.
_THREAD_ID_RE = re.compile(
    r"\bthread(?:\s+id)?\s*[:#=]?\s*([A-Za-z0-9_-]*[0-9_-][A-Za-z0-9_-]*)", re.IGNORECASE
)
_SENDER_RE = re.compile(r"\bfrom\s+([\w.@-]+(?:\s+[A-Z][\w.-]*)?)")
#: "from" words that name a time, not a sender: "the email from this morning".
_NOT_SENDERS = frozenset({"this", "today", "yesterday", "last", "earlier", "my", "the", "a"})


def parse_communications_request(
    text: str, *, thread_id: str | None = None
) -> CommunicationsRequest:
    """The task and thread a request names. Never fails: an unclear request triages.

    `thread_id`, when the caller already knows it (the UI, a test), wins over
    anything in the text.
    """
    lowered = (text or "").lower()
    named_thread = thread_id
    if named_thread is None:
        match = _THREAD_ID_RE.search(text or "")
        named_thread = match.group(1) if match else None

    task: CommunicationsTask | None = None
    for candidate, keywords in _TASK_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            task = candidate
            break
    if task is None:
        task = CommunicationsTask.SUMMARIZE_THREAD if named_thread else CommunicationsTask.TRIAGE

    sender: str | None = None
    if task in THREAD_TASKS and not named_thread:
        for match in _SENDER_RE.finditer(text or ""):
            first = match.group(1).split()[0].lower()
            if first not in _NOT_SENDERS:
                sender = match.group(1)
                break
    try:
        return CommunicationsRequest(task=task, thread_id=named_thread, sender_hint=sender)
    except (CommunicationsContractError, ValidationError):
        # Not an id Gmail would accept: drop it rather than pass it on.
        return CommunicationsRequest(task=task, sender_hint=sender)


# --- Mail, as read -----------------------------------------------------------------


_LINK_RE = re.compile(r"\bhttps?://[^\s<>\"')\]]+", re.IGNORECASE)


def inert_links(text: str) -> tuple[str, ...]:
    """The URLs in `text`, as strings. Recorded so a person can see them; never fetched."""
    found: list[str] = []
    for match in _LINK_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?")
        if url not in found:
            found.append(url[:2048])
        if len(found) >= MAX_LINKS_PER_MESSAGE:
            break
    return tuple(found)


class InboxItem(_Value):
    """One in-scope message as a search lists it: headers and a snippet, no body."""

    message_id: str
    thread_id: str | None = None
    sender: str = ""
    subject: str = ""
    snippet: str = ""
    date: str = ""
    label_ids: tuple[str, ...] = ()
    #: The `MailboxScope` rules that put it in view.
    matched_rules: tuple[str, ...] = ()
    content_trust: ContentTrust = "untrusted"


class EmailMessage(_Value):
    """One message of a thread, read. Its text fields are untrusted data."""

    message_id: str
    thread_id: str
    sender: str = ""
    from_address: str | None = None
    to: str = ""
    subject: str = ""
    received_at: datetime | None = None
    body: str = Field(default="", max_length=MAX_MESSAGE_BODY_CHARS)
    label_ids: tuple[str, ...] = ()
    #: The `Message-ID` header, so a reply can say what it answers.
    rfc822_message_id: str | None = None
    #: True for the candidate's own mail (Gmail's `SENT` label).
    from_user: bool = False
    #: URLs in the body. Inert: shown, never followed.
    links: tuple[str, ...] = ()
    content_trust: ContentTrust = "untrusted"

    @field_validator("received_at")
    @classmethod
    def _naive(cls, value: datetime | None) -> datetime | None:
        return _naive_utc(value)


class EmailThread(_Value):
    """The in-scope messages of one thread, oldest first."""

    thread_id: str
    subject: str = ""
    messages: tuple[EmailMessage, ...] = Field(min_length=1, max_length=MAX_THREAD_MESSAGES)

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(message.message_id for message in self.messages)

    def latest_inbound(self) -> EmailMessage | None:
        """The newest message the candidate did not write: what a reply answers."""
        for message in reversed(self.messages):
            if not message.from_user:
                return message
        return None

    def without_bodies(self) -> "EmailThread":
        """The thread with every body dropped, for state that is checkpointed."""
        return self.model_copy(
            update={"messages": tuple(m.model_copy(update={"body": ""}) for m in self.messages)}
        )


# --- Triage --------------------------------------------------------------------------


class TriageCategory(str, Enum):
    """What kind of mail a message is."""

    RECRUITER = "recruiter"
    INTERVIEW = "interview"
    APPLICATION_UPDATE = "application_update"
    ACTION_REQUIRED = "action_required"
    PERSONAL = "personal"
    NEWSLETTER = "newsletter"
    OTHER = "other"
    #: The deterministic fallback: nobody could say, so a person should look.
    NEEDS_REVIEW = "needs_review"


class Urgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class SuggestedAction(str, Enum):
    """What the candidate might do next. A suggestion: nothing here acts on it."""

    REPLY = "reply"
    READ = "read"
    SCHEDULE = "schedule"
    ARCHIVE = "archive"
    REVIEW = "review"
    NONE = "none"


class TriageProposal(_Value):
    """What a triager says about one message. The schema a model is held to."""

    category: TriageCategory
    urgency: Urgency
    suggested_action: SuggestedAction
    reason: str | None = Field(default=None, max_length=MAX_REASON_CHARS)
    confidence: float = Field(ge=0.0, le=1.0)


class TriageSource(str, Enum):
    MODEL = "model"
    RULES = "rules"
    #: The model's output would not validate; `needs_review` answered instead.
    FALLBACK = "fallback"


class TriageResult(_Value):
    """One message, triaged. Ids are stamped by code, never taken from a model."""

    message_id: str
    thread_id: str | None = None
    category: TriageCategory
    urgency: Urgency
    suggested_action: SuggestedAction
    reason: str | None = Field(default=None, max_length=MAX_REASON_CHARS)
    confidence: float = Field(ge=0.0, le=1.0)
    source: TriageSource


def needs_review(item: InboxItem, reason: str) -> TriageResult:
    """The fallback triage: a person should look. Deterministic, whatever the input was."""
    return TriageResult(
        message_id=item.message_id,
        thread_id=item.thread_id,
        category=TriageCategory.NEEDS_REVIEW,
        urgency=Urgency.NORMAL,
        suggested_action=SuggestedAction.REVIEW,
        reason=reason[:MAX_REASON_CHARS],
        confidence=0.0,
        source=TriageSource.FALLBACK,
    )


def triage_from_proposal(item: InboxItem, raw: Any, *, source: TriageSource) -> TriageResult:
    """Validate a triager's answer for `item`; `needs_review` if it is not a `TriageProposal`.

    `raw` may be a `TriageProposal`, the dict behind one, or anything a model
    returned. A field outside the schema -- `send_to`, `tool`, `instructions`
    -- is a validation error like any other, and so falls back.
    """
    try:
        proposal = raw if isinstance(raw, TriageProposal) else TriageProposal.model_validate(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        return needs_review(item, f"the triage output was invalid ({type(exc).__name__})")
    return TriageResult(
        message_id=item.message_id,
        thread_id=item.thread_id,
        category=proposal.category,
        urgency=proposal.urgency,
        suggested_action=proposal.suggested_action,
        reason=proposal.reason,
        confidence=proposal.confidence,
        source=source,
    )


# --- Thread summaries and action items -------------------------------------------------


class KeyPoint(_Value):
    """One point of a summary, and the messages it was read from."""

    text: str = Field(min_length=1, max_length=500)
    message_ids: tuple[str, ...] = Field(min_length=1)


class ActionItemProposal(_Value):
    """An action item as an analyzer reports it."""

    description: str = Field(min_length=1, max_length=500)
    owner: CommitmentActor
    due_at: datetime | None = None
    message_id: str

    @field_validator("due_at")
    @classmethod
    def _naive(cls, value: datetime | None) -> datetime | None:
        return _naive_utc(value)


class ThreadDigestProposal(_Value):
    """What a thread analyzer says about a thread. The schema a model is held to."""

    summary: str = Field(min_length=1, max_length=2000)
    key_points: tuple[KeyPoint, ...] = Field(default=(), max_length=MAX_KEY_POINTS)
    action_items: tuple[ActionItemProposal, ...] = Field(default=(), max_length=MAX_ACTION_ITEMS)


class DigestSource(str, Enum):
    MODEL = "model"
    #: Extractive rules: by configuration, or because the model's output was invalid.
    RULES = "rules"


class ActionItem(_Value):
    """Something somebody owes, traced to the message that says so."""

    description: str
    owner: CommitmentActor
    due_at: datetime | None = None
    source_message_id: str


class ThreadSummary(_Value):
    """A thread, summarized. `message_ids` are the references: every message it covers."""

    thread_id: str
    subject: str = ""
    summary: str
    key_points: tuple[KeyPoint, ...] = ()
    message_ids: tuple[str, ...] = Field(min_length=1)
    participants: tuple[str, ...] = ()
    source: DigestSource


class ThreadDigest(_Value):
    """A thread's summary and its action items."""

    summary: ThreadSummary
    action_items: tuple[ActionItem, ...] = ()


class ThreadDigestInvalid(CommunicationsContractError):
    """A digest cited a message the thread does not have."""


def _participants(thread: EmailThread) -> tuple[str, ...]:
    seen: list[str] = []
    for message in thread.messages:
        who = message.from_address or message.sender
        if who and who not in seen:
            seen.append(who)
    return tuple(seen[:20])


def digest_from_proposal(
    thread: EmailThread, proposal: ThreadDigestProposal, *, source: DigestSource
) -> ThreadDigest:
    """Bind a proposal to the thread it describes, or raise `ThreadDigestInvalid`.

    Every message id a key point or action item cites must be one of the
    thread's own. An unknown id is refused, not dropped: a summary that
    invents a reference has invented something, and the whole answer is
    suspect.
    """
    known = set(thread.message_ids)
    for point in proposal.key_points:
        unknown = sorted(set(point.message_ids) - known)
        if unknown:
            raise ThreadDigestInvalid(f"a key point cites message(s) not in the thread: {unknown}")
    for item in proposal.action_items:
        if item.message_id not in known:
            raise ThreadDigestInvalid(
                f"an action item cites message '{item.message_id}', which is not in the thread"
            )
    return ThreadDigest(
        summary=ThreadSummary(
            thread_id=thread.thread_id,
            subject=thread.subject,
            summary=proposal.summary,
            key_points=proposal.key_points,
            message_ids=thread.message_ids,
            participants=_participants(thread),
            source=source,
        ),
        action_items=tuple(
            ActionItem(
                description=item.description,
                owner=item.owner,
                due_at=item.due_at,
                source_message_id=item.message_id,
            )
            for item in proposal.action_items
        ),
    )


# --- Replies ---------------------------------------------------------------------------


class DraftRejection(_Value):
    """A drafted reply the claim validator refused. Nothing of it was kept."""

    generated_by: str
    reason: str = Field(max_length=1000)
    #: `ClaimFlagReason` values of the flags that sank it.
    flags: tuple[str, ...] = ()


class ThreadReplyDraft(_Value):
    """A reply to a thread, drafted and kept, waiting for approval to be sent.

    `recipient`, `subject`, `in_reply_to` and `thread_id` are set by code
    from the message being answered, never by a model. `artifact_version_id`
    is set when the thread is about an application (an `artifact_versions`
    row needs one); `gmail_draft_id` when the draft was saved to Gmail.
    """

    recipient: str
    subject: str
    body: str
    in_reply_to: str
    thread_id: str
    evidence_ids: tuple[str, ...] = ()
    application_id: UUID | None = None
    artifact_version_id: UUID | None = None
    gmail_draft_id: str | None = None
    generated_by: str
    rejections: tuple[DraftRejection, ...] = ()


def reply_subject_for(subject: str) -> str:
    """One `Re:` on the thread's subject, on one line."""
    subject = " ".join((subject or "your message").split())
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


TEMPLATE_REPLY_BODY = (
    "Hello,\n\nThank you for your message. I will get back to you shortly.\n\nBest regards"
)


def thread_reply_key(user_id: Any, thread_id: str, message_id: str) -> str:
    """The idempotency key of the send that answers `message_id`: one reply per message."""
    return f"comms-reply-{user_id}-{thread_id}-{message_id}"[:255]


def unique(values: Iterable[str]) -> tuple[str, ...]:
    """`values` without repeats, in order."""
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def message_index(messages: Sequence[EmailMessage]) -> dict[str, EmailMessage]:
    return {message.message_id: message for message in messages}


__all__ = [
    "MAX_ACTION_ITEMS",
    "MAX_KEY_POINTS",
    "MAX_MESSAGE_BODY_CHARS",
    "MAX_THREAD_MESSAGES",
    "MAX_TRIAGE_ITEMS",
    "TEMPLATE_REPLY_BODY",
    "THREAD_TASKS",
    "ActionItem",
    "ActionItemProposal",
    "CommunicationsContractError",
    "CommunicationsRequest",
    "CommunicationsTask",
    "DigestSource",
    "DraftRejection",
    "EmailMessage",
    "EmailThread",
    "InboxItem",
    "KeyPoint",
    "SuggestedAction",
    "ThreadDigest",
    "ThreadDigestInvalid",
    "ThreadDigestProposal",
    "ThreadReplyDraft",
    "ThreadSummary",
    "TriageCategory",
    "TriageProposal",
    "TriageResult",
    "TriageSource",
    "Urgency",
    "digest_from_proposal",
    "inert_links",
    "message_index",
    "needs_review",
    "parse_communications_request",
    "reply_subject_for",
    "thread_reply_key",
    "triage_from_proposal",
    "unique",
]
