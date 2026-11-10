"""Outbound email: what a send may carry, how it is found again, and what a reply may claim.

Three rules live here, because they are properties of the email itself rather
than of whichever adapter sends it.

**A send is built from its approved payload and nothing else.** An email
leaves this system only as an approved `ActionIntent` of kind `SEND_EMAIL`
(or `SEND_RECRUITER_MESSAGE`). `OutboundEmail.from_intent` reads the message
out of that intent's payload -- the same payload `ActionIntent.fingerprint`
hashes and `ActionPreview` shows the reviewer -- and refuses a payload with
any key it does not know. There is no key for a sender, an account, a
credential, a `Bcc` or a header: whose mailbox sends is the gateway-attached
`ToolAuthorization`, and nothing a model wrote into a payload can reach it.
Changing the recipient, subject, body or attachments after approval changes
the hash, so the approval no longer authorizes the action
(`ApprovalDecision.authorizes`) and a new one has to be asked for.

**A send can be found again by its idempotency key.** Gmail's `messages.send`
is not idempotent: sent twice, it sends twice. So every message gets a
`Message-ID` derived from the action's idempotency key
(`outbound_message_id`), and Gmail's `rfc822msgid:` search over the Sent
folder answers "did this send land?" without the response it may have lost.

**A reply may not invent claims.** A drafted reply is model output, and so a
set of claims. `finalize_reply` holds it to the Phase E rule
(`personalos.domain.artifacts.validate_draft`): any checkable fact about the
candidate -- an employer, a tool, a number, a degree -- must appear in an
evidence record the segment cites. One difference from a tailored resume: a
reply that makes no claim about the candidate at all ("thank you, Tuesday at
10 works") is fine, so the whole-draft `NO_EVIDENCE` flag does not apply.
Every other flag rejects it.
"""

import calendar
import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from personalos.domain.artifacts import (
    ClaimFlagReason,
    DraftProposal,
    DraftSegment,
    DraftValidation,
    UntraceableClaim,
    validate_draft,
)
from personalos.domain.job_search import (
    PAYLOAD_ATTACHMENTS,
    PAYLOAD_BODY,
    PAYLOAD_IN_REPLY_TO,
    PAYLOAD_PREVIOUS_BODY,
    PAYLOAD_RECIPIENT,
    PAYLOAD_SUBJECT,
    ActionIntent,
    ActionKind,
    AttachmentPreview,
    EvidenceRecord,
    JobSearchContractError,
    _Value,
)
from personalos.domain.models import ArtifactType

#: The action kinds whose payload is an `OutboundEmail`.
EMAIL_ACTION_KINDS = frozenset({ActionKind.SEND_EMAIL, ActionKind.SEND_RECRUITER_MESSAGE})

#: The Gmail thread a reply belongs in, when the proposer knew it.
PAYLOAD_THREAD_ID = "thread_id"
#: The Gmail draft the message was saved as before anyone approved it.
PAYLOAD_GMAIL_DRAFT_ID = "gmail_draft_id"
#: The `artifact_versions` row the body was stored as.
PAYLOAD_ARTIFACT_VERSION_ID = "artifact_version_id"
#: The evidence records the body's claims cite.
PAYLOAD_EVIDENCE_IDS = "evidence_ids"

#: Every key an email action's payload may carry. Anything else -- `from`,
#: `cc`, `bcc`, `headers`, `credential`, `token`, `account` -- is refused, not
#: ignored: a payload that tries to say more than this is not one to send.
EMAIL_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        PAYLOAD_RECIPIENT,
        PAYLOAD_SUBJECT,
        PAYLOAD_BODY,
        PAYLOAD_PREVIOUS_BODY,
        PAYLOAD_ATTACHMENTS,
        PAYLOAD_IN_REPLY_TO,
        PAYLOAD_THREAD_ID,
        PAYLOAD_GMAIL_DRAFT_ID,
        PAYLOAD_ARTIFACT_VERSION_ID,
        PAYLOAD_EVIDENCE_IDS,
        # Bookkeeping the recruiter flows attach; never part of the message.
        "application_id",
        "classification",
        "checkpoint_id",
        "kind",
        "unmet_condition",
    }
)

#: Domain of the `Message-ID`s this system stamps. Not a real host: it only
#: has to be unique to this system, which a key-derived local part makes it.
MESSAGE_ID_DOMAIN = "personalos.local"

#: One mailbox address, the same shape `mcp_servers.google.schemas.EmailAddress` accepts.
_ADDRESS = re.compile(r"^[^\s<>()\[\],;:\"\\@]+@[^\s<>()\[\],;:\"\\@]+\.[^\s<>()\[\],;:\"\\@]+$")


def outbound_message_id(idempotency_key: str) -> str:
    """The RFC 5322 `Message-ID` a send with this idempotency key carries.

    Derived, not random, so a reconciler that has only the action can compute
    what to search the Sent folder for.
    """
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:40]
    return f"<pos-{digest}@{MESSAGE_ID_DOMAIN}>"


def sent_folder_query(idempotency_key: str) -> str:
    """The Gmail search that finds this send in the Sent folder, if it landed."""
    return f"in:sent rfc822msgid:{outbound_message_id(idempotency_key)}"


class OutboundEmail(_Value):
    """One email, exactly as the reviewer approved it."""

    recipient: str
    subject: str = Field(max_length=998)
    body: str
    #: The provider id of the message this answers; `None` starts a new thread.
    in_reply_to: str | None = None
    thread_id: str | None = None
    attachments: tuple[AttachmentPreview, ...] = ()

    @field_validator("recipient")
    @classmethod
    def _one_address(cls, value: str) -> str:
        if not _ADDRESS.match(value or ""):
            raise JobSearchContractError(
                f"an email goes to exactly one plain address; '{value}' is not one"
            )
        return value

    @field_validator("subject")
    @classmethod
    def _one_line(cls, value: str) -> str:
        if "\r" in value or "\n" in value:
            raise JobSearchContractError("an email subject is one line")
        return value

    @field_validator("body")
    @classmethod
    def _has_body(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("an email with no body is not one to send")
        return value

    @property
    def is_reply(self) -> bool:
        return bool(self.in_reply_to)

    @classmethod
    def from_intent(cls, intent: ActionIntent) -> "OutboundEmail":
        """Read the email out of an approved action's payload, or raise.

        Raises `JobSearchContractError` for an action that is not an email, a
        payload with a key outside `EMAIL_PAYLOAD_KEYS`, or a message that is
        incomplete (no recipient, no body).
        """
        if intent.kind not in EMAIL_ACTION_KINDS:
            raise JobSearchContractError(f"'{intent.kind.value}' is not an email action")
        payload = intent.payload
        unknown = sorted(set(payload) - EMAIL_PAYLOAD_KEYS)
        if unknown:
            raise JobSearchContractError(
                f"email action {intent.action_id} carries keys an email may not set: {unknown}"
            )
        for key in (PAYLOAD_RECIPIENT, PAYLOAD_SUBJECT, PAYLOAD_BODY):
            if not payload.get(key):
                raise JobSearchContractError(
                    f"email action {intent.action_id} has no '{key}'; there is no message to send"
                )
        in_reply_to = payload.get(PAYLOAD_IN_REPLY_TO)
        thread_id = payload.get(PAYLOAD_THREAD_ID)
        return cls(
            recipient=str(payload[PAYLOAD_RECIPIENT]),
            subject=str(payload[PAYLOAD_SUBJECT]),
            body=str(payload[PAYLOAD_BODY]),
            in_reply_to=str(in_reply_to) if in_reply_to else None,
            thread_id=str(thread_id) if thread_id else None,
            attachments=tuple(
                AttachmentPreview.model_validate(item)
                for item in payload.get(PAYLOAD_ATTACHMENTS) or ()
            ),
        )


def send_email_intent(
    *,
    recipient: str,
    subject: str,
    body: str,
    idempotency_key: str,
    requested_by: str,
    in_reply_to: str | None = None,
    thread_id: str | None = None,
    attachments: Sequence[AttachmentPreview] = (),
    summary: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ActionIntent:
    """Propose one email as a `SEND_EMAIL` action. Proposing is all this does.

    The message is validated first (`OutboundEmail`), so an action that could
    never be sent is never put to a reviewer. The target names the recipient
    and whether it is a reply, which `ActionIntent.fingerprint` covers too.
    """
    email = OutboundEmail(
        recipient=recipient,
        subject=subject,
        body=body,
        in_reply_to=in_reply_to,
        thread_id=thread_id,
        attachments=tuple(attachments),
    )
    payload: dict[str, Any] = {
        **(extra or {}),
        PAYLOAD_RECIPIENT: email.recipient,
        PAYLOAD_SUBJECT: email.subject,
        PAYLOAD_BODY: email.body,
    }
    if email.in_reply_to:
        payload[PAYLOAD_IN_REPLY_TO] = email.in_reply_to
    if email.thread_id:
        payload[PAYLOAD_THREAD_ID] = email.thread_id
    if email.attachments:
        payload[PAYLOAD_ATTACHMENTS] = [a.model_dump(mode="json") for a in email.attachments]
    unknown = sorted(set(payload) - EMAIL_PAYLOAD_KEYS)
    if unknown:
        raise JobSearchContractError(f"an email action may not carry {unknown}")
    where = (
        f"reply in thread {email.thread_id or email.in_reply_to}"
        if email.is_reply
        else ("new thread")
    )
    return ActionIntent(
        kind=ActionKind.SEND_EMAIL,
        target=f"{email.recipient} ({where})",
        summary=summary or f"Email {email.recipient}: {email.subject}",
        payload=payload,
        idempotency_key=idempotency_key,
        requested_by=requested_by,
    )


def draft_idempotency_key(base: str, body: str) -> str:
    """The key a Gmail draft is saved under: one draft per reply and per body."""
    return f"{base}:draft:{hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]}"[:255]


# --- Reply grounding ---------------------------------------------------------


class ReplyApplication(_Value):
    """What a reply needs to know about the application a message is about."""

    application_id: UUID
    user_id: UUID
    company: str
    title: str
    location: str | None = None
    #: The posting's skills: a reply naming one must cite evidence for it.
    skills: tuple[str, ...] = ()

    def context_terms(self) -> tuple[str, ...]:
        """Names a reply may use without that being a claim about the candidate."""
        return tuple(term for term in (self.company, self.title, self.location) if term)

    def summary(self) -> dict[str, str]:
        """What the writer is told about the application."""
        return {k: v for k, v in (("company", self.company), ("title", self.title)) if v}


class ReplyGrounding(_Value):
    """What a reply to one message may be written from."""

    user_id: UUID
    #: The only records a citation may resolve against.
    evidence: tuple[EvidenceRecord, ...] = ()
    #: Names that are not claims about the candidate: company, role, location.
    context_terms: tuple[str, ...] = ()
    #: Terms that count as a claim wherever they appear: the posting's skills.
    protected_terms: tuple[str, ...] = ()
    #: What the writer is told about the application (company, title).
    application: dict[str, str] = Field(default_factory=dict)


# --- Reply integrity ---------------------------------------------------------


#: Words a reply may use about *when* without that being a claim about the
#: candidate: "Thursday at 10 AM works" names no employer, tool or degree.
CALENDAR_TERMS: tuple[str, ...] = tuple(
    name for name in (*calendar.day_name, *calendar.month_name, *calendar.month_abbr) if name
) + ("AM", "PM", "UTC")


class ReplyProposal(_Value):
    """A reply body as a model wrote it: segments, each citing the evidence it restates.

    No recipient and no subject: those are set by code from the message being
    answered, so a model cannot choose where its text goes.
    """

    segments: tuple[DraftSegment, ...] = ()


def validate_reply(
    proposal: ReplyProposal,
    evidence: Sequence[EvidenceRecord],
    *,
    context_terms: Iterable[str] = (),
    protected_terms: Iterable[str] = (),
) -> DraftValidation:
    """`validate_draft` for a reply: every flag but the whole-draft `NO_EVIDENCE` counts.

    `CALENDAR_TERMS` are always context: day and month names are capitalized,
    and would otherwise read to the validator as proper nouns. A number is
    still checked -- "Thursday at 10" needs the 10 in the context (the
    drafter adds the times the message itself named), and "8 years" needs it
    in the cited evidence.
    """
    validation = validate_draft(
        DraftProposal(artifact_type=ArtifactType.RECRUITER_REPLY, segments=proposal.segments),
        evidence,
        context_terms=(*CALENDAR_TERMS, *context_terms),
        protected_terms=protected_terms,
    )
    return validation.model_copy(
        update={
            "flags": tuple(
                flag for flag in validation.flags if flag.reason is not ClaimFlagReason.NO_EVIDENCE
            )
        }
    )


def finalize_reply(
    proposal: ReplyProposal,
    evidence: Sequence[EvidenceRecord],
    *,
    context_terms: Iterable[str] = (),
    protected_terms: Iterable[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """The reply's body and the evidence ids it cites, or raise `UntraceableClaim`."""
    validation = validate_reply(
        proposal, evidence, context_terms=context_terms, protected_terms=protected_terms
    )
    if not validation.ok:
        raise UntraceableClaim(validation)
    body = "\n\n".join(s.text.strip() for s in proposal.segments if s.text.strip())
    if not body:
        raise JobSearchContractError("the drafted reply is empty")
    return body, validation.cited_evidence_ids


__all__ = [
    "CALENDAR_TERMS",
    "EMAIL_ACTION_KINDS",
    "EMAIL_PAYLOAD_KEYS",
    "MESSAGE_ID_DOMAIN",
    "PAYLOAD_ARTIFACT_VERSION_ID",
    "PAYLOAD_EVIDENCE_IDS",
    "PAYLOAD_GMAIL_DRAFT_ID",
    "PAYLOAD_THREAD_ID",
    "OutboundEmail",
    "ReplyApplication",
    "ReplyGrounding",
    "ReplyProposal",
    "draft_idempotency_key",
    "finalize_reply",
    "outbound_message_id",
    "send_email_intent",
    "sent_folder_query",
    "validate_reply",
]
