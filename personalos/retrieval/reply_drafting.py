"""Reply drafting: ground a reply in evidence, write it, check it, keep it.

`GroundedReplyDrafter` satisfies the Job Search graph's `RecruiterReplyDrafter`
port. For one recorded recruiter event it

1. **grounds** the reply (`ReplyGroundingSource`): the application it is
   about, the candidate's evidence nearest the message, and the names a reply
   may use without that being a claim (the company, the role, the recruiter);
2. **writes** the body through the `ReplyWriter` port -- a model, given those
   records and nothing else about the candidate, returning segments that cite
   them (`personalos.models.reply_drafting`);
3. **validates** it with `personalos.domain.outbound_email.finalize_reply`, the
   Phase E claim validator. A reply with an untraceable claim is rejected:
   never stored, never saved to Gmail, never proposed. The deterministic
   template reply -- which states nothing about the candidate -- is drafted in
   its place, unless the drafter was built with `fallback_on_rejection=False`,
   in which case `UntraceableClaim` propagates;
4. **keeps** what survives, twice: as an `artifact_versions` row
   (`ReplyDraftStore`, type `recruiter_reply`) linked to the evidence records
   it cites and to the message it answers, and as a draft in the candidate's
   Gmail Drafts folder (`ReplyDraftMailbox`). Both are `WRITE_REVERSIBLE`.

It stops there. The `ReplyDraft` it returns becomes a `SEND_RECRUITER_MESSAGE`
intent that goes through the approval interrupt; nothing in this module can
send, and the recipient and subject it returns were set by code from the
message being answered, not by the model.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

from personalos.domain.artifacts import EXCERPT_CHARS, UntraceableClaim
from personalos.domain.errors import FailureClass
from personalos.domain.job_search import EvidenceRecord, EvidenceRef, JobSearchContractError
from personalos.domain.outbound_email import (
    ReplyApplication,
    ReplyGrounding,
    ReplyProposal,
    draft_idempotency_key,
    finalize_reply,
)
from personalos.domain.recruiter_events import (
    RecruiterEvent,
    ReplyDraft,
    reply_recipient,
    reply_subject,
    template_reply,
)
from personalos.persistence.evidence import SqlEvidenceSource
from personalos.retrieval.artifact_prep import Embedder, EvidenceIndex

logger = logging.getLogger(__name__)

#: `artifact_versions.evidence` type of the entry linking a reply to the message it answers.
RECRUITER_MESSAGE_EVIDENCE = "recruiter_message"

#: `generated_by` of a reply drafted from the template.
TEMPLATE_GENERATOR = "reply_drafter:template"

#: Evidence records a reply is written from.
DEFAULT_REPLY_TOP_K = 6

#: Message characters embedded as the retrieval query.
MAX_QUERY_CHARS = 4000


class ReplyGroundingSource(Protocol):
    """Looks up the application a message is about and the evidence for a reply."""

    async def ground(self, event: RecruiterEvent, application_id: UUID) -> ReplyGrounding: ...


class ReplyApplicationLookup(Protocol):
    """Finds the application a message was tied to. Implemented in `personalos.persistence`."""

    async def lookup(self, application_id: UUID) -> ReplyApplication: ...


class EvidenceReplyGrounding:
    """`ReplyGroundingSource`: the application, and the candidate's evidence nearest the message.

    Nearest by `embedder` over `index` when an embedder is wired; otherwise the
    first `top_k` of the candidate's evidence on file. Every lookup is scoped
    to the application's own user, so another candidate's record is never a
    source.
    """

    def __init__(
        self,
        applications: ReplyApplicationLookup,
        *,
        embedder: Embedder | None = None,
        index: EvidenceIndex | None = None,
        evidence_source: SqlEvidenceSource | None = None,
        top_k: int = DEFAULT_REPLY_TOP_K,
        min_similarity: float = 0.0,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if embedder is not None and index is None:
            raise ValueError("an embedder needs an index to search")
        if embedder is None and evidence_source is None:
            raise ValueError("without an embedder, EvidenceReplyGrounding needs an evidence source")
        self.applications = applications
        self.embedder = embedder
        self.index = index
        self.evidence_source = evidence_source
        self.top_k = top_k
        self.min_similarity = min_similarity

    async def ground(self, event: RecruiterEvent, application_id: UUID) -> ReplyGrounding:
        application = await self.applications.lookup(application_id)
        if self.embedder is not None and self.index is not None:
            message = event.message
            query = "\n".join(
                part for part in (message.subject or "", message.body[:MAX_QUERY_CHARS]) if part
            )
            retrieved = await self.index.search(
                user_id=application.user_id,
                query_embedding=await self.embedder.embed(query),
                embedding_model=self.embedder.model,
                top_k=self.top_k,
                min_similarity=self.min_similarity,
            )
            evidence = tuple(item.record for item in retrieved)
        else:
            assert self.evidence_source is not None
            evidence = tuple((await self.evidence_source.load(application.user_id))[: self.top_k])
        return ReplyGrounding(
            user_id=application.user_id,
            evidence=evidence,
            context_terms=application.context_terms(),
            protected_terms=tuple(skill.lower() for skill in application.skills),
            application=application.summary(),
        )


class ReplyWriter(Protocol):
    """Writes a reply body as segments citing the evidence they restate."""

    @property
    def name(self) -> str: ...

    async def write(
        self,
        event: RecruiterEvent,
        evidence: Sequence[EvidenceRecord],
        *,
        application: dict[str, Any] | None = None,
    ) -> ReplyProposal: ...


class ReplyDraftStore(Protocol):
    """Stores a reply as the next `recruiter_reply` version of an application's artifacts."""

    async def save(
        self,
        *,
        application_id: UUID,
        content: str,
        evidence: Sequence[EvidenceRef],
        generated_by: str,
    ) -> UUID: ...


class ReplyDraftMailbox(Protocol):
    """Saves a draft to the candidate's mailbox. Has no way to send one."""

    async def create_draft(
        self,
        *,
        recipient: str,
        subject: str,
        body: str,
        idempotency_key: str,
        thread_id: str | None = None,
    ) -> str: ...


def _commitment_terms(event: RecruiterEvent) -> list[str]:
    """The dates and times the message itself named, as a reply would restate them."""
    terms: list[str] = []
    for commitment in event.commitments:
        if commitment.due_at is None:
            continue
        due = commitment.due_at
        terms += [
            f"{due:%Y-%m-%d}",
            str(due.year),
            str(due.day),
            str(due.month),
            f"{due:%H:%M}",
            str(due.hour),
            str(due.hour % 12 or 12),
            f"{due:%M}",
        ]
    return terms


class GroundedReplyDrafter:
    """`RecruiterReplyDrafter`: an evidence-grounded, validated, stored reply draft."""

    def __init__(
        self,
        *,
        grounding: ReplyGroundingSource,
        writer: ReplyWriter,
        store: ReplyDraftStore,
        mailbox: ReplyDraftMailbox | None = None,
        fallback_on_rejection: bool = True,
        context_terms: Sequence[str] = (),
    ):
        """Wire the ports. Without a `mailbox` the reply is stored but not saved to Gmail.

        `context_terms` are names every reply may use without citing evidence
        -- the candidate's own name, typically.
        """
        if grounding is None or writer is None or store is None:
            raise ValueError("GroundedReplyDrafter requires a grounding source, writer and store")
        self.grounding = grounding
        self.writer = writer
        self.store = store
        self.mailbox = mailbox
        self.fallback_on_rejection = fallback_on_rejection
        self.context_terms = tuple(context_terms)

    async def draft(self, event: RecruiterEvent) -> ReplyDraft:
        """Return a proposed reply to `event`'s message, already stored."""
        application_id = event.correlation.application_id
        if application_id is None:
            raise JobSearchContractError(
                f"message {event.message.provider_message_id} is not tied to an application; "
                f"a reply is drafted only for a matched event"
            )
        grounding = await self.grounding.ground(event, application_id)
        records = {record.evidence_id: record for record in grounding.evidence}

        proposal = await self.writer.write(
            event, grounding.evidence, application=grounding.application
        )
        try:
            body, cited = finalize_reply(
                proposal,
                grounding.evidence,
                context_terms=self._context(event, grounding),
                protected_terms=grounding.protected_terms,
            )
            generated_by = self.writer.name
        except UntraceableClaim as rejected:
            if not self.fallback_on_rejection:
                raise
            logger.warning(
                "drafted reply to message %s was rejected and not kept (%s); drafting the "
                "template reply instead",
                event.message.provider_message_id,
                rejected,
            )
            body, cited, generated_by = template_reply(event).body, (), TEMPLATE_GENERATOR

        return await self._keep(
            event, application_id, body, [records[eid] for eid in cited], generated_by
        )

    def _context(self, event: RecruiterEvent, grounding: ReplyGrounding) -> list[str]:
        message = event.message
        sender = message.from_address or ""
        return [
            *self.context_terms,
            *grounding.context_terms,
            message.subject or "",
            sender,
            sender.split("@", 1)[0].replace(".", " "),
            *_commitment_terms(event),
        ]

    async def _keep(
        self,
        event: RecruiterEvent,
        application_id: UUID,
        body: str,
        cited: Sequence[EvidenceRecord],
        generated_by: str,
    ) -> ReplyDraft:
        message = event.message
        recipient, subject = reply_recipient(event), reply_subject(event)
        evidence = [
            *(
                EvidenceRef(
                    type=record.source_type.value,
                    ref=record.evidence_id,
                    excerpt=record.text[:EXCERPT_CHARS],
                )
                for record in cited
            ),
            EvidenceRef(
                type=RECRUITER_MESSAGE_EVIDENCE,
                ref=f"gmail:{message.provider_message_id}",
                excerpt=(message.subject or "")[:EXCERPT_CHARS] or None,
            ),
        ]
        version_id = await self.store.save(
            application_id=application_id,
            content=body,
            evidence=evidence,
            generated_by=generated_by,
        )

        draft_id: str | None = None
        if self.mailbox is not None and recipient:
            try:
                draft_id = await self.mailbox.create_draft(
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    idempotency_key=draft_idempotency_key(
                        f"reply-{application_id}-{message.provider_message_id}", body
                    ),
                    thread_id=message.thread_id,
                )
            except Exception as exc:
                if getattr(exc, "failure_class", None) is FailureClass.POLICY_DENIAL:
                    # A refusal is an answer about this system's configuration,
                    # not an outage; it is surfaced, not papered over.
                    raise
                # The stored version is the draft of record; the Gmail copy is a
                # convenience, and a Gmail outage is no reason to drop the reply.
                logger.exception(
                    "could not save the reply to message %s as a Gmail draft; it is stored as "
                    "artifact version %s",
                    message.provider_message_id,
                    version_id,
                )

        return ReplyDraft(
            recipient=recipient,
            subject=subject,
            body=body,
            evidence_ids=tuple(record.evidence_id for record in cited),
            artifact_version_id=version_id,
            gmail_draft_id=draft_id,
            generated_by=generated_by,
        )


__all__ = [
    "DEFAULT_REPLY_TOP_K",
    "EvidenceReplyGrounding",
    "RECRUITER_MESSAGE_EVIDENCE",
    "ReplyApplicationLookup",
    "TEMPLATE_GENERATOR",
    "GroundedReplyDrafter",
    "ReplyDraftMailbox",
    "ReplyDraftStore",
    "ReplyGroundingSource",
    "ReplyWriter",
]
