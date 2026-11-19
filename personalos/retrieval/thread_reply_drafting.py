"""Reply drafting for any in-scope thread: grounded, validated, kept, never sent.

`ThreadReplyDrafter` satisfies the Communications subgraph's `ReplyDrafter`
port. It is the J5 recruiter-reply pipeline (`retrieval.reply_drafting`)
generalised from "a recruiter event on an application" to "the latest message
of a thread", and it reuses that pipeline's pieces rather than restating them:

1. **Who and what.** The reply goes to the sender of the thread's latest
   inbound message and nobody else, under the thread's subject with one
   `Re:`. Both are set here, by code; the writer is never asked.
2. **Grounding.** The candidate's own evidence records (and, when the thread
   is about one of their applications, that application's company, role and
   skills via `ReplyApplicationLookup`). An application belonging to another
   user is refused: a draft is grounded in its own candidate's records only.
3. **Writing and checking.** The writer returns a `ReplyProposal`, which
   `finalize_reply` -- the Phase E claim validator -- checks against the
   evidence. A rejected draft is recorded as a `DraftRejection` and nothing of
   it is kept: not stored, not saved to Gmail, not proposed. The neutral
   template reply, which states nothing about the candidate, takes its place.
4. **Keeping** (`WRITE_REVERSIBLE`, both): an `artifact_versions` row when the
   thread is about an application (that table hangs off one), and a draft in
   the candidate's Gmail Drafts folder.

Nothing here can send. The `ThreadReplyDraft` it returns becomes a
`SEND_EMAIL` intent that waits for a person's approval.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

from personalos.domain.artifacts import EXCERPT_CHARS, UntraceableClaim
from personalos.domain.communications import (
    TEMPLATE_REPLY_BODY,
    CommunicationsContractError,
    DraftRejection,
    EmailThread,
    ThreadReplyDraft,
    reply_subject_for,
    thread_reply_key,
)
from personalos.domain.errors import FailureClass
from personalos.domain.job_search import EvidenceRecord, EvidenceRef
from personalos.domain.outbound_email import (
    ReplyApplication,
    ReplyProposal,
    draft_idempotency_key,
    finalize_reply,
)
from personalos.retrieval.reply_drafting import (
    ReplyApplicationLookup,
    ReplyDraftMailbox,
    ReplyDraftStore,
)

logger = logging.getLogger(__name__)

#: `artifact_versions.evidence` type of the entry linking a reply to the message it answers.
EMAIL_MESSAGE_EVIDENCE = "email_message"

#: `generated_by` of a reply drafted from the template.
THREAD_TEMPLATE_GENERATOR = "thread_reply_drafter:template"

DEFAULT_TOP_K = 6


class CandidateEvidence(Protocol):
    """The candidate's evidence records. `persistence.evidence.SqlEvidenceSource`."""

    async def load(self, user_id: UUID) -> list[EvidenceRecord]: ...


class ThreadReplyWriterPort(Protocol):
    """`personalos.models.communications.ThreadReplyWriter`, as this layer sees it."""

    @property
    def name(self) -> str: ...

    async def write(
        self,
        thread: EmailThread,
        evidence: Sequence[EvidenceRecord],
        *,
        application: dict[str, Any] | None = None,
    ) -> ReplyProposal: ...


class ThreadReplyDrafter:
    """A validated reply to a thread's latest inbound message, stored and saved as a draft."""

    def __init__(
        self,
        *,
        evidence: CandidateEvidence,
        writer: ThreadReplyWriterPort | None = None,
        applications: ReplyApplicationLookup | None = None,
        store: ReplyDraftStore | None = None,
        mailbox: ReplyDraftMailbox | None = None,
        top_k: int = DEFAULT_TOP_K,
        context_terms: Sequence[str] = (),
    ):
        """Wire the ports. Without a `writer` every reply is the template.

        Without a `store`, nothing is written to `artifact_versions`; without a
        `mailbox`, nothing is saved to Gmail.
        """
        if evidence is None:
            raise ValueError("ThreadReplyDrafter requires an evidence source")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.evidence = evidence
        self.writer = writer
        self.applications = applications
        self.store = store
        self.mailbox = mailbox
        self.top_k = top_k
        self.context_terms = tuple(context_terms)

    async def draft(
        self,
        thread: EmailThread,
        *,
        user_id: UUID,
        application_id: UUID | None = None,
    ) -> ThreadReplyDraft:
        target = thread.latest_inbound()
        if target is None or not target.from_address:
            raise CommunicationsContractError(
                f"thread {thread.thread_id} has no inbound message with a sender to reply to"
            )
        application = await self._application(application_id, user_id)
        evidence = list(await self.evidence.load(user_id))[: self.top_k]
        records = {record.evidence_id: record for record in evidence}

        body, cited, generated_by, rejections = await self._write(thread, evidence, application)
        recipient = target.from_address
        subject = reply_subject_for(thread.subject or target.subject)

        version_id: UUID | None = None
        if application is not None and self.store is not None:
            version_id = await self.store.save(
                application_id=application.application_id,
                content=body,
                evidence=[
                    *(
                        EvidenceRef(
                            type=records[eid].source_type.value,
                            ref=eid,
                            excerpt=records[eid].text[:EXCERPT_CHARS],
                        )
                        for eid in cited
                    ),
                    EvidenceRef(
                        type=EMAIL_MESSAGE_EVIDENCE,
                        ref=f"gmail:{target.message_id}",
                        excerpt=(target.subject or "")[:EXCERPT_CHARS] or None,
                    ),
                ],
                generated_by=generated_by,
            )

        draft_id: str | None = None
        if self.mailbox is not None:
            try:
                draft_id = await self.mailbox.create_draft(
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    idempotency_key=draft_idempotency_key(
                        thread_reply_key(user_id, thread.thread_id, target.message_id), body
                    ),
                    thread_id=thread.thread_id,
                )
            except Exception as exc:
                if getattr(exc, "failure_class", None) is FailureClass.POLICY_DENIAL:
                    raise
                logger.exception(
                    "could not save the reply on thread %s as a Gmail draft; it is kept as %s",
                    thread.thread_id,
                    version_id or "the proposed action only",
                )

        return ThreadReplyDraft(
            recipient=recipient,
            subject=subject,
            body=body,
            in_reply_to=target.message_id,
            thread_id=thread.thread_id,
            evidence_ids=tuple(cited),
            application_id=application.application_id if application else None,
            artifact_version_id=version_id,
            gmail_draft_id=draft_id,
            generated_by=generated_by,
            rejections=tuple(rejections),
        )

    async def _application(
        self, application_id: UUID | None, user_id: UUID
    ) -> ReplyApplication | None:
        if application_id is None or self.applications is None:
            return None
        application = await self.applications.lookup(application_id)
        if application.user_id != user_id:
            raise CommunicationsContractError(
                f"application {application_id} is not this candidate's; a reply is grounded "
                f"in its own candidate's records only"
            )
        return application

    async def _write(
        self,
        thread: EmailThread,
        evidence: Sequence[EvidenceRecord],
        application: ReplyApplication | None,
    ) -> tuple[str, tuple[str, ...], str, list[DraftRejection]]:
        rejections: list[DraftRejection] = []
        if self.writer is None:
            return TEMPLATE_REPLY_BODY, (), THREAD_TEMPLATE_GENERATOR, rejections
        try:
            proposal = await self.writer.write(
                thread, evidence, application=application.summary() if application else None
            )
        except Exception as exc:  # noqa: BLE001 - no reply text is not a reason to fail the task
            logger.warning(
                "the reply writer failed on thread %s (%s); drafting the template reply",
                thread.thread_id,
                type(exc).__name__,
            )
            return TEMPLATE_REPLY_BODY, (), THREAD_TEMPLATE_GENERATOR, rejections
        try:
            body, cited = finalize_reply(
                proposal,
                evidence,
                context_terms=self._context(thread, application),
                protected_terms=(
                    tuple(skill.lower() for skill in application.skills) if application else ()
                ),
            )
            return body, tuple(cited), self.writer.name, rejections
        except UntraceableClaim as rejected:
            logger.warning(
                "the drafted reply on thread %s was rejected and not kept (%s); drafting the "
                "template reply instead",
                thread.thread_id,
                rejected,
            )
            rejections.append(
                DraftRejection(
                    generated_by=self.writer.name,
                    reason=str(rejected)[:1000],
                    flags=tuple(flag.reason.value for flag in rejected.validation.flags),
                )
            )
            return TEMPLATE_REPLY_BODY, (), THREAD_TEMPLATE_GENERATOR, rejections

    def _context(self, thread: EmailThread, application: ReplyApplication | None) -> list[str]:
        terms = [*self.context_terms, thread.subject]
        for message in thread.messages:
            sender = message.from_address or ""
            terms += [message.subject, sender, sender.split("@", 1)[0].replace(".", " ")]
            name = message.sender.split("<", 1)[0].strip().strip('"')
            if name:
                terms.append(name)
        if application is not None:
            terms += list(application.context_terms())
        return [term for term in terms if term]


__all__ = [
    "DEFAULT_TOP_K",
    "EMAIL_MESSAGE_EVIDENCE",
    "THREAD_TEMPLATE_GENERATOR",
    "CandidateEvidence",
    "ThreadReplyDrafter",
    "ThreadReplyWriterPort",
]
