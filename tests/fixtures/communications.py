"""A Communications subgraph over the real Gmail tool path, against a fake Google API.

Nothing is faked between the graph and Google's JSON: the mailbox is
`GmailThreadReader`, drafts are saved by `GmailOutbound` as the `draft_reply`
node, and approved sends are carried out by `GmailSendExecutor` -- all
dispatching `google.gmail_*` intents through the policy gateway (the default
rule chain, with the production per-node allowlists) to an in-process
`GoogleMCPServer` over `FakeGoogleApi`. Recruiter mail is recorded by the
real `SqlRecruiterEventStore` on SQLite, and replies are stored by the real
`SqlReplyDraftStore`.

A test reads outcomes off `world.google.api` (drafts, sends, calls), the
database, and the graph's final state.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from personalos.domain.artifacts import DraftSegment
from personalos.domain.job_search import (
    ActionIntent,
    ActionReceipt,
    ApprovalDecision,
    ApprovalVerdict,
    EvidenceRecord,
)
from personalos.domain.mailbox import MailboxScope
from personalos.domain.models import ApplicationStatus, EvidenceSourceType
from personalos.domain.outbound_email import ReplyProposal
from personalos.executor.gmail_outbound import GmailOutbound, GmailSendExecutor
from personalos.executor.gmail_threads import GmailThreadReader
from personalos.graphs.communications import DRAFT_REPLY, CommunicationsGraph, requester
from personalos.models.recruiter_events import RuleBasedRecruiterEventExtractor
from personalos.persistence.models import Base, UserModel
from personalos.persistence.recruiter_events import SqlRecruiterEventStore
from personalos.persistence.reply_drafts import SqlReplyApplicationLookup, SqlReplyDraftStore
from personalos.persistence.repositories import ApplicationRepository, JobPostingRepository
from personalos.policy.rules import GOOGLE_NODE_TOOL_ALLOWLISTS
from personalos.retrieval.thread_reply_drafting import ThreadReplyDrafter
from tests.fixtures.fake_google import GoogleHarness, gmail_message

NOW = datetime(2026, 11, 19, 12, 0)
LABEL = "Recruiting"
RECRUITER = "priya@acme.example"
REFERENCE = "REQ-7781"

EVIDENCE = (
    EvidenceRecord(
        evidence_id="ev-pipelines",
        source_type=EvidenceSourceType.RESUME,
        source_ref="resume#experience",
        text="Built Python data pipelines at Initech processing 2 million events a day.",
    ),
)

INVITE_BODY = (
    "Hi Sam,\n\nThanks for applying to REQ-7781. We would like to schedule a technical "
    "interview with the team. Please send your availability by Friday.\n\n"
    "Details: https://acme.example/careers/interview-prep\n\nPriya"
)


def sqlite_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def applied_application(factory, *, company: str = "Acme", reference: str = REFERENCE):
    """An APPLIED application for a new user. Returns `(user_id, application_id)`."""
    session = factory()
    try:
        user = UserModel(email=f"candidate-{uuid4()}@example.com")
        session.add(user)
        session.commit()
        posting = JobPostingRepository(session).create(
            source="greenhouse",
            source_job_id=reference,
            title="Senior Backend Engineer",
            company=company,
            description_hash="a" * 64,
            dedupe_key=f"posting:{uuid4()}",
        )
        applications = ApplicationRepository(session)
        application = applications.create(job_posting_id=posting.id, user_id=user.id)
        for status in (
            ApplicationStatus.SAVED,
            ApplicationStatus.PREPARING,
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.APPLIED,
        ):
            applications.update_status(application.id, status, now=NOW - timedelta(days=3))
        return user.id, application.id
    finally:
        session.close()


def new_user(factory) -> UUID:
    session = factory()
    try:
        user = UserModel(email=f"candidate-{uuid4()}@example.com")
        session.add(user)
        session.commit()
        return user.id
    finally:
        session.close()


class StaticEvidence:
    def __init__(self, records: Sequence[EvidenceRecord] = EVIDENCE):
        self.records = list(records)
        self.calls: list[UUID] = []

    async def load(self, user_id: UUID) -> list[EvidenceRecord]:
        self.calls.append(user_id)
        return list(self.records)


class ScriptedReplyWriter:
    """A reply writer that returns fixed segments, and records what it was shown."""

    name = "thread_reply_drafter:scripted"

    def __init__(self, *segments: DraftSegment):
        self.segments = segments or (
            DraftSegment(text="Hello Priya,"),
            DraftSegment(text="Thank you, I would be glad to interview."),
            DraftSegment(
                text="I built Python data pipelines at Initech.", evidence_ids=("ev-pipelines",)
            ),
            DraftSegment(text="Best regards"),
        )
        self.threads: list[Any] = []

    async def write(self, thread, evidence, *, application=None) -> ReplyProposal:
        self.threads.append(thread)
        return ReplyProposal(segments=self.segments)


class SpyExecutor:
    """Wraps the real send executor and records every call it is handed."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.calls: list[tuple[ActionIntent, ApprovalDecision]] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.calls.append((intent, decision))
        return await self.inner.execute(intent, decision)


@dataclass
class CommsWorld:
    google: GoogleHarness
    factory: Any
    graph: CommunicationsGraph
    reader: GmailThreadReader
    executor: SpyExecutor
    writer: ScriptedReplyWriter
    store: SqlRecruiterEventStore
    label_id: str
    extra: dict[str, Any] = field(default_factory=dict)

    def compile(self):
        return self.graph.build()

    def add_mail(
        self,
        message_id: str,
        *,
        thread_id: str = "t-acme",
        sender: str = f"Priya <{RECRUITER}>",
        subject: str = f"Interview for {REFERENCE} at Acme",
        body: str = INVITE_BODY,
        labelled: bool = True,
        labels: Sequence[str] = ("INBOX",),
        minute: int = 0,
    ) -> None:
        all_labels = [*labels, self.label_id] if labelled else list(labels)
        stamp = int((datetime(2026, 11, 18, 9, minute)).timestamp() * 1000)
        self.google.api.add_message(
            gmail_message(
                message_id,
                sender=sender,
                subject=subject,
                body=body,
                thread_id=thread_id,
                labels=all_labels,
                extra={"internalDate": str(stamp)},
            )
        )

    def drafts(self) -> list[dict[str, Any]]:
        return list(self.google.api.drafts.values())

    def sends(self) -> list[Any]:
        return [c for c in self.google.api.calls if c.url.endswith("/messages/send")]


async def build_world(*, variant: str = "in_process", **kwargs: Any) -> CommsWorld:
    """The world over a started harness: in-process, or over MCP stdio."""
    google = await GoogleHarness(variant, node_allowlists=GOOGLE_NODE_TOOL_ALLOWLISTS).start()
    return make_world(google, **kwargs)


def build_world_in_process(**kwargs: Any) -> CommsWorld:
    """`build_world` for synchronous callers: the in-process harness needs no start-up I/O."""
    google = GoogleHarness("in_process", node_allowlists=GOOGLE_NODE_TOOL_ALLOWLISTS)
    google.manager.register_server(google.server)
    return make_world(google, **kwargs)


def make_world(
    google: GoogleHarness,
    *,
    factory: Any = None,
    writer: ScriptedReplyWriter | None = None,
    triager: Any = None,
    analyzer: Any = None,
    evidence: Sequence[EvidenceRecord] = EVIDENCE,
) -> CommsWorld:
    google.api.evaluate_queries = True
    label_id = google.api.add_label(LABEL)
    factory = factory or sqlite_factory()
    reader = GmailThreadReader(google.gateway, scope=MailboxScope(labels=(LABEL,)))
    writer = writer or ScriptedReplyWriter()
    store = SqlRecruiterEventStore(factory, clock=lambda: NOW)
    drafter = ThreadReplyDrafter(
        evidence=StaticEvidence(evidence),
        writer=writer,
        applications=SqlReplyApplicationLookup(factory),
        store=SqlReplyDraftStore(factory),
        mailbox=GmailOutbound(google.gateway, drafting_node=requester(DRAFT_REPLY)),
        context_terms=("Sam",),
    )
    executor = SpyExecutor(GmailSendExecutor(GmailOutbound(google.gateway)))
    graph = CommunicationsGraph(
        reader,
        triager=triager,
        analyzer=analyzer,
        reply_drafter=drafter,
        action_executor=executor,
        recruiter_event_extractor=RuleBasedRecruiterEventExtractor(),
        application_directory=store,
        recruiter_event_recorder=store,
        clock=lambda: NOW,
    )
    return CommsWorld(google, factory, graph, reader, executor, writer, store, label_id)


def thread(name: str = "comms") -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{name}-{uuid4()}"}}


def approve(request: dict[str, Any], *, decided_by: str = "sam") -> ApprovalDecision:
    """A person's approval of the `ApprovalRequest` in an interrupt payload."""
    return ApprovalDecision(
        action_id=request["action_id"],
        action_fingerprint=request["action_hash"],
        verdict=ApprovalVerdict.APPROVED,
        decided_by=decided_by,
        decided_at=NOW,
        request_id=request["request_id"],
    )


def interrupt_requests(final: dict[str, Any]) -> list[dict[str, Any]]:
    (pending,) = final["__interrupt__"]
    return pending.value["requests"]
