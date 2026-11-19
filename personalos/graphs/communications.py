"""The Communications subgraph: triage, summarize, extract action items, draft a reply.

Pipeline:

    parse_request -> triage_inbox -> END                                  (triage)
    parse_request -> load_thread -> link_application -> END               (summary, action items)
                  -> load_thread -> link_application -> draft_reply
                  -> request_send_approval -> send_approval_checkpoint (interrupts)
                  -> execute_send -> END                                  (draft a reply)

It builds on what Phase J already has -- the Gmail scope the inbox poller
watches (J3), the Google MCP tools (J2), and reply drafting and the
exactly-once send executor (J5) -- and adds no provider code.

**Who may touch the mailbox, and how.** This module holds no gateway and no
token. Its ports (`MailboxReader`, `ReplyDrafter`, `ActionExecutor`) are bound
by the composition root to adapters that dispatch `google.gmail_*` intents
through the policy gateway, each stamped with the `requested_by` of the node
that makes it. `COMMUNICATIONS_NODE_TOOL_ALLOWLISTS` gives triage and
`load_thread` read tools, `draft_reply` reads and `gmail_create_draft`, and no
node `gmail_send_message`: a reply leaves only as an approved `SEND_EMAIL`
action, carried out by the send executor as `GMAIL_SENDER` after
`execute_send` has checked the approval.

| Effect | Class | Where |
| --- | --- | --- |
| Inbox listings, thread reads | `READ_EXTERNAL` | `triage_inbox`, `load_thread`, `draft_reply` |
| Gmail draft, `artifact_versions` row | `WRITE_REVERSIBLE` | `draft_reply` |
| Send | `WRITE_EXTERNAL`, `require_approval` | `execute_send`, only after approval |

**Mail is data.** Message text reaches the triager, the thread analyzer, the
recruiter-event extractor and the reply writer as quoted data, and comes back
as closed schemas -- `TriageResult`, `ThreadDigest`, `ExtractionOutcome`,
`ReplyProposal` -- none of which has a field for a tool, a recipient or a new
step. Routing reads only the parsed request and what the nodes concluded,
never message text. Bodies are not checkpointed: `load_thread` keeps the
thread without them, and `draft_reply` reads the thread again when it needs
them. Links are recorded as inert strings; nothing fetches one, and no tool
reachable here can open an attachment.

**Approval shows the exact message, and is bound to it.** The `SEND_EMAIL`
action's payload is recipient, subject, body, attachments and whether it is a
reply (`in_reply_to`, `thread_id`); its `ApprovalRequest` carries the preview
(recipient, subject, body diff, attachments, reply vs new thread) and the
hash. `execute_send` recomputes the hash; an action edited after approval --
a new body, a new recipient -- no longer matches and is refused, not sent.

**Applications.** A thread whose inbound mail correlates to one of the
candidate's applications is recorded against it the way the inbox poller's
mail is: `triage` (from `domain.recruiter_events`) decides, and the
`RecruiterEventRecorder` writes the communication event and its proposed
transition once per message, through the lifecycle's transition rules. A
second pass over the same thread records nothing new.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol, TypedDict
from uuid import NAMESPACE_URL, UUID, uuid5

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from personalos.domain.communications import (
    CommunicationsContractError,
    CommunicationsRequest,
    CommunicationsTask,
    EmailThread,
    InboxItem,
    ThreadDigest,
    ThreadReplyDraft,
    TriageResult,
    needs_review,
    parse_communications_request,
    thread_reply_key,
)
from personalos.domain.errors import PersonalOSError
from personalos.domain.job_search import (
    PAYLOAD_PREVIOUS_BODY,
    ActionIntent,
    ActionReceipt,
    ApprovalDecision,
    ApprovalRequest,
    EmittedEvent,
    JobSearchContractError,
    JobSearchEventType,
    RecruiterMessage,
    authorize_execution,
)
from personalos.domain.outbound_email import (
    PAYLOAD_ARTIFACT_VERSION_ID,
    PAYLOAD_EVIDENCE_IDS,
    PAYLOAD_GMAIL_DRAFT_ID,
    send_email_intent,
)
from personalos.domain.recruiter_events import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_MIN_CLASSIFICATION_CONFIDENCE,
    CorrelationOutcome,
    RecruiterEvent,
    RecruiterEventRecord,
    RecruiterEventReview,
    TriageAction,
    correlate_application,
    triage,
)
from personalos.domain.redaction import redact, redact_text
from personalos.domain.workflow import communications_thread_id
from personalos.graphs.job_search import (
    ActionExecutor,
    ApplicationDirectory,
    EventEmitter,
    RecruiterEventExtractor,
    RecruiterEventRecorder,
    _events_for_recorded,
)
from personalos.models.communications import (
    MessageTriager,
    ResilientThreadAnalyzer,
    RuleBasedTriager,
    ThreadAnalyzer,
)
from personalos.observability.structured_logging import LogEvent
from personalos.observability.tracing import traced_node
from personalos.policy import PolicyError
from personalos.policy.rules import COMMUNICATIONS_REQUESTER_PREFIX

logger = logging.getLogger(__name__)

GRAPH_NAME = "communications"

PARSE_REQUEST = "parse_request"
TRIAGE_INBOX = "triage_inbox"
LOAD_THREAD = "load_thread"
LINK_APPLICATION = "link_application"
DRAFT_REPLY = "draft_reply"
REQUEST_SEND_APPROVAL = "request_send_approval"
SEND_APPROVAL_CHECKPOINT = "send_approval_checkpoint"
EXECUTE_SEND = "execute_send"

#: `approval_stage` / the interrupt's `stage`.
STAGE_SEND_REPLY = "send_reply"


def requester(node: str) -> str:
    """The `requested_by` a node stamps on its tool calls; policy scopes tools by it."""
    return f"{COMMUNICATIONS_REQUESTER_PREFIX}{node}"


# --- Ports ----------------------------------------------------------------------


class MailboxReader(Protocol):
    """Reads in-scope mail. `personalos.executor.gmail_threads.GmailThreadReader`.

    Every call is a read through the policy gateway as `requested_by`; a
    thread with nothing inside the mailbox scope raises `NotFound`.
    """

    async def list_inbox(self, *, requested_by: str, limit: int = ...) -> Sequence[InboxItem]: ...

    async def find_thread(self, *, sender_hint: str | None, requested_by: str) -> str | None: ...

    async def read_thread(self, thread_id: str, *, requested_by: str) -> EmailThread: ...


class ReplyDrafter(Protocol):
    """Drafts, validates and keeps a reply. Has no way to send one.

    `personalos.retrieval.thread_reply_drafting.ThreadReplyDrafter`.
    """

    async def draft(
        self, thread: EmailThread, *, user_id: UUID, application_id: UUID | None = None
    ) -> ThreadReplyDraft: ...


class CommunicationsState(TypedDict, total=False):
    """Graph state. JSON only; no message body is ever kept in it."""

    request: str
    user_id: str | None
    #: A thread the caller already knows, bypassing the request's own.
    thread_id: str | None
    comms_request: dict[str, Any] | None
    triage: list[dict[str, Any]]
    thread: dict[str, Any] | None
    digest: dict[str, Any] | None
    recruiter_events: list[dict[str, Any]]
    recruiter_event_records: list[dict[str, Any]]
    recruiter_event_reviews: list[dict[str, Any]]
    application_id: str | None
    reply_draft: dict[str, Any] | None
    approval_stage: str | None
    pending_actions: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    approval_refusals: list[dict[str, Any]]
    action_receipts: list[dict[str, Any]]
    errors: list[str]


def _dump(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [value.model_dump(mode="json") for value in values]


def _load_request(state: CommunicationsState) -> CommunicationsRequest | None:
    raw = state.get("comms_request")
    return CommunicationsRequest.model_validate(raw) if raw else None


def _user_id(state: CommunicationsState) -> UUID:
    raw = state.get("user_id")
    if not raw:
        raise JobSearchContractError("a Communications task needs the candidate's user_id")
    return UUID(str(raw))


def _failure(exc: Exception) -> str:
    return redact_text(f"{type(exc).__name__}: {exc}")[:500]


# --- Graph ----------------------------------------------------------------------


class CommunicationsGraph:
    """Builds and compiles the Communications subgraph."""

    def __init__(
        self,
        mailbox: MailboxReader,
        *,
        triager: MessageTriager | None = None,
        analyzer: ThreadAnalyzer | None = None,
        reply_drafter: ReplyDrafter | None = None,
        action_executor: ActionExecutor | None = None,
        recruiter_event_extractor: RecruiterEventExtractor | None = None,
        application_directory: ApplicationDirectory | None = None,
        recruiter_event_recorder: RecruiterEventRecorder | None = None,
        event_emitter: EventEmitter | None = None,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        min_classification_confidence: float = DEFAULT_MIN_CLASSIFICATION_CONFIDENCE,
        approval_ttl: timedelta | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        """Wire the ports.

        `mailbox` is required. Without a `reply_drafter` and an
        `action_executor`, a reply request ends after the summary with an
        error saying so. Without the extractor, directory and recorder,
        threads are summarized but not linked to applications.
        """
        if mailbox is None:
            raise ValueError("CommunicationsGraph requires a MailboxReader")
        self.mailbox = mailbox
        self.triager = triager or RuleBasedTriager()
        self.analyzer = analyzer or ResilientThreadAnalyzer(None)
        self.reply_drafter = reply_drafter
        self.action_executor = action_executor
        self.recruiter_event_extractor = recruiter_event_extractor
        self.application_directory = application_directory
        self.recruiter_event_recorder = recruiter_event_recorder
        self.event_emitter = event_emitter
        self.correlation_threshold = correlation_threshold
        self.min_classification_confidence = min_classification_confidence
        self.approval_ttl = approval_ttl
        self.clock = clock
        self.checkpointer = checkpointer or InMemorySaver()

    @property
    def links_applications(self) -> bool:
        return (
            self.recruiter_event_extractor is not None
            and self.application_directory is not None
            and self.recruiter_event_recorder is not None
        )

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(CommunicationsState)
        for name, method in (
            (PARSE_REQUEST, self.parse_request),
            (TRIAGE_INBOX, self.triage_inbox),
            (LOAD_THREAD, self.load_thread),
            (LINK_APPLICATION, self.link_application),
            (DRAFT_REPLY, self.draft_reply),
            (REQUEST_SEND_APPROVAL, self.request_send_approval),
            (SEND_APPROVAL_CHECKPOINT, self.send_approval_checkpoint),
            (EXECUTE_SEND, self.execute_send),
        ):
            graph.add_node(name, traced_node(name, method, graph=GRAPH_NAME))

        graph.add_edge(START, PARSE_REQUEST)
        graph.add_conditional_edges(
            PARSE_REQUEST,
            self.route_after_parse,
            {TRIAGE_INBOX: TRIAGE_INBOX, LOAD_THREAD: LOAD_THREAD},
        )
        graph.add_edge(TRIAGE_INBOX, END)
        graph.add_conditional_edges(
            LOAD_THREAD, self.route_after_load, {LINK_APPLICATION: LINK_APPLICATION, END: END}
        )
        graph.add_conditional_edges(
            LINK_APPLICATION, self.route_after_link, {DRAFT_REPLY: DRAFT_REPLY, END: END}
        )
        graph.add_conditional_edges(
            DRAFT_REPLY,
            self.route_after_draft,
            {REQUEST_SEND_APPROVAL: REQUEST_SEND_APPROVAL, END: END},
        )
        graph.add_edge(REQUEST_SEND_APPROVAL, SEND_APPROVAL_CHECKPOINT)
        graph.add_edge(SEND_APPROVAL_CHECKPOINT, EXECUTE_SEND)
        graph.add_edge(EXECUTE_SEND, END)
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def parse_request(self, state: CommunicationsState) -> dict[str, Any]:
        """Turn the request into a typed `CommunicationsRequest`. Deterministic; no model."""
        request = parse_communications_request(
            state.get("request", ""), thread_id=state.get("thread_id") or None
        )
        logger.info(
            "communications request: %s%s",
            request.task.value,
            f" on thread {request.thread_id}" if request.thread_id else "",
        )
        return {"comms_request": request.model_dump(mode="json"), "errors": []}

    async def triage_inbox(self, state: CommunicationsState) -> dict[str, Any]:
        """Triage the newest in-scope messages: headers and snippets, no bodies.

        Every result is a typed `TriageResult`; a triager that raises or
        answers outside the schema yields `needs_review` for that message.
        """
        try:
            items = list(await self.mailbox.list_inbox(requested_by=requester(TRIAGE_INBOX)))
        except (PolicyError, PersonalOSError) as exc:
            return {"triage": [], "errors": [*(state.get("errors") or []), _failure(exc)]}
        results: list[TriageResult] = []
        for item in items:
            try:
                result = await self.triager.triage(item)
            except Exception as exc:  # noqa: BLE001 - one bad triage is a review, not a crash
                result = needs_review(item, f"the triager failed ({type(exc).__name__})")
            # Ids are the item's, whatever the triager said they were.
            results.append(
                redact(result).model_copy(
                    update={"message_id": item.message_id, "thread_id": item.thread_id}
                )
            )
        return {"triage": _dump(results)}

    async def load_thread(self, state: CommunicationsState) -> dict[str, Any]:
        """Read the thread, summarize it, and say what its inbound mail is about.

        The node that reads bodies. What it returns has none: the thread is
        kept without them, the digest is redacted, and each recruiter event's
        message is blanked before it becomes state.
        """
        request = _load_request(state)
        errors = list(state.get("errors") or [])
        if request is None:  # pragma: no cover - parse_request always runs first
            raise JobSearchContractError("load_thread reached without a parsed request")
        who = requester(LOAD_THREAD)
        try:
            thread_id = request.thread_id or await self.mailbox.find_thread(
                sender_hint=request.sender_hint, requested_by=who
            )
            if thread_id is None:
                errors.append("no in-scope thread matches the request")
                return {"thread": None, "digest": None, "recruiter_events": [], "errors": errors}
            thread = await self.mailbox.read_thread(thread_id, requested_by=who)
        except (PolicyError, PersonalOSError) as exc:
            errors.append(_failure(exc))
            return {"thread": None, "digest": None, "recruiter_events": [], "errors": errors}

        digest: ThreadDigest = redact(await self.analyzer.analyze(thread))
        events = await self._recruiter_events(state, thread)
        logger.info(
            "read thread %s: %d message(s), %d action item(s), %d inbound event(s)",
            thread.thread_id,
            len(thread.messages),
            len(digest.action_items),
            len(events),
        )
        return {
            "thread": thread.without_bodies().model_dump(mode="json"),
            "digest": digest.model_dump(mode="json"),
            "recruiter_events": _dump(events),
            "errors": errors,
        }

    async def _recruiter_events(
        self, state: CommunicationsState, thread: EmailThread
    ) -> list[RecruiterEvent]:
        """Classify and correlate each inbound message, as J3 does for polled mail."""
        if not self.links_applications:
            return []
        assert self.recruiter_event_extractor is not None
        assert self.application_directory is not None
        user_id = _user_id(state)
        candidates = list(await self.application_directory.candidates(user_id))
        events: list[RecruiterEvent] = []
        for message in thread.messages:
            if message.from_user:
                continue
            inbound = RecruiterMessage(
                provider_message_id=message.message_id,
                received_at=message.received_at or self.clock(),
                body=message.body,
                subject=message.subject or None,
                from_address=message.from_address,
                thread_id=thread.thread_id,
            )
            outcome = await self.recruiter_event_extractor.extract(inbound)
            correlation = correlate_application(
                inbound, candidates, threshold=self.correlation_threshold
            )
            event = RecruiterEvent.build(inbound, outcome, correlation)
            events.append(
                event.model_copy(update={"message": inbound.model_copy(update={"body": ""})})
            )
        return events

    async def link_application(self, state: CommunicationsState) -> dict[str, Any]:
        """Record matched recruiter mail against its application, once per message.

        `triage` decides; the recorder writes the communication event and the
        transition it proposes in one transaction, through the lifecycle's
        rules. A message recorded before comes back `created=False` and
        nothing else happens for it. Anything short of a confident match is a
        review, not a transition.
        """
        events = [RecruiterEvent.model_validate(raw) for raw in state.get("recruiter_events") or ()]
        now = self.clock()
        records: list[RecruiterEventRecord] = []
        reviews: list[RecruiterEventReview] = []
        application_id: UUID | None = None
        for event in events:
            if event.correlation.outcome is CorrelationOutcome.MATCHED:
                application_id = event.correlation.application_id
            decision = triage(
                event, min_classification_confidence=self.min_classification_confidence
            )
            if decision.action is TriageAction.IGNORE:
                continue
            record: RecruiterEventRecord | None = None
            reason = decision.review_reason
            if decision.action is TriageAction.RECORD and self.recruiter_event_recorder:
                record = await self.recruiter_event_recorder.record(
                    event,
                    transition=decision.transition,
                    emit=_events_for_recorded(event, decision, now),
                )
                records.append(record)
                if not record.created:
                    continue
                if record.refused_transition is not None:
                    reason = (
                        f"the application is '{record.status.value}', and the lifecycle does "
                        f"not allow '{record.refused_transition.value}' from there"
                    )
            if reason:
                review = RecruiterEventReview(
                    dedupe_key=event.dedupe_key,
                    provider_message_id=event.message.provider_message_id,
                    classification=event.classification,
                    reason=reason,
                    application_id=record.application_id if record else None,
                    candidates=event.correlation.candidates,
                )
                reviews.append(review)
                if self.event_emitter is not None:
                    await self.event_emitter.emit(
                        EmittedEvent(
                            type=JobSearchEventType.RECRUITER_EVENT_REVIEW_REQUIRED,
                            aggregate_id=(
                                record.application_id
                                if record
                                else uuid5(NAMESPACE_URL, event.dedupe_key)
                            ),
                            payload=review.model_dump(mode="json"),
                            dedupe_key=f"recruiter_event_review:{event.dedupe_key}",
                            occurred_at=now,
                        )
                    )
        return {
            "recruiter_event_records": _dump(records),
            "recruiter_event_reviews": _dump(reviews),
            "application_id": str(application_id) if application_id else None,
            # Consumed: the events are recorded, and a later run must not record them again.
            "recruiter_events": [],
        }

    async def draft_reply(self, state: CommunicationsState) -> dict[str, Any]:
        """Draft, check and keep a reply, and propose sending it.

        Reads the thread again -- bodies are never checkpointed -- under this
        node's own allowlist, which can save a Gmail draft and cannot send.
        The proposal is a `SEND_EMAIL` intent whose recipient, subject and
        thread are the drafter's code-set values.
        """
        errors = list(state.get("errors") or [])
        raw_thread = state.get("thread")
        if not raw_thread or self.reply_drafter is None or self.action_executor is None:
            errors.append("no reply drafter and sender are wired; nothing was drafted")
            return {"reply_draft": None, "pending_actions": [], "errors": errors}
        thread_id = EmailThread.model_validate(raw_thread).thread_id
        user_id = _user_id(state)
        raw_application = state.get("application_id")
        try:
            thread = await self.mailbox.read_thread(thread_id, requested_by=requester(DRAFT_REPLY))
            draft = await self.reply_drafter.draft(
                thread,
                user_id=user_id,
                application_id=UUID(raw_application) if raw_application else None,
            )
            intent = self._send_intent(draft, user_id)
        except (PolicyError, PersonalOSError, CommunicationsContractError) as exc:
            errors.append(_failure(exc))
            return {"reply_draft": None, "pending_actions": [], "errors": errors}
        if draft.rejections:
            logger.info(
                "the drafted reply on thread %s was rejected by the claim validator; the "
                "template reply was drafted instead",
                thread_id,
            )
        return {
            "reply_draft": draft.model_dump(mode="json"),
            "pending_actions": _dump([intent]),
            "approval_stage": STAGE_SEND_REPLY,
            "errors": errors,
        }

    def _send_intent(self, draft: ThreadReplyDraft, user_id: UUID) -> ActionIntent:
        extra: dict[str, Any] = {PAYLOAD_PREVIOUS_BODY: ""}
        if draft.gmail_draft_id:
            extra[PAYLOAD_GMAIL_DRAFT_ID] = draft.gmail_draft_id
        if draft.artifact_version_id is not None:
            extra[PAYLOAD_ARTIFACT_VERSION_ID] = str(draft.artifact_version_id)
        if draft.evidence_ids:
            extra[PAYLOAD_EVIDENCE_IDS] = list(draft.evidence_ids)
        if draft.application_id is not None:
            extra["application_id"] = str(draft.application_id)
        return send_email_intent(
            recipient=draft.recipient,
            subject=draft.subject,
            body=draft.body,
            idempotency_key=thread_reply_key(user_id, draft.thread_id, draft.in_reply_to),
            requested_by=requester(DRAFT_REPLY),
            in_reply_to=draft.in_reply_to,
            thread_id=draft.thread_id,
            summary=f"Reply to {draft.recipient}: {draft.subject}"[:500],
            extra=extra,
        )

    def request_send_approval(self, state: CommunicationsState) -> dict[str, Any]:
        """Mint the `ApprovalRequest` for the send, before the interrupt checkpoints it."""
        now = self.clock()
        requests = [
            ApprovalRequest.for_intent(intent, now=now, ttl=self.approval_ttl)
            for intent in (
                ActionIntent.model_validate(raw) for raw in state.get("pending_actions") or ()
            )
        ]
        for request in requests:
            logger.info(
                "approval requested to send %s to %s; expires %s",
                request.action_id,
                request.target,
                request.expires_at.isoformat(),
            )
        return {"approval_requests": [*(state.get("approval_requests") or []), *_dump(requests)]}

    def send_approval_checkpoint(self, state: CommunicationsState) -> dict[str, Any]:
        """Park the run until a person answers. Nothing is ever sent without that answer."""
        intents = [ActionIntent.model_validate(raw) for raw in state.get("pending_actions") or ()]
        if not intents:
            return {}
        requests = {
            request.action_id: request
            for request in (
                ApprovalRequest.model_validate(raw) for raw in state.get("approval_requests") or ()
            )
        }
        outstanding = [requests[i.action_id] for i in intents if i.action_id in requests]
        if len(outstanding) != len(intents):
            raise JobSearchContractError(
                "a send reached the approval checkpoint with no approval request on file"
            )
        for request in outstanding:
            logger.info(
                "send %s awaits approval",
                request.action_id,
                extra={
                    "event": LogEvent.APPROVAL_REQUESTED,
                    "action_id": request.action_id,
                    "approval_id": request.request_id,
                },
            )
        resumed = interrupt(
            {
                "workflow": GRAPH_NAME,
                "stage": state.get("approval_stage") or STAGE_SEND_REPLY,
                "requests": [request.model_dump(mode="json") for request in outstanding],
            }
        )
        decisions = _decisions_from_resume(resumed)
        return {"approvals": [*(state.get("approvals") or []), *_dump(decisions)]}

    async def execute_send(self, state: CommunicationsState) -> dict[str, Any]:
        """Send what was approved, and only that.

        `authorize_execution` checks the request, the decision, the expiry,
        the approver and -- recomputed here -- the action's hash. Anything
        else is an `ApprovalRefusal` and the executor is not called.
        """
        requests = {
            r.action_id: r
            for r in (
                ApprovalRequest.model_validate(raw) for raw in state.get("approval_requests") or ()
            )
        }
        decisions = {
            d.action_id: d
            for d in (ApprovalDecision.model_validate(raw) for raw in state.get("approvals") or ())
        }
        refusals = list(state.get("approval_refusals") or [])
        receipts: list[ActionReceipt] = []
        for raw in state.get("pending_actions") or ():
            intent = ActionIntent.model_validate(raw)
            refusal = authorize_execution(
                intent=intent,
                request=requests.get(intent.action_id),
                decision=decisions.get(intent.action_id),
                now=self.clock(),
            )
            if refusal is not None:
                (logger.warning if refusal.suspicious else logger.info)(
                    "not sending %s: %s", intent.action_id, refusal.detail
                )
                refusals.append(refusal.model_dump(mode="json"))
                receipts.append(refusal.to_receipt())
                continue
            if self.action_executor is None:  # pragma: no cover - draft_reply checks first
                raise JobSearchContractError("execute_send reached without an action executor")
            receipts.append(await self.action_executor.execute(intent, decisions[intent.action_id]))
        return {"action_receipts": _dump(receipts), "approval_refusals": refusals}

    # ------------------------------------------------------------------
    # Routing: reads the parsed request and the nodes' conclusions, never mail
    # ------------------------------------------------------------------

    def route_after_parse(self, state: CommunicationsState) -> str:
        request = _load_request(state)
        return LOAD_THREAD if request is not None and request.is_thread_task else TRIAGE_INBOX

    def route_after_load(self, state: CommunicationsState) -> str:
        return LINK_APPLICATION if state.get("thread") else END

    def route_after_link(self, state: CommunicationsState) -> str:
        request = _load_request(state)
        if request is not None and request.task is CommunicationsTask.DRAFT_REPLY:
            return DRAFT_REPLY
        return END

    def route_after_draft(self, state: CommunicationsState) -> str:
        return REQUEST_SEND_APPROVAL if state.get("pending_actions") else END


def _decisions_from_resume(resumed: Any) -> list[ApprovalDecision]:
    """The reviewer's decisions, from whatever the interrupt was resumed with."""
    if resumed is None:
        return []
    if isinstance(resumed, ApprovalDecision):
        return [resumed]
    if isinstance(resumed, dict):
        if "action_id" in resumed:
            return [ApprovalDecision.model_validate(resumed)]
        return _decisions_from_resume(list(resumed.values()))
    if isinstance(resumed, Sequence) and not isinstance(resumed, str | bytes):
        return [d for item in resumed for d in _decisions_from_resume(item)]
    raise JobSearchContractError(
        f"cannot read an approval decision out of a resume value of type {type(resumed).__name__}"
    )


# --- Supervisor adapter -----------------------------------------------------------


class CommunicationsSubgraphRunner:
    """Satisfies the Supervisor's `CommunicationsSubgraphRunner` port with this subgraph.

    Runs on its own stable thread (`communications_thread_id`), continued
    rather than restarted while it has steps left -- a reply parked at its
    send approval, say.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        user_id: UUID | None = None,
        thread_id: str | None = None,
        workflow_id: UUID | None = None,
    ):
        self.graph = graph
        self.user_id = user_id
        self.thread_id = thread_id or communications_thread_id(user_id or "anonymous")
        self.workflow_id = workflow_id

    async def __call__(self, task_dag: dict[str, Any], state: Any) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": self.thread_id}
        if self.workflow_id is not None:
            configurable["workflow_id"] = str(self.workflow_id)
        config = {"configurable": configurable}
        stored = await self.graph.aget_state(config)
        initial: CommunicationsState | None = None
        if not stored.next:
            initial = {
                "request": (state or {}).get("message", "") if hasattr(state, "get") else "",
                "user_id": str(self.user_id) if self.user_id else None,
            }
        else:
            logger.info(
                "communications thread '%s' has %s still to run; continuing it",
                self.thread_id,
                ", ".join(stored.next),
            )
        final = await self.graph.ainvoke(initial, config=config)
        return communications_result(task_dag, final)


def communications_result(task_dag: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """What the Supervisor records of a Communications task. No inbound message body."""
    request = final.get("comms_request") or {}
    digest = final.get("digest") or {}
    return {
        "goal": task_dag.get("goal"),
        "domain": "communications",
        "task": request.get("task"),
        "thread_id": (final.get("thread") or {}).get("thread_id"),
        "triage": final.get("triage") or [],
        "summary": digest.get("summary"),
        "action_items": digest.get("action_items") or [],
        "recruiter_event_records": final.get("recruiter_event_records") or [],
        "recruiter_event_reviews": final.get("recruiter_event_reviews") or [],
        "reply_draft": final.get("reply_draft"),
        "awaiting_approval": bool(final.get("__interrupt__")),
        "action_receipts": final.get("action_receipts") or [],
        "errors": final.get("errors") or [],
    }


__all__ = [
    "DRAFT_REPLY",
    "EXECUTE_SEND",
    "GRAPH_NAME",
    "LINK_APPLICATION",
    "LOAD_THREAD",
    "PARSE_REQUEST",
    "REQUEST_SEND_APPROVAL",
    "SEND_APPROVAL_CHECKPOINT",
    "STAGE_SEND_REPLY",
    "TRIAGE_INBOX",
    "CommunicationsGraph",
    "CommunicationsState",
    "CommunicationsSubgraphRunner",
    "MailboxReader",
    "ReplyDrafter",
    "communications_result",
    "requester",
]
