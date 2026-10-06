"""The Job Search subgraph: the one domain subgraph this build implements.

Pipeline, in order:

    load_search_profile -> search_providers -> normalize_jobs -> deduplicate
    -> persist_postings -> hard_filter -> score_candidates -> evidence_check -> rank -> shortlist
    -> [optional] prepare_application_packet
    -> request_approval_for_external_write
    -> approval_checkpoint_for_external_submission   (interrupts here)
    -> execute_approved_actions
    -> persist_application -> emit application.created -> END

with a recruiter-response branch hanging off the end (see
`JobSearchGraph.handle_recruiter_response` and
`JobSearchGraph.create_follow_up_checkpoint`).

There is a second way in. A run invoked with `fired_checkpoints` in its input
is a *durable wait coming due* -- `apps.worker.checkpoint_monitor` found a
scheduled follow-up whose trigger had arrived and whose condition was still
unmet, and started this thread to act on it. It enters at `draft_follow_up`
rather than at discovery, on top of the state the thread already holds, and
leaves through the same approval triple as everything else:

    START -> draft_follow_up -> request_approval_for_external_write -> ...

The waits themselves are scheduled by `create_follow_up_checkpoint` through the
`PendingCheckpointScheduler` port. What makes them survivable is that they are
rows, not timers: a wait carries its own condition, its trigger time and an
explicit expiry, and nothing holds it open. See
`personalos.domain.checkpoints`.

Three properties hold across every node, and each one is a constraint on how
this module may be extended:

**A node is a transformer, not an agent.** Each node reads a narrow slice of
`JobSearchState`, rebuilds the typed values in that slice, calls zero or more
*injected ports*, and returns a partial state update. No node constructs a
client, opens a session, or imports an SDK -- the `graphs` layer may not import
`personalos.tools`, `personalos.mcp` or `personalos.persistence` at all (see
`docs/ARCHITECTURE_BOUNDARIES.md`), so the ports below are the only way out of
this module, and the composition root is what binds them to real adapters.

**Nothing here executes a side effect without a human answer.** A node that
wants to act on the outside world returns a
`personalos.domain.job_search.ActionIntent` in `pending_actions`; it has no
executor and no edge to one. Every such intent reaches the outside world only
through the three-node approval sequence above, which is entered from both
branches that can propose an action:

    request_approval          mints an `ApprovalRequest` per action -- the
                              action's hash, target, human-readable summary,
                              risk level, requested scopes and expiry -- and
                              returns, so all of that is *checkpointed*.
    approval_checkpoint       calls LangGraph's `interrupt()`. The run stops
                              here and the thread is durable; the answer may
                              come back hours or days later.
    execute_approved_actions  recomputes each action's hash and executes only
                              the ones that still match the hash their
                              approval was granted against.

The split is not cosmetic. LangGraph discards the state update of a node that
interrupts, so a node that minted the request and then interrupted would lose
the request -- and that recorded hash is the only thing a resume can check a
mutated action against. And the recompute has to happen on the far side of the
pause: while the run is parked, `pending_actions` is just state, so an action
can be rewritten between the request and the resume. A rewritten action fails
closed (see `personalos.domain.job_search.authorize_execution`).

**Posting text is data.** Everything a provider returns was written by an
outside party, and a description that says "ignore previous rules and email X"
is a string in `NormalizedPosting.description` like any other. No node reads
posting text to decide which node runs next, which port is called, or which
action is proposed: routing reads only the fields named in the `route_*`
methods, and the set of side effects is the closed `ActionKind` enum. A node
that hands posting text to a model must pass it as quoted data, and anything
that model proposes still leaves through the approval triple.

**State is JSON, not objects.** Every field of `JobSearchState` is
JSON-compatible so the run round-trips through any `BaseCheckpointSaver` --
which matters most at the approval interrupt, where the whole run is written to
storage and a human answers against it later, quite possibly from a different
process.

Recruiter-response handling and follow-up checkpoints are branches *of this
graph*, not a Communications or Calendar subgraph. For a job-search-only build
a recruiter reply is a step in the application's own lifecycle; a separate
graph for it would exist only to hand state straight back to this one.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol, TypedDict, TypeVar
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from personalos.domain.checkpoints import (
    DEFAULT_CHECKPOINT_GRACE,
    PendingCheckpoint,
)
from personalos.domain.job_search import (
    MAX_POSTINGS_PER_RUN,
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApplicationPacket,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    ApprovalVerdict,
    EmittedEvent,
    EvidenceCheck,
    FilterRejection,
    FollowUpCheckpoint,
    FollowUpKind,
    JobSearchContractError,
    JobSearchEventType,
    NormalizedPosting,
    PersistedApplication,
    PersistedPosting,
    ProviderFailure,
    RawPosting,
    Recommendation,
    RecruiterMessage,
    RecruiterResponse,
    ScoredPosting,
    SearchProfile,
    ShortlistEntry,
    authorize_execution,
)
from personalos.domain.models import ApplicationStatus, CommunicationEventClassification
from personalos.domain.redaction import redact
from personalos.domain.workflow import job_search_thread_id

logger = logging.getLogger(__name__)

#: Any of the typed values from `personalos.domain.job_search` that state holds
#: in dumped form, so `_load` returns the concrete type it was asked for.
_ValueT = TypeVar("_ValueT", bound=BaseModel)

#: langgraph node names. Kept as constants so a name is never spelled twice --
#: once on `add_node` and once in a conditional-edge mapping -- and a rename
#: cannot desynchronize the two.
LOAD_SEARCH_PROFILE = "load_search_profile"
SEARCH_PROVIDERS = "search_providers"
NORMALIZE_JOBS = "normalize_jobs"
DEDUPLICATE = "deduplicate"
PERSIST_POSTINGS = "persist_postings"
HARD_FILTER = "hard_filter"
SCORE_CANDIDATES = "score_candidates"
EVIDENCE_CHECK = "evidence_check"
RANK = "rank"
SHORTLIST = "shortlist"
PREPARE_APPLICATION_PACKET = "prepare_application_packet"
REQUEST_APPROVAL = "request_approval_for_external_write"
APPROVAL_CHECKPOINT = "approval_checkpoint_for_external_submission"
EXECUTE_APPROVED_ACTIONS = "execute_approved_actions"
PERSIST_APPLICATION = "persist_application"
EMIT_APPLICATION_CREATED = "emit_application_created"
HANDLE_RECRUITER_RESPONSE = "handle_recruiter_response"
CREATE_FOLLOW_UP_CHECKPOINT = "create_follow_up_checkpoint"
DRAFT_FOLLOW_UP = "draft_follow_up_for_triggered_checkpoint"

#: Values of `JobSearchState["approval_stage"]`. The approval node is entered
#: from two places (a submission, and a recruiter reply), and this is how its
#: router knows which branch to return to -- rather than inferring it from
#: which fields happen to be populated.
STAGE_SUBMISSION = "external_submission"
STAGE_RECRUITER_OUTREACH = "recruiter_outreach"

#: How far out a follow-up is scheduled when a recruiter has gone quiet, and
#: when one is owed a reply. The quiet period is longer because the ball is in
#: their court; an owed reply is the candidate's own commitment.
NO_RESPONSE_FOLLOW_UP_DAYS = 7
AWAITING_REPLY_FOLLOW_UP_DAYS = 2
INTERVIEW_PREP_FOLLOW_UP_DAYS = 1

#: Classifications that mean the candidate owes the recruiter something.
_REPLY_OWED_CLASSIFICATIONS = frozenset(
    {
        CommunicationEventClassification.ACTION_REQUIRED,
        CommunicationEventClassification.INTERVIEW_INVITE,
    }
)


# ---------------------------------------------------------------------------
# Ports
#
# Every outward-facing capability a node needs, as a Protocol defined here and
# implemented elsewhere. They are deliberately narrow: a port is what one node
# needs, so a test can substitute exactly that one behaviour, and so a node's
# reach is readable from its constructor rather than from its body.
# ---------------------------------------------------------------------------


class SearchProfileStore(Protocol):
    """Resolves the candidate's current targeting profile."""

    async def load(self, user_id: UUID) -> SearchProfile:
        """Return the profile `load_search_profile` will drive the run from."""
        ...


class JobBoardProvider(Protocol):
    """One job board or aggregator the run fans out to.

    `name` is used to attribute both results and failures, so a thin result
    set can be traced to the provider that produced it.
    """

    name: str

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return this provider's postings for the profile, unnormalized."""
        ...


class PostingNormalizer(Protocol):
    """Interprets one provider's payload into the shared posting shape.

    Returning `None` drops the posting: a payload too malformed to normalize is
    skipped with a log line rather than failing the run, since one bad row from
    a provider should not lose the other ninety-nine.
    """

    def normalize(self, raw: RawPosting) -> NormalizedPosting | None:
        """Return the normalized posting, or `None` to drop it."""
        ...


class PostingCatalog(Protocol):
    """The durable record of every posting discovered, one row per opening.

    What makes dedupe hold *across* runs: `deduplicate` only sees one run's
    postings, while this resolves each to the row an earlier run, or another
    provider, already created for the same `dedupe_key`.
    """

    async def record(self, postings: Sequence[NormalizedPosting]) -> Sequence[PersistedPosting]:
        """Store each posting once and return the row each resolved to."""
        ...


class CandidateScorer(Protocol):
    """Scores one posting against the profile, with its reasoning."""

    async def score(self, posting: NormalizedPosting, profile: SearchProfile) -> ScoredPosting:
        """Return the posting with a score, per-signal components, and reasons."""
        ...


class EvidenceChecker(Protocol):
    """Checks a match against the candidate's own record.

    Separate from scoring because they answer different questions: the scorer
    says how well the posting matches the stated profile, the checker says
    whether that match is backed by something actually in the candidate's
    history. A posting can score well and still fail this.
    """

    async def check(self, scored: ScoredPosting, profile: SearchProfile) -> EvidenceCheck:
        """Return the grounding verdict and citations for this match."""
        ...


class ApplicationPacketBuilder(Protocol):
    """Assembles the tailored documents for one shortlisted posting."""

    async def build(self, entry: ShortlistEntry, profile: SearchProfile) -> ApplicationPacket:
        """Return an unsubmitted packet for this shortlist entry."""
        ...


class ApprovalGate(Protocol):
    """Answers "is there already a decision on file for this action?".

    Deliberately *not* a second approver. The human-in-the-loop seam is
    LangGraph's `interrupt()` in `approval_checkpoint`; this port exists for the
    answers that already exist before the graph gets there -- a standing grant,
    a decision an operator recorded through the API, a policy that pre-clears a
    specific low-risk action.

    `PENDING` is the honest answer to "nobody has decided yet", and it is what
    makes the run interrupt. A deployment with no source of pre-existing
    decisions wires `InterruptOnlyApprovalGate` and pauses on every outward-
    facing write, which is the default shape of this graph.

    A gate that returns `APPROVED` does not skip the safety checks: whatever it
    returns is still bound to the checkpointed `ApprovalRequest` and re-checked
    against a freshly recomputed action hash by `execute_approved_actions`.
    """

    async def review(self, intent: ActionIntent) -> ApprovalDecision:
        """Return the verdict on this intent, bound to its fingerprint."""
        ...


class ActionExecutor(Protocol):
    """Redeems an approved action against the outside world.

    Held only by the approval node, and called only with a decision that
    `ApprovalDecision.authorizes()` accepted. Implementations are expected to
    re-check that themselves: this is the same belt-and-braces re-check
    `personalos.mcp.adapter.MCPToolInvoker` performs on an `ApprovedIntent`.
    """

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Carry out the approved action and return its receipt."""
        ...


class ApplicationStore(Protocol):
    """Writes the application record and the signals attached to it.

    The graph layer cannot import `personalos.persistence`, so this is how an
    application reaches storage. `create` takes a requested `status` rather
    than setting one: the implementation is expected to move the row through
    `personalos.domain.models.validate_application_status_transition`, so a
    classification can propose a lifecycle move but never write one directly.
    """

    async def create(
        self,
        *,
        packet: ApplicationPacket,
        profile: SearchProfile,
        status: ApplicationStatus,
        submitted: bool,
        receipt: ActionReceipt | None,
    ) -> PersistedApplication:
        """Persist the application (and its artifacts) and return the stored row."""
        ...

    async def record_recruiter_response(
        self, application_id: UUID, response: RecruiterResponse
    ) -> None:
        """Record one classified recruiter message against the application."""
        ...

    async def record_follow_up(self, checkpoint: FollowUpCheckpoint) -> None:
        """Record a dated follow-up reminder for the application."""
        ...


class PendingCheckpointScheduler(Protocol):
    """Stores a conditional wait that outlives this run, and this process.

    The one port whose whole purpose is to be *unheld*. Everything else a node
    calls happens now; this hands over a wait that becomes actionable days
    later, when neither this graph, this thread nor this process still exists.
    That is why the node cannot simply keep a timer: a timer is a thing a
    process holds, and the thing being scheduled has to survive the process.

    The implementation is
    `personalos.persistence.pending_checkpoints.StorePendingCheckpointScheduler`;
    it deduplicates on `PendingCheckpoint.dedupe_key` and returns whichever
    wait now owns that key, so a node re-entering this branch on a resume gets
    the original wait back rather than a second one dated from today.
    """

    async def schedule(self, checkpoint: PendingCheckpoint) -> PendingCheckpoint:
        """Persist the wait and return the stored one."""
        ...


class EventEmitter(Protocol):
    """Hands a domain event to the transactional outbox."""

    async def emit(self, event: EmittedEvent) -> None:
        """Enqueue the event for relay."""
        ...


class RecruiterInbox(Protocol):
    """Reads inbound recruiter messages for one application."""

    async def fetch(self, application_id: UUID) -> Sequence[RecruiterMessage]:
        """Return the messages received for this application."""
        ...


class RecruiterMessageClassifier(Protocol):
    """Classifies one recruiter message into a typed response."""

    async def classify(self, message: RecruiterMessage) -> RecruiterResponse:
        """Return the classification and the lifecycle move it implies."""
        ...


# ---------------------------------------------------------------------------
# Default pure implementations
# ---------------------------------------------------------------------------


class InterruptOnlyApprovalGate:
    """An `ApprovalGate` with nothing on file, so every action interrupts.

    The default a deployment wants unless it has somewhere to look up decisions
    made outside this run. It is a real implementation rather than an `| None`
    default on the constructor: `JobSearchGraph` requires every port so its
    reach is readable at the construction site, and "pause for a human every
    time" is a policy worth naming rather than an absence.
    """

    async def review(self, intent: ActionIntent) -> ApprovalDecision:
        """Report that nobody has answered, which is what parks the run."""
        return ApprovalDecision(
            action_id=intent.action_id,
            action_fingerprint=intent.fingerprint(),
            verdict=ApprovalVerdict.PENDING,
            decided_by="system:interrupt_only_gate",
            note="no decision on file; awaiting a human answer at the interrupt",
        )


class DictPostingNormalizer:
    """Normalizer for providers that already return posting-shaped dicts.

    A pure function object -- no I/O, no client -- so a node using it still
    calls "zero tool interfaces". It reads a conventional set of keys and drops
    anything missing the two fields a posting cannot exist without (a title and
    a company). A provider with its own vocabulary gets its own normalizer
    injected instead of this one.
    """

    #: Keys accepted for each normalized field, in precedence order. Providers
    #: disagree on names far more often than on meaning.
    _ALIASES: dict[str, tuple[str, ...]] = {
        "title": ("title", "job_title", "position"),
        "company": ("company", "company_name", "employer"),
        "location": ("location", "job_location", "city"),
        "url": ("url", "job_url", "link"),
        "description": ("description", "job_description", "summary"),
        "source_job_id": ("id", "job_id", "source_job_id"),
        "salary_min": ("salary_min", "min_salary"),
        "salary_max": ("salary_max", "max_salary"),
        "remote": ("remote", "is_remote"),
        "skills": ("skills", "tags"),
        "posted_at": ("posted_at", "date_posted"),
    }

    def normalize(self, raw: RawPosting) -> NormalizedPosting | None:
        """Build a `NormalizedPosting`, or return `None` if the payload is unusable."""
        payload = raw.payload
        picked = {
            field: next(
                (payload[key] for key in keys if payload.get(key) not in (None, "")),
                None,
            )
            for field, keys in self._ALIASES.items()
        }
        if not picked["title"] or not picked["company"]:
            logger.warning(
                "dropping posting from provider '%s': missing title or company", raw.provider
            )
            return None

        try:
            return NormalizedPosting(
                source=raw.provider,
                source_job_id=_as_str(picked["source_job_id"]),
                title=str(picked["title"]),
                company=str(picked["company"]),
                location=_as_str(picked["location"]),
                url=_as_str(picked["url"]),
                description=str(picked["description"] or ""),
                salary_min=_as_int(picked["salary_min"]),
                salary_max=_as_int(picked["salary_max"]),
                remote=bool(picked["remote"]),
                posted_at=_as_datetime(picked["posted_at"]),
                skills=tuple(str(skill) for skill in (picked["skills"] or ())),
                raw=dict(payload),
            )
        except (JobSearchContractError, ValueError, TypeError):
            logger.warning(
                "dropping unnormalizable posting from provider '%s'", raw.provider, exc_info=True
            )
            return None


def _as_str(value: Any) -> str | None:
    """Coerce to a non-blank string, or `None`."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> int | None:
    """Coerce to an int, or `None` when the provider sent something else."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    """Coerce an ISO-8601 string or datetime to a datetime, or `None`."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class JobSearchState(TypedDict, total=False):
    """State carried between Job Search nodes.

    Every value is JSON-compatible: typed values from
    `personalos.domain.job_search` are stored as `.model_dump(mode="json")` and
    rebuilt by whichever node needs them. That is what lets the whole run --
    including a pause at the approval checkpoint -- survive a checkpoint
    round-trip without registering custom serializers.

    Intermediate stages are all retained rather than overwritten. The extra
    state costs one JSON blob; what it buys is a run whose outcome can be
    explained after the fact ("nothing shortlisted, because all eleven
    survivors scored below your floor") instead of just observed.
    """

    # Inputs.
    user_id: str
    query: str
    prepare_application: bool

    # Discovery.
    search_profile: dict[str, Any] | None
    raw_postings: list[dict[str, Any]]
    provider_failures: list[dict[str, Any]]
    normalized_postings: list[dict[str, Any]]
    deduplicated_postings: list[dict[str, Any]]
    duplicate_dedupe_keys: list[str]
    persisted_postings: list[dict[str, Any]]

    # Selection.
    filtered_postings: list[dict[str, Any]]
    filter_rejections: list[dict[str, Any]]
    scored_postings: list[dict[str, Any]]
    evidence_checks: list[dict[str, Any]]
    ranked_postings: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]

    # Application.
    application_packet: dict[str, Any] | None
    pending_actions: list[dict[str, Any]]
    approval_stage: str | None
    approval_requests: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    approval_refusals: list[dict[str, Any]]
    action_receipts: list[dict[str, Any]]
    application: dict[str, Any] | None
    emitted_events: list[dict[str, Any]]

    # Recruiter lifecycle.
    recruiter_messages: list[dict[str, Any]]
    recruiter_responses: list[dict[str, Any]]
    follow_up_checkpoints: list[dict[str, Any]]
    #: The durable waits `create_follow_up_checkpoint` scheduled, in dumped
    #: form. Carried for provenance: the authoritative copy is the row, which
    #: is the only version that outlives this run.
    pending_checkpoints: list[dict[str, Any]]
    #: Durable waits whose trigger came round with their condition still unmet,
    #: supplied as *input* by `apps.worker.checkpoint_monitor`. Their presence
    #: is what routes a run into the follow-up path at START instead of into
    #: discovery -- see `JobSearchGraph.route_from_start`.
    fired_checkpoints: list[dict[str, Any]]


def _dump(values: Sequence[BaseModel]) -> list[dict[str, Any]]:
    """Dump a sequence of pydantic values to JSON-compatible dicts."""
    return [value.model_dump(mode="json") for value in values]


def _load(model: type[_ValueT], raws: Sequence[dict[str, Any]] | None) -> list[_ValueT]:
    """Rebuild typed values from the dict form held in state."""
    return [model.model_validate(raw) for raw in raws or ()]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class JobSearchGraph:
    """Builds and compiles the Job Search subgraph from its injected ports.

    Every port is a required constructor argument except the three that are
    genuinely optional capabilities: the normalizer (which has a pure default),
    and the recruiter inbox plus its classifier, which together turn the
    recruiter-response branch on. Everything else is required for the same
    reason `JobSearchExecutor` requires a `ToolGateway` -- a graph that can
    fall back to a global default is a graph whose reach is not visible at its
    construction site.
    """

    def __init__(
        self,
        *,
        profile_store: SearchProfileStore,
        providers: Sequence[JobBoardProvider],
        scorer: CandidateScorer,
        evidence_checker: EvidenceChecker,
        packet_builder: ApplicationPacketBuilder,
        approval_gate: ApprovalGate,
        action_executor: ActionExecutor,
        application_store: ApplicationStore,
        event_emitter: EventEmitter,
        normalizer: PostingNormalizer | None = None,
        posting_catalog: PostingCatalog | None = None,
        recruiter_inbox: RecruiterInbox | None = None,
        recruiter_classifier: RecruiterMessageClassifier | None = None,
        checkpoint_scheduler: PendingCheckpointScheduler | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        approval_ttl: timedelta | None = None,
        checkpoint_grace: timedelta | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Wire the ports this graph's nodes delegate to."""
        required = {
            "profile_store": profile_store,
            "scorer": scorer,
            "evidence_checker": evidence_checker,
            "packet_builder": packet_builder,
            "approval_gate": approval_gate,
            "action_executor": action_executor,
            "application_store": application_store,
            "event_emitter": event_emitter,
        }
        missing = sorted(name for name, port in required.items() if port is None)
        if missing:
            raise ValueError(f"JobSearchGraph requires: {', '.join(missing)}")
        if not providers:
            raise ValueError("JobSearchGraph requires at least one JobBoardProvider")
        if (recruiter_inbox is None) != (recruiter_classifier is None):
            raise ValueError(
                "recruiter_inbox and recruiter_classifier must be provided together; "
                "an inbox with no classifier cannot produce a typed response, and a "
                "classifier with no inbox has nothing to classify"
            )

        self.profile_store = profile_store
        self.providers = tuple(providers)
        self.normalizer = normalizer or DictPostingNormalizer()
        # Optional: without it a run still deduplicates within itself, it just
        # leaves no `job_postings` rows behind for the next run to collapse onto.
        self.posting_catalog = posting_catalog
        self.scorer = scorer
        self.evidence_checker = evidence_checker
        self.packet_builder = packet_builder
        self.approval_gate = approval_gate
        self.action_executor = action_executor
        self.application_store = application_store
        self.event_emitter = event_emitter
        self.recruiter_inbox = recruiter_inbox
        self.recruiter_classifier = recruiter_classifier
        # Optional for the same reason the recruiter inbox is: a deployment
        # with no monitor process sweeping `pending_checkpoints` would schedule
        # waits nothing ever picks up, and a wait nobody will act on is worse
        # than no wait -- it reads, in SQL, as a follow-up that is coming.
        # Without it the branch still records its `FollowUpCheckpoint`s in
        # state and emits `follow_up_scheduled`; only the durable half is off.
        self.checkpoint_scheduler = checkpoint_scheduler
        # How long past its trigger a scheduled wait stays actionable. `None`
        # means `DEFAULT_CHECKPOINT_GRACE`; a deployment that would rather drop
        # a late follow-up than send one passes something shorter.
        self.checkpoint_grace = checkpoint_grace or DEFAULT_CHECKPOINT_GRACE
        # `None` means each action kind's own `approval_ttl` applies; an
        # explicit value overrides all of them, which is what a deployment with
        # a stricter review window (or a test with a much shorter one) passes.
        self.approval_ttl = approval_ttl
        # Injected so the expiry a request is minted with and the `now` the
        # executor checks it against come from the same source, and a test can
        # move time without sleeping.
        self.clock = clock
        # In-memory checkpointing is for tests only -- it loses every thread when
        # the process ends. A real deployment passes
        # `personalos.persistence.checkpointer.SqlAlchemyCheckpointSaver`, built
        # by `personalos.bootstrap.build_durable_checkpointer`, and nothing in
        # this module changes for it.
        self.checkpointer = checkpointer or InMemorySaver()

    def build(self) -> CompiledStateGraph:
        """Assemble and compile the pipeline with this instance's checkpointer."""
        graph = StateGraph(JobSearchState)

        graph.add_node(LOAD_SEARCH_PROFILE, self.load_search_profile)
        graph.add_node(SEARCH_PROVIDERS, self.search_providers)
        graph.add_node(NORMALIZE_JOBS, self.normalize_jobs)
        graph.add_node(DEDUPLICATE, self.deduplicate)
        graph.add_node(PERSIST_POSTINGS, self.persist_postings)
        graph.add_node(HARD_FILTER, self.hard_filter)
        graph.add_node(SCORE_CANDIDATES, self.score_candidates)
        graph.add_node(EVIDENCE_CHECK, self.evidence_check)
        graph.add_node(RANK, self.rank)
        graph.add_node(SHORTLIST, self.shortlist)
        graph.add_node(PREPARE_APPLICATION_PACKET, self.prepare_application_packet)
        graph.add_node(REQUEST_APPROVAL, self.request_approval)
        graph.add_node(APPROVAL_CHECKPOINT, self.approval_checkpoint)
        graph.add_node(EXECUTE_APPROVED_ACTIONS, self.execute_approved_actions)
        graph.add_node(PERSIST_APPLICATION, self.persist_application)
        graph.add_node(EMIT_APPLICATION_CREATED, self.emit_application_created)
        graph.add_node(HANDLE_RECRUITER_RESPONSE, self.handle_recruiter_response)
        graph.add_node(CREATE_FOLLOW_UP_CHECKPOINT, self.create_follow_up_checkpoint)
        graph.add_node(DRAFT_FOLLOW_UP, self.draft_follow_up)

        # Two ways into this graph, decided at START from the input alone.
        #
        # A run handed `fired_checkpoints` is a durable wait coming due: some
        # sweep found a checkpoint whose trigger had arrived and whose
        # condition was *still* unmet, and started this thread to act on it.
        # It must not re-enter discovery -- the application it is following up
        # on was found weeks ago, and searching the boards again would rebuild
        # a shortlist nobody asked for -- so it enters at the draft node, on
        # top of the state this thread already holds.
        #
        # Everything else starts at the beginning.
        graph.add_conditional_edges(
            START,
            self.route_from_start,
            {DRAFT_FOLLOW_UP: DRAFT_FOLLOW_UP, LOAD_SEARCH_PROFILE: LOAD_SEARCH_PROFILE},
        )
        graph.add_edge(LOAD_SEARCH_PROFILE, SEARCH_PROVIDERS)
        graph.add_edge(SEARCH_PROVIDERS, NORMALIZE_JOBS)
        graph.add_edge(NORMALIZE_JOBS, DEDUPLICATE)
        graph.add_edge(DEDUPLICATE, PERSIST_POSTINGS)
        graph.add_edge(PERSIST_POSTINGS, HARD_FILTER)
        graph.add_edge(HARD_FILTER, SCORE_CANDIDATES)
        graph.add_edge(SCORE_CANDIDATES, EVIDENCE_CHECK)
        graph.add_edge(EVIDENCE_CHECK, RANK)
        graph.add_edge(RANK, SHORTLIST)

        # The packet step is optional: a discovery-only run stops at the
        # shortlist, and so does a run that shortlisted nothing.
        graph.add_conditional_edges(
            SHORTLIST,
            self.route_after_shortlist,
            {PREPARE_APPLICATION_PACKET: PREPARE_APPLICATION_PACKET, END: END},
        )
        graph.add_edge(PREPARE_APPLICATION_PACKET, REQUEST_APPROVAL)

        # The approval triple is entered from both branches that can propose an
        # outward-facing action, and routes back by stage. This is the whole
        # point of routing through it: there is no edge from a node that builds
        # an ActionIntent to anything that acts on one.
        #
        # Three nodes rather than one, and the split is load-bearing rather
        # than cosmetic. LangGraph checkpoints between super-steps and discards
        # the partial update of a node that interrupts, so a node that both
        # minted the `ApprovalRequest` and then interrupted would lose the
        # request it had just written -- and the recorded hash is the only
        # thing a resume can check a mutated action against. So:
        #
        #   request_approval  mints the requests and *returns*, which
        #                     checkpoints them;
        #   approval_checkpoint  interrupts, and its resume value is the
        #                     reviewer's decisions;
        #   execute_approved_actions  re-derives each action's hash and
        #                     executes only what still matches.
        graph.add_edge(REQUEST_APPROVAL, APPROVAL_CHECKPOINT)
        graph.add_edge(APPROVAL_CHECKPOINT, EXECUTE_APPROVED_ACTIONS)
        graph.add_conditional_edges(
            EXECUTE_APPROVED_ACTIONS,
            self.route_after_approval,
            {PERSIST_APPLICATION: PERSIST_APPLICATION, END: END},
        )
        graph.add_edge(PERSIST_APPLICATION, EMIT_APPLICATION_CREATED)
        graph.add_conditional_edges(
            EMIT_APPLICATION_CREATED,
            self.route_after_emit,
            {HANDLE_RECRUITER_RESPONSE: HANDLE_RECRUITER_RESPONSE, END: END},
        )
        graph.add_edge(HANDLE_RECRUITER_RESPONSE, CREATE_FOLLOW_UP_CHECKPOINT)
        graph.add_conditional_edges(
            CREATE_FOLLOW_UP_CHECKPOINT,
            self.route_after_follow_up,
            {REQUEST_APPROVAL: REQUEST_APPROVAL, END: END},
        )
        # A drafted follow-up is an outward-facing message like any other, so
        # it leaves by the same door: the approval triple, never an executor.
        graph.add_conditional_edges(
            DRAFT_FOLLOW_UP,
            self.route_after_follow_up,
            {REQUEST_APPROVAL: REQUEST_APPROVAL, END: END},
        )

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Discovery nodes
    # ------------------------------------------------------------------

    async def load_search_profile(self, state: JobSearchState) -> dict[str, Any]:
        """Resolve the candidate's targeting profile, honouring an inline query.

        A free-text query from the caller is folded into the profile's keywords
        rather than carried alongside them, so `hard_filter` and
        `score_candidates` have exactly one source of criteria to read.
        """
        user_id = _require_uuid(state.get("user_id"), "user_id")
        profile = await self.profile_store.load(user_id)

        query = (state.get("query") or "").strip()
        if query:
            merged = tuple(dict.fromkeys((*profile.keywords, query)))
            profile = profile.model_copy(update={"keywords": merged})

        return {"search_profile": profile.model_dump(mode="json")}

    async def search_providers(self, state: JobSearchState) -> dict[str, Any]:
        """Fan out to every provider, recording failures instead of raising.

        One dead job board must not fail the run. A provider that raises is
        recorded in `provider_failures` and the run continues, so the caller
        can tell a thin result set from a genuinely empty one.
        """
        profile = _require_profile(state)
        raw: list[RawPosting] = []
        failures: list[ProviderFailure] = []

        for provider in self.providers:
            try:
                postings = await provider.search(profile)
            except Exception as exc:
                logger.warning("provider '%s' failed during search", provider.name, exc_info=True)
                failures.append(ProviderFailure(provider=provider.name, error=str(exc)))
                continue
            raw.extend(postings)

        if len(raw) > MAX_POSTINGS_PER_RUN:
            logger.info(
                "truncating %s postings to the per-run bound of %s",
                len(raw),
                MAX_POSTINGS_PER_RUN,
            )
            raw = raw[:MAX_POSTINGS_PER_RUN]

        return {"raw_postings": _dump(raw), "provider_failures": _dump(failures)}

    def normalize_jobs(self, state: JobSearchState) -> dict[str, Any]:
        """Interpret each provider payload into the shared posting shape."""
        raws = _load(RawPosting, state.get("raw_postings"))
        normalized = [
            posting for posting in (self.normalizer.normalize(raw) for raw in raws) if posting
        ]
        return {"normalized_postings": _dump(normalized)}

    def deduplicate(self, state: JobSearchState) -> dict[str, Any]:
        """Collapse postings that describe the same opening, keeping the first.

        Identity is `NormalizedPosting.dedupe_key` -- company, title and a hash
        of the description -- not a provider id, so the same role cross-posted
        to two boards collapses onto one entry rather than surviving twice
        under two ids.
        """
        postings = _load(NormalizedPosting, state.get("normalized_postings"))
        seen: dict[str, NormalizedPosting] = {}
        duplicates: list[str] = []
        for posting in postings:
            key = posting.dedupe_key
            if key in seen:
                duplicates.append(key)
                continue
            seen[key] = posting
        return {
            "deduplicated_postings": _dump(list(seen.values())),
            "duplicate_dedupe_keys": duplicates,
        }

    async def persist_postings(self, state: JobSearchState) -> dict[str, Any]:
        """Record every distinct posting in the catalog, before any is filtered.

        Ahead of `hard_filter` on purpose: a posting the profile rejects today
        is still a posting that was discovered, and the next run should
        recognise it rather than store it again.
        """
        if self.posting_catalog is None:
            return {"persisted_postings": []}
        postings = _load(NormalizedPosting, state.get("deduplicated_postings"))
        persisted = await self.posting_catalog.record(postings)
        return {"persisted_postings": _dump(persisted)}

    def hard_filter(self, state: JobSearchState) -> dict[str, Any]:
        """Drop postings that violate a stated constraint, recording why.

        These are the candidate's non-negotiables, so they are applied *before*
        scoring: no score is high enough to survive an excluded employer or a
        salary below the floor, and letting one compete on score would mean it
        could.
        """
        profile = _require_profile(state)
        postings = _load(NormalizedPosting, state.get("deduplicated_postings"))

        kept: list[NormalizedPosting] = []
        rejections: list[FilterRejection] = []
        excluded = {name.strip().lower() for name in profile.excluded_companies}

        for posting in postings:
            reason = _hard_filter_reason(posting, profile, excluded)
            if reason is None:
                kept.append(posting)
            else:
                rejections.append(
                    FilterRejection(
                        dedupe_key=posting.dedupe_key,
                        title=posting.title,
                        company=posting.company,
                        reason=reason,
                    )
                )

        return {"filtered_postings": _dump(kept), "filter_rejections": _dump(rejections)}

    async def score_candidates(self, state: JobSearchState) -> dict[str, Any]:
        """Score every surviving posting against the profile."""
        profile = _require_profile(state)
        postings = _load(NormalizedPosting, state.get("filtered_postings"))
        scored = [await self.scorer.score(posting, profile) for posting in postings]
        return {"scored_postings": _dump(scored)}

    async def evidence_check(self, state: JobSearchState) -> dict[str, Any]:
        """Check each scored match against the candidate's own record."""
        profile = _require_profile(state)
        scored = _load(ScoredPosting, state.get("scored_postings"))
        checks = [await self.evidence_checker.check(item, profile) for item in scored]
        return {"evidence_checks": _dump(checks)}

    def rank(self, state: JobSearchState) -> dict[str, Any]:
        """Order scored postings best-first.

        Ties break on `dedupe_key`, which is content-derived and therefore
        stable: the same inputs rank the same way on every run, so a shortlist
        is reproducible rather than dependent on provider response ordering.
        """
        scored = _load(ScoredPosting, state.get("scored_postings"))
        ranked = sorted(scored, key=lambda item: (-item.score, item.posting.dedupe_key))
        return {"ranked_postings": _dump(ranked)}

    def shortlist(self, state: JobSearchState) -> dict[str, Any]:
        """Take the top-ranked postings that clear the score floor and are grounded.

        Two gates, both able to empty the shortlist: a posting below
        `profile.min_score` is never recommended however thin the field is, and
        a posting whose match is not grounded in the candidate's record is
        dropped even if it scored well -- an ungrounded match is a coincidence
        in the job text, not a reason to apply.
        """
        profile = _require_profile(state)
        ranked = _load(ScoredPosting, state.get("ranked_postings"))
        checks = {
            check.dedupe_key: check for check in _load(EvidenceCheck, state.get("evidence_checks"))
        }

        entries: list[ShortlistEntry] = []
        for item in ranked:
            if len(entries) >= profile.max_shortlist:
                break
            if item.score < profile.min_score:
                continue
            # A SKIP verdict stands whatever the floor is set to: a profile
            # with `min_score=0` must not shortlist a rejected posting.
            if item.match is not None and item.match.recommendation == Recommendation.SKIP:
                continue
            check = checks.get(item.posting.dedupe_key)
            if check is None or not check.grounded:
                continue
            entries.append(ShortlistEntry(rank=len(entries) + 1, scored=item, evidence=check))

        return {"shortlist": _dump(entries)}

    # ------------------------------------------------------------------
    # Application nodes
    # ------------------------------------------------------------------

    async def prepare_application_packet(self, state: JobSearchState) -> dict[str, Any]:
        """Build the packet for the top shortlist entry and propose submitting it.

        The two halves are deliberately asymmetric: the packet is a local
        artifact this node produces outright, while the submission is only
        *proposed*, as an `ActionIntent` in `pending_actions`. This node has no
        executor and no gate, so the proposal is the furthest it can go.
        """
        entries = _load(ShortlistEntry, state.get("shortlist"))
        if not entries:  # pragma: no cover - the router only reaches here with entries
            raise JobSearchContractError(
                "prepare_application_packet reached with an empty shortlist"
            )
        profile = _require_profile(state)
        top = entries[0]

        packet = await self.packet_builder.build(top, profile)
        intent = ActionIntent(
            kind=ActionKind.SUBMIT_APPLICATION,
            # Where the write lands, as a reviewer would check it: the posting's
            # own URL when the provider gave one, and otherwise the company and
            # role it was listed under. Part of the action's hash, so
            # re-pointing an otherwise identical submission invalidates its
            # approval.
            target=(
                packet.posting.url or f"{packet.posting.company} / {packet.posting.title}"
            ),
            summary=(
                f"Submit an application to {packet.posting.company} " f"for {packet.posting.title}"
            ),
            payload={
                "dedupe_key": packet.dedupe_key,
                "company": packet.posting.company,
                "title": packet.posting.title,
                "url": packet.posting.url,
                "artifact_types": [draft.artifact_type.value for draft in packet.artifacts],
            },
            # Content-derived, so retrying the run proposes the same submission
            # rather than a second one.
            idempotency_key=f"submit-{packet.dedupe_key}"[:255],
            requested_by=f"graph:job_search#{PREPARE_APPLICATION_PACKET}",
        )

        return {
            "application_packet": packet.model_dump(mode="json"),
            "pending_actions": _dump([intent]),
            "approval_stage": STAGE_SUBMISSION,
        }

    def request_approval(self, state: JobSearchState) -> dict[str, Any]:
        """Raise an `ApprovalRequest` for every action about to leave the system.

        Runs immediately before the interrupt and does nothing else, because
        what it writes has to be *checkpointed* before the run pauses. Each
        request pins the action's hash, target, summary, risk level, requested
        scopes and expiry as they stood at this moment; that pinned hash is the
        fixed point `execute_approved_actions` compares against when the run
        comes back, possibly days later, and it cannot be re-derived then --
        re-deriving it would compare a mutated action against itself.

        Appended rather than replaced: the node runs twice in a run that also
        answers a recruiter, and the submission's request stays part of the
        audit trail.
        """
        intents = _load(ActionIntent, state.get("pending_actions"))
        now = self.clock()
        requests = [
            ApprovalRequest.for_intent(intent, now=now, ttl=self.approval_ttl)
            for intent in intents
        ]
        for request in requests:
            logger.info(
                "approval requested for action %s (%s, risk=%s) on target '%s'; "
                "expires %s",
                request.action_id,
                request.kind.value,
                request.risk.value,
                request.target,
                request.expires_at.isoformat(),
            )
        return {
            "approval_requests": [*(state.get("approval_requests") or []), *_dump(requests)]
        }

    async def approval_checkpoint(self, state: JobSearchState) -> dict[str, Any]:
        """Park the run until every proposed external write has an answer.

        The single chokepoint for every outward-facing action this subgraph can
        take, and the only node in the graph that calls LangGraph's
        `interrupt()`. Reaching it means the next thing that would happen is a
        write somebody outside this system can see.

        The `ApprovalGate` is consulted first, and it is *not* a second
        approver: it answers only "is there already a decision on file for
        this?" -- a standing grant, an answer recorded through the API before
        the graph got here. Anything it returns `PENDING` for is genuinely
        unanswered, and that is what triggers the interrupt. A deployment with
        no such source wires `InterruptOnlyApprovalGate` and interrupts on
        every action, which is the default this design assumes.

        The interrupt's value is the list of live `ApprovalRequest`s; its resume
        value is the reviewer's `ApprovalDecision`s. Nothing is executed here --
        a decision collected at this node has not yet been checked against the
        action it will be spent on, and doing both in one node would put that
        check on the same side of the checkpoint boundary as the answer it is
        supposed to be auditing.
        """
        intents = _load(ActionIntent, state.get("pending_actions"))
        if not intents:
            return {}

        requests = {
            request.action_id: request
            for request in _load(ApprovalRequest, state.get("approval_requests"))
        }
        answered: dict[UUID, ApprovalDecision] = {}
        outstanding: list[ActionIntent] = []

        for intent in intents:
            if intent.action_id not in requests:
                raise JobSearchContractError(
                    f"action {intent.action_id} ({intent.kind.value}) reached the approval "
                    f"checkpoint with no request on file; request_approval must run first"
                )
            decision = await self.approval_gate.review(intent)
            if decision.verdict == ApprovalVerdict.PENDING:
                outstanding.append(intent)
            else:
                answered[intent.action_id] = decision

        if outstanding:
            payload = {
                "workflow": "job_search",
                "stage": state.get("approval_stage"),
                "requests": [
                    requests[intent.action_id].model_dump(mode="json") for intent in outstanding
                ],
            }
            logger.info(
                "pausing the run: %s action(s) await approval (%s)",
                len(outstanding),
                ", ".join(intent.kind.value for intent in outstanding),
            )
            # Raises `GraphInterrupt` the first time through, which checkpoints
            # the thread and ends the invocation; on a resume it returns the
            # value the caller supplied instead. Everything the answer is
            # checked against was written by the previous super-step, so the
            # gap between these two moments can be days long.
            for decision in _decisions_from_resume(interrupt(payload)):
                answered[decision.action_id] = decision

        decisions = [
            answered[intent.action_id] for intent in intents if intent.action_id in answered
        ]
        # Appended, not replaced: this node runs twice in a run that handles a
        # recruiter reply, and the submission's decision is part of the audit
        # trail the final state has to carry.
        return {"approvals": [*(state.get("approvals") or []), *_dump(decisions)]}

    async def execute_approved_actions(self, state: JobSearchState) -> dict[str, Any]:
        """Redeem the actions whose approval still fits them, and refuse the rest.

        The one place an outward-facing action actually happens, and it runs a
        super-step *after* the decision was collected -- so it re-reads the
        pending action from the checkpoint rather than trusting the copy the
        reviewer was shown. `authorize_execution` then recomputes the action's
        hash and refuses if it is not the hash the request went out with.

        That recomputation is the point of the whole arrangement. While the run
        is parked, `pending_actions` is just state, and a later node (or a model
        driving one) can rewrite it. An approval of "apply to the backend role
        at Acme" must not be spendable on whatever the action says by the time
        it is redeemed, so a changed action fails closed: no call is made, and
        an `ApprovalRefusal` is recorded in its place.
        """
        intents = _load(ActionIntent, state.get("pending_actions"))
        if not intents:
            return {}

        requests = {
            request.action_id: request
            for request in _load(ApprovalRequest, state.get("approval_requests"))
        }
        decisions = {
            decision.action_id: decision
            for decision in _load(ApprovalDecision, state.get("approvals"))
        }
        now = self.clock()

        receipts: list[ActionReceipt] = []
        refusals: list[ApprovalRefusal] = []

        for intent in intents:
            refusal = authorize_execution(
                intent=intent,
                request=requests.get(intent.action_id),
                decision=decisions.get(intent.action_id),
                now=now,
            )
            if refusal is not None:
                # Logged at warning only when the refusal means the action and
                # its approval drifted apart; a reviewer saying no is an
                # ordinary outcome, not an incident.
                (logger.warning if refusal.suspicious else logger.info)(
                    "refusing to execute action %s (%s): %s",
                    intent.action_id,
                    intent.kind.value,
                    refusal.detail,
                )
                refusals.append(refusal)
                continue

            receipt = await self.action_executor.execute(intent, decisions[intent.action_id])
            # The one point where what an executor did re-enters reasoning
            # state. A receipt's `detail` is provider-authored text, and from
            # here it is read by later nodes and may be shown to a model, so it
            # is redacted before it becomes state rather than only on its way
            # into a checkpoint.
            receipts.append(redact(receipt))

        return {
            "action_receipts": [*(state.get("action_receipts") or []), *_dump(receipts)],
            "approval_refusals": [*(state.get("approval_refusals") or []), *_dump(refusals)],
        }

    async def persist_application(self, state: JobSearchState) -> dict[str, Any]:
        """Store the application, recording whether it was actually submitted.

        Runs whether or not the submission was approved. A rejected submission
        still produced a real packet, and losing that work would mean rebuilding
        it on the next run; it is persisted in `READY_TO_APPLY` instead, which
        is exactly the lifecycle state for "prepared, not sent".
        """
        packet = _require_packet(state)
        profile = _require_profile(state)
        receipt = _receipt_for_kind(state, ActionKind.SUBMIT_APPLICATION)
        submitted = receipt is not None and receipt.ok

        application = await self.application_store.create(
            packet=packet,
            profile=profile,
            status=ApplicationStatus.APPLIED if submitted else ApplicationStatus.READY_TO_APPLY,
            submitted=submitted,
            receipt=receipt,
        )
        return {"application": application.model_dump(mode="json")}

    async def emit_application_created(self, state: JobSearchState) -> dict[str, Any]:
        """Emit `application.created`, plus a rejection event when not submitted.

        Emitted through the outbox port rather than published directly, so the
        event is enqueued in the same transaction as the row that caused it and
        a crash between the two cannot lose it.
        """
        application = _require_application(state)
        events = [
            EmittedEvent(
                type=JobSearchEventType.APPLICATION_CREATED,
                aggregate_id=application.application_id,
                payload={
                    "application_id": str(application.application_id),
                    "job_posting_id": str(application.job_posting_id),
                    "user_id": str(application.user_id),
                    "dedupe_key": application.dedupe_key,
                    "status": application.status.value,
                    "submitted": application.submitted,
                    "external_reference": application.external_reference,
                },
                dedupe_key=f"application.created:{application.application_id}",
            )
        ]
        if not application.submitted:
            # A prepared-but-unsent application is a distinct, reportable
            # outcome, not the absence of one.
            events.append(
                EmittedEvent(
                    type=JobSearchEventType.APPLICATION_SUBMISSION_REJECTED,
                    aggregate_id=application.application_id,
                    payload={
                        "application_id": str(application.application_id),
                        "dedupe_key": application.dedupe_key,
                    },
                    dedupe_key=(f"application.submission_rejected:{application.application_id}"),
                )
            )

        for event in events:
            await self.event_emitter.emit(event)

        return {"emitted_events": _dump(events)}

    # ------------------------------------------------------------------
    # Recruiter lifecycle branch
    # ------------------------------------------------------------------

    async def handle_recruiter_response(self, state: JobSearchState) -> dict[str, Any]:
        """Classify inbound recruiter messages and record each against the application.

        A branch of this subgraph rather than a Communications subgraph: for a
        job-search-only build, a recruiter reply is a step in the application's
        lifecycle, and the only consumer of its classification is this pipeline.

        A reply the recruiter is owed becomes an `ActionIntent`, not a sent
        message -- this node has no executor, so the outbound message routes
        back through the approval checkpoint like any other external action.
        """
        application = _require_application(state)
        if (  # pragma: no cover - the router does not reach here without both
            self.recruiter_inbox is None or self.recruiter_classifier is None
        ):
            raise JobSearchContractError(
                "handle_recruiter_response reached without a recruiter inbox"
            )

        messages = list(await self.recruiter_inbox.fetch(application.application_id))
        responses: list[RecruiterResponse] = []
        events: list[EmittedEvent] = []
        intents: list[ActionIntent] = []

        for message in messages:
            response = await self.recruiter_classifier.classify(message)
            await self.application_store.record_recruiter_response(
                application.application_id, response
            )
            responses.append(response)

            event = EmittedEvent(
                type=JobSearchEventType.RECRUITER_RESPONSE_RECORDED,
                aggregate_id=application.application_id,
                payload={
                    "application_id": str(application.application_id),
                    "provider_message_id": response.provider_message_id,
                    "classification": response.classification.value,
                    "implied_status": (
                        response.implied_status.value if response.implied_status else None
                    ),
                },
                dedupe_key=(
                    f"recruiter_response:{application.application_id}:"
                    f"{response.provider_message_id}"
                ),
            )
            await self.event_emitter.emit(event)
            events.append(event)

            if response.requires_reply:
                intents.append(
                    ActionIntent(
                        kind=ActionKind.SEND_RECRUITER_MESSAGE,
                        target=(
                            f"{message.from_address or 'recruiter'} "
                            f"(thread {response.provider_message_id})"
                        ),
                        summary=(
                            f"Reply to the recruiter's "
                            f"{response.classification.value.replace('_', ' ')} for "
                            f"{application.dedupe_key}"
                        ),
                        payload={
                            "application_id": str(application.application_id),
                            "in_reply_to": response.provider_message_id,
                            "classification": response.classification.value,
                        },
                        idempotency_key=(
                            f"reply-{application.application_id}-" f"{response.provider_message_id}"
                        )[:255],
                        requested_by=f"graph:job_search#{HANDLE_RECRUITER_RESPONSE}",
                    )
                )

        return {
            "recruiter_messages": _dump(messages),
            "recruiter_responses": _dump(responses),
            "emitted_events": [*(state.get("emitted_events") or []), *_dump(events)],
            "pending_actions": _dump(intents),
            "approval_stage": STAGE_RECRUITER_OUTREACH,
        }

    async def create_follow_up_checkpoint(
        self, state: JobSearchState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Schedule the dated reminders this application's state calls for.

        Also a branch of this subgraph rather than a Calendar one, and for the
        same reason. A checkpoint is an internal record, so it needs no
        approval; the outbound message that a checkpoint might eventually
        prompt is a separate action that does.

        Each reminder is written twice, into two things with different
        lifetimes. The `FollowUpCheckpoint` goes into state, where it is part
        of this run's story. The `PendingCheckpoint` goes through
        `checkpoint_scheduler` into storage, and that copy is the one that
        matters: a reminder due in seven days has to outlive this run, this
        worker and this deploy, and state -- durable as it is -- is only ever
        read by something that already decided to look at this thread. The row
        is what makes something *come and look*.

        Three things go into the durable copy that the state copy has no use
        for:

        - **the condition**, stored as a question rather than an answer, so it
          is asked again at trigger time. The whole value of waiting a week is
          that the recruiter might reply during it, and a condition evaluated
          now would be blind to exactly that;
        - **an explicit expiry**, `checkpoint_grace` past the trigger, so a
          follow-up that nothing swept in time is written off rather than sent
          weeks late or left pending forever;
        - **the thread to resume**, read from the run's own config. That is why
          this node takes `config`: `thread_id` is not state, it is the
          identity of the thread state is stored under, and a wait that did not
          record it would have nowhere to come back to.
        """
        application = _require_application(state)
        responses = _load(RecruiterResponse, state.get("recruiter_responses"))
        now = self.clock()

        checkpoints = _follow_ups_for(application.application_id, responses, now)
        events: list[EmittedEvent] = []
        scheduled: list[PendingCheckpoint] = []
        for checkpoint in checkpoints:
            await self.application_store.record_follow_up(checkpoint)
            if self.checkpoint_scheduler is not None:
                scheduled.append(
                    await self.checkpoint_scheduler.schedule(
                        self._durable_wait(checkpoint, now=now, config=config)
                    )
                )
            event = EmittedEvent(
                type=JobSearchEventType.FOLLOW_UP_SCHEDULED,
                aggregate_id=application.application_id,
                payload={
                    "application_id": str(application.application_id),
                    "kind": checkpoint.kind.value,
                    "due_at": checkpoint.due_at.isoformat(),
                    "reason": checkpoint.reason,
                },
                dedupe_key=(f"follow_up:{application.application_id}:{checkpoint.kind.value}"),
            )
            await self.event_emitter.emit(event)
            events.append(event)

        return {
            "follow_up_checkpoints": _dump(checkpoints),
            "pending_checkpoints": _dump(scheduled),
            "emitted_events": [*(state.get("emitted_events") or []), *_dump(events)],
        }

    async def draft_follow_up(self, state: JobSearchState) -> dict[str, Any]:
        """Propose the follow-up a triggered checkpoint asked for. Never sends it.

        Where a durable wait lands when it comes due with its condition still
        unmet. The monitor has already done the only thing it is allowed to do
        on its own -- re-ask the condition and find it still false -- and this
        node turns that into an `ActionIntent`, exactly like every other node
        that wants to touch the outside world. It holds no executor and has no
        edge to one; the message leaves only through the approval triple, which
        means a follow-up drafted while nobody was watching still waits for a
        human.

        The application comes from the thread's stored state rather than from
        the monitor's input. That is deliberate: the run was started on the
        thread the checkpoint named, so the application here is the one that
        thread has been about all along, and a monitor that passed its own copy
        could follow up on an application that had since moved on.
        """
        application = _require_application(state)
        fired = _load(PendingCheckpoint, state.get("fired_checkpoints"))
        now = self.clock()

        intents: list[ActionIntent] = []
        events: list[EmittedEvent] = []
        for checkpoint in fired:
            if checkpoint.application_id != application.application_id:
                # A checkpoint for a different application reached this thread.
                # Refused rather than followed up on: the thread's state is
                # about one application, so drafting from it would write a
                # message about the wrong one.
                raise JobSearchContractError(
                    f"checkpoint {checkpoint.checkpoint_id} is for application "
                    f"{checkpoint.application_id}, but thread state holds "
                    f"{application.application_id}"
                )
            intents.append(
                ActionIntent(
                    kind=ActionKind.SEND_RECRUITER_MESSAGE,
                    target=f"recruiter thread for {application.dedupe_key}",
                    summary=(
                        f"Follow up on {application.dedupe_key}: {checkpoint.reason} "
                        f"(due {checkpoint.trigger_at.date().isoformat()})"
                    ),
                    payload={
                        "application_id": str(application.application_id),
                        "checkpoint_id": str(checkpoint.checkpoint_id),
                        "kind": checkpoint.kind.value,
                        "unmet_condition": checkpoint.condition.kind.value,
                    },
                    idempotency_key=f"follow-up-{checkpoint.checkpoint_id}"[:255],
                    requested_by=f"graph:job_search#{DRAFT_FOLLOW_UP}",
                    created_at=now,
                )
            )
            event = EmittedEvent(
                type=JobSearchEventType.FOLLOW_UP_TRIGGERED,
                aggregate_id=application.application_id,
                payload={
                    "application_id": str(application.application_id),
                    "checkpoint_id": str(checkpoint.checkpoint_id),
                    "kind": checkpoint.kind.value,
                    "trigger_at": checkpoint.trigger_at.isoformat(),
                    "unmet_condition": checkpoint.condition.kind.value,
                },
                dedupe_key=f"follow_up_triggered:{checkpoint.checkpoint_id}",
                occurred_at=now,
            )
            await self.event_emitter.emit(event)
            events.append(event)

        return {
            "pending_actions": _dump(intents),
            "approval_stage": STAGE_RECRUITER_OUTREACH,
            "emitted_events": [*(state.get("emitted_events") or []), *_dump(events)],
            # Consumed, so it is cleared. `fired_checkpoints` is an *input*
            # channel that decides which way a run enters the graph, and thread
            # state persists between runs: left in place, the next ordinary job
            # search on this thread would be routed into the follow-up path by
            # a checkpoint that fired a month ago.
            "fired_checkpoints": [],
        }

    def _durable_wait(
        self,
        checkpoint: FollowUpCheckpoint,
        *,
        now: datetime,
        config: RunnableConfig | None,
    ) -> PendingCheckpoint:
        """Build the storable wait behind one in-run follow-up checkpoint."""
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise JobSearchContractError(
                "cannot schedule a durable follow-up on a run with no thread_id; the "
                "wait would have nowhere to resume. Invoke the graph with a config "
                "naming its thread (see personalos.domain.workflow.WorkflowThread)"
            )
        raw_workflow_id = configurable.get("workflow_id")
        return PendingCheckpoint.for_follow_up(
            application_id=checkpoint.application_id,
            kind=checkpoint.kind,
            due_at=checkpoint.due_at,
            reason=checkpoint.reason,
            thread_id=str(thread_id),
            workflow_id=UUID(str(raw_workflow_id)) if raw_workflow_id else None,
            created_at=now,
            grace=self.checkpoint_grace,
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    def route_from_start(self, state: JobSearchState) -> str:
        """Enter at the follow-up draft when a durable wait fired, else at discovery.

        The only routing decision made from the graph's input rather than from
        work it has done, because it answers a question the run cannot answer
        for itself: *why* was this thread invoked? A monitor acting on a
        triggered checkpoint and a caller starting a fresh search hand the same
        graph the same thread, and only the `fired_checkpoints` in the input
        tells the two apart.
        """
        if state.get("fired_checkpoints"):
            return DRAFT_FOLLOW_UP
        return LOAD_SEARCH_PROFILE

    def route_after_shortlist(self, state: JobSearchState) -> str:
        """Prepare a packet only when asked to, and only with something to apply to."""
        if not state.get("prepare_application"):
            return END
        if not state.get("shortlist"):
            return END
        return PREPARE_APPLICATION_PACKET

    def route_after_approval(self, state: JobSearchState) -> str:
        """Return to the branch that proposed the action, by stage.

        Runs after `execute_approved_actions`, so by here every action has
        either a receipt or a recorded refusal. A submission that was refused
        still routes on to `persist_application`: the packet is real work, and
        it is stored as prepared-but-unsent rather than thrown away.

        A `PENDING` verdict ends the run regardless of stage. That is a
        reviewer who resumed the interrupt with "not yet" -- the run holds
        where it is rather than proceeding on an unanswered request.
        """
        if _has_pending_verdict(state):
            return END
        if state.get("approval_stage") == STAGE_SUBMISSION:
            return PERSIST_APPLICATION
        return END

    def route_after_emit(self, state: JobSearchState) -> str:
        """Continue into the recruiter branch only when an inbox is wired."""
        if self.recruiter_inbox is None or self.recruiter_classifier is None:
            return END
        if not state.get("application"):  # pragma: no cover - persist always sets one
            return END
        return HANDLE_RECRUITER_RESPONSE

    def route_after_follow_up(self, state: JobSearchState) -> str:
        """Send a proposed message back through the approval checkpoint, or finish.

        This is the edge that makes the invariant structural: neither the
        recruiter branch nor the triggered-checkpoint branch can reach an
        executor except by going back through the approval node. Shared by both
        for exactly that reason -- a second router would be a second place the
        rule could be written differently.
        """
        if state.get("pending_actions"):
            return REQUEST_APPROVAL
        return END


# ---------------------------------------------------------------------------
# Node helpers
#
# Pure functions, kept out of the node bodies so each rule can be tested on
# its own and so a node body reads as the sequence of steps it performs.
# ---------------------------------------------------------------------------


def _hard_filter_reason(
    posting: NormalizedPosting, profile: SearchProfile, excluded: set[str]
) -> str | None:
    """Return why this posting fails a stated constraint, or `None` if it passes."""
    if posting.company.strip().lower() in excluded:
        return f"company '{posting.company}' is on the exclusion list"
    if profile.remote_only and not posting.remote:
        return "profile requires remote and the posting is not remote"
    if profile.salary_min is not None:
        # An unstated salary is not a violation: most postings omit one, and
        # treating silence as a failure would filter out most of the market.
        if posting.salary_max is not None and posting.salary_max < profile.salary_min:
            return (
                f"posting salary ceiling {posting.salary_max} is below the floor "
                f"{profile.salary_min}"
            )
    if profile.must_have_skills:
        haystack = f"{posting.title}\n{posting.description}\n{' '.join(posting.skills)}".lower()
        missing = [
            skill for skill in profile.must_have_skills if skill.strip().lower() not in haystack
        ]
        if missing:
            return f"posting does not mention required skill(s): {', '.join(missing)}"
    return None


def _follow_ups_for(
    application_id: UUID, responses: Sequence[RecruiterResponse], now: datetime
) -> list[FollowUpCheckpoint]:
    """Decide which follow-up checkpoints this application's state calls for.

    At most one checkpoint per kind, so re-running the branch over the same
    messages does not pile up duplicate reminders.
    """
    if not responses:
        return [
            FollowUpCheckpoint(
                application_id=application_id,
                kind=FollowUpKind.NO_RESPONSE,
                due_at=now + timedelta(days=NO_RESPONSE_FOLLOW_UP_DAYS),
                reason="no recruiter response received yet",
            )
        ]

    by_kind: dict[FollowUpKind, FollowUpCheckpoint] = {}
    for response in responses:
        if response.classification == CommunicationEventClassification.INTERVIEW_INVITE:
            by_kind.setdefault(
                FollowUpKind.INTERVIEW_PREP,
                FollowUpCheckpoint(
                    application_id=application_id,
                    kind=FollowUpKind.INTERVIEW_PREP,
                    due_at=now + timedelta(days=INTERVIEW_PREP_FOLLOW_UP_DAYS),
                    reason=f"prepare for the interview offered in {response.provider_message_id}",
                ),
            )
        if response.requires_reply or response.classification in _REPLY_OWED_CLASSIFICATIONS:
            by_kind.setdefault(
                FollowUpKind.AWAITING_CANDIDATE_REPLY,
                FollowUpCheckpoint(
                    application_id=application_id,
                    kind=FollowUpKind.AWAITING_CANDIDATE_REPLY,
                    due_at=now + timedelta(days=AWAITING_REPLY_FOLLOW_UP_DAYS),
                    reason=f"reply owed on {response.provider_message_id}",
                ),
            )
    return list(by_kind.values())


def _decisions_from_resume(resumed: Any) -> list[ApprovalDecision]:
    """Parse whatever a caller resumed the interrupt with into typed decisions.

    Accepts one decision or several, typed or in dumped form, because the value
    arrives from outside the graph -- an API handler, an operator's CLI, a test
    -- and normalizing it once here is better than each caller having to know
    the graph's internal shape. What it will *not* do is invent a decision:
    anything it cannot parse raises, so a malformed resume stops the run rather
    than silently leaving an action unapproved (which
    `authorize_execution` would then refuse anyway, but with a misleading
    reason).
    """
    if resumed is None:
        return []
    if isinstance(resumed, ApprovalDecision):
        return [resumed]
    if isinstance(resumed, dict):
        # A mapping of request id -> decision is a natural shape for a caller
        # answering several requests at once; a bare decision is the common one.
        if "action_id" in resumed:
            return [ApprovalDecision.model_validate(resumed)]
        return _decisions_from_resume(list(resumed.values()))
    if isinstance(resumed, Sequence) and not isinstance(resumed, str | bytes):
        decisions: list[ApprovalDecision] = []
        for item in resumed:
            decisions.extend(_decisions_from_resume(item))
        return decisions
    raise JobSearchContractError(
        f"cannot read an approval decision out of a resume value of type "
        f"{type(resumed).__name__}; resume the approval checkpoint with "
        f"ApprovalDecision(s)"
    )


def _has_pending_verdict(state: JobSearchState) -> bool:
    """True when a reviewer has not yet answered one of the actions just proposed.

    Scoped to the current `pending_actions` rather than the whole accumulated
    approval list, so a `PENDING` verdict recorded for an earlier stage cannot
    keep stalling later ones.
    """
    action_ids = {raw.get("action_id") for raw in state.get("pending_actions") or ()}
    return any(
        raw.get("verdict") == ApprovalVerdict.PENDING.value and raw.get("action_id") in action_ids
        for raw in state.get("approvals") or ()
    )


def _receipt_for_kind(state: JobSearchState, kind: ActionKind) -> ActionReceipt | None:
    """The receipt for the approved action of this kind, if one was redeemed."""
    action_ids = {
        raw["action_id"]
        for raw in state.get("pending_actions") or ()
        if raw.get("kind") == kind.value
    }
    for raw in state.get("action_receipts") or ():
        if raw.get("action_id") in action_ids:
            return ActionReceipt.model_validate(raw)
    return None


def _require_uuid(value: Any, field: str) -> UUID:
    """Parse a required UUID out of state, failing with a typed contract error."""
    if value is None:
        raise JobSearchContractError(f"{field} is required to start a job search run")
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except ValueError as exc:
        raise JobSearchContractError(f"{field} must be a UUID, got {value!r}") from exc


def _require_profile(state: JobSearchState) -> SearchProfile:
    """Rebuild the search profile, failing loudly if a node ran out of order."""
    raw = state.get("search_profile")
    if not raw:
        raise JobSearchContractError(
            "no search profile in state; load_search_profile must run first"
        )
    return SearchProfile.model_validate(raw)


def _require_packet(state: JobSearchState) -> ApplicationPacket:
    """Rebuild the application packet, failing loudly if a node ran out of order."""
    raw = state.get("application_packet")
    if not raw:
        raise JobSearchContractError(
            "no application packet in state; prepare_application_packet must run first"
        )
    return ApplicationPacket.model_validate(raw)


def _require_application(state: JobSearchState) -> PersistedApplication:
    """Rebuild the persisted application, failing loudly if a node ran out of order."""
    raw = state.get("application")
    if not raw:
        raise JobSearchContractError("no application in state; persist_application must run first")
    return PersistedApplication.model_validate(raw)


# ---------------------------------------------------------------------------
# Supervisor adapter
# ---------------------------------------------------------------------------


class JobSearchSubgraphRunner:
    """Satisfies `personalos.graphs.supervisor.JobSubgraphRunner` with this subgraph.

    The Supervisor plans a bounded `TaskDAG` and hands it to a runner; this
    translates that hand-off into an initial `JobSearchState`. It lives here
    rather than in `bootstrap` because the mapping is knowledge about this
    graph's state, not wiring -- the composition root still owns constructing
    the graph and its ports.

    The DAG itself is carried through for provenance rather than interpreted:
    this subgraph's step sequence is fixed by `build()`, and a planner cannot
    reorder or extend it.

    `thread_id` is supplied, not minted here, and that is load-bearing. This
    subgraph runs on its own thread -- the Supervisor's conversation and this
    domain run are separately resumable units of work, so they do not share one
    -- but a thread id invented per call would give every invocation a fresh
    thread, and a fresh thread has no state to resume from no matter how durable
    the checkpointer underneath it is. The composition root derives a stable id
    (`personalos.domain.workflow.job_search_thread_id`) and registers it against
    the workflow before the first run; see `personalos.bootstrap`.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        user_id: UUID,
        thread_id: str | None = None,
        workflow_id: UUID | None = None,
        prepare_application: bool = False,
    ):
        """Take the compiled subgraph, the candidate, and the thread to run on.

        `thread_id` defaults to `job_search_thread_id(user_id)` -- the same
        recipe `personalos.bootstrap.register_job_search_thread` registers under,
        deliberately, so a runner built without an explicit id still lands on the
        thread that was registered for it. A deployment running several
        concurrent searches per candidate passes the id that says which one.
        """
        self.graph = graph
        self.user_id = user_id
        self.thread_id = thread_id or job_search_thread_id(user_id)
        self.workflow_id = workflow_id
        self.prepare_application = prepare_application

    async def __call__(self, task_dag: dict[str, Any], state: Any) -> dict[str, Any]:
        """Run the subgraph for the Supervisor's planned DAG and return its final state."""
        configurable: dict[str, Any] = {"thread_id": self.thread_id}
        if self.workflow_id is not None:
            configurable["workflow_id"] = str(self.workflow_id)
        initial: JobSearchState = {
            "user_id": str(self.user_id),
            "query": (state or {}).get("message", "") if hasattr(state, "get") else "",
            "prepare_application": self.prepare_application,
        }
        final = await self.graph.ainvoke(initial, config={"configurable": configurable})
        return {
            "goal": task_dag.get("goal"),
            "shortlist": final.get("shortlist") or [],
            "application": final.get("application"),
            "emitted_events": final.get("emitted_events") or [],
            "provider_failures": final.get("provider_failures") or [],
        }


__all__ = [
    "JobSearchGraph",
    "JobSearchState",
    "JobSearchSubgraphRunner",
    "DictPostingNormalizer",
    "InterruptOnlyApprovalGate",
    # Ports
    "SearchProfileStore",
    "JobBoardProvider",
    "PostingNormalizer",
    "CandidateScorer",
    "EvidenceChecker",
    "ApplicationPacketBuilder",
    "ApprovalGate",
    "ActionExecutor",
    "ApplicationStore",
    "EventEmitter",
    "RecruiterInbox",
    "RecruiterMessageClassifier",
    "PendingCheckpointScheduler",
    # Node names
    "LOAD_SEARCH_PROFILE",
    "SEARCH_PROVIDERS",
    "NORMALIZE_JOBS",
    "DEDUPLICATE",
    "HARD_FILTER",
    "SCORE_CANDIDATES",
    "EVIDENCE_CHECK",
    "RANK",
    "SHORTLIST",
    "PREPARE_APPLICATION_PACKET",
    "REQUEST_APPROVAL",
    "APPROVAL_CHECKPOINT",
    "EXECUTE_APPROVED_ACTIONS",
    "PERSIST_APPLICATION",
    "EMIT_APPLICATION_CREATED",
    "HANDLE_RECRUITER_RESPONSE",
    "CREATE_FOLLOW_UP_CHECKPOINT",
    "DRAFT_FOLLOW_UP",
    "STAGE_SUBMISSION",
    "STAGE_RECRUITER_OUTREACH",
]
