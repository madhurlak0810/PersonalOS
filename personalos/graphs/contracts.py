"""Typed input/output contracts for every node registered in a graph.

A node reads a narrow slice of its graph's state and returns a partial update.
`JobSearchState` and `SupervisorState` are `TypedDict`s holding JSON, which is
what lets them round-trip through a checkpointer -- but a `TypedDict` says
nothing about which keys one node reads, which it writes, or what the dicts in
them must rebuild into. This module says that, once per node, as a pair of
Pydantic models:

- **The input model** names the state keys the node reads and the typed value
  each one must rebuild into. It ignores every other key, because a node is
  handed the whole state. A required field is one the node cannot run without.
- **The output model** names every key the node may return and is
  `extra="forbid"`, so a node that starts writing a neighbour's key fails its
  contract instead of making the graph work by accident. A required field is
  one the node always returns; an optional one it returns only sometimes.

Consumed input channels (`fired_checkpoints`, `inbound_messages`, ...) are
cleared by the node that reads them, so their output type is `Consumed`: a
value that must be empty.

Nothing here runs in the graph. The contracts are enforced by the node
contract tests (`tests/unit/test_*_nodes.py`), and
`tests/contracts/test_node_contract_coverage.py` fails if a node is registered
in a graph without an entry in `NODE_CONTRACTS` or without a contract test.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from personalos.domain.applications import RequestedTransition, TransitionOutcome
from personalos.domain.checkpoints import PendingCheckpoint
from personalos.domain.interview_scheduling import (
    CalendarEvent,
    InterviewRequest,
    InterviewSchedulePlan,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionReceipt,
    ApplicationPacket,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    EmittedEvent,
    EvidenceCheck,
    FilterRejection,
    FollowUpCheckpoint,
    NormalizedPosting,
    PersistedApplication,
    PersistedPosting,
    ProviderFailure,
    RawPosting,
    RecruiterMessage,
    RecruiterResponse,
    ScoredPosting,
    SearchProfile,
    ShortlistEntry,
)
from personalos.domain.recruiter_events import (
    RecruiterEvent,
    RecruiterEventRecord,
    RecruiterEventReview,
)
from personalos.domain.routing import RouteDecision
from personalos.domain.tasks import TaskDAG
from personalos.graphs import job_search as jsg
from personalos.graphs import supervisor as sup

#: Graph names, as `traced_node` labels them.
SUPERVISOR_GRAPH = "supervisor"
JOB_SEARCH_GRAPH = "job_search"


def _must_be_empty(value: Any) -> Any:
    if value:
        raise ValueError("a consumed input channel must be cleared, not carried forward")
    return value


#: An input channel the node has consumed. Thread state outlives the run, so
#: one left in place would route the next run on the thread back here.
Consumed = Annotated[list[Any], AfterValidator(_must_be_empty)]
ConsumedMapping = Annotated[dict[str, Any], AfterValidator(_must_be_empty)]

#: Values `approval_stage` may take; see `personalos.graphs.job_search`.
ApprovalStage = Literal[
    "external_submission",
    "recruiter_outreach",
    "interview_scheduling",
    "lifecycle_transition",
]


class NodeInput(BaseModel):
    """The slice of state a node reads. Every other key is ignored."""

    model_config = ConfigDict(extra="ignore")


class NodeOutput(BaseModel):
    """The partial update a node returns. A key not declared here is a violation."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class NodeContract:
    """One node's typed I/O, and the graph method that implements it."""

    graph: str
    node: str
    method: str
    input: type[NodeInput]
    output: type[NodeOutput]

    def validate_input(self, state: Mapping[str, Any]) -> NodeInput:
        """Rebuild the slice of `state` this node reads; raise if it does not fit."""
        return self.input.model_validate(dict(state))

    def validate_output(self, update: Mapping[str, Any]) -> NodeOutput:
        """Rebuild the update this node returned; raise if it does not fit."""
        return self.output.model_validate(dict(update))


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class LoadContextInput(NodeInput):
    """Reads no state: the actor comes from the run's config, not its state."""


class LoadContextOutput(NodeOutput):
    actor_facts: dict[str, Any]


class ClassifyIntentInput(NodeInput):
    message: str = ""


class ClassifyIntentOutput(NodeOutput):
    route_decision: RouteDecision


class PlanWorkInput(NodeInput):
    message: str = ""
    route_decision: RouteDecision


class PlanWorkOutput(NodeOutput):
    task_dag: TaskDAG


class RunJobSubgraphInput(NodeInput):
    task_dag: TaskDAG


class RunJobSubgraphOutput(NodeOutput):
    result: dict[str, Any]


class ClarifyInput(NodeInput):
    route_decision: RouteDecision | None = None


class ClarifyOutput(NodeOutput):
    clarification: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Job Search: discovery and selection
# ---------------------------------------------------------------------------


class LoadSearchProfileInput(NodeInput):
    user_id: UUID
    query: str | None = None


class LoadSearchProfileOutput(NodeOutput):
    search_profile: SearchProfile


class SearchProvidersInput(NodeInput):
    search_profile: SearchProfile


class SearchProvidersOutput(NodeOutput):
    raw_postings: list[RawPosting]
    provider_failures: list[ProviderFailure]


class NormalizeJobsInput(NodeInput):
    raw_postings: list[RawPosting] = []


class NormalizeJobsOutput(NodeOutput):
    normalized_postings: list[NormalizedPosting]


class DeduplicateInput(NodeInput):
    normalized_postings: list[NormalizedPosting] = []


class DeduplicateOutput(NodeOutput):
    deduplicated_postings: list[NormalizedPosting]
    duplicate_dedupe_keys: list[str]


class PersistPostingsInput(NodeInput):
    deduplicated_postings: list[NormalizedPosting] = []


class PersistPostingsOutput(NodeOutput):
    persisted_postings: list[PersistedPosting]


class HardFilterInput(NodeInput):
    search_profile: SearchProfile
    deduplicated_postings: list[NormalizedPosting] = []


class HardFilterOutput(NodeOutput):
    filtered_postings: list[NormalizedPosting]
    filter_rejections: list[FilterRejection]


class ScoreCandidatesInput(NodeInput):
    search_profile: SearchProfile
    filtered_postings: list[NormalizedPosting] = []


class ScoreCandidatesOutput(NodeOutput):
    scored_postings: list[ScoredPosting]


class EvidenceCheckInput(NodeInput):
    search_profile: SearchProfile
    scored_postings: list[ScoredPosting] = []


class EvidenceCheckOutput(NodeOutput):
    evidence_checks: list[EvidenceCheck]


class RankInput(NodeInput):
    scored_postings: list[ScoredPosting] = []


class RankOutput(NodeOutput):
    ranked_postings: list[ScoredPosting]


class ShortlistInput(NodeInput):
    search_profile: SearchProfile
    ranked_postings: list[ScoredPosting] = []
    evidence_checks: list[EvidenceCheck] = []


class ShortlistOutput(NodeOutput):
    shortlist: list[ShortlistEntry]


# ---------------------------------------------------------------------------
# Job Search: application and the approval triple
# ---------------------------------------------------------------------------


class PrepareApplicationPacketInput(NodeInput):
    search_profile: SearchProfile
    shortlist: list[ShortlistEntry] = Field(min_length=1)


class PrepareApplicationPacketOutput(NodeOutput):
    application_packet: ApplicationPacket
    pending_actions: list[ActionIntent] = Field(min_length=1, max_length=1)
    approval_stage: Literal["external_submission"]


class RequestApprovalInput(NodeInput):
    pending_actions: list[ActionIntent] = []
    approval_requests: list[ApprovalRequest] = []


class RequestApprovalOutput(NodeOutput):
    approval_requests: list[ApprovalRequest]


class ApprovalCheckpointInput(NodeInput):
    pending_actions: list[ActionIntent] = []
    approval_requests: list[ApprovalRequest] = []
    approvals: list[ApprovalDecision] = []
    approval_stage: ApprovalStage | None = None


class ApprovalCheckpointOutput(NodeOutput):
    #: Absent when nothing was pending.
    approvals: list[ApprovalDecision] | None = None


class ExecuteApprovedActionsInput(NodeInput):
    pending_actions: list[ActionIntent] = []
    approval_requests: list[ApprovalRequest] = []
    approvals: list[ApprovalDecision] = []
    action_receipts: list[ActionReceipt] = []
    approval_refusals: list[ApprovalRefusal] = []


class ExecuteApprovedActionsOutput(NodeOutput):
    #: Both absent when nothing was pending.
    action_receipts: list[ActionReceipt] | None = None
    approval_refusals: list[ApprovalRefusal] | None = None


class PersistApplicationInput(NodeInput):
    search_profile: SearchProfile
    application_packet: ApplicationPacket
    pending_actions: list[ActionIntent] = []
    action_receipts: list[ActionReceipt] = []


class PersistApplicationOutput(NodeOutput):
    application: PersistedApplication


class EmitApplicationCreatedInput(NodeInput):
    application: PersistedApplication


class EmitApplicationCreatedOutput(NodeOutput):
    emitted_events: list[EmittedEvent] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Job Search: recruiter lifecycle and durable follow-ups
# ---------------------------------------------------------------------------


class HandleRecruiterResponseInput(NodeInput):
    application: PersistedApplication
    emitted_events: list[EmittedEvent] = []


class HandleRecruiterResponseOutput(NodeOutput):
    recruiter_messages: list[RecruiterMessage]
    recruiter_responses: list[RecruiterResponse]
    emitted_events: list[EmittedEvent]
    pending_actions: list[ActionIntent]
    approval_stage: Literal["recruiter_outreach"]


class CreateFollowUpCheckpointInput(NodeInput):
    application: PersistedApplication
    recruiter_responses: list[RecruiterResponse] = []
    emitted_events: list[EmittedEvent] = []


class CreateFollowUpCheckpointOutput(NodeOutput):
    follow_up_checkpoints: list[FollowUpCheckpoint]
    pending_checkpoints: list[PendingCheckpoint]
    emitted_events: list[EmittedEvent]


class DraftFollowUpInput(NodeInput):
    fired_checkpoints: list[PendingCheckpoint] = Field(min_length=1)
    #: Required unless every fired checkpoint is an interview-prep reminder.
    application: PersistedApplication | None = None
    emitted_events: list[EmittedEvent] = []


class DraftFollowUpOutput(NodeOutput):
    pending_actions: list[ActionIntent]
    approval_stage: Literal["recruiter_outreach"]
    emitted_events: list[EmittedEvent]
    fired_checkpoints: Consumed


class ProposeRequestedTransitionsInput(NodeInput):
    requested_transitions: list[RequestedTransition] = Field(min_length=1)


class ProposeRequestedTransitionsOutput(NodeOutput):
    pending_actions: list[ActionIntent]
    approval_stage: Literal["lifecycle_transition"]


class ApplyRequestedTransitionsInput(NodeInput):
    requested_transitions: list[RequestedTransition] = []
    action_receipts: list[ActionReceipt] = []
    approval_refusals: list[ApprovalRefusal] = []
    transition_outcomes: list[TransitionOutcome] = []


class ApplyRequestedTransitionsOutput(NodeOutput):
    transition_outcomes: list[TransitionOutcome]
    requested_transitions: Consumed


# ---------------------------------------------------------------------------
# Job Search: inbound recruiter mail and interview scheduling
# ---------------------------------------------------------------------------


class ClassifyRecruiterEventsInput(NodeInput):
    user_id: UUID
    inbound_messages: list[RecruiterMessage] = Field(min_length=1)
    confirmed_correlations: dict[str, UUID] = {}


class ClassifyRecruiterEventsOutput(NodeOutput):
    recruiter_events: list[RecruiterEvent]
    inbound_messages: Consumed
    confirmed_correlations: ConsumedMapping


class RecordRecruiterEventsInput(NodeInput):
    recruiter_events: list[RecruiterEvent] = []
    emitted_events: list[EmittedEvent] = []


class RecordRecruiterEventsOutput(NodeOutput):
    recruiter_event_records: list[RecruiterEventRecord]
    recruiter_event_reviews: list[RecruiterEventReview]
    emitted_events: list[EmittedEvent]
    pending_actions: list[ActionIntent]
    approval_stage: Literal["recruiter_outreach"]
    interview_requests: list[InterviewRequest]


class PlanInterviewScheduleInput(NodeInput):
    user_id: UUID
    calendar_changes: list[CalendarEvent] = []
    interview_requests: list[InterviewRequest] = []
    pending_actions: list[ActionIntent] = []
    pending_checkpoints: list[PendingCheckpoint] = []
    emitted_events: list[EmittedEvent] = []


class PlanInterviewScheduleOutput(NodeOutput):
    interview_schedules: list[InterviewSchedulePlan]
    pending_actions: list[ActionIntent]
    pending_checkpoints: list[PendingCheckpoint]
    emitted_events: list[EmittedEvent]
    interview_requests: Consumed
    calendar_changes: Consumed
    #: Set only when no replies from an earlier node are being carried along.
    approval_stage: Literal["interview_scheduling"] | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _contract(
    graph: str, node: str, method: str, input: type[NodeInput], output: type[NodeOutput]
) -> tuple[tuple[str, str], NodeContract]:
    return (graph, node), NodeContract(graph, node, method, input, output)


#: Every node registered in a graph, keyed by `(graph, node)`.
NODE_CONTRACTS: Mapping[tuple[str, str], NodeContract] = dict(
    [
        _contract(
            SUPERVISOR_GRAPH,
            sup.LOAD_CONTEXT,
            "_load_context",
            LoadContextInput,
            LoadContextOutput,
        ),
        _contract(
            SUPERVISOR_GRAPH,
            sup.CLASSIFY_INTENT,
            "_classify_intent",
            ClassifyIntentInput,
            ClassifyIntentOutput,
        ),
        _contract(SUPERVISOR_GRAPH, sup.PLAN_WORK, "_plan_work", PlanWorkInput, PlanWorkOutput),
        _contract(
            SUPERVISOR_GRAPH,
            sup.RUN_JOB_SUBGRAPH,
            "_run_job_subgraph",
            RunJobSubgraphInput,
            RunJobSubgraphOutput,
        ),
        _contract(SUPERVISOR_GRAPH, sup.CLARIFY, "_clarify", ClarifyInput, ClarifyOutput),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.LOAD_SEARCH_PROFILE,
            "load_search_profile",
            LoadSearchProfileInput,
            LoadSearchProfileOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.SEARCH_PROVIDERS,
            "search_providers",
            SearchProvidersInput,
            SearchProvidersOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.NORMALIZE_JOBS,
            "normalize_jobs",
            NormalizeJobsInput,
            NormalizeJobsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.DEDUPLICATE,
            "deduplicate",
            DeduplicateInput,
            DeduplicateOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.PERSIST_POSTINGS,
            "persist_postings",
            PersistPostingsInput,
            PersistPostingsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.HARD_FILTER,
            "hard_filter",
            HardFilterInput,
            HardFilterOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.SCORE_CANDIDATES,
            "score_candidates",
            ScoreCandidatesInput,
            ScoreCandidatesOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.EVIDENCE_CHECK,
            "evidence_check",
            EvidenceCheckInput,
            EvidenceCheckOutput,
        ),
        _contract(JOB_SEARCH_GRAPH, jsg.RANK, "rank", RankInput, RankOutput),
        _contract(JOB_SEARCH_GRAPH, jsg.SHORTLIST, "shortlist", ShortlistInput, ShortlistOutput),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.PREPARE_APPLICATION_PACKET,
            "prepare_application_packet",
            PrepareApplicationPacketInput,
            PrepareApplicationPacketOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.REQUEST_APPROVAL,
            "request_approval",
            RequestApprovalInput,
            RequestApprovalOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.APPROVAL_CHECKPOINT,
            "approval_checkpoint",
            ApprovalCheckpointInput,
            ApprovalCheckpointOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.EXECUTE_APPROVED_ACTIONS,
            "execute_approved_actions",
            ExecuteApprovedActionsInput,
            ExecuteApprovedActionsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.PERSIST_APPLICATION,
            "persist_application",
            PersistApplicationInput,
            PersistApplicationOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.EMIT_APPLICATION_CREATED,
            "emit_application_created",
            EmitApplicationCreatedInput,
            EmitApplicationCreatedOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.HANDLE_RECRUITER_RESPONSE,
            "handle_recruiter_response",
            HandleRecruiterResponseInput,
            HandleRecruiterResponseOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.CREATE_FOLLOW_UP_CHECKPOINT,
            "create_follow_up_checkpoint",
            CreateFollowUpCheckpointInput,
            CreateFollowUpCheckpointOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.DRAFT_FOLLOW_UP,
            "draft_follow_up",
            DraftFollowUpInput,
            DraftFollowUpOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.PROPOSE_REQUESTED_TRANSITIONS,
            "propose_requested_transitions",
            ProposeRequestedTransitionsInput,
            ProposeRequestedTransitionsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.APPLY_REQUESTED_TRANSITIONS,
            "apply_requested_transitions",
            ApplyRequestedTransitionsInput,
            ApplyRequestedTransitionsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.CLASSIFY_RECRUITER_EVENTS,
            "classify_recruiter_events",
            ClassifyRecruiterEventsInput,
            ClassifyRecruiterEventsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.RECORD_RECRUITER_EVENTS,
            "record_recruiter_events",
            RecordRecruiterEventsInput,
            RecordRecruiterEventsOutput,
        ),
        _contract(
            JOB_SEARCH_GRAPH,
            jsg.PLAN_INTERVIEW_SCHEDULE,
            "plan_interview_schedule",
            PlanInterviewScheduleInput,
            PlanInterviewScheduleOutput,
        ),
    ]
)


def contracts_for(graph: str) -> dict[str, NodeContract]:
    """This graph's contracts, keyed by node name."""
    return {node: contract for (name, node), contract in NODE_CONTRACTS.items() if name == graph}


__all__ = [
    "SUPERVISOR_GRAPH",
    "JOB_SEARCH_GRAPH",
    "Consumed",
    "ConsumedMapping",
    "ApprovalStage",
    "NodeInput",
    "NodeOutput",
    "NodeContract",
    "NODE_CONTRACTS",
    "contracts_for",
]
