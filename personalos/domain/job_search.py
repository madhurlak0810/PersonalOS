"""Typed contracts carried between Job Search subgraph nodes.

Every node of the Job Search subgraph (see `personalos.graphs.job_search`) is
a transformer over the values defined here: it reads a narrow slice of them,
calls zero or more injected tool interfaces, and returns more of them. Keeping
the shapes in `domain` rather than in the graph module is what makes that
contract checkable -- a node's input and output can be validated in a unit
test without building a graph, a checkpointer, or a provider.

Three properties are load-bearing:

- **Value objects, not records.** Everything here is `frozen=True` and
  `extra="forbid"`. A node cannot mutate a posting in place (so a later node
  can never silently depend on an earlier one's side effect), and a provider
  cannot smuggle an unexpected field through normalization.
- **Provenance survives.** A `NormalizedPosting` keeps the `raw` payload it
  was built from, and a `ScoredPosting` keeps the per-component scores and
  human-readable reasons behind its number, so a shortlist can always be
  explained rather than just asserted.
- **Side effects are proposals.** `ActionIntent` describes an outward-facing
  action (submitting an application, messaging a recruiter) *without*
  performing it. The only thing that turns one into something executable is
  an `ApprovalDecision` bound to its fingerprint, which mirrors how
  `personalos.policy.intents.ApprovalGrant` binds a human grant to one
  specific `ToolIntent`.
"""

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.models import (
    ApplicationStatus,
    ArtifactType,
    CommunicationEventClassification,
    validate_evidence_links,
    validate_idempotency_key,
)

#: Hard cap on how many postings a single run may shortlist. The shortlist is
#: the hand-off point to work that costs money and attention (packet
#: generation, a human approval), so it is bounded here rather than left to
#: whatever the scorer happened to rank.
MAX_SHORTLIST_SIZE = 25

#: Cap on postings carried forward from the provider fan-out. A provider that
#: returns an unbounded page still cannot make the rest of the pipeline
#: unbounded.
MAX_POSTINGS_PER_RUN = 500

#: Score below which a posting is never shortlisted, regardless of rank. A
#: run that finds nothing good shortlists nothing, rather than shortlisting
#: its least-bad option.
DEFAULT_MIN_SCORE = 0.5


class JobSearchContractError(ValidationFailed, ValueError):
    """A value crossing between Job Search nodes violated its contract.

    Subclasses both `ValidationFailed` (reports through the shared error
    taxonomy) and `ValueError`, matching `InvalidIdempotencyKey` and
    `UnsupportedRouteDomain` elsewhere in `personalos.domain`.
    """


class _Value(BaseModel):
    """Base for every value passed between nodes: immutable and closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_hash(payload: Any) -> str:
    """Stable SHA-256 of a JSON-ish value.

    Mirrors `personalos.policy.intents.fingerprint_intent`: canonical JSON so
    key order cannot change the hash, and `default=repr` so an unserializable
    value degrades to a stable string instead of raising and blocking an
    otherwise valid posting.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()


# --- Search profile ----------------------------------------------------------


class SearchProfile(_Value):
    """What the candidate is looking for, as `load_search_profile` resolved it.

    This is the only place the rest of the pipeline reads targeting criteria
    from: `hard_filter` and `score_candidates` take a profile, not a bag of
    request parameters, so a criterion that is not on this model cannot
    influence a filtering or scoring decision.
    """

    user_id: UUID
    profile_version: int = 1
    target_roles: tuple[str, ...] = ()
    target_locations: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    must_have_skills: tuple[str, ...] = ()
    excluded_companies: tuple[str, ...] = ()
    salary_min: int | None = None
    salary_max: int | None = None
    remote_only: bool = False
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=0.0, le=1.0)
    max_shortlist: int = Field(default=5, ge=1, le=MAX_SHORTLIST_SIZE)

    @field_validator("salary_max")
    @classmethod
    def _max_not_below_min(cls, value: int | None, info) -> int | None:
        minimum = info.data.get("salary_min")
        if value is not None and minimum is not None and value < minimum:
            raise JobSearchContractError(
                f"salary_max ({value}) cannot be below salary_min ({minimum})"
            )
        return value


# --- Postings ----------------------------------------------------------------


class RawPosting(_Value):
    """One posting exactly as a provider returned it, before normalization.

    Kept as an opaque `payload` on purpose: the provider's shape is the
    provider's business, and `normalize_jobs` is the single place that
    interprets it. Nothing downstream of normalization reads `payload`.
    """

    provider: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def _provider_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("provider must not be blank")
        return value


class ProviderFailure(_Value):
    """A provider that failed during fan-out, recorded rather than raised.

    One dead job board must not fail the whole search: the run continues with
    the providers that answered, and this is how the caller learns which did
    not, so a thin result set is distinguishable from a genuinely empty one.
    """

    provider: str
    error: str
    error_code: str | None = None


class NormalizedPosting(_Value):
    """A posting in the one shape the rest of the pipeline understands.

    `dedupe_key` and `description_hash` are derived here rather than supplied,
    so two providers describing the same opening collapse on identical
    content instead of on whatever id each of them happened to mint. Both
    fields line up with `personalos.persistence.models.JobPostingModel`'s
    columns of the same name.
    """

    source: str
    source_job_id: str | None = None
    title: str
    company: str
    location: str | None = None
    url: str | None = None
    description: str = ""
    salary_min: int | None = None
    salary_max: int | None = None
    remote: bool = False
    posted_at: datetime | None = None
    skills: tuple[str, ...] = ()
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "company")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("title and company must not be blank")
        return value

    @property
    def description_hash(self) -> str:
        """Content hash of the posting text, for near-duplicate detection."""
        return _canonical_hash(self.description.strip().lower())

    @property
    def dedupe_key(self) -> str:
        """Content-derived identity: same company, same title, same text.

        Truncated to fit `job_postings.dedupe_key` (varchar(150)); the
        embedded hash is what carries the uniqueness, the readable prefix is
        there so a duplicate is diagnosable by eye.
        """
        company = self.company.strip().lower()
        title = self.title.strip().lower()
        return f"{company}|{title}|{self.description_hash}"[:150]


# --- Scoring and evidence ----------------------------------------------------


class FilterRejection(_Value):
    """Why one posting did not survive `hard_filter`.

    Rejections are retained rather than dropped: a run that filtered out
    everything is a reviewable outcome ("all of them were below your salary
    floor"), not an unexplained empty list.
    """

    dedupe_key: str
    title: str
    company: str
    reason: str


class ScoredPosting(_Value):
    """A posting with its match score and the components behind it.

    `components` is the per-signal breakdown and `reasons` the human-readable
    summary. A score with neither is not accepted -- an unexplainable ranking
    is the failure mode this model exists to prevent.
    """

    posting: NormalizedPosting
    score: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @field_validator("components")
    @classmethod
    def _components_present(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise JobSearchContractError(
                "a scored posting must carry at least one scoring component"
            )
        return value


class EvidenceRef(_Value):
    """One citation from the candidate's own record backing a claim.

    `type`/`ref` match what `personalos.domain.models.validate_evidence_links`
    requires, so an `ApplicationPacket`'s citations can be handed straight to
    the artifact-version validator without reshaping.
    """

    type: str
    ref: str
    excerpt: str | None = None

    @field_validator("type", "ref")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("evidence type and ref must not be blank")
        return value


class EvidenceCheck(_Value):
    """Whether a scored posting's match is grounded in the candidate's record.

    A posting that scores well on keywords but cites nothing is not shortlisted
    (`grounded` is False), which is what stops the pipeline from recommending a
    role on the strength of a coincidence in the job text.
    """

    dedupe_key: str
    grounded: bool
    citations: tuple[EvidenceRef, ...] = ()
    unsupported_claims: tuple[str, ...] = ()

    @field_validator("citations")
    @classmethod
    def _grounded_needs_citations(
        cls, value: tuple[EvidenceRef, ...], info
    ) -> tuple[EvidenceRef, ...]:
        if info.data.get("grounded") and not value:
            raise JobSearchContractError(
                "an evidence check cannot report grounded=True with no citations"
            )
        return value


class ShortlistEntry(_Value):
    """One posting the run is recommending, at its final rank."""

    rank: int = Field(ge=1)
    scored: ScoredPosting
    evidence: EvidenceCheck


# --- Application packet ------------------------------------------------------


class ArtifactDraft(_Value):
    """A tailored resume or cover letter, with the evidence it was built from.

    `evidence` is validated with the same
    `personalos.domain.models.validate_evidence_links` the persistence layer
    applies to `artifact_versions`, so a draft that cannot be persisted is
    rejected here -- at the node that produced it -- rather than several steps
    later at the write.
    """

    artifact_type: ArtifactType
    content: str
    evidence: tuple[EvidenceRef, ...]

    @field_validator("evidence")
    @classmethod
    def _cites_evidence(cls, value: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        validate_evidence_links([ref.model_dump(mode="json") for ref in value])
        return value


class ApplicationPacket(_Value):
    """Everything assembled for one application, still unsubmitted.

    Producing a packet is deliberately separate from submitting one: this is a
    local artifact, and the only thing that can turn it into an outward-facing
    submission is an approved `ActionIntent`.
    """

    dedupe_key: str
    posting: NormalizedPosting
    artifacts: tuple[ArtifactDraft, ...]
    answers: dict[str, str] = Field(default_factory=dict)

    @field_validator("artifacts")
    @classmethod
    def _needs_a_resume(cls, value: tuple[ArtifactDraft, ...]) -> tuple[ArtifactDraft, ...]:
        if not any(draft.artifact_type == ArtifactType.RESUME for draft in value):
            raise JobSearchContractError("an application packet must include a resume draft")
        return value


# --- Actions, approvals ------------------------------------------------------


class ActionKind(str, Enum):
    """The outward-facing actions this subgraph may propose.

    A closed set, for the same reason `RouteDomain` is closed: a node cannot
    invent a new kind of side effect, and the approval node's handling of each
    kind is reviewable code rather than a generic passthrough.
    """

    SUBMIT_APPLICATION = "submit_application"
    SEND_RECRUITER_MESSAGE = "send_recruiter_message"


class ActionIntent(_Value):
    """A proposed outward-facing action. Never executed by the node that made it.

    The analogue of `personalos.policy.intents.ToolIntent` one level up: a
    proposal, carrying enough provenance to audit and enough identity
    (`fingerprint`) that an approval cannot be replayed against different
    arguments. A node that wants to submit an application returns one of these
    and the graph routes it to the approval checkpoint; there is no code path
    from a node to a submission.
    """

    action_id: UUID = Field(default_factory=uuid4)
    kind: ActionKind
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str
    requested_by: str = "graph:job_search"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("idempotency_key")
    @classmethod
    def _check_idempotency_key(cls, value: str) -> str:
        return validate_idempotency_key(value)

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError(
                "an action intent must carry a human-readable summary; it is what a "
                "reviewer approves"
            )
        return value

    def fingerprint(self) -> str:
        """Stable hash of the side effect this intent describes."""
        return _canonical_hash({"kind": self.kind.value, "payload": self.payload})


class ApprovalVerdict(str, Enum):
    """What a reviewer decided about one proposed action."""

    APPROVED = "approved"
    REJECTED = "rejected"
    #: The reviewer has not answered yet. The run stops at the checkpoint;
    #: it does not proceed on an unanswered request.
    PENDING = "pending"


class ApprovalDecision(_Value):
    """A reviewer's verdict on one `ActionIntent`, bound to its fingerprint.

    Bound rather than merely referenced, exactly as
    `personalos.policy.intents.ApprovalGrant.matches` is: an approval of
    "apply to the backend role at Acme" must not clear a mutated payload that
    applies somewhere else.
    """

    action_id: UUID
    action_fingerprint: str
    verdict: ApprovalVerdict
    decided_by: str
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    note: str | None = None

    def authorizes(self, intent: ActionIntent) -> bool:
        """True only for an approval issued for exactly this intent."""
        return (
            self.verdict == ApprovalVerdict.APPROVED
            and self.action_id == intent.action_id
            and self.action_fingerprint == intent.fingerprint()
        )


class ActionReceipt(_Value):
    """What came back from redeeming one approved `ActionIntent`.

    The analogue of `personalos.tools.gateway.ToolResult` at this level: it
    ties an outcome back to the action it settles, so a persisted application
    can record the external reference the submission returned rather than
    assuming one.
    """

    action_id: UUID
    ok: bool
    external_reference: str | None = None
    detail: str | None = None


# --- Persistence outcome and events -----------------------------------------


class PersistedApplication(_Value):
    """The application row the run produced, as the store reported it back.

    `submitted` is separate from `status`: an application can be persisted in
    `READY_TO_APPLY` because the reviewer rejected the submission, and the
    distinction between "we have a record" and "we sent it" must survive into
    the final state.
    """

    application_id: UUID
    job_posting_id: UUID
    user_id: UUID
    dedupe_key: str
    status: ApplicationStatus
    submitted: bool = False
    artifact_version_ids: tuple[UUID, ...] = ()
    external_reference: str | None = None


class JobSearchEventType(str, Enum):
    """Domain events this subgraph emits.

    Values are the wire names written to `outbox_events.type` /
    `event_log.event_type`, which is why they are dotted strings rather than
    Python-shaped identifiers.
    """

    APPLICATION_CREATED = "application.created"
    APPLICATION_SUBMISSION_REJECTED = "application.submission_rejected"
    RECRUITER_RESPONSE_RECORDED = "application.recruiter_response_recorded"
    FOLLOW_UP_SCHEDULED = "application.follow_up_scheduled"


class EmittedEvent(_Value):
    """One domain event handed to the outbox, with its dedupe identity.

    `dedupe_key` maps onto `outbox_events.dedupe_key`, so re-running a step
    after a crash enqueues the same event rather than a second copy of it.
    """

    type: JobSearchEventType
    aggregate_id: UUID
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str | None = None
    occurred_at: datetime = Field(default_factory=datetime.utcnow)


# --- Recruiter responses and follow-ups -------------------------------------


class RecruiterMessage(_Value):
    """One inbound recruiter-side message tied to an application.

    Handled inside this subgraph rather than by a Communications subgraph:
    for a job-search-only build, a recruiter reply is a step in the
    application's own lifecycle, and splitting it out would mean a second
    graph whose only job is to hand state back to this one.
    """

    provider_message_id: str
    received_at: datetime
    body: str = ""
    subject: str | None = None
    from_address: str | None = None


class RecruiterResponse(_Value):
    """A recruiter message after classification, plus the status move it implies.

    `implied_status` is a *proposal*, not a write: `persist_application`'s
    store runs it through
    `personalos.domain.models.validate_application_status_transition`, so a
    classification cannot drive an application into a state the lifecycle
    does not allow.
    """

    provider_message_id: str
    classification: CommunicationEventClassification
    occurred_at: datetime
    implied_status: ApplicationStatus | None = None
    requires_reply: bool = False
    summary: str | None = None


class FollowUpKind(str, Enum):
    """Why a follow-up checkpoint was created."""

    #: Nothing came back at all; check in after the quiet period.
    NO_RESPONSE = "no_response"
    #: The recruiter asked for something; the candidate owes a reply.
    AWAITING_CANDIDATE_REPLY = "awaiting_candidate_reply"
    #: An interview is booked; prepare before it.
    INTERVIEW_PREP = "interview_prep"


class FollowUpCheckpoint(_Value):
    """A dated reminder attached to an application.

    Created as a branch of this subgraph, for the same reason recruiter
    responses are: the checkpoint exists to move *this* application forward,
    and a standalone Calendar subgraph is out of scope for this build.
    Scheduling one is an internal record, so it needs no approval -- messaging
    the recruiter about it is a separate `ActionIntent` that does.
    """

    application_id: UUID
    kind: FollowUpKind
    due_at: datetime
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("a follow-up checkpoint must record why it exists")
        return value


__all__ = [
    "MAX_SHORTLIST_SIZE",
    "MAX_POSTINGS_PER_RUN",
    "DEFAULT_MIN_SCORE",
    "JobSearchContractError",
    "SearchProfile",
    "RawPosting",
    "ProviderFailure",
    "NormalizedPosting",
    "FilterRejection",
    "ScoredPosting",
    "EvidenceRef",
    "EvidenceCheck",
    "ShortlistEntry",
    "ArtifactDraft",
    "ApplicationPacket",
    "ActionKind",
    "ActionIntent",
    "ApprovalVerdict",
    "ApprovalDecision",
    "ActionReceipt",
    "PersistedApplication",
    "JobSearchEventType",
    "EmittedEvent",
    "RecruiterMessage",
    "RecruiterResponse",
    "FollowUpKind",
    "FollowUpCheckpoint",
]
