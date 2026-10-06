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
import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
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


#: Characters removed from provider text before it is stored: C0/C1 controls
#: other than tab and newline, plus the zero-width and bidirectional-override
#: code points. None of them carry meaning in a job posting, and all of them
#: are ways to make stored text read differently to a human than to a model.
_UNSAFE_CHARS = re.compile(
    "[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]"
)

#: Legal-form suffixes dropped when comparing company names, so "Acme, Inc."
#: on one board and "Acme" on another are the same employer.
_COMPANY_SUFFIXES = frozenset(
    {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "gmbh", "plc"}
)

#: Room left for the readable part of a dedupe key: `job_postings.dedupe_key`
#: is varchar(150), and the trailing separator plus SHA-256 digest take 65.
_DEDUPE_PREFIX_LENGTH = 85


def strip_unsafe_chars(text: str) -> str:
    """Remove control, zero-width and bidi-override characters from provider text."""
    return _UNSAFE_CHARS.sub("", text)


def canonical_text(text: str) -> str:
    """Fold text to the form postings are compared in.

    Unicode-normalized, case-folded, with every run of non-alphanumerics
    collapsed to one space. Two boards rendering the same description with
    different whitespace, bullets or quote styles therefore hash the same.
    """
    folded = unicodedata.normalize("NFKC", strip_unsafe_chars(text)).casefold()
    return " ".join(re.findall(r"\w+", folded))


def canonical_company(name: str) -> str:
    """`canonical_text` for a company name, minus trailing legal-form suffixes."""
    words = canonical_text(name).split()
    while len(words) > 1 and words[-1] in _COMPANY_SUFFIXES:
        words.pop()
    return " ".join(words)


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

    @field_validator("title", "company", "location", "description", mode="before")
    @classmethod
    def _strip_unsafe(cls, value: Any) -> Any:
        return strip_unsafe_chars(value) if isinstance(value, str) else value

    @field_validator("title", "company")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("title and company must not be blank")
        return value

    @property
    def description_hash(self) -> str:
        """Content hash of the posting text, for near-duplicate detection."""
        return _canonical_hash(canonical_text(self.description))

    @property
    def dedupe_key(self) -> str:
        """Content-derived identity: same company, same title, same text.

        A readable `company|title` prefix followed by a digest of all three.
        The prefix is truncated to keep the key inside
        `job_postings.dedupe_key` (varchar(150)); the digest never is, since it
        is what carries the uniqueness -- the prefix is only there so a
        duplicate is diagnosable by eye.
        """
        company = canonical_company(self.company)
        title = canonical_text(self.title)
        digest = _canonical_hash([company, title, self.description_hash])
        return f"{f'{company}|{title}'[:_DEDUPE_PREFIX_LENGTH]}|{digest}"


class PersistedPosting(_Value):
    """The `job_postings` row a discovered posting resolved to.

    `created` is False when the row already existed -- the posting was stored
    by an earlier run, or by another provider earlier in this one.
    """

    job_posting_id: UUID
    dedupe_key: str
    created: bool


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


class RiskLevel(str, Enum):
    """How much an action costs to get wrong, from the reviewer's point of view.

    Not a probability and not a severity score -- a reviewer's triage label.
    The question it answers is "how carefully do I have to read this before
    saying yes", and the ordering is by how recoverable the action is: a
    message can be followed by a correction, an application cannot be
    unsubmitted.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionRiskProfile(_Value):
    """The review terms attached to one `ActionKind`.

    A table rather than a judgement made per call: what an application
    submission risks, and which capabilities it needs, is a property of the
    kind of action, and deriving it per intent would let a node quietly
    request a cheaper review for the same side effect.

    `scopes` names the capabilities the action consumes, in the same
    `resource:verb` vocabulary `policy_decisions.requested_scopes` records, so
    an approval can be checked against what the approver is actually willing
    to delegate rather than against a free-text summary.
    """

    kind: ActionKind
    risk: RiskLevel
    scopes: tuple[str, ...]
    #: How long an approval for this kind stays good for. Bounded because an
    #: approval is a statement about the world as the reviewer saw it, and the
    #: world moves: a posting closes, a recruiter thread goes cold.
    approval_ttl: timedelta

    @field_validator("scopes")
    @classmethod
    def _scopes_present(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise JobSearchContractError(
                "an outward-facing action must name the scopes it consumes; a request "
                "for no scopes is a request a reviewer cannot evaluate"
            )
        return value


#: The review terms for every kind of outward-facing action, keyed by kind.
#: Read through `risk_profile_for`, which fails loudly on a kind that was added
#: to `ActionKind` without deciding what reviewing it costs.
ACTION_RISK_PROFILES: Mapping[ActionKind, ActionRiskProfile] = MappingProxyType(
    {
        ActionKind.SUBMIT_APPLICATION: ActionRiskProfile(
            kind=ActionKind.SUBMIT_APPLICATION,
            # Unrecoverable: an application cannot be withdrawn from the
            # company's side of the transaction once it has landed.
            risk=RiskLevel.HIGH,
            scopes=("applications:submit", "artifacts:read"),
            approval_ttl=timedelta(days=7),
        ),
        ActionKind.SEND_RECRUITER_MESSAGE: ActionRiskProfile(
            kind=ActionKind.SEND_RECRUITER_MESSAGE,
            # Embarrassing rather than unrecoverable: a wrong message can be
            # followed by a correction to the same thread.
            risk=RiskLevel.MEDIUM,
            scopes=("communications:send",),
            approval_ttl=timedelta(days=3),
        ),
    }
)


def risk_profile_for(kind: ActionKind) -> ActionRiskProfile:
    """Return the review terms for an action kind.

    Raises rather than defaulting: a new `ActionKind` with no entry in
    `ACTION_RISK_PROFILES` is a side effect nobody has decided how to review,
    and a permissive default would let it reach a reviewer labelled as cheap.
    """
    try:
        return ACTION_RISK_PROFILES[kind]
    except KeyError as exc:
        raise JobSearchContractError(
            f"no risk profile registered for action kind "
            f"'{getattr(kind, 'value', kind)}'; every outward-facing action must "
            f"declare its risk and scopes"
        ) from exc


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
    #: Who or what receives the write, as a reviewer would name it -- the
    #: posting's URL, the recruiter thread. Separate from `summary` because a
    #: reviewer checks the destination and the description independently: the
    #: whole failure this guards against is a plausible-sounding summary
    #: pointing somewhere else.
    target: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str
    requested_by: str = "graph:job_search"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("idempotency_key")
    @classmethod
    def _check_idempotency_key(cls, value: str) -> str:
        return validate_idempotency_key(value)

    @field_validator("summary", "target")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError(
                "an action intent must carry a human-readable summary and target; they "
                "are what a reviewer approves"
            )
        return value

    def fingerprint(self) -> str:
        """Stable hash of the side effect this intent describes.

        Covers the kind, the target and the payload: everything that decides
        *what happens outside this system*. It deliberately excludes
        `action_id`, `created_at` and `requested_by`, so re-proposing the same
        submission hashes the same -- and equally deliberately includes
        `target`, so redirecting an otherwise identical action at a different
        recipient produces a different hash and invalidates any approval held
        against the old one.
        """
        return _canonical_hash(
            {"kind": self.kind.value, "target": self.target, "payload": self.payload}
        )

    def risk_profile(self) -> "ActionRiskProfile":
        """The review terms this action is subject to."""
        return risk_profile_for(self.kind)


class ApprovalRequest(_Value):
    """What a reviewer is shown, and what an approval is later checked against.

    Minted by the node that parks the run and written to state *before* the
    graph interrupts, so it is part of the checkpoint the reviewer's answer
    eventually resumes. That ordering is the whole mechanism: `action_hash` is
    the hash of the action as it stood when the request went out, and it is the
    fixed point an executor compares a freshly recomputed hash against hours or
    days later. Recomputing the request instead of storing it would compare the
    mutated action against itself and always agree.

    Bounded by `expires_at` for the same reason `ApprovalGrant` is bound to a
    fingerprint: an approval is a statement about a specific action at a
    specific time, and neither the action nor the moment may drift out from
    under it.
    """

    request_id: UUID = Field(default_factory=uuid4)
    action_id: UUID
    #: `ActionIntent.fingerprint()` as of the moment the request was raised.
    action_hash: str
    kind: ActionKind
    target: str
    summary: str
    risk: RiskLevel
    requested_scopes: tuple[str, ...] = ()
    idempotency_key: str
    requested_by: str = "graph:job_search"
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def _expiry_after_request(cls, value: datetime, info) -> datetime:
        requested_at = info.data.get("requested_at")
        if requested_at is not None and value <= requested_at:
            raise JobSearchContractError(
                f"an approval request must expire after it was raised "
                f"(expires_at={value}, requested_at={requested_at})"
            )
        return value

    @classmethod
    def for_intent(
        cls,
        intent: ActionIntent,
        *,
        now: datetime | None = None,
        ttl: timedelta | None = None,
    ) -> "ApprovalRequest":
        """Raise the request a reviewer answers for this intent.

        `ttl` overrides the kind's own `approval_ttl`, which is what a
        deployment with a stricter (or, in a test, much shorter) review window
        passes. The hash is taken here, once, from the intent as it stands.
        """
        profile = intent.risk_profile()
        raised_at = now or datetime.utcnow()
        return cls(
            action_id=intent.action_id,
            action_hash=intent.fingerprint(),
            kind=intent.kind,
            target=intent.target,
            summary=intent.summary,
            risk=profile.risk,
            requested_scopes=profile.scopes,
            idempotency_key=intent.idempotency_key,
            requested_by=intent.requested_by,
            requested_at=raised_at,
            expires_at=raised_at + (ttl or profile.approval_ttl),
        )

    def is_expired(self, now: datetime) -> bool:
        """True once this request may no longer be answered."""
        return now >= self.expires_at

    def describes(self, intent: ActionIntent) -> bool:
        """True when this request was raised for that intent, whatever it now says."""
        return self.action_id == intent.action_id


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
    #: The `ApprovalRequest` this answers, when the decision came back through
    #: one. Optional so a standing approval resolved without ever raising a
    #: request is still expressible; when it is set, it is checked.
    request_id: UUID | None = None

    def authorizes(self, intent: ActionIntent) -> bool:
        """True only for an approval issued for exactly this intent."""
        return (
            self.verdict == ApprovalVerdict.APPROVED
            and self.action_id == intent.action_id
            and self.action_fingerprint == intent.fingerprint()
        )

    def answers(self, request: ApprovalRequest) -> bool:
        """True when this decision was issued for exactly that request."""
        return self.action_id == request.action_id and (
            self.request_id is None or self.request_id == request.request_id
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


class RefusalReason(str, Enum):
    """Why an approved-looking action was not executed after all.

    A closed set so a refusal is classifiable rather than a string an operator
    has to read: `HASH_MISMATCH` and `EXPIRED` mean something went wrong
    between the request and the resume and want investigating, while
    `NOT_APPROVED` is the ordinary outcome of a reviewer saying no.
    """

    #: The run reached the executor with no request on file for this action.
    NO_REQUEST = "no_approval_request"
    #: No decision came back for this action, or it was not an approval.
    NOT_APPROVED = "not_approved"
    #: A decision that belongs to some other action or some other request.
    MISDIRECTED_DECISION = "misdirected_decision"
    #: The reviewer approved a hash the action no longer has.
    HASH_MISMATCH = "action_hash_mismatch"
    #: The approval window closed before the run got back to the executor.
    EXPIRED = "approval_expired"


class ApprovalRefusal(_Value):
    """A refusal to execute, recorded rather than raised.

    Recorded because a refusal is an outcome the run has to carry forward: a
    submission refused at the executor still leaves a real application packet
    that should be persisted as prepared-but-unsent, and the reason it was
    refused is what an operator needs to decide whether to re-request approval
    or to investigate why the action changed.
    """

    action_id: UUID
    request_id: UUID | None
    reason: RefusalReason
    detail: str
    approved_hash: str | None = None
    recomputed_hash: str | None = None

    @property
    def suspicious(self) -> bool:
        """True for the reasons that mean something tampered with the action."""
        return self.reason in (RefusalReason.HASH_MISMATCH, RefusalReason.MISDIRECTED_DECISION)

    def to_receipt(self) -> ActionReceipt:
        """The not-ok receipt this refusal settles the action with."""
        return ActionReceipt(action_id=self.action_id, ok=False, detail=self.detail)


def authorize_execution(
    *,
    intent: ActionIntent,
    request: ApprovalRequest | None,
    decision: ApprovalDecision | None,
    now: datetime,
) -> ApprovalRefusal | None:
    """Decide whether an approved action may still be executed. `None` means yes.

    The last gate before a side effect, and the only one that runs *after* the
    checkpoint the approval was granted against. Everything it checks is a way
    the action and its approval can have drifted apart while the run was parked:

    - the request is the one raised for this action, and the decision answers
      that request (not a different action's, and not a different request for
      the same action);
    - the decision is an approval, of the hash the request went out with;
    - **the intent still hashes to what the request went out with.** This is the
      check the whole interrupt design exists for: between raising the request
      and resuming, the pending action lives in graph state, where a later node
      -- or a model driving one -- could rewrite it. Recomputing the hash here
      and comparing it to the stored `action_hash` is what makes a rewritten
      action fail closed instead of executing under someone else's approval;
    - the approval has not expired.

    Pure, and takes `now` rather than reading a clock, so every branch is
    testable without waiting and a caller cannot get a different answer than
    the one it will record.
    """
    if request is None:
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=None,
            reason=RefusalReason.NO_REQUEST,
            detail=(
                f"no approval request on file for action {intent.action_id} "
                f"({intent.kind.value}); it was never put to a reviewer"
            ),
            recomputed_hash=intent.fingerprint(),
        )

    if not request.describes(intent):
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.MISDIRECTED_DECISION,
            detail=(
                f"approval request {request.request_id} was raised for action "
                f"{request.action_id}, not {intent.action_id}"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=intent.fingerprint(),
        )

    if decision is None:
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.NOT_APPROVED,
            detail=(
                f"no decision recorded for action {intent.action_id} "
                f"({intent.kind.value}); the request is still unanswered"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=intent.fingerprint(),
        )

    if not decision.answers(request):
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.MISDIRECTED_DECISION,
            detail=(
                f"decision for action {decision.action_id} / request "
                f"{decision.request_id} does not answer request {request.request_id} "
                f"for action {request.action_id}"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=intent.fingerprint(),
        )

    if decision.verdict != ApprovalVerdict.APPROVED:
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.NOT_APPROVED,
            detail=(
                f"action {intent.action_id} ({intent.kind.value}) was not approved: "
                f"verdict={decision.verdict.value}"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=intent.fingerprint(),
        )

    if decision.action_fingerprint != request.action_hash:
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.HASH_MISMATCH,
            detail=(
                f"the approval for action {intent.action_id} is bound to hash "
                f"{decision.action_fingerprint}, which is not the hash the request "
                f"was raised with"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=intent.fingerprint(),
        )

    recomputed = intent.fingerprint()
    if recomputed != request.action_hash:
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.HASH_MISMATCH,
            detail=(
                f"action {intent.action_id} ({intent.kind.value}) changed after it was "
                f"approved: approved hash {request.action_hash}, now {recomputed}. "
                f"Refusing to execute under an approval given for a different action"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=recomputed,
        )

    if request.is_expired(now):
        return ApprovalRefusal(
            action_id=intent.action_id,
            request_id=request.request_id,
            reason=RefusalReason.EXPIRED,
            detail=(
                f"the approval for action {intent.action_id} expired at "
                f"{request.expires_at.isoformat()}; it must be requested again"
            ),
            approved_hash=request.action_hash,
            recomputed_hash=recomputed,
        )

    return None


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
    #: A scheduled follow-up's trigger came round with its condition still
    #: unmet, so a draft was proposed. Distinct from FOLLOW_UP_SCHEDULED
    #: because most scheduled follow-ups never become this one: the recruiter
    #: replies, the wait resolves silently, and nothing is emitted at all.
    FOLLOW_UP_TRIGGERED = "application.follow_up_triggered"


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
    "PersistedPosting",
    "strip_unsafe_chars",
    "canonical_text",
    "canonical_company",
    "FilterRejection",
    "ScoredPosting",
    "EvidenceRef",
    "EvidenceCheck",
    "ShortlistEntry",
    "ArtifactDraft",
    "ApplicationPacket",
    "ActionKind",
    "RiskLevel",
    "ActionRiskProfile",
    "ACTION_RISK_PROFILES",
    "risk_profile_for",
    "ActionIntent",
    "ApprovalRequest",
    "ApprovalVerdict",
    "ApprovalDecision",
    "ActionReceipt",
    "RefusalReason",
    "ApprovalRefusal",
    "authorize_execution",
    "PersistedApplication",
    "JobSearchEventType",
    "EmittedEvent",
    "RecruiterMessage",
    "RecruiterResponse",
    "FollowUpKind",
    "FollowUpCheckpoint",
]
