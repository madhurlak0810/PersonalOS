"""Fake implementations of every Job Search subgraph port.

One fake per port, each recording what it was called with. They are shared by
the graph scenario test and the per-node contract tests so both exercise the
same substitutes: a node test that passed against a hand-rolled stub while the
scenario test used a different one would not tell you the node and the graph
agree about the port.

Every fake is deterministic. Nothing here sleeps, reads a clock it did not
receive, or hashes on identity, so a scenario assertion can name exact scores
and ranks rather than ranges.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from personalos.domain.checkpoints import (
    CheckpointCondition,
    ConditionKind,
    PendingCheckpoint,
)
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApplicationPacket,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    ArtifactDraft,
    EmittedEvent,
    EvidenceCheck,
    EvidenceRef,
    FollowUpCheckpoint,
    NormalizedPosting,
    PersistedApplication,
    PersistedPosting,
    RawPosting,
    RecruiterMessage,
    RecruiterResponse,
    ScoredPosting,
    SearchProfile,
    ShortlistEntry,
)
from personalos.domain.models import (
    ApplicationStatus,
    ArtifactType,
    CommunicationEventClassification,
    validate_application_status_transition,
)

#: Fixed ids so assertions can name them.
USER_ID = UUID("11111111-1111-1111-1111-111111111111")
POSTING_ID = UUID("22222222-2222-2222-2222-222222222222")
APPLICATION_ID = UUID("33333333-3333-3333-3333-333333333333")

#: Fixed clock, so `posted_at` and follow-up due dates are reproducible.
NOW = datetime(2026, 9, 28, 12, 0, 0)


def profile(**overrides: Any) -> SearchProfile:
    """A search profile with sensible defaults for the happy path."""
    defaults: dict[str, Any] = {
        "user_id": USER_ID,
        "profile_version": 3,
        "target_roles": ("Backend Engineer",),
        "target_locations": ("Remote",),
        "keywords": ("python", "postgres"),
        "must_have_skills": ("python",),
        "excluded_companies": ("Stealth Co",),
        "salary_min": 120_000,
        "remote_only": True,
        "min_score": 0.5,
        "max_shortlist": 3,
    }
    defaults.update(overrides)
    return SearchProfile(**defaults)


def posting(
    *,
    source: str = "fakeboard",
    title: str = "Senior Backend Engineer",
    company: str = "Acme",
    description: str = "We need python and postgres experience.",
    remote: bool = True,
    salary_max: int | None = 190_000,
    skills: tuple[str, ...] = ("python", "postgres"),
    **overrides: Any,
) -> NormalizedPosting:
    """A normalized posting that clears the default profile's hard filters."""
    payload: dict[str, Any] = {
        "source": source,
        "source_job_id": "fb-1",
        "title": title,
        "company": company,
        "location": "Remote",
        "url": f"https://example.test/{company.lower().replace(' ', '-')}",
        "description": description,
        "salary_min": 140_000,
        "salary_max": salary_max,
        "remote": remote,
        "skills": skills,
        "posted_at": NOW,
    }
    payload.update(overrides)
    return NormalizedPosting(**payload)


def raw_posting(provider: str = "fakeboard", **payload: Any) -> RawPosting:
    """A provider payload in the shape `DictPostingNormalizer` understands."""
    defaults: dict[str, Any] = {
        "id": "fb-1",
        "title": "Senior Backend Engineer",
        "company": "Acme",
        "location": "Remote",
        "url": "https://example.test/acme",
        "description": "We need python and postgres experience.",
        "salary_min": 140_000,
        "salary_max": 190_000,
        "remote": True,
        "skills": ["python", "postgres"],
        "posted_at": NOW.isoformat(),
    }
    defaults.update(payload)
    return RawPosting(provider=provider, payload=defaults)


def scored(
    target: NormalizedPosting | None = None, score: float = 0.9, **overrides: Any
) -> ScoredPosting:
    """A scored posting carrying a component breakdown, as the contract requires."""
    return ScoredPosting(
        posting=target or posting(),
        score=score,
        components=overrides.pop("components", {"keywords": score, "salary": 1.0}),
        reasons=overrides.pop("reasons", ("matches python and postgres",)),
    )


def evidence_check(
    target: NormalizedPosting | None = None, *, grounded: bool = True
) -> EvidenceCheck:
    """A grounding verdict for a posting, with a citation when grounded."""
    target = target or posting()
    return EvidenceCheck(
        dedupe_key=target.dedupe_key,
        grounded=grounded,
        citations=(
            (
                EvidenceRef(
                    type="resume_section", ref="experience.acme", excerpt="Built a Python API"
                ),
            )
            if grounded
            else ()
        ),
        unsupported_claims=() if grounded else ("no python experience on record",),
    )


def shortlist_entry(rank: int = 1, score: float = 0.9) -> ShortlistEntry:
    """One shortlist entry, ready to hand to a packet builder."""
    target = posting()
    return ShortlistEntry(
        rank=rank, scored=scored(target, score=score), evidence=evidence_check(target)
    )


def packet(target: NormalizedPosting | None = None) -> ApplicationPacket:
    """An application packet with the resume draft the contract requires."""
    target = target or posting()
    return ApplicationPacket(
        dedupe_key=target.dedupe_key,
        posting=target,
        artifacts=(
            ArtifactDraft(
                artifact_type=ArtifactType.RESUME,
                content="Tailored resume for Acme",
                evidence=(EvidenceRef(type="resume_section", ref="experience.acme"),),
            ),
        ),
        answers={"why_us": "I like distributed systems."},
    )


def submit_intent(target: NormalizedPosting | None = None) -> ActionIntent:
    """A submission `ActionIntent`, as `prepare_application_packet` builds one."""
    target = target or posting()
    return ActionIntent(
        kind=ActionKind.SUBMIT_APPLICATION,
        target=target.url or f"{target.company} / {target.title}",
        summary=f"Submit an application to {target.company} for {target.title}",
        payload={"dedupe_key": target.dedupe_key, "company": target.company},
        idempotency_key=f"submit-{target.dedupe_key}"[:255],
    )


def approval_request(intent: ActionIntent | None = None, **overrides: Any) -> ApprovalRequest:
    """The `ApprovalRequest` `request_approval` raises for an intent.

    Minted from the intent rather than hand-written, so a test that checks a
    refusal is comparing against the hash the graph would really have recorded.
    """
    intent = intent or submit_intent()
    request = ApprovalRequest.for_intent(intent, now=NOW)
    return request.model_copy(update=overrides) if overrides else request


def approval_decision(
    request: ApprovalRequest,
    verdict: ApprovalVerdict = ApprovalVerdict.APPROVED,
    *,
    decided_by: str = "reviewer@example.test",
    **overrides: Any,
) -> ApprovalDecision:
    """A reviewer's answer to one request, correctly bound to it.

    This is the shape a caller resumes the approval interrupt with.
    """
    decision = ApprovalDecision(
        action_id=request.action_id,
        action_fingerprint=request.action_hash,
        verdict=verdict,
        decided_by=decided_by,
        request_id=request.request_id,
    )
    return decision.model_copy(update=overrides) if overrides else decision


def recruiter_message(
    provider_message_id: str = "msg-1", body: str = "Are you available for an interview?"
) -> RecruiterMessage:
    """One inbound recruiter message."""
    return RecruiterMessage(
        provider_message_id=provider_message_id,
        received_at=NOW,
        body=body,
        subject="Next steps",
        from_address="recruiter@acme.test",
    )


# --- Port fakes ---------------------------------------------------------------


class FakeProfileStore:
    """Returns a fixed profile, recording which user it was asked about."""

    def __init__(self, result: SearchProfile | None = None):
        self.result = result or profile()
        self.calls: list[UUID] = []

    async def load(self, user_id: UUID) -> SearchProfile:
        self.calls.append(user_id)
        return self.result


class FakeProvider:
    """A job board that returns a fixed list of raw postings."""

    def __init__(self, name: str = "fakeboard", results: Sequence[RawPosting] | None = None):
        self.name = name
        self.results = list(results if results is not None else [raw_posting(name)])
        self.calls: list[SearchProfile] = []

    async def search(self, search_profile: SearchProfile) -> Sequence[RawPosting]:
        self.calls.append(search_profile)
        return self.results


class FailingProvider:
    """A job board that always raises, to exercise the fan-out's failure handling."""

    def __init__(self, name: str = "deadboard", error: str = "upstream 503"):
        self.name = name
        self.error = error
        self.calls: list[SearchProfile] = []

    async def search(self, search_profile: SearchProfile) -> Sequence[RawPosting]:
        self.calls.append(search_profile)
        raise RuntimeError(self.error)


class FakePostingCatalog:
    """An in-memory posting catalog keyed on `dedupe_key`, as the table is.

    Returns the existing entry for a key it has seen, exactly as
    `personalos.persistence.job_postings.SqlPostingCatalog` does.
    """

    def __init__(self):
        self.rows: dict[str, UUID] = {}
        self.calls: list[list[NormalizedPosting]] = []

    async def record(self, postings: Sequence[NormalizedPosting]) -> list[PersistedPosting]:
        self.calls.append(list(postings))
        persisted = []
        for target in postings:
            created = target.dedupe_key not in self.rows
            row_id = self.rows.setdefault(target.dedupe_key, uuid4())
            persisted.append(
                PersistedPosting(
                    job_posting_id=row_id, dedupe_key=target.dedupe_key, created=created
                )
            )
        return persisted


class FakeScorer:
    """Scores on keyword overlap, deterministically.

    Deliberately simple arithmetic rather than a stub returning a constant: the
    ranking and shortlist tests need distinguishable scores, and a scorer that
    derives them from the posting keeps the fixtures honest about which posting
    should win.
    """

    def __init__(self):
        self.calls: list[tuple[NormalizedPosting, SearchProfile]] = []

    async def score(
        self, target: NormalizedPosting, search_profile: SearchProfile
    ) -> ScoredPosting:
        self.calls.append((target, search_profile))
        haystack = f"{target.title} {target.description} {' '.join(target.skills)}".lower()
        hits = [word for word in search_profile.keywords if word.lower() in haystack]
        overlap = len(hits) / max(len(search_profile.keywords), 1)
        return ScoredPosting(
            posting=target,
            score=round(overlap, 4),
            components={"keyword_overlap": round(overlap, 4)},
            reasons=(f"matched {len(hits)}/{len(search_profile.keywords)} keywords",),
        )


class FakeEvidenceChecker:
    """Grounds any posting whose text mentions one of the profile's must-have skills."""

    def __init__(self, *, always_grounded: bool | None = None):
        self.always_grounded = always_grounded
        self.calls: list[ScoredPosting] = []

    async def check(self, item: ScoredPosting, search_profile: SearchProfile) -> EvidenceCheck:
        self.calls.append(item)
        if self.always_grounded is not None:
            grounded = self.always_grounded
        else:
            haystack = f"{item.posting.description} {' '.join(item.posting.skills)}".lower()
            grounded = any(skill.lower() in haystack for skill in search_profile.must_have_skills)
        return EvidenceCheck(
            dedupe_key=item.posting.dedupe_key,
            grounded=grounded,
            citations=(
                (EvidenceRef(type="resume_section", ref="experience.primary"),) if grounded else ()
            ),
            unsupported_claims=() if grounded else ("nothing on record supports this match",),
        )


class FakePacketBuilder:
    """Builds a resume-bearing packet for whatever entry it is given."""

    def __init__(self):
        self.calls: list[ShortlistEntry] = []

    async def build(
        self, entry: ShortlistEntry, search_profile: SearchProfile
    ) -> ApplicationPacket:
        self.calls.append(entry)
        target = entry.scored.posting
        return ApplicationPacket(
            dedupe_key=target.dedupe_key,
            posting=target,
            artifacts=(
                ArtifactDraft(
                    artifact_type=ArtifactType.RESUME,
                    content=f"Resume tailored for {target.company}",
                    evidence=tuple(entry.evidence.citations)
                    or (EvidenceRef(type="resume_section", ref="experience.primary"),),
                ),
            ),
            answers={},
        )


class FakeApprovalGate:
    """Returns a scripted verdict, correctly bound to each intent's fingerprint.

    The binding is the point: a gate that returned an `APPROVED` verdict with a
    stale fingerprint would be rejected by `ApprovalDecision.authorizes`, which
    is what `test_approval_checkpoint_rejects_a_mismatched_fingerprint` checks.
    """

    def __init__(
        self,
        verdict: ApprovalVerdict = ApprovalVerdict.APPROVED,
        *,
        decided_by: str = "reviewer@example.test",
        fingerprint_override: str | None = None,
    ):
        self.verdict = verdict
        self.decided_by = decided_by
        self.fingerprint_override = fingerprint_override
        self.reviewed: list[ActionIntent] = []

    async def review(self, intent: ActionIntent) -> ApprovalDecision:
        self.reviewed.append(intent)
        return ApprovalDecision(
            action_id=intent.action_id,
            action_fingerprint=self.fingerprint_override or intent.fingerprint(),
            verdict=self.verdict,
            decided_by=self.decided_by,
        )


class NoStandingApprovalGate:
    """An approval gate with nothing on file, so the run interrupts.

    The same behaviour as
    `personalos.graphs.job_search.InterruptOnlyApprovalGate`, re-declared here
    so an approval-interrupt test can also see *which* intents were asked
    about. A test that wants the run to pause uses this; a test about some
    other part of the pipeline uses `FakeApprovalGate`, whose standing
    `APPROVED` keeps the run moving.
    """

    def __init__(self):
        self.reviewed: list[ActionIntent] = []

    async def review(self, intent: ActionIntent) -> ApprovalDecision:
        self.reviewed.append(intent)
        return ApprovalDecision(
            action_id=intent.action_id,
            action_fingerprint=intent.fingerprint(),
            verdict=ApprovalVerdict.PENDING,
            decided_by="system:no_standing_approval",
        )


class FakeActionExecutor:
    """Records redeemed actions and refuses anything the decision does not authorize.

    The refusal is not defensive padding: the executor is the last place an
    unapproved action could slip through, and a fake that executed anything it
    was handed would hide a graph bug rather than surface it.
    """

    def __init__(self, *, ok: bool = True):
        self.ok = ok
        self.executed: list[tuple[ActionIntent, ApprovalDecision]] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        if not decision.authorizes(intent):
            raise AssertionError(
                f"executor called with a decision that does not authorize {intent.kind.value}"
            )
        self.executed.append((intent, decision))
        return ActionReceipt(
            action_id=intent.action_id,
            ok=self.ok,
            external_reference=f"ext-{intent.kind.value}" if self.ok else None,
            detail=None if self.ok else "provider rejected the submission",
        )


class FakeApplicationStore:
    """In-memory application store that enforces the real lifecycle rules.

    `create` runs the requested status through
    `validate_application_status_transition` from DISCOVERED, so a node asking
    for an unreachable status fails here exactly as it would against the
    repository -- a fake that accepted any status would make the scenario test
    agree with a graph the database would reject.
    """

    def __init__(self, application_id: UUID = APPLICATION_ID):
        self.application_id = application_id
        self.created: list[PersistedApplication] = []
        self.recruiter_responses: list[tuple[UUID, RecruiterResponse]] = []
        self.follow_ups: list[FollowUpCheckpoint] = []

    async def create(
        self,
        *,
        packet: ApplicationPacket,
        profile: SearchProfile,
        status: ApplicationStatus,
        submitted: bool,
        receipt: ActionReceipt | None,
    ) -> PersistedApplication:
        current = ApplicationStatus.DISCOVERED
        for step in (
            ApplicationStatus.SAVED,
            ApplicationStatus.PREPARING,
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.APPLIED,
        ):
            current = validate_application_status_transition(current, step)
            if current == status:
                break

        application = PersistedApplication(
            application_id=self.application_id,
            job_posting_id=POSTING_ID,
            user_id=profile.user_id,
            dedupe_key=packet.dedupe_key,
            status=status,
            submitted=submitted,
            artifact_version_ids=tuple(uuid4() for _ in packet.artifacts),
            external_reference=receipt.external_reference if receipt else None,
        )
        self.created.append(application)
        return application

    async def record_recruiter_response(
        self, application_id: UUID, response: RecruiterResponse
    ) -> None:
        self.recruiter_responses.append((application_id, response))

    async def record_follow_up(self, checkpoint: FollowUpCheckpoint) -> None:
        self.follow_ups.append(checkpoint)


class FakePendingCheckpointScheduler:
    """An in-memory stand-in for the durable wait store.

    Deduplicates on `dedupe_key` exactly as
    `personalos.persistence.pending_checkpoints.PendingCheckpointStore` does,
    and returns the *existing* wait when one is already held. That is not
    convenience: a node test whose fake accepted a second wait for the same
    application would agree with a graph the unique constraint would reject.
    """

    def __init__(self):
        self.scheduled: dict[str, PendingCheckpoint] = {}

    async def schedule(self, checkpoint: PendingCheckpoint) -> PendingCheckpoint:
        return self.scheduled.setdefault(checkpoint.dedupe_key, checkpoint)

    def waits(self) -> list[PendingCheckpoint]:
        """Every wait scheduled, in the order it was first seen."""
        return list(self.scheduled.values())


class ScriptedConditionEvaluator:
    """Answers checkpoint conditions from a script, recording each question.

    Keyed by `ConditionKind` rather than by checkpoint id, because that is the
    only thing the evaluator is actually given -- the port exists so the
    decision can be made from the stored condition alone, and a fake that
    cheated by looking the checkpoint up would hide a condition that failed to
    round-trip.
    """

    def __init__(self, met: dict[ConditionKind, bool] | None = None, *, default: bool = False):
        self.met = dict(met or {})
        self.default = default
        self.asked: list[CheckpointCondition] = []

    async def is_met(self, condition: CheckpointCondition) -> bool:
        self.asked.append(condition)
        return self.met.get(condition.kind, self.default)


class FakeEventEmitter:
    """Collects emitted events in order."""

    def __init__(self):
        self.events: list[EmittedEvent] = []

    async def emit(self, event: EmittedEvent) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        """Emitted event type values, in emission order."""
        return [event.type.value for event in self.events]


class FakeRecruiterInbox:
    """Returns a fixed list of inbound messages for any application."""

    def __init__(self, messages: Sequence[RecruiterMessage] | None = None):
        self.messages = list(messages or ())
        self.calls: list[UUID] = []

    async def fetch(self, application_id: UUID) -> Sequence[RecruiterMessage]:
        self.calls.append(application_id)
        return self.messages


class FakeRecruiterClassifier:
    """Classifies on a keyword in the message body, deterministically."""

    def __init__(self):
        self.calls: list[RecruiterMessage] = []

    async def classify(self, message: RecruiterMessage) -> RecruiterResponse:
        self.calls.append(message)
        body = message.body.lower()
        if "interview" in body:
            classification = CommunicationEventClassification.INTERVIEW_INVITE
            implied = ApplicationStatus.INTERVIEWING
            requires_reply = True
        elif "unfortunately" in body or "not moving forward" in body:
            classification = CommunicationEventClassification.REJECTION
            implied = ApplicationStatus.REJECTED
            requires_reply = False
        else:
            classification = CommunicationEventClassification.GENERAL_UPDATE
            implied = None
            requires_reply = False
        return RecruiterResponse(
            provider_message_id=message.provider_message_id,
            classification=classification,
            occurred_at=message.received_at,
            implied_status=implied,
            requires_reply=requires_reply,
            summary=message.subject,
        )
