"""Hybrid, evidence-grounded scoring of postings against the candidate's record.

`HybridJobMatcher` produces one `JobMatch` per posting from two sources that
are deliberately kept apart:

- **Deterministic features** -- location, level, employment type, and how many
  of the posting's stated requirements appear in the candidate's evidence.
  Pure functions of the posting, the profile and the evidence records; the
  same inputs always give the same numbers.
- **A semantic assessment** -- a model's structured read of the posting,
  obtained through the `SemanticAssessor` port. It never contributes a number
  of its own. Its claims are first resolved against real evidence records
  (`personalos.domain.job_search.ground_assessment`), and the semantic
  component is then *computed* from what survived, so a model cannot raise a
  score by asserting experience the candidate does not have.

The final score is explainable by construction:

    weighted_score = sum(ScoringWeights[c] * components[c] for c in components)
    score          = 0.0                         if a failed constraint rejects
                   = min(weighted_score, cap)    if a failed constraint caps
                   = weighted_score              otherwise

Every weight, threshold and constraint effect is a field of `ScoringConfig`,
and every `JobMatch` carries the components, weights and constraint outcomes
it was computed from.

Posting text is data here as everywhere else: it is matched against and handed
to the assessor as a quoted value, and nothing in this module branches on what
it says beyond the token tables below.
"""

import logging
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personalos.domain.job_search import (
    ConstraintEffect,
    ConstraintResult,
    ConstraintStatus,
    EmploymentType,
    EvidenceCheck,
    EvidenceRecord,
    EvidenceRef,
    GapSeverity,
    GroundedAssessment,
    JobMatch,
    MatchedRequirement,
    MatchStrength,
    MissingRequirement,
    NormalizedPosting,
    Recommendation,
    ScoredPosting,
    SearchProfile,
    SemanticAssessment,
    SeniorityLevel,
    canonical_text,
    ground_assessment,
)

logger = logging.getLogger(__name__)

#: Component names, as they appear in `JobMatch.components` and `.weights`.
LOCATION = "location"
LEVEL = "level"
EMPLOYMENT_TYPE = "employment_type"
REQUIREMENT_COVERAGE = "requirement_coverage"
SEMANTIC = "semantic"

#: Name of the constraint raised by a BLOCKING entry in the gap analysis.
BLOCKING_REQUIREMENT = "blocking_requirement"

#: How much one matched requirement counts toward the semantic component.
STRENGTH_VALUES: dict[MatchStrength, float] = {
    MatchStrength.STRONG: 1.0,
    MatchStrength.PARTIAL: 0.6,
    MatchStrength.WEAK: 0.3,
}

#: How much one missing requirement counts against it. A minor gap dilutes the
#: component far less than a blocking one.
SEVERITY_WEIGHTS: dict[GapSeverity, float] = {
    GapSeverity.BLOCKING: 1.0,
    GapSeverity.MAJOR: 0.6,
    GapSeverity.MINOR: 0.25,
}

#: Title tokens that state a level, highest level first so "Senior Staff
#: Engineer" reads as staff.
_LEVEL_TOKENS: tuple[tuple[SeniorityLevel, frozenset[str]], ...] = (
    (SeniorityLevel.PRINCIPAL, frozenset({"principal", "distinguished", "fellow"})),
    (SeniorityLevel.STAFF, frozenset({"staff"})),
    (SeniorityLevel.SENIOR, frozenset({"senior", "sr", "lead"})),
    (SeniorityLevel.MID, frozenset({"mid", "intermediate"})),
    (SeniorityLevel.JUNIOR, frozenset({"junior", "jr", "entry", "graduate", "associate"})),
    (SeniorityLevel.INTERN, frozenset({"intern", "internship"})),
)

#: Title phrases that state an engagement type, in canonical-text form.
_EMPLOYMENT_PHRASES: tuple[tuple[EmploymentType, tuple[str, ...]], ...] = (
    (EmploymentType.INTERNSHIP, ("intern", "internship")),
    (EmploymentType.CONTRACT, ("contract", "contractor", "freelance")),
    (EmploymentType.PART_TIME, ("part time",)),
    (EmploymentType.TEMPORARY, ("temporary", "temp", "seasonal")),
    (EmploymentType.FULL_TIME, ("full time",)),
)


class ScoringWeights(BaseModel):
    """How much each component contributes to the weighted score. Sums to 1.

    The split is 35% hard-constraint fit (location, level, employment type),
    30% measured requirement overlap, 35% grounded semantic fit: the two halves
    that look at the candidate's actual record together outweigh the half that
    only looks at the posting's logistics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    location: float = Field(default=0.15, ge=0.0)
    level: float = Field(default=0.10, ge=0.0)
    employment_type: float = Field(default=0.10, ge=0.0)
    requirement_coverage: float = Field(default=0.30, ge=0.0)
    semantic: float = Field(default=0.35, ge=0.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> "ScoringWeights":
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")
        return self


class ScoringConfig(BaseModel):
    """Every tunable of the scorer, in one place."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    #: Score at or above which a posting is recommended APPLY / MAYBE.
    apply_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    maybe_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    #: Value of a feature the posting gives no information about. Halfway, so
    #: silence neither helps nor sinks a posting.
    unknown_feature_value: float = Field(default=0.5, ge=0.0, le=1.0)
    #: What failing each hard constraint does. Location and employment type
    #: reject: the candidate cannot take the job as posted. Level and a
    #: blocking gap cap: the job is takeable, just not a strong recommendation.
    on_location_mismatch: ConstraintEffect = ConstraintEffect.REJECT
    on_employment_type_mismatch: ConstraintEffect = ConstraintEffect.REJECT
    on_level_mismatch: ConstraintEffect = ConstraintEffect.CAP
    on_blocking_requirement: ConstraintEffect = ConstraintEffect.CAP
    #: Ceiling applied by a CAP effect. Below `apply_threshold`, so a capped
    #: posting can be MAYBE at best.
    capped_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> "ScoringConfig":
        if self.maybe_threshold > self.apply_threshold:
            raise ValueError("maybe_threshold cannot exceed apply_threshold")
        if self.capped_score >= self.apply_threshold:
            raise ValueError(
                "capped_score must be below apply_threshold, or a capped posting could "
                "still be recommended APPLY"
            )
        return self


class EvidenceSource(Protocol):
    """Loads the candidate's citable resume and project records."""

    async def load(self, user_id: UUID) -> Sequence[EvidenceRecord]:
        """Return every evidence record on file for this user."""
        ...


class SemanticAssessor(Protocol):
    """Produces a model's structured assessment of one posting.

    The returned assessment is untrusted: the matcher grounds it before use.
    """

    async def assess(
        self,
        posting: NormalizedPosting,
        profile: SearchProfile,
        evidence: Sequence[EvidenceRecord],
    ) -> SemanticAssessment:
        """Return the claimed matches, gaps, risks and tailoring suggestions."""
        ...


# --- Deterministic features --------------------------------------------------


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-token containment over `canonical_text` strings."""
    return bool(phrase) and f" {phrase} " in f" {haystack} "


def infer_seniority(title: str) -> SeniorityLevel | None:
    """The level a posting title states, or `None` if it states none."""
    tokens = set(canonical_text(title).split())
    for level, markers in _LEVEL_TOKENS:
        if tokens & markers:
            return level
    return None


def infer_employment_type(title: str) -> EmploymentType | None:
    """The engagement type a posting title states, or `None` if it states none."""
    text = canonical_text(title)
    for kind, phrases in _EMPLOYMENT_PHRASES:
        if any(_contains_phrase(text, phrase) for phrase in phrases):
            return kind
    return None


def _result(
    name: str,
    status: ConstraintStatus,
    detail: str,
    config: ScoringConfig,
    on_failure: ConstraintEffect = ConstraintEffect.NONE,
) -> ConstraintResult:
    effect = on_failure if status == ConstraintStatus.FAILED else ConstraintEffect.NONE
    return ConstraintResult(
        name=name,
        status=status,
        effect=effect,
        cap=config.capped_score if effect == ConstraintEffect.CAP else None,
        detail=detail,
    )


def location_feature(
    posting: NormalizedPosting, profile: SearchProfile, config: ScoringConfig
) -> tuple[float, ConstraintResult]:
    """Score the posting's location against where the candidate will work."""
    fail = config.on_location_mismatch
    targets = [text for text in map(canonical_text, profile.target_locations) if text]
    location = canonical_text(posting.location or "")
    remote = posting.remote or _contains_phrase(location, "remote")

    if profile.remote_only and not remote:
        detail = "profile requires remote and the posting is not remote"
        return 0.0, _result(LOCATION, ConstraintStatus.FAILED, detail, config, fail)
    if not targets:
        return 1.0, _result(LOCATION, ConstraintStatus.PASSED, "no location preference", config)
    if remote:
        return 1.0, _result(LOCATION, ConstraintStatus.PASSED, "posting is remote", config)
    if not location:
        detail = "posting does not state a location"
        return config.unknown_feature_value, _result(
            LOCATION, ConstraintStatus.UNKNOWN, detail, config
        )
    for target in targets:
        if _contains_phrase(location, target) or _contains_phrase(target, location):
            detail = f"'{posting.location}' is a target location"
            return 1.0, _result(LOCATION, ConstraintStatus.PASSED, detail, config)
    detail = f"'{posting.location}' is not a target location and the posting is not remote"
    return 0.0, _result(LOCATION, ConstraintStatus.FAILED, detail, config, fail)


def level_feature(
    posting: NormalizedPosting, profile: SearchProfile, config: ScoringConfig
) -> tuple[float, ConstraintResult]:
    """Score the posting's level against the levels the candidate accepts.

    One step off the ladder (a senior candidate, a staff posting) is a stretch,
    not a violation: half credit and no constraint failure. Two or more fails.
    """
    if not profile.seniority_levels:
        return 1.0, _result(LEVEL, ConstraintStatus.PASSED, "no level preference", config)
    level = infer_seniority(posting.title)
    if level is None:
        detail = "posting title does not state a level"
        return config.unknown_feature_value, _result(
            LEVEL, ConstraintStatus.UNKNOWN, detail, config
        )
    distance = min(abs(level.rank - wanted.rank) for wanted in profile.seniority_levels)
    if distance == 0:
        detail = f"posting is {level.value}, an accepted level"
        return 1.0, _result(LEVEL, ConstraintStatus.PASSED, detail, config)
    if distance == 1:
        detail = f"posting is {level.value}, one step from an accepted level"
        return 0.5, _result(LEVEL, ConstraintStatus.PASSED, detail, config)
    detail = f"posting is {level.value}, {distance} steps from any accepted level"
    return 0.0, _result(LEVEL, ConstraintStatus.FAILED, detail, config, config.on_level_mismatch)


def employment_type_feature(
    posting: NormalizedPosting, profile: SearchProfile, config: ScoringConfig
) -> tuple[float, ConstraintResult]:
    """Score the posting's engagement type against what the candidate accepts."""
    name = EMPLOYMENT_TYPE
    if not profile.employment_types:
        detail = "no employment type preference"
        return 1.0, _result(name, ConstraintStatus.PASSED, detail, config)
    kind = infer_employment_type(posting.title)
    if kind is None:
        detail = "posting title does not state an employment type"
        return config.unknown_feature_value, _result(name, ConstraintStatus.UNKNOWN, detail, config)
    if kind in profile.employment_types:
        detail = f"posting is {kind.value}, an accepted employment type"
        return 1.0, _result(name, ConstraintStatus.PASSED, detail, config)
    detail = f"posting is {kind.value}, which the profile does not accept"
    return 0.0, _result(
        name, ConstraintStatus.FAILED, detail, config, config.on_employment_type_mismatch
    )


def requirement_coverage(
    posting: NormalizedPosting, evidence: Sequence[EvidenceRecord]
) -> tuple[list[tuple[str, EvidenceRecord]], list[str]]:
    """Split the posting's stated requirements by whether evidence mentions them.

    Requirements are `posting.skills`, compared as whole canonical phrases
    against each record's text. Returns `(covered, uncovered)`, each in the
    posting's own order; a covered requirement comes with the first record
    that mentions it.
    """
    texts = [(record, canonical_text(record.text)) for record in evidence]
    covered: list[tuple[str, EvidenceRecord]] = []
    uncovered: list[str] = []
    seen: set[str] = set()
    for skill in posting.skills:
        phrase = canonical_text(skill)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        record = next((rec for rec, text in texts if _contains_phrase(text, phrase)), None)
        if record is None:
            uncovered.append(skill)
        else:
            covered.append((skill, record))
    return covered, uncovered


def semantic_value(assessment: GroundedAssessment) -> float:
    """The semantic component, computed from the grounded assessment.

    Strength-weighted matches over everything assessed, where each gap counts
    by its severity. Nothing assessed is 0.0: no evidence of fit is not fit.
    """
    earned = sum(STRENGTH_VALUES[item.strength] for item in assessment.matched_requirements)
    owed = len(assessment.matched_requirements) + sum(
        SEVERITY_WEIGHTS[item.severity] for item in assessment.missing_requirements
    )
    return earned / owed if owed else 0.0


# --- Matcher -----------------------------------------------------------------


class HybridJobMatcher:
    """Scores postings and grounds the result in the candidate's record.

    Satisfies both the `CandidateScorer` and `EvidenceChecker` ports of
    `personalos.graphs.job_search.JobSearchGraph`, so one instance is passed as
    both: `check` reads the match `score` already attached and makes no second
    model call.
    """

    def __init__(
        self,
        *,
        evidence_source: EvidenceSource,
        assessor: SemanticAssessor,
        config: ScoringConfig | None = None,
    ):
        """Bind the evidence source and the semantic assessor."""
        if evidence_source is None or assessor is None:
            raise ValueError("HybridJobMatcher requires an evidence_source and an assessor")
        self.evidence_source = evidence_source
        self.assessor = assessor
        self.config = config or ScoringConfig()

    async def match(self, posting: NormalizedPosting, profile: SearchProfile) -> JobMatch:
        """Return the scored, explained, evidence-grounded verdict on one posting."""
        config = self.config
        evidence = tuple(await self.evidence_source.load(profile.user_id))

        location, location_check = location_feature(posting, profile, config)
        level, level_check = level_feature(posting, profile, config)
        employment, employment_check = employment_type_feature(posting, profile, config)
        covered, uncovered = requirement_coverage(posting, evidence)
        stated = len(covered) + len(uncovered)
        coverage = len(covered) / stated if stated else config.unknown_feature_value

        grounded = await self._assess(posting, profile, evidence, covered, uncovered)

        blocking = [
            gap.requirement
            for gap in grounded.missing_requirements
            if gap.severity == GapSeverity.BLOCKING
        ]
        blocking_check = _result(
            BLOCKING_REQUIREMENT,
            ConstraintStatus.FAILED if blocking else ConstraintStatus.PASSED,
            (
                f"no evidence for blocking requirement(s): {', '.join(blocking)}"
                if blocking
                else "no blocking gaps"
            ),
            config,
            config.on_blocking_requirement,
        )
        constraints = (location_check, level_check, employment_check, blocking_check)

        components = {
            LOCATION: round(location, 4),
            LEVEL: round(level, 4),
            EMPLOYMENT_TYPE: round(employment, 4),
            REQUIREMENT_COVERAGE: round(coverage, 4),
            SEMANTIC: round(semantic_value(grounded), 4),
        }
        weights = config.weights.model_dump()
        weighted = round(sum(weights[name] * value for name, value in components.items()), 4)

        caps = [item.cap for item in constraints if item.cap is not None]
        rejected = any(item.effect == ConstraintEffect.REJECT for item in constraints)
        score = 0.0 if rejected else min([weighted, *caps])

        if rejected:
            recommendation = Recommendation.SKIP
        elif score >= config.apply_threshold:
            recommendation = Recommendation.APPLY
        elif score >= config.maybe_threshold:
            recommendation = Recommendation.MAYBE
        else:
            recommendation = Recommendation.SKIP

        reasons = [
            f"{name}: {value:.2f} x weight {weights[name]:.2f}"
            for name, value in components.items()
        ]
        reasons.append(f"weighted score {weighted:.4f}")
        reasons.extend(
            f"{item.name} {item.status.value} ({item.effect.value}): {item.detail}"
            for item in constraints
            if item.status != ConstraintStatus.PASSED
        )
        if stated:
            reasons.append(f"{len(covered)}/{stated} stated requirements found in evidence")

        return JobMatch(
            dedupe_key=posting.dedupe_key,
            score=score,
            recommendation=recommendation,
            matched_requirements=grounded.matched_requirements,
            missing_requirements=grounded.missing_requirements,
            risks=grounded.risks,
            tailoring_suggestions=grounded.tailoring_suggestions,
            weighted_score=weighted,
            components=components,
            weights=weights,
            constraints=constraints,
            ungrounded_claims=grounded.ungrounded_claims,
            reasons=tuple(reasons),
        )

    async def score(self, posting: NormalizedPosting, profile: SearchProfile) -> ScoredPosting:
        """`CandidateScorer`: the posting with its score and the match behind it."""
        match = await self.match(posting, profile)
        return ScoredPosting(
            posting=posting,
            score=match.score,
            components=match.components,
            reasons=match.reasons,
            match=match,
        )

    async def check(self, scored: ScoredPosting, profile: SearchProfile) -> EvidenceCheck:
        """`EvidenceChecker`: grounded when the match cites at least one record."""
        match = scored.match
        if match is None:
            return EvidenceCheck(
                dedupe_key=scored.posting.dedupe_key,
                grounded=False,
                unsupported_claims=("posting was scored without an evidence-grounded match",),
            )
        citations = tuple(
            EvidenceRef(type=item.evidence_type.value, ref=item.evidence_id)
            for item in match.matched_requirements
        )
        return EvidenceCheck(
            dedupe_key=match.dedupe_key,
            grounded=bool(citations),
            citations=citations,
            unsupported_claims=match.ungrounded_claims,
        )

    async def _assess(
        self,
        posting: NormalizedPosting,
        profile: SearchProfile,
        evidence: Sequence[EvidenceRecord],
        covered: Sequence[tuple[str, EvidenceRecord]],
        uncovered: Sequence[str],
    ) -> GroundedAssessment:
        """Ground the assessor's output, or fall back to keyword-level matches.

        A failing assessor costs the posting its semantic nuance, not its
        place in the run: the deterministic overlap stands in, marked as such.
        """
        try:
            assessment = await self.assessor.assess(posting, profile, evidence)
        except Exception:
            logger.warning(
                "semantic assessment failed for posting '%s'; using keyword-level matches",
                posting.dedupe_key,
                exc_info=True,
            )
            return GroundedAssessment(
                matched_requirements=tuple(
                    MatchedRequirement(
                        requirement=skill,
                        evidence_id=record.evidence_id,
                        evidence_type=record.source_type,
                        strength=MatchStrength.PARTIAL,
                        rationale="keyword match only",
                    )
                    for skill, record in covered
                ),
                missing_requirements=tuple(
                    MissingRequirement(requirement=skill, severity=GapSeverity.MAJOR)
                    for skill in uncovered
                ),
                risks=("semantic assessment unavailable; matches are keyword-level only",),
            )
        return ground_assessment(assessment, evidence)


__all__ = [
    "LOCATION",
    "LEVEL",
    "EMPLOYMENT_TYPE",
    "REQUIREMENT_COVERAGE",
    "SEMANTIC",
    "BLOCKING_REQUIREMENT",
    "STRENGTH_VALUES",
    "SEVERITY_WEIGHTS",
    "ScoringWeights",
    "ScoringConfig",
    "EvidenceSource",
    "SemanticAssessor",
    "infer_seniority",
    "infer_employment_type",
    "location_feature",
    "level_feature",
    "employment_type_feature",
    "requirement_coverage",
    "semantic_value",
    "HybridJobMatcher",
]
