"""Contract tests for the match and gap-analysis values.

The point of these models is what they refuse to hold: a matched requirement
with no evidence behind it, and a score that survived a failed hard constraint.
"""

import pytest
from pydantic import ValidationError

from personalos.domain.job_search import (
    ClaimedMatch,
    ConstraintEffect,
    ConstraintResult,
    ConstraintStatus,
    GapSeverity,
    JobMatch,
    JobSearchContractError,
    MatchStrength,
    MissingRequirement,
    Recommendation,
    ScoredPosting,
    SemanticAssessment,
    TailoringSuggestion,
    ground_assessment,
)
from personalos.domain.models import EvidenceSourceType
from tests.fixtures import job_matching as fx
from tests.fixtures import job_search_fakes as fakes


def match(**overrides) -> dict:
    """Keyword arguments for a valid, unconstrained `JobMatch`."""
    values = {
        "dedupe_key": fakes.posting().dedupe_key,
        "score": 0.8,
        "weighted_score": 0.8,
        "recommendation": Recommendation.APPLY,
        "components": {"semantic": 0.8},
        "weights": {"semantic": 1.0},
    }
    values.update(overrides)
    return values


def failed(effect: ConstraintEffect, cap: float | None = None) -> ConstraintResult:
    return ConstraintResult(
        name="location",
        status=ConstraintStatus.FAILED,
        effect=effect,
        cap=cap,
        detail="not a target location",
    )


class TestMatchedRequirementsMustCiteEvidence:
    def test_a_job_match_with_an_uncited_matched_requirement_fails_validation(self):
        with pytest.raises(ValidationError) as excinfo:
            JobMatch(
                **match(
                    matched_requirements=[
                        {
                            "requirement": "python",
                            "strength": "strong",
                            "evidence_type": "resume",
                        }
                    ]
                )
            )

        assert "evidence_id" in str(excinfo.value)

    @pytest.mark.parametrize("evidence_id", [None, "", "   "])
    def test_a_null_or_blank_evidence_id_fails_validation(self, evidence_id):
        with pytest.raises(ValidationError):
            JobMatch(
                **match(
                    matched_requirements=[
                        {
                            "requirement": "python",
                            "strength": "strong",
                            "evidence_type": "resume",
                            "evidence_id": evidence_id,
                        }
                    ]
                )
            )

    def test_a_cited_matched_requirement_is_accepted(self):
        built = JobMatch(
            **match(
                matched_requirements=[
                    {
                        "requirement": "python",
                        "strength": "strong",
                        "evidence_type": "resume",
                        "evidence_id": fx.EV_ACME,
                    }
                ]
            )
        )

        assert built.matched_requirements[0].evidence_id == fx.EV_ACME
        built.ensure_grounded_in({record.evidence_id for record in fx.EVIDENCE})

    def test_a_citation_that_is_not_on_record_is_caught(self):
        built = JobMatch(
            **match(
                matched_requirements=[
                    {
                        "requirement": "rust",
                        "strength": "strong",
                        "evidence_type": "resume",
                        "evidence_id": "ev-invented",
                    }
                ]
            )
        )

        with pytest.raises(JobSearchContractError, match="ev-invented"):
            built.ensure_grounded_in({record.evidence_id for record in fx.EVIDENCE})


class TestGroundAssessment:
    def test_cited_claims_become_matches_typed_from_the_record(self):
        grounded = ground_assessment(
            SemanticAssessment(
                matched_requirements=(
                    ClaimedMatch(
                        requirement="search", evidence_id=fx.EV_SEARCH, strength=MatchStrength.WEAK
                    ),
                )
            ),
            fx.EVIDENCE,
        )

        (matched,) = grounded.matched_requirements
        assert matched.evidence_id == fx.EV_SEARCH
        assert matched.evidence_type == EvidenceSourceType.PROJECT
        assert grounded.ungrounded_claims == ()

    @pytest.mark.parametrize("evidence_id", [None, "ev-invented"])
    def test_a_claim_with_no_real_citation_becomes_a_gap(self, evidence_id):
        grounded = ground_assessment(
            SemanticAssessment(
                matched_requirements=(
                    ClaimedMatch(requirement="rust", evidence_id=evidence_id),
                )
            ),
            fx.EVIDENCE,
        )

        assert grounded.matched_requirements == ()
        assert grounded.ungrounded_claims == ("rust",)
        assert [(gap.requirement, gap.severity) for gap in grounded.missing_requirements] == [
            ("rust", GapSeverity.MAJOR)
        ]

    def test_with_no_evidence_on_file_nothing_can_match(self):
        grounded = ground_assessment(fx.GLOWING, ())

        assert grounded.matched_requirements == ()
        assert len(grounded.missing_requirements) == 3
        assert grounded.tailoring_suggestions == ()

    def test_a_suggestion_citing_unknown_evidence_is_dropped(self):
        grounded = ground_assessment(
            SemanticAssessment(
                missing_requirements=(
                    MissingRequirement(requirement="go", severity=GapSeverity.MINOR),
                ),
                tailoring_suggestions=(
                    TailoringSuggestion(suggestion="Mention the Go rewrite.", evidence_id="nope"),
                    TailoringSuggestion(suggestion="Address the Go gap up front."),
                    TailoringSuggestion(suggestion="Lead with Acme.", evidence_id=fx.EV_ACME),
                ),
            ),
            fx.EVIDENCE,
        )

        assert [item.suggestion for item in grounded.tailoring_suggestions] == [
            "Address the Go gap up front.",
            "Lead with Acme.",
        ]


class TestConstraintsBindTheScore:
    def test_a_rejected_match_cannot_carry_a_score(self):
        with pytest.raises(ValidationError, match="rejected by a hard constraint"):
            JobMatch(**match(constraints=(failed(ConstraintEffect.REJECT),)))

    def test_a_rejected_match_must_be_skip(self):
        with pytest.raises(ValidationError):
            JobMatch(
                **match(
                    score=0.0,
                    recommendation=Recommendation.MAYBE,
                    constraints=(failed(ConstraintEffect.REJECT),),
                )
            )

    def test_a_capped_match_cannot_exceed_its_cap(self):
        with pytest.raises(ValidationError, match="exceeds the cap"):
            JobMatch(**match(constraints=(failed(ConstraintEffect.CAP, cap=0.5),)))

    def test_a_score_cannot_exceed_the_weighted_sum(self):
        with pytest.raises(ValidationError):
            JobMatch(**match(score=0.9, weighted_score=0.8))

    def test_an_effect_requires_a_failure(self):
        with pytest.raises(ValidationError):
            ConstraintResult(
                name="location",
                status=ConstraintStatus.PASSED,
                effect=ConstraintEffect.REJECT,
                detail="",
            )

    def test_a_cap_effect_requires_a_cap(self):
        with pytest.raises(ValidationError):
            failed(ConstraintEffect.CAP)


class TestScoredPostingCarriesItsMatch:
    def test_a_match_for_another_posting_is_refused(self):
        with pytest.raises(ValidationError):
            ScoredPosting(
                posting=fakes.posting(),
                score=0.8,
                components={"semantic": 0.8},
                match=JobMatch(**match(dedupe_key="someone|else|abc")),
            )

    def test_a_match_with_a_different_score_is_refused(self):
        with pytest.raises(ValidationError):
            ScoredPosting(
                posting=fakes.posting(),
                score=0.6,
                components={"semantic": 0.8},
                match=JobMatch(**match()),
            )

    def test_the_match_round_trips_through_json_state(self):
        scored = ScoredPosting(
            posting=fakes.posting(),
            score=0.8,
            components={"semantic": 0.8},
            match=JobMatch(**match()),
        )

        assert ScoredPosting.model_validate(scored.model_dump(mode="json")) == scored
