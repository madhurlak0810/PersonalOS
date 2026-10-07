"""The integrity rule for tailored drafts: rephrase what is supported, invent nothing.

`validate_draft` is pure, so every case here is a proposal, the evidence it was
written from, and the flags that come back. The acceptance criterion this file
exists for is `test_a_draft_with_a_claim_not_traceable_to_evidence_is_rejected`.
"""

import pytest

from personalos.domain.artifacts import (
    ClaimFlagReason,
    DraftProposal,
    DraftSegment,
    UntraceableClaim,
    finalize_draft,
    validate_draft,
)
from personalos.domain.errors import ErrorCode
from personalos.domain.models import ArtifactType
from tests.fixtures import artifact_prep as prep

EVIDENCE = prep.records()
CONTEXT = ("Initech", "Senior Backend Engineer", "Remote")


def _draft(*segments: tuple[str, tuple[str, ...]]) -> DraftProposal:
    return DraftProposal(
        artifact_type=ArtifactType.RESUME,
        segments=tuple(DraftSegment(text=text, evidence_ids=ids) for text, ids in segments),
    )


def _flags(proposal: DraftProposal, **kwargs):
    return validate_draft(proposal, EVIDENCE, context_terms=CONTEXT, **kwargs).flags


# --- The acceptance criterion ------------------------------------------------


def test_a_draft_with_a_claim_not_traceable_to_evidence_is_rejected():
    """One invented bullet among honest ones sinks the draft, and says which."""
    proposal = prep.fabricated_resume_proposal()

    with pytest.raises(UntraceableClaim) as excinfo:
        finalize_draft(proposal, EVIDENCE, context_terms=CONTEXT)

    (flag,) = excinfo.value.validation.flags
    assert flag.reason == ClaimFlagReason.UNSUPPORTED_FACT
    assert flag.segment_index == len(proposal.segments) - 1
    # The team size, the tool and the employer are each named as untraceable.
    assert set(flag.terms) == {"12", "Kubernetes", "Google"}
    assert excinfo.value.code == ErrorCode.VALIDATION
    assert excinfo.value.details["flags"][0]["reason"] == "unsupported_fact"


def test_a_draft_that_only_rephrases_its_evidence_passes_and_links_to_it():
    draft = finalize_draft(prep.resume_proposal(), EVIDENCE, context_terms=CONTEXT)

    assert draft.artifact_type == ArtifactType.RESUME
    assert "1,200 requests per second" in draft.content
    # Linked to the evidence used, in first-cited order, typed from the record.
    assert [(ref.type, ref.ref) for ref in draft.evidence] == [
        ("resume", prep.ACME),
        ("project", prep.PGSYNC),
    ]
    assert draft.evidence[0].excerpt.startswith("Senior Software Engineer at Acme Corp")


def test_a_cover_letter_may_frame_its_claims_without_citing_the_framing():
    """Greeting, the role applied for and a sign-off carry no fact to trace."""
    draft = finalize_draft(prep.cover_letter_proposal(), EVIDENCE, context_terms=CONTEXT)

    assert draft.content.startswith("Dear Hiring Manager,")
    assert [ref.ref for ref in draft.evidence] == [prep.ACME]


# --- Each thing the rule forbids inventing -----------------------------------


@pytest.mark.parametrize(
    ("text", "untraceable"),
    [
        ("Staff Engineer at Hooli before joining Acme Corp.", {"Hooli"}),  # employer
        ("Cut p95 latency by 65% for Python APIs.", {"65"}),  # metric
        ("Built Python APIs backed by Postgres and DynamoDB.", {"DynamoDB"}),  # tool
        ("Senior Software Engineer at Acme Corp since 2015.", {"2015"}),  # date
        ("Built Python APIs serving 12,000 requests per second.", {"12,000"}),  # magnitude
        ("Shipped the Atlas platform on Python and Postgres.", {"Atlas"}),  # project
    ],
)
def test_an_invented_fact_in_a_cited_claim_is_flagged(text, untraceable):
    (flag,) = _flags(_draft((text, (prep.ACME,))))

    assert flag.reason == ClaimFlagReason.UNSUPPORTED_FACT
    assert set(flag.terms) == untraceable


def test_a_fact_must_come_from_the_records_the_claim_cites_not_from_any_record():
    """Kubernetes is on the candidate's record -- at Globex, not at Acme."""
    (flag,) = _flags(_draft(("Ran Kubernetes clusters at Acme Corp.", (prep.ACME,))))

    assert flag.terms == ("Kubernetes",)
    assert _flags(_draft(("Ran Kubernetes clusters at Globex.", (prep.INFRA,)))) == ()


def test_an_invented_degree_is_flagged_even_in_lower_case_and_uncited():
    (flag,) = _flags(_draft(("holds a masters degree in computer science.", ())))

    assert flag.reason == ClaimFlagReason.UNCITED
    assert {"masters", "degree"} <= set(flag.terms)


def test_a_claim_that_names_a_fact_and_cites_nothing_is_flagged():
    """True or not, a substantive claim has to point at its evidence."""
    (flag,) = _flags(_draft(("Built Python APIs at Acme Corp.", ())))

    assert flag.reason == ClaimFlagReason.UNCITED
    assert {"Python", "APIs", "Acme", "Corp"} <= set(flag.terms)


def test_a_posting_skill_cannot_be_acquired_by_mentioning_it():
    """The commonest tailoring lie: echoing a required tool the record lacks."""
    proposal = _draft(("Built APIs with a strong focus on terraform automation.", (prep.ACME,)))

    assert _flags(proposal) == ()  # lower-case and unlisted: not checkable
    (flag,) = _flags(proposal, protected_terms=["terraform", "python"])
    assert flag.terms == ("terraform",)


def test_a_citation_that_does_not_resolve_is_flagged():
    (flag,) = _flags(_draft(("Built Python APIs.", ("ev-does-not-exist",))))

    assert flag.reason == ClaimFlagReason.UNKNOWN_EVIDENCE
    assert flag.terms == ("ev-does-not-exist",)


def test_a_draft_that_cites_nothing_at_all_is_flagged():
    (flag,) = _flags(_draft(("Dear Hiring Manager,", ()), ("Thank you for your time.", ())))

    assert flag.reason == ClaimFlagReason.NO_EVIDENCE


def test_validation_reports_every_flag_and_is_deterministic():
    proposal = _draft(
        ("Built Python APIs at Hooli.", (prep.ACME,)),
        ("Certified in Rust.", ()),
        ("Author of pgsync.", (prep.PGSYNC,)),
    )

    first = validate_draft(proposal, EVIDENCE, context_terms=CONTEXT)

    assert first == validate_draft(proposal, EVIDENCE, context_terms=CONTEXT)
    assert not first.ok
    assert [flag.segment_index for flag in first.flags] == [0, 1]
    assert first.cited_evidence_ids == (prep.ACME, prep.PGSYNC)
