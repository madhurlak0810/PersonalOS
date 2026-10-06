"""A fixed synthetic candidate, posting set and scripted model for job matching.

Everything the golden ranking test depends on lives here: one profile, three
evidence records, seven postings chosen so each exercises a different path
through `HybridJobMatcher` (a clean match, a partial one, a stretch level, a
rejected location, a capped level, a rejected employment type, and a model
that cites evidence that does not exist), and the assessment a model "returned"
for each. `ScriptedAssessor` replays those assessments, so a run makes no model
call and its output is a pure function of this file.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from personalos.domain.job_search import (
    ClaimedMatch,
    EmploymentType,
    EvidenceRecord,
    GapSeverity,
    MatchStrength,
    MissingRequirement,
    NormalizedPosting,
    RawPosting,
    SearchProfile,
    SemanticAssessment,
    SeniorityLevel,
    TailoringSuggestion,
)
from personalos.domain.models import EvidenceSourceType
from tests.fixtures import job_search_fakes as fakes

GOLDEN_RANKING = Path(__file__).parent / "golden" / "job_match_ranking.json"

EV_ACME = "ev-resume-acme"
EV_INFRA = "ev-resume-infra"
EV_SEARCH = "ev-project-search"

EVIDENCE: tuple[EvidenceRecord, ...] = (
    EvidenceRecord(
        evidence_id=EV_ACME,
        source_type=EvidenceSourceType.RESUME,
        source_ref="experience.acme",
        text="Built Python APIs backed by Postgres at Acme for four years.",
    ),
    EvidenceRecord(
        evidence_id=EV_INFRA,
        source_type=EvidenceSourceType.RESUME,
        source_ref="experience.infra",
        text="Operated Kafka pipelines and Kubernetes clusters in production.",
    ),
    EvidenceRecord(
        evidence_id=EV_SEARCH,
        source_type=EvidenceSourceType.PROJECT,
        source_ref="projects.search",
        text="Open-source search service written in Python with Elasticsearch.",
    ),
)


def profile(**overrides: Any) -> SearchProfile:
    """A senior, full-time candidate open to remote or New York."""
    defaults: dict[str, Any] = {
        "target_locations": ("Remote", "New York"),
        "seniority_levels": (SeniorityLevel.SENIOR,),
        "employment_types": (EmploymentType.FULL_TIME,),
        "remote_only": False,
        "salary_min": None,
        "excluded_companies": (),
        "min_score": 0.45,
        "max_shortlist": 5,
    }
    defaults.update(overrides)
    return fakes.profile(**defaults)


def _strong(requirement: str, evidence_id: str | None) -> ClaimedMatch:
    return ClaimedMatch(
        requirement=requirement, evidence_id=evidence_id, strength=MatchStrength.STRONG
    )


#: A model answer that could not be more favourable, fully cited. Reused for
#: the postings that must lose anyway on a hard constraint.
GLOWING = SemanticAssessment(
    matched_requirements=(
        _strong("python", EV_ACME),
        _strong("postgres", EV_ACME),
        _strong("kafka", EV_INFRA),
    ),
    tailoring_suggestions=(
        TailoringSuggestion(suggestion="Lead with the Acme API work.", evidence_id=EV_ACME),
    ),
)

#: Provider payload overrides and the scripted assessment, keyed by company.
CASES: dict[str, tuple[dict[str, Any], SemanticAssessment]] = {
    "Northwind": (
        {"title": "Senior Backend Engineer", "skills": ["python", "postgres", "kafka"]},
        GLOWING,
    ),
    "Globex": (
        {
            "title": "Senior Platform Engineer",
            "location": "New York, NY",
            "remote": False,
            "skills": ["python", "kubernetes", "go"],
        },
        SemanticAssessment(
            matched_requirements=(
                _strong("python", EV_ACME),
                ClaimedMatch(
                    requirement="kubernetes", evidence_id=EV_INFRA, strength=MatchStrength.PARTIAL
                ),
            ),
            missing_requirements=(
                MissingRequirement(requirement="go", severity=GapSeverity.MAJOR),
            ),
            risks=("No production Go on record.",),
        ),
    ),
    "Initech": (
        {"title": "Staff Data Engineer", "skills": ["python", "spark", "airflow"]},
        SemanticAssessment(
            matched_requirements=(
                ClaimedMatch(
                    requirement="python", evidence_id=EV_SEARCH, strength=MatchStrength.PARTIAL
                ),
            ),
            missing_requirements=(
                MissingRequirement(requirement="spark", severity=GapSeverity.MAJOR),
                MissingRequirement(requirement="airflow", severity=GapSeverity.MINOR),
            ),
        ),
    ),
    # Fails location: onsite in a city the candidate did not list.
    "Hooli": (
        {
            "title": "Senior Backend Engineer",
            "location": "Berlin",
            "remote": False,
            "skills": ["python", "postgres", "kafka"],
        },
        GLOWING,
    ),
    # Fails level by two steps: capped, not rejected.
    "Umbrella": (
        {"title": "Junior Backend Engineer", "skills": ["python", "postgres", "kafka"]},
        GLOWING,
    ),
    # Fails employment type.
    "Vandelay": (
        {"title": "Senior Backend Engineer (Contract)", "skills": ["python", "postgres", "kafka"]},
        GLOWING,
    ),
    # The model credits PyTorch to a record that does not exist.
    "Stark": (
        {"title": "Senior ML Engineer", "skills": ["python", "pytorch"]},
        SemanticAssessment(
            matched_requirements=(
                _strong("python", EV_ACME),
                _strong("pytorch", "ev-resume-does-not-exist"),
            ),
        ),
    ),
}


def raw_postings() -> list[RawPosting]:
    """One provider payload per case, in declaration order."""
    return [
        fakes.raw_posting(
            id=f"gold-{company.lower()}",
            company=company,
            url=f"https://example.test/{company.lower()}",
            description=f"{company} is hiring. We use python.",
            **payload,
        )
        for company, (payload, _) in CASES.items()
    ]


def posting(company: str, **overrides: Any) -> NormalizedPosting:
    """The normalized posting for one case."""
    payload = {
        "description": f"{company} is hiring. We use python.",
        **CASES[company][0],
        **overrides,
    }
    payload["skills"] = tuple(payload.get("skills", ()))
    return fakes.posting(company=company, **payload)


def golden_ranking() -> dict[str, Any]:
    """The checked-in expected ranking and shortlist."""
    return json.loads(GOLDEN_RANKING.read_text())


class StaticEvidenceSource:
    """An `EvidenceSource` over a fixed tuple of records."""

    def __init__(self, records: Sequence[EvidenceRecord] = EVIDENCE):
        self.records = tuple(records)

    async def load(self, user_id) -> Sequence[EvidenceRecord]:
        return self.records


class ScriptedAssessor:
    """A `SemanticAssessor` that replays a fixed assessment per company."""

    def __init__(self, script: dict[str, SemanticAssessment] | None = None):
        self.script = script or {company: answer for company, (_, answer) in CASES.items()}
        self.calls: list[str] = []

    async def assess(self, target, search_profile, evidence) -> SemanticAssessment:
        self.calls.append(target.company)
        return self.script[target.company]


class FailingAssessor:
    """A `SemanticAssessor` whose model call always fails."""

    async def assess(self, target, search_profile, evidence) -> SemanticAssessment:
        raise RuntimeError("model unavailable")
