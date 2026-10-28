"""Loading the golden dataset under `evals/golden/` into domain values.

The JSON files are the source of truth and are meant to be read and edited by
hand; this module only turns their entries into the `SearchProfile`s,
`RawPosting`s and `RecruiterMessage`s a graph takes, and fingerprints the set
so a results file records exactly which version of it was run.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from personalos.domain.job_search import (
    EvidenceRecord,
    NormalizedPosting,
    RawPosting,
    RecruiterMessage,
    SearchProfile,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.recruiter_events import ApplicationCandidate

from . import world

GOLDEN_DIR = Path(__file__).parent / "golden"

#: The files a run reads. Hashed together into `dataset_sha256`.
GOLDEN_FILES = (
    "routing_requests.json",
    "recruiter_emails.json",
    "candidate_profiles.json",
    "job_descriptions.json",
    "scenarios.json",
    "adversarial.json",
    "human_review.json",
)


def load(name: str) -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def dataset_sha256() -> str:
    digest = hashlib.sha256()
    for name in GOLDEN_FILES:
        digest.update(name.encode())
        digest.update((GOLDEN_DIR / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Candidate:
    name: str
    profile: SearchProfile
    evidence: tuple[EvidenceRecord, ...]


@cache
def candidates() -> dict[str, Candidate]:
    return {
        name: Candidate(
            name=name,
            profile=SearchProfile(user_id=world.USER_ID, **entry["profile"]),
            evidence=tuple(EvidenceRecord(**record) for record in entry["evidence"]),
        )
        for name, entry in load("candidate_profiles.json")["candidates"].items()
    }


@cache
def job_descriptions() -> dict[str, dict[str, Any]]:
    """Golden postings by id, in file order."""
    return {entry["id"]: entry for entry in load("job_descriptions.json")["postings"]}


def raw_posting(entry: dict[str, Any], posted_at: str | None = None) -> RawPosting:
    """A provider payload for one golden posting entry."""
    posted_at = posted_at or load("job_descriptions.json")["posted_at"]
    payload = {
        "id": entry["id"],
        "url": f"https://jobs.example/{entry['id']}",
        "posted_at": posted_at,
        **entry["payload"],
    }
    return RawPosting(provider="golden", payload=payload)


def raw_postings(selection: str | list[str], extra: list[dict[str, Any]] = ()) -> list[RawPosting]:
    """`"all"` or a list of golden posting ids, plus any inline entries."""
    golden = job_descriptions()
    ids = list(golden) if selection == "all" else list(selection)
    return [raw_posting(golden[posting_id]) for posting_id in ids] + [
        raw_posting(entry) for entry in extra
    ]


def normalized_posting(payload: dict[str, Any], posting_id: str) -> NormalizedPosting:
    """The normalized form of a golden payload, for calling a draft writer directly."""
    return NormalizedPosting(
        source="golden",
        source_job_id=posting_id,
        url=f"https://jobs.example/{posting_id}",
        title=payload["title"],
        company=payload["company"],
        location=payload.get("location"),
        description=payload.get("description", ""),
        remote=payload.get("remote", False),
        salary_min=payload.get("salary_min"),
        salary_max=payload.get("salary_max"),
        skills=tuple(payload.get("skills", ())),
    )


def application(entry: dict[str, Any]) -> ApplicationCandidate:
    return ApplicationCandidate(
        application_id=world.APPLICATION_ID,
        company=entry["company"],
        title=entry["title"],
        status=ApplicationStatus(entry["status"]),
        reference=entry.get("reference"),
        thread_ids=tuple(entry.get("thread_ids", ())),
        contact_addresses=tuple(entry.get("contact_addresses", ())),
    )


def message(case_id: str, entry: dict[str, Any], received_at: str) -> RecruiterMessage:
    return RecruiterMessage(
        provider_message_id=f"<{case_id}@golden.example>",
        received_at=datetime.fromisoformat(received_at),
        subject=entry.get("subject"),
        from_address=entry.get("from_address"),
        body=entry.get("body", ""),
        thread_id=entry.get("thread_id"),
    )


__all__ = [
    "GOLDEN_DIR",
    "GOLDEN_FILES",
    "load",
    "dataset_sha256",
    "Candidate",
    "candidates",
    "job_descriptions",
    "raw_posting",
    "raw_postings",
    "normalized_posting",
    "application",
    "message",
]
