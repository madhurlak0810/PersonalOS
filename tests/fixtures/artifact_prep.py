"""A fixed candidate record, embedder and scripted writer for artifact prep.

Nothing here calls a model. `HashingEmbedder` turns text into a bag-of-words
vector, so "nearest evidence" means "shares the most words" and a test can say
which chunk should win. `ScriptedDraftWriter` replays the proposal a model
"returned", so what a test asserts about validation is a pure function of the
proposals below.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from personalos.domain.artifacts import DraftProposal, DraftSegment
from personalos.domain.job_search import EvidenceRecord, NormalizedPosting, SearchProfile
from personalos.domain.models import ArtifactType
from personalos.persistence.models import UserModel
from personalos.persistence.repositories import EvidenceChunkRepository
from tests.fixtures import job_search_fakes as fakes

OTHER_USER_ID = UUID("44444444-4444-4444-4444-444444444444")

ACME = "experience.acme"
INFRA = "experience.infra"
PGSYNC = "projects.pgsync"
GARDEN = "volunteering.garden"

#: The candidate's record, keyed by `source_ref`: (source_type, text).
CHUNKS: dict[str, tuple[str, str]] = {
    ACME: (
        "resume",
        "Senior Software Engineer at Acme Corp, 2019-2023. Built Python APIs backed by "
        "Postgres serving 1,200 requests per second; cut p95 latency by 40%.",
    ),
    INFRA: (
        "resume",
        "Operated Kafka pipelines and Kubernetes clusters in production at Globex, 2016-2019.",
    ),
    PGSYNC: (
        "project",
        "Open-source project pgsync: a Python tool that replicates Postgres tables.",
    ),
    GARDEN: (
        "resume",
        "Volunteer treasurer for a community garden; managed its annual budget.",
    ),
}

#: Somebody else's record, close to the posting, which must never be selected.
OTHER_USERS_CHUNK = "Built Python APIs backed by Postgres at Hooli for a decade."


class HashingEmbedder:
    """A deterministic bag-of-words embedding: shared words, similar vectors."""

    def __init__(self, model: str = "fake-embed-v1", dim: int = 256):
        self.model = model
        self.dim = dim
        self.calls: list[str] = []

    def vector(self, text: str) -> list[float]:
        counts = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % self.dim
            counts[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in counts)) or 1.0
        return [value / norm for value in counts]

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector(text)


class ScriptedDraftWriter:
    """Returns the proposal scripted for each artifact type, recording the ask."""

    def __init__(self, proposals: dict[ArtifactType, DraftProposal]):
        self.proposals = proposals
        self.calls: list[tuple[ArtifactType, tuple[str, ...]]] = []

    async def write(
        self,
        artifact_type: ArtifactType,
        posting: NormalizedPosting,
        profile: SearchProfile,
        evidence: Sequence[EvidenceRecord],
    ) -> DraftProposal:
        self.calls.append((artifact_type, tuple(record.evidence_id for record in evidence)))
        return self.proposals[artifact_type]


def seed_evidence(
    factory: Callable[[], Any], embedder: HashingEmbedder, user_id: UUID = fakes.USER_ID
) -> dict[str, str]:
    """Store the candidate's chunks (and one stranger's); return ids by `source_ref`."""
    session = factory()
    try:
        session.add(UserModel(id=user_id, email="candidate@example.test"))
        session.add(UserModel(id=OTHER_USER_ID, email="stranger@example.test"))
        session.commit()
        chunks = EvidenceChunkRepository(session)
        ids = {
            ref: str(
                chunks.create(
                    user_id=user_id,
                    source_type=source_type,
                    source_ref=ref,
                    chunk_text=text,
                    embedding=embedder.vector(text),
                    embedding_model=embedder.model,
                ).id
            )
            for ref, (source_type, text) in CHUNKS.items()
        }
        chunks.create(
            user_id=OTHER_USER_ID,
            source_type="resume",
            source_ref="experience.hooli",
            chunk_text=OTHER_USERS_CHUNK,
            embedding=embedder.vector(OTHER_USERS_CHUNK),
            embedding_model=embedder.model,
        )
        return ids
    finally:
        session.close()


def records(ids: dict[str, str] | None = None) -> list[EvidenceRecord]:
    """The candidate's chunks as `EvidenceRecord`s, ids defaulting to the `source_ref`."""
    return [
        EvidenceRecord(
            evidence_id=(ids or {}).get(ref, ref),
            source_type=source_type,
            source_ref=ref,
            text=text,
        )
        for ref, (source_type, text) in CHUNKS.items()
    ]


def posting(**overrides: Any) -> NormalizedPosting:
    """The posting drafts are tailored to: a different company from any on record."""
    return fakes.posting(**{"company": "Initech", **overrides})


def _proposal(artifact_type: ArtifactType, *segments: tuple[str, tuple[str, ...]]):
    return DraftProposal(
        artifact_type=artifact_type,
        segments=tuple(DraftSegment(text=text, evidence_ids=ids) for text, ids in segments),
    )


def resume_proposal(ids: dict[str, str] | None = None) -> DraftProposal:
    """A resume that rephrases the record and adds nothing to it."""
    ids = ids or {ref: ref for ref in CHUNKS}
    return _proposal(
        ArtifactType.RESUME,
        ("Senior Software Engineer, Acme Corp (2019-2023)", (ids[ACME],)),
        (
            "Designed and shipped Python APIs on Postgres handling 1,200 requests per "
            "second, reducing p95 latency by 40%.",
            (ids[ACME],),
        ),
        (
            "Author of pgsync, an open-source Python tool for replicating Postgres tables.",
            (ids[PGSYNC],),
        ),
    )


def cover_letter_proposal(ids: dict[str, str] | None = None) -> DraftProposal:
    """A cover letter: uncited framing around one cited claim."""
    ids = ids or {ref: ref for ref in CHUNKS}
    return _proposal(
        ArtifactType.COVER_LETTER,
        ("Dear Hiring Manager,", ()),
        ("I'm writing to apply for the Senior Backend Engineer role at Initech.", ()),
        (
            "At Acme Corp I built Python APIs backed by Postgres that served 1,200 "
            "requests per second.",
            (ids[ACME],),
        ),
        ("Thank you for your time.", ()),
    )


def fabricated_resume_proposal(ids: dict[str, str] | None = None) -> DraftProposal:
    """The honest resume plus one bullet the record does not support."""
    ids = ids or {ref: ref for ref in CHUNKS}
    honest = resume_proposal(ids)
    invented = DraftSegment(
        text="Led a team of 12 engineers migrating services to Kubernetes at Google.",
        evidence_ids=(ids[ACME],),
    )
    return honest.model_copy(update={"segments": (*honest.segments, invented)})


__all__ = [
    "OTHER_USER_ID",
    "ACME",
    "INFRA",
    "PGSYNC",
    "GARDEN",
    "CHUNKS",
    "OTHER_USERS_CHUNK",
    "HashingEmbedder",
    "ScriptedDraftWriter",
    "seed_evidence",
    "records",
    "posting",
    "resume_proposal",
    "cover_letter_proposal",
    "fabricated_resume_proposal",
]
