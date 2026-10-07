"""Artifact prep: pick the evidence for a posting, draft from it, check the draft.

`TailoredPacketBuilder` satisfies the `ApplicationPacketBuilder` port of
`personalos.graphs.job_search.JobSearchGraph`. For one shortlisted posting it

1. **selects evidence** -- embeds the posting and takes the candidate's nearest
   resume and project chunks (`EvidenceSelector`, over the pgvector-backed
   `EvidenceIndex`);
2. **drafts** each artifact through the `DraftWriter` port, which is handed
   those records and nothing else about the candidate;
3. **validates** every draft with `personalos.domain.artifacts.finalize_draft`,
   against the same records the writer saw. A draft with an untraceable claim
   raises `UntraceableClaim` and no packet is produced;
4. **stores** the drafts through the `DraftSink` port, which returns them
   carrying their `artifact_versions` ids.

It stops there. The packet it returns is unsubmitted, and this module has no
way to submit it: sending, submitting and overwriting are `ActionIntent`s the
graph routes through its approval interrupt.

The evidence set is fixed at step 1 and is the only thing a citation can
resolve against, so the writer cannot widen its own support by citing a record
it was not shown.
"""

import logging
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from personalos.domain.artifacts import DraftProposal, RetrievedEvidence, finalize_draft
from personalos.domain.job_search import (
    ApplicationPacket,
    ArtifactDraft,
    EvidenceRecord,
    JobSearchContractError,
    NormalizedPosting,
    SearchProfile,
    ShortlistEntry,
)
from personalos.domain.models import ArtifactType

logger = logging.getLogger(__name__)

#: Evidence chunks handed to the writer for one posting.
DEFAULT_TOP_K = 8

#: Cap on posting description characters in the retrieval query, so one padded
#: posting cannot drown its own title and skill list.
MAX_QUERY_DESCRIPTION_CHARS = 4000

#: The documents a packet is built with unless the caller asks otherwise.
DEFAULT_ARTIFACT_TYPES: tuple[ArtifactType, ...] = (ArtifactType.RESUME, ArtifactType.COVER_LETTER)


class Embedder(Protocol):
    """Turns text into a vector in one named embedding space."""

    #: The model the vectors come from. Must match `evidence_chunks.embedding_model`
    #: for the chunks to be comparable at all.
    model: str

    async def embed(self, text: str) -> list[float]:
        """Return the embedding of `text`."""
        ...


class EvidenceIndex(Protocol):
    """Nearest-neighbour search over one candidate's evidence chunks."""

    async def search(
        self,
        *,
        user_id: UUID,
        query_embedding: list[float],
        embedding_model: str,
        top_k: int,
        min_similarity: float,
    ) -> Sequence[RetrievedEvidence]:
        """Return the closest chunks, most similar first."""
        ...


class DraftWriter(Protocol):
    """Writes one tailored document from a fixed set of evidence records.

    Its output is a `DraftProposal`: claims, each citing the records it
    restates. An implementation backed by a model must treat the posting as
    data and may not be given any other source of facts about the candidate.
    """

    async def write(
        self,
        artifact_type: ArtifactType,
        posting: NormalizedPosting,
        profile: SearchProfile,
        evidence: Sequence[EvidenceRecord],
    ) -> DraftProposal:
        """Return the proposed draft of `artifact_type` for this posting."""
        ...


class DraftSink(Protocol):
    """Stores validated drafts and returns them with their row ids."""

    async def record(
        self,
        *,
        user_id: UUID,
        posting: NormalizedPosting,
        drafts: Sequence[ArtifactDraft],
        generated_by: str,
    ) -> Sequence[ArtifactDraft]:
        """Persist the drafts as artifact versions."""
        ...


def retrieval_query(posting: NormalizedPosting) -> str:
    """The text a posting is embedded as when looking for evidence."""
    parts = [
        posting.title,
        ", ".join(posting.skills),
        posting.description[:MAX_QUERY_DESCRIPTION_CHARS],
    ]
    return "\n".join(part for part in parts if part)


class EvidenceSelector:
    """Selects the resume and project evidence relevant to one posting."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        index: EvidenceIndex,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = 0.0,
    ):
        """Bind the embedder and the index it queries."""
        if embedder is None or index is None:
            raise ValueError("EvidenceSelector requires an embedder and an index")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.embedder = embedder
        self.index = index
        self.top_k = top_k
        self.min_similarity = min_similarity

    async def select(self, posting: NormalizedPosting, user_id: UUID) -> list[RetrievedEvidence]:
        """The candidate's evidence nearest this posting, most similar first."""
        embedding = await self.embedder.embed(retrieval_query(posting))
        return list(
            await self.index.search(
                user_id=user_id,
                query_embedding=embedding,
                embedding_model=self.embedder.model,
                top_k=self.top_k,
                min_similarity=self.min_similarity,
            )
        )


class TailoredPacketBuilder:
    """Builds an evidence-grounded, validated, stored application packet."""

    def __init__(
        self,
        *,
        selector: EvidenceSelector,
        writer: DraftWriter,
        sink: DraftSink | None = None,
        artifact_types: Sequence[ArtifactType] = DEFAULT_ARTIFACT_TYPES,
        context_terms: Sequence[str] = (),
        generated_by: str = "artifact_prep",
    ):
        """Wire the selector, writer and (optionally) the draft sink.

        `context_terms` are names every draft may use without citing evidence,
        beyond the posting's own company, title and location -- the candidate's
        name, typically. Without a `sink` the drafts are returned unstored.
        """
        if selector is None or writer is None:
            raise ValueError("TailoredPacketBuilder requires a selector and a writer")
        if ArtifactType.RESUME not in artifact_types:
            raise ValueError("an application packet must include a resume draft")
        self.selector = selector
        self.writer = writer
        self.sink = sink
        self.artifact_types = tuple(dict.fromkeys(artifact_types))
        self.context_terms = tuple(context_terms)
        self.generated_by = generated_by

    async def build(self, entry: ShortlistEntry, profile: SearchProfile) -> ApplicationPacket:
        """`ApplicationPacketBuilder`: the tailored, unsubmitted packet for this entry."""
        posting = entry.scored.posting
        retrieved = await self.selector.select(posting, profile.user_id)
        evidence = [item.record for item in retrieved]
        if not evidence:
            raise JobSearchContractError(
                f"no resume or project evidence was retrieved for posting "
                f"'{posting.dedupe_key}'; a draft with nothing to cite cannot be written"
            )

        context_terms = [
            posting.company,
            posting.title,
            posting.location or "",
            *self.context_terms,
        ]
        protected_terms = [*posting.skills, *profile.must_have_skills, *profile.keywords]

        drafts: list[ArtifactDraft] = []
        for artifact_type in self.artifact_types:
            proposal = await self.writer.write(artifact_type, posting, profile, evidence)
            if proposal.artifact_type != artifact_type:
                raise JobSearchContractError(
                    f"asked for a {artifact_type.value} draft and was given a "
                    f"{proposal.artifact_type.value}"
                )
            drafts.append(
                finalize_draft(
                    proposal,
                    evidence,
                    context_terms=context_terms,
                    protected_terms=protected_terms,
                )
            )

        if self.sink is not None:
            drafts = list(
                await self.sink.record(
                    user_id=profile.user_id,
                    posting=posting,
                    drafts=drafts,
                    generated_by=self.generated_by,
                )
            )
        logger.info(
            "prepared %s draft(s) for posting '%s' from %s evidence record(s)",
            len(drafts),
            posting.dedupe_key,
            len(evidence),
        )
        return ApplicationPacket(dedupe_key=posting.dedupe_key, posting=posting, artifacts=drafts)


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_QUERY_DESCRIPTION_CHARS",
    "DEFAULT_ARTIFACT_TYPES",
    "Embedder",
    "EvidenceIndex",
    "DraftWriter",
    "DraftSink",
    "retrieval_query",
    "EvidenceSelector",
    "TailoredPacketBuilder",
]
