"""Worker graphs that run a job search end to end without a network or a model.

The local quick start (`IMPLEMENTATION_GUIDE.md`) and its smoke test
(`scripts/smoke_quickstart.py`) name this factory in `WORKER_GRAPHS`. It is
`apps.worker.graphs.build_worker_graphs` with two ports swapped:

- `providers`: a `FakeJobProvider` serving `SAMPLE_POSTINGS`, still reached
  through the tool gateway, so every search is a recorded policy decision;
- the draft writer inside `packet_builder`: `ExtractiveDraftWriter`, which
  restates the selected evidence verbatim, so the packet step completes
  without `LLM_GATEWAY_CONFIG` and every draft passes the grounding check.

Everything else -- policy, persistence, checkpoints, approvals, the outbox --
is the real wiring, which is the point: a search started through the API
reaches a real approval interrupt. The ports the default factory leaves
unbound stay unbound here, so an approved `submit_application` fails as a
retryable `PortNotConfigured` naming `application_submitter`.
"""

from collections.abc import Callable, Sequence
from typing import Any

from apps.worker.graphs import build_worker_graphs as build_default_worker_graphs
from apps.worker.main import WorkerGraphs
from apps.worker.unconfigured import NotConfiguredEmbedder
from personalos.bootstrap import (
    build_artifact_packet_builder,
    build_embedder,
    build_job_providers,
    build_policy_engine,
)
from personalos.domain.artifacts import ArtifactType, DraftProposal, DraftSegment
from personalos.domain.job_search import EvidenceRecord, NormalizedPosting, SearchProfile
from personalos.persistence.leases import DEFAULT_LEASE_TTL_SECONDS
from personalos.providers.fake import FakeJobProvider

#: The fake board's name, as it appears on postings and policy decisions.
FAKE_BOARD = "fakeboard"

#: What the fake board serves on every search. The first matches the quick
#: start's sample profile (`docs/quickstart/profile.json`); the second is
#: on-site, so the remote-only profile filters it out.
SAMPLE_POSTINGS: tuple[dict[str, Any], ...] = (
    {
        "id": "fb-backend-1",
        "title": "Senior Backend Engineer",
        "company": "Example Robotics",
        "location": "Remote",
        "url": "https://jobs.example.test/example-robotics/backend",
        "description": (
            "Build Python services on Postgres for our fleet platform. "
            "Experience with FastAPI and Docker is a plus."
        ),
        "salary_min": 150_000,
        "salary_max": 190_000,
        "remote": True,
        "skills": ["python", "postgres", "fastapi"],
    },
    {
        "id": "fb-frontend-1",
        "title": "Frontend Engineer",
        "company": "Example Media",
        "location": "Berlin",
        "url": "https://jobs.example.test/example-media/frontend",
        "description": "Build React interfaces for our newsroom tools.",
        "salary_min": 90_000,
        "salary_max": 120_000,
        "remote": False,
        "skills": ["react", "typescript"],
    },
)


class ExtractiveDraftWriter:
    """A `DraftWriter` that writes each document as the evidence it was given.

    One segment per record, its text verbatim and citing that record alone,
    so the draft states nothing the record does not. Not a writer anyone
    would send, and not meant to be: it is what lets the packet step run
    without a model.
    """

    async def write(
        self,
        artifact_type: ArtifactType,
        posting: NormalizedPosting,
        profile: SearchProfile,
        evidence: Sequence[EvidenceRecord],
    ) -> DraftProposal:
        return DraftProposal(
            artifact_type=artifact_type,
            segments=tuple(
                DraftSegment(text=record.text, evidence_ids=(record.evidence_id,))
                for record in evidence
            ),
        )


def build_worker_graphs(
    session_factory: Callable[[], Any],
    *,
    lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> WorkerGraphs:
    """The default worker graphs, searching the fake board and drafting extractively."""
    policy = build_policy_engine(session_factory)
    embedder: Any = build_embedder() or NotConfiguredEmbedder()
    return build_default_worker_graphs(
        session_factory,
        overrides={
            "providers": build_job_providers(
                [FakeJobProvider(FAKE_BOARD, SAMPLE_POSTINGS)], policy=policy
            ),
            "packet_builder": build_artifact_packet_builder(
                ExtractiveDraftWriter(), embedder, session_factory, policy=policy
            ),
        },
        lease_ttl_seconds=lease_ttl_seconds,
    )


__all__ = ["FAKE_BOARD", "SAMPLE_POSTINGS", "ExtractiveDraftWriter", "build_worker_graphs"]
