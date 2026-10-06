"""Shared wiring for the job discovery tests.

Builds the Job Search graph the way a deployment does for its discovery half
-- provider adapters behind the policy gateway, postings recorded in a real
`job_postings` table -- with fakes for the ports discovery does not touch.
"""

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.bootstrap import build_job_providers
from personalos.graphs.job_search import JobSearchGraph
from personalos.persistence.job_postings import SqlPostingCatalog
from personalos.persistence.models import Base, JobPostingModel
from personalos.policy import PolicyDecision, ToolIntent, default_policy_engine
from tests.fixtures import job_search_fakes as fakes


def posting_payload(**overrides: Any) -> dict[str, Any]:
    """A provider payload that clears the default profile's hard filters."""
    return fakes.raw_posting(**overrides).payload


def session_factory(tmp_path):
    """A session factory over a fresh file-backed SQLite database."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{tmp_path / 'discovery.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def stored_postings(factory) -> list[JobPostingModel]:
    """Every `job_postings` row, oldest first."""
    session = factory()
    try:
        return session.query(JobPostingModel).order_by(JobPostingModel.created_at).all()
    finally:
        session.close()


def build_discovery_graph(adapters, factory, **overrides):
    """Compile the graph over gateway-backed providers and a SQL catalog.

    Returns the graph, its ports, and the list every policy decision made
    during the run is appended to.
    """
    decisions: list[tuple[ToolIntent, PolicyDecision]] = []
    policy = default_policy_engine(decision_sink=lambda i, d: decisions.append((i, d)))
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": build_job_providers(adapters, policy=policy),
        "posting_catalog": SqlPostingCatalog(factory),
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": fakes.FakeApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
    }
    ports.update(overrides)
    return JobSearchGraph(**ports).build(), ports, decisions
