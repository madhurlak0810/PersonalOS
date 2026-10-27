"""A real worker process, running the real worker wiring, that can be killed.

Exists for the restart test in `tests/unit/test_worker_process.py`: the API
queues a conversation, *this* process claims it and dies part way through the
job search, and a second copy of this process -- knowing nothing but the
database -- has to finish the workflow without searching the job boards again.

It is `apps.worker.main.build_worker` over the Supervisor and the Job Search
subgraph from `tests.fixtures.durable_workflow`, whose ports log to a shared
file so the parent can see what each process did after it is gone. `kill_at`
is passed straight through to those ports: `approval` dies at the approval
review, after discovery and scoring are checkpointed.

Run as `python -m tests.fixtures.worker_process <db> <log> <lease-ttl-seconds>
[kill_at]`. Drains the command queue and exits, logging `worker.drained`.
"""

import asyncio
import os
import sys

from apps.worker.main import WorkerGraphs, WorkflowScopedJobSubgraph, build_worker
from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.domain.routing import RouteDecision, RouteDomain
from personalos.graphs.supervisor import SupervisorGraph
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import WorkflowLeaseStore
from tests.fixtures import durable_workflow as durable

#: Logged once the queue has nothing left this worker can claim.
EVENT_WORKER_DRAINED = "worker.drained"


class JobClassifier:
    """Routes every message to job search, confidently."""

    def classify(self, message: str) -> RouteDecision:
        return RouteDecision(domain=RouteDomain.JOB, confidence=0.95)


def build_graphs(factory, log, *, lease_ttl_seconds: int, kill_at: str | None = None):
    """The graphs this worker runs: a Supervisor delegating to the durable job search."""
    registry = WorkflowThreadRegistry(factory)
    subgraph, _ports = durable.build_graph(factory, log, kill_at=kill_at, registry=registry)
    supervisor = SupervisorGraph(
        JobClassifier(),
        WorkflowScopedJobSubgraph(subgraph, registry, prepare_application=True),
        checkpointer=SqlAlchemyCheckpointSaver(factory, registry),
    ).build()
    runner = DurableWorkflowRunner(
        supervisor,
        registry=registry,
        leases=WorkflowLeaseStore(factory, ttl_seconds=lease_ttl_seconds),
        owner=f"worker:{os.getpid()}",
    )
    return WorkerGraphs(runners={"supervisor": runner})


def main(argv: list[str]) -> int:
    db_path, log_path, ttl = argv[0], argv[1], int(argv[2])
    kill_at = argv[3] if len(argv) > 3 else None
    log = durable.EventLog(log_path)
    factory = durable.session_factory(db_path, create=False)
    graphs = build_graphs(factory, log, lease_ttl_seconds=ttl, kill_at=kill_at)
    worker = build_worker(factory, graphs, lease_ttl_seconds=ttl, idle_seconds=0.05)
    asyncio.run(worker.drain())
    log.append(EVENT_WORKER_DRAINED)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main(sys.argv[1:]))
