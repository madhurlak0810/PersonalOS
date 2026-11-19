"""A File subgraph over the real stack, on a temporary directory.

Nothing is faked between the graph and the disk: the graph's `FileOperations`
port is `GatewayFileOperations`, dispatching through the policy-enforcing
gateway (the default rule chain) to an in-process `FilesMCPServer` whose
sandbox is confined to `root`. A test reads outcomes off the disk, the
server's denial log and the policy engine's decision sink.
"""

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp_servers.files.sandbox import PathDenial, PathSandbox
from mcp_servers.files.server import FilesMCPServer
from personalos.domain.files import FileApprovalPolicy
from personalos.domain.job_search import ApprovalDecision, ApprovalVerdict
from personalos.executor.file_operations import GatewayFileOperations
from personalos.graphs.file_task import FileTaskGraph
from personalos.mcp.adapter import MCPToolInvoker
from personalos.mcp.manager import MCPServerManager
from personalos.models.file_planning import ResilientFilePlanner
from personalos.persistence.idempotency import InMemoryOperationStore, OperationStore
from personalos.policy import PolicyDecision, PolicyEngine, ToolIntent, default_rules
from personalos.tools.gateway import PolicyEnforcingToolGateway

NOW = datetime(2026, 11, 19, 12, 0)

RESUME_V1 = "# Ada Lovelace\n\nResume, 2025 edition.\n\n- Analytical Engine programs\n"
RESUME_V2 = "# Ada Lovelace\n\nResume, 2026 edition.\n\n- Python, Postgres\n- Staff engineer\n"
COVER_LETTER = "# Cover letter: Initech\n\nDear hiring team,\n\nI would like to apply.\n"


def write(path: Path, text: str, *, age_days: int = 0) -> Path:
    """Write `text` at `path` with an mtime `age_days` before `NOW`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    stamp = (NOW.replace(tzinfo=UTC) - timedelta(days=age_days)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def seed_documents(root: Path) -> dict[str, Path]:
    """Two resumes (the 2026 one newest), a cover letter, and a non-document."""
    return {
        "old_resume": write(root / "resume-2025.md", RESUME_V1, age_days=300),
        "resume": write(root / "resume-2026.md", RESUME_V2, age_days=2),
        "cover_letter": write(root / "cover-letter-initech.md", COVER_LETTER, age_days=10),
        "notes": write(root / "notes.json", "{}", age_days=1),
    }


@dataclass
class FileWorld:
    """Everything a File subgraph test reads its outcome from."""

    root: Path
    server: FilesMCPServer
    gateway: PolicyEnforcingToolGateway
    policy: PolicyEngine
    files: GatewayFileOperations
    graph: FileTaskGraph
    denials: list[PathDenial] = field(default_factory=list)
    decisions: list[tuple[ToolIntent, PolicyDecision]] = field(default_factory=list)

    def compile(self):
        return self.graph.build()


def build_world(
    root: Path,
    *,
    operation_store: OperationStore | None = None,
    planner: ResilientFilePlanner | None = None,
    summarizer: Any = None,
    files: Any = None,
    max_unattended: int = 3,
    checkpointer: Any = None,
) -> FileWorld:
    """The File subgraph over a real files server confined to `root`."""
    denials: list[PathDenial] = []
    decisions: list[tuple[ToolIntent, PolicyDecision]] = []
    server = FilesMCPServer(
        PathSandbox([root]),
        operation_store=operation_store or InMemoryOperationStore(),
        denial_sink=denials.append,
    )
    manager = MCPServerManager()
    manager.register_server(server)
    policy = PolicyEngine(
        default_rules(max_unattended_files=max_unattended),
        decision_sink=lambda intent, decision: decisions.append((intent, decision)),
    )
    gateway = PolicyEnforcingToolGateway(policy=policy, invoker=MCPToolInvoker(manager))
    operations = GatewayFileOperations(gateway)
    graph = FileTaskGraph(
        files if files is not None else operations,
        planner=planner,
        summarizer=summarizer,
        approval_policy=FileApprovalPolicy(max_files_without_approval=max_unattended),
        clock=lambda: NOW,
        checkpointer=checkpointer,
    )
    return FileWorld(root, server, gateway, policy, operations, graph, denials, decisions)


def thread(name: str = "file") -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{name}-{uuid4()}"}}


def approve(request: dict[str, Any], *, decided_by: str = "ada") -> ApprovalDecision:
    """A person's approval of the `ApprovalRequest` in an interrupt payload."""
    return ApprovalDecision(
        action_id=request["action_id"],
        action_fingerprint=request["action_hash"],
        verdict=ApprovalVerdict.APPROVED,
        decided_by=decided_by,
        decided_at=NOW,
        request_id=request["request_id"],
    )


def interrupt_requests(final: dict[str, Any]) -> list[dict[str, Any]]:
    """The approval requests a run parked on."""
    (pending,) = final["__interrupt__"]
    return pending.value["requests"]
