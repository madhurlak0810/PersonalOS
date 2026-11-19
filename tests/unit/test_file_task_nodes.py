"""Per-node input/output contract tests for the File subgraph.

One class per registered node, each calling the node on a graph whose node
methods check their output against `personalos.graphs.contracts`. The graph
runs over the real stack (`tests.fixtures.file_task`): policy gateway, MCP
invoker and a files server confined to a temporary directory.

Whole-graph behaviour -- traversal, approval, crash replay, prompt injection,
routing -- is in `tests/adversarial/test_file_task_boundaries.py`.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from personalos.domain.files import (
    FileOp,
    FilePlan,
    FileReceipt,
    FileStep,
    StepOutcome,
    step_idempotency_key,
)
from personalos.domain.job_search import ActionKind, ApprovalDecision, ApprovalVerdict
from personalos.graphs import file_task as fsg
from personalos.graphs.contracts import FILE_TASK_GRAPH, contracts_for
from personalos.models.file_planning import ResilientFilePlanner
from tests.fixtures import file_task as ft
from tests.fixtures.node_contracts import covers_node, enforce_contracts

CONTRACTS = contracts_for(FILE_TASK_GRAPH)


class StaticPlanner:
    """A primary planner that always proposes one fixed plan."""

    def __init__(self, plan: FilePlan):
        self.plan_value = plan
        self.calls = 0

    async def plan(self, request, inventory, *, feedback=None):
        self.calls += 1
        return self.plan_value


@pytest.fixture
def root(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    ft.seed_documents(docs)
    return docs


def graph(root: Path, **kwargs):
    return enforce_contracts(ft.build_world(root, **kwargs).graph, FILE_TASK_GRAPH)


async def discovered(root: Path, **kwargs):
    g = graph(root, **kwargs)
    return g, await g.discover_files({})


async def planned(root: Path, request: str, **kwargs):
    g, state = await discovered(root, **kwargs)
    state = {"request": request, **state}
    state.update(await g.plan_file_task(state))
    return g, state


@covers_node(FILE_TASK_GRAPH, fsg.DISCOVER_FILES)
class TestDiscoverFiles:
    async def test_lists_documents_with_their_hashes_and_kinds_but_no_content(self, root):
        _, update = await discovered(root)

        inventory = update["inventory"]
        names = sorted(entry["name"] for entry in inventory["entries"])
        assert names == ["cover-letter-initech.md", "resume-2025.md", "resume-2026.md"]
        assert inventory["roots"] == [str(root.resolve())]
        kinds = {entry["name"]: entry["category"] for entry in inventory["entries"]}
        assert kinds["resume-2026.md"] == "resume"
        assert kinds["cover-letter-initech.md"] == "cover_letter"
        assert all(len(entry["sha256"]) == 64 for entry in inventory["entries"])
        assert "Ada Lovelace" not in str(inventory)

    async def test_hidden_files_and_denied_directories_are_not_listed(self, root):
        ft.write(root / ".ssh" / "resume.md", "secret")
        ft.write(root / ".hidden-resume.md", "secret")

        _, update = await discovered(root)

        assert all(".ssh" not in e["path"] for e in update["inventory"]["entries"])
        assert all(".hidden" not in e["path"] for e in update["inventory"]["entries"])


@covers_node(FILE_TASK_GRAPH, fsg.PLAN_FILE_TASK)
class TestPlanFileTask:
    async def test_find_my_latest_resume_locates_the_newest_resume_only(self, root):
        _, state = await planned(root, "find my latest resume")

        plan = FilePlan.model_validate(state["plan"])
        assert [(s.op, Path(s.path).name) for s in plan.steps] == [
            (FileOp.LOCATE, "resume-2026.md")
        ]
        assert state["pending_actions"] == []
        assert "approval_stage" not in state

    async def test_a_plan_over_the_limit_is_proposed_as_one_action_listing_every_path(self, root):
        _, state = await planned(root, "organize my documents", max_unattended=1)

        (action,) = state["pending_actions"]
        assert action["kind"] == ActionKind.APPLY_FILE_PLAN.value
        plan = FilePlan.model_validate(state["plan"])
        assert action["payload"]["plan_hash"] == plan.plan_hash()
        assert set(action["payload"]["affected_paths"]) == set(plan.affected_paths())
        assert all(path in action["payload"]["body"] for path in plan.affected_paths())
        assert state["approval_stage"] == fsg.STAGE_FILE_PLAN
        assert "changes 3 files" in state["approval_reasons"][0]

    async def test_a_model_plan_naming_an_unknown_file_is_repaired_then_falls_back(self, root):
        bad = FilePlan(
            goal="read",
            steps=(FileStep(step_id="s1", op=FileOp.READ, path=str(root / "nope.md")),),
        )
        primary = StaticPlanner(bad)

        _, state = await planned(
            root, "find my latest resume", planner=ResilientFilePlanner(primary)
        )

        assert primary.calls == 2  # the first answer and one repair
        assert state["plan_source"] == "fallback"
        assert any("did not find" in error for error in state["plan_errors"])

    async def test_a_request_nothing_matches_plans_nothing(self, root):
        _, state = await planned(root, "delete every file")

        assert state["plan"] is None
        assert state["plan_errors"]

    def test_its_input_contract_requires_an_inventory(self):
        with pytest.raises(ValidationError):
            CONTRACTS[fsg.PLAN_FILE_TASK].validate_input({"request": "find my resume"})


@covers_node(FILE_TASK_GRAPH, fsg.READ_DOCUMENTS)
class TestReadDocuments:
    async def test_summarize_reads_through_the_gateway_and_keeps_only_the_summary(self, root):
        g, state = await planned(root, "summarize my cover letter")

        update = await g.read_documents(state)

        (receipt,) = [FileReceipt.model_validate(r) for r in update["read_receipts"]]
        assert receipt.outcome == StepOutcome.APPLIED
        assert receipt.summary is not None
        assert receipt.summary.title == "Cover letter: Initech"
        assert receipt.summary.path == receipt.path
        assert receipt.excerpt is None

    async def test_a_plan_with_only_mutations_reads_nothing(self, root):
        g, state = await planned(root, "organize my documents")

        assert await g.read_documents(state) == {"read_receipts": []}

    async def test_no_plan_reads_nothing(self, root):
        g = graph(root)

        assert await g.read_documents({}) == {"read_receipts": []}


@covers_node(FILE_TASK_GRAPH, fsg.REQUEST_FILE_APPROVAL)
class TestRequestFileApproval:
    async def test_mints_a_request_pinning_the_actions_hash(self, root):
        g, state = await planned(root, "organize my documents", max_unattended=1)

        update = g.request_file_approval(state)

        (request,) = update["approval_requests"]
        (action,) = state["pending_actions"]
        assert request["action_id"] == action["action_id"]
        assert request["kind"] == ActionKind.APPLY_FILE_PLAN.value
        assert request["risk"] == "high"
        assert request["payload"]["affected_paths"] == action["payload"]["affected_paths"]

    def test_its_input_contract_requires_a_pending_action(self):
        with pytest.raises(ValidationError):
            CONTRACTS[fsg.REQUEST_FILE_APPROVAL].validate_input({"pending_actions": []})


@covers_node(FILE_TASK_GRAPH, fsg.FILE_APPROVAL_CHECKPOINT)
class TestFileApprovalCheckpoint:
    async def test_nothing_pending_is_a_no_op(self, root):
        assert graph(root).file_approval_checkpoint({}) == {}

    async def test_a_pending_action_without_a_request_is_a_contract_error(self, root):
        g, state = await planned(root, "organize my documents", max_unattended=1)

        with pytest.raises(ValueError, match="no request on file"):
            g.file_approval_checkpoint(state)


@covers_node(FILE_TASK_GRAPH, fsg.APPLY_FILE_CHANGES)
class TestApplyFileChanges:
    async def test_an_unattended_plan_moves_with_hash_keys_and_undo_records(self, root):
        g, state = await planned(root, "organize my resumes", max_unattended=3)
        assert state["pending_actions"] == []

        update = await g.apply_file_changes(state)

        receipts = [FileReceipt.model_validate(r) for r in update["change_receipts"]]
        plan = FilePlan.model_validate(state["plan"])
        assert [r.outcome for r in receipts] == [StepOutcome.APPLIED, StepOutcome.APPLIED]
        for step, receipt in zip(plan.mutating_steps, receipts, strict=True):
            assert receipt.idempotency_key == step_idempotency_key(plan, step)
            assert receipt.undo is not None
            assert receipt.undo.path == step.destination
            assert receipt.undo.destination == step.path
        assert (root / "Resumes" / "resume-2026.md").read_text() == ft.RESUME_V2

    async def test_a_rejected_plan_skips_every_mutation(self, root):
        g, state = await planned(root, "organize my documents", max_unattended=1)
        state.update(g.request_file_approval(state))
        (request,) = state["approval_requests"]
        state["approvals"] = [
            ApprovalDecision(
                action_id=request["action_id"],
                action_fingerprint=request["action_hash"],
                verdict=ApprovalVerdict.REJECTED,
                decided_by="ada",
                request_id=request["request_id"],
            ).model_dump(mode="json")
        ]

        update = await g.apply_file_changes(state)

        assert {r["outcome"] for r in update["change_receipts"]} == {"skipped"}
        assert update["approval_refusals"][0]["reason"] == "denied_by_user"
        assert (root / "resume-2026.md").exists()

    async def test_no_plan_applies_nothing(self, root):
        assert await graph(root).apply_file_changes({}) == {
            "change_receipts": [],
            "approval_refusals": [],
        }


@covers_node(FILE_TASK_GRAPH, fsg.VERIFY_FILE_CHANGES)
class TestVerifyFileChanges:
    async def test_applied_moves_are_checked_against_the_disk(self, root):
        g, state = await planned(root, "organize my resumes")
        state.update(await g.apply_file_changes(state))

        update = await g.verify_file_changes(state)

        assert update["verified"] is True
        assert len(update["verifications"]) == 2

    async def test_a_change_undone_behind_its_back_fails_verification(self, root):
        g, state = await planned(root, "organize my resumes")
        state.update(await g.apply_file_changes(state))
        (root / "Resumes" / "resume-2026.md").rename(root / "resume-2026.md")

        update = await g.verify_file_changes(state)

        assert update["verified"] is False
        assert [v["ok"] for v in update["verifications"]].count(False) == 1

    async def test_nothing_applied_verifies_trivially(self, root):
        assert await graph(root).verify_file_changes({}) == {"verifications": [], "verified": True}
