"""The files server's discovery and reorganization tools: list, stat, move, delete."""

import hashlib
from pathlib import Path

import pytest

from mcp_servers.files.sandbox import PathSandbox
from mcp_servers.files.server import FilesMCPServer
from personalos.domain.models import ActionTarget, ToolCallErrorCode, ToolCallRequest


def request(tool: str, **params) -> ToolCallRequest:
    return ToolCallRequest(target=ActionTarget(server="files", tool=tool), params=params)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    allowed = tmp_path / "docs"
    allowed.mkdir()
    (allowed / "resume.md").write_text("# Resume")
    (allowed / "nested").mkdir()
    (allowed / "nested" / "cover-letter.txt").write_text("Dear team")
    (allowed / "nested" / "data.json").write_text("{}")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.md").write_text("not yours")
    return allowed


@pytest.fixture
def denials() -> list:
    return []


@pytest.fixture
def server(root, denials) -> FilesMCPServer:
    return FilesMCPServer(PathSandbox([root]), denial_sink=denials.append)


async def test_list_files_reports_documents_with_hashes_and_no_content(server, root):
    result = await server.execute(request("list_files"))

    assert result.ok
    entries = {Path(e["path"]).name: e for e in result.result["entries"]}
    assert set(entries) == {"resume.md", "cover-letter.txt"}
    assert entries["resume.md"]["sha256"] == sha256("# Resume")
    assert "content" not in entries["resume.md"]
    assert result.result["roots"] == [str(root.resolve())]


async def test_list_files_skips_a_symlink_out_of_the_roots(server, root, tmp_path, denials):
    (root / "link.md").symlink_to(tmp_path / "outside" / "secret.md")

    result = await server.execute(request("list_files"))

    assert all("secret" not in e["path"] for e in result.result["entries"])
    assert all(Path(e["path"]).name != "link.md" for e in result.result["entries"])


async def test_list_files_of_a_directory_outside_is_denied_and_reported(server, denials):
    result = await server.execute(request("list_files", directory="../outside"))

    assert result.error.code == ToolCallErrorCode.PERMISSION_DENIED
    assert denials[0].tool == "list_files"


async def test_stat_file_reports_a_missing_file_without_failing(server):
    result = await server.execute(request("stat_file", path="nope.md"))

    assert result.ok and result.result["exists"] is False


async def test_move_is_hash_guarded_never_clobbers_and_returns_its_undo(server, root):
    moved = await server.execute(
        request(
            "move_file",
            source="resume.md",
            destination="Resumes/resume.md",
            expected_sha256=sha256("# Resume"),
            plan_size=1,
            idempotency_key="move-resume-0001",
        )
    )

    assert moved.ok, moved.error
    assert (root / "Resumes" / "resume.md").read_text() == "# Resume"
    assert not (root / "resume.md").exists()
    undo = moved.result["undo"]
    assert undo["path"].endswith("Resumes/resume.md") and undo["destination"].endswith("resume.md")

    (root / "resume.md").write_text("another")
    clobber = await server.execute(
        request(
            "move_file",
            source="resume.md",
            destination="Resumes/resume.md",
            expected_sha256=sha256("another"),
            plan_size=1,
            idempotency_key="move-resume-0002",
        )
    )
    assert not clobber.ok and "already exists" in clobber.error.message
    assert (root / "Resumes" / "resume.md").read_text() == "# Resume"


async def test_a_move_of_a_changed_file_is_refused(server, root):
    result = await server.execute(
        request(
            "move_file",
            source="resume.md",
            destination="r.md",
            expected_sha256=sha256("an older version"),
            plan_size=1,
            idempotency_key="move-resume-0003",
        )
    )

    assert not result.ok and "changed since it was read" in result.error.message
    assert (root / "resume.md").exists()


async def test_move_requires_its_plan_size(server):
    result = await server.execute(
        request(
            "move_file",
            source="resume.md",
            destination="r.md",
            expected_sha256=sha256("# Resume"),
            idempotency_key="move-resume-0004",
        )
    )

    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR


async def test_delete_removes_only_the_version_named(server, root):
    stale = await server.execute(
        request(
            "delete_file",
            path="resume.md",
            expected_sha256=sha256("not this"),
            idempotency_key="delete-resume-01",
        )
    )
    assert not stale.ok and (root / "resume.md").exists()

    deleted = await server.execute(
        request(
            "delete_file",
            path="resume.md",
            expected_sha256=sha256("# Resume"),
            idempotency_key="delete-resume-02",
        )
    )
    assert deleted.ok and not (root / "resume.md").exists()


async def test_a_retried_move_with_the_same_key_is_replayed_not_repeated(server, root):
    params = {
        "source": "resume.md",
        "destination": "Resumes/resume.md",
        "expected_sha256": sha256("# Resume"),
        "plan_size": 1,
        "idempotency_key": "move-resume-0005",
    }
    first = await server.execute(request("move_file", **params))
    second = await server.execute(request("move_file", **params))

    assert first.ok and second.ok and second.replayed is True
    assert (root / "Resumes" / "resume.md").read_text() == "# Resume"
