"""Tests for the files MCP server's path sandbox and hash-guarded writes."""

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import pytest

from mcp_servers.files.sandbox import PathNotAllowed, PathSandbox
from mcp_servers.files.server import FilesMCPServer
from personalos.domain.models import ActionTarget, ToolCallErrorCode, ToolCallRequest


def request(tool: str, **params) -> ToolCallRequest:
    """Build a ToolCallRequest against the files server."""
    return ToolCallRequest(target=ActionTarget(server="files", tool=tool), params=params)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An allowed root with one resume in it, next to a directory that is not allowed."""
    allowed = tmp_path / "artifacts"
    allowed.mkdir()
    (allowed / "resume.md").write_text("# Resume v1")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not yours")
    return allowed


@pytest.fixture
def sandbox(root: Path) -> PathSandbox:
    return PathSandbox([root])


@pytest.fixture
def server(sandbox: PathSandbox) -> FilesMCPServer:
    return FilesMCPServer(sandbox)


# ----------------------------------------------------------------------
# Canonicalization and allowed roots
# ----------------------------------------------------------------------


def test_path_inside_root_resolves_to_its_canonical_form(sandbox, root):
    assert sandbox.resolve("resume.md") == root / "resume.md"
    assert sandbox.resolve(str(root / "drafts" / ".." / "resume.md")) == root / "resume.md"


def test_relative_path_is_anchored_on_the_root_not_the_working_directory(
    sandbox, root, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path / "outside")
    assert sandbox.resolve("secret.txt") == root / "secret.txt"


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "drafts/../../outside/secret.txt",
        "../outside/secret.txt",
    ],
)
def test_relative_traversal_out_of_the_root_is_rejected(sandbox, path):
    with pytest.raises(PathNotAllowed, match="outside the allowed roots"):
        sandbox.resolve(path)


def test_traversal_is_rejected_even_when_the_raw_string_starts_with_an_allowed_root(
    sandbox, root
):
    """The prefix is the allowed root; the canonical path is not under it."""
    for raw in (
        f"{root}/../../{root.parent.name}/outside/secret.txt",
        f"{root}/drafts/../../outside/secret.txt",
        f"{root}/../../../../../../etc/passwd",
    ):
        assert raw.startswith(str(root))
        with pytest.raises(PathNotAllowed, match="outside the allowed roots"):
            sandbox.resolve(raw)


def test_absolute_path_outside_every_root_is_rejected(sandbox, tmp_path):
    with pytest.raises(PathNotAllowed):
        sandbox.resolve(str(tmp_path / "outside" / "secret.txt"))


def test_sibling_directory_sharing_the_roots_name_prefix_is_rejected(tmp_path, root):
    """`/x/artifacts-private` is not inside `/x/artifacts`."""
    sibling = tmp_path / "artifacts-private"
    sibling.mkdir()
    (sibling / "notes.md").write_text("x")

    with pytest.raises(PathNotAllowed):
        PathSandbox([root]).resolve(str(sibling / "notes.md"))


def test_symlink_inside_root_pointing_outside_is_rejected(sandbox, root, tmp_path):
    (root / "innocent.md").symlink_to(tmp_path / "outside" / "secret.txt")

    with pytest.raises(PathNotAllowed, match="outside the allowed roots"):
        sandbox.resolve("innocent.md")


def test_symlinked_directory_inside_root_pointing_outside_is_rejected(sandbox, root, tmp_path):
    (root / "shared").symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(PathNotAllowed, match="outside the allowed roots"):
        sandbox.resolve("shared/secret.txt")
    # A file that does not exist yet is still resolved through the link.
    with pytest.raises(PathNotAllowed, match="outside the allowed roots"):
        sandbox.resolve("shared/new-file.md")


def test_dangling_symlink_pointing_outside_is_rejected(sandbox, root, tmp_path):
    (root / "later.md").symlink_to(tmp_path / "outside" / "not-created-yet.md")

    with pytest.raises(PathNotAllowed):
        sandbox.resolve("later.md")


def test_symlink_that_stays_inside_the_root_is_allowed(sandbox, root):
    (root / "latest.md").symlink_to(root / "resume.md")

    assert sandbox.resolve("latest.md") == root / "resume.md"


def test_any_configured_root_is_accepted(root, tmp_path):
    second = tmp_path / "resumes"
    second.mkdir()
    sandbox = PathSandbox([root, second])

    assert sandbox.resolve(str(second / "cv.pdf")) == second / "cv.pdf"


@pytest.mark.parametrize("path", ["", "   ", "resume\x00.md"])
def test_malformed_paths_are_rejected(sandbox, path):
    with pytest.raises(PathNotAllowed):
        sandbox.resolve(path)


def test_the_root_itself_is_not_a_file(sandbox, root):
    with pytest.raises(PathNotAllowed):
        sandbox.resolve(str(root))


def test_no_configured_roots_rejects_everything(tmp_path):
    with pytest.raises(PathNotAllowed, match="no allowed roots"):
        PathSandbox([]).resolve(str(tmp_path / "resume.md"))


# ----------------------------------------------------------------------
# Default-denied directories
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".ssh/id_rsa",
        ".aws/credentials",
        ".config/gcloud/credentials.db",
        "gcloud/application_default_credentials.json",
        ".mozilla/firefox/profile/logins.json",
        "google-chrome/Default/Cookies",
        "backup/.SSH/id_ed25519",
        ".env",
        "notes/.hidden/file.md",
    ],
)
def test_denied_directories_are_rejected_inside_an_allowed_root(sandbox, root, path):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret")

    with pytest.raises(PathNotAllowed, match="denied directory"):
        sandbox.resolve(path)
    with pytest.raises(PathNotAllowed, match="denied directory"):
        sandbox.resolve(str(target))


def test_symlink_into_a_denied_directory_is_rejected(sandbox, root):
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id_rsa").write_text("key")
    (root / "resume-final.md").symlink_to(root / ".ssh" / "id_rsa")

    with pytest.raises(PathNotAllowed, match="denied directory"):
        sandbox.resolve("resume-final.md")


def test_denied_directory_reached_through_dot_dot_is_rejected(sandbox, root):
    (root / ".ssh").mkdir()

    with pytest.raises(PathNotAllowed, match="denied directory"):
        sandbox.resolve("drafts/../.ssh/id_rsa")


@pytest.mark.parametrize("name", [".ssh", ".aws", ".config"])
def test_a_denied_directory_cannot_be_configured_as_a_root(tmp_path, name):
    denied = tmp_path / name
    denied.mkdir()

    with pytest.raises(ValueError, match="denied directory"):
        PathSandbox([denied])


def test_root_must_be_absolute_and_not_the_filesystem_root():
    with pytest.raises(ValueError, match="absolute"):
        PathSandbox(["artifacts"])
    with pytest.raises(ValueError, match="filesystem root"):
        PathSandbox([os.path.abspath(os.sep)])


# ----------------------------------------------------------------------
# The tools enforce the sandbox
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_returns_content_and_hash(server, root):
    result = await server.execute(request("read_file", path="resume.md"))

    assert result.ok is True
    assert result.result["content"] == "# Resume v1"
    assert result.result["encoding"] == "utf-8"
    assert result.result["sha256"] == sha256("# Resume v1")
    assert result.result["path"] == str(root / "resume.md")


@pytest.mark.asyncio
async def test_read_file_returns_binary_content_as_base64(server, root):
    (root / "resume.pdf").write_bytes(b"%PDF\xff\xfe")

    result = await server.execute(request("read_file", path="resume.pdf"))

    assert result.ok is True
    assert result.result["encoding"] == "base64"
    assert result.result["sha256"] == hashlib.sha256(b"%PDF\xff\xfe").hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../../etc/passwd", "../outside/secret.txt"])
async def test_read_file_rejects_traversal(server, path):
    result = await server.execute(request("read_file", path=path))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.EXECUTION_ERROR
    assert "outside the allowed roots" in result.error.message


@pytest.mark.asyncio
async def test_read_file_rejects_a_symlink_pointing_outside(server, root, tmp_path):
    (root / "innocent.md").symlink_to(tmp_path / "outside" / "secret.txt")

    result = await server.execute(request("read_file", path="innocent.md"))

    assert result.ok is False
    assert "not yours" not in str(result)


@pytest.mark.asyncio
async def test_read_and_write_reject_a_denied_directory_when_explicitly_requested(server, root):
    (root / ".ssh").mkdir()
    key = root / ".ssh" / "id_rsa"
    key.write_text("PRIVATE KEY")

    read = await server.execute(request("read_file", path=str(key)))
    write = await server.execute(
        request(
            "write_file",
            path=".ssh/authorized_keys",
            content="ssh-ed25519 AAAA attacker",
            idempotency_key=str(uuid4()),
        )
    )

    assert read.ok is False
    assert "denied directory" in read.error.message
    assert "PRIVATE KEY" not in str(read)
    assert write.ok is False
    assert "denied directory" in write.error.message
    assert not (root / ".ssh" / "authorized_keys").exists()


@pytest.mark.asyncio
async def test_write_file_rejects_traversal_and_writes_nothing(server, tmp_path):
    result = await server.execute(
        request(
            "write_file",
            path="../outside/planted.md",
            content="x",
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is False
    assert not (tmp_path / "outside" / "planted.md").exists()


@pytest.mark.asyncio
async def test_write_file_through_a_symlink_pointing_outside_leaves_the_target_untouched(
    server, root, tmp_path
):
    secret = tmp_path / "outside" / "secret.txt"
    (root / "innocent.md").symlink_to(secret)

    result = await server.execute(
        request(
            "write_file",
            path="innocent.md",
            content="overwritten",
            expected_sha256=sha256("not yours"),
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is False
    assert secret.read_text() == "not yours"


@pytest.mark.asyncio
async def test_read_file_reports_a_missing_file(server):
    result = await server.execute(request("read_file", path="nope.md"))

    assert result.ok is False
    assert "does not exist" in result.error.message


# ----------------------------------------------------------------------
# Content-hashed writes
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_creates_a_new_file_and_its_parents(server, root):
    result = await server.execute(
        request(
            "write_file",
            path="applications/acme/cover-letter.md",
            content="Dear Acme",
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is True
    assert result.result["created"] is True
    assert result.result["previous_sha256"] is None
    assert result.result["sha256"] == sha256("Dear Acme")
    assert (root / "applications" / "acme" / "cover-letter.md").read_text() == "Dear Acme"


@pytest.mark.asyncio
async def test_write_file_replaces_the_version_that_was_read(server, root):
    read = await server.execute(request("read_file", path="resume.md"))

    result = await server.execute(
        request(
            "write_file",
            path="resume.md",
            content="# Resume v2",
            expected_sha256=read.result["sha256"],
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is True
    assert result.result["created"] is False
    assert result.result["previous_sha256"] == sha256("# Resume v1")
    assert (root / "resume.md").read_text() == "# Resume v2"
    assert os.listdir(root) == ["resume.md"]  # no temp file left behind


@pytest.mark.asyncio
async def test_stale_write_is_rejected_and_the_newer_content_survives(server, root):
    read = await server.execute(request("read_file", path="resume.md"))
    (root / "resume.md").write_text("# Resume edited by hand")

    result = await server.execute(
        request(
            "write_file",
            path="resume.md",
            content="# Resume v2",
            expected_sha256=read.result["sha256"],
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is False
    assert "changed since it was read" in result.error.message
    assert (root / "resume.md").read_text() == "# Resume edited by hand"
    assert os.listdir(root) == ["resume.md"]


@pytest.mark.asyncio
async def test_existing_file_cannot_be_overwritten_without_its_hash(server, root):
    result = await server.execute(
        request("write_file", path="resume.md", content="blind", idempotency_key=str(uuid4()))
    )

    assert result.ok is False
    assert "already exists" in result.error.message
    assert (root / "resume.md").read_text() == "# Resume v1"


@pytest.mark.asyncio
async def test_write_expecting_a_file_that_was_deleted_is_rejected(server, root):
    result = await server.execute(
        request(
            "write_file",
            path="gone.md",
            content="x",
            expected_sha256=sha256("whatever it was"),
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is False
    assert "no longer exists" in result.error.message
    assert not (root / "gone.md").exists()


@pytest.mark.asyncio
async def test_second_writer_on_the_same_version_loses(server, root):
    """Two writers read v1; only the first write lands."""
    base = sha256("# Resume v1")

    first = await server.execute(
        request(
            "write_file",
            path="resume.md",
            content="# From writer A",
            expected_sha256=base,
            idempotency_key=str(uuid4()),
        )
    )
    second = await server.execute(
        request(
            "write_file",
            path="resume.md",
            content="# From writer B",
            expected_sha256=base,
            idempotency_key=str(uuid4()),
        )
    )

    assert first.ok is True
    assert second.ok is False
    assert (root / "resume.md").read_text() == "# From writer A"


@pytest.mark.asyncio
async def test_retried_write_with_the_same_key_replays_instead_of_going_stale(server, root):
    params = {
        "path": "resume.md",
        "content": "# Resume v2",
        "expected_sha256": sha256("# Resume v1"),
        "idempotency_key": str(uuid4()),
    }

    first = await server.execute(request("write_file", **params))
    second = await server.execute(request("write_file", **params))

    assert first.ok is True
    assert second.ok is True
    assert second.replayed is True
    assert second.result == first.result


@pytest.mark.asyncio
async def test_malformed_expected_hash_is_a_validation_error(server):
    result = await server.execute(
        request(
            "write_file",
            path="resume.md",
            content="x",
            expected_sha256="not-a-hash",
            idempotency_key=str(uuid4()),
        )
    )

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR


@pytest.mark.asyncio
async def test_oversized_content_is_rejected(sandbox, root):
    server = FilesMCPServer(sandbox, max_bytes=8)

    write = await server.execute(
        request("write_file", path="big.md", content="123456789", idempotency_key=str(uuid4()))
    )
    read = await server.execute(request("read_file", path="resume.md"))

    assert write.ok is False
    assert not (root / "big.md").exists()
    assert read.ok is False
    assert "limit" in read.error.message


@pytest.mark.asyncio
async def test_directory_is_not_readable_as_a_file(server, root):
    (root / "drafts").mkdir()

    result = await server.execute(request("read_file", path="drafts"))

    assert result.ok is False
