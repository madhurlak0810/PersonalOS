"""Files MCP Server - sandboxed read/write of resumes and generated artifacts.

Every path a tool here receives goes through `PathSandbox.resolve` before
anything is opened, so the tool surface is exactly the configured allowed
roots regardless of what the caller sent. See `mcp_servers/files/sandbox.py`.

Writes are compare-and-swap on content: the caller names the SHA-256 of the
version it read, and the write is refused if the file on disk no longer hashes
to that. A caller that never read the file can only create it.
"""

import base64
import binascii
import hashlib
import logging
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator

from mcp_servers.files.sandbox import PathNotAllowed, PathSandbox, StaleWrite
from personalos.domain.errors import NotFound, ValidationFailed
from personalos.domain.models import Intent, MutatingIntent
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.persistence.idempotency import InMemoryOperationStore, OperationStore

logger = logging.getLogger(__name__)

#: Largest file a tool will read or write. A resume or cover letter is far
#: below this; the cap is what keeps a tool result from carrying a disk image.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024

ContentEncoding = Literal["utf-8", "base64"]

_SHA256_HEX_LENGTH = 64


class ReadFileIntent(Intent):
    """Typed parameters for `read_file`."""

    path: str


class WriteFileIntent(MutatingIntent):
    """Typed parameters for the mutating `write_file`."""

    path: str
    content: str
    encoding: ContentEncoding = "utf-8"
    expected_sha256: str | None = None

    @field_validator("expected_sha256")
    @classmethod
    def _check_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if len(value) != _SHA256_HEX_LENGTH or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("must be a hex-encoded SHA-256 digest")
        return value


def content_sha256(data: bytes) -> str:
    """Hex SHA-256 of file content: the version token `write_file` compares."""
    return hashlib.sha256(data).hexdigest()


class FilesMCPServer(MCPServer):
    """MCP Server for reading and writing files inside the allowed roots."""

    def __init__(
        self,
        sandbox: PathSandbox,
        operation_store: OperationStore | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        """Initialize Files MCP Server.

        `sandbox` is required rather than defaulted: which directories this
        server may touch is a deployment decision, made at the composition
        root, and a server that could fall back to a default root is one whose
        reach is not visible where it is constructed.
        """
        super().__init__(
            "files",
            "Read and write resume and artifact files inside the allowed roots",
            operation_store=operation_store or InMemoryOperationStore(),
        )
        self.sandbox = sandbox
        self.max_bytes = max_bytes
        # Serializes compare-then-write within this process. Writers in other
        # processes are caught by the hash check only up to the final rename.
        self._write_lock = threading.Lock()
        self.initialize()

    def initialize(self):
        """Register the file tools."""
        read_schema = ToolSchema(
            name="read_file",
            description=(
                "Read a file inside the allowed roots. Returns its content and "
                "the sha256 to pass as expected_sha256 when writing it back."
            ),
            intent_type=ReadFileIntent,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path of the file. Relative paths are resolved "
                            "against the primary allowed root."
                        ),
                    },
                },
            },
            required=["path"],
        )
        self.register_tool(read_schema, self._read_file)

        write_schema = ToolSchema(
            name="write_file",
            description=(
                "Create or replace a file inside the allowed roots. Replacing "
                "requires expected_sha256 to match the file's current content."
            ),
            intent_type=WriteFileIntent,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path of the file. Relative paths are resolved "
                            "against the primary allowed root."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "New content, encoded as `encoding` says",
                    },
                    "encoding": {
                        "type": "string",
                        "enum": ["utf-8", "base64"],
                        "default": "utf-8",
                        "description": "How `content` is encoded",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "description": (
                            "sha256 of the version being replaced, from "
                            "read_file. Omit only when creating a new file."
                        ),
                    },
                },
            },
            required=["path", "content"],
        )
        self.register_tool(write_schema, self._write_file)

    def _read_file(self, path: str) -> dict[str, Any]:
        """Read a file inside the allowed roots."""
        target = self.sandbox.resolve(path)
        data = self._read_bytes(target, path)
        if data is None:
            raise NotFound(f"file {path!r} does not exist")
        logger.info(f"Read {len(data)} bytes from {target}")

        try:
            content, encoding = data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            content, encoding = base64.b64encode(data).decode("ascii"), "base64"
        return {
            "path": str(target),
            "content": content,
            "encoding": encoding,
            "size": len(data),
            "sha256": content_sha256(data),
        }

    def _write_file(
        self,
        path: str,
        content: str,
        encoding: ContentEncoding = "utf-8",
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Create or replace a file inside the allowed roots."""
        target = self.sandbox.resolve(path)
        data = _decode_content(content, encoding)
        if len(data) > self.max_bytes:
            raise ValidationFailed(
                f"content is {len(data)} bytes; the limit is {self.max_bytes}"
            )

        with self._write_lock:
            current = self._read_bytes(target, path)
            previous_sha256 = None if current is None else content_sha256(current)
            if current is None and expected_sha256 is not None:
                raise StaleWrite(
                    f"file {path!r} no longer exists; omit expected_sha256 to create it"
                )
            if current is not None and expected_sha256 is None:
                raise StaleWrite(
                    f"file {path!r} already exists; read it and pass its sha256 "
                    f"as expected_sha256 to replace it"
                )
            if current is not None and previous_sha256 != expected_sha256:
                raise StaleWrite(
                    f"file {path!r} changed since it was read "
                    f"(current sha256 {previous_sha256}); re-read it before writing"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            # Creating the parents is the one step that happens after the
            # check, so confirm the target still canonicalizes to where the
            # check said it would.
            if self.sandbox.resolve(path) != target:
                raise PathNotAllowed(f"path {path!r} changed while it was being written")
            self._replace_atomically(target, data, path, creating=current is None)

        logger.info(f"Wrote {len(data)} bytes to {target}")
        return {
            "path": str(target),
            "size": len(data),
            "sha256": content_sha256(data),
            "previous_sha256": previous_sha256,
            "created": current is None,
        }

    def _read_bytes(self, target: Path, requested: str) -> bytes | None:
        """Read a canonical path's bytes, or return None if nothing is there.

        `target` has already had its symlinks resolved, so the final component
        being a symlink now means it was swapped in after the check:
        `O_NOFOLLOW` refuses it rather than following it out of the sandbox.
        """
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(target, flags)
        except FileNotFoundError:
            return None
        except NotADirectoryError:
            return None
        except OSError as e:
            raise PathNotAllowed(f"path {requested!r} could not be opened safely") from e
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise PathNotAllowed(f"path {requested!r} is not a regular file")
            if info.st_size > self.max_bytes:
                raise ValidationFailed(
                    f"file {requested!r} is {info.st_size} bytes; the limit is {self.max_bytes}"
                )
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                return handle.read(self.max_bytes + 1)[: self.max_bytes]
        finally:
            if fd != -1:
                os.close(fd)

    def _replace_atomically(
        self, target: Path, data: bytes, requested: str, *, creating: bool
    ) -> None:
        """Write `data` beside `target`, then move it into place in one step.

        A reader never sees a half-written file. When creating, the move is a
        hard link, which fails if something appeared at `target` since the
        check -- so two racing creates cannot both win.
        """
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".personalos-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if creating:
                try:
                    os.link(tmp_name, target)
                except FileExistsError as e:
                    raise StaleWrite(
                        f"file {requested!r} was created by another writer; "
                        f"read it before writing"
                    ) from e
            else:
                os.replace(tmp_name, target)
                tmp_name = None
        finally:
            if tmp_name is not None:
                os.unlink(tmp_name)


def _decode_content(content: str, encoding: ContentEncoding) -> bytes:
    if encoding == "utf-8":
        return content.encode("utf-8")
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValidationFailed("content is not valid base64") from e
