"""Reading a resume for ingestion, through the sandboxed files tool.

`GatewayDocumentReader` satisfies the `DocumentReader` port of
`personalos.retrieval.evidence_ingestion.ResumeIngestor`. It holds no file
handle and does not resolve paths itself: every read is a `files.read_file`
intent dispatched through the tool gateway, so policy records a `READ_LOCAL`
decision for it and the files server's sandbox decides -- on the canonical
path, after `..` and symlinks are resolved -- whether the file is inside the
allowed roots. A path outside them is refused there, with nothing read, and
that refusal is raised here as `DocumentReadRefused`.
"""

import base64
import hashlib
import logging

from personalos.domain.errors import PolicyDeniedError, ValidationFailed
from personalos.domain.evidence import SourceDocument
from personalos.policy import IntentOrigin, ToolIntent
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

FILES_SERVER = "files"
READ_TOOL = "read_file"


class DocumentReadRefused(PolicyDeniedError):
    """The files tool would not read the document: outside the roots, missing, or unsafe."""

    default_message = "the files tool refused to read the document"


class GatewayDocumentReader:
    """Reads one file's bytes and hash through `files.read_file`."""

    def __init__(self, gateway: ToolGateway, *, requested_by: str = "evidence_ingestion"):
        """Take the gateway reads are dispatched through, and who they are for."""
        if gateway is None:
            raise ValueError("GatewayDocumentReader requires a ToolGateway")
        self.gateway = gateway
        self.requested_by = requested_by

    async def read(self, path: str) -> SourceDocument:
        """The file at `path`, as the sandbox resolved it, or `DocumentReadRefused`."""
        intent = ToolIntent(
            server=FILES_SERVER,
            tool=READ_TOOL,
            arguments={"path": path},
            origin=IntentOrigin.USER,
            requested_by=self.requested_by,
        )
        result = await self.gateway.dispatch(intent)
        if not result.success:
            logger.warning("files.read_file refused %r: %s", path, result.error)
            raise DocumentReadRefused(result.error or f"could not read {path!r}")

        payload = result.result or {}
        try:
            if payload["encoding"] == "base64":
                content = base64.b64decode(payload["content"], validate=True)
            else:
                content = payload["content"].encode("utf-8")
            canonical, sha256 = str(payload["path"]), str(payload["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationFailed("files.read_file returned an unreadable payload") from exc
        if hashlib.sha256(content).hexdigest() != sha256:
            raise ValidationFailed(f"the content read from {canonical!r} does not match its hash")
        return SourceDocument(path=canonical, content=content, sha256=sha256)


__all__ = ["FILES_SERVER", "READ_TOOL", "DocumentReadRefused", "GatewayDocumentReader"]
