"""A sandboxed resume, the production read path, and builders for binary formats.

`ingestor` wires `build_resume_ingestor` exactly as the CLI and API do -- the
policy engine, the MCP adapter, the files server confined to one allowed root
-- over a SQLite database, so a test exercises the real read path and the real
sync, with only the embedder swappable. `minimal_docx` and `minimal_pdf` build
the smallest files of those formats that carry the given paragraphs, so no
binary fixture has to be checked in.
"""

import io
import shutil
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID
from xml.sax.saxutils import escape

from mcp_servers.files.sandbox import PathSandbox
from mcp_servers.files.server import FilesMCPServer
from personalos.bootstrap import build_resume_ingestor, build_tool_gateway
from personalos.mcp.manager import MCPServerManager
from personalos.models.embedding import HashingEmbedder
from personalos.persistence.models import (
    ApplicationModel,
    ArtifactVersionModel,
    DocumentModel,
    EvidenceChunkModel,
    JobPostingModel,
)
from personalos.policy import default_policy_engine
from personalos.retrieval.evidence_ingestion import ResumeIngestor

FIXTURE_RESUME = Path(__file__).parent / "resumes" / "candidate_resume.md"


class CountingEmbedder(HashingEmbedder):
    """The default embedder, recording every text it was asked to embed."""

    def __init__(self, model: str | None = None, version: str | None = None):
        if model is not None:
            self.model = model
        if version is not None:
            self.version = version
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector(text)

    async def embed_many(self, texts) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


def allowed_root(tmp_path: Path) -> Path:
    """An allowed root holding the fixture resume, with a sibling directory outside it."""
    root = tmp_path / "job-search"
    root.mkdir(parents=True)
    shutil.copy(FIXTURE_RESUME, root / "resume.md")
    return root


def ingestor(
    root: Path,
    factory: Callable[[], Any],
    embedder: HashingEmbedder | None = None,
) -> ResumeIngestor:
    """The production ingestor, confined to `root`, over `factory`'s database."""
    manager = MCPServerManager()
    manager.register_server(FilesMCPServer(PathSandbox([root])))
    gateway = build_tool_gateway(manager=manager, policy=default_policy_engine())
    return build_resume_ingestor(factory, gateway=gateway, embedder=embedder or CountingEmbedder())


def snapshot(factory: Callable[[], Any]) -> dict[str, Any]:
    """Every `documents` and `evidence_chunks` row, as comparable dicts."""
    session = factory()
    try:
        documents = [
            row.to_dict() for row in session.query(DocumentModel).order_by(DocumentModel.path)
        ]
        chunks = {
            str(row.id): {**row.to_dict(), "embedding": list(row.embedding)}
            for row in session.query(EvidenceChunkModel)
        }
        return {"documents": documents, "chunks": chunks}
    finally:
        session.close()


def cite(factory: Callable[[], Any], user_id: UUID, evidence_ids: Sequence[str]) -> None:
    """Store an artifact version citing `evidence_ids`, as artifact prep would."""
    session = factory()
    try:
        posting = JobPostingModel(
            source="fake",
            source_job_id="cited-1",
            title="Senior Backend Engineer",
            company="Northwind",
            description_hash="0" * 64,
            dedupe_key="northwind:senior-backend-engineer",
        )
        session.add(posting)
        session.flush()
        application = ApplicationModel(job_posting_id=posting.id, user_id=user_id)
        session.add(application)
        session.flush()
        session.add(
            ArtifactVersionModel(
                application_id=application.id,
                artifact_type="resume",
                content="Built Python APIs backed by Postgres.",
                evidence=[{"type": "resume", "ref": ref} for ref in evidence_ids],
            )
        )
        session.commit()
    finally:
        session.close()


_DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def minimal_docx(paragraphs: Sequence[tuple[str | None, str]]) -> bytes:
    """A DOCX of `(style, text)` paragraphs; style `"Heading2"`, `"List"` or None."""
    body = []
    for style, text in paragraphs:
        props = ""
        if style == "List":
            props = '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        elif style:
            props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        body.append(f"<w:p>{props}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>")
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_DOCX_NS}"><w:body>{"".join(body)}</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def minimal_pdf(lines: Sequence[str]) -> bytes:
    """A one-page PDF showing `lines` in Helvetica, one per line."""

    def literal(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream = (
        "BT /F1 11 Tf 14 TL 72 720 Td "
        + " ".join(f"({literal(line)}) Tj T*" for line in lines)
        + " ET"
    )
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
