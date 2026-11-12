"""Loading the caller's resume into the evidence index over HTTP. Version 1.

- `POST /v1/evidence/documents` ingests the resume at `path` (PDF, DOCX,
  Markdown or text): the file is read through the sandboxed files tool,
  chunked into sections, roles, projects and skills, embedded and indexed as
  the evidence job matching and artifact prep cite. 201 when the path is new;
  200 when it was re-ingested (`updated`) or had not changed (`unchanged`, a
  no-op).

`path` names a file on the server, inside `FILES_ALLOWED_ROOTS`; a relative
path is resolved against the first root. Anything outside them -- including by
`..` or a symlink -- is a 403 and nothing is read. Nothing is uploaded.

Like `/v1/profile`, the evidence belongs to the authenticated actor
(`apps.api.auth.current_actor`); a caller cannot name another.
"""

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field

from apps.api.auth import current_actor
from personalos.bootstrap import build_resume_ingestor
from personalos.domain.evidence import IngestionResult, IngestionStatus
from personalos.domain.search_profiles import actor_user_id
from personalos.persistence.database import SessionLocal
from personalos.retrieval.evidence_ingestion import ResumeIngestor

router = APIRouter()


class IngestResumeRequest(BaseModel):
    """Which file to ingest."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)


def get_resume_ingestor() -> ResumeIngestor:
    """FastAPI dependency: an ingestor over the configured database and files tool."""
    return build_resume_ingestor(SessionLocal, requested_by="api:ingest_resume")


@router.post("/evidence/documents", response_model=IngestionResult)
async def ingest_resume(
    body: IngestResumeRequest,
    http_response: Response,
    ingestor: ResumeIngestor = Depends(get_resume_ingestor),
    actor_id: str = Depends(current_actor),
):
    """Ingest the resume at `path` as the actor's evidence."""
    result = await ingestor.ingest(actor_user_id(actor_id), body.path, display_name=actor_id)
    http_response.status_code = 201 if result.status == IngestionStatus.INDEXED else 200
    return result


__all__ = ["router", "get_resume_ingestor", "IngestResumeRequest"]
