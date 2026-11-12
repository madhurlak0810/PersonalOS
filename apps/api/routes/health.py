"""`GET /health`: the one unauthenticated route.

It answers whether the process is up and serving, and nothing else -- no
version, no configuration, no integration state (that is
`GET /v1/integrations`, which needs a token). A liveness probe has no token
to send, and an answer this small gives an unauthenticated caller nothing.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """The process is up."""

    status: str = "ok"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: 200 while the API process can serve a request."""
    return HealthResponse()


__all__ = ["router", "HealthResponse"]
