"""Maps every error a route can raise onto one transport-safe envelope.

`PersonalOSError` covers deterministic failures raised anywhere in the call
graph (API, executor, MCP, persistence); `RequestValidationError` is FastAPI's
own pydantic-driven request validation; the catch-all handles anything left,
so a bug never reaches a caller as a raw stack trace. Each handler logs with a
stack trace internally (`exc_info`) and returns only the sanitized envelope
externally.

Kept in its own module, separate from `apps.api.main.create_app`, so a test
building a bare router-only app can register the same handlers rather than
asserting on default FastAPI/Starlette error behavior.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from personalos.domain.errors import InternalError, PersonalOSError, ValidationFailed

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Attach the shared error-envelope handlers to `app`."""

    @app.exception_handler(PersonalOSError)
    async def handle_personalos_error(request: Request, exc: PersonalOSError):
        logger.error(
            "%s %s failed (error_code=%s, context_id=%s): %s",
            request.method,
            request.url.path,
            exc.code.value,
            exc.context_id,
            exc.message,
            exc_info=exc,
        )
        return JSONResponse(status_code=exc.http_status, content=exc.to_envelope())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        error = ValidationFailed(details={"errors": exc.errors()})
        logger.info(
            "%s %s failed request validation (context_id=%s)",
            request.method,
            request.url.path,
            error.context_id,
        )
        return JSONResponse(status_code=error.http_status, content=error.to_envelope())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        error = InternalError()
        logger.exception(
            "%s %s failed with an unhandled exception (context_id=%s)",
            request.method,
            request.url.path,
            error.context_id,
        )
        return JSONResponse(status_code=error.http_status, content=error.to_envelope())


__all__ = ["register_error_handlers"]
