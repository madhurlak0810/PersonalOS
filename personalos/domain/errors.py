"""Shared, deterministic error taxonomy and the transport-safe envelope built from it.

Every layer that can fail either raises one of these types directly, or wraps
its own exception in one at the boundary where the failure gets reported (an
API route, a job's terminal state). The hierarchy lives in `domain` rather
than in `policy` or `tools` -- where the specific failure types already live
-- because `domain` is the one layer every other checked layer is allowed to
depend on (see `tests/architecture/boundaries.py`); that is what lets every
layer report the same finite set of `error_code` values instead of each
inventing its own strings.

`to_envelope()` is deliberately narrow: `error_code`, `message`, and
`context_id` only. The original exception message, arguments, and stack trace
belong in the log line that reports the error (via `logger.exception` /
`exc_info=True`), not in anything handed to a caller outside the process.
"""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class ErrorCode(str, Enum):
    """Stable, wire-visible failure categories.

    A client (or a test) may match on this string, so renaming or repurposing
    a member is a breaking change.
    """

    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    POLICY_DENIED = "policy_denied"
    APPROVAL_REQUIRED = "approval_required"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    TOOL_FAILURE = "tool_failure"
    RETRYABLE = "retryable"
    INTERNAL = "internal"


class PersonalOSError(Exception):
    """Base for every deterministic, taxonomy-mapped error in the system.

    Subclasses fix `code`, `http_status`, and `retryable` as class attributes,
    so the mapping from failure type to wire representation cannot drift
    between call sites. `context_id` is generated per instance and is the
    thread that ties a sanitized external response back to the full internal
    log entry for that same failure.
    """

    code: ErrorCode = ErrorCode.INTERNAL
    http_status: int = 500
    retryable: bool = False
    default_message: str = "an internal error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        context_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message or self.default_message
        self.context_id = context_id or uuid4()
        self.details = details or {}
        super().__init__(self.message)

    def to_envelope(self) -> dict[str, str]:
        """Transport-safe representation: stable fields only, no internals."""
        return {
            "error_code": self.code.value,
            "message": self.message,
            "context_id": str(self.context_id),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code.value!r}, "
            f"context_id={self.context_id}, message={self.message!r})"
        )


class ValidationFailed(PersonalOSError):
    """A caller-supplied request or argument did not meet its contract."""

    code = ErrorCode.VALIDATION
    http_status = 422
    default_message = "request failed validation"


class NotFound(PersonalOSError):
    """The referenced resource does not exist."""

    code = ErrorCode.NOT_FOUND
    http_status = 404
    default_message = "resource not found"


class PolicyDeniedError(PersonalOSError):
    """The policy engine refused the action outright."""

    code = ErrorCode.POLICY_DENIED
    http_status = 403
    default_message = "action denied by policy"


class ApprovalRequiredError(PersonalOSError):
    """The action needs a human approval that was not supplied."""

    code = ErrorCode.APPROVAL_REQUIRED
    http_status = 428
    default_message = "action requires approval"


class IdempotencyConflict(PersonalOSError):
    """An idempotency key could not be honored as requested."""

    code = ErrorCode.IDEMPOTENCY_CONFLICT
    http_status = 409
    default_message = "idempotency key conflict"


class ToolFailure(PersonalOSError):
    """An approved tool call reached its adapter and failed there."""

    code = ErrorCode.TOOL_FAILURE
    http_status = 502
    retryable = True
    default_message = "tool execution failed"


class RetryableFailure(PersonalOSError):
    """A transient condition; the same request may succeed if retried later."""

    code = ErrorCode.RETRYABLE
    http_status = 409
    retryable = True
    default_message = "temporarily unavailable; retry later"


class InternalError(PersonalOSError):
    """An unclassified failure. Used at boundaries that wrap unknown exceptions."""

    code = ErrorCode.INTERNAL
    http_status = 500
    default_message = "an internal error occurred"


__all__ = [
    "ErrorCode",
    "PersonalOSError",
    "ValidationFailed",
    "NotFound",
    "PolicyDeniedError",
    "ApprovalRequiredError",
    "IdempotencyConflict",
    "ToolFailure",
    "RetryableFailure",
    "InternalError",
]
