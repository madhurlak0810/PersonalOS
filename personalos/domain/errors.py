"""Shared, deterministic error taxonomy and the transport-safe envelope built from it.

Every layer that can fail either raises one of these types directly, or wraps
its own exception in one at the boundary where the failure gets reported (an
API route, a job's terminal state). The hierarchy lives in `domain` rather
than in `policy` or `tools` -- where the specific failure types already live
-- because `domain` is the one layer every other checked layer is allowed to
depend on (see `tests/architecture/boundaries.py`); that is what lets every
layer report the same finite set of `error_code` values instead of each
inventing its own strings.

Every error also names a `FailureClass` -- the Phase A recovery model. The
`ErrorCode` says what went wrong to a caller; the failure class says what can
be done about it once a workflow step has failed: retried, reconciled first,
reconnected, or only acknowledged. It is part of `repr()` because `repr()` is
all of an exception that survives a checkpoint (LangGraph stores a node's
error as its repr), and `personalos.domain.failures` reads it back from there.

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
    CONFLICT = "conflict"
    TOOL_FAILURE = "tool_failure"
    RETRYABLE = "retryable"
    INTERNAL = "internal"


class FailureClass(str, Enum):
    """Why a workflow step failed, in terms of how it may be recovered.

    Wire-visible on `GET /v1/workflows/{id}/failures`. Which of these a user
    may retry is `personalos.domain.failures`'s rule, not a property of the
    exception: `PersonalOSError.retryable` is about *automatic* retries.
    """

    #: The model timed out, was rate limited or was briefly unavailable.
    MODEL_TRANSIENT = "model_transient"
    #: The model answered, but with output that did not meet its contract.
    MODEL_SEMANTIC = "model_semantic"
    #: A provider timed out or was briefly unavailable.
    PROVIDER_TRANSIENT = "provider_transient"
    #: A provider refused in a way retrying cannot change: revoked OAuth, a
    #: missing scope. The integration is degraded until it is reconnected.
    PROVIDER_PERMANENT = "provider_permanent"
    #: Another writer or worker got there first.
    CONCURRENCY = "concurrency"
    #: The step had already caused side effects when it failed; a retry has to
    #: reconcile them rather than repeat them.
    PARTIAL_SIDE_EFFECT = "partial_side_effect"
    #: Policy refused the action. An answer, not a failure: never retried.
    POLICY_DENIAL = "policy_denial"
    #: A human did not answer in time.
    HUMAN_TIMEOUT = "human_timeout"
    #: Raised by something outside this taxonomy.
    UNCLASSIFIED = "unclassified"


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
    #: How a step that raised this is recovered; `None` leaves it to `code`.
    failure_class: FailureClass | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        context_id: UUID | None = None,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ):
        self.message = message or self.default_message
        self.context_id = context_id or uuid4()
        self.details = details or {}
        #: Seconds the provider asked to be left alone for (its `Retry-After`),
        #: when it said. An automatic retry never comes sooner; see
        #: `personalos.domain.retry_policy`.
        self.retry_after = retry_after
        super().__init__(self.message)

    def to_envelope(self) -> dict[str, str]:
        """Transport-safe representation: stable fields only, no internals."""
        return {
            "error_code": self.code.value,
            "message": self.message,
            "context_id": str(self.context_id),
        }

    def repr_fields(self) -> dict[str, str]:
        """Extra `key='value'` pairs for `repr()`, before the message.

        For facts recovery needs that the message is the wrong place for --
        which provider to reconnect, say. Values must not be secret: a repr is
        what ends up in a checkpoint.
        """
        return {}

    def __repr__(self) -> str:
        fields = {"code": self.code.value}
        if self.failure_class is not None:
            fields["failure_class"] = self.failure_class.value
        fields.update(self.repr_fields())
        rendered = "".join(f"{key}={value!r}, " for key, value in fields.items())
        return (
            f"{type(self).__name__}({rendered}"
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
    failure_class = FailureClass.POLICY_DENIAL


class ApprovalRequiredError(PersonalOSError):
    """The action needs a human approval that was not supplied."""

    code = ErrorCode.APPROVAL_REQUIRED
    http_status = 428
    default_message = "action requires approval"
    failure_class = FailureClass.POLICY_DENIAL


class IdempotencyConflict(PersonalOSError):
    """An idempotency key could not be honored as requested."""

    code = ErrorCode.IDEMPOTENCY_CONFLICT
    http_status = 409
    default_message = "idempotency key conflict"
    failure_class = FailureClass.CONCURRENCY


class Conflict(PersonalOSError):
    """The request is valid but contradicts the resource's current state.

    Distinct from `ValidationFailed`: the same request would be accepted
    against a resource in a different state -- resuming a workflow that is
    parked on an approval, say, rather than one that has already finished.
    """

    code = ErrorCode.CONFLICT
    http_status = 409
    default_message = "request conflicts with the resource's current state"
    failure_class = FailureClass.CONCURRENCY


class ToolFailure(PersonalOSError):
    """An approved tool call reached its adapter and failed there."""

    code = ErrorCode.TOOL_FAILURE
    http_status = 502
    retryable = True
    default_message = "tool execution failed"
    failure_class = FailureClass.PROVIDER_TRANSIENT


class RetryableFailure(PersonalOSError):
    """A transient condition; the same request may succeed if retried later."""

    code = ErrorCode.RETRYABLE
    http_status = 409
    retryable = True
    default_message = "temporarily unavailable; retry later"


class ModelTransientFailure(RetryableFailure):
    """The model call timed out, was rate limited, or the model was unavailable."""

    default_message = "the model was temporarily unavailable"
    failure_class = FailureClass.MODEL_TRANSIENT


class ModelSemanticFailure(ValidationFailed):
    """The model answered with output that did not meet its contract.

    Not retried automatically -- the same prompt tends to get the same answer
    -- but a user may retry it deliberately.
    """

    default_message = "the model's output did not meet its contract"
    failure_class = FailureClass.MODEL_SEMANTIC


class ProviderPermanentFailure(PersonalOSError):
    """A provider refused in a way no retry can change: revoked OAuth, a missing scope.

    Never retried, automatically or otherwise: the integration is marked
    degraded and the user is asked to reconnect it. `provider` and `reason`
    are carried into `repr()` so that survives a checkpoint.
    """

    code = ErrorCode.TOOL_FAILURE
    http_status = 502
    default_message = "the provider rejected the request; reconnect the integration"
    failure_class = FailureClass.PROVIDER_PERMANENT

    #: Values for `reason`.
    REVOKED_AUTH = "revoked_auth"
    MISSING_SCOPE = "missing_scope"

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str,
        reason: str = REVOKED_AUTH,
        context_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.reason = reason
        super().__init__(
            message,
            context_id=context_id,
            details={"provider": provider, "reason": reason, **(details or {})},
        )

    def repr_fields(self) -> dict[str, str]:
        return {"provider": self.provider, "reason": self.reason}


class PartialSideEffect(PersonalOSError):
    """A step failed after some of its side effects had already taken effect."""

    code = ErrorCode.TOOL_FAILURE
    http_status = 502
    default_message = "the step failed after some of its actions had taken effect"
    failure_class = FailureClass.PARTIAL_SIDE_EFFECT


class HumanTimeout(PersonalOSError):
    """A human did not answer a request in time."""

    code = ErrorCode.CONFLICT
    http_status = 409
    default_message = "the request expired before anyone answered it"
    failure_class = FailureClass.HUMAN_TIMEOUT


class InternalError(PersonalOSError):
    """An unclassified failure. Used at boundaries that wrap unknown exceptions."""

    code = ErrorCode.INTERNAL
    http_status = 500
    default_message = "an internal error occurred"


__all__ = [
    "ErrorCode",
    "FailureClass",
    "PersonalOSError",
    "ValidationFailed",
    "NotFound",
    "PolicyDeniedError",
    "ApprovalRequiredError",
    "IdempotencyConflict",
    "Conflict",
    "ToolFailure",
    "RetryableFailure",
    "ModelTransientFailure",
    "ModelSemanticFailure",
    "ProviderPermanentFailure",
    "PartialSideEffect",
    "HumanTimeout",
    "InternalError",
]
