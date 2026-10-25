"""A failed workflow step as a user sees it: what broke, and what may be done about it.

`GET /v1/workflows/{id}/failures` lists these, and `POST .../retry` and
`POST .../dismiss` act on them. The rules here are the whole recovery policy,
as pure functions of what the stored rows say:

- **The failure class is read back from the stored error.** LangGraph keeps a
  failed node's exception as its `repr()`, which for a `PersonalOSError`
  carries `failure_class` (see `personalos.domain.errors`). Anything else is
  classified by its `code`, then by a short list of builtin network errors,
  and is otherwise `unclassified`.
- **Side effects outrank the error.** A step that had already sent something
  when it failed is a `partial_side_effect`, whatever it failed on: what
  matters for its retry is that the send must be replayed, not repeated.
- **A policy denial is an answer, not a failure.** It is shown as `denied`,
  and the only thing offered is to acknowledge it.
- **A provider-permanent failure is a broken integration.** Retrying cannot
  fix revoked OAuth or a missing scope, so the offer is to reconnect, and the
  retry is refused rather than allowed to loop.
- **Manual retries are bounded** by `MAX_FAILURE_ATTEMPTS`, counting the
  attempt that failed first.

A failure is identified by the stored error it was read from: the thread, the
checkpoint it was recorded against, and the task that raised it. A retry
re-runs that task at that checkpoint, so a retry that fails again overwrites
the same error and keeps the same id -- which is what lets attempts be counted.

An outbox event the worker gave up delivering -- dead-lettered after its
retries, or quarantined as malformed -- is an `EventFailure`: always open to a
retry by hand, since what stopped it (a broken handler, a bad producer) is
fixed outside the event, and delivery is idempotent on its `dedupe_key`.

Pure values, like the rest of `domain`; nothing here reads or writes.
"""

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from personalos.domain.errors import ErrorCode, FailureClass, PersonalOSError
from personalos.domain.redaction import redact, redact_text

#: Total attempts a failed step may have, the first one included.
MAX_FAILURE_ATTEMPTS = 5

#: Longest error text returned to a caller.
MAX_ERROR_LENGTH = 500

#: Namespace for failure ids; fixed, so an id is the same in every process.
_FAILURE_NAMESPACE = UUID("3c8a7f0e-6b1d-4f2a-9e57-0d4f1c2b8a61")

#: Classes a user may retry. The two left out are answers, not accidents.
RETRYABLE_CLASSES = frozenset(FailureClass) - {
    FailureClass.POLICY_DENIAL,
    FailureClass.PROVIDER_PERMANENT,
}

#: A stored error with no `failure_class` of its own, classified by its code.
_CLASS_BY_CODE: dict[str, FailureClass] = {
    ErrorCode.POLICY_DENIED.value: FailureClass.POLICY_DENIAL,
    ErrorCode.APPROVAL_REQUIRED.value: FailureClass.POLICY_DENIAL,
    ErrorCode.CONFLICT.value: FailureClass.CONCURRENCY,
    ErrorCode.IDEMPOTENCY_CONFLICT.value: FailureClass.CONCURRENCY,
    ErrorCode.TOOL_FAILURE.value: FailureClass.PROVIDER_TRANSIENT,
    ErrorCode.RETRYABLE.value: FailureClass.PROVIDER_TRANSIENT,
}

#: Builtin exceptions that mean a network call did not complete.
_TRANSIENT_BUILTINS = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "BrokenPipeError",
    }
)

#: Operation statuses meaning the call may have reached the provider and
#: either landed or may have landed.
_SIDE_EFFECT_STATUSES = frozenset({"completed", "unknown", "in_progress"})

_REPR = re.compile(r"^\s*(?P<type>[A-Za-z_][\w.]*)\((?P<body>.*)\)\s*$", re.DOTALL)
_FIELD = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
_MESSAGE = re.compile(r"message=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*$", re.DOTALL)
_CONTEXT_ID = re.compile(r"\bcontext_id=")


class FailureStatus(str, Enum):
    """Where a failure stands. Wire-visible."""

    #: Failed, and waiting on a user.
    FAILED = "failed"
    #: Policy refused the action. Never retried.
    DENIED = "denied"
    #: A retry has been handed to the worker and has not finished.
    RETRY_QUEUED = "retry_queued"
    #: Acknowledged and closed without a retry.
    DISMISSED = "dismissed"


class RecoveryAction(str, Enum):
    """What a user is offered for a failure."""

    RETRY = "retry"
    DISMISS = "dismiss"
    #: Reconnect the integration; a retry would only fail the same way.
    RECONNECT = "reconnect"


class OperationRecovery(str, Enum):
    """What a retry does about one side effect of the failed step."""

    #: It took effect. Its stored receipt is returned; nothing is sent.
    REPLAY_STORED_RECEIPT = "replay_stored_receipt"
    #: Its outcome is unknown. The provider is asked, under the operation's
    #: idempotency key, before anything could be sent again.
    RECONCILE_BEFORE_SEND = "reconcile_before_send"


@dataclass(frozen=True)
class StoredError:
    """What a stored `repr()` says about the exception it was made from."""

    error_type: str | None = None
    failure_class: FailureClass | None = None
    code: str | None = None
    provider: str | None = None
    reason: str | None = None
    message: str = ""


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FailedOperation(_Value):
    """A side effect the failed step started, from its `tool_executions` row."""

    operation_id: UUID
    tool_name: str
    idempotency_key: str
    #: `tool_executions.status`: completed, failed, unknown or in_progress.
    status: str
    #: Provider calls made for it so far.
    attempts: int
    recovery: OperationRecovery
    #: The stored receipt, redacted; present when the operation completed.
    receipt: dict[str, Any] | None = None
    error: str | None = None


class IntegrationHealth(_Value):
    """The state of the integration a provider-permanent failure broke."""

    provider: str
    #: `healthy` or `degraded`.
    status: str
    reason: str | None = None
    reconnect_required: bool = False
    updated_at: datetime | None = None


class WorkflowFailure(_Value):
    """One failed step of a workflow, and what may be done about it."""

    failure_id: UUID
    workflow_id: UUID
    thread_id: str
    #: The node that raised, when the stored task path names one.
    step: str | None = None
    failure_class: FailureClass
    #: What the error itself was classed as, when the step's side effects
    #: made it a `partial_side_effect` instead.
    cause_class: FailureClass | None = None
    status: FailureStatus
    retryable: bool
    #: Why `retryable` is false; absent when it is true.
    not_retryable_reason: str | None = None
    actions: list[RecoveryAction] = Field(default_factory=list)
    #: The stored error text, redacted and truncated.
    last_error: str
    error_type: str | None = None
    #: Attempts at this step so far: the first, plus every retry queued.
    attempts: int
    max_attempts: int = MAX_FAILURE_ATTEMPTS
    operations: list[FailedOperation] = Field(default_factory=list)
    integration: IntegrationHealth | None = None
    #: When the checkpoint the step started from was written.
    failed_at: datetime | None = None
    dismissed_by: str | None = None
    dismissed_at: datetime | None = None


def failure_id(thread_id: str, checkpoint_id: str | None, task_id: str | None) -> UUID:
    """The stable id of the error a task recorded against a checkpoint."""
    return uuid5(_FAILURE_NAMESPACE, f"{thread_id}|{checkpoint_id or ''}|{task_id or ''}")


def parse_stored_error(text: str) -> StoredError:
    """Read a stored exception `repr()` back into the facts recovery needs.

    Tolerant by design: a repr this does not recognise still yields its text
    as the message, and classifies as `unclassified`.
    """
    match = _REPR.match(text or "")
    if match is None:
        return StoredError(message=text or "")
    error_type, body = match.group("type"), match.group("body")

    if _CONTEXT_ID.search(body):
        # A `PersonalOSError`: `key='value'` fields, then context_id, then the
        # message. Fields are read only from before context_id, so a message
        # that happens to contain "provider='x'" cannot set one.
        head = _CONTEXT_ID.split(body, maxsplit=1)[0]
        fields = {key: _literal(value) for key, value in _FIELD.findall(head)}
        message_match = _MESSAGE.search(body)
        message = _literal(message_match.group(1)) if message_match else body
        return StoredError(
            error_type=error_type,
            failure_class=_failure_class(fields.get("failure_class")),
            code=fields.get("code"),
            provider=fields.get("provider"),
            reason=fields.get("reason"),
            message=message,
        )

    try:
        args = ast.literal_eval(f"({body},)") if body.strip() else ()
        message = "; ".join(str(arg) for arg in args)
    except (ValueError, SyntaxError):
        message = body
    return StoredError(error_type=error_type, message=message)


def classify(error: StoredError) -> FailureClass:
    """The failure class of a stored error, on its own terms."""
    if error.failure_class is not None:
        return error.failure_class
    if error.code in _CLASS_BY_CODE:
        return _CLASS_BY_CODE[error.code]
    if error.error_type and error.error_type.rsplit(".", 1)[-1] in _TRANSIENT_BUILTINS:
        return FailureClass.PROVIDER_TRANSIENT
    return FailureClass.UNCLASSIFIED


def sanitize_error(error: StoredError) -> str:
    """The error text fit to hand a caller: redacted, one line, bounded."""
    text = error.message or ""
    if error.error_type and error.failure_class is None and error.code is None:
        # A builtin's message alone ("timed out") does not say what timed out.
        text = f"{error.error_type}: {text}" if text else error.error_type
    text = " ".join(redact_text(text).split())
    if len(text) > MAX_ERROR_LENGTH:
        text = text[: MAX_ERROR_LENGTH - 1].rstrip() + "…"
    return text


def describe_exception(error: BaseException) -> str:
    """A live exception as `sanitize_error` would show its stored repr."""
    if isinstance(error, PersonalOSError):
        return sanitize_error(parse_stored_error(repr(error)))
    return sanitize_error(StoredError(error_type=type(error).__name__, message=str(error)))


def operation_recovery(status: str) -> OperationRecovery:
    """What a retry does about an operation in `status`."""
    if status == "completed":
        return OperationRecovery.REPLAY_STORED_RECEIPT
    return OperationRecovery.RECONCILE_BEFORE_SEND


def assess_failure(
    *,
    workflow_id: UUID,
    thread_id: str,
    step: str | None,
    checkpoint_id: str | None,
    task_id: str | None,
    stored_error: str,
    failed_at: datetime | None = None,
    operations: list[FailedOperation] | None = None,
    integration: IntegrationHealth | None = None,
    retries_queued: int = 0,
    retry_pending: bool = False,
    dismissed_by: str | None = None,
    dismissed_at: datetime | None = None,
) -> WorkflowFailure:
    """Decide everything about one failure from what the rows say about it.

    `retries_queued` counts every retry ever queued for this failure, finished
    or not; `retry_pending` is whether one is still waiting on the worker.
    """
    error = parse_stored_error(stored_error)
    cause = classify(error)
    operations = list(operations or [])

    failure_class = cause
    if cause not in (FailureClass.POLICY_DENIAL, FailureClass.PROVIDER_PERMANENT) and any(
        operation.status in _SIDE_EFFECT_STATUSES for operation in operations
    ):
        failure_class = FailureClass.PARTIAL_SIDE_EFFECT

    if integration is None and failure_class == FailureClass.PROVIDER_PERMANENT and error.provider:
        # The integration row is written by the worker that saw the failure;
        # until it has been, the stored error is enough to say which one broke.
        integration = IntegrationHealth(
            provider=error.provider,
            status="degraded",
            reason=error.reason,
            reconnect_required=True,
        )

    attempts = 1 + retries_queued
    status, retryable, reason, actions = _disposition(
        failure_class=failure_class,
        attempts=attempts,
        retry_pending=retry_pending,
        dismissed=dismissed_by is not None,
        integration=integration,
    )
    return WorkflowFailure(
        failure_id=failure_id(thread_id, checkpoint_id, task_id),
        workflow_id=workflow_id,
        thread_id=thread_id,
        step=step,
        failure_class=failure_class,
        cause_class=cause if cause != failure_class else None,
        status=status,
        retryable=retryable,
        not_retryable_reason=reason,
        actions=actions,
        last_error=sanitize_error(error),
        error_type=error.error_type,
        attempts=attempts,
        operations=operations,
        integration=integration,
        failed_at=failed_at,
        dismissed_by=dismissed_by,
        dismissed_at=dismissed_at,
    )


class EventDisposition(str, Enum):
    """Why an outbox event stopped being delivered. Wire-visible."""

    #: Its automatic retries ran out, or it failed in a way none could fix.
    DEAD_LETTER = "dead_letter"
    #: Its payload did not meet the event schema; it was never retried.
    QUARANTINED = "quarantined"


class EventFailure(_Value):
    """An outbox event the worker gave up delivering, and what may be done about it.

    Recoverable by an operator: a retry puts the event back in line with a
    fresh attempt budget, for after the handler -- or, for a quarantined
    event, whatever produced it -- has been fixed. Delivery is idempotent on
    the event's `dedupe_key`, so a retry of an event that was in fact applied
    applies nothing twice.
    """

    #: The `outbox_events` row id.
    failure_id: UUID
    event_id: UUID
    event_type: str
    dedupe_key: str | None = None
    #: The workflow the event belongs to, when it could be told.
    workflow_id: UUID | None = None
    disposition: EventDisposition
    status: FailureStatus = FailureStatus.FAILED
    retryable: bool = True
    actions: list[RecoveryAction] = Field(default_factory=lambda: [RecoveryAction.RETRY])
    #: The last delivery error, redacted and truncated.
    last_error: str
    #: Deliveries attempted since it was staged or last retried by hand.
    attempts: int
    created_at: datetime | None = None
    failed_at: datetime | None = None


def assess_event_failure(
    *,
    outbox_id: UUID,
    event_id: UUID | None,
    event_type: str,
    status: str,
    last_error: str | None,
    attempts: int,
    dedupe_key: str | None = None,
    workflow_id: UUID | None = None,
    created_at: datetime | None = None,
    failed_at: datetime | None = None,
) -> EventFailure:
    """An outbox row given up on, as a recoverable failure."""
    disposition = (
        EventDisposition.QUARANTINED
        if status == EventDisposition.QUARANTINED.value
        else EventDisposition.DEAD_LETTER
    )
    text = " ".join(redact_text(last_error or "").split()) or "no error was recorded"
    if len(text) > MAX_ERROR_LENGTH:
        text = text[: MAX_ERROR_LENGTH - 1].rstrip() + "…"
    return EventFailure(
        failure_id=outbox_id,
        event_id=event_id or outbox_id,
        event_type=event_type,
        dedupe_key=dedupe_key,
        workflow_id=workflow_id,
        disposition=disposition,
        last_error=text,
        attempts=attempts,
        created_at=created_at,
        failed_at=failed_at,
    )


def redacted_receipt(receipt: Any) -> dict[str, Any] | None:
    """A stored receipt as it may be shown."""
    return redact(receipt) if isinstance(receipt, dict) else None


def _disposition(
    *,
    failure_class: FailureClass,
    attempts: int,
    retry_pending: bool,
    dismissed: bool,
    integration: IntegrationHealth | None,
) -> tuple[FailureStatus, bool, str | None, list[RecoveryAction]]:
    if dismissed:
        return FailureStatus.DISMISSED, False, "the failure was dismissed", []
    if failure_class == FailureClass.POLICY_DENIAL:
        return (
            FailureStatus.DENIED,
            False,
            "policy denied this action; asking again gets the same answer",
            [RecoveryAction.DISMISS],
        )
    if failure_class == FailureClass.PROVIDER_PERMANENT:
        provider = integration.provider if integration else "the integration"
        return (
            FailureStatus.FAILED,
            False,
            f"{provider} rejected the request permanently; reconnect it instead of retrying",
            [RecoveryAction.RECONNECT, RecoveryAction.DISMISS],
        )
    if retry_pending:
        return FailureStatus.RETRY_QUEUED, False, "a retry is already queued", []
    if attempts >= MAX_FAILURE_ATTEMPTS:
        return (
            FailureStatus.FAILED,
            False,
            f"the step has failed {attempts} times, the most allowed",
            [RecoveryAction.DISMISS],
        )
    return FailureStatus.FAILED, True, None, [RecoveryAction.RETRY, RecoveryAction.DISMISS]


def _literal(quoted: str) -> str:
    try:
        value = ast.literal_eval(quoted)
    except (ValueError, SyntaxError):
        return quoted.strip("'\"")
    return value if isinstance(value, str) else str(value)


def _failure_class(value: str | None) -> FailureClass | None:
    try:
        return FailureClass(value) if value else None
    except ValueError:
        return None


__all__ = [
    "MAX_ERROR_LENGTH",
    "MAX_FAILURE_ATTEMPTS",
    "RETRYABLE_CLASSES",
    "EventDisposition",
    "EventFailure",
    "FailedOperation",
    "FailureStatus",
    "IntegrationHealth",
    "OperationRecovery",
    "RecoveryAction",
    "StoredError",
    "WorkflowFailure",
    "assess_event_failure",
    "assess_failure",
    "classify",
    "describe_exception",
    "failure_id",
    "operation_recovery",
    "parse_stored_error",
    "redacted_receipt",
    "sanitize_error",
]
