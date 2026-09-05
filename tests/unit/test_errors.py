"""Tests for the shared error taxonomy and its transport-safe envelope.

Covers `personalos.domain.errors` directly, plus the mapping at each layer
boundary that is supposed to report through it: policy, persistence
(idempotency), the tool gateway, and the MCP adapter. The API-level envelope
(status codes, sanitization of unclassified failures) is covered in
`test_api_routes.py`.
"""

from uuid import UUID, uuid4

import pytest

from personalos.domain.errors import (
    ErrorCode,
    IdempotencyConflict,
    InternalError,
    NotFound,
    PersonalOSError,
    PolicyDeniedError,
    RetryableFailure,
    ToolFailure,
    ValidationFailed,
)
from personalos.domain.models import (
    ActionTarget,
    InvalidIdempotencyKey,
    ToolCallErrorCode,
)
from personalos.mcp.adapter import _ERROR_CODE_MAP
from personalos.persistence.idempotency import (
    IdempotencyError,
    IdempotencyKeyReused,
    OperationInProgress,
)
from personalos.policy.errors import (
    ApprovalRequired,
    InvalidApproval,
    PolicyDenied,
    PolicyError,
    PolicyViolation,
)
from personalos.policy.intents import Decision, PolicyDecision
from personalos.tools.gateway import ToolExecutionError, ToolResult


def a_decision(decision: Decision = Decision.DENY) -> PolicyDecision:
    """A minimal, well-formed decision to construct policy errors from."""
    return PolicyDecision(
        intent_id=uuid4(),
        tool_ref="jobs.rm_rf",
        decision=decision,
        rule="tool_allowlist",
        reason="not on the allowlist",
    )


# ---------------------------------------------------------------------------
# Envelope shape
# ---------------------------------------------------------------------------


def test_envelope_has_exactly_the_stable_fields():
    """No internal detail (details dict, args, traceback) leaks into the envelope."""
    error = ValidationFailed("bad input", details={"field": "title", "secret": "s3cr3t"})

    envelope = error.to_envelope()

    assert set(envelope) == {"error_code", "message", "context_id"}
    assert envelope["error_code"] == "validation"
    assert envelope["message"] == "bad input"
    assert UUID(envelope["context_id"]) == error.context_id
    assert "s3cr3t" not in str(envelope)


def test_context_id_is_generated_when_not_supplied():
    """Every instance gets its own correlation id even with no explicit one."""
    first = InternalError()
    second = InternalError()
    assert first.context_id != second.context_id


def test_default_message_is_used_when_none_given():
    assert NotFound().message == "resource not found"


@pytest.mark.parametrize(
    "error_cls,expected_code,expected_retryable",
    [
        (ValidationFailed, ErrorCode.VALIDATION, False),
        (NotFound, ErrorCode.NOT_FOUND, False),
        (PolicyDeniedError, ErrorCode.POLICY_DENIED, False),
        (IdempotencyConflict, ErrorCode.IDEMPOTENCY_CONFLICT, False),
        (ToolFailure, ErrorCode.TOOL_FAILURE, True),
        (RetryableFailure, ErrorCode.RETRYABLE, True),
        (InternalError, ErrorCode.INTERNAL, False),
    ],
)
def test_each_subclass_fixes_its_own_code(error_cls, expected_code, expected_retryable):
    """The code/retryable pairing is a class property, not per-instance state."""
    error = error_cls()
    assert error.code is expected_code
    assert error.retryable is expected_retryable
    assert isinstance(error, PersonalOSError)


# ---------------------------------------------------------------------------
# Cross-layer consistency: every layer's specific exceptions report through
# the same taxonomy rather than each inventing its own error shape.
# ---------------------------------------------------------------------------


def test_domain_invalid_idempotency_key_reports_as_validation():
    error = InvalidIdempotencyKey("too short")
    assert isinstance(error, PersonalOSError)
    assert isinstance(error, ValueError)  # original base, kept for compatibility
    assert error.code is ErrorCode.VALIDATION
    assert error.to_envelope()["error_code"] == "validation"


def test_policy_denied_reports_as_policy_denied():
    error = PolicyDenied(a_decision(Decision.DENY))
    assert isinstance(error, PersonalOSError)
    assert error.code is ErrorCode.POLICY_DENIED
    assert error.http_status == 403
    assert error.decision.rule == "tool_allowlist"


def test_approval_required_reports_as_approval_required():
    error = ApprovalRequired(a_decision(Decision.REQUIRE_APPROVAL))
    assert isinstance(error, PersonalOSError)
    assert error.code is ErrorCode.APPROVAL_REQUIRED
    assert error.http_status == 428


def test_policy_violation_stays_an_internal_error():
    """A bypass attempt is a bug in this codebase, not a normal denial."""
    error = PolicyViolation("fabricated approval")
    assert isinstance(error, PersonalOSError)
    assert error.code is ErrorCode.INTERNAL


def test_invalid_approval_reports_as_validation():
    error = InvalidApproval("grant does not match intent")
    assert isinstance(error, PersonalOSError)
    assert error.code is ErrorCode.VALIDATION


def test_policy_error_hierarchy_is_intact():
    assert issubclass(PolicyDenied, PolicyError)
    assert issubclass(ApprovalRequired, PolicyError)
    assert issubclass(PolicyError, PersonalOSError)


def test_idempotency_key_reused_reports_as_idempotency_conflict():
    error = IdempotencyKeyReused("key already used for a different request")
    assert isinstance(error, PersonalOSError)
    assert isinstance(error, IdempotencyError)
    assert error.code is ErrorCode.IDEMPOTENCY_CONFLICT


def test_operation_in_progress_is_retryable():
    """Unlike a reused key, a concurrent in-flight operation is worth retrying."""
    error = OperationInProgress("operation still running")
    assert error.code is ErrorCode.RETRYABLE
    assert error.retryable is True


def test_tool_execution_error_reports_as_tool_failure():
    error = ToolExecutionError("jobs.search_jobs", "upstream timeout")
    assert isinstance(error, PersonalOSError)
    assert isinstance(error, ToolFailure)
    assert error.code is ErrorCode.TOOL_FAILURE
    assert error.retryable is True
    assert error.tool_ref == "jobs.search_jobs"


def test_tool_result_unwrap_maps_error_code_to_the_matching_exception():
    """A gateway result carrying `error_code="not_found"` raises `NotFound`, not a generic failure."""
    result = ToolResult(
        intent_id=uuid4(),
        tool_ref="jobs.search_jobs",
        success=False,
        error="no such server",
        error_code=ErrorCode.NOT_FOUND.value,
    )

    with pytest.raises(NotFound):
        result.unwrap()


def test_tool_result_unwrap_falls_back_without_a_specific_error_code():
    """An adapter that reports only a string still raises a `PersonalOSError`."""
    result = ToolResult(
        intent_id=uuid4(),
        tool_ref="jobs.search_jobs",
        success=False,
        error="upstream timeout",
        error_code=None,
    )

    with pytest.raises(ToolExecutionError) as excinfo:
        result.unwrap()
    assert excinfo.value.code is ErrorCode.TOOL_FAILURE


def test_mcp_adapter_maps_every_tool_call_error_code():
    """Every `ToolCallErrorCode` the MCP layer can produce has a taxonomy mapping.

    A code left unmapped would silently fall back to `INTERNAL` at the
    adapter, hiding a real client- or not-found-shaped failure behind a
    generic 500 -- this pins the mapping so a newly added `ToolCallErrorCode`
    forces a decision here too.
    """
    assert set(_ERROR_CODE_MAP) == set(ToolCallErrorCode)


@pytest.mark.parametrize(
    "tool_call_code,expected",
    [
        (ToolCallErrorCode.SERVER_NOT_FOUND, ErrorCode.NOT_FOUND),
        (ToolCallErrorCode.TOOL_NOT_FOUND, ErrorCode.NOT_FOUND),
        (ToolCallErrorCode.VALIDATION_ERROR, ErrorCode.VALIDATION),
        (ToolCallErrorCode.IDEMPOTENCY_CONFLICT, ErrorCode.IDEMPOTENCY_CONFLICT),
        (ToolCallErrorCode.EXECUTION_ERROR, ErrorCode.TOOL_FAILURE),
    ],
)
def test_mcp_adapter_maps_specific_codes_as_expected(tool_call_code, expected):
    assert _ERROR_CODE_MAP[tool_call_code] is expected


def test_action_target_is_unaffected_by_error_wiring():
    """Sanity check that unrelated domain models still construct normally."""
    target = ActionTarget(server="jobs", tool="search_jobs")
    assert target.tool == "search_jobs"
