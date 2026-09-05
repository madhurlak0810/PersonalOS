"""Errors raised when the policy boundary is crossed incorrectly.

These subclass :class:`~personalos.domain.errors.PersonalOSError` (the policy
layer is allowed to depend on `domain`, see
`tests/architecture/boundaries.py`) so a policy failure reports through the
same `error_code` taxonomy as every other layer, while keeping the richer
`decision`-carrying constructors callers here already depend on.
"""

from typing import TYPE_CHECKING

from personalos.domain.errors import ErrorCode, PersonalOSError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from personalos.policy.intents import PolicyDecision


class PolicyError(PersonalOSError):
    """Base class for all policy failures."""


class PolicyDenied(PolicyError):
    """The policy engine refused the intent outright."""

    code = ErrorCode.POLICY_DENIED
    http_status = 403

    def __init__(self, decision: "PolicyDecision"):
        self.decision = decision
        super().__init__(
            f"policy denied intent '{decision.tool_ref}' "
            f"(rule={decision.rule}): {decision.reason}",
            details={"rule": decision.rule, "tool_ref": decision.tool_ref},
        )


class ApprovalRequired(PolicyError):
    """The intent needs a human approval that was not supplied."""

    code = ErrorCode.APPROVAL_REQUIRED
    http_status = 428

    def __init__(self, decision: "PolicyDecision"):
        self.decision = decision
        super().__init__(
            f"intent '{decision.tool_ref}' requires approval "
            f"(rule={decision.rule}): {decision.reason}",
            details={"rule": decision.rule, "tool_ref": decision.tool_ref},
        )


class PolicyViolation(PolicyError):
    """A caller tried to bypass the policy layer.

    Raised when code attempts to fabricate an approval rather than obtain one
    from the policy engine, or hands an executor something other than an
    approved intent. Left at the base class's `INTERNAL` code: this signals a
    programming error in this codebase, not a normal user-facing denial.
    """


class InvalidApproval(PolicyError):
    """The supplied approval does not match the intent it was presented with."""

    code = ErrorCode.VALIDATION
    http_status = 400


__all__ = [
    "PolicyError",
    "PolicyDenied",
    "ApprovalRequired",
    "PolicyViolation",
    "InvalidApproval",
]
