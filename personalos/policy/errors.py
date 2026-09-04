"""Errors raised when the policy boundary is crossed incorrectly."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from personalos.policy.intents import PolicyDecision


class PolicyError(Exception):
    """Base class for all policy failures."""


class PolicyDenied(PolicyError):
    """The policy engine refused the intent outright."""

    def __init__(self, decision: "PolicyDecision"):
        self.decision = decision
        super().__init__(
            f"policy denied intent '{decision.tool_ref}' "
            f"(rule={decision.rule}): {decision.reason}"
        )


class ApprovalRequired(PolicyError):
    """The intent needs a human approval that was not supplied."""

    def __init__(self, decision: "PolicyDecision"):
        self.decision = decision
        super().__init__(
            f"intent '{decision.tool_ref}' requires approval "
            f"(rule={decision.rule}): {decision.reason}"
        )


class PolicyViolation(PolicyError):
    """A caller tried to bypass the policy layer.

    Raised when code attempts to fabricate an approval rather than obtain one
    from the policy engine, or hands an executor something other than an
    approved intent.
    """


class InvalidApproval(PolicyError):
    """The supplied approval does not match the intent it was presented with."""


__all__ = [
    "PolicyError",
    "PolicyDenied",
    "ApprovalRequired",
    "PolicyViolation",
    "InvalidApproval",
]
