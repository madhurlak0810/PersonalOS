"""The tool boundary: ports plus the policy-enforcing gateway.

Every tool call in PersonalOS goes through a :class:`ToolGateway`. The gateway
is the only component that talks to both the policy layer and a tool adapter,
which is what lets the rule hold everywhere else: orchestration and execution
code can express *intents* but cannot reach a tool without a decision.

Two ports are defined here:

``ToolInvoker``
    Implemented by adapters (see ``personalos.mcp.adapter``). It accepts only
    an :class:`~personalos.policy.intents.ApprovedIntent`, so an adapter cannot
    be driven from raw arguments even by accident.

``ToolGateway``
    Depended on by executors. Executors hold this type, never a concrete
    adapter or server manager.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, Field

from personalos.domain.errors import (
    ErrorCode,
    IdempotencyConflict,
    InternalError,
    NotFound,
    PersonalOSError,
    ToolFailure,
    ValidationFailed,
)
from personalos.policy import (
    ApprovalGrant,
    ApprovedIntent,
    PolicyEngine,
    PolicyViolation,
    ToolIntent,
)
from personalos.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


#: Maps the generic `ErrorCode` an adapter attached to a failed payload onto
#: the specific exception `unwrap()` raises for it. An adapter that cannot (or
#: does not) classify its failure this precisely -- or reports `TOOL_FAILURE`
#: itself -- falls back to :class:`ToolExecutionError` below.
_ERROR_CODE_TO_EXCEPTION: dict[str, type[PersonalOSError]] = {
    ErrorCode.VALIDATION.value: ValidationFailed,
    ErrorCode.NOT_FOUND.value: NotFound,
    ErrorCode.IDEMPOTENCY_CONFLICT.value: IdempotencyConflict,
    ErrorCode.INTERNAL.value: InternalError,
}


class ToolResult(BaseModel):
    """Outcome of one approved tool call, tied back to its intent."""

    intent_id: UUID
    tool_ref: str
    success: bool
    result: Any | None = None
    error: str | None = None
    #: A `personalos.domain.errors.ErrorCode` value, when the adapter could
    #: classify the failure that precisely. `None` for a success, and for a
    #: failure the adapter reported only as an opaque string.
    error_code: str | None = None
    replayed: bool = False
    rule: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_adapter_payload(
        cls, approved: ApprovedIntent, payload: dict[str, Any]
    ) -> "ToolResult":
        """Normalize the ``{success, result, error, error_code}`` dict adapters return."""
        return cls(
            intent_id=approved.intent.intent_id,
            tool_ref=approved.intent.tool_ref,
            success=bool(payload.get("success")),
            result=payload.get("result"),
            error=payload.get("error"),
            error_code=payload.get("error_code"),
            replayed=bool(payload.get("replayed", False)),
            rule=approved.decision.rule,
        )

    def unwrap(self) -> Any:
        """Return the result, raising a mapped :class:`PersonalOSError` on failure.

        The exception class reflects `error_code` when the adapter set one to
        something more specific than a plain tool failure (an MCP failure
        carries its `ToolCallErrorCode`, translated by the adapter into this
        shared taxonomy); otherwise falls back to :class:`ToolExecutionError`.
        """
        if not self.success:
            exc_cls = _ERROR_CODE_TO_EXCEPTION.get(self.error_code)
            if exc_cls is not None:
                raise exc_cls(
                    f"tool '{self.tool_ref}' failed: {self.error or 'unknown error'}",
                    details={"tool_ref": self.tool_ref},
                )
            raise ToolExecutionError(self.tool_ref, self.error or "unknown error")
        return self.result


class ToolExecutionError(ToolFailure):
    """An approved tool call failed inside the adapter with no finer classification."""

    def __init__(self, tool_ref: str, error: str):
        self.tool_ref = tool_ref
        self.error = error
        super().__init__(
            f"tool '{tool_ref}' failed: {error}",
            details={"tool_ref": tool_ref, "error": error},
        )


@runtime_checkable
class ToolInvoker(Protocol):
    """Port for anything that can carry out an approved intent."""

    async def invoke(self, approved: ApprovedIntent) -> dict[str, Any]:
        """Run the approved intent, returning an adapter payload dict."""
        ...


class ToolGateway(ABC):
    """Port executors depend on: submit an intent, get a result."""

    @abstractmethod
    async def dispatch(
        self,
        intent: ToolIntent,
        approval: ApprovalGrant | None = None,
    ) -> ToolResult:
        """Authorize the intent, then execute it if policy permits."""


class PolicyEnforcingToolGateway(ToolGateway):
    """The production gateway: policy first, adapter second.

    Denials are raised rather than folded into a failed ``ToolResult``. A
    blocked tool call is a control-flow event the caller must handle, not an
    ordinary error it might skip past while continuing to act.
    """

    def __init__(self, policy: PolicyEngine, invoker: ToolInvoker):
        """Wire a policy engine to a tool adapter."""
        self.policy = policy
        self.invoker = invoker

    async def dispatch(
        self,
        intent: ToolIntent,
        approval: ApprovalGrant | None = None,
    ) -> ToolResult:
        """Authorize then execute.

        Raises :class:`~personalos.policy.errors.PolicyDenied` or
        :class:`~personalos.policy.errors.ApprovalRequired` if the intent does
        not clear policy.
        """
        if not isinstance(intent, ToolIntent):
            raise PolicyViolation(
                f"gateway accepts ToolIntent only, got {type(intent).__name__}; "
                f"raw tool arguments cannot be dispatched"
            )

        approved = self.policy.authorize(intent, approval)
        return await self.execute_approved(approved)

    async def execute_approved(self, approved: ApprovedIntent) -> ToolResult:
        """Execute an already-approved intent.

        Separate from ``dispatch`` so a queued approval can be redeemed later
        without re-deriving it, and so the type check below is the only way in.
        """
        if not isinstance(approved, ApprovedIntent):
            raise PolicyViolation(
                f"expected an ApprovedIntent, got {type(approved).__name__}; "
                f"execution requires a policy decision"
            )

        logger.info(
            "executing %s (intent=%s, rule=%s)",
            approved.intent.tool_ref,
            approved.intent.intent_id,
            approved.decision.rule,
        )
        payload = await self.invoker.invoke(approved)
        return ToolResult.from_adapter_payload(approved, payload)


class ToolRegistryInvoker:
    """Adapter that satisfies :class:`ToolInvoker` from a local registry.

    Used for in-process tools that are not backed by an MCP server. The
    registry itself takes raw arguments, so this wrapper is the only place
    allowed to call it.
    """

    def __init__(self, registry: ToolRegistry):
        """Take the registry whose tools this invoker exposes."""
        self.registry = registry

    async def invoke(self, approved: ApprovedIntent) -> dict[str, Any]:
        """Execute the approved intent against the registry."""
        return await self.registry.execute(approved.tool, **approved.arguments)


__all__ = [
    "ToolResult",
    "ToolExecutionError",
    "ToolInvoker",
    "ToolGateway",
    "PolicyEnforcingToolGateway",
    "ToolRegistryInvoker",
]
