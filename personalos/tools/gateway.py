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

from personalos.policy import (
    ApprovalGrant,
    ApprovedIntent,
    PolicyEngine,
    PolicyViolation,
    ToolIntent,
)
from personalos.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Outcome of one approved tool call, tied back to its intent."""

    intent_id: UUID
    tool_ref: str
    success: bool
    result: Any | None = None
    error: str | None = None
    replayed: bool = False
    rule: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_adapter_payload(
        cls, approved: ApprovedIntent, payload: dict[str, Any]
    ) -> "ToolResult":
        """Normalize the ``{success, result, error}`` dict adapters return."""
        return cls(
            intent_id=approved.intent.intent_id,
            tool_ref=approved.intent.tool_ref,
            success=bool(payload.get("success")),
            result=payload.get("result"),
            error=payload.get("error"),
            replayed=bool(payload.get("replayed", False)),
            rule=approved.decision.rule,
        )

    def unwrap(self) -> Any:
        """Return the result, raising :class:`ToolExecutionError` on failure."""
        if not self.success:
            raise ToolExecutionError(self.tool_ref, self.error or "unknown error")
        return self.result


class ToolExecutionError(RuntimeError):
    """An approved tool call failed inside the adapter."""

    def __init__(self, tool_ref: str, error: str):
        self.tool_ref = tool_ref
        self.error = error
        super().__init__(f"tool '{tool_ref}' failed: {error}")


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
