"""Intents: the only currency an executor is allowed to act on.

A :class:`ToolIntent` is a *proposal*. It may have been assembled by
deterministic orchestration code or emitted by a language model, so nothing in
the system may treat it as trustworthy. An :class:`ApprovedIntent` is a
proposal that the policy engine has cleared, and it is the only thing an
executor will run.

The distinction is enforced at construction time: an ``ApprovedIntent`` cannot
be built outside this module, so no amount of convenience code in the graph or
executor layers can quietly manufacture an approval.
"""

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from personalos.policy.errors import InvalidApproval, PolicyViolation


class IntentOrigin(str, Enum):
    """Where an intent came from, which determines how much it is trusted."""

    SYSTEM = "system"  # built by deterministic orchestration code
    USER = "user"  # requested directly by the human operator
    LLM = "llm"  # proposed by a model; never trusted on its own


class Decision(str, Enum):
    """Outcome of evaluating an intent against policy."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


def fingerprint_intent(server: str, tool: str, arguments: dict[str, Any]) -> str:
    """Hash the identity of a side effect.

    Canonical JSON keeps the hash stable across key ordering; unserializable
    values fall back to their repr so fingerprinting never raises and blocks an
    otherwise valid intent. Deliberately duplicated rather than reused from
    ``personalos.persistence``: the policy layer must not depend on storage.
    """
    payload = json.dumps(
        {"server": server, "tool": tool, "arguments": arguments},
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolIntent(BaseModel):
    """A proposed tool call. Untrusted until the policy engine clears it."""

    intent_id: UUID = Field(default_factory=uuid4)
    server: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    origin: IntentOrigin = IntentOrigin.SYSTEM
    mutating: bool = False

    # Provenance, for audit trails and for rules that key off the caller.
    requested_by: str = "unknown"
    job_id: UUID | None = None
    agent_id: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def tool_ref(self) -> str:
        """Fully-qualified tool name, e.g. ``jobs.search_jobs``."""
        return f"{self.server}.{self.tool}"

    def fingerprint(self) -> str:
        """Stable hash of the side effect this intent describes."""
        return fingerprint_intent(self.server, self.tool, self.arguments)

    class Config:
        use_enum_values = False


class PolicyDecision(BaseModel):
    """The verdict on one intent, retained for audit."""

    intent_id: UUID
    tool_ref: str
    decision: Decision
    rule: str
    reason: str
    decided_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def allowed(self) -> bool:
        """True only for an outright allow."""
        return self.decision == Decision.ALLOW

    class Config:
        use_enum_values = False


class ApprovalGrant(BaseModel):
    """Evidence that a human approved one specific intent.

    Bound to the fingerprint of the intent so a grant cannot be replayed
    against a different set of arguments.
    """

    intent_id: UUID
    intent_fingerprint: str
    approved_by: str
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    note: str | None = None

    def matches(self, intent: ToolIntent) -> bool:
        """True when this grant was issued for exactly this intent."""
        return (
            self.intent_id == intent.intent_id
            and self.intent_fingerprint == intent.fingerprint()
        )


# Only code inside this module holds the mint token, so an ApprovedIntent can
# only be produced through mint_approved_intent, which the policy engine calls
# after a rule chain has actually cleared the intent.
_MINT_TOKEN = object()


class ApprovedIntent:
    """A policy-cleared intent. The only thing an executor will run.

    Instances are immutable and cannot be constructed directly: attempting to
    do so raises :class:`PolicyViolation`. Obtain one from
    ``PolicyEngine.authorize``.
    """

    __slots__ = ("_intent", "_decision", "_approval", "_approved_at")

    def __init__(
        self,
        intent: ToolIntent,
        decision: PolicyDecision,
        approval: ApprovalGrant | None = None,
        *,
        _token: Any = None,
    ):
        if _token is not _MINT_TOKEN:
            raise PolicyViolation(
                "ApprovedIntent cannot be constructed directly; obtain one from "
                "PolicyEngine.authorize() so the policy layer is not bypassed"
            )
        object.__setattr__(self, "_intent", intent)
        object.__setattr__(self, "_decision", decision)
        object.__setattr__(self, "_approval", approval)
        object.__setattr__(self, "_approved_at", datetime.utcnow())

    @property
    def intent(self) -> ToolIntent:
        """The intent that was cleared."""
        return self._intent

    @property
    def decision(self) -> PolicyDecision:
        """The decision that cleared it."""
        return self._decision

    @property
    def approval(self) -> ApprovalGrant | None:
        """The human grant, when the decision required one."""
        return self._approval

    @property
    def approved_at(self) -> datetime:
        """When the approval was minted."""
        return self._approved_at

    # Convenience passthroughs so adapters never reach for raw kwargs.
    @property
    def server(self) -> str:
        """Target server name."""
        return self._intent.server

    @property
    def tool(self) -> str:
        """Target tool name."""
        return self._intent.tool

    @property
    def arguments(self) -> dict[str, Any]:
        """Arguments as cleared by policy."""
        return dict(self._intent.arguments)

    def __setattr__(self, name: str, value: Any):
        raise PolicyViolation("ApprovedIntent is immutable once minted")

    def __repr__(self) -> str:
        return (
            f"ApprovedIntent(tool_ref={self._intent.tool_ref!r}, "
            f"intent_id={self._intent.intent_id!s}, rule={self._decision.rule!r})"
        )


def mint_approved_intent(
    intent: ToolIntent,
    decision: PolicyDecision,
    approval: ApprovalGrant | None = None,
) -> ApprovedIntent:
    """Mint an approval. Internal to the policy layer; callers use the engine.

    Re-checks the decision so a mistake in a rule chain cannot turn a denial
    into an approval, and re-checks that any grant belongs to this intent.
    """
    if decision.intent_id != intent.intent_id:
        raise InvalidApproval(
            f"decision {decision.intent_id} does not belong to intent {intent.intent_id}"
        )
    if decision.decision == Decision.DENY:
        raise PolicyViolation(f"cannot approve a denied intent: {decision.reason}")
    if decision.decision == Decision.REQUIRE_APPROVAL:
        if approval is None:
            raise InvalidApproval("decision requires an approval grant, none supplied")
        if not approval.matches(intent):
            raise InvalidApproval("approval grant does not match this intent")
    return ApprovedIntent(intent, decision, approval, _token=_MINT_TOKEN)


__all__ = [
    "IntentOrigin",
    "Decision",
    "ToolIntent",
    "PolicyDecision",
    "ApprovalGrant",
    "ApprovedIntent",
    "fingerprint_intent",
    "mint_approved_intent",
]
