"""Policy layer: decides what may run, and never runs it.

This package owns one question -- *is this tool call permitted?* -- and answers
it by turning an untrusted :class:`ToolIntent` into an
:class:`ApprovedIntent`. It depends only on ``personalos.domain``: no storage,
no tool adapters, no execution. See ``docs/ARCHITECTURE_BOUNDARIES.md``.
"""

from .engine import DecisionSink, PolicyDecisionLog, PolicyEngine, default_policy_engine
from .errors import (
    ApprovalRequired,
    InvalidApproval,
    PolicyDenied,
    PolicyError,
    PolicyViolation,
)
from .intents import (
    ApprovalGrant,
    ApprovedIntent,
    Decision,
    IntentOrigin,
    PolicyDecision,
    ToolIntent,
    fingerprint_intent,
)
from .permissions import (
    DEFAULT_CLASS_OUTCOMES,
    DEFAULT_TOOL_PERMISSIONS,
    PermissionClass,
    Provenance,
    ToolPermission,
)
from .rules import (
    ArgumentAllowlistRule,
    DocumentOverwriteRule,
    MutatingToolRule,
    PolicyRule,
    RequireProvenanceRule,
    ToolAllowlistRule,
    UntrustedOriginRule,
    default_rules,
)

__all__ = [
    # Engine
    "PolicyEngine",
    "DecisionSink",
    "PolicyDecisionLog",
    "default_policy_engine",
    # Permission classes
    "PermissionClass",
    "ToolPermission",
    "Provenance",
    "DEFAULT_CLASS_OUTCOMES",
    "DEFAULT_TOOL_PERMISSIONS",
    # Intents
    "ToolIntent",
    "ApprovedIntent",
    "ApprovalGrant",
    "PolicyDecision",
    "Decision",
    "IntentOrigin",
    "fingerprint_intent",
    # Rules
    "PolicyRule",
    "RequireProvenanceRule",
    "ToolAllowlistRule",
    "ArgumentAllowlistRule",
    "MutatingToolRule",
    "DocumentOverwriteRule",
    "UntrustedOriginRule",
    "default_rules",
    # Errors
    "PolicyError",
    "PolicyDenied",
    "ApprovalRequired",
    "PolicyViolation",
    "InvalidApproval",
]
