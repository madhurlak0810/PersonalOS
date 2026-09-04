"""Policy rules.

A rule inspects a :class:`~personalos.policy.intents.ToolIntent` and either
expresses an opinion (allow / deny / require approval) or abstains by returning
``None``. Rules never perform side effects and never touch storage or tool
adapters: they are pure functions of the intent, which is what makes the policy
layer cheap to test exhaustively.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from personalos.policy.intents import Decision, IntentOrigin, PolicyDecision, ToolIntent

IDEMPOTENCY_KEY_ARG = "idempotency_key"


class PolicyRule(ABC):
    """One check against an intent."""

    #: Stable identifier recorded on the decision, for audit and tests.
    name: str = "unnamed"

    @abstractmethod
    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Return a decision, or None to abstain."""

    # Helpers so rules do not each rebuild the decision payload.
    def _decide(
        self, intent: ToolIntent, decision: Decision, reason: str
    ) -> PolicyDecision:
        return PolicyDecision(
            intent_id=intent.intent_id,
            tool_ref=intent.tool_ref,
            decision=decision,
            rule=self.name,
            reason=reason,
        )

    def allow(self, intent: ToolIntent, reason: str) -> PolicyDecision:
        """Build an allow decision attributed to this rule."""
        return self._decide(intent, Decision.ALLOW, reason)

    def deny(self, intent: ToolIntent, reason: str) -> PolicyDecision:
        """Build a deny decision attributed to this rule."""
        return self._decide(intent, Decision.DENY, reason)

    def require_approval(self, intent: ToolIntent, reason: str) -> PolicyDecision:
        """Build an approval-required decision attributed to this rule."""
        return self._decide(intent, Decision.REQUIRE_APPROVAL, reason)


class RequireProvenanceRule(PolicyRule):
    """Reject intents that do not say who asked for them.

    Provenance is what makes every other rule auditable, so an intent without
    it is treated as malformed rather than merely untidy.
    """

    name = "require_provenance"

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Deny when ``requested_by`` was left unset."""
        if not intent.requested_by or intent.requested_by == "unknown":
            return self.deny(intent, "intent has no requested_by provenance")
        return None


class ToolAllowlistRule(PolicyRule):
    """Only tools on the allowlist may run at all.

    This is the rule that makes the system default-deny in practice: a newly
    added MCP tool is unreachable until it is listed here.
    """

    name = "tool_allowlist"

    def __init__(self, allowed: Iterable[str]):
        """Take fully-qualified ``server.tool`` references."""
        self.allowed: set[str] = set(allowed)

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Allow listed tools, deny everything else."""
        if intent.tool_ref not in self.allowed:
            return self.deny(intent, f"tool '{intent.tool_ref}' is not on the allowlist")
        return self.allow(intent, f"tool '{intent.tool_ref}' is allowlisted")


class ArgumentAllowlistRule(PolicyRule):
    """Reject arguments a tool did not declare.

    Tool adapters splat arguments into handlers, so an unexpected key is a
    confused-deputy risk: it lets a caller (in particular a model) reach
    parameters the orchestration layer never meant to expose.
    """

    name = "argument_allowlist"

    def __init__(self, allowed_arguments: Mapping[str, Iterable[str]]):
        """Map ``server.tool`` to the argument names it accepts."""
        self.allowed_arguments = {
            tool_ref: set(names) | {IDEMPOTENCY_KEY_ARG}
            for tool_ref, names in allowed_arguments.items()
        }

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Deny unknown argument names; abstain for undeclared tools."""
        allowed = self.allowed_arguments.get(intent.tool_ref)
        if allowed is None:
            # No declaration for this tool: ToolAllowlistRule owns that call.
            return None
        unexpected = sorted(set(intent.arguments) - allowed)
        if unexpected:
            return self.deny(
                intent,
                f"unexpected arguments for '{intent.tool_ref}': {unexpected}",
            )
        return None


class MutatingToolRule(PolicyRule):
    """Mutating tools need an idempotency key, and normally need approval.

    The key requirement mirrors the guard in
    ``personalos.persistence.idempotency``: a side effect that cannot be
    deduplicated must not be attempted at all. Approval is required unless the
    tool is explicitly listed as safe to run unattended.
    """

    name = "mutating_tool"

    def __init__(self, auto_approved: Iterable[str] | None = None):
        """Take ``server.tool`` refs that may mutate without human sign-off."""
        self.auto_approved: set[str] = set(auto_approved or ())

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Abstain for read-only intents; gate mutating ones."""
        if not intent.mutating:
            return None
        key = intent.arguments.get(IDEMPOTENCY_KEY_ARG)
        if not key:
            return self.deny(
                intent,
                f"mutating tool '{intent.tool_ref}' requires an {IDEMPOTENCY_KEY_ARG}",
            )
        if intent.tool_ref in self.auto_approved:
            return self.allow(
                intent, f"mutating tool '{intent.tool_ref}' is auto-approved"
            )
        return self.require_approval(
            intent, f"mutating tool '{intent.tool_ref}' needs human approval"
        )


class UntrustedOriginRule(PolicyRule):
    """Escalate anything a model proposed.

    A model-authored intent has already passed the allowlist by this point, so
    the remaining question is trust: reads proceed, side effects wait for a
    human regardless of any auto-approval granted to system code paths.
    """

    name = "untrusted_origin"

    def __init__(self, untrusted: Iterable[IntentOrigin] = (IntentOrigin.LLM,)):
        """Take the origins to treat as untrusted."""
        self.untrusted: set[IntentOrigin] = set(untrusted)

    def evaluate(self, intent: ToolIntent) -> PolicyDecision | None:
        """Require approval for mutating intents from untrusted origins."""
        if intent.origin not in self.untrusted:
            return None
        if intent.mutating:
            return self.require_approval(
                intent,
                f"mutating intent from untrusted origin '{intent.origin.value}'",
            )
        return None


#: Tools exposed by the jobs MCP server, with the argument surface each accepts.
#: Being listed here makes a tool reachable; whether it can run unattended is
#: still decided by MutatingToolRule and UntrustedOriginRule.
JOB_SEARCH_TOOL_ARGUMENTS = {
    "jobs.search_jobs": {"keywords", "locations", "job_type", "limit"},
    "jobs.scrape_job_details": {"job_id", "job_url"},
    "jobs.filter_jobs": {
        "jobs",
        "salary_min",
        "salary_max",
        "experience_level",
        "remote_only",
    },
    # Mutating: reachable, but MutatingToolRule holds it for approval.
    "jobs.save_favorite_job": {"job_id", "notes"},
}


def default_rules() -> list[PolicyRule]:
    """The rule chain the application boots with.

    Read-only job search tools are allowlisted; nothing else can run until it
    is added here deliberately.
    """
    return [
        RequireProvenanceRule(),
        ToolAllowlistRule(JOB_SEARCH_TOOL_ARGUMENTS.keys()),
        ArgumentAllowlistRule(JOB_SEARCH_TOOL_ARGUMENTS),
        MutatingToolRule(),
        UntrustedOriginRule(),
    ]


__all__ = [
    "PolicyRule",
    "RequireProvenanceRule",
    "ToolAllowlistRule",
    "ArgumentAllowlistRule",
    "MutatingToolRule",
    "UntrustedOriginRule",
    "JOB_SEARCH_TOOL_ARGUMENTS",
    "default_rules",
    "IDEMPOTENCY_KEY_ARG",
]
