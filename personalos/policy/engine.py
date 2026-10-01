"""The policy engine: the single place where intents become authorized.

Nothing else in the system may decide that a tool call is acceptable. Callers
hand the engine a :class:`~personalos.policy.intents.ToolIntent` and receive
either an :class:`~personalos.policy.intents.ApprovedIntent` or an exception.

Every verdict is written to the decision log (the ``policy_decisions`` table in
a real deployment) before it is returned, so by the time a caller can act on an
outcome the record of it already exists.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol
from uuid import UUID

from personalos.policy.errors import ApprovalRequired, PolicyDenied
from personalos.policy.intents import (
    ApprovalGrant,
    ApprovedIntent,
    Decision,
    IntentOrigin,
    PolicyDecision,
    ToolIntent,
    mint_approved_intent,
)
from personalos.policy.permissions import (
    DEFAULT_CLASS_OUTCOMES,
    DEFAULT_TOOL_PERMISSIONS,
    WRITE_CLASSES,
    PermissionClass,
    Provenance,
    ToolPermission,
    strictest,
)
from personalos.policy.rules import PolicyRule, default_rules

logger = logging.getLogger(__name__)

#: Called with every decision the engine reaches, for audit trails. Kept as a
#: plain callable so the policy layer does not depend on the event bus.
DecisionSink = Callable[[ToolIntent, PolicyDecision], None]


class PolicyDecisionLog(Protocol):
    """Port for the durable record of every verdict.

    Defined here and implemented in ``personalos.persistence`` (wired by the
    composition root), so the policy layer is backed by the ``policy_decisions``
    table without importing storage. Takes plain values only, for the same
    reason: the implementation must not need a policy type to satisfy it.

    ``record`` must not return until the row is durable. The engine calls it
    before handing back an outcome, and lets its exceptions propagate: an
    action whose decision could not be recorded does not get an outcome at all.
    """

    def record(
        self,
        *,
        principal: str,
        workflow_id: UUID | None,
        tool: str,
        args_hash: str,
        decision: str,
        requested_scopes: list[str],
    ) -> None:
        """Persist one verdict."""
        ...


class PolicyEngine:
    """Evaluates proposed actions against permission classes and a rule chain.

    Two entry points share one verdict and one decision log:

    ``evaluate``
        Classifies an action from its identity alone -- principal, tool,
        scopes, provenance -- using the tool's :class:`PermissionClass`.

    ``evaluate_intent`` / ``authorize``
        Additionally run the rule chain over a full :class:`ToolIntent`. The
        stricter of the two verdicts wins.

    Rule chain resolution order, chosen so that adding a rule can only ever
    tighten behaviour:

    1. The first rule that denies wins, and evaluation stops.
    2. Otherwise, any rule that demanded approval wins.
    3. Otherwise, an allow from any rule wins.
    4. Otherwise the default applies, which is deny.
    """

    def __init__(
        self,
        rules: Sequence[PolicyRule] | None = None,
        *,
        default_decision: Decision = Decision.DENY,
        decision_sink: DecisionSink | None = None,
        decision_log: PolicyDecisionLog | None = None,
        tool_permissions: Mapping[str, ToolPermission] | None = None,
        class_outcomes: Mapping[PermissionClass, Decision] | None = None,
    ):
        """Build an engine. With no rules supplied, every intent is denied."""
        self.rules: list[PolicyRule] = list(rules or ())
        self.default_decision = default_decision
        self.decision_sink = decision_sink
        self.decision_log = decision_log
        self.tool_permissions: Mapping[str, ToolPermission] = (
            DEFAULT_TOOL_PERMISSIONS if tool_permissions is None else tool_permissions
        )
        self.class_outcomes: Mapping[PermissionClass, Decision] = {
            **DEFAULT_CLASS_OUTCOMES,
            **(class_outcomes or {}),
        }

    def evaluate(
        self,
        principal: str,
        workflow_id: UUID | None,
        tool: str,
        args_hash: str,
        requested_scopes: Sequence[str],
        provenance: Provenance,
    ) -> Decision:
        """Decide ALLOW, REQUIRE_APPROVAL or DENY for one proposed tool call.

        ``tool`` is the fully-qualified ``server.tool`` reference. The verdict
        is written to the decision log before it is returned, whatever it is.
        """
        decision, rule, reason = self._classify(tool, requested_scopes, provenance)
        self._persist(principal, workflow_id, tool, args_hash, decision, requested_scopes)
        logger.info(
            "policy %s %s (rule=%s, origin=%s, requested_by=%s, principal=%s, "
            "workflow_id=%s): %s",
            decision.value,
            tool,
            rule,
            provenance.origin.value,
            provenance.requested_by,
            principal,
            workflow_id,
            reason,
        )
        return decision

    def evaluate_intent(self, intent: ToolIntent) -> PolicyDecision:
        """Reach a verdict on an intent without acting on it."""
        decision = self._evaluate_rules(intent)

        # The permission class can only tighten what the rule chain said.
        class_decision, class_rule, class_reason = self._classify(
            intent.tool_ref,
            self._declared_scopes(intent.tool_ref),
            Provenance(origin=intent.origin, requested_by=intent.requested_by),
        )
        if strictest(decision.decision, class_decision) != decision.decision:
            decision = PolicyDecision(
                intent_id=intent.intent_id,
                tool_ref=intent.tool_ref,
                decision=class_decision,
                rule=class_rule,
                reason=class_reason,
            )
        return self._record(intent, decision)

    def _evaluate_rules(self, intent: ToolIntent) -> PolicyDecision:
        """Run the rule chain. Pure: nothing is logged or recorded here."""
        escalation: PolicyDecision | None = None
        approval: PolicyDecision | None = None

        for rule in self.rules:
            decision = rule.evaluate(intent)
            if decision is None:
                continue
            if decision.decision == Decision.DENY:
                return decision
            if decision.decision == Decision.REQUIRE_APPROVAL and escalation is None:
                escalation = decision
            elif decision.decision == Decision.ALLOW and approval is None:
                approval = decision

        if escalation is not None:
            return escalation
        if approval is not None:
            return approval

        return PolicyDecision(
            intent_id=intent.intent_id,
            tool_ref=intent.tool_ref,
            decision=self.default_decision,
            rule="default",
            reason=(
                f"no rule allowed '{intent.tool_ref}'; "
                f"default is {self.default_decision.value}"
            ),
        )

    def authorize(
        self,
        intent: ToolIntent,
        approval: ApprovalGrant | None = None,
    ) -> ApprovedIntent:
        """Clear an intent for execution, or raise.

        Raises :class:`PolicyDenied` when a rule refused the intent, and
        :class:`ApprovalRequired` when a human grant is needed and none (or a
        mismatched one) was supplied.
        """
        decision = self.evaluate_intent(intent)

        if decision.decision == Decision.DENY:
            raise PolicyDenied(decision)

        if decision.decision == Decision.REQUIRE_APPROVAL:
            if approval is None or not approval.matches(intent):
                raise ApprovalRequired(decision)

        return mint_approved_intent(intent, decision, approval)

    def _classify(
        self,
        tool: str,
        requested_scopes: Sequence[str],
        provenance: Provenance,
    ) -> tuple[Decision, str, str]:
        """Verdict from the tool's permission class, as (decision, rule, reason)."""
        if not provenance.attributed:
            return Decision.DENY, "require_provenance", "action has no requested_by provenance"

        permission = self.tool_permissions.get(tool)
        if permission is None:
            return (
                Decision.DENY,
                "permission_class",
                f"tool '{tool}' has no permission class; unclassified tools are denied",
            )

        excess = sorted(set(requested_scopes) - permission.scopes)
        if excess:
            return (
                Decision.DENY,
                "requested_scopes",
                f"tool '{tool}' may not be granted scopes {excess}",
            )

        permission_class = permission.permission_class
        untrusted = provenance.origin == IntentOrigin.LLM

        # Not configurable: no class_outcomes override can hand this to a model.
        if permission_class == PermissionClass.SENSITIVE and untrusted:
            return (
                Decision.DENY,
                "permission_class",
                f"'{tool}' is {permission_class.value}; never delegated to the LLM",
            )

        decision = self.class_outcomes[permission_class]
        reason = f"'{tool}' is {permission_class.value}"
        if untrusted and permission_class in WRITE_CLASSES:
            decision = strictest(decision, Decision.REQUIRE_APPROVAL)
            reason += f", proposed by untrusted origin '{provenance.origin.value}'"
        return decision, "permission_class", reason

    def _declared_scopes(self, tool: str) -> list[str]:
        """The scopes a tool is classified as consuming; empty if unclassified."""
        permission = self.tool_permissions.get(tool)
        return sorted(permission.scopes) if permission else []

    def _persist(
        self,
        principal: str,
        workflow_id: UUID | None,
        tool: str,
        args_hash: str,
        decision: Decision,
        requested_scopes: Sequence[str],
    ) -> None:
        """Write the verdict to the decision log. Failures propagate."""
        if self.decision_log is None:
            return
        self.decision_log.record(
            principal=principal,
            workflow_id=workflow_id,
            tool=tool,
            args_hash=args_hash,
            decision=decision.value,
            requested_scopes=list(requested_scopes),
        )

    def _record(self, intent: ToolIntent, decision: PolicyDecision) -> PolicyDecision:
        """Persist the decision, log it, and hand it to the audit sink."""
        self._persist(
            intent.context.actor_id,
            intent.context.workflow_id,
            intent.tool_ref,
            intent.fingerprint(),
            decision.decision,
            self._declared_scopes(intent.tool_ref),
        )
        logger.info(
            "policy %s %s (rule=%s, origin=%s, requested_by=%s, %s): %s",
            decision.decision.value,
            decision.tool_ref,
            decision.rule,
            intent.origin.value,
            intent.requested_by,
            intent.context.as_log_str(),
            decision.reason,
        )
        if self.decision_sink is not None:
            self.decision_sink(intent, decision)
        return decision


def default_policy_engine(
    decision_sink: DecisionSink | None = None,
    decision_log: PolicyDecisionLog | None = None,
) -> PolicyEngine:
    """The engine the application boots with: default-deny plus the base rules."""
    return PolicyEngine(default_rules(), decision_sink=decision_sink, decision_log=decision_log)


__all__ = ["PolicyEngine", "DecisionSink", "PolicyDecisionLog", "default_policy_engine"]
