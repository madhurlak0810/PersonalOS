"""The policy engine: the single place where intents become authorized.

Nothing else in the system may decide that a tool call is acceptable. Callers
hand the engine a :class:`~personalos.policy.intents.ToolIntent` and receive
either an :class:`~personalos.policy.intents.ApprovedIntent` or an exception.
"""

import logging
from collections.abc import Callable, Sequence

from personalos.policy.errors import ApprovalRequired, PolicyDenied
from personalos.policy.intents import (
    ApprovalGrant,
    ApprovedIntent,
    Decision,
    PolicyDecision,
    ToolIntent,
    mint_approved_intent,
)
from personalos.policy.rules import PolicyRule, default_rules

logger = logging.getLogger(__name__)

#: Called with every decision the engine reaches, for audit trails. Kept as a
#: plain callable so the policy layer does not depend on the event bus.
DecisionSink = Callable[[ToolIntent, PolicyDecision], None]


class PolicyEngine:
    """Evaluates intents against an ordered rule chain.

    Resolution order, chosen so that adding a rule can only ever tighten
    behaviour:

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
    ):
        """Build an engine. With no rules supplied, every intent is denied."""
        self.rules: list[PolicyRule] = list(rules or ())
        self.default_decision = default_decision
        self.decision_sink = decision_sink

    def evaluate(self, intent: ToolIntent) -> PolicyDecision:
        """Reach a verdict on an intent without acting on it."""
        escalation: PolicyDecision | None = None
        approval: PolicyDecision | None = None

        for rule in self.rules:
            decision = rule.evaluate(intent)
            if decision is None:
                continue
            if decision.decision == Decision.DENY:
                return self._record(intent, decision)
            if decision.decision == Decision.REQUIRE_APPROVAL and escalation is None:
                escalation = decision
            elif decision.decision == Decision.ALLOW and approval is None:
                approval = decision

        if escalation is not None:
            return self._record(intent, escalation)
        if approval is not None:
            return self._record(intent, approval)

        return self._record(
            intent,
            PolicyDecision(
                intent_id=intent.intent_id,
                tool_ref=intent.tool_ref,
                decision=self.default_decision,
                rule="default",
                reason=(
                    f"no rule allowed '{intent.tool_ref}'; "
                    f"default is {self.default_decision.value}"
                ),
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
        decision = self.evaluate(intent)

        if decision.decision == Decision.DENY:
            raise PolicyDenied(decision)

        if decision.decision == Decision.REQUIRE_APPROVAL:
            if approval is None or not approval.matches(intent):
                raise ApprovalRequired(decision)

        return mint_approved_intent(intent, decision, approval)

    def _record(self, intent: ToolIntent, decision: PolicyDecision) -> PolicyDecision:
        """Log the decision and hand it to the audit sink."""
        logger.info(
            "policy %s %s (rule=%s, origin=%s, requested_by=%s): %s",
            decision.decision.value,
            decision.tool_ref,
            decision.rule,
            intent.origin.value,
            intent.requested_by,
            decision.reason,
        )
        if self.decision_sink is not None:
            self.decision_sink(intent, decision)
        return decision


def default_policy_engine(
    decision_sink: DecisionSink | None = None,
) -> PolicyEngine:
    """The engine the application boots with: default-deny plus the base rules."""
    return PolicyEngine(default_rules(), decision_sink=decision_sink)


__all__ = ["PolicyEngine", "DecisionSink", "default_policy_engine"]
