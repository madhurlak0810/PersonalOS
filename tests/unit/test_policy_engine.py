"""Tests for the policy layer: default deny, rule order, unforgeable approvals."""

import pytest

from personalos.policy import (
    ApprovalGrant,
    ApprovalRequired,
    ApprovedIntent,
    ArgumentAllowlistRule,
    Decision,
    IntentOrigin,
    MutatingToolRule,
    PolicyDenied,
    PolicyEngine,
    PolicyRule,
    PolicyViolation,
    RequireProvenanceRule,
    ToolAllowlistRule,
    ToolIntent,
    UntrustedOriginRule,
    default_policy_engine,
)
from personalos.policy.intents import mint_approved_intent


def make_intent(**overrides) -> ToolIntent:
    """A well-formed read-only intent, overridable per test."""
    defaults = {
        "server": "jobs",
        "tool": "search_jobs",
        "arguments": {"keywords": ["python"], "locations": ["remote"]},
        "origin": IntentOrigin.SYSTEM,
        "requested_by": "test",
    }
    defaults.update(overrides)
    return ToolIntent(**defaults)


def grant_for(intent: ToolIntent, approved_by: str = "operator") -> ApprovalGrant:
    """A matching approval grant for an intent."""
    return ApprovalGrant(
        intent_id=intent.intent_id,
        intent_fingerprint=intent.fingerprint(),
        approved_by=approved_by,
    )


# ----------------------------------------------------------------------
# Default deny
# ----------------------------------------------------------------------


def test_engine_with_no_rules_denies_everything():
    """An unconfigured engine is closed, not open."""
    decision = PolicyEngine().evaluate(make_intent())
    assert decision.decision == Decision.DENY
    assert decision.rule == "default"


def test_unlisted_tool_is_denied():
    """A tool nobody allowlisted cannot run."""
    engine = default_policy_engine()
    with pytest.raises(PolicyDenied) as excinfo:
        engine.authorize(make_intent(tool="delete_everything"))
    assert "not on the allowlist" in str(excinfo.value)


def test_allowlisted_read_tool_is_authorized():
    """The happy path still works."""
    approved = default_policy_engine().authorize(make_intent())
    assert isinstance(approved, ApprovedIntent)
    assert approved.intent.tool_ref == "jobs.search_jobs"
    assert approved.decision.decision == Decision.ALLOW
    assert approved.approval is None


def test_intent_without_provenance_is_denied():
    """Unattributed intents are treated as malformed."""
    with pytest.raises(PolicyDenied) as excinfo:
        default_policy_engine().authorize(make_intent(requested_by="unknown"))
    assert "provenance" in str(excinfo.value)


def test_unexpected_arguments_are_denied():
    """An argument the tool never declared is a confused-deputy risk."""
    intent = make_intent(
        arguments={"keywords": ["python"], "locations": ["remote"], "callback": "rm -rf"}
    )
    with pytest.raises(PolicyDenied) as excinfo:
        default_policy_engine().authorize(intent)
    assert "callback" in str(excinfo.value)


# ----------------------------------------------------------------------
# Mutating tools
# ----------------------------------------------------------------------


def test_mutating_tool_without_idempotency_key_is_denied():
    """A side effect that cannot be deduplicated is not attempted."""
    intent = make_intent(tool="save_favorite_job", arguments={"job_id": "j1"}, mutating=True)
    with pytest.raises(PolicyDenied) as excinfo:
        default_policy_engine().authorize(intent)
    assert "idempotency_key" in str(excinfo.value)


def test_mutating_tool_requires_approval():
    """With a key but no grant, the intent waits for a human."""
    intent = make_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j1", "idempotency_key": "k" * 12},
        mutating=True,
    )
    engine = default_policy_engine()
    with pytest.raises(ApprovalRequired):
        engine.authorize(intent)

    approved = engine.authorize(intent, grant_for(intent))
    assert approved.decision.decision == Decision.REQUIRE_APPROVAL
    assert approved.approval.approved_by == "operator"


def test_approval_for_a_different_intent_is_rejected():
    """A grant is bound to one intent and cannot be reused."""
    intent = make_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j1", "idempotency_key": "k" * 12},
        mutating=True,
    )
    other = make_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j2", "idempotency_key": "k" * 12},
        mutating=True,
    )
    with pytest.raises(ApprovalRequired):
        default_policy_engine().authorize(intent, grant_for(other))


def test_approval_does_not_survive_argument_tampering():
    """Changing arguments after approval invalidates the grant."""
    intent = make_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j1", "idempotency_key": "k" * 12},
        mutating=True,
    )
    grant = grant_for(intent)
    intent.arguments["job_id"] = "j999"
    with pytest.raises(ApprovalRequired):
        default_policy_engine().authorize(intent, grant)


def test_auto_approved_mutating_tool_runs_unattended():
    """Explicit opt-in is the only way to skip the human."""
    engine = PolicyEngine(
        [
            RequireProvenanceRule(),
            ToolAllowlistRule({"jobs.save_favorite_job"}),
            MutatingToolRule(auto_approved={"jobs.save_favorite_job"}),
        ]
    )
    intent = make_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j1", "idempotency_key": "k" * 12},
        mutating=True,
    )
    assert engine.authorize(intent).decision.decision == Decision.ALLOW


# ----------------------------------------------------------------------
# Untrusted origins
# ----------------------------------------------------------------------


def test_model_proposed_read_is_allowed_when_allowlisted():
    """Reads from a model are fine once the tool and arguments check out."""
    intent = make_intent(origin=IntentOrigin.LLM, requested_by="llm:planner")
    assert default_policy_engine().authorize(intent).intent.origin == IntentOrigin.LLM


def test_model_proposed_mutation_is_escalated_even_when_auto_approved():
    """Auto-approval covers reviewed code paths, not model output."""
    engine = PolicyEngine(
        [
            RequireProvenanceRule(),
            ToolAllowlistRule({"jobs.save_favorite_job"}),
            MutatingToolRule(auto_approved={"jobs.save_favorite_job"}),
            UntrustedOriginRule(),
        ]
    )
    intent = make_intent(
        tool="save_favorite_job",
        arguments={"job_id": "j1", "idempotency_key": "k" * 12},
        mutating=True,
        origin=IntentOrigin.LLM,
        requested_by="llm:planner",
    )
    with pytest.raises(ApprovalRequired):
        engine.authorize(intent)


def test_model_proposed_tool_outside_the_allowlist_is_denied_not_escalated():
    """A denial from any rule beats an escalation from a later one."""
    intent = make_intent(
        tool="exfiltrate",
        arguments={},
        mutating=True,
        origin=IntentOrigin.LLM,
        requested_by="llm:planner",
    )
    with pytest.raises(PolicyDenied):
        default_policy_engine().authorize(intent)


# ----------------------------------------------------------------------
# Rule ordering
# ----------------------------------------------------------------------


class _AlwaysAllow(PolicyRule):
    name = "always_allow"

    def evaluate(self, intent):
        return self.allow(intent, "test rule")


class _AlwaysDeny(PolicyRule):
    name = "always_deny"

    def evaluate(self, intent):
        return self.deny(intent, "test rule")


class _AlwaysEscalate(PolicyRule):
    name = "always_escalate"

    def evaluate(self, intent):
        return self.require_approval(intent, "test rule")


def test_deny_beats_allow_regardless_of_order():
    """Adding a rule can only tighten behaviour."""
    for rules in ([_AlwaysAllow(), _AlwaysDeny()], [_AlwaysDeny(), _AlwaysAllow()]):
        decision = PolicyEngine(rules).evaluate(make_intent())
        assert decision.decision == Decision.DENY, rules


def test_escalation_beats_allow_regardless_of_order():
    """An approval demand is never overridden by a permissive rule."""
    for rules in ([_AlwaysAllow(), _AlwaysEscalate()], [_AlwaysEscalate(), _AlwaysAllow()]):
        decision = PolicyEngine(rules).evaluate(make_intent())
        assert decision.decision == Decision.REQUIRE_APPROVAL, rules


def test_decision_sink_sees_every_decision():
    """Decisions are auditable without policy depending on the event bus."""
    seen = []
    engine = default_policy_engine(decision_sink=lambda i, d: seen.append((i, d)))
    engine.evaluate(make_intent())
    with pytest.raises(PolicyDenied):
        engine.authorize(make_intent(tool="nope"))
    assert len(seen) == 2
    assert [d.decision for _, d in seen] == [Decision.ALLOW, Decision.DENY]


# ----------------------------------------------------------------------
# Approvals cannot be forged
# ----------------------------------------------------------------------


def test_approved_intent_cannot_be_constructed_directly():
    """The type itself refuses to be manufactured outside the policy layer."""
    intent = make_intent()
    decision = default_policy_engine().evaluate(intent)
    with pytest.raises(PolicyViolation):
        ApprovedIntent(intent, decision)


def test_approved_intent_is_immutable():
    """An approval cannot be edited after it is minted."""
    approved = default_policy_engine().authorize(make_intent())
    with pytest.raises(PolicyViolation):
        approved._intent = make_intent(tool="something_else")


def test_a_denied_decision_cannot_be_minted_into_an_approval():
    """Even inside the policy layer, a denial cannot become an approval."""
    intent = make_intent(tool="not_allowlisted")
    decision = default_policy_engine().evaluate(intent)
    assert decision.decision == Decision.DENY
    with pytest.raises(PolicyViolation):
        mint_approved_intent(intent, decision)


def test_argument_allowlist_abstains_for_undeclared_tools():
    """Argument checking does not double as an allowlist."""
    rule = ArgumentAllowlistRule({"jobs.search_jobs": {"keywords"}})
    assert rule.evaluate(make_intent(tool="other", arguments={"anything": 1})) is None
