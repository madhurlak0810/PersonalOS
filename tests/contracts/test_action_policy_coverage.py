"""CI gate: every approval-gated action kind has a policy decision that holds it.

`ActionKind` is the closed set of side effects a graph may propose. Adding a
member is adding a new way to write to the outside world, and this file is
what makes that impossible to do by accident:

1. **Every `ActionKind` has a row in `ACTION_POLICIES`**, the policy-layer
   table naming the calls the action makes (`personalos.policy.actions`).
2. **Policy holds every one of those calls for a person**, on the path the
   call really takes. A subgraph-run kind's probes go through the gateway's
   rule chain and must come back `require_approval`: not `allow` (it would
   run unattended) and not `deny` (a row that can never run is a mistake
   too, usually an unclassified tool or a requester that is not allowlisted).
   An executor-run kind is decided by `ToolExecutor` from its tool's
   permission class and its risk profile's scopes, and that verdict must be
   `require_approval`; through the rule chain its probe may be held or
   refused, never allowed. A model-originated probe is never allowed.
3. **An executor-run kind's approval terms fit its tool**: the scopes its
   risk profile requests are ones the tool may be granted, or the verdict in
   (2) would be `deny`.
4. **The evals can score it.** Every kind is labelled in
   `evals/golden/approval_boundaries.json`, so approval precision and recall
   cover it.

The last tests remove a rule, a permission class or a row and check that the
gate notices -- the gate is only worth having if it fails.
"""

from collections.abc import Mapping, Sequence

import pytest

from evals import dataset
from personalos.domain.job_search import ActionKind, risk_profile_for
from personalos.executor.tool_executor import ACTION_TOOLS
from personalos.policy.actions import (
    ACTION_POLICIES,
    ActionExecution,
    ActionPolicy,
    ToolProbe,
    UnpolicedActionKind,
    action_policy_for,
)
from personalos.policy.engine import PolicyEngine
from personalos.policy.intents import Decision, IntentOrigin, ToolIntent
from personalos.policy.permissions import (
    DEFAULT_TOOL_PERMISSIONS,
    WRITE_CLASSES,
    PermissionClass,
    Provenance,
    ToolPermission,
)
from personalos.policy.rules import (
    IDEMPOTENCY_KEY_ARG,
    DestructiveFileRule,
    FileBulkMutationRule,
    PolicyRule,
    default_rules,
)


def _intent(probe: ToolProbe, origin: IntentOrigin) -> ToolIntent:
    server, tool = probe.tool.split(".", 1)
    return ToolIntent(
        server=server,
        tool=tool,
        arguments={**probe.arguments, IDEMPOTENCY_KEY_ARG: f"contract-{probe.tool}"},
        origin=origin,
        mutating=True,
        requested_by=probe.requested_by,
    )


def _probe_violations(
    engine: PolicyEngine, kind: ActionKind, policy: ActionPolicy, probe: ToolProbe
) -> list[str]:
    """How one call an action makes escapes the approval it needs, if it does."""
    where = f"{kind.value} via {probe.tool} as {probe.requested_by}"
    violations = []
    permission = engine.tool_permissions.get(probe.tool)
    if permission is None:
        violations.append(f"{where}: tool has no permission class")
    elif permission.permission_class not in WRITE_CLASSES:
        violations.append(f"{where}: classified {permission.permission_class.value}, not a write")
    system = engine.evaluate_intent(_intent(probe, IntentOrigin.SYSTEM))
    # A subgraph's calls go through the gateway's rule chain, so that is where
    # they must be held. An executor-run action is decided by its permission
    # class (below); through the gateway it may be held or refused, never let
    # through.
    held = (
        system.decision is Decision.REQUIRE_APPROVAL
        if policy.execution is ActionExecution.SUBGRAPH
        else system.decision is not Decision.ALLOW
    )
    if not held:
        violations.append(
            f"{where}: policy says {system.decision.value} "
            f"({system.rule}: {system.reason}), expected require_approval"
        )
    if engine.evaluate_intent(_intent(probe, IntentOrigin.LLM)).decision is Decision.ALLOW:
        violations.append(f"{where}: a model-originated call is allowed unattended")
    return violations


def _executor_violations(engine: PolicyEngine, kind: ActionKind, policy: ActionPolicy) -> list[str]:
    """The verdict `ToolExecutor` reaches for an approved action of this kind."""
    requester = policy.probes[0].requested_by
    verdict = engine.evaluate_action(
        requester,
        None,
        policy.tool,
        "contract",
        list(risk_profile_for(kind).scopes),
        Provenance(origin=IntentOrigin.SYSTEM, requested_by=requester),
    )
    if verdict.decision is Decision.REQUIRE_APPROVAL:
        return []
    return [
        f"{kind.value}: the executor's verdict is {verdict.decision.value} "
        f"({verdict.reason}), expected require_approval"
    ]


def action_policy_violations(
    policies: Mapping[ActionKind, ActionPolicy] = ACTION_POLICIES,
    *,
    rules: Sequence[PolicyRule] | None = None,
    tool_permissions: Mapping[str, ToolPermission] = DEFAULT_TOOL_PERMISSIONS,
    class_outcomes: Mapping[PermissionClass, Decision] | None = None,
) -> list[str]:
    """Every way the given policy configuration fails to hold an action kind."""
    engine = PolicyEngine(
        default_rules() if rules is None else rules,
        tool_permissions=tool_permissions,
        class_outcomes=class_outcomes,
    )
    violations: list[str] = []
    for kind in ActionKind:
        policy = policies.get(kind)
        if policy is None:
            violations.append(f"{kind.value}: no row in ACTION_POLICIES")
            continue
        if not policy.probes:
            violations.append(f"{kind.value}: names no tool call for policy to hold")
            continue
        for probe in policy.probes:
            violations += _probe_violations(engine, kind, policy, probe)
        if policy.execution is ActionExecution.EXECUTOR:
            violations += _executor_violations(engine, kind, policy)
    return violations


# --- The gate ---------------------------------------------------------------------------


def test_every_action_kind_is_held_for_approval_by_policy():
    assert action_policy_violations() == []


@pytest.mark.parametrize("kind", list(ActionKind), ids=lambda kind: kind.value)
def test_every_action_kind_has_a_policy_row_and_a_risk_profile(kind):
    assert action_policy_for(kind).probes
    assert risk_profile_for(kind).scopes


def test_the_executor_evaluates_each_kind_as_its_policy_row_says():
    executed = {
        kind: policy.tool
        for kind, policy in ACTION_POLICIES.items()
        if policy.execution is ActionExecution.EXECUTOR
    }
    assert dict(ACTION_TOOLS) == executed


@pytest.mark.parametrize("kind", list(ActionKind), ids=lambda kind: kind.value)
def test_every_action_kind_is_labelled_for_the_approval_evals(kind):
    labels = dataset.load("approval_boundaries.json")["action_kinds"]
    assert kind.value in labels, "label it in evals/golden/approval_boundaries.json"
    assert labels[kind.value]["consequential"] is True


# --- The gate fails when it should ------------------------------------------------------


def test_an_action_kind_with_no_policy_row_fails_the_gate(monkeypatch):
    without = {k: v for k, v in ACTION_POLICIES.items() if k is not ActionKind.SEND_EMAIL}

    assert action_policy_violations(without) == ["send_email: no row in ACTION_POLICIES"]
    monkeypatch.setattr("personalos.policy.actions.ACTION_POLICIES", without)
    with pytest.raises(UnpolicedActionKind, match="send_email"):
        action_policy_for(ActionKind.SEND_EMAIL)


def test_removing_the_permission_class_of_a_calendar_write_fails_the_gate():
    permissions = {
        tool: permission
        for tool, permission in DEFAULT_TOOL_PERMISSIONS.items()
        if tool != "google.calendar_delete_event"
    }

    violations = action_policy_violations(tool_permissions=permissions)

    assert violations, "an unclassified calendar delete must fail the gate"
    assert all(v.startswith("cancel_calendar_event") for v in violations)
    assert any("tool has no permission class" in v for v in violations)


def test_removing_the_file_rules_fails_the_gate_for_file_plans():
    rules = [
        rule
        for rule in default_rules()
        if not isinstance(rule, (DestructiveFileRule, FileBulkMutationRule))
    ]

    violations = action_policy_violations(rules=rules)

    # Delete is still DESTRUCTIVE by class; a bulk move is reversible by class and
    # only `FileBulkMutationRule` held it.
    (violation,) = violations
    assert violation.startswith(
        "apply_file_plan via files.move_file as graph:file_task#apply_file_changes: "
        "policy says allow"
    )


def test_letting_external_writes_run_unattended_fails_the_gate():
    violations = action_policy_violations(
        class_outcomes={PermissionClass.WRITE_EXTERNAL: Decision.ALLOW}
    )

    flagged = {v.split(" ", 1)[0].rstrip(":") for v in violations}
    assert {
        "submit_application",
        "send_email",
        "send_recruiter_message",
        "create_calendar_event",
        "update_calendar_event",
    } <= flagged
