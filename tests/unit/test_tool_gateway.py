"""Tests for the tool boundary: nothing reaches an adapter unapproved."""

from typing import Any

import pytest

from personalos.mcp.adapter import MCPToolInvoker
from personalos.mcp.manager import MCPServerManager
from personalos.policy import (
    ApprovalRequired,
    ApprovedIntent,
    IntentOrigin,
    PolicyDenied,
    PolicyEngine,
    PolicyViolation,
    ToolIntent,
    default_policy_engine,
)
from personalos.tools.gateway import (
    PolicyEnforcingToolGateway,
    ToolExecutionError,
    ToolGateway,
    ToolInvoker,
    ToolResult,
)


class RecordingInvoker:
    """Invoker that records what it was handed and returns a canned payload."""

    def __init__(self, payload: dict[str, Any] = None):
        self.payload = payload or {"success": True, "result": {"jobs": []}}
        self.calls: list[ApprovedIntent] = []

    async def invoke(self, approved: ApprovedIntent) -> dict[str, Any]:
        """Record the approved intent and return the canned payload."""
        assert isinstance(approved, ApprovedIntent), "adapter got unapproved input"
        self.calls.append(approved)
        return self.payload


def read_intent(**overrides) -> ToolIntent:
    """A well-formed, allowlisted read intent."""
    defaults = {
        "server": "jobs",
        "tool": "search_jobs",
        "arguments": {"keywords": ["python"]},
        "origin": IntentOrigin.SYSTEM,
        "requested_by": "test",
    }
    defaults.update(overrides)
    return ToolIntent(**defaults)


def test_invoker_satisfies_the_port():
    """The recording double and the real adapter share one protocol."""
    assert isinstance(RecordingInvoker(), ToolInvoker)
    assert isinstance(MCPToolInvoker(MCPServerManager()), ToolInvoker)


def test_gateway_is_the_tool_gateway_port():
    """Executors can depend on the abstract type."""
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), RecordingInvoker())
    assert isinstance(gateway, ToolGateway)


@pytest.mark.asyncio
async def test_allowed_intent_reaches_the_adapter_as_an_approval():
    """The adapter only ever sees an ApprovedIntent."""
    invoker = RecordingInvoker()
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), invoker)

    result = await gateway.dispatch(read_intent())

    assert isinstance(result, ToolResult)
    assert result.success
    assert len(invoker.calls) == 1
    approved = invoker.calls[0]
    assert isinstance(approved, ApprovedIntent)
    assert approved.tool == "search_jobs"
    assert approved.server == "jobs"


@pytest.mark.asyncio
async def test_denied_intent_never_reaches_the_adapter():
    """A denial stops the call rather than surfacing as a failed result."""
    invoker = RecordingInvoker()
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), invoker)

    with pytest.raises(PolicyDenied):
        await gateway.dispatch(read_intent(tool="rm_rf"))

    assert invoker.calls == []


@pytest.mark.asyncio
async def test_intent_awaiting_approval_never_reaches_the_adapter():
    """Approval-required intents are held, not attempted."""
    invoker = RecordingInvoker()
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), invoker)

    with pytest.raises(ApprovalRequired):
        await gateway.dispatch(
            read_intent(
                tool="save_favorite_job",
                arguments={"job_id": "j1", "idempotency_key": "k" * 12},
                mutating=True,
            )
        )

    assert invoker.calls == []


@pytest.mark.asyncio
async def test_gateway_with_an_empty_policy_denies_everything():
    """A misconfigured engine fails closed."""
    invoker = RecordingInvoker()
    gateway = PolicyEnforcingToolGateway(PolicyEngine(), invoker)

    with pytest.raises(PolicyDenied):
        await gateway.dispatch(read_intent())

    assert invoker.calls == []


@pytest.mark.asyncio
async def test_gateway_refuses_raw_arguments():
    """A dict that looks like a tool call is not a tool call."""
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), RecordingInvoker())

    with pytest.raises(PolicyViolation):
        await gateway.dispatch({"tool": "search_jobs", "arguments": {}})


@pytest.mark.asyncio
async def test_execute_approved_refuses_anything_unapproved():
    """The post-policy entry point cannot be used to skip policy."""
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), RecordingInvoker())

    with pytest.raises(PolicyViolation):
        await gateway.execute_approved(read_intent())


@pytest.mark.asyncio
async def test_mcp_adapter_refuses_anything_unapproved():
    """Defence in depth: the adapter checks too."""
    invoker = MCPToolInvoker(MCPServerManager())

    with pytest.raises(PolicyViolation):
        await invoker.invoke(read_intent())


@pytest.mark.asyncio
async def test_adapter_failure_becomes_a_failed_result():
    """Tool failures are ordinary results; only policy denials raise."""
    invoker = RecordingInvoker({"success": False, "error": "upstream timeout"})
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), invoker)

    result = await gateway.dispatch(read_intent())

    assert not result.success
    assert result.error == "upstream timeout"
    with pytest.raises(ToolExecutionError):
        result.unwrap()


@pytest.mark.asyncio
async def test_result_carries_the_intent_and_deciding_rule():
    """Results are traceable back to the decision that permitted them."""
    invoker = RecordingInvoker()
    gateway = PolicyEnforcingToolGateway(default_policy_engine(), invoker)
    intent = read_intent()

    result = await gateway.dispatch(intent)

    assert result.intent_id == intent.intent_id
    assert result.tool_ref == "jobs.search_jobs"
    assert result.rule == "tool_allowlist"
