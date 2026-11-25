"""Which policy decision governs each kind of approval-gated action.

`personalos.domain.job_search.ActionKind` is the closed set of side effects a
graph may propose. Each one reaches the outside world through tool calls, and
it is policy -- not the graph that proposed it -- that holds those calls for a
person. This table says, for every kind, *which* calls those are, as a probe:
the riskiest call the action makes, with the requester and arguments it makes
it with.

`tests/contracts/test_action_policy_coverage.py` puts every probe to the
default policy engine and fails unless policy answers `require_approval`. So
an action kind added without a row here, a row whose tool has no permission
class or is not allowlisted, or a rule removed from under an existing row,
fails CI instead of shipping a side effect nobody decided how to gate.

`ActionExecution.EXECUTOR` kinds are carried out by
`personalos.executor.tool_executor.ToolExecutor`, which evaluates the action
as its first probe's tool (`ACTION_TOOLS` there is derived from this table).
`ActionExecution.SUBGRAPH` kinds are carried out by their own subgraph, one
tool call at a time through the policy gateway: each call is a probe.

Pure data, like the rest of this package.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from personalos.domain.job_search import ActionKind
from personalos.policy.rules import (
    CALENDAR_REQUESTER_PREFIX,
    FILE_TASK_REQUESTER_PREFIX,
    GMAIL_SENDER,
)


class ActionExecution(str, Enum):
    """Who carries an approved action out."""

    #: `ToolExecutor`, evaluating the action as one classified tool.
    EXECUTOR = "executor"
    #: The proposing subgraph, call by call through the policy gateway.
    SUBGRAPH = "subgraph"


@dataclass(frozen=True)
class ToolProbe:
    """One tool call an action makes, as the code that makes it would make it."""

    tool: str
    requested_by: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionPolicy:
    """The calls policy must hold for one kind of action."""

    execution: ActionExecution
    probes: tuple[ToolProbe, ...]

    @property
    def tool(self) -> str:
        """The tool an `EXECUTOR` action is evaluated as: its first probe's."""
        return self.probes[0].tool


_CALENDAR_WRITER = f"{CALENDAR_REQUESTER_PREFIX}plan_calendar_change"
_FILE_WRITER = f"{FILE_TASK_REQUESTER_PREFIX}apply_file_changes"

ACTION_POLICIES: Mapping[ActionKind, ActionPolicy] = MappingProxyType(
    {
        ActionKind.SUBMIT_APPLICATION: ActionPolicy(
            ActionExecution.EXECUTOR,
            (
                ToolProbe(
                    "jobs.submit_application",
                    "graph:job_search#execute_approved_actions",
                    {"job_id": "probe"},
                ),
            ),
        ),
        ActionKind.SEND_RECRUITER_MESSAGE: ActionPolicy(
            ActionExecution.EXECUTOR,
            (ToolProbe("google.gmail_send_message", GMAIL_SENDER, {"to": "probe@example.com"}),),
        ),
        ActionKind.SEND_EMAIL: ActionPolicy(
            ActionExecution.EXECUTOR,
            (ToolProbe("google.gmail_send_message", GMAIL_SENDER, {"to": "probe@example.com"}),),
        ),
        ActionKind.OVERWRITE_DOCUMENT: ActionPolicy(
            ActionExecution.EXECUTOR,
            (
                ToolProbe(
                    "files.overwrite_document",
                    "graph:job_search#prepare_application_packet",
                    {"path": "probe.md", "content": "", "expected_sha256": "probe"},
                ),
            ),
        ),
        ActionKind.CREATE_CALENDAR_EVENT: ActionPolicy(
            ActionExecution.EXECUTOR,
            (ToolProbe("google.calendar_create_event", _CALENDAR_WRITER, {"summary": "probe"}),),
        ),
        ActionKind.UPDATE_CALENDAR_EVENT: ActionPolicy(
            ActionExecution.EXECUTOR,
            (ToolProbe("google.calendar_update_event", _CALENDAR_WRITER, {"event_id": "probe"}),),
        ),
        ActionKind.CANCEL_CALENDAR_EVENT: ActionPolicy(
            ActionExecution.EXECUTOR,
            (ToolProbe("google.calendar_delete_event", _CALENDAR_WRITER, {"event_id": "probe"}),),
        ),
        # A plan needs sign-off when it deletes, or moves more files than may
        # move unattended; each of those calls is held by policy on its own,
        # so a plan cannot be split into calls that each look harmless.
        ActionKind.APPLY_FILE_PLAN: ActionPolicy(
            ActionExecution.SUBGRAPH,
            (
                ToolProbe(
                    "files.delete_file",
                    _FILE_WRITER,
                    {"path": "probe.md", "expected_sha256": "probe"},
                ),
                ToolProbe(
                    "files.move_file",
                    _FILE_WRITER,
                    {
                        "source": "a.md",
                        "destination": "b.md",
                        "expected_sha256": "probe",
                        "plan_size": 1000,
                    },
                ),
            ),
        ),
    }
)


class UnpolicedActionKind(LookupError):
    """An `ActionKind` with no row in `ACTION_POLICIES`: nobody decided how to gate it."""


def action_policy_for(kind: ActionKind) -> ActionPolicy:
    """The row for `kind`; raises `UnpolicedActionKind` for a kind with none."""
    try:
        return ACTION_POLICIES[kind]
    except KeyError:
        raise UnpolicedActionKind(
            f"action kind '{kind.value}' has no entry in ACTION_POLICIES; decide which "
            "policy rule holds it before it can be proposed"
        ) from None


__all__ = [
    "ActionExecution",
    "ActionPolicy",
    "ToolProbe",
    "ACTION_POLICIES",
    "UnpolicedActionKind",
    "action_policy_for",
]
