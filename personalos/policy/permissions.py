"""Permission classes: what kind of side effect a tool is, and what that costs.

A rule in :mod:`personalos.policy.rules` inspects one intent's arguments. A
permission class answers the coarser question that does not need the arguments
at all: *how recoverable is this kind of action?* Every tool the job-search
scope can reach is assigned exactly one class here, and the class -- not the
caller -- decides the default outcome.

Like the rules, everything in this module is a pure function of its inputs.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

from personalos.policy.intents import Decision, IntentOrigin


class PermissionClass(str, Enum):
    """How much damage a tool can do, ordered from least to most."""

    READ_LOCAL = "read_local"  # reads state this system already holds
    READ_EXTERNAL = "read_external"  # job search, read Gmail, list calendar
    WRITE_REVERSIBLE = "write_reversible"  # draft creation, tentative calendar block
    WRITE_EXTERNAL = "write_external"  # send email, create/update event, submit application
    DESTRUCTIVE = "destructive"  # delete file, overwrite important document
    SENSITIVE = "sensitive"  # credential/permission changes; never delegated to the LLM


#: Classes that change something. An untrusted origin never gets one of these
#: unattended, whatever the configured outcome for the class says.
WRITE_CLASSES = frozenset(
    {
        PermissionClass.WRITE_REVERSIBLE,
        PermissionClass.WRITE_EXTERNAL,
        PermissionClass.DESTRUCTIVE,
        PermissionClass.SENSITIVE,
    }
)

#: The outcome each class gets under default configuration.
DEFAULT_CLASS_OUTCOMES: Mapping[PermissionClass, Decision] = MappingProxyType(
    {
        PermissionClass.READ_LOCAL: Decision.ALLOW,
        PermissionClass.READ_EXTERNAL: Decision.ALLOW,
        PermissionClass.WRITE_REVERSIBLE: Decision.ALLOW,
        PermissionClass.WRITE_EXTERNAL: Decision.REQUIRE_APPROVAL,
        PermissionClass.DESTRUCTIVE: Decision.REQUIRE_APPROVAL,
        PermissionClass.SENSITIVE: Decision.DENY,
    }
)

#: Strictness order used whenever two verdicts on the same action are combined:
#: the stricter one always wins, so no source of opinion can loosen another.
_STRICTNESS = {Decision.ALLOW: 0, Decision.REQUIRE_APPROVAL: 1, Decision.DENY: 2}


def strictest(first: Decision, second: Decision) -> Decision:
    """Return whichever of two decisions is the more restrictive."""
    return first if _STRICTNESS[first] >= _STRICTNESS[second] else second


class Provenance(BaseModel):
    """Where a proposed action came from.

    The same two facts :class:`~personalos.policy.intents.ToolIntent` carries
    as ``origin`` and ``requested_by``, as a value of their own so an action
    can be evaluated without first being built into an intent.
    """

    model_config = ConfigDict(frozen=True)

    origin: IntentOrigin = IntentOrigin.SYSTEM
    requested_by: str = "unknown"

    @property
    def attributed(self) -> bool:
        """True when the action says who asked for it."""
        return bool(self.requested_by) and self.requested_by != "unknown"


@dataclass(frozen=True)
class ToolPermission:
    """The permission class of one tool, and the scopes it may be granted."""

    permission_class: PermissionClass
    #: Capabilities the tool consumes, in the ``resource:verb`` vocabulary
    #: ``policy_decisions.requested_scopes`` records. A request for a scope
    #: outside this set is a request for more than the tool needs.
    scopes: frozenset[str] = frozenset()


def _permission(permission_class: PermissionClass, scopes: Iterable[str]) -> ToolPermission:
    return ToolPermission(permission_class, frozenset(scopes))


#: Every tool the job-search scope knows how to classify, by ``server.tool``.
#: Being listed here classifies a tool; it does not make it reachable. An
#: intent must still clear ``ToolAllowlistRule``, so the entries for servers
#: that are not implemented yet stay unreachable until they are allowlisted.
DEFAULT_TOOL_PERMISSIONS: Mapping[str, ToolPermission] = MappingProxyType(
    {
        # READ_LOCAL
        "jobs.filter_jobs": _permission(PermissionClass.READ_LOCAL, {"jobs:read"}),
        "files.read_file": _permission(PermissionClass.READ_LOCAL, {"artifacts:read"}),
        # READ_EXTERNAL
        "jobs.search_jobs": _permission(PermissionClass.READ_EXTERNAL, {"jobs:read"}),
        "jobs.scrape_job_details": _permission(PermissionClass.READ_EXTERNAL, {"jobs:read"}),
        "google.gmail_read_message": _permission(
            PermissionClass.READ_EXTERNAL, {"communications:read"}
        ),
        "google.calendar_list_events": _permission(
            PermissionClass.READ_EXTERNAL, {"calendar:read"}
        ),
        # WRITE_REVERSIBLE
        "jobs.save_favorite_job": _permission(PermissionClass.WRITE_REVERSIBLE, {"jobs:write"}),
        "google.gmail_create_draft": _permission(
            PermissionClass.WRITE_REVERSIBLE, {"communications:draft"}
        ),
        "google.calendar_create_tentative_block": _permission(
            PermissionClass.WRITE_REVERSIBLE, {"calendar:draft"}
        ),
        # WRITE_EXTERNAL
        "google.gmail_send_message": _permission(
            PermissionClass.WRITE_EXTERNAL, {"communications:send"}
        ),
        "google.calendar_create_event": _permission(
            PermissionClass.WRITE_EXTERNAL, {"calendar:write"}
        ),
        "google.calendar_update_event": _permission(
            PermissionClass.WRITE_EXTERNAL, {"calendar:write"}
        ),
        "jobs.submit_application": _permission(
            PermissionClass.WRITE_EXTERNAL, {"applications:submit", "artifacts:read"}
        ),
        # DESTRUCTIVE
        "files.delete_file": _permission(PermissionClass.DESTRUCTIVE, {"artifacts:delete"}),
        "files.overwrite_document": _permission(
            PermissionClass.DESTRUCTIVE, {"artifacts:write"}
        ),
        # SENSITIVE
        "google.update_credentials": _permission(
            PermissionClass.SENSITIVE, {"credentials:write"}
        ),
        "google.change_permissions": _permission(
            PermissionClass.SENSITIVE, {"permissions:write"}
        ),
    }
)


__all__ = [
    "PermissionClass",
    "WRITE_CLASSES",
    "DEFAULT_CLASS_OUTCOMES",
    "DEFAULT_TOOL_PERMISSIONS",
    "Provenance",
    "ToolPermission",
    "strictest",
]
