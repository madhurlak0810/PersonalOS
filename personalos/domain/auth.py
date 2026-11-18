"""Who is calling the API: the verified principal, and the credential shapes.

A request's actor is whatever its credential verified as -- never a header or
a body field the caller filled in. `Principal` is that verified identity; the
API derives it in `apps.api.auth` and every route takes its actor from there.

A principal also says what kind of identity it is and what it may do: the
token's `scopes` (`read`, `write`, `approve`, `admin`), and whether it is a
person or a process (`IdentityKind`). The worker, the scheduler and every
model-driven node are system identities, which can never hold `approve` -- so
the agent can propose an action but never approve it. See
`docs/AUTHORIZATION.md`.

The credential shapes live here, not next to the code that mints them,
because `personalos.domain.redaction` matches on them too: a token is
redacted by shape even in a process that never loaded it (the CLI rotating it
while the API logs).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

#: Prefix of the long-lived local API token: `pos_at_<43 url-safe chars>`.
API_TOKEN_PREFIX = "pos_at_"

#: Prefix of a short-lived, workflow-scoped SSE stream ticket.
STREAM_TICKET_PREFIX = "pos_st_"


class AuthMethod(str, Enum):
    """How a principal was established. Recorded, never trusted for scoping."""

    #: The locally generated bearer token (`personalos.cli auth token`).
    LOCAL_TOKEN = "local_token"
    #: A short-lived ticket for one workflow's event stream.
    STREAM_TICKET = "stream_ticket"
    #: Authentication is turned off (`API_AUTH_ENABLED=false`, loopback only).
    DISABLED = "disabled"
    #: A process's own identity (the worker, the scheduler); never from a request.
    SYSTEM = "system"


class Scope(str, Enum):
    """What a token lets its holder do. Carried by the principal, checked per route.

    - `read`: see the actor's own workflows, approvals, applications, audit.
    - `write`: start, resume, retry and move them.
    - `approve`: decide an approval (`POST /v1/approvals/{id}/decision`, or a
      resume that carries decisions). Never held by a system identity.
    - `admin`: install-wide operations that belong to no one actor -- the
      Google connection, dead-lettered events, the integration health board.

    `admin` does not widen ownership: an admin still sees only its own
    workflows. Ownership is `ActorScope`'s, not a scope's.
    """

    READ = "read"
    WRITE = "write"
    APPROVE = "approve"
    ADMIN = "admin"


#: Everything a human's token holds unless configured otherwise.
HUMAN_SCOPES: frozenset[Scope] = frozenset(Scope)

#: Everything a system identity may ever hold: anything but `approve`.
SYSTEM_SCOPES: frozenset[Scope] = frozenset({Scope.READ, Scope.WRITE, Scope.ADMIN})


class IdentityKind(str, Enum):
    """Whether a principal is a person or a process acting on their behalf."""

    #: A person, holding a token they were given.
    HUMAN = "human"
    #: The worker, the scheduler, a model-driven node, an event source. It may
    #: propose an action; it may never be the one who approves it.
    SYSTEM = "system"


#: Actor ids that name a process, not a person. `system` is the default actor
#: of a run nobody started; the prefixed forms are the worker's own identities
#: (`system:worker`, `system:approval_expiry`), event sources (`event:gmail`)
#: and model-driven nodes (`model:...`, `agent:...`).
SYSTEM_ACTOR_IDS: frozenset[str] = frozenset({"system", "worker", "scheduler"})
SYSTEM_ACTOR_PREFIXES: tuple[str, ...] = (
    "system:",
    "worker:",
    "scheduler:",
    "event:",
    "model:",
    "agent:",
)


def is_system_actor(actor_id: str | None) -> bool:
    """Whether `actor_id` names a process rather than a person.

    Checked by shape, so it holds in a process that never built the
    principal: the graph's execution gate refuses an approval decided by
    such an actor even if the decision was written straight into state.
    """
    if not actor_id:
        return True
    lowered = actor_id.strip().lower()
    return lowered in SYSTEM_ACTOR_IDS or lowered.startswith(SYSTEM_ACTOR_PREFIXES)


class SystemIdentityCannotApprove(ValueError):
    """A system principal was built holding the `approve` scope."""


@dataclass(frozen=True)
class Principal:
    """A verified caller: who it is, what kind of identity, and what it may do.

    `actor_id` is what routes record as the actor and what every repository
    scopes by (`personalos.domain.ownership.ActorScope`). `scopes` is what the
    token carries; `kind` is fixed by the actor id's shape -- a system actor
    id is a system principal whatever the caller asked for -- and a system
    principal can never hold `approve`: constructing one that does raises.
    """

    actor_id: str
    method: AuthMethod
    #: `None` is the default for the kind: `HUMAN_SCOPES` or `SYSTEM_SCOPES`.
    scopes: frozenset[Scope] | None = field(default=None, compare=True)
    kind: IdentityKind = IdentityKind.HUMAN

    def __post_init__(self) -> None:
        if is_system_actor(self.actor_id):
            object.__setattr__(self, "kind", IdentityKind.SYSTEM)
        if self.scopes is None:
            default = SYSTEM_SCOPES if self.kind is IdentityKind.SYSTEM else HUMAN_SCOPES
            object.__setattr__(self, "scopes", default)
        scopes = frozenset(Scope(s) for s in self.scopes)  # type: ignore[union-attr]
        object.__setattr__(self, "scopes", scopes)
        if self.kind is IdentityKind.SYSTEM and Scope.APPROVE in scopes:
            raise SystemIdentityCannotApprove(
                f"system identity {self.actor_id!r} cannot hold the approve scope"
            )

    @property
    def is_system(self) -> bool:
        """Whether this is a process rather than a person."""
        return self.kind is IdentityKind.SYSTEM

    def has(self, scope: Scope) -> bool:
        """Whether the token carries `scope`."""
        return scope in self.scopes  # type: ignore[operator]

    def __repr__(self) -> str:
        return f"Principal(actor_id={self.actor_id!r}, method={self.method.value})"


def parse_scopes(values: Iterable[str] | str) -> frozenset[Scope]:
    """Scopes from configuration: `"read,write"` or a list. Unknown names raise."""
    if isinstance(values, str):
        values = values.split(",")
    return frozenset(Scope(value.strip().lower()) for value in values if value.strip())


def system_principal(actor_id: str, scopes: Iterable[Scope] = SYSTEM_SCOPES) -> Principal:
    """The principal a worker, scheduler or model-driven node acts as.

    `actor_id` must look like a system actor (`system:worker`, ...), so the
    identity stays recognisable wherever it is recorded.
    """
    if not is_system_actor(actor_id):
        raise ValueError(f"{actor_id!r} is not a system actor id (e.g. 'system:worker')")
    return Principal(
        actor_id=actor_id,
        method=AuthMethod.SYSTEM,
        scopes=frozenset(scopes),
        kind=IdentityKind.SYSTEM,
    )


#: The worker's identity: what it records when it acts on its own account.
WORKER_ACTOR = "system:worker"
#: The scheduler's identity: sweeps, expiries, health checks.
SCHEDULER_ACTOR = "system:scheduler"


__all__ = [
    "API_TOKEN_PREFIX",
    "HUMAN_SCOPES",
    "SCHEDULER_ACTOR",
    "STREAM_TICKET_PREFIX",
    "SYSTEM_ACTOR_IDS",
    "SYSTEM_ACTOR_PREFIXES",
    "SYSTEM_SCOPES",
    "WORKER_ACTOR",
    "AuthMethod",
    "IdentityKind",
    "Principal",
    "Scope",
    "SystemIdentityCannotApprove",
    "is_system_actor",
    "parse_scopes",
    "system_principal",
]
