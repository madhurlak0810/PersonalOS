"""Whose data a read or write may touch: the actor scope every repository takes.

**Ownership is per actor, and it is decided in the data layer.** Every store
the API reads workflows, workflow events, approvals, applications, artifacts,
audit records or credential references through takes an `access` argument --
required, keyword-only, with no default -- and filters by it in its own query.
A route that forgets to check ownership cannot exist, because a route cannot
call the store without saying whose data it wants.

An access is one of two things:

- an `ActorScope`, built from the verified principal (never from anything the
  caller sent): the store returns only that actor's rows, and an object that
  belongs to someone else is indistinguishable from one that does not exist;
- `SYSTEM_ACCESS`, which the worker and scheduler pass explicitly. It sees
  everything, and is never built from a request.

**How an actor maps to what it owns.** Rows record their owner as the
candidate id they are about (`users.id`, the `user_id` column on
`workflow_runs`, `applications`, `credentials`, ...). An actor owns the rows
keyed by `actor_user_id(actor_id)`: a stable id derived from the actor string
(or the actor itself, when it is already a UUID). So the mapping needs no
table of its own, and two actors -- `alice` and `bob` -- own disjoint sets of
rows with the schema as it is. A workflow is owned through its runs: it is
the actor's when a run of it is keyed by the actor's id, and
`WorkflowThreadRegistry.register` refuses to add a run for a different owner
to an existing workflow, so a workflow never has two.
"""

from dataclasses import dataclass
from typing import Final
from uuid import UUID

from personalos.domain.auth import Principal
from personalos.domain.search_profiles import actor_user_id


@dataclass(frozen=True)
class ActorScope:
    """One actor's view of the data: what its owner id is, and who is asking."""

    actor_id: str
    #: The `users.id` the actor's rows are keyed by: `actor_user_id(actor_id)`.
    owner_id: UUID

    @classmethod
    def of(cls, actor_id: str) -> "ActorScope":
        """The scope of `actor_id`."""
        return cls(actor_id=actor_id, owner_id=actor_user_id(actor_id))

    @classmethod
    def for_principal(cls, principal: Principal) -> "ActorScope":
        """The scope of a verified principal."""
        return cls.of(principal.actor_id)

    def __repr__(self) -> str:
        return f"ActorScope(actor_id={self.actor_id!r})"


class _SystemAccess:
    """Unscoped access, for the worker and scheduler. A singleton: `SYSTEM_ACCESS`."""

    _instance: "_SystemAccess | None" = None

    def __new__(cls) -> "_SystemAccess":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SYSTEM_ACCESS"


#: Unscoped: every row. Passed by the worker and scheduler, never by a route.
SYSTEM_ACCESS: Final = _SystemAccess()

#: What a scoped store method takes.
Access = ActorScope | _SystemAccess


def owner_of(access: Access) -> UUID | None:
    """The owner id `access` is limited to, or `None` for `SYSTEM_ACCESS`."""
    if isinstance(access, ActorScope):
        return access.owner_id
    if access is SYSTEM_ACCESS:
        return None
    raise TypeError(f"expected an ActorScope or SYSTEM_ACCESS, got {access!r}")


def can_see(access: Access, owner_id: UUID | None) -> bool:
    """Whether `access` may see a row owned by `owner_id`."""
    scoped = owner_of(access)
    return scoped is None or (owner_id is not None and owner_id == scoped)


__all__ = ["SYSTEM_ACCESS", "Access", "ActorScope", "can_see", "owner_of"]
