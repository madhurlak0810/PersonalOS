"""Ownership in SQL, and the audit row every authorization denial leaves.

The query half of `personalos.domain.ownership`: the predicates scoped stores
put in their `WHERE` clauses, so a row another actor owns never leaves the
database for a scoped caller.

- A **workflow** is owned through its runs: `workflow_runs.user_id` is the
  owner of the run, and a workflow is the actor's when one of its runs is.
  (`WorkflowThreadRegistry.register` keeps a workflow to one owner.)
- An **application** -- and its artifact versions, status events and
  recruiter events -- is owned by `applications.user_id`.
- An **audit event** is owned through its workflow; one recorded against no
  workflow is the actor's own when it names that actor (an authorization
  denial, say).
- A **credential reference** is owned by `credentials.user_id`; one with no
  owner is install-wide and visible only to `SYSTEM_ACCESS` or a holder of
  the `admin` scope (the routes decide the second).

**Denials are audited.** `record_denial` appends an `audit_events` row --
actor, route, object type, object id, reason -- through
`AuditEventRepository`, which redacts every free-text field first. The route
comes from `authorization_context` (set per request by the API) and is a
method and path only: never a query string, which is where a stream ticket
travels, and never a header, which is where the token does.

A by-id lookup that a scoped caller cannot see is audited the same way whether
the object belongs to someone else or does not exist, under the one reason
`not_visible`. The denied actor can read its own denials, so recording the two
differently would tell it which ids are real.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from uuid import UUID

from sqlalchemy import exists, literal, select
from sqlalchemy.orm import Session

from personalos.domain.models import AuditEventResult
from personalos.domain.ownership import Access, owner_of
from personalos.persistence.models import WorkflowModel, WorkflowRunModel
from personalos.persistence.repositories import AuditEventRepository

logger = logging.getLogger(__name__)

#: `audit_events.action` prefix of an authorization denial; the reason follows.
AUDIT_DENIED_PREFIX = "authz.denied."


class DenialReason(str, Enum):
    """Why a request was refused. Recorded in `audit_events.action`."""

    #: A by-id lookup the caller cannot see: another actor's, or no such object.
    NOT_VISIBLE = "not_visible"
    #: The token lacks the scope the route needs.
    MISSING_SCOPE = "missing_scope"
    #: A system identity tried to do what only a person may (decide an approval).
    SYSTEM_IDENTITY = "system_identity"
    #: A stream ticket for a different workflow than the stream asked for.
    TICKET_WRONG_WORKFLOW = "ticket_wrong_workflow"
    #: A stream ticket past its expiry.
    TICKET_EXPIRED = "ticket_expired"


class ObjectType(str, Enum):
    """What kind of object a denial was about."""

    WORKFLOW = "workflow"
    WORKFLOW_EVENTS = "workflow_events"
    THREAD = "thread"
    APPROVAL = "approval"
    APPLICATION = "application"
    AUDIT = "audit"
    CREDENTIAL = "credential"
    EVENT_FAILURE = "event_failure"
    ROUTE = "route"


#: The request a denial happens during: `"GET /v1/workflows/<id>"`. Set by the
#: API per request; `None` outside one (a test calling a store directly).
_route: ContextVar[str | None] = ContextVar("authorization_route", default=None)


@contextmanager
def authorization_context(route: str | None) -> Iterator[None]:
    """Name the route denials inside this block are recorded against."""
    token = _route.set(route)
    try:
        yield
    finally:
        _route.reset(token)


def current_route() -> str | None:
    """The route set by `authorization_context`, if any."""
    return _route.get()


# --- Predicates --------------------------------------------------------------------


def owned_workflow_ids(access: Access):
    """A subquery of the workflow ids `access` may see; `None` for unscoped access."""
    owner = owner_of(access)
    if owner is None:
        return None
    return select(WorkflowRunModel.workflow_id).where(WorkflowRunModel.user_id == owner)


def workflow_column_filter(column, access: Access):
    """A clause limiting `column` (a workflow id) to what `access` may see."""
    owned = owned_workflow_ids(access)
    return literal(True) if owned is None else column.in_(owned)


def workflow_visible(session: Session, workflow_id: UUID, access: Access) -> bool:
    """Whether the workflow exists and `access` may see it."""
    owner = owner_of(access)
    if owner is None:
        return session.get(WorkflowModel, workflow_id) is not None
    return bool(
        session.query(
            exists().where(
                WorkflowRunModel.workflow_id == workflow_id,
                WorkflowRunModel.user_id == owner,
            )
        ).scalar()
    )


def workflow_owner(session: Session, workflow_id: UUID) -> UUID | None:
    """The owner of a workflow: the user id its runs are keyed by, if any is."""
    row = (
        session.query(WorkflowRunModel.user_id)
        .filter(
            WorkflowRunModel.workflow_id == workflow_id,
            WorkflowRunModel.user_id.isnot(None),
        )
        .order_by(WorkflowRunModel.created_at.asc())
        .first()
    )
    return row[0] if row else None


# --- Denials -----------------------------------------------------------------------


def record_denial(
    session_factory: Callable[[], Session],
    *,
    actor_id: str,
    object_type: ObjectType | str,
    reason: DenialReason,
    object_id: object | None = None,
    route: str | None = None,
    detail: str | None = None,
) -> None:
    """Append the audit row for one authorization denial. Never raises.

    The row is filed under no workflow, so it is the denied actor's record and
    nobody else's; its `target_ref` is `"<type>:<id> <route>"`. A failure to
    write it is logged and swallowed: the request is refused either way, and
    an audit outage must not turn a 404 into a 500.
    """
    route = route or current_route() or "-"
    kind = object_type.value if isinstance(object_type, ObjectType) else str(object_type)
    target = f"{kind}:{object_id if object_id is not None else '-'} {route}"
    if detail:
        target = f"{target} ({detail})"
    session = session_factory()
    try:
        AuditEventRepository(session).create(
            actor=actor_id,
            action=f"{AUDIT_DENIED_PREFIX}{reason.value}",
            target_ref=target[:500],
            result=AuditEventResult.FAILURE.value,
        )
    except Exception:  # pragma: no cover - depends on the database failing
        logger.exception("could not audit an authorization denial (%s)", reason.value)
        session.rollback()
    finally:
        session.close()
    logger.info(
        "authorization denied: %s %s for %s (%s)",
        kind,
        object_id if object_id is not None else "-",
        actor_id,
        reason.value,
        extra={"event": "authz.denied", "actor_id": actor_id, "reason": reason.value},
    )


def deny_if_not_visible(
    session_factory: Callable[[], Session],
    access: Access,
    *,
    visible: bool,
    object_type: ObjectType,
    object_id: object,
) -> bool:
    """Audit a scoped by-id miss; return `visible` so a caller can `if not ...: return None`."""
    if not visible and owner_of(access) is not None:
        record_denial(
            session_factory,
            actor_id=access.actor_id,  # type: ignore[union-attr]
            object_type=object_type,
            object_id=object_id,
            reason=DenialReason.NOT_VISIBLE,
        )
    return visible


__all__ = [
    "AUDIT_DENIED_PREFIX",
    "DenialReason",
    "ObjectType",
    "authorization_context",
    "current_route",
    "deny_if_not_visible",
    "owned_workflow_ids",
    "record_denial",
    "workflow_column_filter",
    "workflow_owner",
    "workflow_visible",
]
