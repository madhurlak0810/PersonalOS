"""The audit row for a path the files server refused.

The sandbox (`mcp_servers.files.sandbox`) refuses any path whose canonical
form -- after `..` and every symlink -- is outside the allowed roots or inside
a credential directory. Each refusal is written here as an `audit_events` row,
so an attempt to reach outside the roots is on the record whoever made it and
whatever the caller does with the error.

Filed under no workflow, as authorization denials are
(`personalos.persistence.ownership.record_denial`): the row is the denied
actor's, and naming a workflow the actor may not own would leak it. The
requested path arrives already redacted.
"""

import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.models import AuditEventResult
from personalos.domain.redaction import redact_text
from personalos.persistence.repositories import AuditEventRepository

logger = logging.getLogger(__name__)

#: `audit_events.action` of a refused path.
FILE_PATH_DENIED_ACTION = "files.path_denied"


def record_file_path_denial(
    session_factory: Callable[[], Session],
    *,
    actor_id: str,
    tool: str,
    requested_path: str,
    reason: str,
    workflow_id: UUID | None = None,
) -> None:
    """Append the audit row for one refused path. Never raises."""
    target = f"files.{tool} {redact_text(requested_path)} ({redact_text(reason)})"
    if workflow_id is not None:
        target = f"{target} workflow={workflow_id}"
    session = session_factory()
    try:
        AuditEventRepository(session).create(
            actor=actor_id,
            action=FILE_PATH_DENIED_ACTION,
            target_ref=target[:500],
            result=AuditEventResult.FAILURE.value,
        )
    except Exception:  # pragma: no cover - depends on the database failing
        logger.exception("could not audit a refused file path")
        session.rollback()
    finally:
        session.close()


__all__ = ["FILE_PATH_DENIED_ACTION", "record_file_path_denial"]
