"""The durable policy decision log: one `policy_decisions` row per verdict.

Implements the `PolicyDecisionLog` port the policy engine calls before it
returns an outcome. It satisfies that port structurally -- persistence may not
import the policy layer -- and is wired to the engine in `personalos.bootstrap`.

This module stores what it is given and decides nothing.
"""

import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.persistence.repositories import PolicyDecisionRepository, WorkflowRepository

logger = logging.getLogger(__name__)


class SqlPolicyDecisionLog:
    """Writes each policy verdict in its own committed transaction.

    Its own session per write, like `JournaledActionExecutor`: the row must be
    durable before the engine returns, and must survive whatever the caller's
    transaction goes on to do -- a denial that rolled back with the request it
    denied would leave no trace of having happened.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Take the factory each write opens its session from."""
        self.session_factory = session_factory

    def record(
        self,
        *,
        principal: str,
        workflow_id: UUID | None,
        tool: str,
        args_hash: str,
        decision: str,
        requested_scopes: list[str],
    ) -> None:
        """Insert and commit one decision row."""
        session = self.session_factory()
        try:
            # `policy_decisions.workflow_id` references `workflows`. A run that
            # was never registered as a workflow still gets its decision
            # recorded, just without the link, rather than failing the insert.
            if (
                workflow_id is not None
                and WorkflowRepository(session).get_by_id(workflow_id) is None
            ):
                logger.debug(
                    "workflow %s is not registered; recording decision unlinked", workflow_id
                )
                workflow_id = None
            PolicyDecisionRepository(session).create(
                principal=principal,
                workflow_id=workflow_id,
                tool=tool,
                args_hash=args_hash,
                decision=decision,
                requested_scopes=requested_scopes,
            )
        finally:
            session.close()


__all__ = ["SqlPolicyDecisionLog"]
