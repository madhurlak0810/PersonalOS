"""The File subgraph's way out: every file access as a `files.*` intent through the gateway.

`GatewayFileOperations` satisfies `personalos.graphs.file_task.FileOperations`.
It holds no file handle and resolves no path: each method builds one
`ToolIntent` for the sandboxed files MCP server and dispatches it through the
policy-enforcing `ToolGateway`, so

- policy decides, from the intent's `requested_by` (the node that asked),
  whether that node may call that tool at all, records the verdict, and holds
  deletes, overwrites and bulk moves for approval;
- the files server resolves every path canonically and refuses -- and
  audits -- anything outside its allowed roots;
- whatever comes back has already been redacted by the gateway.

**An approval grant is minted here, and only from a person's plan approval.**
Policy binds an `ApprovalGrant` to one intent's id and fingerprint, and the
intent is built here, at call time; the person approved the *plan* hours
earlier. So when a call needs approval, a grant for exactly this intent is
minted from the `PlanApproval` the graph passed -- and only after checking
that the decision approves that plan action and was made by a person, not a
system identity. With no `PlanApproval`, a call that needs one is refused by
the policy engine.

Every intent carries the run's `ExecutionContext`, so the decision log and the
audit trail attribute the call to the actor the run belongs to.
"""

import base64
import binascii
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langgraph.config import get_config

from personalos.domain.auth import is_system_actor
from personalos.domain.context import ExecutionContext
from personalos.domain.errors import ValidationFailed
from personalos.domain.files import (
    MAX_INVENTORY_ENTRIES,
    FileContent,
    FileEntry,
    FileInventory,
    FileStat,
    PlanApproval,
    categorize,
)
from personalos.policy import ApprovalGrant, IntentOrigin, ToolIntent
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

FILES_SERVER = "files"

#: Turns a document's bytes into text: `(path, content) -> text`, or `None`
#: for a document with none to give. Supplied by the composition root (PDF and
#: DOCX extraction live in `retrieval`); UTF-8 text needs nothing.
TextExtractor = Callable[[str, bytes], str | None]


class GatewayFileOperations:
    """`FileOperations` over the tool gateway."""

    def __init__(
        self,
        gateway: ToolGateway,
        *,
        text_extractor: TextExtractor | None = None,
        context: Callable[[], ExecutionContext] | None = None,
    ):
        """Take the gateway every call is dispatched through.

        `context` supplies the run's `ExecutionContext`; by default it is read
        from the LangGraph run config (`workflow_id`, `correlation_id`,
        `actor_id`), falling back to a fresh system context outside a run.
        """
        if gateway is None:
            raise ValueError("GatewayFileOperations requires a ToolGateway")
        self.gateway = gateway
        self.text_extractor = text_extractor
        self.context = context or _context_from_run

    # --- reads ----------------------------------------------------------------

    async def list_files(self, *, requested_by: str) -> FileInventory:
        payload = await self._call(
            "list_files", {"max_entries": MAX_INVENTORY_ENTRIES}, requested_by=requested_by
        )
        entries = []
        for raw in payload.get("entries") or ():
            try:
                entries.append(
                    FileEntry(
                        path=str(raw["path"]),
                        name=str(raw["name"]),
                        size=int(raw["size"]),
                        modified_at=datetime.fromisoformat(str(raw["modified_at"])),
                        sha256=str(raw["sha256"]),
                        category=categorize(str(raw["name"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("files.list_files returned an unreadable entry; skipped")
        return FileInventory(
            roots=tuple(str(root) for root in payload.get("roots") or ()),
            entries=tuple(entries),
            truncated=bool(payload.get("truncated")),
        )

    async def stat_file(self, path: str, *, requested_by: str) -> FileStat:
        payload = await self._call("stat_file", {"path": path}, requested_by=requested_by)
        modified = payload.get("modified_at")
        return FileStat(
            path=str(payload.get("path") or path),
            exists=bool(payload.get("exists")),
            size=payload.get("size"),
            sha256=payload.get("sha256"),
            modified_at=datetime.fromisoformat(modified) if modified else None,
        )

    async def read_file(self, path: str, *, requested_by: str) -> FileContent:
        payload = await self._call("read_file", {"path": path}, requested_by=requested_by)
        try:
            canonical, sha256 = str(payload["path"]), str(payload["sha256"])
            content, encoding = payload["content"], payload.get("encoding", "utf-8")
            size = int(payload.get("size") or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationFailed("files.read_file returned an unreadable payload") from exc
        if encoding == "utf-8":
            text: str | None = str(content)
        else:
            try:
                data = base64.b64decode(str(content), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValidationFailed("files.read_file returned invalid base64") from exc
            text = self._extract(canonical, data)
        return FileContent(path=canonical, sha256=sha256, size=size, text=text)

    def _extract(self, path: str, data: bytes) -> str | None:
        if self.text_extractor is None:
            return None
        try:
            return self.text_extractor(path, data) or None
        except Exception:  # noqa: BLE001 - an unreadable document has no text, not an error
            logger.info("no text could be extracted from %s", path)
            return None

    # --- writes ---------------------------------------------------------------

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str | None,
        idempotency_key: str,
        requested_by: str,
        approval: PlanApproval | None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "path": path,
            "content": content,
            "encoding": "utf-8",
            "idempotency_key": idempotency_key,
        }
        if expected_sha256 is not None:
            arguments["expected_sha256"] = expected_sha256
        return await self._call(
            "write_file", arguments, requested_by=requested_by, mutating=True, approval=approval
        )

    async def move_file(
        self,
        source: str,
        destination: str,
        *,
        expected_sha256: str,
        plan_size: int,
        idempotency_key: str,
        requested_by: str,
        approval: PlanApproval | None,
    ) -> dict[str, Any]:
        return await self._call(
            "move_file",
            {
                "source": source,
                "destination": destination,
                "expected_sha256": expected_sha256,
                "plan_size": plan_size,
                "idempotency_key": idempotency_key,
            },
            requested_by=requested_by,
            mutating=True,
            approval=approval,
        )

    async def delete_file(
        self,
        path: str,
        *,
        expected_sha256: str,
        idempotency_key: str,
        requested_by: str,
        approval: PlanApproval | None,
    ) -> dict[str, Any]:
        return await self._call(
            "delete_file",
            {"path": path, "expected_sha256": expected_sha256, "idempotency_key": idempotency_key},
            requested_by=requested_by,
            mutating=True,
            approval=approval,
        )

    # --- dispatch ---------------------------------------------------------------

    async def _call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        requested_by: str,
        mutating: bool = False,
        approval: PlanApproval | None = None,
    ) -> dict[str, Any]:
        """Dispatch one `files.<tool>` intent; raise on refusal or failure.

        SYSTEM origin: the call's shape is fixed by the validated plan, and
        whatever a model contributed to the plan is what the reviewer saw.
        """
        intent = ToolIntent(
            server=FILES_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=mutating,
            requested_by=requested_by,
            context=self.context(),
        )
        result = await self.gateway.dispatch(intent, approval=_grant_for(intent, approval))
        payload = result.unwrap()
        return payload if isinstance(payload, dict) else {}


def _grant_for(intent: ToolIntent, approval: PlanApproval | None) -> ApprovalGrant | None:
    """A grant for exactly this intent, from a person's approval of its plan; else None."""
    if approval is None:
        return None
    decision = approval.decision
    if not decision.authorizes(approval.action):
        logger.warning(
            "a plan approval that does not authorize its action was passed with files.%s; "
            "no grant minted",
            intent.tool,
        )
        return None
    if is_system_actor(decision.decided_by):
        logger.warning(
            "a plan approval decided by system identity %r was passed with files.%s; "
            "no grant minted",
            decision.decided_by,
            intent.tool,
        )
        return None
    return ApprovalGrant(
        intent_id=intent.intent_id,
        intent_fingerprint=intent.fingerprint(),
        approved_by=decision.decided_by,
        note=f"file plan {approval.action.action_id}",
    )


def _context_from_run() -> ExecutionContext:
    """The run's correlation identity, from the LangGraph config when there is one."""
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return ExecutionContext()
    fields: dict[str, Any] = {}
    for key in ("workflow_id", "correlation_id"):
        if configurable.get(key):
            fields[key] = configurable[key]
    if configurable.get("actor_id"):
        fields["actor_id"] = str(configurable["actor_id"])
    try:
        return ExecutionContext(**fields)
    except ValueError:
        return ExecutionContext()


__all__ = ["FILES_SERVER", "GatewayFileOperations", "TextExtractor"]
