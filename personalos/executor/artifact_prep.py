"""The two writes artifact prep makes, each behind a policy decision.

**Storing a draft** (`PolicyGatedDraftSink`). A tailored draft becomes an
`artifact_versions` row as soon as it has been validated. That write is
classified `WRITE_REVERSIBLE` -- a new version in this system's own database,
nothing replaced, nothing sent -- so policy allows it unattended. It is still
*asked*: the verdict is recorded like any other, and reclassifying the tool is
all it takes to put drafts behind an approval too.

**Overwriting a document** (`DocumentOverwriteExecutor`). Replacing a file the
candidate already has is the opposite case. It reaches this module only as an
approved `ActionIntent` of kind `OVERWRITE_DOCUMENT`, handed over by
`personalos.executor.tool_executor.ToolExecutor` after the graph's approval
interrupt, and is carried out as `files.overwrite_document` through the tool
gateway -- where policy is asked again, the files server checks the hash the
reviewer's diff was computed against, and the old version is backed up first.

Sending and submitting have no adapter here on purpose. This module can store
a draft and, with a human's approval, replace a file; it cannot put a draft in
front of anyone outside this system.
"""

import logging
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from personalos.domain.artifacts import DocumentOverwrite, content_sha256
from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ArtifactDraft,
    JobSearchContractError,
    NormalizedPosting,
)
from personalos.policy import (
    ApprovalGrant,
    ApprovalRequired,
    Decision,
    IntentOrigin,
    PolicyDenied,
    PolicyEngine,
    PolicyViolation,
    ToolIntent,
    fingerprint_intent,
)
from personalos.policy.permissions import Provenance
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

#: The classified tool storing a draft is evaluated as.
DRAFT_TOOL = "artifacts.create_draft"
DRAFT_SCOPES: tuple[str, ...] = ("artifacts:draft",)

FILES_SERVER = "files"
OVERWRITE_TOOL = "overwrite_document"


class DraftStore(Protocol):
    """Writes drafts as artifact versions. Implemented in `personalos.persistence`."""

    async def save(
        self,
        *,
        user_id: UUID,
        posting: NormalizedPosting,
        drafts: Sequence[ArtifactDraft],
        generated_by: str,
    ) -> Sequence[ArtifactDraft]:
        """Store the drafts and return them with their row ids."""
        ...


class PolicyGatedDraftSink:
    """Stores validated drafts, under a recorded `WRITE_REVERSIBLE` decision."""

    def __init__(
        self,
        store: DraftStore,
        policy: PolicyEngine,
        *,
        workflow_id: UUID | None = None,
    ):
        """Wrap `store`, authorizing every save through `policy`."""
        if store is None:
            raise ValueError("PolicyGatedDraftSink requires a DraftStore")
        if policy is None:
            raise ValueError(
                "PolicyGatedDraftSink requires a PolicyEngine; a draft cannot be stored "
                "without a policy decision to record against it"
            )
        self.store = store
        self.policy = policy
        self.workflow_id = workflow_id

    async def record(
        self,
        *,
        user_id: UUID,
        posting: NormalizedPosting,
        drafts: Sequence[ArtifactDraft],
        generated_by: str,
    ) -> Sequence[ArtifactDraft]:
        """Ask policy, then store. Raises `PolicyDenied` or `ApprovalRequired`."""
        verdict = self.policy.evaluate_action(
            generated_by,
            self.workflow_id,
            DRAFT_TOOL,
            fingerprint_intent(
                "artifacts",
                "create_draft",
                {
                    "user_id": str(user_id),
                    "dedupe_key": posting.dedupe_key,
                    "drafts": [
                        {
                            "artifact_type": draft.artifact_type.value,
                            "sha256": content_sha256(draft.content),
                            "evidence_ids": [ref.ref for ref in draft.evidence],
                        }
                        for draft in drafts
                    ],
                },
            ),
            list(DRAFT_SCOPES),
            # SYSTEM: the write's shape is fixed here, and the model-authored
            # content it carries has already been through the claim validator.
            Provenance(origin=IntentOrigin.SYSTEM, requested_by=generated_by),
        )
        if verdict.decision == Decision.DENY:
            raise PolicyDenied(verdict)
        if verdict.decision == Decision.REQUIRE_APPROVAL:
            raise ApprovalRequired(verdict)

        return await self.store.save(
            user_id=user_id, posting=posting, drafts=drafts, generated_by=generated_by
        )


class DocumentOverwriteExecutor:
    """Redeems an approved `OVERWRITE_DOCUMENT` action through the files tool.

    Satisfies `ActionExecutorPort`, so it is what `ToolExecutor` wraps for this
    kind of action. The reviewer's `ApprovalDecision` is restated as an
    `ApprovalGrant` on the tool intent it authorizes, which is the form the
    gateway's policy engine checks: the approval is carried across, not
    assumed, and an intent with no authorizing decision gets no grant at all.
    """

    def __init__(self, gateway: ToolGateway):
        """Take the gateway the overwrite is dispatched through."""
        if gateway is None:
            raise ValueError("DocumentOverwriteExecutor requires a ToolGateway")
        self.gateway = gateway

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        """Overwrite the document the approved intent names, backing it up first."""
        if intent.kind != ActionKind.OVERWRITE_DOCUMENT:
            raise JobSearchContractError(
                f"DocumentOverwriteExecutor cannot execute a {intent.kind.value} action"
            )
        if decision is None or not decision.authorizes(intent):
            raise PolicyViolation(
                f"overwrite action {intent.action_id} reached its executor without an "
                f"approval that authorizes it"
            )

        overwrite = DocumentOverwrite.from_payload(intent.payload)
        tool_intent = ToolIntent(
            server=FILES_SERVER,
            tool=OVERWRITE_TOOL,
            arguments={
                "path": overwrite.path,
                "content": overwrite.content,
                "expected_sha256": overwrite.expected_sha256,
                "idempotency_key": intent.idempotency_key,
            },
            origin=IntentOrigin.SYSTEM,
            mutating=True,
            requested_by=intent.requested_by,
        )
        grant = ApprovalGrant(
            intent_id=tool_intent.intent_id,
            intent_fingerprint=tool_intent.fingerprint(),
            approved_by=decision.decided_by,
            note=f"approval of action {intent.action_id} ({intent.fingerprint()})",
        )

        result = await self.gateway.dispatch(tool_intent, grant)
        if not result.success:
            logger.warning("overwrite of '%s' did not happen: %s", overwrite.path, result.error)
            return ActionReceipt(action_id=intent.action_id, ok=False, detail=result.error)

        backup_path = (result.result or {}).get("backup_path")
        return ActionReceipt(
            action_id=intent.action_id,
            ok=True,
            external_reference=backup_path,
            detail=f"replaced {overwrite.path}; previous version backed up at {backup_path}",
        )


__all__ = [
    "DRAFT_TOOL",
    "DRAFT_SCOPES",
    "DraftStore",
    "PolicyGatedDraftSink",
    "DocumentOverwriteExecutor",
]
