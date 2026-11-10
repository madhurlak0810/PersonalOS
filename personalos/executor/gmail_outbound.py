"""Gmail drafts and sends, over the Google MCP tools, that cannot send twice.

A Gmail `messages.send` is not idempotent: sent twice, it is two emails, and
the second cannot be taken back. The ordinary way to send twice is a lost
response -- the message goes out, the answer is lost to a timeout, and
whoever retries has no message id to show for the first attempt. So nothing
here ever re-sends on the strength of a missing response:

- every message is stamped with a `Message-ID` derived from the action's
  idempotency key (`personalos.domain.outbound_email.outbound_message_id`);
- `GmailSendExecutor` searches the Sent folder for that `Message-ID` before
  it sends, and returns the message it finds instead of sending another;
- `GmailSentReconciler` is the `ProviderReconciler` `ToolExecutor` asks when
  an earlier attempt left no recorded outcome (a failure, or a crash between
  the send and the receipt). It answers `APPLIED` with the message it found,
  or `NOT_APPLIED` only after the Sent folder has twice come back without it.
  That answer -- not the absence of a receipt -- is what permits a second
  call, and the send retry policy (`SEND`, no blind retries) means nothing
  else does.

**Drafts.** `GmailOutbound.create_draft` saves a message to the candidate's
Drafts folder (`google.gmail_create_draft`, `WRITE_REVERSIBLE`). It is made
as the node that drafted it, whose allowlist holds that one Gmail tool.

**Sends.** Every send and every lookup around one is `requested_by`
`GMAIL_SENDER`, a principal only this module stamps, and only after the
approved action has been checked against its `ApprovalDecision`. The send is
dispatched with that approval restated as the gateway's `ApprovalGrant`.
What is sent is read from the approved payload (`OutboundEmail.from_intent`)
and nowhere else: no argument names an account or a credential, and a payload
that tries to is refused before anything is dispatched.

**Attachments** are shown to the reviewer but the Gmail tool cannot carry
them, so an approved email with attachments is refused, not sent without
them: what goes out has to be what was approved.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from personalos.domain.job_search import (
    ActionIntent,
    ActionReceipt,
    ApprovalDecision,
    JobSearchContractError,
)
from personalos.domain.outbound_email import (
    EMAIL_ACTION_KINDS,
    OutboundEmail,
    outbound_message_id,
    sent_folder_query,
)
from personalos.executor.calendar import WriteAuthority
from personalos.executor.tool_executor import (
    ActionExecutorPort,
    ReconcileOutcome,
    Reconciliation,
)
from personalos.policy import ApprovalGrant, IntentOrigin, PolicyViolation, ToolIntent
from personalos.policy.rules import GMAIL_SENDER
from personalos.tools.gateway import ToolGateway

logger = logging.getLogger(__name__)

GOOGLE_SERVER = "google"

#: The node replies are drafted in, and so the one that saves them to Gmail.
DEFAULT_DRAFTING_NODE = "graph:job_search#record_recruiter_events"

#: Seconds `GmailSentReconciler` waits before asking the Sent folder a second
#: time. Gmail's search index can trail a send by a moment; one miss is not
#: proof the send never landed.
DEFAULT_CONFIRM_DELAY_SECONDS = 2.0


class GmailOutbound:
    """The Gmail half of drafting and sending, over the `google.gmail_*` tools.

    Holds no token and no HTTP client: each call is a `ToolIntent` through the
    policy gateway, and the Google MCP server brokers a token for that one call.
    """

    def __init__(
        self,
        gateway: ToolGateway,
        *,
        drafting_node: str = DEFAULT_DRAFTING_NODE,
        sender: str = GMAIL_SENDER,
    ):
        if gateway is None:
            raise ValueError("GmailOutbound requires a ToolGateway")
        self.gateway = gateway
        self.drafting_node = drafting_node
        self.sender = sender

    async def _read(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=False,
            requested_by=self.sender,
        )
        return (await self.gateway.dispatch(intent)).unwrap() or {}

    # --- drafts ---------------------------------------------------------

    async def create_draft(
        self,
        *,
        recipient: str,
        subject: str,
        body: str,
        idempotency_key: str,
        thread_id: str | None = None,
    ) -> str:
        """Save a draft in the candidate's Drafts folder; return its id. Sends nothing."""
        arguments: dict[str, Any] = {
            "to": [recipient],
            "subject": subject,
            "body": body,
            "idempotency_key": idempotency_key,
        }
        if thread_id:
            arguments["thread_id"] = thread_id
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool="gmail_create_draft",
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=True,
            requested_by=self.drafting_node,
        )
        answer = (await self.gateway.dispatch(intent)).unwrap() or {}
        return str(answer["draft_id"])

    # --- sends ----------------------------------------------------------

    async def find_sent(self, idempotency_key: str) -> str | None:
        """The Gmail id of the message sent under this key, if the Sent folder has it."""
        answer = await self._read(
            "gmail_search", {"query": sent_folder_query(idempotency_key), "max_results": 5}
        )
        messages = answer.get("messages") or []
        if len(messages) > 1:
            logger.warning(
                "%s sent messages carry the Message-ID of idempotency key '%s'; using %s",
                len(messages),
                idempotency_key,
                messages[0].get("message_id"),
            )
        return str(messages[0]["message_id"]) if messages else None

    async def original(self, provider_message_id: str) -> tuple[str | None, str | None]:
        """`(Message-ID header, thread id)` of the message a reply answers, if it still exists."""
        answer = await self._read(
            "gmail_read_message", {"message_id": provider_message_id, "missing_ok": True}
        )
        if not answer.get("found", True):
            return None, None
        header = answer.get("rfc822_message_id") or None
        if header and not (header.startswith("<") and header.endswith(">")):
            header = None
        return header, answer.get("thread_id") or None

    async def send(
        self,
        email: OutboundEmail,
        *,
        authority: WriteAuthority,
        in_reply_to_header: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Send `email` under the approval in `authority`; return Gmail's answer."""
        action, decision = authority.intent, authority.decision
        if decision is None or not decision.authorizes(action):
            raise PolicyViolation(
                f"google.gmail_send_message for action {action.action_id}: the approval does "
                f"not authorize this action"
            )
        arguments: dict[str, Any] = {
            "to": [email.recipient],
            "subject": email.subject,
            "body": email.body,
            "message_id": outbound_message_id(action.idempotency_key),
            "idempotency_key": action.idempotency_key,
        }
        if thread_id:
            arguments["thread_id"] = thread_id
        if in_reply_to_header:
            arguments["in_reply_to"] = in_reply_to_header
        intent = ToolIntent(
            server=GOOGLE_SERVER,
            tool="gmail_send_message",
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=True,
            requested_by=self.sender,
        )
        grant = ApprovalGrant(
            intent_id=intent.intent_id,
            intent_fingerprint=intent.fingerprint(),
            approved_by=decision.decided_by,
            note=f"approval of action {action.action_id} ({action.fingerprint()})",
        )
        return (await self.gateway.dispatch(intent, grant)).unwrap() or {}


def _refused(intent: ActionIntent, why: str) -> ActionReceipt:
    logger.warning("not sending email action %s: %s", intent.action_id, why)
    return ActionReceipt(action_id=intent.action_id, ok=False, detail=f"not sent: {why}")


class GmailSendExecutor:
    """Carries out approved `SEND_EMAIL` and `SEND_RECRUITER_MESSAGE` actions.

    The provider adapter `ToolExecutor` wraps for those kinds; other kinds go
    to `fallback`. Looks in the Sent folder before every send.
    """

    def __init__(self, client: GmailOutbound, *, fallback: ActionExecutorPort | None = None):
        if client is None:
            raise ValueError("GmailSendExecutor requires a GmailOutbound client")
        self.client = client
        self.fallback = fallback

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        if intent.kind not in EMAIL_ACTION_KINDS:
            if self.fallback is None:
                raise JobSearchContractError(
                    f"GmailSendExecutor cannot execute '{intent.kind.value}' and has no fallback"
                )
            return await self.fallback.execute(intent, decision)

        if decision is None or not decision.authorizes(intent):
            raise PolicyViolation(
                f"email action {intent.action_id} reached its executor without an approval "
                f"that authorizes it"
            )
        try:
            email = OutboundEmail.from_intent(intent)
        except (JobSearchContractError, ValueError) as exc:
            return _refused(intent, str(exc))
        if email.attachments:
            return _refused(
                intent,
                f"it has {len(email.attachments)} attachment(s), which the Gmail tool cannot "
                f"send; sending it without them would not be the message that was approved",
            )

        existing = await self.client.find_sent(intent.idempotency_key)
        if existing is not None:
            logger.info(
                "email for idempotency key '%s' is already in the Sent folder as %s; "
                "not sending again",
                intent.idempotency_key,
                existing,
            )
            return ActionReceipt(
                action_id=intent.action_id,
                ok=True,
                external_reference=existing,
                detail="already in the Sent folder; not sent again",
            )

        header, thread_id = None, email.thread_id
        if email.in_reply_to:
            header, original_thread = await self.client.original(email.in_reply_to)
            thread_id = thread_id or original_thread
        sent = await self.client.send(
            email,
            authority=WriteAuthority(intent, decision),
            in_reply_to_header=header,
            thread_id=thread_id,
        )
        return ActionReceipt(
            action_id=intent.action_id,
            ok=True,
            external_reference=str(sent.get("message_id")),
            detail="sent as a reply" if email.is_reply else "sent as a new thread",
        )


class GmailSentReconciler:
    """Tells `ToolExecutor` whether an email whose outcome was never recorded went out.

    A lookup that raises is left to raise: `ToolExecutor` reads that as "the
    provider could not be asked" and sends nothing.
    """

    def __init__(
        self,
        client: GmailOutbound,
        *,
        confirm_delay: float = DEFAULT_CONFIRM_DELAY_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.client = client
        self.confirm_delay = confirm_delay
        self.sleep = sleep

    async def reconcile(self, intent: ActionIntent) -> Reconciliation:
        if intent.kind not in EMAIL_ACTION_KINDS:
            return Reconciliation(ReconcileOutcome.INDETERMINATE)
        found = await self.client.find_sent(intent.idempotency_key)
        if found is None and self.confirm_delay > 0:
            await self.sleep(self.confirm_delay)
            found = await self.client.find_sent(intent.idempotency_key)
        if found is None:
            return Reconciliation(ReconcileOutcome.NOT_APPLIED)
        return Reconciliation(
            ReconcileOutcome.APPLIED,
            ActionReceipt(
                action_id=intent.action_id,
                ok=True,
                external_reference=found,
                detail="found in the Sent folder; not sent again",
            ),
        )


__all__ = [
    "DEFAULT_CONFIRM_DELAY_SECONDS",
    "DEFAULT_DRAFTING_NODE",
    "GmailOutbound",
    "GmailSendExecutor",
    "GmailSentReconciler",
]
