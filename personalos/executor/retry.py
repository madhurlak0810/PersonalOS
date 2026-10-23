"""Automatic retries for tool dispatch, and what is never retried.

A transient failure (`PersonalOSError.retryable`) is worth another attempt. A
policy verdict is not: `PolicyDenied` and `ApprovalRequired` are answers, and
asking again gets the same answer. Both are terminal here, so a denied action
surfaces to the caller on the first attempt instead of being quietly re-asked.

How many retries a transient failure gets is the tool's `RetryPolicy`
(`personalos.domain.retry_policy`), chosen from its permission class: reads
get a few, drafts fewer, sends none. A calendar write's one retry needs a
reconciliation first, which only `ToolExecutor` can do, so through this path a
calendar write is not retried at all. Retries back off exponentially with
jitter, and never come sooner than a provider's `Retry-After` -- a provider
asking for longer than the policy's `max_delay` gets the failure back instead
of a caller parked on a long sleep.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from personalos.domain.errors import PersonalOSError
from personalos.domain.retry_policy import (
    CALENDAR_WRITE,
    DRAFT,
    READ,
    SEND,
    RetryPolicy,
    retry_after_seconds,
)
from personalos.policy import ApprovalGrant, PolicyError, ToolIntent
from personalos.policy.permissions import DEFAULT_TOOL_PERMISSIONS, PermissionClass
from personalos.tools.gateway import ToolGateway, ToolResult

logger = logging.getLogger(__name__)

#: Total attempts for a read, including the first. Kept for callers that
#: predate per-class policies.
DEFAULT_MAX_ATTEMPTS = READ.max_attempts

#: Delay window before the second attempt; doubles on each one after.
DEFAULT_BASE_DELAY_SECONDS = READ.base_delay

#: Tools whose writes are recovered by reconciliation before their one retry.
CALENDAR_WRITE_TOOLS = frozenset({"google.calendar_create_event", "google.calendar_update_event"})

_POLICY_BY_CLASS: dict[PermissionClass, RetryPolicy] = {
    PermissionClass.READ_LOCAL: READ,
    PermissionClass.READ_EXTERNAL: READ,
    PermissionClass.WRITE_REVERSIBLE: DRAFT,
    PermissionClass.WRITE_EXTERNAL: SEND,
    PermissionClass.DESTRUCTIVE: SEND,
    PermissionClass.SENSITIVE: SEND,
}


def retry_policy_for(tool_ref: str) -> RetryPolicy:
    """The retry budget for a classified tool. An unclassified one gets none."""
    if tool_ref in CALENDAR_WRITE_TOOLS:
        return CALENDAR_WRITE
    permission = DEFAULT_TOOL_PERMISSIONS.get(tool_ref)
    if permission is None:
        return SEND
    return _POLICY_BY_CLASS.get(permission.permission_class, SEND)


def is_retryable(error: BaseException) -> bool:
    """True when an automatic retry of the same dispatch is allowed.

    The `PolicyError` check comes first and does not consult `retryable`: a
    policy outcome stays terminal even if a subclass were ever mislabelled.
    """
    if isinstance(error, PolicyError):
        return False
    return isinstance(error, PersonalOSError) and error.retryable


async def dispatch_with_retry(
    gateway: ToolGateway,
    intent: ToolIntent,
    approval: ApprovalGrant | None = None,
    *,
    policy: RetryPolicy | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> ToolResult:
    """Dispatch an intent, retrying only failures that are safe to retry.

    `policy` defaults to the tool's own (`retry_policy_for`); `max_attempts`
    and `base_delay` override its budget and backoff. Every attempt goes back
    through the gateway, so every attempt is re-authorized and gets its own
    policy decision row.
    """
    policy = policy or retry_policy_for(intent.tool_ref)
    if policy.requires_reconciliation:
        # Nothing here can ask the provider whether the write landed.
        policy = policy.with_max_attempts(1)
    if max_attempts is not None:
        policy = policy.with_max_attempts(max_attempts)
    if base_delay is not None:
        policy = RetryPolicy(
            policy.operation_class,
            max_retries=policy.max_retries,
            base_delay=base_delay,
            max_delay=max(policy.max_delay, base_delay),
        )

    attempt = 1
    while True:
        try:
            return await gateway.dispatch(intent, approval)
        except Exception as error:
            if not policy.allows_retry(attempt) or not is_retryable(error):
                raise
            retry_after = retry_after_seconds(error)
            if retry_after is not None and retry_after > policy.max_delay:
                logger.warning(
                    "dispatch of %s failed (attempt %s/%s, %s) and the provider asked for "
                    "%.0fs, longer than this call waits; not retrying: %s",
                    intent.tool_ref,
                    attempt,
                    policy.max_attempts,
                    intent.context.as_log_str(),
                    retry_after,
                    error,
                )
                raise
            delay = policy.delay(attempt, retry_after=retry_after, rand=rand)
            logger.warning(
                "dispatch of %s failed (attempt %s/%s, %s); retrying in %.2fs: %s",
                intent.tool_ref,
                attempt,
                policy.max_attempts,
                intent.context.as_log_str(),
                delay,
                error,
            )
            await sleep(delay)
            attempt += 1


__all__ = [
    "CALENDAR_WRITE_TOOLS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_BASE_DELAY_SECONDS",
    "dispatch_with_retry",
    "is_retryable",
    "retry_policy_for",
]
