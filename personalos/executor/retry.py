"""Automatic retries for tool dispatch, and what is never retried.

A transient failure (`PersonalOSError.retryable`) is worth another attempt. A
policy verdict is not: `PolicyDenied` and `ApprovalRequired` are answers, and
asking again gets the same answer. Both are terminal here, so a denied action
surfaces to the caller on the first attempt instead of being quietly re-asked.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from personalos.domain.errors import PersonalOSError
from personalos.policy import ApprovalGrant, PolicyError, ToolIntent
from personalos.tools.gateway import ToolGateway, ToolResult

logger = logging.getLogger(__name__)

#: Total attempts, including the first.
DEFAULT_MAX_ATTEMPTS = 3

#: Delay before the second attempt; doubles on each one after.
DEFAULT_BASE_DELAY_SECONDS = 0.2


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
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ToolResult:
    """Dispatch an intent, retrying only failures that are safe to retry.

    Every attempt goes back through the gateway, so every attempt is
    re-authorized and gets its own policy decision row.
    """
    attempt = 1
    while True:
        try:
            return await gateway.dispatch(intent, approval)
        except Exception as error:
            if attempt >= max_attempts or not is_retryable(error):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "dispatch of %s failed (attempt %s/%s, %s); retrying in %.1fs: %s",
                intent.tool_ref,
                attempt,
                max_attempts,
                intent.context.as_log_str(),
                delay,
                error,
            )
            await sleep(delay)
            attempt += 1


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_BASE_DELAY_SECONDS",
    "is_retryable",
    "dispatch_with_retry",
]
