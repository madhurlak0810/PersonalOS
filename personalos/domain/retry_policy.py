"""How many times an operation may be retried automatically, and how far apart.

The budget depends on what a repeat could do, not on what failed:

- **Reads** are retried freely (`READ`, 3 retries): asking again changes nothing.
- **Drafts** are retried a little (`DRAFT`, 2 retries): a duplicate draft is a
  nuisance in this system's own store, never something a third party sees.
- **Sends** are never retried blindly (`SEND`, 0 retries): a send that timed
  out may have landed, and a second one cannot be taken back. Recovering one
  is reconciliation's job, under its idempotency key, not a loop's.
- **Calendar writes** get one retry, and only after reconciliation
  (`CALENDAR_WRITE`): the provider is asked first whether the write landed,
  and the retry is made only when it positively did not.
- **Event delivery** (`EVENT_DELIVERY`, 4 retries) is what the outbox
  dispatcher allows a handler; handlers are idempotent on the event's
  `dedupe_key`, so a redelivery applies nothing twice.

Delays grow exponentially with "equal jitter": retry *n* waits between half
and all of `base_delay * 2**(n-1)` (capped at `max_delay`). Half the window is
fixed, so each retry waits at least as long as the one before it; the other
half is random, so workers that failed together do not retry together. A
provider's `Retry-After` is a floor: the retry never comes sooner than the
provider asked.

Pure values, like the rest of `domain`; nothing here sleeps.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum


class OperationClass(str, Enum):
    """What kind of operation a retry budget is for."""

    READ = "read"
    DRAFT = "draft"
    SEND = "send"
    CALENDAR_WRITE = "calendar_write"
    EVENT_DELIVERY = "event_delivery"


@dataclass(frozen=True)
class RetryPolicy:
    """An automatic-retry budget and the backoff between its attempts."""

    operation_class: OperationClass
    #: Retries after the first attempt. Zero means the first failure is final.
    max_retries: int
    #: Delay window before the first retry, in seconds; doubles for each after.
    base_delay: float = 0.2
    #: Largest backoff window, in seconds. A `Retry-After` may exceed it.
    max_delay: float = 30.0
    #: A retry is allowed only once the provider has confirmed the failed
    #: attempt did not take effect.
    requires_reconciliation: bool = False

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.base_delay < 0 or self.max_delay < self.base_delay:
            raise ValueError("delays must satisfy 0 <= base_delay <= max_delay")

    @property
    def max_attempts(self) -> int:
        """Attempts in all, the first one included."""
        return 1 + self.max_retries

    def allows_retry(self, attempt: int) -> bool:
        """Whether attempt number `attempt` (counting from 1) may be followed by another."""
        return attempt <= self.max_retries

    def with_max_attempts(self, max_attempts: int) -> "RetryPolicy":
        """This policy with a different total attempt budget."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        return replace(self, max_retries=max_attempts - 1)

    def delay(
        self,
        retry: int,
        *,
        retry_after: float | None = None,
        rand: Callable[[], float] = random.random,
    ) -> float:
        """Seconds to wait before retry number `retry` (counting from 1)."""
        if retry < 1:
            raise ValueError("retry counts from 1")
        window = min(self.max_delay, self.base_delay * 2 ** (retry - 1))
        delay = window / 2 + rand() * window / 2
        if retry_after is not None and retry_after > delay:
            return float(retry_after)
        return delay


READ = RetryPolicy(OperationClass.READ, max_retries=3)
DRAFT = RetryPolicy(OperationClass.DRAFT, max_retries=2)
SEND = RetryPolicy(OperationClass.SEND, max_retries=0)
CALENDAR_WRITE = RetryPolicy(
    OperationClass.CALENDAR_WRITE, max_retries=1, base_delay=1.0, requires_reconciliation=True
)
EVENT_DELIVERY = RetryPolicy(
    OperationClass.EVENT_DELIVERY, max_retries=4, base_delay=5.0, max_delay=600.0
)


def retry_after_seconds(error: BaseException) -> float | None:
    """The `Retry-After` a provider attached to `error`, in seconds, if any.

    Read from a `retry_after` attribute, which `PersonalOSError` carries; a
    negative or unparseable value is ignored rather than trusted.
    """
    value = getattr(error, "retry_after", None)
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


__all__ = [
    "CALENDAR_WRITE",
    "DRAFT",
    "EVENT_DELIVERY",
    "READ",
    "SEND",
    "OperationClass",
    "RetryPolicy",
    "retry_after_seconds",
]
