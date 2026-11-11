"""The one HTTP path every job board adapter reads through.

Job boards are third-party services that owe this system nothing, so every
adapter talks to them the same polite, defensive way:

- **Rate limited.** Requests from one adapter instance start at least
  `min_interval` seconds apart, across every board it watches. A `Retry-After`
  pushes that floor out for the whole adapter, not only the request that got
  it, so a rate-limited board is left alone rather than hit by the next call.
- **Retried with backoff.** A transport error, a 429 or a 5xx is retried under
  the `READ` budget from `personalos.domain.retry_policy`: exponential
  backoff with jitter, never sooner than the board's `Retry-After`. A
  `Retry-After` longer than `max_retry_after` is not slept through; the call
  fails at once with that value attached, so the caller (and the gateway's own
  retry) can decide what to do with a board that asked for minutes.
- **Read-only.** There is no method here that sends anything but a query.
  `get_json` is a GET; `query_json` is a POST only because some boards
  (Workday's career sites) take their search criteria as a JSON body. No
  other verb is reachable from an adapter.
- **Untrusting.** A 404 is `None`; anything else that is not a 2xx with a
  JSON body is `ProviderUnavailable`. What the body *says* is never
  interpreted here -- see "Untrusted posting text" in the boundaries doc.
"""

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import httpx

from personalos.domain.retry_policy import READ, RetryPolicy
from personalos.providers.base import ProviderUnavailable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
#: Default spacing between request starts, per adapter instance.
DEFAULT_MIN_INTERVAL_SECONDS = 0.5
#: The longest `Retry-After` an adapter will sleep through before giving up.
DEFAULT_MAX_RETRY_AFTER_SECONDS = 60.0
#: Identifies the client without carrying anything about its user.
USER_AGENT = "PersonalOS-job-discovery/1.0 (read-only)"

#: Statuses that say "try again later" rather than "this request is wrong".
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

Sleep = Callable[[float], Awaitable[None]]


class RateLimiter:
    """Spaces request starts at least `min_interval` seconds apart.

    One instance is shared by every request an adapter makes, so watching
    twenty boards on one provider is twenty requests at that provider's pace,
    not twenty at once.
    """

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ):
        """Take the spacing, and optionally a clock and sleep to test with."""
        if min_interval < 0:
            raise ValueError("min_interval must not be negative")
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next_allowed = float("-inf")

    async def acquire(self) -> None:
        """Wait until the next request may start, and claim that slot."""
        async with self._lock:
            wait = self._next_allowed - self._clock()
            if wait > 0:
                await self._sleep(wait)
            self._next_allowed = self._clock() + self.min_interval

    def defer(self, seconds: float) -> None:
        """Hold every request back for at least `seconds` from now."""
        self._next_allowed = max(self._next_allowed, self._clock() + seconds)


class ProviderHttp:
    """Rate-limited, retrying, read-only JSON client for one adapter."""

    def __init__(
        self,
        provider: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        retry_policy: RetryPolicy = READ,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Sleep = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ):
        """Take the adapter's name (for messages) and its pacing.

        `client` is reused when given (tests pass one over a mock transport);
        otherwise each request opens and closes its own.
        """
        self.provider = provider
        self.retry_policy = retry_policy
        self.max_retry_after = max_retry_after
        self.limiter = RateLimiter(min_interval, clock=clock, sleep=sleep)
        self._client = client
        self._timeout = timeout
        self._now = now
        self._sleep = sleep
        self._rand = rand

    async def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        """GET `url`, returning parsed JSON, or `None` on a 404."""
        return await self._request("GET", url, params=params)

    async def query_json(self, url: str, body: Mapping[str, Any]) -> Any:
        """POST a search query as JSON, returning parsed JSON, or `None` on a 404.

        For boards whose *read* API takes its criteria as a body. Nothing an
        adapter sends this way may create or change anything upstream.
        """
        return await self._request("POST", url, json=dict(body))

    async def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            attempt += 1
            await self.limiter.acquire()
            try:
                response = await self._send(method, url, **kwargs)
            except httpx.TransportError as exc:
                failure = ProviderUnavailable(f"{self.provider} request to {url} failed: {exc}")
            else:
                if response.status_code == 404:
                    return None
                if response.is_success:
                    return self._json(response, url)
                failure = self._retryable_failure(response, url)

            if not self.retry_policy.allows_retry(attempt):
                raise failure
            delay = self.retry_policy.delay(
                attempt, retry_after=failure.retry_after, rand=self._rand
            )
            logger.info(
                "%s: retrying %s %s in %.2fs (attempt %d): %s",
                self.provider,
                method,
                url,
                delay,
                attempt,
                failure.message,
            )
            await self._sleep(delay)

    def _json(self, response: httpx.Response, url: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderUnavailable(
                f"{self.provider} answered {url} with a body that is not JSON"
            ) from exc

    def _retryable_failure(self, response: httpx.Response, url: str) -> ProviderUnavailable:
        """The failure a non-2xx answer is, raised now if retrying cannot help.

        A `Retry-After` holds back every later request from this adapter, even
        when the call gives up instead of waiting it out.
        """
        retry_after = parse_retry_after(response.headers.get("Retry-After"), self._now())
        failure = ProviderUnavailable(
            f"{self.provider} answered {url} with HTTP {response.status_code}",
            retry_after=retry_after,
        )
        if response.status_code not in _RETRYABLE_STATUSES:
            # A 4xx other than "slow down" will not improve with time.
            raise failure
        if retry_after is not None:
            self.limiter.defer(retry_after)
            if retry_after > self.max_retry_after:
                raise failure
        return failure

    async def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if self._client is not None:
            return await self._client.request(method, url, headers=headers, **kwargs)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.request(method, url, headers=headers, **kwargs)


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """A `Retry-After` header in seconds: delta-seconds or an HTTP-date.

    A past date is zero; a negative, empty or unparseable value is ignored
    rather than trusted.
    """
    if value is None or not value.strip():
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - now).total_seconds())
    return seconds if seconds >= 0 else None


__all__ = [
    "DEFAULT_MAX_RETRY_AFTER_SECONDS",
    "DEFAULT_MIN_INTERVAL_SECONDS",
    "ProviderHttp",
    "RateLimiter",
    "USER_AGENT",
    "parse_retry_after",
]
