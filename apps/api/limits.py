"""How much a single caller may ask of the API: rates, concurrency, auth failures.

A buggy UI loop, a stolen token or a hostile client is bounded here. Every
limit is keyed on what the server verified -- the authenticated actor, or for
authentication failures the connection's source address -- and never on a
header or body field the client chose. See docs/LIMITS.md for the defaults.

**Rates.** Each limited route spends one token from the actor's bucket for
that limit (`RateLimit`: up to `capacity` at once, refilled at `capacity` per
`period_seconds`). An empty bucket is a 429 with `Retry-After` -- the seconds
until the next token. The routes and the limit each spends from:

====================================  =========================  ==================
route                                 limit                      setting
====================================  =========================  ==================
``POST /v1/chat`` (a workflow start)  ``workflow_start``         ``API_RATE_WORKFLOW_START_PER_MINUTE``
``POST /v1/workflows/{id}/resume``    ``workflow_resume``        ``API_RATE_WORKFLOW_RESUME_PER_MINUTE``
``POST /v1/approvals/{id}/decision``  ``approval_decision``      ``API_RATE_APPROVAL_DECISION_PER_MINUTE``
``POST /v1/workflows/{id}/retry``,    ``failure_action``         ``API_RATE_FAILURE_ACTION_PER_MINUTE``
``.../dismiss``, event retry
``POST /v1/evidence/documents``       ``evidence_ingest``        ``API_RATE_EVIDENCE_INGEST_PER_MINUTE``
====================================  =========================  ==================

The buckets live behind `RateLimitStore`. `InMemoryRateLimitStore` is the one
there is: per process, so with `API_WORKERS=4` an actor gets up to four times
each budget. A Postgres or Redis store implements the same `consume` and is
passed to `ApiLimits`; nothing else changes.

**Concurrency.** An actor may have at most `API_MAX_ACTIVE_WORKFLOWS`
workflows with a command queued or running (read from the command queue, so
it holds across processes, and frees itself when the worker finishes or
parks a run), and at most `API_MAX_SSE_CONNECTIONS` open event streams (in
this process). Over either is a 429, not an unbounded queue.

**Authentication failures.** `AuthFailureThrottle` counts failed credentials
per source address. The first `API_AUTH_FREE_FAILURES` cost nothing -- a typo
is not a lockout -- and each one after doubles the wait, from
`API_AUTH_BACKOFF_BASE_SECONDS` up to `API_AUTH_BACKOFF_MAX_SECONDS`. While an
address is waiting, every credential from it is refused with a 429 *before*
it is checked, the right one included, so guessing gets no faster by
guessing harder. A success clears the address's count.

**Idempotent replays are free.** `POST /v1/chat` with an `Idempotency-Key`
it has already answered returns that answer without spending budget or
creating a workflow (`apps.api.routes.workflows`).

**Health is never limited.** `/health` has no dependencies and the body
middlewares pass it straight through.

Every refusal is reported by `personalos.observability.limits.record_limit_hit`.
"""

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn, Protocol, runtime_checkable

from fastapi import Depends, Request

from personalos.config import Settings, settings
from personalos.domain.auth import Principal
from personalos.domain.errors import RateLimited
from personalos.observability.limits import (
    LimitHitCounter,
    default_hit_counter,
    normalize_path,
    record_limit_hit,
)


class LimitName:
    """The `limit` a refusal names: in the 429's details, the log line and the metric."""

    WORKFLOW_START = "workflow_start"
    WORKFLOW_RESUME = "workflow_resume"
    APPROVAL_DECISION = "approval_decision"
    FAILURE_ACTION = "failure_action"
    EVIDENCE_INGEST = "evidence_ingest"
    ACTIVE_WORKFLOWS = "active_workflows"
    SSE_CONNECTIONS = "sse_connections"
    AUTH_FAILURES = "auth_failures"
    BODY_SIZE = "body_size"
    JSON_DEPTH = "json_depth"
    JSON_STRING_LENGTH = "json_string_length"
    JSON_ENCODING = "json_encoding"


#: The largest page any list endpoint returns. A route may set a lower bound;
#: none may set a higher one (`tests/unit/test_api_limits.py` checks).
MAX_PAGE_SIZE = 200

#: `Retry-After` for a concurrency cap: there is no refill to wait for, only
#: someone else's work finishing, so this is a polite polling interval.
CONCURRENCY_RETRY_AFTER_SECONDS = 5.0


# --- Rates -------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimit:
    """A token bucket: `capacity` at once, refilled at `capacity` per `period_seconds`."""

    capacity: int
    period_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.period_seconds <= 0:
            raise ValueError("a rate limit needs a positive capacity and period")

    @property
    def refill_per_second(self) -> float:
        return self.capacity / self.period_seconds


@dataclass(frozen=True)
class RateDecision:
    """Whether one request may proceed, and if not, how long until it could."""

    allowed: bool
    remaining: int
    retry_after: float = 0.0


@runtime_checkable
class RateLimitStore(Protocol):
    """Where buckets live. The seam a shared (Postgres, Redis) store replaces."""

    def consume(self, key: str, limit: RateLimit) -> RateDecision:
        """Spend one token from `key`'s bucket under `limit`, if there is one."""
        ...


@dataclass
class _Bucket:
    tokens: float
    updated: float


class InMemoryRateLimitStore:
    """Token buckets in this process's memory, bounded to `max_keys` buckets.

    Past `max_keys`, the least recently used bucket is forgotten -- which
    gives that key a full bucket back, so the bound is set well above the
    number of actors a local install has.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000):
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def consume(self, key: str, limit: RateLimit) -> RateDecision:
        with self._lock:
            now = self._clock()
            bucket = self._buckets.pop(key, None)
            if bucket is None:
                bucket = _Bucket(tokens=float(limit.capacity), updated=now)
            else:
                elapsed = max(0.0, now - bucket.updated)
                bucket.tokens = min(
                    float(limit.capacity), bucket.tokens + elapsed * limit.refill_per_second
                )
                bucket.updated = now
            self._buckets[key] = bucket
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return RateDecision(allowed=True, remaining=int(bucket.tokens))
            wait = (1.0 - bucket.tokens) / limit.refill_per_second
            return RateDecision(allowed=False, remaining=0, retry_after=wait)


# --- Concurrency ------------------------------------------------------------------


class Slot:
    """One held unit of a `ConcurrencyLimiter`. Releasing it twice is harmless."""

    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()

    @property
    def released(self) -> bool:
        return self._released


class ConcurrencyLimiter:
    """Counts what each key holds open right now, in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[str, int] = {}

    def acquire(self, key: str, maximum: int) -> Slot | None:
        """A slot for `key`, or `None` when it already holds `maximum`."""
        with self._lock:
            held = self._held.get(key, 0)
            if held >= maximum:
                return None
            self._held[key] = held + 1
        return Slot(lambda: self._release(key))

    def held(self, key: str) -> int:
        with self._lock:
            return self._held.get(key, 0)

    def _release(self, key: str) -> None:
        with self._lock:
            held = self._held.get(key, 0) - 1
            if held > 0:
                self._held[key] = held
            else:
                self._held.pop(key, None)


# --- Authentication failures ------------------------------------------------------


@dataclass
class _FailureRecord:
    failures: int = 0
    blocked_until: float = 0.0
    last_failure: float = 0.0


class AuthFailureThrottle:
    """Exponential backoff on failed credentials, per source address.

    `free_failures` failures cost nothing; failure number `free_failures + n`
    blocks the address for `base_seconds * 2 ** (n - 1)`, capped at
    `max_seconds`. An address that has not failed for `forget_after_seconds`
    starts over, and at most `max_sources` addresses are remembered.
    """

    def __init__(
        self,
        *,
        free_failures: int = 3,
        base_seconds: float = 1.0,
        max_seconds: float = 300.0,
        forget_after_seconds: float = 900.0,
        max_sources: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        if free_failures < 1:
            raise ValueError("API_AUTH_FREE_FAILURES must be at least 1: a typo is not an attack")
        self.free_failures = free_failures
        self.base_seconds = base_seconds
        self.max_seconds = max_seconds
        self.forget_after_seconds = max(forget_after_seconds, max_seconds)
        self._max_sources = max_sources
        self._clock = clock
        self._lock = threading.Lock()
        self._sources: OrderedDict[str, _FailureRecord] = OrderedDict()

    def retry_after(self, source: str) -> float | None:
        """Seconds `source` must still wait before a credential is checked, or `None`."""
        with self._lock:
            record = self._current(source)
            if record is None:
                return None
            remaining = record.blocked_until - self._clock()
            return remaining if remaining > 0 else None

    def failed(self, source: str) -> float | None:
        """Count a failure; the wait it imposes, if any."""
        with self._lock:
            now = self._clock()
            record = self._current(source) or _FailureRecord()
            record.failures += 1
            record.last_failure = now
            over = record.failures - self.free_failures
            wait: float | None = None
            if over > 0:
                wait = min(self.max_seconds, self.base_seconds * 2 ** min(over - 1, 62))
                record.blocked_until = now + wait
            self._sources.pop(source, None)
            self._sources[source] = record
            while len(self._sources) > self._max_sources:
                self._sources.popitem(last=False)
            return wait

    def succeeded(self, source: str) -> None:
        with self._lock:
            self._sources.pop(source, None)

    def failures(self, source: str) -> int:
        with self._lock:
            record = self._current(source)
            return record.failures if record else 0

    def _current(self, source: str) -> _FailureRecord | None:
        record = self._sources.get(source)
        if record is None:
            return None
        if self._clock() - record.last_failure >= self.forget_after_seconds:
            self._sources.pop(source, None)
            return None
        return record


# --- What the app holds -----------------------------------------------------------


def _default_rates(config: Settings) -> dict[str, RateLimit]:
    return {
        LimitName.WORKFLOW_START: RateLimit(config.api_rate_workflow_start_per_minute),
        LimitName.WORKFLOW_RESUME: RateLimit(config.api_rate_workflow_resume_per_minute),
        LimitName.APPROVAL_DECISION: RateLimit(config.api_rate_approval_decision_per_minute),
        LimitName.FAILURE_ACTION: RateLimit(config.api_rate_failure_action_per_minute),
        LimitName.EVIDENCE_INGEST: RateLimit(config.api_rate_evidence_ingest_per_minute),
    }


@dataclass
class ApiLimits:
    """The API's limits: the rates, the caps, and where their state lives."""

    rates: dict[str, RateLimit]
    store: RateLimitStore = field(default_factory=InMemoryRateLimitStore)
    max_active_workflows: int = 5
    max_sse_connections: int = 5
    auth_failures: AuthFailureThrottle = field(default_factory=AuthFailureThrottle)
    streams: ConcurrencyLimiter = field(default_factory=ConcurrencyLimiter)
    hits: LimitHitCounter = field(default_factory=lambda: default_hit_counter)
    enabled: bool = True

    def consume(self, actor_id: str, limit: str, *, route: str) -> None:
        """Spend one of `actor_id`'s `limit` budget, or raise `RateLimited`."""
        rate = self.rates.get(limit)
        if not self.enabled or rate is None:
            return
        decision = self.store.consume(f"{limit}:{actor_id}", rate)
        if decision.allowed:
            return
        retry_after = _whole_seconds(decision.retry_after)
        self.refuse(
            actor_id,
            limit,
            route=route,
            retry_after=retry_after,
            message=(
                f"too many requests: at most {rate.capacity} {limit.replace('_', ' ')} "
                f"requests per {_period(rate.period_seconds)}; retry in {int(retry_after)}s"
            ),
        )

    def check_active_workflows(self, actor_id: str, active: int, *, route: str) -> None:
        """Refuse a new workflow when `actor_id` already has `active` at the cap."""
        if not self.enabled or active < self.max_active_workflows:
            return
        self.refuse(
            actor_id,
            LimitName.ACTIVE_WORKFLOWS,
            route=route,
            retry_after=CONCURRENCY_RETRY_AFTER_SECONDS,
            message=(
                f"too many active workflows: {active} queued or running, at most "
                f"{self.max_active_workflows}; wait for one to finish or park before starting another"
            ),
            details={"active": active, "max": self.max_active_workflows},
        )

    def open_stream(self, actor_id: str, *, route: str) -> Slot:
        """A slot for one more event stream for `actor_id`, or raise `RateLimited`."""
        if not self.enabled:
            return Slot(lambda: None)
        slot = self.streams.acquire(actor_id, self.max_sse_connections)
        if slot is not None:
            return slot
        self.refuse(
            actor_id,
            LimitName.SSE_CONNECTIONS,
            route=route,
            retry_after=CONCURRENCY_RETRY_AFTER_SECONDS,
            message=(
                f"too many open event streams: at most {self.max_sse_connections}; "
                "close one before opening another"
            ),
        )

    def refuse(
        self,
        actor_id: str | None,
        limit: str,
        *,
        route: str,
        retry_after: float,
        message: str,
        details: dict | None = None,
        source: str | None = None,
    ) -> NoReturn:
        """Report the hit and raise the 429."""
        record_limit_hit(
            actor_id=actor_id,
            route=route,
            limit=limit,
            retry_after=retry_after,
            source=source,
            counter=self.hits,
        )
        raise RateLimited(message, limit=limit, retry_after=retry_after, details=details)


def _whole_seconds(seconds: float) -> float:
    """`Retry-After` is whole seconds; never round a wait down to zero."""
    return float(max(1, math.ceil(seconds)))


def _period(seconds: float) -> str:
    return "minute" if seconds == 60 else f"{seconds:g}s"


def build_api_limits(
    config: Settings = settings, *, clock: Callable[[], float] = time.monotonic
) -> ApiLimits:
    """The limits as configured, over in-process state."""
    return ApiLimits(
        rates=_default_rates(config),
        store=InMemoryRateLimitStore(clock=clock),
        max_active_workflows=config.api_max_active_workflows,
        max_sse_connections=config.api_max_sse_connections,
        auth_failures=AuthFailureThrottle(
            free_failures=config.api_auth_free_failures,
            base_seconds=config.api_auth_backoff_base_seconds,
            max_seconds=config.api_auth_backoff_max_seconds,
            clock=clock,
        ),
        enabled=config.api_limits_enabled,
    )


# --- FastAPI dependencies ---------------------------------------------------------


def get_api_limits(request: Request) -> ApiLimits:
    """The app's `ApiLimits`, built from settings the first time it is needed."""
    limits = getattr(request.app.state, "api_limits", None)
    if limits is None:
        limits = build_api_limits()
        request.app.state.api_limits = limits
    return limits


def route_label(request: Request) -> str:
    """`METHOD /path` with ids as `{id}`, as the middlewares label it; never the query string."""
    return f"{request.method} {normalize_path(request.url.path)}"


def client_address(request: Request) -> str:
    """The connection's peer address -- not `X-Forwarded-For`, which the client writes."""
    return request.client.host if request.client else "unknown"


def rate_limited(limit: str):
    """A dependency that spends one of the verified actor's `limit` budget per request."""

    def spend(request: Request) -> None:
        principal = getattr(request.state, "principal", None)
        if not isinstance(principal, Principal):  # pragma: no cover - mounted behind auth
            return
        get_api_limits(request).consume(principal.actor_id, limit, route=route_label(request))

    spend.__name__ = f"rate_limit_{limit}"
    return spend


def rate_limited_deps(*limits: str) -> list:
    """`dependencies=[...]` for a route that spends from each of `limits`."""
    return [Depends(rate_limited(limit)) for limit in limits]


__all__ = [
    "CONCURRENCY_RETRY_AFTER_SECONDS",
    "MAX_PAGE_SIZE",
    "ApiLimits",
    "AuthFailureThrottle",
    "ConcurrencyLimiter",
    "InMemoryRateLimitStore",
    "LimitName",
    "RateDecision",
    "RateLimit",
    "RateLimitStore",
    "Slot",
    "build_api_limits",
    "client_address",
    "get_api_limits",
    "rate_limited",
    "rate_limited_deps",
    "route_label",
]
