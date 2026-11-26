"""Limit hits as an OpenTelemetry counter, a structured log line and a span event.

Every refusal the API's limits make (`apps.api.limits`, `apps.api.security`)
goes through `record_limit_hit`, which:

- adds one to `personalos.api.limit_hits` (counter), with `route` and
  `limit` attributes -- not the actor, so the metric's cardinality is bounded
  by the routes, not by who calls;
- writes `api.limit_hit` at WARNING with `actor_id`, `route`, `limit_name` and
  `retry_after_seconds`;
- records an `api.limit_hit` span (under whatever span is current), with
  `actor_id`, `route`, `limit` and `retry_after_seconds` as attributes;
- counts the hit in this process's `LimitHitCounter`, which
  `GET /v1/failures/limits` reads, so repeated hits are visible without a
  metrics backend.

**Nothing about the request goes in.** No token, no body, no query string:
`route` is a method and a path *template* (`POST /v1/workflows/{workflow_id}/resume`),
or, for a refusal made before routing, the path with every id-like segment
replaced by `{id}` (`normalize_path`). The source address of a throttled
authentication failure is logged (it is what is being throttled), never put
on the metric or the span.
"""

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from personalos.observability import metrics, tracing
from personalos.observability.metrics import Metric

logger = logging.getLogger("personalos.api.limits")

#: `event` of the log line a limit hit writes.
LIMIT_HIT_EVENT = "api.limit_hit"
#: Name of the span a limit hit records.
LIMIT_HIT_SPAN = "api.limit_hit"

#: The actor of a hit refused before anyone was authenticated.
ANONYMOUS_ACTOR = "anonymous"

#: Path segments that are ids, not names: UUIDs, numbers, long opaque tokens.
_ID_SEGMENT = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|\d+|[A-Za-z0-9_\-:.]{24,})$"
)


def normalize_path(path: str) -> str:
    """`path` with every id-like segment replaced by `{id}`, for a route label."""
    return "/".join("{id}" if _ID_SEGMENT.match(part) else part for part in path.split("/"))


@dataclass(frozen=True)
class LimitHitCount:
    """How often one actor hit one limit on one route, in this process."""

    actor_id: str
    route: str
    limit: str
    count: int
    last_hit_at: datetime


class LimitHitCounter:
    """Per-process tally of limit hits, keyed by (actor, route, limit).

    Bounded: past `max_entries` keys the least recently hit is dropped, so
    a flood of distinct routes cannot grow it without end.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, str], LimitHitCount] = {}

    def add(self, actor_id: str, route: str, limit: str) -> None:
        key = (actor_id, route, limit)
        with self._lock:
            previous = self._counts.pop(key, None)
            self._counts[key] = LimitHitCount(
                actor_id=actor_id,
                route=route,
                limit=limit,
                count=(previous.count if previous else 0) + 1,
                last_hit_at=self._clock(),
            )
            while len(self._counts) > self._max_entries:
                self._counts.pop(next(iter(self._counts)))

    def snapshot(self, *, actor_id: str | None = None) -> list[LimitHitCount]:
        """Every tally, most hits first; only `actor_id`'s when given."""
        with self._lock:
            counts = list(self._counts.values())
        if actor_id is not None:
            counts = [c for c in counts if c.actor_id == actor_id]
        return sorted(counts, key=lambda c: (-c.count, c.actor_id, c.route, c.limit))

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()


#: The process's tally, unless an `ApiLimits` is given its own.
default_hit_counter = LimitHitCounter()


def record_limit_hit(
    *,
    actor_id: str | None,
    route: str,
    limit: str,
    retry_after: float | None = None,
    source: str | None = None,
    counter: LimitHitCounter | None = None,
) -> None:
    """Report one refused request. See the module docstring for where it goes."""
    actor = actor_id or ANONYMOUS_ACTOR
    metrics.add(Metric.API_LIMIT_HITS, route=route, limit=limit)
    (counter or default_hit_counter).add(actor, route, limit)
    attributes: dict[str, str | float] = {"actor_id": actor, "route": route, "limit": limit}
    if retry_after is not None:
        attributes["retry_after_seconds"] = float(retry_after)
    with tracing.span(LIMIT_HIT_SPAN, attributes):
        pass
    extra: dict[str, object] = {
        "event": LIMIT_HIT_EVENT,
        "actor_id": actor,
        "route": route,
        "limit_name": limit,
    }
    if retry_after is not None:
        extra["retry_after_seconds"] = round(float(retry_after), 3)
    if source is not None:
        extra["source_address"] = source
    logger.warning("%s refused by the %s limit (actor_id=%s)", route, limit, actor, extra=extra)


__all__ = [
    "ANONYMOUS_ACTOR",
    "LIMIT_HIT_EVENT",
    "LIMIT_HIT_SPAN",
    "LimitHitCount",
    "LimitHitCounter",
    "default_hit_counter",
    "normalize_path",
    "record_limit_hit",
]
