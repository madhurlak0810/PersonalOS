"""Recorded-fixture harness for the real job board adapters.

Every adapter is built over an `httpx.MockTransport` that answers from the
routes recorded in `evals/golden/provider_normalization.json`, with a fake
clock and sleep, so a contract test exercises the adapter's real request,
retry and pacing code with no network and no wall-clock waiting.
"""

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from personalos.domain.job_search import NormalizedPosting, RawPosting
from personalos.domain.retry_policy import READ
from personalos.graphs.job_search import DictPostingNormalizer
from personalos.providers import (
    AshbyProvider,
    GreenhouseProvider,
    JobProvider,
    LeverProvider,
    ProviderHttp,
    WorkdayProvider,
)

GOLDEN_PATH = Path(__file__).parents[2] / "evals" / "golden" / "provider_normalization.json"

#: Adapter class per provider name, in the order the suites run them.
ADAPTERS: dict[str, type] = {
    "greenhouse": GreenhouseProvider,
    "lever": LeverProvider,
    "ashby": AshbyProvider,
    "workday": WorkdayProvider,
}
PROVIDERS = tuple(ADAPTERS)

Handler = Callable[[httpx.Request], httpx.Response]


def golden() -> dict[str, Any]:
    """The golden document, freshly parsed (callers may mutate it)."""
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def case(provider: str) -> dict[str, Any]:
    """One provider's entry in the golden document."""
    return golden()["providers"][provider]


def route_key(request: httpx.Request) -> str:
    """`METHOD scheme://host/path`, the way routes are keyed in the golden file."""
    url = request.url
    return f"{request.method} {url.scheme}://{url.host}{url.path}"


class FakeClock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class Recorded:
    """A built adapter and everything it did."""

    provider: JobProvider
    requests: list[httpx.Request] = field(default_factory=list)
    started_at: list[float] = field(default_factory=list)
    clock: FakeClock = field(default_factory=FakeClock)

    @property
    def sleeps(self) -> list[float]:
        return self.clock.sleeps


def routes_handler(routes: dict[str, Any]) -> Handler:
    """Answer each request from the recorded routes, or 404."""

    def handle(request: httpx.Request) -> httpx.Response:
        key = route_key(request)
        if key in routes:
            return httpx.Response(200, json=routes[key])
        return httpx.Response(404, json={"error": "not found"})

    return handle


def build(
    provider: str,
    *,
    handler: Handler | None = None,
    routes: dict[str, Any] | None = None,
    min_interval: float = 0.0,
    max_retry_after: float = 60.0,
    rand: Callable[[], float] = lambda: 0.5,
) -> Recorded:
    """Build `provider` over recorded routes (or a custom handler)."""
    entry = case(provider)
    answer = handler or routes_handler(routes if routes is not None else entry["routes"])
    recorded = Recorded(provider=None)  # type: ignore[arg-type]

    def record(request: httpx.Request) -> httpx.Response:
        recorded.requests.append(request)
        recorded.started_at.append(recorded.clock.now)
        return answer(request)

    http = ProviderHttp(
        provider,
        client=httpx.AsyncClient(transport=httpx.MockTransport(record)),
        min_interval=min_interval,
        retry_policy=READ,
        max_retry_after=max_retry_after,
        clock=recorded.clock,
        sleep=recorded.clock.sleep,
        rand=rand,
    )
    recorded.provider = ADAPTERS[provider](copy.deepcopy(entry["config"]), http=http)
    return recorded


def normalize(raws: list[RawPosting]) -> list[NormalizedPosting]:
    """Every raw posting the default normalizer accepts, in order."""
    normalizer = DictPostingNormalizer()
    return [posting for raw in raws if (posting := normalizer.normalize(raw)) is not None]


def canonical(posting: NormalizedPosting) -> dict[str, Any]:
    """A normalized posting as golden JSON: every field but `raw`, plus its identity."""
    document = posting.model_dump(mode="json", exclude={"raw"})
    document["dedupe_key"] = posting.dedupe_key
    document["description_hash"] = posting.description_hash
    return document


__all__ = [
    "ADAPTERS",
    "GOLDEN_PATH",
    "PROVIDERS",
    "FakeClock",
    "Recorded",
    "build",
    "canonical",
    "case",
    "golden",
    "normalize",
    "route_key",
    "routes_handler",
]
