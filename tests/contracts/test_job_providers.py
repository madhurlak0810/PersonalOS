"""The contract every real `JobProvider` adapter passes: Greenhouse, Lever, Ashby, Workday.

Each test runs once per provider in `PROVIDERS`, against the responses
recorded in `evals/golden/provider_normalization.json` and served by
`httpx.MockTransport` -- no network, and a fake clock so retries and pacing
are asserted without waiting. A new board joins by adding its adapter to
`tests/fixtures/job_provider_contract.ADAPTERS` and its recorded routes to the
golden file; it is done when this file passes for it unchanged.

What the contract pins down:

- **Shape.** `search` returns `RawPosting`s tagged with the adapter's name
  whose payloads `DictPostingNormalizer` accepts; `get_job` round-trips the
  id `search` assigned to the same posting, answers `None` for one that is
  gone, and never sends a malformed id upstream.
- **Read-only.** Only GETs, plus a POST to a search endpoint for a board
  whose search takes a body. Nothing else is ever sent.
- **Polite.** Requests are spaced by the rate limit; a 429 or 5xx is retried
  with jittered exponential backoff, never sooner than `Retry-After`; a
  `Retry-After` past the cap, or a 4xx that will not improve, is not retried.
- **Untrusting.** A garbage body is `ProviderUnavailable` (or no postings),
  never a crash.
"""

import httpx
import pytest

from personalos.domain.job_search import RawPosting
from personalos.domain.retry_policy import READ
from personalos.providers import JobProvider, ProviderUnavailable
from personalos.providers.http import USER_AGENT
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.job_provider_contract import (
    PROVIDERS,
    build,
    case,
    normalize,
    route_key,
    routes_handler,
)

pytestmark = pytest.mark.parametrize("provider", PROVIDERS)

EVERYTHING = fakes.profile(target_roles=(), keywords=())

#: Endpoints a board may be sent a POST to: searches that take a JSON body.
QUERY_ENDPOINTS = {"workday": "/jobs"}


def search_endpoint_fails(provider: str, response: httpx.Response):
    """A handler answering the provider's search endpoint with `response`, the rest as recorded."""
    routes = case(provider)["routes"]
    search_key = next(iter(routes))
    fallback = routes_handler(routes)

    def handle(request: httpx.Request) -> httpx.Response:
        return response if route_key(request) == search_key else fallback(request)

    return handle


def flaky(provider: str, failures: list[httpx.Response]):
    """A handler that answers with `failures` first, then the recorded routes."""
    fallback = routes_handler(case(provider)["routes"])
    queue = list(failures)

    def handle(request: httpx.Request) -> httpx.Response:
        return queue.pop(0) if queue else fallback(request)

    return handle


# --- Shape -------------------------------------------------------------------


def test_is_a_job_provider_with_a_stable_name(provider):
    adapter = build(provider).provider

    assert isinstance(adapter, JobProvider)
    assert adapter.name == provider
    assert type(adapter).name == provider


async def test_search_returns_normalizable_postings_tagged_with_the_adapter(provider):
    raws = await build(provider).provider.search(EVERYTHING)

    assert raws and all(isinstance(raw, RawPosting) for raw in raws)
    assert {raw.provider for raw in raws} == {provider}
    postings = normalize(list(raws))
    assert postings
    for posting in postings:
        assert posting.source == provider
        assert posting.source_job_id
        assert posting.url and posting.url.startswith("https://")
        assert posting.description
        assert len(posting.description_hash) == 64
        assert len(posting.dedupe_key) <= 150
    assert len({p.source_job_id for p in postings}) == len(postings)


async def test_search_filters_on_the_profile(provider):
    adapter = build(provider).provider

    found = await adapter.search(fakes.profile(target_roles=("Backend Engineer",), keywords=()))
    nothing = await adapter.search(fakes.profile(target_roles=("Lighthouse Keeper",), keywords=()))

    titles = {raw.payload["title"] for raw in found}
    assert "Senior Backend Engineer" in titles
    assert all("backend engineer" in str(title).lower() for title in titles)
    assert nothing == []


async def test_get_job_round_trips_the_id_search_assigned(provider):
    adapter = build(provider).provider
    known = case(provider)["known_id"]

    (searched,) = [
        p for p in normalize(list(await adapter.search(EVERYTHING))) if p.source_job_id == known
    ]
    fetched = await adapter.get_job(known)

    assert fetched is not None and fetched.provider == provider
    (refetched,) = normalize([fetched])
    assert refetched.source_job_id == known
    assert refetched.dedupe_key == searched.dedupe_key
    assert refetched.model_dump(exclude={"raw"}) == searched.model_dump(exclude={"raw"})


async def test_get_job_returns_none_for_a_posting_that_is_gone(provider):
    assert await build(provider).provider.get_job(case(provider)["missing_id"]) is None


async def test_malformed_ids_are_never_sent_upstream(provider):
    recorded = build(provider)

    for job_id in case(provider)["malformed_ids"]:
        assert await recorded.provider.get_job(job_id) is None

    assert recorded.requests == []


# --- Read-only ---------------------------------------------------------------


async def test_only_reads_are_ever_sent(provider):
    recorded = build(provider)

    await recorded.provider.search(EVERYTHING)
    await recorded.provider.search(fakes.profile())
    await recorded.provider.get_job(case(provider)["known_id"])
    await recorded.provider.get_job(case(provider)["missing_id"])

    for request in recorded.requests:
        if request.method == "POST":
            assert request.url.path.endswith(QUERY_ENDPOINTS[provider]), request.url
        else:
            assert request.method == "GET", request.url
        assert request.url.scheme == "https"
        assert request.headers["User-Agent"] == USER_AGENT
        # Nothing identifying the user, and no credential, goes upstream.
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers


# --- Polite ------------------------------------------------------------------


async def test_requests_are_spaced_by_the_rate_limit(provider):
    recorded = build(provider, min_interval=2.0)

    await recorded.provider.search(EVERYTHING)
    await recorded.provider.get_job(case(provider)["known_id"])

    assert len(recorded.started_at) >= 2
    gaps = [b - a for a, b in zip(recorded.started_at, recorded.started_at[1:], strict=False)]
    assert all(gap >= 2.0 for gap in gaps), gaps


async def test_a_429_is_retried_no_sooner_than_retry_after(provider):
    recorded = build(
        provider,
        handler=flaky(provider, [httpx.Response(429, headers={"Retry-After": "7"})]),
    )

    raws = await recorded.provider.search(EVERYTHING)

    assert raws
    assert recorded.sleeps[0] >= 7
    assert recorded.started_at[1] - recorded.started_at[0] >= 7


async def test_an_http_date_retry_after_is_honoured(provider):
    recorded = build(
        provider,
        handler=flaky(
            provider,
            [httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})],
        ),
        max_retry_after=float("inf"),
    )

    await recorded.provider.search(EVERYTHING)

    assert recorded.sleeps[0] > 60 * 60 * 24 * 365


async def test_5xx_backs_off_exponentially_with_jitter_then_gives_up(provider):
    """Every attempt fails: the READ budget is spent, each wait inside its jitter window."""
    jitter = iter([0.0, 1.0, 0.5])
    recorded = build(
        provider,
        handler=search_endpoint_fails(provider, httpx.Response(503)),
        rand=lambda: next(jitter),
    )

    with pytest.raises(ProviderUnavailable):
        await recorded.provider.search(EVERYTHING)

    assert len(recorded.requests) == READ.max_attempts
    windows = [READ.base_delay * 2**n for n in range(READ.max_retries)]
    assert recorded.sleeps == [windows[0] / 2, windows[1], windows[2] * 0.75]


async def test_a_retry_after_past_the_cap_fails_fast_with_it_attached(provider):
    recorded = build(
        provider,
        handler=search_endpoint_fails(
            provider, httpx.Response(429, headers={"Retry-After": "3600"})
        ),
        max_retry_after=60,
    )

    with pytest.raises(ProviderUnavailable) as raised:
        await recorded.provider.search(EVERYTHING)

    assert len(recorded.requests) == 1
    assert recorded.sleeps == []
    # The adapter-level error says "no board answered"; the rate limit is
    # remembered for every later request instead.
    assert recorded.provider.http.limiter._next_allowed >= recorded.clock.now + 3600
    assert raised.value.retryable


@pytest.mark.parametrize("status", [400, 401, 403, 410])
async def test_a_client_error_is_not_retried(provider, status):
    recorded = build(provider, handler=search_endpoint_fails(provider, httpx.Response(status)))

    with pytest.raises(ProviderUnavailable):
        await recorded.provider.search(EVERYTHING)

    assert len(recorded.requests) == 1
    assert recorded.sleeps == []


async def test_a_transport_error_is_retried(provider):
    calls = {"n": 0}
    fallback = routes_handler(case(provider)["routes"])

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return fallback(request)

    recorded = build(provider, handler=handle)

    assert await recorded.provider.search(EVERYTHING)
    assert len(recorded.sleeps) == 1


# --- Untrusting --------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [b"<html>maintenance</html>", b'"just a string"', b"[1, 2, 3]", b'{"jobs": "nope"}', b"null"],
    ids=["html", "string", "list", "wrong-type", "null"],
)
async def test_a_garbage_body_is_unavailable_or_empty_never_a_crash(provider, body):
    recorded = build(
        provider,
        handler=search_endpoint_fails(
            provider,
            httpx.Response(200, content=body, headers={"Content-Type": "application/json"}),
        ),
    )

    try:
        raws = await recorded.provider.search(EVERYTHING)
    except ProviderUnavailable:
        return
    assert normalize(list(raws)) == []
