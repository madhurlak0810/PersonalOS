"""Unit tests for the job provider adapters and their tool-boundary plumbing.

The Greenhouse adapter is exercised against `httpx.MockTransport`, so the
requests it makes are asserted without a network.
"""

from datetime import datetime, timezone

import httpx
import pytest

from personalos.bootstrap import build_job_providers, build_tool_gateway
from personalos.domain.job_search import RawPosting
from personalos.executor.job_discovery import GatewayJobProvider
from personalos.graphs.job_search import DictPostingNormalizer
from personalos.policy import (
    DEFAULT_TOOL_PERMISSIONS,
    ApprovalGrant,
    IntentOrigin,
    PermissionClass,
    PolicyDenied,
    PolicyViolation,
    ToolIntent,
    default_policy_engine,
)
from personalos.providers import (
    AshbyProvider,
    FakeJobProvider,
    GreenhouseProvider,
    JobProvider,
    LeverProvider,
    ProviderHttp,
    ProviderUnavailable,
    RateLimiter,
    WorkdayProvider,
)
from personalos.providers.greenhouse import html_to_text
from personalos.providers.http import parse_retry_after
from personalos.providers.lever import EU_BASE_URL
from personalos.providers.workday import WorkdaySite
from personalos.tools.gateway import ToolExecutionError
from tests.fixtures import job_search_fakes as fakes

GREENHOUSE_JOB = {
    "id": 4012345,
    "title": "Senior Backend Engineer",
    "absolute_url": "https://boards.greenhouse.io/acme/jobs/4012345",
    "location": {"name": "Remote - US"},
    "first_published": "2026-09-20T10:00:00-04:00",
    "updated_at": "2026-09-25T10:00:00-04:00",
    "company_name": "Acme",
    "departments": [{"id": 1, "name": "Engineering"}],
    "content": (
        "&lt;p&gt;We need &lt;strong&gt;Python&lt;/strong&gt; &amp;amp; Postgres.&lt;/p&gt;"
        "&lt;ul&gt;&lt;li&gt;Ship APIs&lt;/li&gt;&lt;/ul&gt;"
        "&lt;script&gt;alert(1)&lt;/script&gt;"
    ),
}
OTHER_JOB = {**GREENHOUSE_JOB, "id": 99, "title": "Office Manager", "content": "Run the office."}


def greenhouse(handler) -> tuple[GreenhouseProvider, list[httpx.Request]]:
    """A Greenhouse adapter over a mock transport, plus the requests it sent."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    http = ProviderHttp("greenhouse", client=client, min_interval=0, sleep=_no_wait)
    return GreenhouseProvider({"acme": "Acme"}, http=http), requests


async def _no_wait(seconds: float) -> None:
    """Retries in these tests back off without spending wall-clock time."""


def board(request: httpx.Request) -> httpx.Response:
    """A one-company board with two jobs."""
    if request.url.path == "/v1/boards/acme/jobs":
        return httpx.Response(200, json={"jobs": [GREENHOUSE_JOB, OTHER_JOB]})
    if request.url.path == "/v1/boards/acme/jobs/4012345":
        return httpx.Response(200, json=GREENHOUSE_JOB)
    return httpx.Response(404, json={"status": 404})


# --- Interface ----------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        FakeJobProvider(),
        GreenhouseProvider({"acme": "Acme"}),
        LeverProvider({"acme": "Acme"}),
        AshbyProvider({"acme": "Acme"}),
        WorkdayProvider({"acme.wd5.myworkdayjobs.com/AcmeCareers": "Acme"}),
        GatewayJobProvider("fakeboard", build_tool_gateway()),
    ],
    ids=["fake", "greenhouse", "lever", "ashby", "workday", "gateway"],
)
def test_adapters_satisfy_the_job_provider_interface(provider):
    """Every adapter is swappable for any other behind `JobProvider`."""
    assert isinstance(provider, JobProvider)
    assert provider.name


async def test_fake_provider_serves_and_fetches_its_postings():
    provider = FakeJobProvider("fakeboard", [{"id": "fb-1", "title": "T", "company": "C"}])

    (found,) = await provider.search(fakes.profile())

    assert found == RawPosting(provider="fakeboard", payload=provider.postings[0])
    assert await provider.get_job("fb-1") == found
    assert await provider.get_job("missing") is None


# --- Greenhouse ---------------------------------------------------------------


def test_html_to_text_unescapes_strips_tags_and_drops_scripts():
    text = html_to_text(GREENHOUSE_JOB["content"])

    assert text == "We need Python & Postgres.\nShip APIs"


async def test_greenhouse_search_returns_matching_postings_in_the_conventional_shape():
    provider, requests = greenhouse(board)

    postings = await provider.search(fakes.profile())

    # "Office Manager" mentions none of the profile's roles or keywords.
    assert [raw.payload["title"] for raw in postings] == ["Senior Backend Engineer"]
    assert [(r.method, r.url.path, r.url.params["content"]) for r in requests] == [
        ("GET", "/v1/boards/acme/jobs", "true")
    ]

    normalized = DictPostingNormalizer().normalize(postings[0])
    assert normalized.source == "greenhouse"
    assert normalized.source_job_id == "acme:4012345"
    assert normalized.company == "Acme"
    assert normalized.location == "Remote - US"
    assert normalized.remote is True
    assert normalized.url == "https://boards.greenhouse.io/acme/jobs/4012345"
    assert normalized.description == "We need Python & Postgres.\nShip APIs"
    assert normalized.posted_at is not None


async def test_greenhouse_get_job_round_trips_the_id_search_assigned():
    provider, requests = greenhouse(board)

    found = await provider.get_job("acme:4012345")

    assert found.payload["id"] == "acme:4012345"
    assert requests[-1].url.path == "/v1/boards/acme/jobs/4012345"


@pytest.mark.parametrize("job_id", ["acme:404", "other-board:1", "acme:../../secrets", "junk"])
async def test_greenhouse_get_job_returns_none_for_unknown_or_malformed_ids(job_id):
    provider, requests = greenhouse(board)

    assert await provider.get_job(job_id) is None
    # Only a well-formed id for a configured board is ever sent upstream.
    assert all(r.url.path.startswith("/v1/boards/acme/jobs/") for r in requests)


async def test_greenhouse_only_ever_issues_get_requests():
    provider, requests = greenhouse(board)

    await provider.search(fakes.profile())
    await provider.get_job("acme:4012345")

    assert {request.method for request in requests} == {"GET"}


async def test_greenhouse_reports_an_unreachable_board_as_provider_unavailable():
    provider, _ = greenhouse(lambda request: httpx.Response(503))

    with pytest.raises(ProviderUnavailable):
        await provider.search(fakes.profile())


# --- Configuration ------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "config"),
    [
        (GreenhouseProvider, {"acme/../x": "Acme"}),
        (LeverProvider, {"acme?x=1": "Acme"}),
        (LeverProvider, {"us/acme": "Acme"}),
        (AshbyProvider, {"ac me": "Acme"}),
        (WorkdayProvider, {"evil.example/Careers": "Acme"}),
        (WorkdayProvider, {"acme.wd5.myworkdayjobs.com.evil.example/Careers": "Acme"}),
        (WorkdayProvider, {"acme.wd5.myworkdayjobs.com/../admin": "Acme"}),
        (WorkdayProvider, {"acme.wd5.myworkdayjobs.com": "Acme"}),
    ],
)
def test_a_board_key_that_could_steer_a_request_elsewhere_is_rejected(adapter, config):
    with pytest.raises(ValueError):
        adapter(config)


@pytest.mark.parametrize(
    "adapter", [GreenhouseProvider, LeverProvider, AshbyProvider, WorkdayProvider]
)
def test_an_adapter_with_nothing_to_watch_is_a_configuration_error(adapter):
    with pytest.raises(ValueError):
        adapter({})


def test_workday_site_keys_resolve_to_the_career_site_endpoint():
    site = WorkdaySite("Acme.WD5.myworkdayjobs.com/AcmeCareers")

    assert site.key == "acme.wd5.myworkdayjobs.com/AcmeCareers"
    assert site.api_base == "https://acme.wd5.myworkdayjobs.com/wday/cxs/acme/AcmeCareers"


async def test_lever_reads_an_eu_site_from_the_eu_region():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = LeverProvider({"eu/globex": "Globex"}, http=ProviderHttp("lever", client=client))

    assert await provider.search(fakes.profile()) == []
    assert str(requests[0].url).startswith(f"{EU_BASE_URL}/globex?")


def test_bootstrap_configures_every_board_that_is_set(monkeypatch):
    from personalos import bootstrap

    monkeypatch.setattr(bootstrap.settings, "greenhouse_boards", {"acme": "Acme"})
    monkeypatch.setattr(bootstrap.settings, "lever_sites", {"acme": "Acme"})
    monkeypatch.setattr(bootstrap.settings, "ashby_boards", {})
    monkeypatch.setattr(
        bootstrap.settings, "workday_sites", {"acme.wd5.myworkdayjobs.com/AcmeCareers": "Acme"}
    )

    assert [p.name for p in bootstrap.configured_job_providers()] == [
        "greenhouse",
        "lever",
        "workday",
    ]


# --- HTTP layer ----------------------------------------------------------------

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("header", "seconds"),
    [
        ("120", 120.0),
        (" 1.5 ", 1.5),
        ("Wed, 11 Nov 2026 12:00:30 GMT", 30.0),
        ("Wed, 11 Nov 2026 11:00:00 GMT", 0.0),
        ("-5", None),
        ("soon", None),
        ("", None),
        (None, None),
        ("nan", None),
    ],
)
def test_retry_after_parses_seconds_and_http_dates_and_ignores_junk(header, seconds):
    assert parse_retry_after(header, NOW) == seconds


async def test_the_rate_limiter_spaces_requests_and_honours_a_deferral():
    now = [0.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(1.5, clock=lambda: now[0], sleep=sleep)

    await limiter.acquire()
    await limiter.acquire()
    limiter.defer(10)
    await limiter.acquire()

    assert slept == [1.5, 10.0]


def test_the_rate_limiter_rejects_a_negative_interval():
    with pytest.raises(ValueError):
        RateLimiter(-1)


# --- Policy -------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["job_providers.search", "job_providers.get_job"])
def test_provider_tools_are_classed_read_external(tool):
    assert DEFAULT_TOOL_PERMISSIONS[tool].permission_class is PermissionClass.READ_EXTERNAL
    assert DEFAULT_TOOL_PERMISSIONS[tool].scopes == {"jobs:read"}


async def test_gateway_provider_reads_through_policy():
    """Both calls reach the adapter, and each is a recorded non-mutating intent."""
    seen: list[ToolIntent] = []
    adapter = FakeJobProvider("fakeboard", [fakes.raw_posting().payload])
    (provider,) = build_job_providers(
        [adapter], policy=default_policy_engine(decision_sink=lambda i, d: seen.append(i))
    )

    (found,) = await provider.search(fakes.profile())
    fetched = await provider.get_job("fb-1")

    assert found.provider == "fakeboard" and fetched == found
    assert await provider.get_job("missing") is None
    assert [i.tool_ref for i in seen] == [
        "job_providers.search",
        "job_providers.get_job",
        "job_providers.get_job",
    ]
    assert all(i.origin is IntentOrigin.SYSTEM and not i.mutating for i in seen)


async def test_gateway_provider_cannot_reach_an_unregistered_adapter():
    gateway = build_tool_gateway(job_providers=[FakeJobProvider("fakeboard")])

    with pytest.raises(Exception, match="unknown job provider 'elsewhere'"):
        await GatewayJobProvider("elsewhere", gateway).search(fakes.profile())


async def test_a_provider_failure_surfaces_as_a_tool_failure():
    class Down(FakeJobProvider):
        async def search(self, profile):
            raise ProviderUnavailable("upstream 503")

    gateway = build_tool_gateway(job_providers=[Down("fakeboard")])
    intent = ToolIntent(
        server="job_providers",
        tool="search",
        arguments={"provider": "fakeboard", "profile": fakes.profile().model_dump(mode="json")},
        requested_by="test",
    )

    result = await gateway.dispatch(intent)

    assert result.success is False
    with pytest.raises(ToolExecutionError, match="upstream 503"):
        result.unwrap()


async def test_a_provider_write_is_refused_even_with_an_approval():
    """Read-only is enforced at the adapter, not only by the allowlist."""
    gateway = build_tool_gateway(job_providers=[FakeJobProvider("fakeboard")])
    unkeyed = ToolIntent(
        server="job_providers",
        tool="search",
        arguments={"provider": "fakeboard", "profile": {}},
        mutating=True,
        requested_by="test",
    )
    with pytest.raises(PolicyDenied):
        await gateway.dispatch(unkeyed)

    keyed = unkeyed.model_copy(
        update={"arguments": {**unkeyed.arguments, "idempotency_key": "write-attempt-0001"}}
    )
    grant = ApprovalGrant(
        intent_id=keyed.intent_id, intent_fingerprint=keyed.fingerprint(), approved_by="reviewer"
    )
    with pytest.raises(PolicyViolation, match="read-only"):
        await gateway.dispatch(keyed, grant)


@pytest.mark.parametrize("tool", ["submit", "apply", "delete_job"])
async def test_no_other_provider_tool_is_reachable(tool):
    gateway = build_tool_gateway(job_providers=[FakeJobProvider("fakeboard")])
    intent = ToolIntent(
        server="job_providers", tool=tool, arguments={"provider": "fakeboard"}, requested_by="test"
    )

    with pytest.raises(PolicyDenied):
        await gateway.dispatch(intent)
