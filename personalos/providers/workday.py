"""Workday career-site adapter. Opt-in; read the risk note before enabling it.

**What it reads.** Every Workday-hosted careers page
(`<tenant>.wd<N>.myworkdayjobs.com/<site>`) is rendered from a JSON endpoint
under `/wday/cxs/<tenant>/<site>/`: a POST to `jobs` searches (criteria in the
body, at most 20 results a page) and a GET to `job/<path>` returns one
posting. Both are unauthenticated and serve exactly what the public page
shows. This adapter uses those two calls and nothing else.

**Why it is opt-in.** Unlike Greenhouse, Lever and Ashby, Workday does not
*publish* this as a developer API: it is the career site's own backend,
undocumented, unversioned, and free to change without notice. Workday's
documented APIs (REST/SOAP) are for a tenant's own integrations and need that
tenant's credentials, so they are not an option for a job seeker. Reading the
endpoint the public page reads -- at a human pace, read-only, only for
employers the user names -- is the narrowest compliant-looking path there is,
but it is the user's call per employer, which is why nothing is watched until
`WORKDAY_SITES` names a site. Check that site's `robots.txt` and terms before
adding it. See "Job board coverage" in `docs/ARCHITECTURE_BOUNDARIES.md`.

**Why it is slow on purpose.** Search results carry no description, so every
candidate costs one more request. The adapter searches only on the profile's
target roles (or its keywords when it has none), stops after `max_pages`
pages per query, fetches at most `max_details` postings per site, and paces
every request through the shared `ProviderHttp` limiter (one second apart by
default).
"""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from personalos.domain.job_search import RawPosting, SearchProfile
from personalos.providers.base import ProviderUnavailable
from personalos.providers.http import DEFAULT_TIMEOUT_SECONDS, ProviderHttp
from personalos.providers.text import (
    html_to_text,
    is_remote,
    items,
    mapping,
    matches,
    number,
    text,
    unique,
)

logger = logging.getLogger(__name__)

#: Workday caps a search page at 20 results.
PAGE_SIZE = 20
DEFAULT_MAX_PAGES = 5
DEFAULT_MAX_DETAILS = 50
DEFAULT_MIN_INTERVAL_SECONDS = 1.0

_HOST = re.compile(r"^[a-z0-9-]{1,63}\.wd\d{1,3}\.myworkdayjobs\.com$")
_SITE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
#: A posting's `externalPath`: `/job/<location>/<slug>_<req id>`.
_EXTERNAL_PATH = re.compile(r"^/job(?:/[A-Za-z0-9._~%()+,'-]{1,200}){1,4}$")


class WorkdaySite:
    """One career site: `<host>/<site>`, e.g. `acme.wd5.myworkdayjobs.com/AcmeCareers`."""

    def __init__(self, key: str):
        """Parse and validate a `WORKDAY_SITES` key."""
        host, _, site = key.partition("/")
        host = host.lower()
        if not _HOST.match(host) or not _SITE.match(site):
            raise ValueError(
                f"invalid workday site '{key}': expected '<tenant>.wd<N>.myworkdayjobs.com/<site>'"
            )
        self.key = f"{host}/{site}"
        self.host = host
        self.site = site
        self.tenant = host.split(".", 1)[0]

    @property
    def api_base(self) -> str:
        """Root of the site's career-site JSON endpoint."""
        return f"https://{self.host}/wday/cxs/{self.tenant}/{self.site}"


class WorkdayProvider:
    """`JobProvider` over one or more Workday career sites.

    `sites` maps a `<host>/<site>` key to the employer's display name. The
    posting's `hiringOrganization` is often a legal entity ("XY00 Acme
    Holdings B.V.") rather than the brand, so the configured name is used.
    """

    name = "workday"

    def __init__(
        self,
        sites: Mapping[str, str],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http: ProviderHttp | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_details: int = DEFAULT_MAX_DETAILS,
    ):
        """Take the sites to watch, the request budget, and optionally an HTTP layer."""
        if not sites:
            raise ValueError("WorkdayProvider requires at least one site")
        if max_pages < 1 or max_details < 1:
            raise ValueError("max_pages and max_details must be at least 1")
        parsed = [(WorkdaySite(key), company) for key, company in sites.items()]
        self.sites: dict[str, tuple[WorkdaySite, str]] = {
            site.key: (site, company) for site, company in parsed
        }
        self.max_pages = max_pages
        self.max_details = max_details
        self.http = http or ProviderHttp(
            self.name, client=client, timeout=timeout, min_interval=DEFAULT_MIN_INTERVAL_SECONDS
        )

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return postings on the configured sites that match the profile.

        A site whose search fails is logged and skipped; only every site
        failing is reported as the provider being down. A single posting that
        fails to load is skipped without failing its site.
        """
        postings: list[RawPosting] = []
        failed: list[str] = []
        for site, company in self.sites.values():
            try:
                paths = await self._search_paths(site, _queries(profile))
            except ProviderUnavailable:
                logger.warning("workday site '%s' failed during search", site.key, exc_info=True)
                failed.append(site.key)
                continue
            for path in paths[: self.max_details]:
                try:
                    payload = await self._payload(site, company, path)
                except ProviderUnavailable:
                    logger.warning("workday posting '%s%s' failed to load", site.key, path)
                    continue
                if payload is not None and matches(payload, profile):
                    postings.append(RawPosting(provider=self.name, payload=payload))
        if failed and len(failed) == len(self.sites):
            raise ProviderUnavailable(f"no workday site answered: {', '.join(failed)}")
        return postings

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Fetch one posting by the `<host>/<site>:<externalPath>` id `search` assigned."""
        key, _, path = source_job_id.partition(":")
        entry = self.sites.get(key)
        if entry is None or not _valid_path(path):
            return None
        site, company = entry
        payload = await self._payload(site, company, path)
        return RawPosting(provider=self.name, payload=payload) if payload else None

    async def _search_paths(self, site: WorkdaySite, queries: Sequence[str]) -> list[str]:
        """Every posting path the queries find, first occurrence kept, in order."""
        paths: list[str] = []
        for query in queries:
            for page in range(self.max_pages):
                offset = page * PAGE_SIZE
                body = mapping(
                    await self.http.query_json(
                        f"{site.api_base}/jobs",
                        {
                            "appliedFacets": {},
                            "limit": PAGE_SIZE,
                            "offset": offset,
                            "searchText": query,
                        },
                    )
                )
                found = items(body.get("jobPostings"))
                for job in found:
                    path = text(mapping(job).get("externalPath"))
                    if path and _valid_path(path):
                        paths.append(path)
                total = number(body.get("total"))
                if len(found) < PAGE_SIZE or (total is not None and offset + PAGE_SIZE >= total):
                    break
        return unique(paths)

    async def _payload(self, site: WorkdaySite, company: str, path: str) -> dict[str, Any] | None:
        body = await self.http.get_json(f"{site.api_base}{path}")
        info = mapping(mapping(body).get("jobPostingInfo"))
        # `posted: false` is a posting taken down since it was listed.
        if not info or info.get("posted") is False:
            return None
        return self._to_payload(info, site, company, path)

    @staticmethod
    def _to_payload(
        info: Mapping[str, Any], site: WorkdaySite, company: str, path: str
    ) -> dict[str, Any]:
        """Map a Workday `jobPostingInfo` onto the conventional posting keys."""
        location = text(info.get("location"))
        additional = [text(value) for value in items(info.get("additionalLocations"))]
        remote_type = text(info.get("remoteType"))
        return {
            # The external path is the only handle the detail endpoint takes.
            "id": f"{site.key}:{path}",
            "title": info.get("title"),
            "company": company,
            "location": location,
            "locations": unique([location, *additional]),
            "url": info.get("externalUrl"),
            "description": html_to_text(text(info.get("jobDescription")) or ""),
            "remote": is_remote(location, remote_type),
            "workplace_type": remote_type,
            "employment_type": text(info.get("timeType")),
            "posted_at": text(info.get("startDate")),
            "requisition_id": text(info.get("jobReqId")),
            "site": site.key,
        }


def _queries(profile: SearchProfile) -> list[str]:
    """The search texts to send: target roles, else keywords, else one empty query."""
    roles = unique([role.strip() for role in profile.target_roles])
    if roles:
        return roles
    keywords = unique([keyword.strip() for keyword in profile.keywords])
    return keywords or [""]


def _valid_path(path: str) -> bool:
    """True for a well-formed `externalPath` that cannot climb out of `job/`."""
    return bool(_EXTERNAL_PATH.match(path)) and not any(
        segment in (".", "..") for segment in path.split("/")
    )


__all__ = ["WorkdayProvider", "WorkdaySite", "PAGE_SIZE"]
