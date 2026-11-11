"""Lever job-site adapter.

Reads Lever's public Postings API (`api.lever.co/v0/postings/<site>`), the
documented, unauthenticated feed behind every `jobs.lever.co/<site>` page. It
exposes published postings only and has no write surface at all. Like
Greenhouse, a site belongs to one employer and there is no text search, so
the adapter is configured with the sites to watch and filters locally.

Sites hosted in Lever's EU region are served from `api.eu.lever.co`; name
those with an `eu/` prefix (`eu/acme`).
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
    epoch_ms_to_iso,
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

BASE_URL = "https://api.lever.co/v0/postings"
EU_BASE_URL = "https://api.eu.lever.co/v0/postings"
EU_PREFIX = "eu/"

_SITE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_POSTING_ID = re.compile(r"^[0-9a-fA-F-]{8,64}$")
#: Lever's `salaryRange.interval` values that mean an annual figure.
_ANNUAL = frozenset({"per-year-salary"})


class LeverProvider:
    """`JobProvider` over one or more Lever job sites.

    `sites` maps a site name (the slug in `jobs.lever.co/<site>`, or
    `eu/<site>` for an EU-hosted one) to the employer's display name. The
    Postings API carries no company name, so this is the only source of it.
    """

    name = "lever"

    def __init__(
        self,
        sites: Mapping[str, str],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http: ProviderHttp | None = None,
    ):
        """Take the sites to watch, and optionally a client or HTTP layer to reuse."""
        if not sites:
            raise ValueError("LeverProvider requires at least one site")
        bad = [site for site in sites if not _SITE.match(site.removeprefix(EU_PREFIX))]
        if bad:
            raise ValueError(f"invalid lever site name(s): {', '.join(bad)}")
        self.sites = dict(sites)
        self.http = http or ProviderHttp(self.name, client=client, timeout=timeout)

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return postings on the configured sites that match the profile.

        A site that fails is logged and skipped; only every site failing is
        reported as the provider being down.
        """
        postings: list[RawPosting] = []
        failed: list[str] = []
        for site, company in self.sites.items():
            try:
                body = await self.http.get_json(_url(site), params={"mode": "json"})
            except ProviderUnavailable:
                logger.warning("lever site '%s' failed during search", site, exc_info=True)
                failed.append(site)
                continue
            for job in items(body):
                payload = self._to_payload(mapping(job), site, company)
                if matches(payload, profile):
                    postings.append(RawPosting(provider=self.name, payload=payload))
        if failed and len(failed) == len(self.sites):
            raise ProviderUnavailable(f"no lever site answered: {', '.join(failed)}")
        return postings

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Fetch one posting by the `<site>:<posting_id>` id `search` assigned."""
        site, _, posting_id = source_job_id.rpartition(":")
        if site not in self.sites or not _POSTING_ID.match(posting_id):
            return None
        job = await self.http.get_json(f"{_url(site)}/{posting_id}", params={"mode": "json"})
        if not isinstance(job, Mapping):
            return None
        return RawPosting(provider=self.name, payload=self._to_payload(job, site, self.sites[site]))

    @staticmethod
    def _to_payload(job: Mapping[str, Any], site: str, company: str) -> dict[str, Any]:
        """Map a Lever posting onto the conventional posting keys."""
        categories = mapping(job.get("categories"))
        location = text(categories.get("location"))
        salary = mapping(job.get("salaryRange"))
        annual = salary.get("interval") in _ANNUAL
        workplace = text(job.get("workplaceType"))
        return {
            # Posting ids are UUIDs, but a site is what get_job needs to find one.
            "id": f"{site}:{job.get('id')}",
            "title": job.get("text"),
            "company": company,
            "location": location,
            "locations": unique([location, *map(text, items(categories.get("allLocations")))]),
            "url": job.get("hostedUrl"),
            "apply_url": text(job.get("applyUrl")),
            "description": _description(job),
            "remote": workplace == "remote" or is_remote(location),
            "workplace_type": workplace,
            "employment_type": text(categories.get("commitment")),
            "posted_at": epoch_ms_to_iso(job.get("createdAt")),
            "salary_min": number(salary.get("min")) if annual else None,
            "salary_max": number(salary.get("max")) if annual else None,
            "salary_currency": text(salary.get("currency")),
            "departments": unique(
                [text(categories.get("department")), text(categories.get("team"))]
            ),
            "site": site,
        }


def _url(site: str) -> str:
    if site.startswith(EU_PREFIX):
        return f"{EU_BASE_URL}/{site.removeprefix(EU_PREFIX)}"
    return f"{BASE_URL}/{site}"


def _description(job: Mapping[str, Any]) -> str:
    """The posting's full text, in the order the hosted page shows it.

    Lever splits a posting into an opening `description`, titled `lists`
    (responsibilities, requirements), and a closing `additional`; a board
    that keeps the same text in one field (Greenhouse, Ashby) flattens to the
    same lines, so the same opening hashes the same on all of them.
    """
    parts = [text(job.get("description")) or ""]
    for section in items(job.get("lists")):
        section = mapping(section)
        heading = text(section.get("text"))
        parts.append(f"<h3>{heading}</h3>" if heading else "")
        parts.append(f"<ul>{text(section.get('content')) or ''}</ul>")
    parts.append(text(job.get("additional")) or "")
    flattened = html_to_text("".join(f"<div>{part}</div>" for part in parts if part))
    return flattened or (text(job.get("descriptionPlain")) or "").strip()


__all__ = ["LeverProvider", "BASE_URL", "EU_BASE_URL"]
