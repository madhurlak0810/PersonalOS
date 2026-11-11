"""Ashby job-board adapter.

Reads Ashby's public Job Postings API
(`api.ashbyhq.com/posting-api/job-board/<board>`), the documented,
unauthenticated feed behind `jobs.ashbyhq.com/<board>`. It returns one
employer's published postings in a single response and has no write surface.
There is no per-posting endpoint and no text search, so `get_job` re-reads the
board and `search` filters locally.

A posting with `isListed: false` is reachable by direct link but was not
advertised; it is dropped, since the employer chose not to publish it.
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

BASE_URL = "https://api.ashbyhq.com/posting-api/job-board"

_BOARD = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_POSTING_ID = re.compile(r"^[0-9a-fA-F-]{8,64}$")
#: Ashby's compensation `interval` for an annual figure.
_ANNUAL = "1 YEAR"


class AshbyProvider:
    """`JobProvider` over one or more Ashby job boards.

    `boards` maps a board name (the slug in `jobs.ashbyhq.com/<board>`) to the
    employer's display name; the API carries no company name of its own.
    """

    name = "ashby"

    def __init__(
        self,
        boards: Mapping[str, str],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http: ProviderHttp | None = None,
    ):
        """Take the boards to watch, and optionally a client or HTTP layer to reuse."""
        if not boards:
            raise ValueError("AshbyProvider requires at least one board")
        bad = [board for board in boards if not _BOARD.match(board)]
        if bad:
            raise ValueError(f"invalid ashby board name(s): {', '.join(bad)}")
        self.boards = dict(boards)
        self.http = http or ProviderHttp(self.name, client=client, timeout=timeout)

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return listed postings on the configured boards that match the profile.

        A board that fails is logged and skipped; only every board failing is
        reported as the provider being down.
        """
        postings: list[RawPosting] = []
        failed: list[str] = []
        for board, company in self.boards.items():
            try:
                jobs = await self._jobs(board)
            except ProviderUnavailable:
                logger.warning("ashby board '%s' failed during search", board, exc_info=True)
                failed.append(board)
                continue
            for job in jobs:
                payload = self._to_payload(job, board, company)
                if matches(payload, profile):
                    postings.append(RawPosting(provider=self.name, payload=payload))
        if failed and len(failed) == len(self.boards):
            raise ProviderUnavailable(f"no ashby board answered: {', '.join(failed)}")
        return postings

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Find one posting by the `<board>:<posting_id>` id `search` assigned."""
        board, _, posting_id = source_job_id.partition(":")
        if board not in self.boards or not _POSTING_ID.match(posting_id):
            return None
        for job in await self._jobs(board):
            if job.get("id") == posting_id:
                return RawPosting(
                    provider=self.name, payload=self._to_payload(job, board, self.boards[board])
                )
        return None

    async def _jobs(self, board: str) -> list[Mapping[str, Any]]:
        """The board's listed postings, as objects."""
        body = await self.http.get_json(
            f"{BASE_URL}/{board}", params={"includeCompensation": "true"}
        )
        jobs = (mapping(job) for job in items(mapping(body).get("jobs")))
        return [job for job in jobs if job and job.get("isListed") is not False]

    @staticmethod
    def _to_payload(job: Mapping[str, Any], board: str, company: str) -> dict[str, Any]:
        """Map an Ashby posting onto the conventional posting keys."""
        location = text(job.get("location"))
        secondary = [text(mapping(s).get("location")) for s in items(job.get("secondaryLocations"))]
        workplace = text(job.get("workplaceType"))
        salary_min, salary_max, currency = _annual_salary(mapping(job.get("compensation")))
        description = html_to_text(text(job.get("descriptionHtml")) or "")
        return {
            "id": f"{board}:{job.get('id')}",
            "title": job.get("title"),
            "company": company,
            "location": location,
            "locations": unique([location, *secondary]),
            "url": job.get("jobUrl"),
            "apply_url": text(job.get("applyUrl")),
            "description": description or (text(job.get("descriptionPlain")) or "").strip(),
            "remote": is_remote(job.get("isRemote"), workplace, location),
            "workplace_type": workplace,
            "employment_type": text(job.get("employmentType")),
            "posted_at": text(job.get("publishedAt")),
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": currency,
            "departments": unique([text(job.get("department")), text(job.get("team"))]),
            "board": board,
        }


def _annual_salary(compensation: Mapping[str, Any]) -> tuple[int | None, int | None, str | None]:
    """The annual salary range from `summaryComponents`, if the board shows one."""
    for component in items(compensation.get("summaryComponents")):
        component = mapping(component)
        if component.get("compensationType") == "Salary" and component.get("interval") == _ANNUAL:
            return (
                number(component.get("minValue")),
                number(component.get("maxValue")),
                text(component.get("currencyCode")),
            )
    return None, None, None


__all__ = ["AshbyProvider", "BASE_URL"]
