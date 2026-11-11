"""Greenhouse job-board adapter.

Reads the public Job Board API (`boards-api.greenhouse.io`), which needs no
credential and exposes nothing but published postings. Greenhouse has no
cross-company search: a board belongs to one employer, so this adapter is
configured with the board tokens to watch and filters their postings against
the profile locally.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from personalos.domain.job_search import RawPosting, SearchProfile
from personalos.providers.base import ProviderUnavailable
from personalos.providers.http import DEFAULT_TIMEOUT_SECONDS, ProviderHttp
from personalos.providers.text import html_to_text, is_remote, items, mapping, matches, text

logger = logging.getLogger(__name__)

BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

_BOARD_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


class GreenhouseProvider:
    """`JobProvider` over one or more Greenhouse boards.

    `boards` maps a board token (the slug in `boards.greenhouse.io/<token>`) to
    the employer's display name; the API's per-job `company_name` is used when
    present and this is the fallback.
    """

    name = "greenhouse"

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
            raise ValueError("GreenhouseProvider requires at least one board token")
        bad = [token for token in boards if not _BOARD_TOKEN.match(token)]
        if bad:
            raise ValueError(f"invalid greenhouse board token(s): {', '.join(bad)}")
        self.boards = dict(boards)
        self.http = http or ProviderHttp(self.name, client=client, timeout=timeout)

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return postings on the configured boards that match the profile.

        A board that fails is logged and skipped so the others still answer;
        only every board failing is reported as the provider being down.
        """
        postings: list[RawPosting] = []
        failed: list[str] = []
        for token, company in self.boards.items():
            try:
                body = await self._get(f"{token}/jobs", params={"content": "true"})
            except ProviderUnavailable:
                logger.warning("greenhouse board '%s' failed during search", token, exc_info=True)
                failed.append(token)
                continue
            for job in items(mapping(body).get("jobs")):
                payload = self._to_payload(mapping(job), token, company)
                if matches(payload, profile):
                    postings.append(RawPosting(provider=self.name, payload=payload))
        if failed and len(failed) == len(self.boards):
            raise ProviderUnavailable(f"no greenhouse board answered: {', '.join(failed)}")
        return postings

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Fetch one posting by the `<board_token>:<job_id>` id `search` assigned."""
        token, _, job_id = source_job_id.partition(":")
        if token not in self.boards or not job_id.isdigit():
            return None
        job = await self._get(f"{token}/jobs/{job_id}")
        if not isinstance(job, Mapping):
            return None
        return RawPosting(
            provider=self.name, payload=self._to_payload(job, token, self.boards[token])
        )

    async def health_check(self) -> None:
        """List the first board's jobs without their content. Raises if it does not answer."""
        token = next(iter(self.boards))
        await self._get(f"{token}/jobs")

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """GET one API path, returning parsed JSON, or `None` on a 404."""
        return await self.http.get_json(f"{BASE_URL}/{path}", params=params)

    @staticmethod
    def _to_payload(job: Mapping[str, Any], token: str, company: str) -> dict[str, Any]:
        """Map a Greenhouse job onto the conventional posting keys."""
        location = text(mapping(job.get("location")).get("name"))
        return {
            # Job ids are only unique within a board.
            "id": f"{token}:{job.get('id')}",
            "title": job.get("title"),
            "company": text(job.get("company_name")) or company,
            "location": location,
            "url": job.get("absolute_url"),
            "description": html_to_text(text(job.get("content")) or ""),
            "remote": is_remote(location),
            "posted_at": text(job.get("first_published")) or text(job.get("updated_at")),
            "departments": [text(mapping(d).get("name")) for d in items(job.get("departments"))],
            "board_token": token,
        }


__all__ = ["GreenhouseProvider", "html_to_text", "BASE_URL"]
