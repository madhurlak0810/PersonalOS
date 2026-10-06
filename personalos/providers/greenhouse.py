"""Greenhouse job-board adapter.

Reads the public Job Board API (`boards-api.greenhouse.io`), which needs no
credential and exposes nothing but published postings. Greenhouse has no
cross-company search: a board belongs to one employer, so this adapter is
configured with the board tokens to watch and filters their postings against
the profile locally.
"""

import html
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from personalos.domain.job_search import RawPosting, SearchProfile
from personalos.providers.base import ProviderUnavailable

logger = logging.getLogger(__name__)

BASE_URL = "https://boards-api.greenhouse.io/v1/boards"
DEFAULT_TIMEOUT_SECONDS = 15.0

#: Tags whose end marks a line break when HTML is flattened to text.
_BLOCK_TAGS = re.compile(r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr)\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
_NON_TEXT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


def html_to_text(content: str) -> str:
    """Flatten a Greenhouse `content` field to plain text.

    The field is HTML that has itself been entity-escaped, so it is unescaped
    before tags are removed and once more after, for entities inside the text.
    """
    text = _NON_TEXT.sub(" ", html.unescape(content))
    text = _ANY_TAG.sub("", _BLOCK_TAGS.sub("\n", text))
    lines = (" ".join(line.split()) for line in html.unescape(text).splitlines())
    return "\n".join(line for line in lines if line)


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
    ):
        """Take the boards to watch, and optionally a client to reuse."""
        if not boards:
            raise ValueError("GreenhouseProvider requires at least one board token")
        self.boards = dict(boards)
        self._client = client
        self._timeout = timeout

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
            for job in (body or {}).get("jobs", []):
                payload = self._to_payload(job, token, company)
                if _matches(payload, profile):
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
        if job is None:
            return None
        return RawPosting(
            provider=self.name, payload=self._to_payload(job, token, self.boards[token])
        )

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """GET one API path, returning parsed JSON, or `None` on a 404."""
        url = f"{BASE_URL}/{path}"
        try:
            if self._client is not None:
                response = await self._client.get(url, params=params)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailable(f"greenhouse request failed for '{path}': {exc}") from exc

    @staticmethod
    def _to_payload(job: Mapping[str, Any], token: str, company: str) -> dict[str, Any]:
        """Map a Greenhouse job onto the conventional posting keys."""
        location = (job.get("location") or {}).get("name")
        return {
            # Job ids are only unique within a board.
            "id": f"{token}:{job.get('id')}",
            "title": job.get("title"),
            "company": job.get("company_name") or company,
            "location": location,
            "url": job.get("absolute_url"),
            "description": html_to_text(job.get("content") or ""),
            "remote": bool(location and "remote" in location.lower()),
            "posted_at": job.get("first_published") or job.get("updated_at"),
            "departments": [d.get("name") for d in job.get("departments") or []],
            "board_token": token,
        }


def _matches(payload: Mapping[str, Any], profile: SearchProfile) -> bool:
    """True when a posting mentions any of the profile's roles or keywords.

    A profile with neither matches everything. This is a coarse recall filter
    standing in for the search endpoint Greenhouse does not have; precision is
    `hard_filter` and the scorer's job.
    """
    terms = [term.lower() for term in (*profile.target_roles, *profile.keywords) if term.strip()]
    if not terms:
        return True
    haystack = f"{payload.get('title') or ''} {payload.get('description') or ''}".lower()
    return any(term in haystack for term in terms)


__all__ = ["GreenhouseProvider", "html_to_text", "BASE_URL"]
