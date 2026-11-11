"""An in-memory `JobProvider`, for tests and for running without a network."""

from collections.abc import Iterable, Sequence
from typing import Any

from personalos.domain.job_search import RawPosting, SearchProfile


class FakeJobProvider:
    """Serves a fixed set of posting payloads and records how it was called."""

    def __init__(self, name: str = "fakeboard", postings: Iterable[dict[str, Any]] = ()):
        """Take posting-shaped payloads; each needs an `id` to be fetchable."""
        self.name = name
        self.postings = [dict(payload) for payload in postings]
        self.search_calls: list[SearchProfile] = []
        self.get_job_calls: list[str] = []
        self.health_calls = 0
        #: Raised by `health_check` while set.
        self.health_error: Exception | None = None

    async def health_check(self) -> None:
        """Answer, or raise `health_error` when a test set one."""
        self.health_calls += 1
        if self.health_error is not None:
            raise self.health_error

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return every posting; filtering is the pipeline's job, not the fake's."""
        self.search_calls.append(profile)
        return [RawPosting(provider=self.name, payload=dict(p)) for p in self.postings]

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Return the posting with this id, or `None`."""
        self.get_job_calls.append(source_job_id)
        for payload in self.postings:
            if str(payload.get("id")) == source_job_id:
                return RawPosting(provider=self.name, payload=dict(payload))
        return None


__all__ = ["FakeJobProvider"]
