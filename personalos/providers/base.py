"""The adapter interface every job source implements.

A provider is a read-only window onto one job board or aggregator. It returns
`RawPosting`s whose payload uses the conventional posting keys
(`id`, `title`, `company`, `location`, `url`, `description`, ...) that
`personalos.graphs.job_search.DictPostingNormalizer` reads, so adapting a new
board means writing one class here and nothing downstream.

Two things a provider is *not*:

- **Not a writer.** The interface has no method that changes anything on the
  provider's side, and `JobProviderInvoker` refuses a mutating intent outright.
  Both calls are classed `READ_EXTERNAL` in `personalos.policy.permissions`.
- **Not a source of instructions.** Everything a provider returns is text an
  outside party wrote. It is carried as data in typed fields and is never
  parsed for directives; see "Untrusted posting text" in the boundaries doc.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from personalos.domain.errors import ToolFailure
from personalos.domain.job_search import RawPosting, SearchProfile


class ProviderUnavailable(ToolFailure):
    """A provider could not be reached or answered with something unusable.

    Retryable (via `ToolFailure`): a job board timing out is the ordinary
    transient failure `dispatch_with_retry` exists for.
    """


@runtime_checkable
class JobProvider(Protocol):
    """One swappable job source.

    Structurally satisfies the graph's `JobBoardProvider` port, which needs
    only `name` and `search`.
    """

    #: Stable identifier, recorded as `job_postings.source`.
    name: str

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Return this provider's postings matching the profile."""
        ...

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Return one posting by the provider's own id, or `None` if it is gone."""
        ...


__all__ = ["JobProvider", "ProviderUnavailable"]
