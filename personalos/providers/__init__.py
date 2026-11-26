"""Job provider adapters: the read-only I/O behind job discovery.

See ``docs/ARCHITECTURE_BOUNDARIES.md``.

``FakeJobProvider`` is imported only when asked for: it serves canned
postings for tests and the quick start, and the production container image
leaves its module out (see ``Dockerfile``).
"""

from typing import Any

from .ashby import AshbyProvider
from .base import JobProvider, ProviderUnavailable
from .greenhouse import GreenhouseProvider
from .http import ProviderHttp, RateLimiter
from .invoker import (
    GET_JOB_TOOL,
    JOB_PROVIDERS_SERVER,
    SEARCH_TOOL,
    JobProviderInvoker,
)
from .lever import LeverProvider
from .workday import WorkdayProvider

__all__ = [
    "JobProvider",
    "ProviderUnavailable",
    "FakeJobProvider",
    "GreenhouseProvider",
    "LeverProvider",
    "AshbyProvider",
    "WorkdayProvider",
    "ProviderHttp",
    "RateLimiter",
    "JobProviderInvoker",
    "JOB_PROVIDERS_SERVER",
    "SEARCH_TOOL",
    "GET_JOB_TOOL",
]


def __getattr__(name: str) -> Any:
    """Import `FakeJobProvider` on first use, so a build without it still imports."""
    if name == "FakeJobProvider":
        from .fake import FakeJobProvider

        return FakeJobProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
