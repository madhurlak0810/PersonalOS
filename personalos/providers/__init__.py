"""Job provider adapters: the read-only I/O behind job discovery.

See ``docs/ARCHITECTURE_BOUNDARIES.md``.
"""

from .base import JobProvider, ProviderUnavailable
from .fake import FakeJobProvider
from .greenhouse import GreenhouseProvider
from .invoker import (
    GET_JOB_TOOL,
    JOB_PROVIDERS_SERVER,
    SEARCH_TOOL,
    JobProviderInvoker,
)

__all__ = [
    "JobProvider",
    "ProviderUnavailable",
    "FakeJobProvider",
    "GreenhouseProvider",
    "JobProviderInvoker",
    "JOB_PROVIDERS_SERVER",
    "SEARCH_TOOL",
    "GET_JOB_TOOL",
]
