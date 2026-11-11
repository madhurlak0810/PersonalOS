"""Health checks for the model servers a configuration names.

Only *local* servers are probed: `ollama` (`GET /api/tags`) and an
`openai_compatible` server with a `base_url` (`GET <base_url>/models`). Both
list the models they serve without running one, so a probe costs nothing. A
hosted provider (`anthropic`, or `openai_compatible` with no `base_url`) is
not probed: there is no free call, and a request to a third party on a timer
is not something this system makes on its own.

What the checks are for is the case the circuit breaker exists for: the
primary local model stops answering. While `model.<provider>` is down, work
that needs the model waits in its queue (`apps.worker.health_monitor`) and
the deterministic paths -- the Gmail poll, event routing -- carry on.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from personalos.domain.health import HealthCheckResult, model_integration, run_check

#: Where Ollama listens unless a spec says otherwise.
DEFAULT_OLLAMA_URL = "http://localhost:11434"
#: Fetches a URL and raises unless it answers with a 2xx. A port: this layer
#: holds no HTTP client, so the composition root binds one
#: (`apps.worker.health_monitor.http_get_ok`).
HttpGet = Callable[[str], Awaitable[Any]]


class LocalModelServerCheck:
    """`HealthCheck` for one local model server."""

    def __init__(self, provider: str, url: str, *, get: HttpGet):
        """Take the provider name, the listing URL to probe, and the HTTP port."""
        self.provider = provider
        self.url = url
        self.integration = model_integration(provider)
        self._get = get

    async def health_check(self) -> HealthCheckResult:
        """Down when the server's model listing does not answer with a 2xx."""
        return await run_check(self.integration, lambda: self._get(self.url))


def probe_url(provider: str, base_url: str | None) -> str | None:
    """The listing URL to probe for a spec, or `None` when it is not a local server."""
    if provider == "ollama":
        return f"{(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/tags"
    if provider == "openai_compatible" and base_url:
        return f"{base_url.rstrip('/')}/models"
    return None


__all__ = ["DEFAULT_OLLAMA_URL", "LocalModelServerCheck", "probe_url"]
