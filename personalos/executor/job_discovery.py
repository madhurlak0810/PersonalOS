"""Job discovery through the tool boundary.

`GatewayJobProvider` is what the Job Search graph is actually handed as a
provider. It holds no HTTP client and no adapter -- only a `ToolGateway` --
so every search and every `get_job` is a `ToolIntent` the policy engine
classifies (`READ_EXTERNAL`), records, and can refuse.
"""

from collections.abc import Sequence
from typing import Any

from personalos.domain.context import ExecutionContext
from personalos.domain.job_search import RawPosting, SearchProfile
from personalos.executor.retry import dispatch_with_retry
from personalos.policy import IntentOrigin, ToolIntent
from personalos.tools.gateway import ToolGateway

#: Mirrors `personalos.providers.invoker`; the executor layer may not import an
#: adapter package, so the address is restated here as the policy tables do.
JOB_PROVIDERS_SERVER = "job_providers"
SEARCH_TOOL = "search"
GET_JOB_TOOL = "get_job"


class GatewayJobProvider:
    """A `JobProvider` whose calls are dispatched as policy-checked intents."""

    def __init__(
        self,
        name: str,
        gateway: ToolGateway,
        *,
        context: ExecutionContext | None = None,
    ):
        """Bind one provider name to the gateway that can reach it."""
        if gateway is None:
            raise ValueError("GatewayJobProvider requires a ToolGateway")
        self.name = name
        self.gateway = gateway
        self.context = context

    async def search(self, profile: SearchProfile) -> Sequence[RawPosting]:
        """Search the provider for postings matching the profile."""
        return await self._read(
            SEARCH_TOOL, {"provider": self.name, "profile": profile.model_dump(mode="json")}
        )

    async def get_job(self, source_job_id: str) -> RawPosting | None:
        """Fetch one posting by the provider's own id."""
        postings = await self._read(
            GET_JOB_TOOL, {"provider": self.name, "source_job_id": source_job_id}
        )
        return postings[0] if postings else None

    async def _read(self, tool: str, arguments: dict[str, Any]) -> list[RawPosting]:
        """Dispatch one read. Never mutating, and `SYSTEM` origin: the intent's
        shape is fixed here, and nothing a posting says can reach it."""
        intent = ToolIntent(
            server=JOB_PROVIDERS_SERVER,
            tool=tool,
            arguments=arguments,
            origin=IntentOrigin.SYSTEM,
            mutating=False,
            requested_by=f"executor:job_discovery#{tool}",
            **({"context": self.context} if self.context else {}),
        )
        result = await dispatch_with_retry(self.gateway, intent)
        return [RawPosting.model_validate(raw) for raw in result.unwrap().get("postings", [])]


__all__ = ["GatewayJobProvider"]
