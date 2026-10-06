"""Tool-boundary adapter for job providers.

Bridges the `ToolInvoker` port to the registered `JobProvider`s, so a provider
call is reachable only as an `ApprovedIntent` -- which is what makes its
`READ_EXTERNAL` classification a decision the policy engine recorded rather
than a comment.
"""

from collections.abc import Iterable
from typing import Any

from personalos.domain.errors import ErrorCode, PersonalOSError
from personalos.domain.job_search import SearchProfile
from personalos.policy import ApprovedIntent, PolicyViolation
from personalos.providers.base import JobProvider

#: Server name provider calls are addressed to, as in `job_providers.search`.
JOB_PROVIDERS_SERVER = "job_providers"
SEARCH_TOOL = "search"
GET_JOB_TOOL = "get_job"


class JobProviderInvoker:
    """Executes approved `job_providers.*` intents against named providers."""

    def __init__(self, providers: Iterable[JobProvider]):
        """Register providers by their `name`."""
        self.providers: dict[str, JobProvider] = {}
        for provider in providers:
            if provider.name in self.providers:
                raise ValueError(f"duplicate job provider name '{provider.name}'")
            self.providers[provider.name] = provider

    async def invoke(self, approved: ApprovedIntent) -> dict[str, Any]:
        """Run one read against one provider, returning an adapter payload."""
        if not isinstance(approved, ApprovedIntent):
            raise PolicyViolation(
                f"JobProviderInvoker requires an ApprovedIntent, got {type(approved).__name__}"
            )
        if approved.intent.mutating:
            raise PolicyViolation(
                f"job providers are read-only; refusing mutating intent '{approved.intent.tool_ref}'"
            )

        arguments = approved.arguments
        provider = self.providers.get(str(arguments.get("provider")))
        if provider is None:
            return _failure(
                f"unknown job provider '{arguments.get('provider')}'", ErrorCode.NOT_FOUND
            )

        try:
            if approved.tool == SEARCH_TOOL:
                profile = SearchProfile.model_validate(arguments.get("profile"))
                found = await provider.search(profile)
                postings = [raw.model_dump(mode="json") for raw in found]
            elif approved.tool == GET_JOB_TOOL:
                raw = await provider.get_job(str(arguments.get("source_job_id")))
                postings = [raw.model_dump(mode="json")] if raw else []
            else:
                return _failure(f"unknown job provider tool '{approved.tool}'", ErrorCode.NOT_FOUND)
        except PersonalOSError as exc:
            return _failure(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001 - a provider bug is a tool failure, not a crash
            return _failure(f"{type(exc).__name__}: {exc}", ErrorCode.TOOL_FAILURE)

        return {"success": True, "result": {"postings": postings}, "error": None}


def _failure(error: str, code: ErrorCode) -> dict[str, Any]:
    return {"success": False, "result": None, "error": error, "error_code": code.value}


__all__ = ["JobProviderInvoker", "JOB_PROVIDERS_SERVER", "SEARCH_TOOL", "GET_JOB_TOOL"]
