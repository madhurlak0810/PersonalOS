"""Integration health: whether each adapter can be used right now, and what to do if not.

Every adapter that talks to something outside this process -- the Gmail
inbox, Google Calendar, the Google MCP server, each job board, the embedder,
each model provider -- exposes a `health_check()` that answers with a
`HealthCheckResult`. The worker runs those checks on a schedule
(`apps.worker.health_monitor`), and the paths that see a real call fail
report the failure too. Both end up in `integration_health`.

Three states:

- **healthy**: the last check or call succeeded.
- **degraded**: it failed. Either the provider refused permanently (revoked
  OAuth, a missing scope), so nothing works until it is reconnected and
  nothing retries; or it failed transiently, fewer than
  `DOWN_AFTER_FAILURES` times in a row, and is still used.
- **down**: it failed transiently `DOWN_AFTER_FAILURES` times in a row. The
  circuit is open: work that needs it waits, and the next successful check
  closes the circuit again.

`action` says what fixes it: `reconnect` (the user must consent again) or
`retry` (a later check will try again by itself; the user may retry
failures early).

Pure values, like the rest of `domain`.
"""

from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from personalos.domain.errors import PersonalOSError, ProviderPermanentFailure
from personalos.domain.redaction import redact_text


class HealthStatus(str, Enum):
    """Whether an integration can be used. Wire-visible."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthAction(str, Enum):
    """What would make an unhealthy integration healthy again. Wire-visible."""

    RECONNECT = "reconnect"
    RETRY = "retry"


# --- Integration names -------------------------------------------------------

#: The Google account itself: the refresh token every Google adapter uses.
GOOGLE = "google"
GMAIL = "gmail"
GOOGLE_CALENDAR = "google_calendar"
GOOGLE_MCP = "google_mcp"
#: The configured embedder.
EMBEDDER = "embedder"

#: Integrations that share the Google account's credential. A credential
#: refusal seen by any of them is a refusal for all of them.
GOOGLE_INTEGRATIONS: tuple[str, ...] = (GOOGLE, GMAIL, GOOGLE_CALENDAR, GOOGLE_MCP)

#: Reasons that mean the credential itself is gone, not just one call refused.
CREDENTIAL_REASONS = frozenset({ProviderPermanentFailure.REVOKED_AUTH, "disconnected"})

#: Transient failures in a row before an integration is `down`.
DOWN_AFTER_FAILURES = 3

#: Longest error text stored or shown.
MAX_HEALTH_ERROR_LENGTH = 500


def model_integration(provider: str) -> str:
    """The integration name of one model provider: `model.ollama`, ..."""
    return f"model.{provider}"


def affected_integrations(provider: str, reason: str | None) -> tuple[str, ...]:
    """Every integration a permanent refusal by `provider` for `reason` breaks.

    A revoked or disconnected Google credential breaks every adapter that
    uses it, so all the Google integrations are marked together and recover
    together. Anything else breaks only the integration that saw it.
    """
    if provider in GOOGLE_INTEGRATIONS and reason in CREDENTIAL_REASONS:
        return GOOGLE_INTEGRATIONS
    return (provider,)


def recovered_integrations(provider: str) -> tuple[str, ...]:
    """Every integration a reconnect of `provider` makes usable again."""
    return GOOGLE_INTEGRATIONS if provider == GOOGLE else (provider,)


def sanitize_health_error(text: str | None) -> str | None:
    """Error text fit to store and show: redacted, one line, bounded."""
    if not text:
        return None
    text = " ".join(redact_text(text).split())
    if len(text) > MAX_HEALTH_ERROR_LENGTH:
        text = text[: MAX_HEALTH_ERROR_LENGTH - 1].rstrip() + "…"
    return text


# --- Check results -------------------------------------------------------------


class HealthCheckResult(BaseModel):
    """What one health check (or one real call) found."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    integration: str
    #: `healthy` when it worked. A failure is `degraded`; the store decides
    #: when enough failures in a row make it `down`.
    status: HealthStatus
    #: A short machine-readable reason: `revoked_auth`, `missing_scope`,
    #: `unavailable`, ...; absent when healthy.
    reason: str | None = None
    #: The error, redacted.
    error: str | None = None
    #: True when only a reconnect can fix it, and nothing should retry.
    permanent: bool = False
    #: The provider the error named, when it named one (`google` for a
    #: refusal Gmail passed on).
    provider: str | None = None
    checked_at: datetime | None = None

    @property
    def ok(self) -> bool:
        """Whether the check succeeded."""
        return self.status is HealthStatus.HEALTHY

    @property
    def action(self) -> HealthAction | None:
        """What would fix it."""
        if self.ok:
            return None
        return HealthAction.RECONNECT if self.permanent else HealthAction.RETRY

    @classmethod
    def healthy(cls, integration: str, *, checked_at: datetime | None = None):
        """A check that succeeded."""
        return cls(integration=integration, status=HealthStatus.HEALTHY, checked_at=checked_at)

    @classmethod
    def from_exception(
        cls, integration: str, error: BaseException, *, checked_at: datetime | None = None
    ) -> "HealthCheckResult":
        """Classify a failed check or call.

        A `ProviderPermanentFailure` anywhere in the cause chain is permanent:
        the integration needs reconnecting and must not be retried. Anything
        else is transient.
        """
        permanent = _permanent_cause(error)
        if permanent is not None:
            return cls(
                integration=integration,
                status=HealthStatus.DEGRADED,
                reason=permanent.reason,
                error=sanitize_health_error(permanent.message),
                permanent=True,
                provider=permanent.provider,
                checked_at=checked_at,
            )
        message = error.message if isinstance(error, PersonalOSError) else str(error)
        text = f"{type(error).__name__}: {message}" if message else type(error).__name__
        return cls(
            integration=integration,
            status=HealthStatus.DEGRADED,
            reason="unavailable",
            error=sanitize_health_error(text),
            checked_at=checked_at,
        )


def _permanent_cause(error: BaseException) -> ProviderPermanentFailure | None:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderPermanentFailure):
            return current
        current = current.__cause__ or current.__context__
    return None


@runtime_checkable
class HealthCheck(Protocol):
    """One adapter's health check. Never raises: a failure is a result."""

    #: The integration it reports on, as named above.
    integration: str

    async def health_check(self) -> HealthCheckResult:
        """Ask the adapter's provider, cheaply, whether it works right now."""
        ...


async def run_check(integration: str, probe, *, checked_at: datetime | None = None):
    """Await `probe()` and turn its outcome into a `HealthCheckResult`.

    The helper every adapter's `health_check` is written with, so they all
    classify failures the same way.
    """
    try:
        await probe()
    except Exception as error:  # noqa: BLE001 - a failed probe is a result, not a crash
        return HealthCheckResult.from_exception(integration, error, checked_at=checked_at)
    return HealthCheckResult.healthy(integration, checked_at=checked_at)


def is_blocking(status: str, reconnect_required: bool) -> bool:
    """Whether work needing an integration in this state should wait.

    `down`, or degraded by a permanent refusal. A transient failure or two
    (`degraded` with `retry`) does not open the circuit.
    """
    return status == HealthStatus.DOWN.value or reconnect_required


__all__ = [
    "CREDENTIAL_REASONS",
    "DOWN_AFTER_FAILURES",
    "EMBEDDER",
    "GMAIL",
    "GOOGLE",
    "GOOGLE_CALENDAR",
    "GOOGLE_INTEGRATIONS",
    "GOOGLE_MCP",
    "MAX_HEALTH_ERROR_LENGTH",
    "HealthAction",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    "affected_integrations",
    "is_blocking",
    "model_integration",
    "recovered_integrations",
    "run_check",
    "sanitize_health_error",
]
