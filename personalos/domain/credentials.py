"""Credential identity: references that may be stored, and secrets that may not.

Two kinds of value, and keeping them apart is the whole design:

`CredentialRef`
    A *name* for a credential -- `cred://google/alice@example.com`. It says
    which secret, not what the secret is. It is safe everywhere: in a database
    row, in graph state, in a log line, in a prompt. This is the only thing the
    rest of the system ever holds.

`SecretValue` / `AccessToken`
    The secret itself. It exists in exactly two places: the OS secret store
    (`personalos.secrets`), and briefly in memory inside an executor that asked
    the credential broker for it. It is built to be hard to write down by
    accident -- its `repr` and `str` are a redaction marker, it refuses to be
    pickled or copied, and no serializer in this codebase knows how to encode
    it -- so a `SecretValue` that strays into graph state fails the checkpoint
    write loudly instead of being persisted quietly.

`AccessTokenProvider` is the port a tool implementation depends on to get a
token at the moment it makes a call. The broker in `personalos.executor`
implements it; the composition root binds it.
"""

import hmac
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

from personalos.domain.errors import NotFound, RetryableFailure, ValidationFailed
from personalos.domain.redaction import REDACTED

#: URI scheme of a credential reference.
CREDENTIAL_REF_SCHEME = "cred"

_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,199}$")


class CredentialKind(str, Enum):
    """What sort of long-lived secret a reference names."""

    #: An OAuth refresh token, exchanged for a short-lived access token.
    OAUTH_REFRESH_TOKEN = "oauth_refresh_token"
    #: A static provider API key. It cannot be exchanged for anything shorter
    #: lived, so the broker leases the key itself, for one call.
    API_KEY = "api_key"
    #: An OAuth client secret, used by an exchanger rather than by a tool.
    OAUTH_CLIENT_SECRET = "oauth_client_secret"


class InvalidCredentialRef(ValidationFailed, ValueError):
    """A string was offered as a credential reference and is not one."""

    default_message = "invalid credential reference"


class CredentialNotFound(NotFound):
    """A reference names a credential the secret store does not hold."""

    default_message = "credential not found"


class CredentialExchangeFailed(RetryableFailure):
    """A reference could not be exchanged for an access token.

    The message carries the reference and the provider's error *code* only,
    never the response body: a token endpoint's error response is exactly the
    kind of text that repeats the credential it was sent.
    """

    default_message = "credential exchange failed"


class CredentialRevoked(ValidationFailed):
    """The provider rejected the stored credential outright; retrying cannot help."""

    default_message = "credential was revoked or is no longer valid"


@dataclass(frozen=True)
class CredentialRef:
    """The name of a credential: `cred://<provider>/<name>`.

    Deliberately tiny. `provider` selects the exchanger (`google`,
    `greenhouse`); `name` distinguishes credentials within it (an account
    address, a key label). Both are restricted to a character set in which a
    token could not plausibly be smuggled -- a reference that could hold a
    secret would defeat the point of storing references.
    """

    provider: str
    name: str

    def __post_init__(self) -> None:
        """Reject anything that is not a well-formed reference."""
        if not isinstance(self.provider, str) or not _PROVIDER_PATTERN.match(self.provider):
            raise InvalidCredentialRef(f"invalid credential provider: {self.provider!r}")
        if not isinstance(self.name, str) or not _NAME_PATTERN.match(self.name):
            raise InvalidCredentialRef("invalid credential name")

    @classmethod
    def parse(cls, value: "str | CredentialRef") -> "CredentialRef":
        """Build a reference from its `cred://provider/name` string form."""
        if isinstance(value, CredentialRef):
            return value
        prefix = f"{CREDENTIAL_REF_SCHEME}://"
        if not isinstance(value, str) or not value.startswith(prefix):
            raise InvalidCredentialRef(f"a credential reference starts with '{prefix}'")
        provider, separator, name = value[len(prefix) :].partition("/")
        if not separator:
            raise InvalidCredentialRef("a credential reference is cred://<provider>/<name>")
        return cls(provider=provider, name=name)

    def __str__(self) -> str:
        return f"{CREDENTIAL_REF_SCHEME}://{self.provider}/{self.name}"


class SecretValue:
    """A secret held in memory, wrapped so it is awkward to leak.

    `reveal()` is the one way to the underlying string, which makes every use
    of a raw secret greppable. Everything else that would turn the object into
    text or bytes either yields the redaction marker or refuses.
    """

    __slots__ = ("_value",)

    #: Marker `personalos.domain.redaction` looks for, so the redactor can
    #: recognise a secret without importing this module.
    __personalos_secret__ = True

    def __init__(self, value: str):
        """Wrap a non-empty secret string."""
        if not isinstance(value, str) or not value:
            raise ValueError("a secret value must be a non-empty string")
        object.__setattr__(self, "_value", value)

    def reveal(self) -> str:
        """Return the secret. Call this at the point of use and nowhere else."""
        return self._value

    def __repr__(self) -> str:
        return f"SecretValue({REDACTED})"

    def __str__(self) -> str:
        return REDACTED

    def __format__(self, format_spec: str) -> str:
        return REDACTED

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SecretValue):
            return NotImplemented
        return hmac.compare_digest(self._value.encode("utf-8"), other._value.encode("utf-8"))

    __hash__ = None  # type: ignore[assignment]

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SecretValue is immutable")

    def __reduce__(self):
        raise TypeError("a SecretValue cannot be pickled, copied or serialized")

    def __getstate__(self):
        raise TypeError("a SecretValue cannot be pickled, copied or serialized")


@dataclass(frozen=True)
class AccessToken:
    """A short-lived credential minted by the broker for one execution.

    Holds the reference it was minted from, so the token's provenance is
    visible without the token being. Its `repr` shows the redaction marker in
    place of the secret.
    """

    credential_ref: CredentialRef
    secret: SecretValue
    expires_at: datetime
    scopes: tuple[str, ...] = ()
    token_type: str = "Bearer"

    def is_expired(self, now: datetime | None = None) -> bool:
        """True once the token may no longer be presented."""
        return (now or datetime.now(timezone.utc)) >= self.expires_at

    def authorization_header(self) -> dict[str, str]:
        """The header a provider call carries. Never log the return value."""
        return {"Authorization": f"{self.token_type} {self.secret.reveal()}"}

    def __reduce__(self):
        raise TypeError("an AccessToken cannot be pickled, copied or serialized")


@runtime_checkable
class AccessTokenProvider(Protocol):
    """Port: turn a credential reference into a token, at the moment of use."""

    async def exchange(
        self, ref: "CredentialRef | str", scopes: Sequence[str] = ()
    ) -> AccessToken:
        """Mint a short-lived token for `ref`, limited to `scopes`."""
        ...


__all__ = [
    "CREDENTIAL_REF_SCHEME",
    "CredentialKind",
    "InvalidCredentialRef",
    "CredentialNotFound",
    "CredentialExchangeFailed",
    "CredentialRevoked",
    "CredentialRef",
    "SecretValue",
    "AccessToken",
    "AccessTokenProvider",
]
