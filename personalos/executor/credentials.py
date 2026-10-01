"""The credential broker: a reference in, a short-lived token out, at execution time.

This is the only component that reads a long-lived secret. It sits in the
executor layer because that is the only layer that acts: orchestration and
model code hold a `CredentialRef` at most, and have no import path to a secret
store (see `tests/architecture/boundaries.py`), so the question "could a
refresh token have reached a prompt or a checkpoint?" is answered by the layer
graph rather than by reading every node.

Three properties are load-bearing:

1. **Just in time.** `exchange` is called by the code about to make the
   provider call, and nothing is cached: no token outlives the execution that
   asked for it, and nothing here is worth stealing between calls.
2. **Nothing leaves but the token.** The long-lived secret is read, handed to
   the provider's exchanger, and dropped. Callers get an `AccessToken`.
3. **Everything it touches becomes redactable.** Both the secret it read and
   the token it minted are registered with the redactor before either is used,
   so if a provider echoes one into an error, every sink already knows to
   remove it.
"""

import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from personalos.domain.credentials import (
    AccessToken,
    CredentialExchangeFailed,
    CredentialNotFound,
    CredentialRef,
)
from personalos.domain.redaction import Redactor, default_redactor
from personalos.secrets.exchange import TokenExchanger
from personalos.secrets.store import SecretStore

logger = logging.getLogger(__name__)

#: Upper bound on how long a brokered token is treated as usable, whatever the
#: provider claimed. "Short-lived" is a property this code guarantees rather
#: than one it hopes the provider supplies.
MAX_ACCESS_TOKEN_TTL = timedelta(hours=1)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CredentialBroker:
    """Exchanges credential references for short-lived access tokens.

    Satisfies `personalos.domain.credentials.AccessTokenProvider`, which is the
    type a tool implementation should depend on.
    """

    def __init__(
        self,
        store: SecretStore,
        exchangers: Mapping[str, TokenExchanger],
        *,
        redactor: Redactor | None = None,
        max_ttl: timedelta = MAX_ACCESS_TOKEN_TTL,
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Take the secret store and one exchanger per provider name."""
        self._store = store
        self._exchangers = dict(exchangers)
        self._redactor = redactor or default_redactor()
        self._max_ttl = max_ttl
        self._clock = clock

    async def exchange(self, ref: CredentialRef | str, scopes: Sequence[str] = ()) -> AccessToken:
        """Mint a short-lived token for `ref`.

        Raises `CredentialNotFound` if the store holds no secret under the
        reference, and `CredentialExchangeFailed` if the provider has no
        exchanger or the exchange fails.
        """
        ref = CredentialRef.parse(ref)
        exchanger = self._exchangers.get(ref.provider)
        if exchanger is None:
            raise CredentialExchangeFailed(
                f"no token exchanger is configured for provider '{ref.provider}'"
            )

        secret = self._store.get(ref)
        if secret is None:
            raise CredentialNotFound(f"no credential is stored for {ref}")
        # Registered before use: an exchanger failure is the likeliest moment
        # for the secret to be echoed back in an error.
        self._redactor.register_secret(secret.reveal())

        token = await exchanger.exchange(ref, secret, tuple(scopes))

        now = self._clock()
        latest = now + self._max_ttl
        if token.expires_at > latest:
            token = AccessToken(
                credential_ref=token.credential_ref,
                secret=token.secret,
                expires_at=latest,
                scopes=token.scopes,
                token_type=token.token_type,
            )
        self._redactor.register_secret(token.secret.reveal(), expires_at=token.expires_at)

        logger.info(
            "brokered access token for %s (scopes=%s, expires_in=%ss)",
            ref,
            ",".join(token.scopes) or "-",
            int((token.expires_at - now).total_seconds()),
        )
        return token

    @asynccontextmanager
    async def lease(
        self, ref: CredentialRef | str, scopes: Sequence[str] = ()
    ) -> AsyncIterator[AccessToken]:
        """Scope a token to one block: `async with broker.lease(ref) as token`.

        The shape to prefer over `exchange` in an executor, because it makes
        the token's lifetime visible in the code -- it exists for the provider
        call inside the block and is not a value to return or store.
        """
        token = await self.exchange(ref, scopes)
        yield token


__all__ = ["MAX_ACCESS_TOKEN_TTL", "CredentialBroker"]
