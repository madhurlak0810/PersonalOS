"""Secret storage: where long-lived credentials actually live.

See `docs/ARCHITECTURE_BOUNDARIES.md`. Only `personalos.executor` and the
composition root may import this package.
"""

from .exchange import ApiKeyExchanger, GoogleOAuthTokenExchanger, TokenExchanger
from .store import (
    DEFAULT_SERVICE_NAME,
    InMemorySecretStore,
    KeyringSecretStore,
    SecretStore,
    SecretStoreUnavailable,
)

__all__ = [
    "DEFAULT_SERVICE_NAME",
    "SecretStore",
    "SecretStoreUnavailable",
    "KeyringSecretStore",
    "InMemorySecretStore",
    "TokenExchanger",
    "GoogleOAuthTokenExchanger",
    "ApiKeyExchanger",
]
