"""Secret stores: the only place a long-lived credential is at rest.

A refresh token or API key is written here and nowhere else. The database
holds a `CredentialRef` (see `personalos.persistence.models.CredentialModel`);
this module turns that reference into the secret, and only the credential
broker asks it to.

`KeyringSecretStore` is the production store, backed by whatever the operating
system provides -- Keychain on macOS, Secret Service / KWallet on Linux,
Credential Locker on Windows -- through the `keyring` package. It refuses to
run on a backend that does not actually protect anything: `keyring` will
happily fall back to a plaintext file or a null store if nothing better is
installed, and a secret store that silently became a text file is worse than
one that fails to start.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from personalos.domain.credentials import CredentialRef, SecretValue
from personalos.domain.errors import InternalError

logger = logging.getLogger(__name__)

#: The service name every PersonalOS secret is filed under in the OS keychain.
DEFAULT_SERVICE_NAME = "personalos"

#: `keyring` backends that store nothing, or store it unprotected.
_UNSAFE_BACKEND_MODULES = (
    "keyring.backends.fail",
    "keyring.backends.null",
    "keyrings.alt",
)


class SecretStoreUnavailable(InternalError):
    """No usable secret store: the package or a protected backend is missing."""

    default_message = "secret store is unavailable"


@runtime_checkable
class SecretStore(Protocol):
    """Port for reading and writing secrets by reference."""

    def get(self, ref: CredentialRef) -> SecretValue | None:
        """Return the secret a reference names, or `None` if it holds none."""
        ...

    def put(self, ref: CredentialRef, secret: SecretValue) -> None:
        """Store (or replace) the secret a reference names."""
        ...

    def delete(self, ref: CredentialRef) -> None:
        """Remove the secret a reference names; a no-op if absent."""
        ...


class KeyringSecretStore:
    """A `SecretStore` over the operating system's keychain."""

    def __init__(self, service_name: str = DEFAULT_SERVICE_NAME, *, backend: Any = None):
        """Bind to a keychain service.

        `backend` is anything with `keyring`'s `get_password` / `set_password`
        / `delete_password`; left unset, the `keyring` package's configured
        backend is used and checked.
        """
        self.service_name = service_name
        self._backend = backend if backend is not None else self._load_backend()

    @staticmethod
    def _load_backend() -> Any:
        try:
            import keyring
        except ImportError as error:
            raise SecretStoreUnavailable(
                "the 'keyring' package is not installed; it is required to store credentials"
            ) from error

        active = keyring.get_keyring()
        module = type(active).__module__
        if module.startswith(_UNSAFE_BACKEND_MODULES):
            raise SecretStoreUnavailable(
                f"keyring backend '{module}' does not protect secrets; install an OS "
                f"keychain backend (Keychain, Secret Service, KWallet, Credential Locker)"
            )
        return keyring

    def get(self, ref: CredentialRef) -> SecretValue | None:
        """Read a secret from the keychain."""
        value = self._backend.get_password(self.service_name, str(ref))
        return SecretValue(value) if value else None

    def put(self, ref: CredentialRef, secret: SecretValue) -> None:
        """Write a secret to the keychain."""
        self._backend.set_password(self.service_name, str(ref), secret.reveal())
        logger.info("stored credential %s", ref)

    def delete(self, ref: CredentialRef) -> None:
        """Remove a secret from the keychain."""
        try:
            self._backend.delete_password(self.service_name, str(ref))
        except Exception as error:
            # `keyring` raises PasswordDeleteError for a missing entry, and the
            # class lives in the package this module imports lazily. Deleting
            # something already gone is the outcome the caller wanted.
            logger.debug("credential %s not deleted: %s", ref, type(error).__name__)
            return
        logger.info("deleted credential %s", ref)


class InMemorySecretStore:
    """A `SecretStore` that forgets everything when the process ends.

    For tests and for a development run with no keychain. Not a fallback: the
    composition root never substitutes this for `KeyringSecretStore` on its own.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._secrets: dict[str, SecretValue] = {}

    def get(self, ref: CredentialRef) -> SecretValue | None:
        """Return the stored secret, if any."""
        return self._secrets.get(str(ref))

    def put(self, ref: CredentialRef, secret: SecretValue) -> None:
        """Store a secret."""
        self._secrets[str(ref)] = secret

    def delete(self, ref: CredentialRef) -> None:
        """Remove a secret."""
        self._secrets.pop(str(ref), None)


__all__ = [
    "DEFAULT_SERVICE_NAME",
    "SecretStoreUnavailable",
    "SecretStore",
    "KeyringSecretStore",
    "InMemorySecretStore",
]
