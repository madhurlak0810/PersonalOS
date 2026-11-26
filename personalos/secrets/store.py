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

`FileSecretStore` is the container store. A container has no OS keychain, so
there the same references are files: Docker secrets (or any read-only mount)
for what an operator provides, and a `0700` directory on a volume for what
PersonalOS writes itself, such as a refresh token from an OAuth connect. The
database still holds only `cred://` references; the file is named after the
reference and holds the value. The local CLI keeps using the keychain.
"""

import logging
import os
import stat
import tempfile
from collections.abc import Sequence
from pathlib import Path
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


def secret_file_name(ref: CredentialRef) -> str:
    """The file a reference is kept in: `<provider>.<name>`.

    Unambiguous because a provider cannot contain a dot, and safe as a single
    path component because neither part can contain a separator or start with
    a dot (`CredentialRef` validates both). Checked again here anyway: this is
    the one place a reference becomes a path.
    """
    name = f"{ref.provider}.{ref.name}"
    if "/" in name or "\\" in name or name.startswith(".") or "\0" in name:
        raise SecretStoreUnavailable(f"credential reference {ref} is not a safe file name")
    return name


class FileSecretStore:
    """A `SecretStore` over files, for containers that have no OS keychain.

    `directory` is where secrets are written: created `0700`, each file
    `0600`, replaced atomically. A file there that group or others can read is
    refused rather than used, as `FileApiTokenStore` does. `read_only_dirs`
    are searched after it, for secrets an operator mounts -- Docker secrets in
    `/run/secrets` -- whose permissions Docker, not this process, controls;
    they are never written or deleted.

    A secret in `directory` takes precedence, so a token PersonalOS rotated
    wins over the one it was first given.
    """

    def __init__(self, directory: Path | str, *, read_only_dirs: Sequence[Path | str] = ()):
        """Bind to a writable directory and any read-only ones; nothing is touched yet."""
        self.directory = Path(directory).expanduser()
        self.read_only_dirs = tuple(Path(d).expanduser() for d in read_only_dirs)

    def _read(self, path: Path, *, check_mode: bool) -> SecretValue | None:
        try:
            info = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not stat.S_ISREG(info.st_mode):
            raise SecretStoreUnavailable(f"secret {path} is not a regular file")
        if check_mode and os.name == "posix" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise SecretStoreUnavailable(
                f"secret file {path} is accessible to other users "
                f"(mode {stat.S_IMODE(info.st_mode):o}); run `chmod 600 {path}`"
            )
        value = path.read_text(encoding="utf-8").strip()
        return SecretValue(value) if value else None

    def get(self, ref: CredentialRef) -> SecretValue | None:
        """Read a secret: the writable directory first, then each read-only one."""
        name = secret_file_name(ref)
        secret = self._read(self.directory / name, check_mode=True)
        if secret is not None:
            return secret
        for directory in self.read_only_dirs:
            secret = self._read(directory / name, check_mode=False)
            if secret is not None:
                return secret
        return None

    def put(self, ref: CredentialRef, secret: SecretValue) -> None:
        """Write a secret into the writable directory, `0600`, atomically."""
        path = self.directory / secret_file_name(ref)
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self.directory, prefix=".secret.")
        except OSError as error:
            raise SecretStoreUnavailable(
                f"secret directory {self.directory} is not writable ({type(error).__name__})"
            ) from None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(secret.reveal() + "\n")
            os.chmod(temp, 0o600)
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
        logger.info("stored credential %s", ref)

    def delete(self, ref: CredentialRef) -> None:
        """Remove a secret from the writable directory; a no-op if absent.

        A secret only a read-only directory holds is left alone (the operator
        mounted it, and only the operator can remove it), with a warning.
        """
        name = secret_file_name(ref)
        try:
            (self.directory / name).unlink()
        except FileNotFoundError:
            pass
        else:
            logger.info("deleted credential %s", ref)
        for directory in self.read_only_dirs:
            if (directory / name).exists():
                logger.warning(
                    "credential %s is still provided by a read-only mount in %s; "
                    "remove the Docker secret to delete it",
                    ref,
                    directory,
                )


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
    "FileSecretStore",
    "InMemorySecretStore",
    "secret_file_name",
]
