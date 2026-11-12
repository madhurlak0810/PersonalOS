"""The local API token: generated on first run, kept outside the repository.

The API's MVP authentication is one bearer token per installation. It is
minted the first time anything needs it (the API starting, or `personalos.cli
auth token`), stored either in a `0600` file under the user's config directory
or in the OS keychain, and printed only by `auth token`.

Every token this module loads or mints is registered with the process-wide
redactor, so it is scrubbed from logs, spans and audit payloads by value; it
also has a fixed `pos_at_` shape the redactor matches on its own.
"""

import logging
import os
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from personalos.domain.auth import API_TOKEN_PREFIX
from personalos.domain.credentials import CredentialRef, SecretValue
from personalos.domain.redaction import Redactor, default_redactor

from .store import SecretStore, SecretStoreUnavailable

logger = logging.getLogger(__name__)

#: Where the token is filed when it lives in the OS keychain.
API_TOKEN_REF = CredentialRef(provider="personalos-api", name="local")

#: Random bytes behind a token: 256 bits.
_TOKEN_BYTES = 32


def generate_api_token() -> SecretValue:
    """A fresh token: `pos_at_` and 43 url-safe characters."""
    return SecretValue(API_TOKEN_PREFIX + secrets.token_urlsafe(_TOKEN_BYTES))


def default_token_path() -> Path:
    """`$XDG_CONFIG_HOME/personalos/api-token`, or `~/.config/...` without it."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "personalos" / "api-token"


@runtime_checkable
class ApiTokenStore(Protocol):
    """Where the one local token is kept."""

    def read(self) -> SecretValue | None:
        """The stored token, or `None` if there is none yet."""
        ...

    def write(self, token: SecretValue) -> None:
        """Store (or replace) the token."""
        ...

    def describe(self) -> str:
        """Where the token is, for a human -- never the token."""
        ...


class FileApiTokenStore:
    """The token in a file only its owner can read.

    Written atomically with mode `0600` in a `0700` directory. A file that
    group or others can read is refused rather than used: a token anyone on
    the machine could have copied authenticates nobody.
    """

    def __init__(self, path: Path | str):
        """Bind to a file path; nothing is touched until a read or write."""
        self.path = Path(path).expanduser()

    def read(self) -> SecretValue | None:
        """Read the token, checking the file's permissions first."""
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        if os.name == "posix" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise SecretStoreUnavailable(
                f"API token file {self.path} is accessible to other users "
                f"(mode {stat.S_IMODE(info.st_mode):o}); run `chmod 600 {self.path}` "
                f"or rotate it with `personalos.cli auth token --rotate`"
            )
        value = self.path.read_text(encoding="utf-8").strip()
        return SecretValue(value) if value else None

    def write(self, token: SecretValue) -> None:
        """Replace the file atomically; it is never visible with wider permissions."""
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkstemp creates the file 0600; os.replace makes the swap atomic, so
        # a reader sees the old token or the new one, never a partial write.
        fd, temp = tempfile.mkstemp(dir=directory, prefix=".api-token.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(token.reveal() + "\n")
            os.chmod(temp, 0o600)
            os.replace(temp, self.path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
        logger.info("wrote the API token to %s", self.path)

    def describe(self) -> str:
        """The file path."""
        return str(self.path)


class KeychainApiTokenStore:
    """The token in the OS keychain, through a `SecretStore`."""

    def __init__(self, store: SecretStore, ref: CredentialRef = API_TOKEN_REF):
        """File the token under `ref` in `store`."""
        self._store = store
        self._ref = ref

    def read(self) -> SecretValue | None:
        """Read the token from the keychain."""
        return self._store.get(self._ref)

    def write(self, token: SecretValue) -> None:
        """Write the token to the keychain."""
        self._store.put(self._ref, token)

    def describe(self) -> str:
        """The keychain reference."""
        return f"the OS keychain ({self._ref})"


class LocalApiToken:
    """Loads, mints and rotates the local token, teaching the redactor each one."""

    def __init__(self, store: ApiTokenStore, *, redactor: Redactor | None = None):
        """Manage the token kept in `store`."""
        self.store = store
        self._redactor = redactor or default_redactor()

    def current(self) -> SecretValue | None:
        """The stored token, if there is one. Never mints."""
        token = self.store.read()
        if token is not None:
            self._redactor.register_secret(token.reveal())
        return token

    def load_or_create(self) -> tuple[SecretValue, bool]:
        """The stored token, minting and storing one on first run.

        Returns the token and whether it was just created.
        """
        token = self.current()
        if token is not None:
            return token, False
        return self._mint(), True

    def rotate(self) -> SecretValue:
        """Replace the token. The old one stops verifying once the API reloads it."""
        return self._mint()

    def _mint(self) -> SecretValue:
        token = generate_api_token()
        # Registered before it is written, so not even the write's own log
        # line could carry it.
        self._redactor.register_secret(token.reveal())
        self.store.write(token)
        return token


__all__ = [
    "API_TOKEN_REF",
    "ApiTokenStore",
    "FileApiTokenStore",
    "KeychainApiTokenStore",
    "LocalApiToken",
    "default_token_path",
    "generate_api_token",
]
