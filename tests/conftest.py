"""Suite-wide guards that keep every test offline and deterministic.

- **No network beyond this machine.** Any socket connection to a
  non-loopback address fails the test that made it. Unit tests talk to fakes
  and recorded model outputs (`tests/fixtures/recorded_model_outputs/`); the
  only real connections the suite makes are to local Postgres and to the API
  and worker processes some tests start on localhost.
- **A fixed seed.** `random` is reseeded before each test, so anything that
  draws from it without being handed its own generator -- retry jitter, say --
  draws the same values on every run and in any test order.
"""

import ipaddress
import random
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

#: The seed every test starts from.
TEST_SEED = 0


class NetworkAccessBlocked(RuntimeError):
    """A test tried to reach a host other than this machine."""


def _is_local(address) -> bool:
    if not isinstance(address, tuple):  # AF_UNIX paths and the like
        return True
    host = address[0]
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname: resolve it without opening a connection.
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        return all(ipaddress.ip_address(info[4][0]).is_loopback for info in infos)


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _guarded_connect(self, address):
    if not _is_local(address):
        raise NetworkAccessBlocked(f"tests may not open network connections (to {address!r})")
    return _connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_local(address):
        raise NetworkAccessBlocked(f"tests may not open network connections (to {address!r})")
    return _connect_ex(self, address)


@pytest.fixture(autouse=True, scope="session")
def _block_network():
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = _connect
        socket.socket.connect_ex = _connect_ex


@pytest.fixture(autouse=True)
def _fixed_seed():
    random.seed(TEST_SEED)


@pytest.fixture(autouse=True)
def _isolated_api_token(monkeypatch):
    """Point the local API token at a per-test file, never the user's real one.

    An app whose authentication is built from settings (`build_api_auth`)
    mints the token on first use; without this, that would be
    `~/.config/personalos/api-token`.

    A plain `tempfile` directory rather than `tmp_path_factory`: using that in
    every test makes every pytest run create a numbered base temp dir, which
    shifts the `pytest-N` in every other test's `tmp_path`.
    """
    from personalos.config import settings

    directory = Path(tempfile.mkdtemp(prefix="personalos-api-token-"))
    path = directory / "api-token"
    monkeypatch.setattr(settings, "api_token_file", str(path))
    monkeypatch.setattr(settings, "api_token_store", "file")
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setenv("API_TOKEN_FILE", str(path))
    monkeypatch.setenv("API_TOKEN_STORE", "file")
    try:
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)
