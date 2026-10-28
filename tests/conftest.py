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
import socket

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
