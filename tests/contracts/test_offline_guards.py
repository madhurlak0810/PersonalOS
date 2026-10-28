"""The suite-wide guards in `tests/conftest.py` do what they claim."""

import random
import socket

import pytest

from tests.conftest import TEST_SEED, NetworkAccessBlocked


def test_a_connection_off_this_machine_is_refused():
    # 203.0.113.0/24 is TEST-NET-3: documentation-only, never routed.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessBlocked):
            sock.connect(("203.0.113.7", 443))


def test_loopback_is_still_reachable():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(server.getsockname())


def test_each_test_starts_from_the_same_seed():
    expected = random.Random(TEST_SEED).random()

    assert random.random() == expected
