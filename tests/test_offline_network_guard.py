"""Prove offline pytest collection fails closed on external network attempts."""
from __future__ import annotations

import socket

import pytest

from tests.offline_network_guard import OfflineNetworkError, is_loopback_address


def test_loopback_addresses_are_allowed():
    assert is_loopback_address(("127.0.0.1", 80))
    assert is_loopback_address(("localhost", 8080))
    assert is_loopback_address(("::1", 443))
    assert not is_loopback_address(("8.8.8.8", 53))


def test_external_socket_connect_fails_closed():
    sock = socket.socket()
    with pytest.raises(OfflineNetworkError, match="Offline regression tests"):
        sock.connect(("1.2.3.4", 443))


def test_external_create_connection_fails_closed():
    with pytest.raises(OfflineNetworkError, match="Offline regression tests"):
        socket.create_connection(("example.com", 443), timeout=0.01)


def test_loopback_connect_is_permitted():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    probe.listen(1)
    host, port = probe.getsockname()
    client = socket.socket()
    client.connect((host, port))
    conn, _ = probe.accept()
    client.close()
    conn.close()
    probe.close()
