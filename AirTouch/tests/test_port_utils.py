"""Tests for loopback port discovery used by AirTouch startup scripts."""

import socket

from AirTouch.core.port_utils import find_free_port, is_port_in_use


def test_find_free_port_skips_bound_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        busy_port = server.getsockname()[1]
        assert is_port_in_use(busy_port) is True
        chosen = find_free_port(start_port=busy_port, max_tries=10)
        assert chosen > busy_port
        assert is_port_in_use(chosen) is False
    finally:
        server.close()

