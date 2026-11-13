"""Port availability helpers for launching the AirTouch dashboard cleanly."""

from __future__ import annotations

import errno
import socket

_BUSY_ERRNOS = {
    errno.EADDRINUSE,
    errno.EACCES,
    getattr(errno, "WSAEADDRINUSE", 10048),
    getattr(errno, "WSAEACCES", 10013),
}


def _loopback_targets() -> list[tuple[int, str]]:
    targets = [(socket.AF_INET, "127.0.0.1")]
    if getattr(socket, "has_ipv6", False):
        targets.append((socket.AF_INET6, "::1"))
    return targets


def _is_bind_blocked(family: int, host: str, port: int) -> bool:
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
            return False
    except OSError as exc:
        return exc.errno in _BUSY_ERRNOS or getattr(exc, "winerror", None) in (10048, 10013)


def _is_accepting(family: int, host: str, port: int, timeout: float) -> bool:
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            if sock.getsockname()[1] == port:
                # Hold the colliding ephemeral port open on sock so alt gets a
                # different source port and cannot TCP self-connect to target.
                with socket.socket(family, socket.SOCK_STREAM) as alt:
                    alt.settimeout(timeout)
                    return alt.connect_ex((host, port)) == 0
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def is_port_in_use(port: int, timeout: float = 0.2) -> bool:
    """Return True if a TCP server is bound or accepting connections on loopback."""
    # Check bind first so bound sockets without accept() are detected cleanly,
    # then probe connect_ex to catch wildcard/SO_REUSEADDR servers (e.g. Uvicorn).
    targets = _loopback_targets()
    return any(_is_bind_blocked(fam, host, port) for fam, host in targets) or any(
        _is_accepting(fam, host, port, timeout) for fam, host in targets
    )


def find_free_port(start_port: int = 8501, max_tries: int = 20) -> int:
    """Return the first unused TCP port starting at ``start_port``."""
    for port in range(start_port, start_port + max_tries):
        if not is_port_in_use(port):
            return port
    return start_port


if __name__ == "__main__":
    print(find_free_port())
