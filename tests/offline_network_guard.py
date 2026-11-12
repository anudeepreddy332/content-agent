"""Deterministic offline guard for pytest — fail closed on external network."""
from __future__ import annotations

import socket
from typing import Any

OFFLINE_NETWORK_ERROR = (
    "Offline regression tests must not call external network services. "
    "Mark the test @pytest.mark.live_network to opt in."
)


class OfflineNetworkError(OSError):
    """Raised when an offline test attempts external network access."""


def _host_from_address(address: Any) -> str | None:
    if isinstance(address, tuple) and address:
        host = address[0]
        return str(host) if host is not None else None
    return None


def is_loopback_address(address: Any) -> bool:
    host = _host_from_address(address)
    if host is None:
        return False
    normalized = host.strip("[]").lower()
    if normalized in {"", "localhost", "::1", "0.0.0.0"}:
        return True
    if normalized.startswith("127."):
        return True
    return False


def install_offline_network_guard(monkeypatch: Any) -> None:
    """Block external socket connects; allow loopback for local harness servers."""
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class GuardedSocket(original_socket):  # type: ignore[misc,valid-type]
        def connect(self, address: Any) -> None:  # type: ignore[override]
            if not is_loopback_address(address):
                raise OfflineNetworkError(
                    f"{OFFLINE_NETWORK_ERROR} attempted connect to {address!r}"
                )
            return super().connect(address)

        def connect_ex(self, address: Any) -> int:  # type: ignore[override]
            if not is_loopback_address(address):
                raise OfflineNetworkError(
                    f"{OFFLINE_NETWORK_ERROR} attempted connect_ex to {address!r}"
                )
            return super().connect_ex(address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if not is_loopback_address(address):
            raise OfflineNetworkError(
                f"{OFFLINE_NETWORK_ERROR} attempted create_connection to {address!r}"
            )
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


def disable_langsmith_and_provider_tracing(monkeypatch: Any) -> None:
    """Ensure offline tests do not enable LangSmith/LangChain tracing."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
