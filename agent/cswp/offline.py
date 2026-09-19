"""Offline guard for deterministic CSWP compilation and evaluation."""

from __future__ import annotations

import os
import sys

from agent.cswp.errors import NetworkForbidden


def install_offline_guard():
    """Install before optional imports; forbids even loopback I/O."""
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ[name] = "false"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def audit(event, args):
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.sendto",
            "socket.sendmsg",
        }:
            raise NetworkForbidden(f"OFFLINE_NETWORK_FORBIDDEN: {event}")

    sys.addaudithook(audit)
