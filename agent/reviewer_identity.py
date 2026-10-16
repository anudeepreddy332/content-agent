"""Single-reviewer pilot credentials and request-scoped graph approval identity.

Only the configured token is accepted; checkpoint state receives the stable
server-configured reviewer ID, never a token or client-supplied identity.
"""
from __future__ import annotations

import hmac
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal


class ReviewerConfigurationError(ValueError):
    """The server has no usable reviewer credential mapping."""


class InvalidReviewerCredential(ValueError):
    """The presented reviewer credential is absent or incorrect."""


_REVIEWER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,63}\Z")


@dataclass(frozen=True)
class ReviewerPrincipal:
    reviewer_id: str
    run_id: str
    origin: Literal["authenticated", "synthetic"]


_CURRENT: ContextVar[ReviewerPrincipal | None] = ContextVar("gate_reviewer", default=None)


def _configured_reviewer() -> tuple[str, str]:
    reviewer_id = os.environ.get("HITL_REVIEWER_ID", "")
    secret = os.environ.get("HITL_REVIEWER_TOKEN", "")
    general_token = os.environ.get("API_BEARER_TOKEN", "")
    if (
        not _REVIEWER_ID.fullmatch(reviewer_id)
        or reviewer_id.startswith("test:")
        or len(secret) < 32
        or not general_token
        or hmac.compare_digest(secret, general_token)
    ):
        raise ReviewerConfigurationError("distinct reviewer credentials are not configured")
    return reviewer_id, secret


def authenticate_reviewer(presented: str) -> str:
    """Map a valid distinct secret to the server-owned stable identity."""
    reviewer_id, secret = _configured_reviewer()
    if not isinstance(presented, str) or not hmac.compare_digest(presented, secret):
        raise InvalidReviewerCredential("invalid or missing reviewer credential")
    return reviewer_id


@contextmanager
def reviewer_context(
    reviewer_id: str | None, run_id: str,
    *, origin: Literal["authenticated", "synthetic"] = "authenticated",
) -> Iterator[None]:
    """Bind an already authenticated principal only for this graph invocation."""
    principal = ReviewerPrincipal(reviewer_id, run_id, origin) if reviewer_id else None
    reset = _CURRENT.set(principal)
    try:
        yield
    finally:
        _CURRENT.reset(reset)


def current_reviewer_id(run_id: str) -> str | None:
    principal = _CURRENT.get()
    if principal is None or principal.run_id != run_id:
        return None
    return principal.reviewer_id


def current_reviewer_origin(run_id: str) -> str | None:
    principal = _CURRENT.get()
    if principal is None or principal.run_id != run_id:
        return None
    return principal.origin


def synthetic_mode_isolated() -> bool:
    return (
        os.environ.get("HITL_TEST_MODE") == "1"
        and os.environ.get("GIT_PUSH_ENABLED", "false").lower() != "true"
        and os.environ.get("PUBLISH_TARGET", "") in ("", "disabled")
    )


def approval_origin_valid(state: dict, gate: int, *, publishing: bool = False) -> bool:
    """Checkpointed origin is authoritative; an ID alone never authorizes a gate."""
    reviewer_id = state.get(f"gate{gate}_reviewer_id")
    origin = state.get(f"gate{gate}_review_origin")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        return False
    if origin == "authenticated":
        return not reviewer_id.startswith("test:")
    if origin == "synthetic" and not publishing:
        return reviewer_id.startswith("test:") and synthetic_mode_isolated()
    return False


def test_auto_reviewer_id() -> str | None:
    """Explicit synthetic identity; publication is forcibly unavailable."""
    if not synthetic_mode_isolated():
        return None
    reviewer_id = os.environ.get("HITL_TEST_REVIEWER_ID", "")
    if not reviewer_id.startswith("test:") or not _REVIEWER_ID.fullmatch(reviewer_id):
        return None
    return reviewer_id
