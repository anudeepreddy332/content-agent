"""Production CSWP error types."""

from __future__ import annotations


class CSWPError(RuntimeError):
    """A CSWP packing, provenance, or contract invariant failed."""


class StructuralError(RuntimeError):
    """Fail closed on an invalid source, identity, or structural representation."""


class NetworkForbidden(BaseException):
    """Cannot be swallowed by a provider's ordinary exception handler."""
