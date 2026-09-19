"""Deterministic identity and hashing helpers for CSWP artifacts."""

from __future__ import annotations

import hashlib
import json


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def experiment_canonical_json(value):
    """Match scripts.phase5a2_shadow_ab canonical JSON for frozen CSWP fingerprints."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(
        data if isinstance(data, bytes) else data.encode("utf-8")
    ).hexdigest()


def sha256_json(value):
    return sha(experiment_canonical_json(value))


def identity(kind, payload):
    return f"ca:{kind}:{sha(canonical_json(payload))}"
