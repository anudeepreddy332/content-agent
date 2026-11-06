"""Durable, metadata-only reservation for production Call B requests.

The journal is deliberately separate from LangGraph's verify-node checkpoint.
An existing reservation is never dispatched again, even if no response was saved.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class ReservationExists(RuntimeError):
    """The logical request was already reserved, possibly by another worker."""


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _path() -> Path:
    return Path(os.environ.get("CALL_B_JOURNAL_PATH", "outputs/call_b_requests.sqlite"))


def _connect() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS call_b_requests (
        run_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('initial', 'retry')),
        verify_key TEXT NOT NULL,
        request_id TEXT NOT NULL,
        draft_sha256 TEXT NOT NULL,
        roster_sha256 TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL,
        reserved_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('reserved', 'responded')),
        response_id TEXT,
        returned_model TEXT,
        finish_reason TEXT,
        usage_json TEXT,
        usage_status TEXT NOT NULL CHECK(usage_status IN ('unknown', 'known')),
        total_tokens INTEGER,
        total_tokens_source TEXT CHECK(total_tokens_source IN ('provider', 'component_sum')),
        estimated_cost_usd REAL,
        call_a_tokens INTEGER,
        call_a_cost_usd REAL,
        PRIMARY KEY (run_id, kind, verify_key)
    )""")
    connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_call_b_retry_per_run
        ON call_b_requests(run_id, kind) WHERE kind = 'retry'""")
    return connection


def lookup(run_id: str, kind: str, verify_key: str | None = None) -> dict | None:
    with closing(_connect()) as connection:
        if kind == "retry":
            row = connection.execute(
                "SELECT * FROM call_b_requests WHERE run_id=? AND kind='retry'",
                (run_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM call_b_requests WHERE run_id=? AND kind='initial' AND verify_key=?",
                (run_id, verify_key),
            ).fetchone()
        return dict(row) if row is not None else None


def reserve(*, run_id: str, kind: str, verify_key: str, request_id: str,
            draft_sha256: str, roster_sha256: str, manifest_sha256: str,
            call_a_tokens: int | None = None, call_a_cost_usd: float | None = None) -> None:
    """Commit the at-most-once dispatch slot before the outbound request."""
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("""INSERT INTO call_b_requests (
                run_id,kind,verify_key,request_id,draft_sha256,roster_sha256,
                manifest_sha256,reserved_at,state,usage_status,call_a_tokens,call_a_cost_usd
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
                run_id, kind, verify_key, request_id, draft_sha256, roster_sha256,
                manifest_sha256, datetime.now(timezone.utc).isoformat(),
                "reserved", "unknown", call_a_tokens, call_a_cost_usd,
            ))
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ReservationExists("Call B request already reserved") from exc


def record_response(*, run_id: str, kind: str, verify_key: str,
                    response_id: str | None, returned_model: str | None,
                    finish_reason: str | None, usage: dict | None,
                    total_tokens: int | None, estimated_cost_usd: float | None) -> None:
    """Save provider metadata immediately after transport, before response validation."""
    usage_known = total_tokens is not None and estimated_cost_usd is not None
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute("""UPDATE call_b_requests SET
            state='responded',response_id=?,returned_model=?,finish_reason=?,
            usage_json=?,usage_status=?,total_tokens=?,total_tokens_source=?,estimated_cost_usd=?
            WHERE run_id=? AND kind=? AND verify_key=? AND state='reserved'""", (
                response_id, returned_model, finish_reason,
                json.dumps(usage, sort_keys=True) if usage is not None else None,
                "known" if usage_known else "unknown", total_tokens,
                ("provider" if usage is not None and usage.get("total_tokens") is not None
                 else "component_sum" if usage_known else None),
                estimated_cost_usd, run_id, kind, verify_key,
            ))
        if cursor.rowcount != 1:
            connection.rollback()
            raise RuntimeError("Call B response has no unique active reservation")
        connection.commit()
