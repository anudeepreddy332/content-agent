"""
api/server.py — FastAPI front end for the content-agent pipeline with durable,
asynchronous human-in-the-loop review.

Design (freeze-scoped, deliberately simple):
  * Graph compiled ONCE with a SqliteSaver checkpointer -> interrupt/resume is
    durable across process restarts (checkpoints in outputs/checkpoints.sqlite).
  * Compute runs on a single-worker executor. interrupt() makes invoke() RETURN
    at the HITL gate, so the worker is free during human-wait; runs only queue
    during actual computation. max_workers=1 also means no concurrent SQLite
    access, so check_same_thread=False is safe here.
  * HITL is MANDATORY on every API run (HITL_AUTO_APPROVE forced "0"). There is
    no auto-approve path through the API — that is B9 territory, deferred.
  * GIT_PUSH_ENABLED is NOT set here; git_node stays in dry-run as configured.

Run state lives in two places:
  * Durable graph state -> SqliteSaver (survives restart).
  * In-memory REGISTRY (status, interrupt payload, result summary) -> volatile.
    On restart, in-flight runs are recoverable from the checkpointer but the
    registry is not rebuilt automatically. Documented freeze limitation.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import hmac
import json
import queue
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

# HITL mode + mandatory review must be set BEFORE the graph/nodes decide behavior.
os.environ["HITL_MODE"] = "api"
os.environ["HITL_AUTO_APPROVE"] = "0"

import sqlite3
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph
from main import _build_initial_state, _make_slug, _write_telemetry
from observability.logger import get_logger

log = get_logger("api")

_CKPT_PATH = "outputs/checkpoints.sqlite"
os.makedirs("outputs", exist_ok=True)
_conn = sqlite3.connect(_CKPT_PATH, check_same_thread=False)
CHECKPOINTER = SqliteSaver(_conn)
GRAPH = build_graph(checkpointer=CHECKPOINTER)

EXECUTOR = ThreadPoolExecutor(max_workers=1)
REGISTRY: dict[str, dict] = {}
LOCK = threading.Lock()

# Demo SPA + Server-Sent-Events live progress (additive; the poll API above is untouched).
_STATIC_DIR = Path(__file__).parent.parent / "static"
_SENTINEL = object()  # marks the end of one streamed compute segment in a run's event queue

def _extract_interrupt(result, config):
    """Return the interrupt payload if the graph is paused, else None.
    Defensive across langgraph versions: checks the returned __interrupt__ first,
    then the persisted snapshot's pending tasks."""
    if isinstance(result, dict):
        ints = result.get("__interrupt__")
        if ints:
            return ints[0].value
    snap = GRAPH.get_state(config)
    for task in getattr(snap, "tasks", ()):
        for it in getattr(task, "interrupts", ()):
            return it.value
    return None

def _advance(run_id: str, invoke_input, finalizing: bool):
    """Run one compute segment to the next interrupt or to termination.
    Used for both the initial run (invoke_input=initial_state) and resumes
    (invoke_input=Command(resume=...))."""
    config = {"configurable": {"thread_id": run_id}}
    with LOCK:
        REGISTRY[run_id]["status"] = "finalizing" if finalizing else "running"
    try:
        result = GRAPH.invoke(invoke_input, config)
    except Exception as e:  # mirror main.py crash handler
        log.error("api.run_crash", run_id=run_id, error=str(e))
        crash = dict(REGISTRY[run_id]["initial_state"])
        crash["error_log"] = [f"pipeline crash: {e}"]
        _write_telemetry(crash)
        with LOCK:
            REGISTRY[run_id].update(status="error", error=str(e))
        return

    payload = _extract_interrupt(result, config)
    with LOCK:
        if payload is not None:
            REGISTRY[run_id].update(status="awaiting_review", interrupt_payload=payload)
            return
        # Terminal. Reject path never runs git; approve path does.
        REGISTRY[run_id]["result"] = result
        rejected = (result.get("hitl_status") == "rejected"
                    or result.get("html_review_status") == "rejected")
        REGISTRY[run_id]["status"] = "rejected" if rejected else "complete"
    _write_telemetry(result)


def _submit(run_id, invoke_input, finalizing):
    """Submit to the worker, or run inline when API_SYNC=1 (tests only)."""
    if os.environ.get("API_SYNC") == "1":
        _advance(run_id, invoke_input, finalizing)
    else:
        EXECUTOR.submit(_advance, run_id, invoke_input, finalizing)


# ---- streaming variant (drives the demo SPA) -------------------------------
# Same single-worker EXECUTOR + checkpointer as _advance, so the SQLite
# single-thread invariant holds. The only difference: GRAPH.stream(...) instead
# of GRAPH.invoke(...), pushing one event per completed node into the run's
# event queue. An SSE endpoint drains that queue. REGISTRY status transitions
# are identical to _advance, so GET /runs/{id} stays accurate for streamed runs.

def _node_headline(node: str, delta: dict) -> dict:
    """Project a node's state delta to a small, display-safe headline."""
    d = {}
    if isinstance(delta.get("web_sources"), list):
        d["web_sources"] = len(delta["web_sources"])
    if isinstance(delta.get("kb_results"), list):
        d["kb_results"] = len(delta["kb_results"])
    if isinstance(delta.get("grounding_report"), list):
        d["claims"] = len(delta["grounding_report"])
    if delta.get("grounding_score") is not None:
        d["grounding_score"] = round(delta["grounding_score"], 3)
    if delta.get("reflection_score") is not None:
        d["reflection_score"] = delta["reflection_score"]
    if delta.get("iterations") is not None:
        d["iterations"] = delta["iterations"]
    if delta.get("html_filename"):
        d["html_filename"] = delta["html_filename"]
    if delta.get("git_status"):
        d["git_status"] = delta["git_status"]
    lat = delta.get("latency_ms")
    if isinstance(lat, dict) and isinstance(lat.get(node), (int, float)):
        d["latency_ms"] = lat[node]
    return d


def _advance_streaming(run_id, invoke_input, finalizing):
    """Run one compute segment with GRAPH.stream, emitting per-node events."""
    config = {"configurable": {"thread_id": run_id}}
    q = REGISTRY[run_id]["events"]
    with LOCK:
        REGISTRY[run_id]["status"] = "finalizing" if finalizing else "running"
    result: dict = {}
    try:
        for mode, chunk in GRAPH.stream(invoke_input, config,
                                        stream_mode=["updates", "values"]):
            if mode == "values":
                result = chunk
                continue
            if not isinstance(chunk, dict) or "__interrupt__" in chunk:
                continue
            for node_name, delta in chunk.items():
                q.put({"event": "node", "node": node_name,
                       "headline": _node_headline(node_name, delta or {})})
    except Exception as e:  # mirror _advance / main.py crash handler
        log.error("api.stream_crash", run_id=run_id, error=str(e))
        crash = dict(REGISTRY[run_id]["initial_state"])
        crash["error_log"] = [f"pipeline crash: {e}"]
        _write_telemetry(crash)
        with LOCK:
            REGISTRY[run_id].update(status="error", error=str(e))
        q.put({"event": "error", "error": str(e)})
        q.put(_SENTINEL)
        return

    if not result:  # defensive: recover final state from the checkpoint
        result = getattr(GRAPH.get_state(config), "values", {}) or {}
    payload = _extract_interrupt(result, config)
    with LOCK:
        if payload is not None:
            REGISTRY[run_id].update(status="awaiting_review", interrupt_payload=payload)
        else:
            REGISTRY[run_id]["result"] = result
            rejected = (result.get("hitl_status") == "rejected"
                        or result.get("html_review_status") == "rejected")
            REGISTRY[run_id]["status"] = "rejected" if rejected else "complete"
    if payload is not None:
        q.put({"event": "gate", "review": payload})
    else:
        _write_telemetry(result)
        q.put({"event": "done", "status": REGISTRY[run_id]["status"], "summary": {
            "grounding_score": result.get("grounding_score"),
            "total_cost_usd": result.get("total_cost_usd"),
            "git_status": result.get("git_status"),
            "html_filename": result.get("html_filename"),
            "slug": result.get("slug"),
        }})
    q.put(_SENTINEL)


def _submit_stream(run_id, invoke_input, finalizing):
    if os.environ.get("API_SYNC") == "1":
        _advance_streaming(run_id, invoke_input, finalizing)
    else:
        EXECUTOR.submit(_advance_streaming, run_id, invoke_input, finalizing)


# ---- auth (fail closed) ----
def require_auth(authorization: str = Header(default="")):
    expected = os.environ.get("API_BEARER_TOKEN")
    if not expected:
        raise HTTPException(503, "API_BEARER_TOKEN not configured on the server")
    presented = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(401, "invalid or missing bearer token")

# ---- request models ----
class CreateRun(BaseModel):
    topic: str = Field(min_length=1)
    card_id: str = "standalone"
    series: str = "Learning Log"

class Feedback(BaseModel):
    feedback: str = Field(min_length=1)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pay the one-time encoder load + BM25 build at startup (M5 warmup), so the
    # first run's retrieve_kb telemetry measures steady-state, same as the CLI.
    from tools.query_kb import warmup as kb_warmup
    log.info("api.warmup", **kb_warmup())
    yield
    EXECUTOR.shutdown(wait=False)
    _conn.close()



app = FastAPI(title="content-agent HITL API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/runs", status_code=202, dependencies=[Depends(require_auth)])
def create_run(body: CreateRun):
    import uuid
    slug = _make_slug(body.topic)
    if not slug:
        raise HTTPException(422, f"topic {body.topic!r} produces an empty slug")
    run_id = str(uuid.uuid4())
    initial_state = _build_initial_state(body.topic, slug, body.card_id, body.series, run_id)
    with LOCK:
        REGISTRY[run_id] = {"status": "queued", "initial_state": initial_state,
                            "interrupt_payload": None, "result": None, "error": None}
    _submit(run_id, initial_state, finalizing=False)
    return {"run_id": run_id, "status": "queued"}


@app.get("/runs/{run_id}", dependencies=[Depends(require_auth)])
def get_run(run_id: str):
    with LOCK:
        reg = REGISTRY.get(run_id)
        if not reg:
            raise HTTPException(404, "unknown run_id")
        out = {"run_id": run_id, "status": reg["status"], "error": reg["error"]}
        if reg["status"] == "awaiting_review":
            out["review"] = reg["interrupt_payload"]
        if reg["result"] is not None:
            r = reg["result"]
            out["summary"] = {
                "grounding_score": r.get("grounding_score"),
                "total_cost_usd": r.get("total_cost_usd"),
                "hitl_status": r.get("hitl_status"),
                "html_review_status": r.get("html_review_status"),
                "git_status": r.get("git_status"),
                "html_filename": r.get("html_filename"),
            }
        return out


def _resume(run_id: str, resume_value: dict):
    with LOCK:
        reg = REGISTRY.get(run_id)
        if not reg:
            raise HTTPException(404, "unknown run_id")
        if reg["status"] != "awaiting_review":
            raise HTTPException(409, f"run is '{reg['status']}', not awaiting_review")
    _submit(run_id, Command(resume=resume_value), finalizing=True)
    return {"run_id": run_id, "status": "resuming"}


@app.post("/runs/{run_id}/approve", dependencies=[Depends(require_auth)])
def approve(run_id: str):
    return _resume(run_id, {"action": "approve"})


@app.post("/runs/{run_id}/reject", dependencies=[Depends(require_auth)])
def reject(run_id: str):
    return _resume(run_id, {"action": "reject"})

@app.post("/runs/{run_id}/feedback", dependencies=[Depends(require_auth)])
def feedback(run_id: str, body: Feedback):
    return _resume(run_id, {"action": "feedback", "feedback": body.feedback})


# ---- demo SPA: streaming UI surface (additive) -----------------------------
# Mirrors the poll API but drives GRAPH.stream so the SPA can show each node as
# it completes and both HITL gates inline. Compute still runs on the one EXECUTOR.

@app.post("/ui/runs", status_code=202, dependencies=[Depends(require_auth)])
def ui_create_run(body: CreateRun):
    import uuid
    slug = _make_slug(body.topic)
    if not slug:
        raise HTTPException(422, f"topic {body.topic!r} produces an empty slug")
    run_id = str(uuid.uuid4())
    initial_state = _build_initial_state(body.topic, slug, body.card_id, body.series, run_id)
    with LOCK:
        REGISTRY[run_id] = {"status": "queued", "initial_state": initial_state,
                            "interrupt_payload": None, "result": None, "error": None,
                            "events": queue.Queue()}
    _submit_stream(run_id, initial_state, finalizing=False)
    return {"run_id": run_id, "status": "queued", "slug": slug}


def _resume_stream(run_id: str, resume_value: dict):
    with LOCK:
        reg = REGISTRY.get(run_id)
        if not reg:
            raise HTTPException(404, "unknown run_id")
        if reg["status"] != "awaiting_review":
            raise HTTPException(409, f"run is '{reg['status']}', not awaiting_review")
    _submit_stream(run_id, Command(resume=resume_value), finalizing=True)
    return {"run_id": run_id, "status": "resuming"}


@app.post("/ui/runs/{run_id}/approve", dependencies=[Depends(require_auth)])
def ui_approve(run_id: str):
    return _resume_stream(run_id, {"action": "approve"})


@app.post("/ui/runs/{run_id}/reject", dependencies=[Depends(require_auth)])
def ui_reject(run_id: str):
    return _resume_stream(run_id, {"action": "reject"})


@app.post("/ui/runs/{run_id}/feedback", dependencies=[Depends(require_auth)])
def ui_feedback(run_id: str, body: Feedback):
    return _resume_stream(run_id, {"action": "feedback", "feedback": body.feedback})


@app.get("/ui/runs/{run_id}/events")
async def ui_events(run_id: str, token: str = ""):
    # EventSource cannot set headers, so the bearer token arrives as a query
    # param. Same fail-closed compare_digest check as require_auth. (Query-param
    # tokens can land in access logs; acceptable for a single-user demo.)
    expected = os.environ.get("API_BEARER_TOKEN")
    if not expected:
        raise HTTPException(503, "API_BEARER_TOKEN not configured on the server")
    if not hmac.compare_digest(token, expected):
        raise HTTPException(401, "invalid or missing token")
    with LOCK:
        reg = REGISTRY.get(run_id)
    if not reg:
        raise HTTPException(404, "unknown run_id")
    q = reg["events"]

    async def gen():
        idle = 0
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                idle += 1
                if idle % 200 == 0:        # ~ every 20s, keep proxies from timing out
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.1)
                continue
            idle = 0
            if item is _SENTINEL:
                yield 'data: {"event": "segment_end"}\n\n'
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", include_in_schema=False)
def demo_index():
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "demo UI not installed (static/index.html missing)")
    return FileResponse(index)

