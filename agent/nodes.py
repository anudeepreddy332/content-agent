""""
agent/nodes.py
--------------
All node and edge functions for the content-agent pipeline.

Each node signature: (state: AgentState) -> dict
    Nodes return ONLY the keys they update.
    LangGraph merges the return dict into the existing state.
    Never return the full state — only the changes.

"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
load_dotenv()

from agent.state import AgentState, DraftSections
from tools.web_search import web_search
from tools.context_assembly import MAX_EVIDENCE_WINDOW_CHARS, context_budget_stats
from tools.query_kb import assemble_child_context, query_kb
from config import (
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    DRAFT_TEMPERATURE,
    MAX_ITERATIONS,
    REFLECTION_THRESHOLD,
    GROUNDING_FLOOR,
    COST_GATE_USD,
    DEEPSEEK_INPUT_COST_PER_M,
    DEEPSEEK_OUTPUT_COST_PER_M,
    TAVILY_MIN_AVG_SCORE,
    LLM_TIMEOUT_S,
)
from observability.logger import get_logger
from observability.tracing import is_tracing_enabled
import html as html_module
import re
import datetime

VERIFY_SYSTEM = Path("prompts/verify_system.md").read_text()
REFLECT_SYSTEM = Path("prompts/reflect_system.md").read_text()

# DeepSeek client
# Initialized once at module load. api_key read at call time (not cached at import)
# so that test_resilience.py invalid-key tests work correctly (Phase 3 lesson).

def _get_client() -> OpenAI:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_TIMEOUT_S,
        max_retries=0,  # tenacity owns retries; SDK-internal retries would stack (3x2=6 attempts)
    )
    if is_tracing_enabled():
        # wrap_openai patches .chat.completions.create() to emit a traced "llm" run
        # per call with token usage attached — DeepSeek's API is OpenAI-compatible,
        # so the same wrapper works against DEEPSEEK_BASE_URL. No-op import cost
        # paid only when tracing is actually on.
        from langsmith.wrappers import wrap_openai
        return wrap_openai(client)
    return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
    )),
    reraise=True,
)

def _llm_call(client: OpenAI, **kwargs):
    """
    Thin retry wrapper around client.chat.completions.create().

    Retries up to 3 times on transient errors only:
        RateLimitError     — 429, back off and retry
        APIConnectionError — network failure, retry
        APITimeoutError    — request timed out, retry
        InternalServerError — 500, server-side transient, retry

    Does NOT retry:
        AuthenticationError — wrong API key, will never succeed
        BadRequestError     — malformed prompt, will never succeed

    reraise=True: after 3 failures the original exception propagates to
    main.py's crash handler, which writes telemetry and re-raises.

    Wait schedule: 2s → 4s → 8s (capped at 10s). Max total wait: ~14s.
    """
    response = client.chat.completions.create(**kwargs)
    if is_tracing_enabled() and response.usage is not None:
        _attach_usage_to_current_run(response.usage)
    return response


def _attach_usage_to_current_run(usage) -> None:
    """Layer DeepSeek's actual per-token price (DEEPSEEK_INPUT_COST_PER_M /
    DEEPSEEK_OUTPUT_COST_PER_M) onto the current LangSmith run's usage_metadata.
    wrap_openai already attaches token counts from response.usage, but LangSmith's
    built-in price table has no entry for "deepseek-chat", so cost would otherwise
    show as zero — this makes it match the cost this pipeline already tracks
    internally (see _cost())."""
    from langsmith.run_helpers import get_current_run_tree

    run = get_current_run_tree()
    if run is None:
        return
    existing = (run.metadata or {}).get("usage_metadata") or {}
    run.metadata["usage_metadata"] = {
        **existing,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "total_cost": _cost(usage),
    }



def _load_system_prompt() -> str:
    """Load draft system prompt from prompts/draft_system.md."""
    prompt_path = Path("prompts/draft_system.md")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Draft system prompt not found at {prompt_path}")

    # Strip the comment header lines (lines starting with #) before the actual prompt
    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    content_lines = [l for l in lines if not l.startswith("#")]
    return "\n".join(content_lines).strip()


def _extract_json_array(raw: str) -> list:
    """
    Tolerant JSON-array extraction for verifier output.

    Layer 1: strip code fences, json.loads (the original path).
    Layer 2: slice from first '[' to last ']' and parse — recovers outputs
             with preamble/postamble text around a valid array.
    Raises ValueError/JSONDecodeError if no array can be recovered.

    Parsing robustness only — does NOT touch the verify rubric. Any change
    here must re-pass evals/verifier_golden_test.py (>=11/12, >=10/12).
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        parsed = json.loads(s[start:end + 1])
        if isinstance(parsed, list):
            return parsed
    raise ValueError("no JSON array found in verifier output")


def _cost(usage) -> float:
    """Compute DeepSeek API cost from a usage object."""
    return (
        usage.prompt_tokens / 1_000_000 * DEEPSEEK_INPUT_COST_PER_M
        +
        usage.completion_tokens / 1_000_000 * DEEPSEEK_OUTPUT_COST_PER_M
    )

def _assemble_markdown(topic: str, sections: DraftSections) -> str:
    """
    Combine DraftSections into a single markdown string.
    Used by verify_node and reflect_node for full-text analysis.
    """
    return f"""# {topic}
        
        ## Problem Framing
        {sections['problem_framing']}
        
        ## Technical Deep-Dive
        {sections['technical_dive']}
        
        ## Code
        {sections['code_snippets']}
        
        ## Takeaways
        {sections['takeaways']} 
        
        """


# Nodes
# Draft Node

def draft_node(state: AgentState) -> dict:
    """
    Generate the structured article draft using DeepSeek.

    Input state keys used:
        topic, series_context, card_id, hitl_feedback (if revision loop),
        iterations, total_tokens, total_cost_usd, latency_ms

    What happens inside:
        1. Builds user message from topic + series context + optional HITL feedback
        2. Calls DeepSeek with draft_system.md as system prompt
        3. Parses JSON response into DraftSections TypedDict
        4. Assembles draft_markdown from sections
        5. Tracks tokens, cost, latency

    Output keys returned:
        draft_sections, draft_markdown, iterations,
        total_tokens, total_cost_usd, latency_ms

    Failure modes:
        - DeepSeek returns malformed JSON: fallback DraftSections with error message
        - API timeout or rate limit: raises exception (caught in main.py retry loop)
        - Cost gate exceeded: does NOT check here — checked in route_after_reflect
    """
    log = get_logger("draft_node")
    t_start = time.time()
    client = _get_client()
    system_prompt = _load_system_prompt()

    # Build user message
    feedback_block = ""
    if state.get("hitl_feedback"):
        feedback_block = f"\n\nREVISION FEEDBACK FROM HUMAN REVIEWER:\n{state['hitl_feedback']}\nAddress this feedback specifically in the new draft."

    # Grounding-report feedback on revision (M4 PASS, locked 2026-06-11):
    # every revision pass feeds the previous iteration's unverified claims back
    # with ground/generalize/cut instructions. Evidence: M4 (18 runs, frozen
    # cache) — Multi-Agent UVR2 0.230 vs control 0.341 with no SV loss; the
    # blind re-roll control regressed UVR on 2/3 topics. Fires on both the
    # reflect-loop and HITL-feedback revision paths (iterations >= 1 in both).
    # Field name m4_feedback_claims kept deliberately for telemetry continuity.
    m4_feedback_claims = 0
    grounding_feedback_block = ""
    if state.get("iterations", 0) >= 1 and state.get("grounding_report"):
        unverified_claims = [
            (r.get("claim") or "").strip()
            for r in state["grounding_report"]
            if r.get("status") == "unverified" and (r.get("claim") or "").strip()
        ]
        if unverified_claims:
            m4_feedback_claims = len(unverified_claims)
            claims_list = "\n".join(f"- {c}" for c in unverified_claims)
            grounding_feedback_block = (
                "\n\nREVISION — GROUNDING REPORT FEEDBACK:\n"
                "Your previous draft contained the following claims that could NOT "
                "be verified against the provided sources:\n"
                f"{claims_list}\n\n"
                "For EACH claim above, do exactly one of the following in the new draft:\n"
                "1. GROUND IT: keep it only if one of the GROUNDING SOURCES below "
                "actually contains that specific detail.\n"
                "2. GENERALIZE IT: replace it with a general statement that drops the "
                "unsourceable specifics (figures, names, mechanisms).\n"
                "3. CUT IT: remove it entirely if it adds little.\n"
                "Do NOT invent new specific claims to replace them. Keep all other "
                "content — especially specific claims that WERE verifiable — intact."
            )


    source_block = ""
    if state.get("web_sources") or state.get("kb_results"):
        web_sources = state.get("web_sources", []) or []
        kb_results = state.get("kb_results", []) or []
        if web_sources or kb_results:
            source_context = _build_source_context(web_sources[:6], kb_results[:3])
            source_block = (
                "\n\nGROUNDING SOURCES (retrieved for this topic):\n"
                f"{source_context}\n\n"
                "Use these sources to make SPECIFIC, substantive technical claims — mechanisms, "
                "conditions, tradeoffs, failure modes, concrete details — but ONLY where a source "
                "actually contains that specific detail. For well-known background NOT covered by "
                "these sources, state it in general terms and do NOT attach specific figures, names, "
                "or mechanisms you cannot source. When choosing between a specific claim you cannot "
                "source and a general statement you can support, choose the general one. Do NOT "
                "extrapolate or combine sources into new specific claims they do not individually support."
            )

    user_message = f"""Write a technical article for The Machinist on the following topic.

                        Topic: {state['topic']}
                        Card ID: {state['card_id']}
                        Series context: {state['series_context']}

                        This article will be published on themachinist.org under the Learning Log section.
                        The audience is engineers learning ML and agentic AI — they are smart but new to this specific topic.
                        {feedback_block}
                        {grounding_feedback_block}
                        {source_block}

                        Return ONLY the JSON object as specified in your instructions. No markdown wrapper."""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    if grounding_feedback_block:
        log.info("draft.grounding_feedback_injected", run_id=state["run_id"],
                 claims=m4_feedback_claims,
                 block_chars=len(grounding_feedback_block),
                 in_user_message=grounding_feedback_block in user_message)

    response = _llm_call(
        client,
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=DRAFT_TEMPERATURE,
        max_tokens=4000,
    )

    latency = int((time.time() - t_start) * 1000)
    run_cost = _cost(response.usage)

    # Parse JSON response
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model wraps JSON anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        sections: DraftSections = {
            "problem_framing": parsed.get("problem_framing", ""),
            "technical_dive": parsed.get("technical_dive", ""),
            "code_snippets": parsed.get("code_snippets", ""),
            "takeaways": parsed.get("takeaways", ""),
        }

    except json.JSONDecodeError as e:
        # Graceful degradation: don't crash the graph
        # Log the raw output so you can debug what the model returned

        log.error("draft.parse_failed", run_id=state["run_id"], error=str(e), raw_preview=raw[:300])

        sections: DraftSections = {
            "problem_framing": f"[PARSE ERROR] {str(e)}",
            "technical_dive": raw,  # preserve raw content for debugging
            "code_snippets": "",
            "takeaways": "",
        }

    # assemble full markdown for downstream nodes
    draft_markdown = _assemble_markdown(state['topic'], sections)

    # Merge latency into existing latency dict
    existing_latency = state.get("latency_ms", {})
    existing_latency["draft"] = latency


    log.info(
        "draft.complete",
        run_id=state["run_id"],
        tokens=response.usage.total_tokens,
        cost=round(run_cost, 5),
        latency_ms=latency,
        sections_count=len([v for v in sections.values() if v])
    )

    return {
        "draft_sections": sections,
        "draft_markdown": draft_markdown,
        "m4_feedback_claims": m4_feedback_claims,
        "iterations": state.get("iterations", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + response.usage.total_tokens,
        "total_cost_usd": state.get("total_cost_usd", 0) + run_cost,
        "latency_ms": existing_latency,
    }

def retrieve_node(state: AgentState) -> dict:
    """
        Retrieve supporting sources from web (Tavily) and local KB (ChromaDB).

        Input state keys used:
            topic, draft_markdown, latency_ms

        What happens inside:
            1. Generates 3 targeted search queries from the topic
            2. Runs each query through Tavily web search (5 results each, deduped)
            3. Queries ChromaDB KB with topic + key terms
            4. Returns combined results

        Why 3 queries instead of 1:
            A single query on "Linear & Logistic Regression" returns mostly intro articles.
            Three targeted queries (intuition, failure modes, production use) return
            more diverse, deeper sources — better grounding coverage.

        Output keys returned:
            web_sources, kb_results, latency_ms

        Failure modes:
            - Tavily returns 0 results: web_sources = [] — verify_node handles this
            - KB is empty: kb_results = [] — verify_node handles this
            - Both empty: draft proceeds to HITL with a low grounding score — human sees it
            - Tavily rate limit: web_search() returns [] and logs, does not crash
    """
    log = get_logger("retrieve_node")

    t_start = time.time()

    topic = state["topic"]

    # Generate targeted search queries
    # Why hardcoded patterns instead of LLM-generated queries?
    # Speed and cost. LLM query generation adds a full round-trip for marginal gain.
    # These three angles cover 90% of what the verify_node needs.


    queries = [
        f"{topic} explained technical",
        f"{topic} failure modes limitations production",
        f"{topic} implementation Python example",
    ]

    # Web search - deduplicate by URL across all queries
    seen_urls = set()
    web_sources = []

    # Declare before the first loop so Tavily errors can be captured there too
    new_error_log = list(state.get("error_log", []))


    for query in queries:
        try:
            results = web_search(query, max_results=5)
        except Exception as e:
            new_error_log.append(f"[retrieve] Tavily error for query '{query[:50]}': {e}")
            log.warning("retrieve.tavily_error",
                        run_id=state["run_id"], query=query, error=str(e))
            continue

        for r in results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                web_sources.append(r)

    # Source quality gate: force cache bypass if first-pass sources are sparse OR low-scoring.
    # COUNT check (< 3): catches empty/near-empty returns.
    # SCORE check (avg < TAVILY_MIN_AVG_SCORE): catches the stale-cache case where the
    #   cache has entries (bypassing the count check) but they are off-topic or outdated.
    #   Confirmed by benchmark 20260604: Topic 1 retrieve=118ms (cache hit), grounding=0.636;
    #   Topics 2 & 3 retrieve=6000ms+ (live), grounding=0.86+.
    # A second pass with force_refresh=True overwrites the cache entry with fresh results.

    avg_score = (
        sum(r.get("score", 0.0) for r in web_sources) / len(web_sources)
        if web_sources else 0.0
    )
    needs_refresh = len(web_sources) < 3 or avg_score < TAVILY_MIN_AVG_SCORE


    if needs_refresh:
        refresh_reason = (
            f"count={len(web_sources)}" if len(web_sources) < 3
            else f"avg_score={round(avg_score, 3)} < {TAVILY_MIN_AVG_SCORE}"
        )
        log.warning("retrieve.low_quality_sources",
                    run_id=state["run_id"],
                    count=len(web_sources),
                    avg_score=round(avg_score, 3),
                    reason=refresh_reason,
                    action="forcing_refresh")

        web_sources = []
        seen_urls = set()
        for query in queries:
            try:
                results = web_search(query, max_results=5, force_refresh=True)
            except Exception as e:
                new_error_log.append(
                    f"[retrieve] Tavily force_refresh error for query '{query[:50]}': {e}"
                )
                log.warning("retrieve.tavily_refresh_error",
                            run_id=state["run_id"], query=query, error=str(e))
                continue

            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    web_sources.append(r)

        new_error_log.append(
            f"retrieve: low quality sources on first pass ({refresh_reason}), "
            f"forced refresh — post-refresh count={len(web_sources)}"
        )
    # Sort by Tavily relevance score descending, then keep top 10
    web_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
    web_sources = web_sources[:10]

    t_web = int((time.time() - t_start) * 1000)

    # KB query — use topic + first 100 chars of problem_framing for richer context
    problem_framing_preview = state.get("draft_sections", {}).get("problem_framing", "")[:100]
    kb_query = f"{topic} {problem_framing_preview}".strip()
    # Retrieve ten MiniLM-safe children, then form bounded source-local evidence
    # windows. Raw child retrieval stays available through query_kb() for
    # diagnostics and evaluator work; only LLM context uses expanded windows.
    kb_children = query_kb(query=kb_query, n_results=10)
    kb_results = assemble_child_context(kb_children, n_windows=5)
    kb_context_stats = {
        "candidate_children": len(kb_children),
        "candidate_unique_sources": len({child.get("source") for child in kb_children}),
        "draft": context_budget_stats(kb_results, n_windows=3),
        "verifier": context_budget_stats(kb_results, n_windows=5),
    }

    latency = int((time.time() - t_start) * 1000)
    existing_latency = state.get("latency_ms", {})
    existing_latency["retrieve"] = latency
    existing_latency["retrieve_web"] = t_web
    existing_latency["retrieve_kb"] = latency - t_web

    log.info(
        "retrieve.complete",
        run_id=state["run_id"],
        web_sources=len(web_sources),
        kb_children=len(kb_children),
        kb_results=len(kb_results),
        kb_context_stats=kb_context_stats,
        latency_ms=latency
    )


    return {
        "web_sources": web_sources,
        "kb_results": kb_results,
        "kb_context_stats": kb_context_stats,
        "latency_ms": existing_latency,
        "error_log": new_error_log,
    }


def _build_source_context(web_sources: list, kb_results: list) -> str:
    """Format sources into a compact string for the verify prompt.

    Truncation limits — do not change without re-running benchmark:
      WEB_CHARS: Tavily content median is ~1923 chars (305-result sample).
                 1500 chars covers ~78% of a typical result and the full
                 lower quartile. 98% of results were cut at the old 500-char
                 limit, which is why obvious claims were marked unverified.
      KB_CHARS:  Context assembly owns the 2400-character evidence-window
                 budget. A window marked ``seed_budget_exceeded`` deliberately
                 exceeds it to preserve retrieved seed evidence, and must reach
                 the consumer intact. This raises the old 2000-char child cap
                 by only ~500 tokens across five verifier windows, leaving >53k
                 of the 64k context budget after the existing web-source allowance.
      Context budget at these limits (worst-case, 5+5 sources):
        source tokens  ~4375  |  total input ~5695  |  headroom 54k / 64k
        cost delta per verify call: +$0.00074 (+14.3%)
    """
    # Context assembly is authoritative for evidence-window budgeting. Retain
    # this cap for legacy/raw results, but never override an explicit,
    # telemetered seed-preserving overflow from that layer.
    WEB_CHARS = 1500
    KB_CHARS = MAX_EVIDENCE_WINDOW_CHARS

    parts = []
    for s in web_sources[:5]:
        parts.append(f"[WEB] {s['url']}\n{s['content'][:WEB_CHARS]}")
    for k in kb_results[:5]:
        text = k["text"]
        kb_text = text if k.get("seed_budget_exceeded") else text[:KB_CHARS]
        parts.append(f"[KB] {k['source']}\n{kb_text}")
    return "\n\n".join(parts) if parts else "No sources available."

def _build_citations(grounding_report: list, web_sources: list) -> str:
    """
    Builds the <ol> HTML block for the sources section of the article,
    Called by html_gen_node - output is HTML, not plain text.

    Priority order:
        1. grounding_report entries with status "verified" or "weak", deduped by URL
        2. If none, fall back to top-3 web_sources by score
        3. If web_sources also empty, return a single fallback <li>
    """
    import urllib.parse

    def _domain(url: str) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain if domain else url
        except Exception:
            return url

    seen_urls: set[str] = set()
    items: list[str] = []

    for entry in grounding_report:
        if entry.get("status") not in ("verified", "weak"):
            continue
        url = entry.get("source_url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append(
            f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">'
            f'{_domain(url)}</a></li>'
        )
    if not items:
        sorted_sources = sorted(web_sources, key=lambda s: s.get("score", 0), reverse=True)
        for s in sorted_sources[:3]:
            url = s.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = s.get("title") or _domain(url)
            items.append(
                f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">'
                f'{title}</a></li>'
            )

    if not items:
        return "<li>Sources retrieved via Tavily web search.</li>"

    return "\n                    ".join(items)



def _deduplicate_grounding_report(report: list[dict], run_id: str) -> list[dict]:
    """
    Remove near-exact duplicate claims from grounding_report.

    Uses difflib.SequenceMatcher (stdlib, no dependency) at threshold 0.85.
    Keeps the first occurrence; discards subsequent near-exact duplicates.

    Threshold rationale:
        - Known duplicate pair (STFT recommendation): ratio = 0.851  → caught
        - Known false-positive tests (different claims): ratio ≤ 0.53 → safe
        - Pair 1 (frequency resolution, semantic duplicate): ratio = 0.41
          → NOT caught by code; handled by verify_system.md instruction instead.
          Lowering threshold below 0.55 to catch it risks merging distinct claims.

    Evidence: FFT benchmark 2026-06-05 showed 2 duplicate pairs inflating
    unverified count. All verified benchmarks showed 0 false-positive merges
    at this threshold.
    """
    import difflib
    log = get_logger("verify_node")

    seen: list[str] = []
    unique: list[dict] = []
    dropped = 0

    for entry in report:
        claim_text = (entry.get("claim") or "").lower().strip()
        if not claim_text:
            continue
        is_duplicate = any(
            difflib.SequenceMatcher(None, claim_text, seen_claim).ratio() >= 0.85
            for seen_claim in seen
        )
        if is_duplicate:
            dropped += 1
        else:
            seen.append(claim_text)
            unique.append(entry)

    if dropped:
        log.info("verify.dedup", run_id=run_id, dropped=dropped,
                 before=len(report), after=len(unique))

    return unique



def _resolve_attributions(report: list[dict], web_sources: list, kb_results: list) -> list[dict]:
    """
       Annotate each grounding_report entry with resolved attribution (M5).

       The verifier's source_url is free text — never validated against the actual
       retrieved set until now. This resolves it post-hoc, with ZERO change to the
       verify prompt or source context (no metric re-baseline).

       Adds to each entry:
           source_kind: "web" | "kb" | "none" | "unresolved"
               none       = verifier attached no source (expected for unverified)
               unresolved = verifier emitted a source_url not in the retrieved set
                            (hallucinated attribution — a signal, not silently passed)
           source_ref:  canonical URL (web) or "kb:<file>" (kb); null otherwise
           kb_chunk_candidates: chunk_index list for the matched KB file (kb only) —
               exact-chunk disambiguation is deferred to M5b (requires verifier-
               visible source tagging, which re-baselines metrics).

       Matching is normalization-tolerant (lowercase, trailing-slash strip) — safe
       because collision risk within one run's <=15-source set is nil.
       """
    def _norm(u: str) -> str:
        return (u or "").strip().lower().rstrip("/")

    web_index = {_norm(s.get("url", "")): s.get("url") for s in web_sources if s.get("url")}
    kb_index: dict[str, list[int]] = {}
    for k in kb_results:
        kb_index.setdefault(k.get("source", "unknown"), []).append(k.get("chunk_index", 0))

    for entry in report:
        su = entry.get("source_url")
        if not su:
            entry["source_kind"] = "none"
            entry["source_ref"] = None
            continue
        nsu = _norm(su)
        if nsu in web_index:
            entry["source_kind"] = "web"
            entry["source_ref"] = web_index[nsu]
            continue
        kb_match = next(
            (src for src in kb_index if nsu == _norm(src) or _norm(src) in nsu or nsu in _norm(src)),
            None,
        )
        if kb_match:
            entry["source_kind"] = "kb"
            entry["source_ref"] = f"kb:{kb_match}"
            entry["kb_chunk_candidates"] = sorted(set(kb_index[kb_match]))
            continue
        entry["source_kind"] = "unresolved"
        entry["source_ref"] = None

    return report


# NODE: verify_node

def verify_node(state: AgentState) -> dict:
    """
    Extracts factual claims from draft and scores each against retrieved sources.
    """
    log = get_logger("verify_node")
    t_start = time.time()
    # cost gate
    if state.get("total_cost_usd", 0) >= COST_GATE_USD:
        log.warning("verify.cost_gate_hit", run_id=state["run_id"],
                    cost=state["total_cost_usd"])
        return {"grounding_report": [], "grounding_score": 0.0,
                "latency_ms": {**state.get("latency_ms", {}), "verify": 0}}

    client = _get_client()
    # 1. Extract claims from drafts
    # Ask llm to pull out every verifiable factual claim as a JSON list

    source_context = _build_source_context(state["web_sources"], state["kb_results"])

    claim_response = _llm_call(
        client,
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": f"""
                            Draft to verify:
                            {state["draft_markdown"]}
                            
                            Available sources:
                            {source_context}
                            
                            Return a JSON array. Each element:
                            {{"claim": "...", "source_url": "..." or null, "confidence": 0.0-1.0,
                              "status": "verified" | "weak" | "unverified",
                              "specificity": "substantive" | "generic"}}
                            
                            Return ONLY the JSON array. No preamble.
                            """
            }
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    latency = int((time.time() - t_start) * 1000)
    run_cost = _cost(claim_response.usage)

    # Parse grounding report
    raw = claim_response.choices[0].message.content.strip()

    try:
        grounding_report = _extract_json_array(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("verify.parse_failed", run_id=state["run_id"], error=str(e),
                  raw_preview=raw[:300])
        grounding_report = []

    # Remove near-exact duplicate claims before scoring.
    # Prompt instruction handles semantic duplicates; this catches string-level dupes.
    grounding_report = _deduplicate_grounding_report(grounding_report, run_id=state["run_id"])

    # M5: resolve each claim's source_url against the actual retrieved set.
    # Pure post-processing — verifier prompt and source context untouched.
    grounding_report = _resolve_attributions(
        grounding_report, state.get("web_sources", []), state.get("kb_results", [])
    )


    # Compute mean confidence
    if grounding_report:
        grounding_score = sum(r.get("confidence", 0) for r in grounding_report) / len(grounding_report)
    else:
        grounding_score = 0


    existing_latency = state.get("latency_ms", {})
    existing_latency["verify"] = latency

    n_verified = sum(1 for r in grounding_report if r.get("status") == "verified")
    n_weak = sum(1 for r in grounding_report if r.get("status") == "weak")
    n_unverified = sum(1 for r in grounding_report if r.get("status") == "unverified")

    # Grounded-depth signal (M3): SV = Substantive and verified
    n_substantive = sum(1 for r in grounding_report if r.get("specificity") == "substantive")
    n_substantive_verified = sum(
        1 for r in grounding_report
        if r.get("specificity") == "substantive" and r.get("status") == "verified"
    )


    log.info("verify.complete",
             run_id=state["run_id"],
             grounding_score=round(grounding_score, 3),
             claims=len(grounding_report),
             verified=n_verified,
             weak=n_weak,
             unverified=n_unverified,
             substantive=n_substantive,
             substantive_verified=n_substantive_verified,
             latency_ms=latency,
             cost=round(run_cost, 5),
             )

    # M4 instrumentation: persist this iteration's metrics. verify runs once per
    # draft iteration; appending here gives per-iteration SV/UVR in telemetry.
    iteration_metrics = list(state.get("iteration_metrics", []) or [])
    iteration_metrics.append({
        "iteration": state.get("iterations", 0),
        "SV": n_substantive_verified,
        "S": n_substantive,
        "V": n_verified,
        "W": n_weak,
        "U": n_unverified,
        "N": len(grounding_report),
        "uvr": round(n_unverified / max(len(grounding_report), 1), 3),
        "grounding_score": round(grounding_score, 3),
        "m4_feedback_claims": state.get("m4_feedback_claims", 0),
        "unverified_claims": [
            (r.get("claim") or "")[:200]
            for r in grounding_report if r.get("status") == "unverified"
        ],
        # M5: full annotated report per iteration — the final grounding_report
        # only reflects the LAST iteration; without this, iteration 1 of a
        # revised run was un-reconstructable.
        "grounding_report": grounding_report,
    })

    return {
        "grounding_report": grounding_report,
        "grounding_score": round(grounding_score, 3),
        "iteration_metrics": iteration_metrics,
        "total_tokens": state.get("total_tokens", 0) + claim_response.usage.total_tokens,
        "total_cost_usd": state.get("total_cost_usd", 0) + run_cost,
        "latency_ms": existing_latency,
    }


def _format_grounding_summary(grounding_report: list) -> str:
    if not grounding_report:
        return "No grounding report available."
    verified = [r for r in grounding_report if r.get("status") == "verified"]
    weak = [r for r in grounding_report if r.get("status") == "weak"]
    unverified = [r for r in grounding_report if r.get("status") == "unverified"]
    return (f"{len(verified)} verified, {len(weak)} weak, {len(unverified)} unverified "
            f"out of {len(grounding_report)} total claims.")


# NODE: reflect_node

def reflect_node(state: AgentState) -> dict:
    """
    Self-evaluates draft on structure, depth, grounding. Scores 1-10.
    """
    log = get_logger("reflect_node")
    t_start = time.time()

    if state.get("total_cost_usd", 0) >= COST_GATE_USD:
        log.warning("reflect.cost_gate_hit", run_id=state["run_id"])
        return {"reflection_score": 7, "reflection_notes": "Cost gate — skipped reflect",
                "latency_ms": {**state.get("latency_ms", {}), "reflect": 0}}

    client = _get_client()
    grounding_summary = _format_grounding_summary(state.get("grounding_report", []))

    response = _llm_call(
        client,
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": REFLECT_SYSTEM,
            },
            {
                "role": "user",
                "content": f"""
            Topic: {state["topic"]}
            Series context: {state["series_context"]}
            
            Draft:
            {state["draft_markdown"]}
            
            Grounding report summary:
            {grounding_summary}
            Grounding score: {state.get("grounding_score", 0):.2f}
            
            Evaluate this draft on:
            1. Structure: Are all 4 sections present and coherent?
            2. Technical depth: Appropriate for senior engineers learning this topic?
            3. Grounding: Are claims backed by sources?
            4. Clarity: Is this human-written quality or AI-slop?
            
            Return JSON only:
            {{"score": <int 1-10>, "notes": "<2-3 sentences of specific critique>"}}
            """
            }
        ],
        temperature=0.1,
        max_tokens=500,
    )

    latency = int((time.time() - t_start) * 1000)
    run_cost = _cost(response.usage)

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        reflection_score = int(parsed.get("score", 7))
        reflection_notes = parsed.get("notes", "")
    except (json.JSONDecodeError, ValueError) as e:
        log.error("reflect.parse_failed", run_id=state["run_id"], error=str(e))
        reflection_score = 7
        reflection_notes = f"Parse error: {e}"


    existing_latency = state.get("latency_ms", {})
    existing_latency["reflect"] = latency

    log.info("reflect.complete", run_id=state["run_id"],
             reflection_score=reflection_score,
             grounding_score=state.get("grounding_score", 0),
             latency=latency, cost=round(run_cost, 5))


    return {
        "reflection_score": reflection_score,
        "reflection_notes": reflection_notes,
        "total_tokens": state.get("total_tokens", 0) + response.usage.total_tokens,
        "total_cost_usd": state.get("total_cost_usd", 0) + run_cost,
        "latency_ms": existing_latency,
    }

# NODE: hitl_node

def hitl_node(state: AgentState) -> dict:
    """
    Interactive HITL gate. Displays draft + grounding report + reflection.
    User inputs: a (approve), r (reject), f (feedback -> types it).
    """
    if os.environ.get("HITL_AUTO_APPROVE") == "1":
        return {"hitl_status": "approved", "hitl_feedback": None}

    # B4: API mode — pause the graph via LangGraph interrupt(). The checkpointer
    # (set by the API server's build_graph) persists full state to SQLite, so the
    # review can outlive a process restart. On resume via Command(resume=...),
    # this node re-executes from the top and interrupt() returns the resume payload
    # instead of pausing again. hitl_node has no LLM calls, so re-execution is free
    # and cannot double-count cost. Fail-safe default is REJECT — never auto-publish.

    if os.environ.get("HITL_MODE") == "api":
        from langgraph.types import interrupt
        decision = interrupt({
            "type": "hitl_review",
            "run_id": state["run_id"],
            "topic": state["topic"],
            "slug": state["slug"],
            "draft_markdown": state["draft_markdown"],
            "grounding_report": state.get("grounding_report", []),
            "grounding_score": state.get("grounding_score", 0.0),
            "reflection_score": state.get("reflection_score", 0),
            "reflection_notes": state.get("reflection_notes", ""),
            "error_log": state.get("error_log", []),
        }) or {}
        action = decision.get("action")
        if action == "approve":
            return {"hitl_status": "approved", "hitl_feedback": None}
        if action == "feedback":
            fb = (decision.get("feedback") or "").strip()
            if fb:
                return {"hitl_status": "feedback", "hitl_feedback": fb}
        return {"hitl_status": "rejected", "hitl_feedback": None}


    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(Panel(f"[bold]DRAFT REVIEW — {state['topic']}[/bold]", style="orange1"))
    console.print(state["draft_markdown"])

    # Grounding table
    table = Table(title="Source Grounding Report")
    table.add_column("Claim", max_width=60)
    table.add_column("Status")
    table.add_column("Confidence")
    table.add_column("Source")
    for r in state.get("grounding_report", []):
        status_color = {"verified": "green", "weak": "yellow", "unverified": "red"}.get(
            r.get("status", ""), "white")
        table.add_row(
            r.get("claim", "")[:60],
            f"[{status_color}]{r.get('status', '')}[/{status_color}]",
            f"{r.get('confidence', 0):.2f}",
            r.get("source_url", "none") or "none",
        )
    console.print(table)

    console.print(f"\n[bold]Reflection:[/bold] {state['reflection_score']}/10 — {state['reflection_notes']}")
    console.print(f"[bold]Grounding score:[/bold] {state['grounding_score']:.2f}")
    if state.get("error_log"):
        console.print(f"[yellow]Warnings:[/yellow] {'; '.join(state['error_log'])}")

    while True:
        choice = input("\n[a]pprove / [r]eject / [f]eedback: ").strip().lower()
        if choice == 'a':
            return {"hitl_status": "approved", "hitl_feedback": None}
        elif choice == 'r':
            return {"hitl_status": "rejected", "hitl_feedback": None}
        elif choice == 'f':
            feedback = input("Feedback: ").strip()
            if feedback:
                return {"hitl_status": "feedback", "hitl_feedback": feedback}
        else:
            print("Enter a, r, or f.")


def _render_problem_framing(raw: str) -> str:
    paragraphs = [p.strip() for p in raw.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        return f"<p>{html_module.escape(raw.strip())}</p>"
    return "\n".join(f"<p>{html_module.escape(p)}</p>" for p in paragraphs)


def _render_code_snippets(raw: str) -> str:
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    matches = pattern.findall(raw)

    if not matches:
        escaped = html_module.escape(raw.strip())
        return (
            '<div class="sl-code-block">'
            '<div class="sl-code-label">TEXT</div>'
            f'<pre><code>{escaped}</code></pre>'
            '</div>'
        )

    blocks = []
    for lang, content in matches:
        lang = lang.strip() or "text"
        escaped = html_module.escape(content.strip())
        blocks.append(
            f'<div class="sl-code-block">'
            f'<div class="sl-code-label">{lang.upper()}</div>'
            f'<pre><code class="language-{lang}">{escaped}</code></pre>'
            f'</div>'
        )
    return "\n\n".join(blocks)


def _render_takeaways(raw: str) -> str:
    lines = raw.strip().splitlines()
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\-\*\•]\s+', '', line)
        line = re.sub(r'^\d+\.\s+', '', line)
        if line:
            items.append(f"<li>{html_module.escape(line)}</li>")
    if not items:
        return "<ul><li>No takeaways generated.</li></ul>"
    return "<ul>\n" + "\n".join(items) + "\n</ul>"


def _render_technical_dive_via_llm(topic: str, technical_dive: str, client, run_id: str) -> tuple[str, int, float]:
    prompt = f"""Convert this technical section into clean HTML for an article about: {topic}

RULES:
- Use <h3> for subsections
- Use <p> for paragraphs
- Use <code> for inline code (never backticks in output)
- Use <strong> for emphasis
- Use <div class="callout callout-info"> for important notes/warnings (both classes required)
- Use <ul><li> for lists
- Do NOT wrap output in any container div
- Do NOT include DOCTYPE, html, head, or body tags
- Return ONLY the HTML. No preamble. No markdown fences.

Content to convert:
{technical_dive}"""

    response = _llm_call(
        client,
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are an HTML conversion engine. Return only valid HTML elements, no markdown, no fences, no preamble."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=3000,
    )

    rendered = response.choices[0].message.content.strip()
    if rendered.startswith("```"):
        rendered = re.sub(r'^```\w*\n?', '', rendered)
        rendered = re.sub(r'\n?```$', '', rendered)
        rendered = rendered.strip()

    return rendered, response.usage.total_tokens, _cost(response.usage)



def hitl_html_node(state: AgentState) -> dict:
    """
    P2 second HITL gate (post-render). The human reviews the GENERATED HTML before publish.
      approve         -> git (publish)
      reject          -> END (nothing published)
      request_changes -> draft (revise): the note is stored in hitl_feedback (the SAME channel
                         draft_node already consumes, lines 32-33), and the graph loops
                         draft -> verify -> reflect -> draft gate -> html_gen -> here again.
    No LLM calls, so interrupt() re-execution on resume is free. Fail-safe default is REJECT.
    """
    if not state.get("html_output"):
        return {"html_review_status": "rejected"}

    if os.environ.get("HITL_AUTO_APPROVE") == "1":
        return {"html_review_status": "approved"}

    if os.environ.get("HITL_MODE") == "api":
        from langgraph.types import interrupt
        decision = interrupt({
            "type": "hitl_html_review",
            "run_id": state["run_id"],
            "topic": state["topic"],
            "slug": state["slug"],
            "html_filename": state.get("html_filename"),
            "html_output": state.get("html_output"),
            "validation_warnings": [e for e in state.get("error_log", []) if "html_gen" in e],
            "grounding_score": state.get("grounding_score", 0.0),
        }) or {}
        action = decision.get("action")
        if action == "approve":
            return {"html_review_status": "approved"}
        # /feedback endpoint sends action="feedback"; accept "request_changes" as an alias.
        if action in ("request_changes", "feedback"):
            note = (decision.get("feedback") or "").strip()
            if note:
                # Layout/design/positioning only — content is frozen after the draft gate.
                return {"html_review_status": "changes", "html_feedback": note}

    # CLI interactive
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    preview = Path("outputs/preview") / f"{state['slug']}.html"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(state["html_output"], encoding="utf-8")
    console.print(Panel(f"[bold]HTML REVIEW — {state['topic']}[/bold]", style="cyan"))
    console.print(f"Rendered file  : {state.get('html_filename')}")
    console.print(f"Open in browser: file://{preview.resolve()}")
    warnings = [e for e in state.get("error_log", []) if "html_gen" in e]
    if warnings:
        console.print(f"[yellow]Validation warnings:[/yellow] {'; '.join(warnings)}")
    while True:
        choice = input("\n[a]pprove (publish) / [c]hanges to LAYOUT (revise) / [r]eject (discard): ").strip().lower()
        if choice == "a":
            return {"html_review_status": "approved"}
        if choice == "r":
            return {"html_review_status": "rejected"}
        if choice == "c":
            note = input("Design/layout/positioning change (NOT content): ").strip()
            if note:
                return {"html_review_status": "changes", "html_feedback": note}


def route_after_html_review(state: AgentState) -> str:
    """approved -> git; changes -> draft (revise loop); anything else -> END."""
    from langgraph.graph import END
    log = get_logger("router")
    status = state.get("html_review_status", "rejected")
    log.info("html_review.decision", run_id=state["run_id"], status=status)
    if status == "approved":
        return "git"
    if status == "changes":
        return "html_revise"
    return END


# NODE: html_gen_node

def html_gen_node(state: AgentState) -> dict:
    log = get_logger("html_gen_node")
    t_start = time.time()

    if state.get("total_cost_usd", 0) >= COST_GATE_USD:
        log.warning("html_gen.cost_gate_hit", run_id=state["run_id"])
        return {"html_output": None, "html_filename": None,
                "latency_ms": {**state.get("latency_ms", {}), "html_gen": 0}}

    template_raw = Path("prompts/html_template.md").read_text(encoding='utf-8')

    if "```html" in template_raw:
        template = template_raw.split("```html", 1)[1]
        template = template.split("```", 1)[0].strip()
    else:
        template = template_raw.strip()



    client = _get_client()
    draft = state.get("draft_sections", {})
    topic = state["topic"]
    topic_raw = topic
    run_id = state["run_id"]

    # Python renders 3 sections deterministically (no LLM, no tokens)
    problem_framing_html = _render_problem_framing(
        draft.get("problem_framing", "No problem framing available.")
    )
    code_snippets_html = _render_code_snippets(draft.get("code_snippets", ""))
    takeaways_html = _render_takeaways(draft.get("takeaways", ""))

    # LLM renders only technical_dive (mixed prose/headings/callouts — genuinely complex)
    log.info("html_gen.technical_dive_start", run_id=run_id)
    technical_dive_html, td_tokens, td_cost = _render_technical_dive_via_llm(
        topic=topic,
        technical_dive=draft.get("technical_dive", ""),
        client=client,
        run_id=run_id,
    )

    citations_html = _build_citations(
        state.get("grounding_report", []),
        state.get("web_sources", []),
    )

    topic_short = topic.lower()

    word_count = len(state.get("draft_markdown", "").split())
    read_time = str(max(5, word_count // 200))

    series_context = state.get("series_context", "Learning Log")
    if "·" in series_context:
        series_label = series_context.split("·")[0].strip()
    else:
        series_label = series_context
    breadcrumb_section = series_label

    # Meta description: first 155 chars of problem_framing, ending at a sentence boundary
    raw_pf = draft.get("problem_framing", "")
    meta_desc = raw_pf[:155]
    if len(raw_pf) > 155 and "." in meta_desc:
        meta_desc = meta_desc[:meta_desc.rfind(".") + 1]
    meta_desc = html_module.escape(meta_desc)

    # Substitute all placeholders — order matters for ones that appear multiple times

    html = template
    html = html.replace("{{TOPIC}}", html_module.escape(topic))
    html = html.replace("{{TOPIC_SHORT}}", html_module.escape(topic_short))
    html = html.replace("{{SLUG}}", state["slug"])
    html = html.replace("{{META_DESCRIPTION}}", meta_desc)
    html = html.replace("{{SERIES_LABEL}}", html_module.escape(series_label))
    html = html.replace("{{BREADCRUMB_SECTION}}", html_module.escape(breadcrumb_section))
    html = html.replace("{{DIFFICULTY}}", "Intermediate")
    html = html.replace("{{READ_TIME}}", read_time)
    html = html.replace("{{PROBLEM_FRAMING}}", problem_framing_html)
    html = html.replace("{{TECHNICAL_DIVE}}", technical_dive_html)
    html = html.replace("{{CODE_SNIPPETS}}", code_snippets_html)
    html = html.replace("{{TAKEAWAYS}}", takeaways_html)
    html = html.replace("{{SOURCES}}", citations_html)


    # Fix HTML entities inside the LD+JSON block — JSON must not contain &amp; etc.
    import re
    def _fix_ldjson(match):
        return match.group(0).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = re.sub(
        r'<script type="application/ld\+json">.*?</script>',
        _fix_ldjson,
        html,
        flags=re.DOTALL
    )

    # Validate
    errors = []
    if "<!DOCTYPE html>" not in html:
        errors.append("Missing DOCTYPE")
    if 'id="main"' not in html:
        errors.append('Missing id="main"')
    if "<h1>" not in html:
        errors.append("Missing h1")
    remaining = re.findall(r'\{\{[A-Z_]+\}\}', html)
    if remaining:
        errors.append(f"Unreplaced placeholders: {remaining}")

    latency = int((time.time() - t_start) * 1000)

    # PUBLISHED filename — ALWAYS the canonical <slug>.html. git_node writes this into
    # the website repo via state["html_filename"], so republishing the same topic targets
    # the same file and git sees a MODIFICATION (-> tagged_and_merged), not a new file.
    # B6 finding: previously this name took a run-id suffix whenever a LOCAL archive copy
    # already existed, leaking local-archive collision-avoidance into the published URL.
    filename = f"{state['slug']}.html"

    # LOCAL archive copy (debug history) — decoupled from the published name. Keep a
    # per-run-unique name here so we never clobber a prior run's HTML in outputs/articles/.
    # This path is write-only and never reaches git_node.
    archive_name = filename
    archive_path = Path("outputs/articles") / archive_name
    if archive_path.exists():
        suffix = state["run_id"].split("-")[-1][:6]
        archive_name = f"{state['slug']}-{suffix}.html"
    archive_path = Path("outputs/articles") / archive_name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(html, encoding="utf-8")


    existing_latency = state.get("latency_ms", {})
    existing_latency["html_gen"] = latency

    error_log = state.get("error_log", [])
    if errors:
        error_log.append(f"[html_gen] Validation warnings: {errors}")

    log.info("html_gen.complete", run_id=run_id, filename=filename,
             archive=archive_name,
             validation_errors=errors, latency_ms=latency,
             td_tokens=td_tokens, td_cost=round(td_cost, 5))

    return {
        "html_output": html,
        "html_filename": filename,
        "total_tokens": state.get("total_tokens", 0) + td_tokens,
        "total_cost_usd": state.get("total_cost_usd", 0) + td_cost,
        "latency_ms": existing_latency,
        "error_log": error_log,
    }


def _visible_words(html_str: str) -> list:
    """Lowercased word tokens of the visible text (tags stripped) — used to prove an HTML
    revision changed only markup/layout, never content."""
    text = html_module.unescape(re.sub(r"<[^>]+>", " ", html_str or ""))
    return re.findall(r"\w+", text.lower())


def html_revise_node(state: AgentState) -> dict:
    """
    P2 layout revision. Applies the reviewer's design/structure/formatting/positioning note to
    the already-rendered HTML via ONE constrained LLM pass, then loops back to hitl_html.
    Content is frozen after the draft gate; this pass must not alter text/claims/code/sources.
    A word-multiset guard ENFORCES that: if the visible word content changes, the revision is
    DISCARDED and the original HTML is kept (with a warning), so the freeze can't be violated
    even by a misbehaving model.
    """
    from collections import Counter
    t0 = time.time()
    log = get_logger("html_revise")
    original = state["html_output"]
    note = state.get("html_feedback") or ""

    system = Path("prompts/html_revise_system.md").read_text(encoding="utf-8")
    user = (
        "Apply ONLY these design/structure/formatting/positioning changes. Do NOT add, remove, "
        "reword, or reorder any sentence, claim, code line, or source — the article text is FROZEN.\n\n"
        f"CHANGES REQUESTED:\n{note}\n\n"
        f"CURRENT HTML (return the COMPLETE revised document, HTML only, no commentary):\n{original}"
    )
    client = _get_client()
    resp = _llm_call(client, model=DEEPSEEK_MODEL,
                     messages=[{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                     temperature=0)
    revised = re.sub(r"^```(?:html)?\s*|\s*```$", "", resp.choices[0].message.content.strip()).strip()

    usage = resp.usage
    cost = round(usage.prompt_tokens / 1_000_000 * DEEPSEEK_INPUT_COST_PER_M
                 + usage.completion_tokens / 1_000_000 * DEEPSEEK_OUTPUT_COST_PER_M, 6)

    # Content-freeze guard.
    before, after = Counter(_visible_words(original)), Counter(_visible_words(revised))
    drift = sum((before - after).values()) + sum((after - before).values())
    looks_like_html = ("<html" in revised.lower()) or revised.lower().startswith("<!doctype")
    errors = list(state.get("error_log", []))
    if drift > 2 or not looks_like_html:
        errors.append(f"html_revise: revision DISCARDED (content drift={drift} words / valid_html={looks_like_html}); kept original")
        log.warning("html_revise.discarded", run_id=state["run_id"], drift=drift, valid_html=looks_like_html)
        out_html = original
    else:
        log.info("html_revise.applied", run_id=state["run_id"], drift=drift, cost=cost)
        out_html = revised

    lat = dict(state.get("latency_ms", {}))
    lat["html_revise"] = int((time.time() - t0) * 1000)
    return {
        "html_output": out_html,
        "html_feedback": None,
        "total_tokens": state.get("total_tokens", 0) + usage.total_tokens,
        "total_cost_usd": round(state.get("total_cost_usd", 0.0) + cost, 6),
        "latency_ms": lat,
        "error_log": errors,
    }


_CATEGORY_LABELS = {
    "concept-exploration": "Concept Exploration",
    "project-deep-dives": "Project Deep Dives",
    "field-notes": "Field Notes",
}


def _derive_excerpt(state: AgentState, max_len: int = 220) -> str:
    """Plain-text excerpt for the Learning Log card, from the (frozen) problem framing.
    No LLM call — strips markdown and truncates at a word boundary."""
    sections = state.get("draft_sections") or {}
    raw = sections.get("problem_framing") or state.get("draft_markdown") or ""
    text = re.sub(r"[#*`_>\[\]]", "", raw)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "…"
    return text or "A technical deep dive."


def _build_index_card(filename: str, title: str, category: str, excerpt: str) -> str:
    """Build one Learning Log <a class="article-card"> block, matching index.html's exact
    indentation so insertion produces a minimal, clean diff."""
    now = datetime.datetime.now(datetime.timezone.utc)
    datetime_attr = now.strftime("%Y-%m-%d")
    display_date = f"{now.strftime('%B')} {now.day}, {now.year}"
    label = _CATEGORY_LABELS.get(category, "Concept Exploration")
    safe_title = html_module.escape(title)
    safe_excerpt = html_module.escape(excerpt)
    return (
        f'                    <a href="{filename}" class="article-card" data-category="{category}">\n'
        f'                        <time class="article-date" datetime="{datetime_attr}">{display_date}</time>\n'
        f'                        <h3 class="article-title">{safe_title}</h3>\n'
        f'                        <p class="article-excerpt">\n'
        f'                            {safe_excerpt}\n'
        f'                        </p>\n'
        f'                        <div class="article-tags">\n'
        f'                            <span class="article-tag">{label}</span>\n'
        f'                        </div>\n'
        f'                    </a>'
    )


def _update_index_html(repo_path: Path, filename: str, title: str, category: str,
                       excerpt: str, log, run_id: str) -> tuple[bool, str]:
    """Insert (newest-first) or update-in-place the Learning Log card for this article.
    Idempotent: re-publishing the same slug REPLACES its card instead of duplicating.
    Non-fatal: if index.html or the grid anchor is missing, skip and let the publish proceed."""
    index_path = repo_path / "index.html"
    if not index_path.exists():
        log.warning("index.not_found", run_id=run_id, path=str(index_path))
        return False, "index.html not found"

    html = index_path.read_text(encoding="utf-8")
    anchor = '<div class="articles-grid" id="articlesGrid">'
    if anchor not in html:
        log.warning("index.anchor_missing", run_id=run_id)
        return False, "articles-grid anchor missing"

    card = _build_index_card(filename, title, category, excerpt)
    existing = re.compile(r'[ \t]*<a href="' + re.escape(filename) + r'"[^>]*>.*?</a>\n?', re.DOTALL)
    if existing.search(html):
        new_html, action = existing.sub(card + "\n", html, count=1), "updated"
    else:
        new_html, action = html.replace(anchor, anchor + "\n" + card, 1), "inserted"

    index_path.write_text(new_html, encoding="utf-8")
    log.info("index.updated", run_id=run_id, action=action, href=filename)
    return True, action


# NODE: git_node

def git_node(state: AgentState) -> dict:
    """
    Push the generated HTML to themachinist-website repo via feature branch.

    Workflow:
        1. Write HTML to ../themachinist-website/<slug>.html
        2. Create and checkout feature/article-<slug>
        3. Commit with standardized message
        4. Diff vs main — if only new file, merge; if existing files changed, tag first
        5. Keep last 5 tags, delete feature branch after merge
        6. Set git_status to "merged", "tagged_and_merged", or "failed"

    Every git operation is wrapped in try/except. Failure does NOT crash the
    pipeline — the article HTML is already saved in outputs/articles/.
    """
    import git
    from git.exc import GitCommandError
    from config import THEMACHINIST_REPO_PATH


    log = get_logger("git_node")
    t_start = time.time()

    slug = state.get("slug", "draft")
    branch = f"feature/article-{slug}"
    filename = state.get("html_filename")
    html_content = state.get("html_output")
    topic = state.get("topic", slug)

    existing_latency = state.get("latency_ms", {})
    existing_latency["git"] = 0
    error_log = state.get("error_log", [])

    # Dry-run guard — if GIT_PUSH_ENABLED is not "true", log intent and skip
    git_push_enabled = os.environ.get("GIT_PUSH_ENABLED", "false").lower() == "true"
    if not git_push_enabled:
        log.info("git.dry_run", run_id=state["run_id"], branch=branch,
                 note="GIT_PUSH_ENABLED is not true — skipping git operations")
        return {
            "branch_name": branch,
            "git_status": "dry_run",
            "latency_ms": existing_latency,
            "error_log": error_log,
        }

    # Validate
    if not html_content or not filename:
        log.error("git.missing_content", run_id=state["run_id"])
        error_log.append("[git_node] No HTML content or filename — nothing to push")
        return {
            "branch_name": branch,
            "git_status": "failed",
            "latency_ms": existing_latency,
            "error_log": error_log,
        }

    repo_path = Path(THEMACHINIST_REPO_PATH).resolve()
    if not repo_path.exists():
        log.error("git.repo_not_found", run_id=state["run_id"], path=str(repo_path))
        error_log.append(f"[git_node] Repo not found at {repo_path}")
        return {
            "branch_name": branch,
            "git_status": "failed",
            "latency_ms": existing_latency,
            "error_log": error_log,
        }

    git_status = "failed"
    repo = None
    original_branch = "main"

    try:
        repo = git.Repo(str(repo_path))
        original_branch = repo.active_branch.name

        # 1. Write html to the website repo
        dest_path = repo_path/filename
        dest_path.write_text(html_content, encoding='utf-8')

        # P2.2: insert/update the Learning Log card in index.html (same commit).
        index_ok, _index_action = _update_index_html(
            repo_path, filename, topic,
            state.get("category", "concept-exploration"),
            _derive_excerpt(state), log, state["run_id"])
        if not index_ok:
            error_log.append("[git_node] index.html not updated (anchor/file missing) — article still published")

        # 2. Create and checkout feature branch
        if branch in repo.heads:
            repo.delete_head(branch, force=True)

        new_branch = repo.create_head(branch)
        new_branch.checkout()

        # 3. Stage and commit
        repo.index.add([filename] + (["index.html"] if index_ok else []))
        commit_msg = f"feat: add {topic} article [content-agent]"
        commit = repo.index.commit(commit_msg)

        # 4. Diff vs main to decide merge strategy
        main_commit = repo.commit("main")
        diff_index = main_commit.diff(commit)

        # index.html is updated on EVERY publish; exclude it so a brand-new article still
        # reports "merged" (and a real re-publish still reports "tagged_and_merged").
        changed_files = [d.b_path for d in diff_index
                         if d.change_type in ("M", "D", "R") and d.b_path != "index.html"]
        new_files = [d.b_path for d in diff_index if d.change_type == "A"]

        # 5. Merge or tag-then-merge
        if changed_files:
            date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
            tag_name = f"v-{date_str}-{slug}"
            existing_tag_names = [t.name for t in repo.tags]
            if tag_name in existing_tag_names:
                repo.delete_tag(tag_name)

            repo.create_tag(tag_name, ref=main_commit, message=f"Pre-merge state before {branch}")
            # Keep last 5 tags only
            tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime if t.commit else 0, reverse=True)
            for old_tag in tags[5:]:
                repo.delete_tag(old_tag)

            repo.heads.main.checkout()
            repo.git.merge(branch, no_ff=True)
            git_status = "tagged_and_merged"

        else:
            repo.heads.main.checkout()
            repo.git.merge(branch, no_ff=True)
            git_status = "merged"

        # 6. Delete the feature branch
        repo.delete_head(branch, force=True)

        log.info("git.success", run_id=state["run_id"], branch=branch,
             status=git_status, changed_files=len(changed_files),
             new_files=len(new_files))

    except GitCommandError as e:
        log.error("git.command_error", run_id=state["run_id"], error=str(e))
        error_log.append(f"[git_node] Git command failed: {e}")

    except Exception as e:
        log.error("git.unexpected_error", run_id=state["run_id"], error=str(e))
        error_log.append(f"[git_node] Unexpected error: {e}")

    finally:
        # Restore original branch if we have a repo reference
        if repo is not None:
            try:
                if repo.active_branch.name != original_branch:
                    repo.heads[original_branch].checkout()
            except Exception:
                pass

    latency = int((time.time() - t_start) * 1000)
    existing_latency["git"] = latency
    return {
        "branch_name": branch,
        "git_status": git_status,
        "latency_ms": existing_latency,
        "error_log": error_log,
    }



# EDGE FUNCTIONS (routing logic)

def route_after_reflect(state: AgentState) -> str:
    """
        Decide whether to revise the draft or proceed to HITL.

        Composite gate (NOT reflection score alone — LLMs inflate self-scores):
            Force rewrite if:
                - grounding_score < GROUNDING_FLOOR (hard floor, regardless of reflection)
                - OR reflection_score < REFLECTION_THRESHOLD AND grounding_score < 0.75
            Proceed if:
                - max iterations reached (always proceed — human decides)
                - composite gate passes

        Returns:
            "draft" — loop back and revise
            "hitl"  — proceed to human review
    """
    log = get_logger("router")
    iterations = state.get("iterations", 0)
    reflection_score = state.get("reflection_score", 8)
    grounding_score = state.get("grounding_score", 0.75)

    # Hard ceiling: never loop more than MAX_ITERATIONS
    if iterations >= MAX_ITERATIONS:
        log.info("route.max_iterations", run_id=state["run_id"], iterations=iterations)
        return "hitl"

    # Cost gate check
    if state.get("total_cost_usd", 0) >= COST_GATE_USD:
        log.info("route.cost_gate", run_id=state["run_id"], cost=round(state.get("total_cost_usd", 0), 4))
        return "hitl"

    # Composite gate
    hard_floor_fail = grounding_score < GROUNDING_FLOOR
    soft_fail = reflection_score < REFLECTION_THRESHOLD and grounding_score < 0.75

    if hard_floor_fail or soft_fail:
        reason = "grounding below floor" if hard_floor_fail else "reflection + grounding both weak"
        log.info("route.revise", run_id=state["run_id"], reason=reason, reflection=reflection_score, grounding=round(grounding_score, 2))
        return "draft"

    log.info("route.proceed", run_id=state["run_id"], reflection=reflection_score, grounding=round(grounding_score, 2))
    return "hitl"

def route_after_hitl(state: AgentState) -> str:
    """
    Route based on HITL decision.

    Returns:
        "html_gen" — approved, generate HTML
        "draft"    — feedback given, revise
        END        — rejected, terminate run
    """
    from langgraph.graph import END
    log = get_logger("router")
    status = state.get("hitl_status", "pending")
    log.info("hitl.decision", run_id=state["run_id"], status=status)

    if status == "approved":
        return "html_gen"
    elif status == "feedback":
        return "draft"
    else:   # rejected or unknown
        return END
