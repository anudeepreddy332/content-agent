""""
agent/nodes.py
--------------
All node and edge functions for the content-agent pipeline.

Day 23 status:
    COMPLETE: draft_node, retrieve_node
    STUBBED:  verify_node, reflect_node, hitl_node, html_gen_node, git_node

Each node signature: (state: AgentState) -> dict
    Nodes return ONLY the keys they update.
    LangGraph merges the return dict into the existing state.
    Never return the full state — only your changes.

Vulnerabilities to watch:
    - draft_node: DeepSeek may return malformed JSON. Handled with try/except + fallback.
    - retrieve_node: Tavily may return 0 results on niche topics. KB may be empty.
      Both are handled — node degrades gracefully, does not crash.
    - draft_node loops back from hitl if feedback is given. On loop, iterations increments.
      Max iterations enforced in route_after_reflect, not here.
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from agent.state import AgentState, DraftSections
from tools.web_search import web_search
from tools.query_kb import query_kb
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
)

# DeepSeek client
# Initialized once at module load. api_key read at call time (not cached at import)
# so that test_resilience.py invalid-key tests work correctly (Phase 3 lesson).

def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

def _load_system_prompt() -> str:
    """Load draft system prompt from prompts/draft_system.md."""
    prompt_path = Path("prompts/draft_system.md")
    if not prompt_path:
        raise FileNotFoundError(f"Draft system prompt not found at {prompt_path}")

    # Strip the comment header lines (lines starting with #) before the actual prompt
    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    content_lines = [l for l in lines if not l.startswith("#")]
    return "\n".join(content_lines).strip()

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

    t_start = time.time()
    client = _get_client()
    system_prompt = _load_system_prompt()

    # Build user message
    feedback_block = ""
    if state.get("hitl_feedback"):
        feedback_block = f"\n\nREVISION FEEDBACK FROM HUMAN REVIEWER:\n{state['hitl_feedback']}\nAddress this feedback specifically in the new draft."

    user_message = f"""Write a technical article for The Machinist on the following topic.

                    Topic: {state['topic']}
                    Card ID: {state['card_id']}
                    Series context: {state['series_context']}
                    
                    This article will be published on themachinist.org under the Learning Log section.
                    The audience is engineers learning ML and agentic AI — they are smart but new to this specific topic.
                    {feedback_block}
                    
                    Return ONLY the JSON object as specified in your instructions. No markdown wrapper."""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
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
        print(f"[draft_node] JSON parse failed: {e}")
        print(f"[draft_node] Raw response: {raw[:500]}")
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

    print(f"[draft_node] Done. Tokens: {response.usage.total_tokens} | Cost: ${run_cost:.5f} | Latency: {latency}ms")

    return {
        "draft_sections": sections,
        "draft_markdown": draft_markdown,
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
    t_start = time.time()

    topic = state["topic"]

    # Generate targeted search queries
    # Why hardcoded patterns instead of LLM-generated queries?
    # Speed and cost. LLM query generation adds a full round-trip for marginal gain.
    # These three angles cover 90% of what the verify_node needs.

    queries = [
        f"{topic} explained technical,"
        f"{topic} failure modes limitations production",
        f"{topic} implementation Python example",
    ]

    # Web search - deduplicate by URL across all queries
    seen_urls = set()
    web_sources = []

    for query in queries:
        results = web_search(query, max_results=5)
        for r in results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                web_sources.append(r)

    # KB query — use topic + first 100 chars of problem_framing for richer context
    problem_framing_preview = state.get("draft_sections", {}).get("problem_framing", "")[:100]
    kb_query = f"{topic} {problem_framing_preview}".strip()
    kb_results = query_kb(query=kb_query, n_results=5)

    latency = int((time.time() - t_start) * 1000)
    existing_latency = state.get("latency_ms", {})
    existing_latency["retrieve"] = latency

    print(f"[retrieve_node] Web: {len(web_sources)} sources | KB: {len(kb_results)} chunks | Latency: {latency}ms")

    return {
        "web_sources": web_sources,
        "kb_results": kb_results,
        "latency_ms": existing_latency,
    }

# NODE: verify_node

def verify_node(state: AgentState) -> dict:
    """
    STUB: Full implementation on Day 24.
    Extracts factual claims from draft and scores each against retrieved sources.

    For now: passes through with a neutral grounding score so the graph runs end-to-end.
    """
    print("[verify_node] STUB — passing through with neutral grounding score")
    existing_latency = state.get("latency_ms", {})
    existing_latency["verify"] = 0

    return {
        "grounding_report": [],
        "grounding_score": 0.75, # Neutral pass-through for testing
        "latency_ms": existing_latency,
    }

# NODE: reflect_node

def reflect_node(state: AgentState) -> dict:
    """
       STUB: Full implementation on Day 24.
       Self-evaluates draft on structure, depth, grounding. Scores 1-10.

       For now: passes with score 8 so route_after_reflect sends to hitl.
       """
    print("[reflect_node] STUB — passing through with score 8")
    existing_latency = state.get("latency_ms", {})
    existing_latency["reflect"] = 0
    return {
        "reflection_score": 8,
        "reflection_notes": "STUB — not yet implemented",
        "latency_ms": existing_latency,
    }

# NODE: hitl_node

def hitl_node(state: AgentState) -> dict:
    """
    STUB: Full implementation on Day 26.
    Displays draft + grounding report to user. Waits for approve/reject/feedback.

    For now: auto-approves so you can test the full graph end-to-end.
    """
    print("\n[hitl_node] STUB — auto-approving draft")
    print(f"\n{'═'*60}")
    print(f"DRAFT PREVIEW: {state['topic']}")
    print(f"{'═'*60}")
    print(state.get("draft_markdown", "")[:800])
    print(f"\n... (truncated for stub) ...")
    print(f"{'═'*60}\n")
    return {
        "hitl_status": "approved",
        "hitl_feedback": None,
    }

# NODE: html_gen_node

def html_gen_node(state: AgentState) -> dict:
    """
    STUB: Full implementation on Day 25.
    Generates themachinist.org-compliant HTML from draft_sections.

    For now: writes a minimal placeholder HTML file to outputs/.
    """
    slug = state.get("slug", "draft")
    filename = f"{slug}.html"
    placeholder = f"<!-- HTML stub for {state['topic']} — Day 25 implementation -->"

    out_path = Path("outputs") / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(placeholder, encoding="utf-8")

    print(f"[html_gen_node] STUB — wrote placeholder to {out_path}")
    existing_latency = state.get("latency_ms", {})
    existing_latency["html_gen"] = 0
    return {
        "html_output": placeholder,
        "html_filename": filename,
        "latency_ms": existing_latency,
    }

# NODE: git_node

def git_node(state: AgentState) -> dict:
    """
    STUB: Full implementation on Day 26.
    Creates feature branch, pushes HTML, diffs vs main, merges or tags.

    For now: logs what it would do without touching any repo.
    """
    slug = state.get("slug", "draft")
    branch = f"feature/article-{slug}"
    print(f"[git_node] STUB — would push {state.get('html_filename')} to branch {branch}")
    existing_latency = state.get("latency_ms", {})
    existing_latency["git"] = 0
    return {
        "branch_name": branch,
        "git_status": "not_started",
        "latency_ms": existing_latency,
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
    iterations = state.get("iterations", 0)
    reflection_score = state.get("reflection_score", 8)
    grounding_score = state.get("grounding_score", 0.75)

    # Hard ceiling: never loop more than MAX_ITERATIONS
    if iterations >= MAX_ITERATIONS:
        print(f"[route_after_reflect] Max iterations ({MAX_ITERATIONS}) reached — proceeding to HITL")
        return "hitl"

    # Cost gate check
    if state.get("total_cost_usd", 0) >= COST_GATE_USD:
        print(f"[route_after_reflect] Cost gate hit (${state['total_cost_usd']:.4f}) — proceeding to HITL")
        return "hitl"

    # Composite gate
    hard_floor_fail = grounding_score < GROUNDING_FLOOR
    soft_fail = reflection_score < REFLECTION_THRESHOLD and grounding_score < 0.75

    if hard_floor_fail or soft_fail:
        reason = "grounding below floor" if hard_floor_fail else "reflection + grounding both weak"
        print(f"[route_after_reflect] Revising draft — {reason} (reflection={reflection_score}, grounding={grounding_score:.2f})")
        return "draft"

    print(f"[route_after_reflect] Proceeding to HITL (reflection={reflection_score}, grounding={grounding_score:.2f})")
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
    status = state.get("hitl_status", "pending")

    if status == "approved":
        return "html_gen"
    elif status == "feedback":
        return "draft"
    else:   # rejected or unknown
        return END


