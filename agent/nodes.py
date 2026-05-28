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
from observability.logger import get_logger
import html as html_module
import re
import datetime

VERIFY_SYSTEM = Path("prompts/verify_system.md").read_text()
REFLECT_SYSTEM = Path("prompts/reflect_system.md").read_text()

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
    log = get_logger("draft_node")
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

    log.info(
        "retrieve.complete",
        run_id=state["run_id"],
        web_sources=len(web_sources),
        kb_results=len(kb_results),
        latency_ms=latency
    )


    return {
        "web_sources": web_sources,
        "kb_results": kb_results,
        "latency_ms": existing_latency,
    }


def _build_source_context(web_sources: list, kb_results: list) -> str:
    """Format sources into a compact string for the verify prompt."""
    parts = []
    for s in web_sources[:5]:
        parts.append(f"[WEB] {s['url']}\n{s['content'][:500]}")
    for k in kb_results[:5]:
        parts.append(f"[KB] {k['source']}\n{k['text'][:800]}")
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

    claim_response = client.chat.completions.create(
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
                              "status": "verified" | "weak" | "unverified"}}
                            
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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        grounding_report = json.loads(raw)
        if not isinstance(grounding_report, list):
            raise ValueError("Expected list")
    except (json.JSONDecodeError, ValueError) as e:
        log.error("verify.parse_failed", run_id=state["run_id"], error=str(e))
        grounding_report = []

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


    log.info("verify.complete",
             run_id=state["run_id"],
             grounding_score=round(grounding_score, 3),
             claims=len(grounding_report),
             verfied=n_verified,
             weak=n_weak,
             unverified=n_unverified,
             latency_ms=latency,
             cost=round(run_cost, 5),
             )

    return {
        "grounding_report": grounding_report,
        "grounding_score": round(grounding_score, 3),
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

    response = client.chat.completions.create(
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

    response = client.chat.completions.create(
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

    base_filename = f"{state['slug']}.html"
    out_path = Path("outputs/articles") / base_filename
    if out_path.exists():
        # Append first 6 chars of run_id to make it unique
        suffix = state["run_id"].split("-")[-1][:6]
        filename = f"{state['slug']}-{suffix}.html"
    else:
        filename = base_filename

    out_path = Path("outputs/articles") / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    existing_latency = state.get("latency_ms", {})
    existing_latency["html_gen"] = latency

    error_log = state.get("error_log", [])
    if errors:
        error_log.append(f"[html_gen] Validation warnings: {errors}")

    log.info("html_gen.complete", run_id=run_id, filename=filename,
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

        # 2. Create and checkout feature branch
        if branch in repo.heads:
            repo.delete_head(branch, force=True)

        new_branch = repo.create_head(branch)
        new_branch.checkout()

        # 3. Stage and commit
        repo.index.add([filename])
        commit_msg = f"feat: add {topic} article [content-agent]"
        commit = repo.index.commit(commit_msg)

        # 4. Diff vs main to decide merge strategy
        main_commit = repo.commit("main")
        diff_index = main_commit.diff(commit)

        changed_files = [d.b_path for d in diff_index if d.change_type in ("M", "D", "R")]
        new_files = [d.b_path for d in diff_index if d.change_type == "A"]

        # 5. Merge or tag-then-merge
        if changed_files:
            date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
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


