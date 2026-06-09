"""tmp/m2_reverify_probe.py — Diagnostic C: self‑contained, generates drafts + reverifies."""
import json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Config ───────────────────────────────────────────────
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "machinist_evergreen")
CACHE_DIR = Path("outputs/m2/tavily_cache.frozen")

CURRENT_PROMPT = Path("prompts/verify_system.md").read_text(encoding="utf-8")
LENIENT_PROMPT = Path("prompts/verify_system_lenient.md").read_text(encoding="utf-8")
DRAFT_SYSTEM_PROMPT = Path("prompts/draft_system.md").read_text(encoding="utf-8")
# strip # comment lines from draft prompt (same as _load_system_prompt)
DRAFT_SYSTEM_PROMPT = "\n".join(l for l in DRAFT_SYSTEM_PROMPT.splitlines() if not l.startswith("#"))

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=BASE_URL)

# ── KB setup (same as query_kb.py) ─────────────────────
encoder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL)

# ── Helper: build source context ───────────────────────
def build_source_context(web_sources, kb_results):
    parts = []
    for s in (web_sources or [])[:5]:
        parts.append(f"[WEB] {s['url']}\n{s.get('content','')[:200]}")
    for k in (kb_results or [])[:3]:
        parts.append(f"[KB] {k['source']}\n{k.get('text','')[:200]}")
    return "\n\n".join(parts) if parts else "No sources available."

# ── Retrieve web (Tavily) ──────────────────────────────
def search_web(query):
    """Read cached Tavily results for the given query."""
    results = []
    for f in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                content = item.get("content", "")
                if query.lower() in content.lower() or query.lower() in item.get("url","").lower():
                    results.append({
                        "url": item.get("url", ""),
                        "content": content,
                        "score": item.get("score", 0),
                    })
        except:
            pass
    # deduplicate by URL, sort by score desc
    seen = set()
    uniq = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        if r["url"] not in seen:
            seen.add(r["url"])
            uniq.append(r)
    return uniq[:10]

# ── Retrieve KB ────────────────────────────────────────
def search_kb(query, n=5):
    if qdrant.get_collection(QDRANT_COLLECTION).points_count == 0:
        return []
    vec = encoder.encode(query).tolist()
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=n,
        with_payload=True,
    )
    results = []
    for h in hits:
        p = h.payload or {}
        results.append({"source": p.get("source","?"), "text": p.get("text","")})
    return results

# ── Draft ──────────────────────────────────────────────
def generate_draft(topic, series, card, web_sources, kb_results):
    source_context = build_source_context(web_sources, kb_results)
    source_block = (
        f"\n\nGROUNDING SOURCES (retrieved for this topic):\n{source_context}\n\n"
        "Ground the article in these sources. Assert only what the sources support. "
        "Synthesize in your own words — do not copy source sentences. "
        "If you cannot support a point from the sources, omit it rather than guessing."
    )
    user_msg = f"""Write a technical article for The Machinist on the following topic.

Topic: {topic}
Card ID: {card}
Series context: {series}

This article will be published on themachinist.org under the Learning Log section.
The audience is engineers learning ML and agentic AI — they are smart but new to this specific topic.
{source_block}

Return ONLY the JSON object as specified in your instructions. No markdown wrapper."""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":DRAFT_SYSTEM_PROMPT},{"role":"user","content":user_msg}],
        temperature=0.3,
        max_tokens=4000,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    draft_json = json.loads(raw)
    # assemble markdown (simplified)
    md = f"# {topic}\n\n## Problem Framing\n{draft_json.get('problem_framing','')}\n\n## Technical Deep-Dive\n{draft_json.get('technical_dive','')}\n\n## Code\n{draft_json.get('code_snippets','')}\n\n## Takeaways\n{draft_json.get('takeaways','')}"
    return md, draft_json

# ── Verify ─────────────────────────────────────────────
def verify(draft_md, web, kb, system_prompt):
    source_ctx = build_source_context(web, kb)
    user_msg = f"""Draft to verify:
{draft_md}

Available sources:
{source_ctx}

Return a JSON array. Each element:
{{"claim": "...", "source_url": "..." or null, "confidence": 0.0-1.0,
  "status": "verified" | "weak" | "unverified"}}

Return ONLY the JSON array. No preamble."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_msg}],
        temperature=0.1,
        max_tokens=4000,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        claims = json.loads(raw)
        if not isinstance(claims, list):
            print("    WARNING: expected list")
            claims = []
    except json.JSONDecodeError as e:
        print(f"    PARSE ERROR: {e}")
        claims = []
    cost = (resp.usage.prompt_tokens/1e6*0.27)+(resp.usage.completion_tokens/1e6*1.10)
    return claims, cost

def uvr(claims):
    v = sum(1 for c in claims if c.get("status")=="verified")
    w = sum(1 for c in claims if c.get("status")=="weak")
    u = sum(1 for c in claims if c.get("status")=="unverified")
    total = v+w+u
    if total==0: return 0.0,0
    return u/total, total

# ── Main ────────────────────────────────────────────────
TOPICS = [
    ("Multi-Agent Systems — When and Why", "standalone", "Agentic AI"),
    ("Embedding Models & Vector Search", "standalone", "Concept Exploration"),
    ("ReAct Agent Pattern", "standalone", "Agentic AI"),
]

total_cost = 0.0
for topic, card, series in TOPICS:
    print(f"\n{'='*60}\n{topic}\n{'='*60}")
    # 1. Retrieve
    web = search_web(topic)
    kb = search_kb(topic)
    print(f"  web sources: {len(web)}, kb chunks: {len(kb)}")

    # 2. Draft
    draft_md, _ = generate_draft(topic, series, card, web, kb)
    print(f"  draft length: {len(draft_md)} chars")

    # 3. Verify with current and lenient prompts
    claims_cur, cost_cur = verify(draft_md, web, kb, CURRENT_PROMPT)
    claims_len, cost_len = verify(draft_md, web, kb, LENIENT_PROMPT)
    total_cost += cost_cur + cost_len

    uvr_cur, n_cur = uvr(claims_cur)
    uvr_len, n_len = uvr(claims_len)
    print(f"  CURRENT: {n_cur} claims, UVR={uvr_cur:.2f}")
    print(f"  LENIENT: {n_len} claims, UVR={uvr_len:.2f}")
    print(f"  Δ UVR = {uvr_cur - uvr_len:+.2f}")
    print(f"  (cost this topic: ${cost_cur+cost_len:.4f})")

print(f"\nTotal cost: ${total_cost:.4f}")