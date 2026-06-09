"""tmp/verifier_unit_test.py — does verify_node correctly judge claims vs a KNOWN source?

No cache dependency. ~8 LLM calls (~$0.02). Each claim verified alone to isolate scoring.
Uses the production verify prompt and client so the result reflects the real verifier.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.nodes import _get_client, _llm_call, DEEPSEEK_MODEL

VERIFY_SYSTEM = Path("prompts/verify_system.md").read_text(encoding="utf-8")

# Paste ~2 real paragraphs that genuinely contain the facts your "supported" claims assert.
# Use an actual KB seed doc excerpt, e.g. kb/seed_docs/multi-agent-systems.md, so the test is real.
SOURCE = """Failure Modes:
- Coordination failure: Agents produce conflicting plans or deadlock on decisions.
- Infinite chatter: Prompt-driven loops without convergence.
- Hallucinated agreements: Agents pretend to agree without true coordination.
- Cost blowup: Multiple LLM calls per step compound inference expense.
- Information cascade: Early errors propagate when agents over-rely on peers.

When to Use / Not Use:
- Use when a task naturally decomposes into parallel or role-based subproblems (code generation + review, multi-perspective analysis).
- Use when error correction via cross-checking is valuable.
- Do NOT use for simple single-step queries.
- Do NOT use for latency-sensitive pipelines where serial agent chaining adds unacceptable delay.
- Do NOT use when the added complexity of coordination outweighs accuracy gains."""

# (kind, expected) where expected in {"supported","unsupported"}.
# Write 4 supported (2 near-verbatim from SOURCE, 2 paraphrased) and 4 unsupported
# (2 plausible-but-absent, 2 clearly not in SOURCE).
CLAIMS = [
    # ── Supported, near‑verbatim (2) ──
    ("verbatim",   "supported",
     "Coordination failure happens when agents produce conflicting plans or deadlock on decisions."),
    ("verbatim",   "supported",
     "Multi-agent systems should not be used for simple single-step queries."),

    # ── Supported, paraphrased (2) ──
    ("paraphrase", "supported",
     "When agents keep talking without reaching a conclusion, the system can get stuck in an infinite chatter loop."),
    ("paraphrase", "supported",
     "One downside of multi-agent setups is that the inference cost can blow up because each step may call multiple LLMs."),

    # ── Plausible but NOT in the source (2) ──
    ("absent",     "unsupported",
     "Multi-agent systems require a dedicated message broker like RabbitMQ to function reliably."),
    ("absent",     "unsupported",
     "Agents in a multi-agent system should always use a round‑robin scheduling algorithm for fairness."),

    # ── Clearly NOT in the source (2) ──
    ("false",      "unsupported",
     "Multi-agent systems are limited to a maximum of 10 agents because of exponential state explosion."),
    ("false",      "unsupported",
     "The first multi-agent system was built by Alan Turing in 1950."),
]

client = _get_client()

def verify_one(claim: str) -> dict:
    user = (
        "Draft to verify:\n" + claim + "\n\n"
        "Available sources:\n[WEB] test://source\n" + SOURCE + "\n\n"
        'Return a JSON array. Each element:\n'
        '{"claim":"...","source_url":"..." or null,"confidence":0.0-1.0,'
        '"status":"verified"|"weak"|"unverified"}\n'
        "Return ONLY the JSON array. No preamble."
    )
    r = _llm_call(client, model=DEEPSEEK_MODEL,
                  messages=[{"role":"system","content":VERIFY_SYSTEM},
                            {"role":"user","content":user}],
                  temperature=0.1, max_tokens=500)
    raw = r.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        arr = json.loads(raw)
        return arr[0] if isinstance(arr, list) and arr else {"status":"EMPTY","raw":raw[:120]}
    except Exception as e:
        return {"status":f"PARSE_FAIL", "err":str(e), "raw":raw[:120]}

print(f"{'kind':<11}{'expected':<12}{'got':<12}{'conf':<6}{'src?':<5} claim")
print("-"*80)
ok = 0
for kind, expected, claim in CLAIMS:
    v = verify_one(claim)
    got = v.get("status","?")
    passed = ((expected=="supported" and got in ("verified","weak")) or
              (expected=="unsupported" and got=="unverified"))
    ok += int(passed)
    print(f"{kind:<11}{expected:<12}{got:<12}{str(v.get('confidence','')):<6}"
          f"{('Y' if v.get('source_url') else 'n'):<5} {claim[:46]}")
print("-"*80)
print(f"verifier accuracy on known ground truth: {ok}/8")
print("\nREAD:")
print("  paraphrase-supported -> 'unverified'  => verifier is PARAPHRASE-BLIND (rubric fix).")
print("  verbatim-supported   -> 'unverified'  => verifier CORE broken (matching/parse).")
print("  all 8 correct                          => verifier is FINE; problem is sources/extraction/over-claiming.")