"""tmp/verifier_unit_test.py — does verify_node correctly judge claims vs a KNOWN source?

No cache dependency. ~8 LLM calls (~$0.02). Each claim verified alone to isolate scoring.
Uses the production verify prompt and client so the result reflects the real verifier.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.nodes import _get_client, _llm_call, DEEPSEEK_MODEL, _extract_json_array

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
# (kind, expected_status, expected_specificity, claim)
# expected_status in {"supported","unsupported"}; expected_specificity in {"substantive","generic"}
# 2x2 grid: specificity x grounding, so both classifiers are calibrated.
CLAIMS = [
    # ── substantive + supported (verbatim) ──
    ("verbatim", "supported", "substantive",
     "Coordination failure happens when agents produce conflicting plans or deadlock on decisions."),
    ("verbatim", "supported", "substantive",
     "Multi-agent systems should not be used for simple single-step queries."),
    # ── substantive + supported (paraphrase) ──
    ("paraphrase", "supported", "substantive",
     "When agents keep talking without reaching a conclusion, the system can get stuck in an infinite chatter loop."),
    ("paraphrase", "supported", "substantive",
     "One downside of multi-agent setups is that inference cost can blow up because each step may call multiple LLMs."),
    # ── substantive + unsupported (plausible but absent) ──
    ("absent", "unsupported", "substantive",
     "Multi-agent systems require a dedicated message broker like RabbitMQ to function reliably."),
    ("absent", "unsupported", "substantive",
     "Agents in a multi-agent system should always use a round-robin scheduling algorithm for fairness."),
    # ── substantive + unsupported (clearly false) ──
    ("false", "unsupported", "substantive",
     "Multi-agent systems are limited to a maximum of 10 agents because of exponential state explosion."),
    ("false", "unsupported", "substantive",
     "The first multi-agent system was built by Alan Turing in 1950."),
    # ── generic + supported (vague restatement of source content) ──
    ("generic_sup", "supported", "generic",
     "Multi-agent systems can be costly to run."),
    ("generic_sup", "supported", "generic",
     "There are situations where multi-agent systems are not a good fit."),
    # ── generic + unsupported (vague background not in source) ──
    ("generic_unsup", "unsupported", "generic",
     "Multi-agent systems are widely used in modern software."),
    ("generic_unsup", "unsupported", "generic",
     "Multi-agent systems are an important area of AI research."),
]

client = _get_client()

def verify_one(claim: str) -> dict:
    user = (
        "Draft to verify:\n" + claim + "\n\n"
        "Available sources:\n[WEB] test://source\n" + SOURCE + "\n\n"
        'Return a JSON array. Each element:\n'
        '{"claim":"...","source_url":"..." or null,"confidence":0.0-1.0,'
        '"status":"verified"|"weak"|"unverified","specificity":"substantive"|"generic"}\n'
        "Return ONLY the JSON array. No preamble."
    )
    r = _llm_call(client, model=DEEPSEEK_MODEL,
                  messages=[{"role":"system","content":VERIFY_SYSTEM},
                            {"role":"user","content":user}],
                  temperature=0.1, max_tokens=500)
    raw = r.choices[0].message.content.strip()
    try:
        arr = _extract_json_array(raw)
        return arr[0] if arr else {"status": "EMPTY", "raw": raw[:500]}
    except Exception as e:
        return {"status": "PARSE_FAIL", "err": str(e), "raw": raw[:500]}



print(f"{'kind':<14}{'exp_status':<11}{'got_status':<11}{'exp_spec':<12}{'got_spec':<12} claim")
print("-" * 100)
ground_ok = 0
spec_ok = 0
for kind, exp_status, exp_spec, claim in CLAIMS:
    v = verify_one(claim)
    got_status = v.get("status", "?")
    got_spec = v.get("specificity", "?")
    g_pass = ((exp_status == "supported" and got_status in ("verified", "weak")) or
              (exp_status == "unsupported" and got_status == "unverified"))
    s_pass = (got_spec == exp_spec)
    ground_ok += int(g_pass)
    spec_ok += int(s_pass)
    print(f"{kind:<14}{exp_status:<11}{got_status:<11}{exp_spec:<12}{got_spec:<12} {claim[:38]}")
    if got_status in ("PARSE_FAIL", "EMPTY"):
        print(f"    err: {v.get('err')}")
        print(f"    raw: {v.get('raw')!r}")
print("-" * 100)
print(f"grounding accuracy:   {ground_ok}/{len(CLAIMS)}")
print(f"specificity accuracy: {spec_ok}/{len(CLAIMS)}")
print("\nGATE TO PROCEED: grounding >= 11/12 AND specificity >= 10/12.")
print("If specificity < 10/12, SV is NOT trustworthy — fix the specificity definition in")
print("verify_system.md and re-run before any experiment.")

sys.exit(0 if (ground_ok >= 11 and spec_ok >= 10) else 1)