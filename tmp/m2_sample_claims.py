"""tmp/m2_sample_claims.py — pull unverified/weak claims from M2 TREATMENT runs for human review."""
import json, random
from pathlib import Path

RUNS = Path("outputs/runs")
TREATMENT = Path("outputs/m2/treatment_ids.txt")
N = 30
random.seed(7)

ids = [l.strip() for l in TREATMENT.read_text().splitlines() if l.strip()]
pool = []
for rid in ids:
    f = RUNS / f"{rid}.json"
    if not f.exists(): continue
    t = json.loads(f.read_text())
    topic = t.get("topic", "?")
    for r in t.get("grounding_report", []):
        if r.get("status") in ("unverified", "weak"):
            pool.append({"topic": topic, "run": rid,
                         "claim": r.get("claim",""), "status": r.get("status"),
                         "confidence": r.get("confidence"), "source_url": r.get("source_url")})

sample = random.sample(pool, min(N, len(pool)))
lines = ["# M2 claim review (human adjudication)\n",
         "For each claim, open the source_url (or check the frozen cache) and fill the last 2 columns.\n",
         "supported = is the claim's substance actually in the provided source(s)? yes / partial / no\n",
         "in_sources = does ANY retrieved source for this topic contain the info? yes / no\n",
         f"\nPool size: {len(pool)} unverified+weak claims. Sample: {len(sample)}.\n",
         "\n| # | topic | status | conf | claim | source_url | supported | in_sources |",
         "|---|---|---|---|---|---|---|---|"]
for i, s in enumerate(sample, 1):
    claim = s["claim"].replace("|"," ")[:160]
    url = (s["source_url"] or "—")
    lines.append(f"| {i} | {s['topic'][:18]} | {s['status']} | {s['confidence']} | {claim} | {url} |  |  |")
Path("tmp/m2_claim_review.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote tmp/m2_claim_review.md with {len(sample)} claims for review.")