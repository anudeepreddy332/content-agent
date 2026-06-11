import json, re
from pathlib import Path
RUNS = Path("outputs/runs")
F = re.compile(r"=|sigmoid|logistic function|log-?odds|logit|odds ratio|cross-?entropy|"
  r"log loss|maximum likelihood|least squares|\bOLS\b|normal equation|gradient|"
  r"partial derivative|cost function|loss function|coefficient|intercept|"
  r"decision boundary|weight update|learning rate", re.I)
def run(idfile):
    rs, ab = [], []
    for rid in Path(idfile).read_text().split():
        t = json.loads((RUNS/f"{rid}.json").read_text())
        rep = t.get("grounding_report", [])
        f = [c for c in rep if F.search(c.get("claim",""))]
        rec = [c for c in f if c.get("status")=="verified"]
        print(f"{rid[:8]} retr={t['latency_ms'].get('retrieve')} |F|={len(f)} "
              f"rec={len(rec)} rate={len(rec)/max(len(f),1):.2f} "
              f"has_src={t['grounding_breakdown']['unverified_has_source']} g={t['grounding_score']}")
        rs.append(len(rec)/max(len(f),1)); ab.append(len(rec))
    print(f"  MEAN rate={sum(rs)/len(rs):.2f} abs={sum(ab)/len(ab):.1f}\n")
print("CONTROL");   run("/tmp/r1_control_ids.txt")
print("TREATMENT"); run("/tmp/r1_treatment_ids.txt")