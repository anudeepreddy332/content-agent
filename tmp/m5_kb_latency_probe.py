""" M5 probe: decompose KB query latency into import / model-load / encode / qdrant / bm25."""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

t0 = time.perf_counter()
from sentence_transformers import SentenceTransformer
t1 = time.perf_counter()

encoder = SentenceTransformer("all-MiniLM-L6-v2")
t2 = time.perf_counter()

encoder.encode("warmup")    # first encode includes lazy torch init
t3 = time.perf_counter()

import tools.query_kb as qk
qk._encoder = encoder       # inject the pre-loaded model into the module singleton

from tools.query_kb import query_kb
r1 = query_kb("CatBoost categorical gradient boosting", n_results=5)  # cold: BM25 build + (already-loaded) encoder
t4 = time.perf_counter()

r2 = query_kb("Ridge regression regularization", n_results=5)  # warm: steady-state per-query cost
t5 = time.perf_counter()

print(f"import sentence_transformers : {(t1-t0)*1000:8.0f} ms")
print(f"model load (cold)            : {(t2-t1)*1000:8.0f} ms")
print(f"first encode (torch init)    : {(t3-t2)*1000:8.0f} ms")
print(f"query_kb cold (results={len(r1)})  : {(t4-t3)*1000:8.0f} ms")
print(f"query_kb warm (results={len(r2)})  : {(t5-t4)*1000:8.0f} ms")