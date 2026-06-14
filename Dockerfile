# content-agent API image. Non-root, embedding model baked in for offline warmup.
FROM python:3.14-slim-bookworm

# uv from the official image (robust across python base tags)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# git: GitPython needs the binary; B6 supervised publish will use it. Dry-run skips it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first (cached unless lock changes). --no-dev drops pytest/ruff.
# --frozen fails the build if uv.lock is stale (the B3 "venv hides missing deps" lesson,
# enforced at build time).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Bake the embedding model so M5 warmup is offline and the first run is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Source last (changes most often)
COPY . .

# Non-root. uid 10001 must own /app (incl. baked model cache) to write outputs/checkpoints.
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# /health is unauthenticated by design (B4) — safe for the container healthcheck.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["python", "main.py", "serve"]