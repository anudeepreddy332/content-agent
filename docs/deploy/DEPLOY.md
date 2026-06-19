# content-agent — single-VM deployment runbook

## Provision (Ubuntu 22.04/24.04 VM, 2 vCPU / 4 GB+)
1. Install Docker Engine + compose plugin (docker.com official script).
2. `git clone <repo> && cd content-agent && git checkout main`
3. `cp .env.example .env`, fill in DEEPSEEK_API_KEY, TAVILY_API_KEY, and
   `API_BEARER_TOKEN` (generate: `python3 -c "import secrets;print(secrets.token_urlsafe(32))"`).
   `chmod 600 .env`.
4. `mkdir -p outputs && sudo chown -R 10001:10001 outputs`
   (the image runs as uid 10001; the bind mount must be writable by it).

## Build, ingest, run
    docker compose -f docker-compose.prod.yml build
    docker compose -f docker-compose.prod.yml up -d qdrant          # start Qdrant first
    docker compose -f docker-compose.prod.yml run --rm app python scripts/ingest.py --source kb/seed_docs/
    docker compose -f docker-compose.prod.yml up -d                 # start the API
    docker compose -f docker-compose.prod.yml ps                    # both should be (healthy)

`ingest` runs inside the compose network, so it reaches `qdrant:6333` even though Qdrant
has no host port. The vector store persists in the `qdrant_data` named volume across restarts.

## Access (no public exposure)
The API binds to 127.0.0.1:8000 on the VM. From your laptop:
    ssh -L 8000:localhost:8000 user@vm-host
    # then locally:
    T=<API_BEARER_TOKEN>
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/runs -H "Authorization: Bearer $T" \
         -H 'Content-Type: application/json' -d '{"topic":"Gradient Descent"}'
    curl -s localhost:8000/runs/<run_id> -H "Authorization: Bearer $T"
    curl -s -X POST localhost:8000/runs/<run_id>/approve -H "Authorization: Bearer $T"

Do NOT open port 8000 in the VM firewall for the freeze. TLS + reverse proxy is post-freeze.

## Security posture (freeze)
- No secrets in the image: .env is dockerignored; injected at runtime via env_file.
- Qdrant network-isolated: no host port; only the app container reaches it.
- GIT_PUSH_ENABLED=false: git_node logs intent and skips before touching any repo path.
  Live publishing is enabled only in the B6/supervised-publish step.
- API bearer auth, fail-closed; loopback bind + SSH tunnel.

## B4 durability limitation (carried forward — KNOWN, documented)
The SqliteSaver checkpoint at outputs/checkpoints.sqlite persists full graph state and lives
on the ./outputs bind mount, so it SURVIVES a container restart. However, the API's in-memory
run REGISTRY does NOT survive restart. Consequence: after `docker compose restart app`, a run
that was awaiting_review returns 404 from GET /runs/{id} even though its checkpoint exists.
  - Recovery today: the article's state is intact in checkpoints.sqlite and can be resumed
    by a process holding the same thread_id; the HTTP wrapper has lost the run_id mapping.
  - Mitigation for the freeze: avoid restarting the app while a run is awaiting human review.
    Approve/reject pending runs before any deploy or restart.
  - Post-freeze fix (B9-adjacent): rehydrate REGISTRY from the checkpointer on startup
    (list thread_ids with pending interrupts -> repopulate awaiting_review entries).

## Operations
- Logs: `docker compose -f docker-compose.prod.yml logs -f app` (structlog JSON to stdout —
  this IS the log sink; a platform collector ingests it).
- Telemetry: outputs/runs/<run_id>.json on the bind mount.
- Update: `git pull && docker compose -f docker-compose.prod.yml build && ... up -d`
  (drain awaiting_review runs first — see durability limitation).
- Rollback: `git checkout <prev tag>` + rebuild; or `docker compose down && up -d` on the
  previously built image. Full rollback script lands in the B6–B8 validation day.