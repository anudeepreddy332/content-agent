# content-agent — public demo deployment runbook (EC2 + Caddy + fork publish)

This deploys the SAME container as DEPLOY.md, with two additions: a Caddy reverse proxy for
TLS + a public port, and a live "Publish to live" path that pushes a FORK of
themachinist-website (never production) so Netlify redeploys it. git_node is unchanged — it
still only does a local merge inside the fork clone; the push is a separate, human-triggered
step (`POST /ui/runs/{id}/publish`, wired to the SPA's Publish button).

## 1. Provision the EC2 instance
- Ubuntu 22.04/24.04, 2 vCPU / 4 GB+ (same sizing as DEPLOY.md).
- Security group:
  - port 22 (SSH) — restrict to YOUR IP only.
  - port 80 and port 443 — open to the public (0.0.0.0/0). This is the only public surface;
    the app container itself is never port-mapped (see step 5).
- Install Docker Engine + the compose plugin (docker.com official install script).

## 2. Clone content-agent and configure secrets
    git clone <repo> && cd content-agent && git checkout main
    cp .env.example .env
    # fill in DEEPSEEK_API_KEY, TAVILY_API_KEY, API_BEARER_TOKEN (generate:
    #   python3 -c "import secrets;print(secrets.token_urlsafe(32))")
    chmod 600 .env
    mkdir -p outputs && sudo chown -R 10001:10001 outputs   # image runs as uid 10001

## 3. Clone the FORK for the demo's publish target
    git clone git@github.com:anudeepreddy332/themachinist-website-fork.git fork-clone
    cd fork-clone
    git config user.name  "content-agent demo"
    git config user.email "demo@content-agent.local"     # commits made by git_node need this
    cd ..
    sudo chown -R 10001:10001 fork-clone                  # container (uid 10001) must write here

The fork's `origin` remote must use a credential the VM can push with non-interactively
(SSH deploy key with write access, or an HTTPS remote with a fine-grained PAT embedded in the
URL). Set this up on `fork-clone/.git/config` directly; do not put credentials in `.env` or
any compose file.

## 4. Point DNS at the box
Easiest path: skip real DNS and use the free wildcard service `sslip.io`, which resolves
`<anything>.<ip-with-dashes>.sslip.io` to that IP automatically — Caddy can then get a real
Let's Encrypt cert with no DNS setup.

    export DEMO_DOMAIN=$(curl -s ifconfig.me | tr '.' '-').sslip.io
    echo "DEMO_DOMAIN=$DEMO_DOMAIN" >> .env

## 5. Build and start

### Primary path (what this demo actually runs today): pull a pre-built image
Building on a small EC2 instance is slow (the `sentence-transformers` baking step alone takes
minutes) and ties up the box every deploy. The path actually used in production for this demo
is: build once on a fast machine, push to Docker Hub, and have the EC2 box only ever `pull`.

**On your laptop** (or any machine with Docker buildx; match `--platform` to the EC2 instance's
CPU architecture — `linux/arm64` for Graviton instances like `t4g.*`, `linux/amd64` otherwise):

    docker buildx build --platform linux/arm64 -t anudeepreddy332/content-agent:demo --push .

This builds and pushes to Docker Hub (`docker.io/anudeepreddy332/content-agent:demo`) in one
step — `docker login` once beforehand if you haven't.

**On the EC2 box**, create `docker-compose.image.yml` (not committed to this repo — it's a
deploy-time artifact, since it hardcodes the registry image tag rather than building from the
checked-out source):

    services:
      app:
        image: anudeepreddy332/content-agent:demo
        pull_policy: always

This is layered on top of `docker-compose.prod.yml` + `docker-compose.demo.yml` exactly like
any other override — it replaces the `app` service's `build: .` with a registry pull and
forces a fresh pull on every `up` (`pull_policy: always`) so re-running step 5 always gets the
latest pushed tag instead of silently reusing a stale local image:

    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml -f docker-compose.image.yml pull
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml -f docker-compose.image.yml up -d qdrant
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml -f docker-compose.image.yml run --rm app \
        python scripts/ingest.py --source kb/seed_docs/
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml -f docker-compose.image.yml up -d
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml -f docker-compose.image.yml ps

Re-deploying a code change with this path is: rebuild+push on your laptop, then on the box
just re-run the `pull` + `up -d` lines above — no git clone/checkout needed on the EC2 box at
all for the `content-agent` source itself (you still need `Caddyfile`, the compose files, and
`.env`/`fork-clone` present, per steps 1-4, since those aren't baked into the image).

### Alternative path: build from source on the VM
This is what `DEPLOY.md` describes for the non-demo deployment, and it still works here too —
useful if you don't want a Docker Hub dependency, or you're iterating directly on the VM:

    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml build
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml up -d qdrant
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml run --rm app \
        python scripts/ingest.py --source kb/seed_docs/
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml up -d
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml ps

Slower per-deploy (rebuilds on the EC2 box itself) and requires a full `git clone` of this repo
on the box; no registry account needed.

### Either path
`docker-compose.demo.yml` REMOVES the app's `127.0.0.1:8000` host port binding (via the
compose `!reset` merge tag — requires Compose v2.24+; `docker compose version` to check) and
adds a `caddy` service bound to 80/443 that reverse-proxies to `app:8000` over the internal
compose network. The API is never directly reachable from the public internet — only through
Caddy. If your compose version doesn't support `!reset`, drop the `ports:` block from
docker-compose.prod.yml's `app` service for this deployment instead.

Visit `https://$DEMO_DOMAIN/` — Caddy auto-provisions the TLS cert on first request (give it
10-20s).

## 6. Smoke-test the live demo
    T=<API_BEARER_TOKEN>
    curl -s https://$DEMO_DOMAIN/health
    curl -s -X POST https://$DEMO_DOMAIN/ui/runs -H "Authorization: Bearer $T" \
         -H 'Content-Type: application/json' -d '{"topic":"Gradient Descent"}'
Then open `https://$DEMO_DOMAIN/` in a browser, paste the token, and run a topic through both
gates. After gate 2 is approved, the "Publish to live" button calls
`POST /ui/runs/{id}/publish`, which runs `git push $PUBLISH_REMOTE main` inside `/app/fork`
(the bind-mounted `fork-clone`) and returns the live URL.

## Local rehearsal before touching the VM (test against ~/tmp/tmw-fork)
Run this on your laptop first — same code path, no EC2/Caddy needed, against the same fork
used in the earlier manual rehearsal:

    export PUBLISH_TARGET=demo            # required; unset disables merge and push
    export GIT_PUSH_ENABLED=true
    export THEMACHINIST_REPO_PATH=$HOME/tmp/tmw-fork   # fork clone only, never production
    export PUBLISH_REMOTE=demo            # add a remote named "demo" pointing at the fork,
                                           # or use "origin" if that's already the fork's remote
    export NETLIFY_BASE_URL=https://tmw-demo-site.netlify.app
    uv run python main.py serve --host 127.0.0.1 --port 8099

    # in another terminal
    T=$(grep -E '^API_BEARER_TOKEN=' .env | cut -d= -f2-)
    RID=$(curl -s -X POST http://localhost:8099/ui/runs -H "Authorization: Bearer $T" \
          -H 'Content-Type: application/json' -d '{"topic":"Test topic"}' | python3 -c \
          'import sys,json;print(json.load(sys.stdin)["run_id"])')
    # drain SSE at http://127.0.0.1:8099/ui/runs/$RID/events (Authorization: Bearer header;
    # query-string tokens are rejected), approve both gates via
    #   curl -s -X POST http://127.0.0.1:8099/ui/runs/$RID/approve -H "Authorization: Bearer $T"
    # once git_status is merged/tagged_and_merged:
    curl -s -X POST http://localhost:8099/ui/runs/$RID/publish -H "Authorization: Bearer $T"

If `PUBLISH_REMOTE=demo` doesn't exist yet on the fork clone:
    git -C ~/tmp/tmw-fork remote add demo git@github.com:anudeepreddy332/themachinist-website-fork.git

## Uptime monitoring (manual, infra-side — docs/PRODUCTION_READINESS.md item 2)
There is no dashboard or metrics endpoint in this project (deliberately out of scope — see
docs/PRODUCTION_READINESS.md's Monitoring section). The one thing worth having for an unattended
demo is an external check that someone notices if the box or the app goes down. Set this up
once, by hand, in a free uptime service (UptimeRobot, Better Stack, or similar):

1. Create a new **HTTP(s)** monitor.
2. URL to monitor: `https://<DEMO_DOMAIN>/health` (the same `DEMO_DOMAIN` from step 4 above —
   e.g. `https://3-91-12-44.sslip.io/health`). `/health` is unauthenticated by design
   (`api/server.py`), safe for an external checker to hit unauthenticated.
3. Expected response: HTTP `200` with JSON body `{"status": "ok"}`. Configure the monitor's
   "keyword" or "response contains" check (if the service supports it) to look for `"ok"`, not
   just the status code — a reverse-proxy misconfiguration that returns a 200 error page from
   Caddy itself would otherwise pass a status-code-only check.
4. Check interval: 5 minutes is plenty for a demo (this is not a paged production SLA).
5. Alert destination: your own email/SMS/Slack webhook, whatever the free tier offers.
6. **What an alert means:** either the EC2 instance is down, the `app` or `caddy` container
   crashed/exited, or Caddy can't reach `app` over the internal network. It does NOT mean a
   *run* failed (a bad DeepSeek/Tavily call doesn't take `/health` down) — for that, see the
   log-based check below. First response: `ssh` in and run
   `docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml ps` to see which
   container is unhealthy, then check its logs.

This is infra configuration, not code — there is nothing to commit for it beyond this runbook
step. No dashboard, no metrics pipeline: a single uptime check is the entire monitoring surface
for this demo, intentionally.

## Log-based publish-failure signal
`api/server.py` logs `api.publish_failed` at **ERROR** level (structlog, JSON to stdout) any
time a run's `git_status` ends up `"failed"` after a publish attempt — distinct from
`git_node`'s own lower-severity log lines, specifically so it's grep-able:

    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml logs app \
        | grep '"event": "api.publish_failed"'

There is no alerting pipeline wired to this (that would be the dashboard/metrics work this
project deliberately doesn't have) — it exists so that if you ever do add a log-shipping
alert (e.g. a Datadog/Better Stack log forwarder watching for this exact event string), the
signal is already there to hook into.

## Security posture (demo, additive to DEPLOY.md)
- The app container is never directly publicly reachable — Caddy is the only public port,
  reverse-proxying over the internal Docker network.
- git_node is unchanged on autonomy: GIT_PUSH_ENABLED only lets it do the LOCAL merge it
  always did; it still has no push capability. Both that merge and `/ui/runs/{id}/publish`
  now also require `PUBLISH_TARGET=demo` with an allowlisted fork remote and demo Netlify
  URL (or production + `CONFIRM_PRODUCTION_PUBLISH=I_UNDERSTAND`). Default is deny.
- The fork's push credential lives only in `fork-clone/.git/config` on the VM, never in an
  image layer, env file, or compose file.
- Same B4 registry-volatility limitation as DEPLOY.md: avoid restarting `app` mid-review.
