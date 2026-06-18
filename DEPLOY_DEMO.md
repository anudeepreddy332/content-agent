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
    git clone <repo> && cd content-agent && git checkout feature/demo-ui   # or main, once merged
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

## 5. Build and start (prod compose + demo override)
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml build
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml up -d qdrant
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml run --rm app \
        python scripts/ingest.py --source kb/seed_docs/
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml up -d
    docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml ps

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

    export GIT_PUSH_ENABLED=true
    export THEMACHINIST_REPO_PATH=$HOME/tmp/tmw-fork
    export PUBLISH_REMOTE=demo            # add a remote named "demo" pointing at the fork,
                                           # or use "origin" if that's already the fork's remote
    export NETLIFY_BASE_URL=https://tmw-demo-site.netlify.app
    uv run python main.py serve --port 8099

    # in another terminal
    T=$(grep -E '^API_BEARER_TOKEN=' .env | cut -d= -f2-)
    RID=$(curl -s -X POST http://localhost:8099/ui/runs -H "Authorization: Bearer $T" \
          -H 'Content-Type: application/json' -d '{"topic":"Test topic"}' | python3 -c \
          'import sys,json;print(json.load(sys.stdin)["run_id"])')
    # drain SSE at http://localhost:8099/ui/runs/$RID/events?token=$T, approve both gates via
    #   curl -s -X POST http://localhost:8099/ui/runs/$RID/approve -H "Authorization: Bearer $T"
    # once git_status is merged/tagged_and_merged:
    curl -s -X POST http://localhost:8099/ui/runs/$RID/publish -H "Authorization: Bearer $T"

If `PUBLISH_REMOTE=demo` doesn't exist yet on the fork clone:
    git -C ~/tmp/tmw-fork remote add demo git@github.com:anudeepreddy332/themachinist-website-fork.git

## Security posture (demo, additive to DEPLOY.md)
- The app container is never directly publicly reachable — Caddy is the only public port,
  reverse-proxying over the internal Docker network.
- git_node is unchanged: GIT_PUSH_ENABLED only lets it do the LOCAL merge it always did; it
  still has no push capability. The push is the separate `/ui/runs/{id}/publish` endpoint,
  human-triggered from the SPA's "Publish to live" button — autonomous publish is still
  impossible.
- The fork's push credential lives only in `fork-clone/.git/config` on the VM, never in an
  image layer, env file, or compose file.
- Same B4 registry-volatility limitation as DEPLOY.md: avoid restarting `app` mid-review.
