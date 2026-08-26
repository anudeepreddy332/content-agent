# CHEATSHEET — Local demo (Mac)

Run the content-agent demo on your Mac and publish to the Netlify **fork**
(`themachinist-website-fork` / tmw-demo-site.netlify.app). Never point a client
rehearsal at the production website.

Publication is fail-closed. `PUBLISH_TARGET=demo` is required, and the configured
Git remote URL must actually be `anudeepreddy332/themachinist-website-fork`.

## 0. Prerequisites (once)
- `.env` exists in the project root with DEEPSEEK_API_KEY, TAVILY_API_KEY, API_BEARER_TOKEN.
  Do not put `PUBLISH_TARGET=production` or `CONFIRM_PRODUCTION_PUBLISH` in `.env` for a demo.
- The fork clone exists and its push remote is the **fork** (creates `demo` if missing):
```bash
  git -C ~/tmp/tmw-fork remote get-url demo \
    || git -C ~/tmp/tmw-fork remote add demo git@github.com:anudeepreddy332/themachinist-website-fork.git
  git -C ~/tmp/tmw-fork remote -v   # must NOT list themachinist-website.git (no -fork)
```
- Git identity is set on that clone (`user.name` / `user.email`) so git_node can commit.
- Your GitHub SSH key is loaded — the local publish pushes over SSH:
```bash
  ssh-add -l    # should list a key; if not: ssh-add ~/.ssh/id_ed25519
```

## 1. First time only — start Qdrant and load the knowledge base
```bash
docker compose up -d qdrant
# starts the local Qdrant vector database on localhost:6333

uv run python scripts/ingest.py --source kb/seed_docs/
# embeds the seed documents into Qdrant. Only needed once — data persists in the Docker volume.
```

## 2. Start the demo server
```bash
PUBLISH_TARGET=demo \
GIT_PUSH_ENABLED=true \
THEMACHINIST_REPO_PATH=$HOME/tmp/tmw-fork \
PUBLISH_REMOTE=demo \
NETLIFY_BASE_URL=https://tmw-demo-site.netlify.app \
uv run python main.py serve --host 127.0.0.1 --port 8099
```
What each piece does:
- `PUBLISH_TARGET=demo` — fail-closed allowlist. Unset = no merge and no push.
- `GIT_PUSH_ENABLED=true` — git_node may merge locally **after** the allowlist passes.
- `THEMACHINIST_REPO_PATH` — the **fork** clone (never the production website clone).
- `PUBLISH_REMOTE=demo` — must resolve to `anudeepreddy332/themachinist-website-fork`.
- `NETLIFY_BASE_URL` — must be exactly the demo site (canonicalized).
- `--host 127.0.0.1` — recommended bind for a screen-share demo (not `0.0.0.0`).
- The server + web page run on port 8099. Leave this terminal open for the whole demo.

## 3. Open the demo
- Browser: http://127.0.0.1:8099
- Paste your API_BEARER_TOKEN into the token box at the top. It is **memory-only** —
  the browser does **not** save it (no localStorage / sessionStorage / cookies).
  The field is cleared after Generate. The same tab reuses the in-memory token for a
  later Generate; a new tab, or a failed auth, requires re-pasting.
- Type an article topic and press Generate.

## 4. Full approve → publish cycle (browser)
1. Watch the pipeline nodes stream live (retrieve, draft, verify, reflect).
2. GATE 1 (content): review the draft + grounding table. Press `a` to approve.
   `r` rejects, `c` requests changes (focuses the feedback box).
3. The article renders. GATE 2 (layout): click "Expand in-page preview" to enlarge
   the HTML iframe (there is no new-tab preview). Press `a` to approve, or `c` to
   request a layout change (the text stays frozen).
4. After gate-2 approval, a "Publish to live" button appears. Press `p` (or click it).
   This pushes the **fork** to GitHub; Netlify redeploys in ~1–2 minutes.
5. A clickable LIVE link appears. Click it → the published article on tmw-demo-site.netlify.app.

## 5. Stop the server
```bash
# in the server terminal: Ctrl+C
docker compose stop qdrant     # optional — stop local Qdrant too
```

## 6. Logs and telemetry
- Live logs: stream in the server terminal as structured JSON.
- Per-run telemetry: `outputs/runs/<run_id>.json` (cost, tokens, grounding, attribution, latency).
- Rendered articles (local archive): `outputs/articles/`
- Durable HITL state: `outputs/checkpoints.sqlite`
