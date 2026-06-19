# CHEATSHEET — Local demo (Mac)

Run the content-agent demo on your Mac and publish to the Netlify fork.

## 0. Prerequisites (once)
- `.env` exists in the project root with DEEPSEEK_API_KEY, TAVILY_API_KEY, API_BEARER_TOKEN.
- The fork's push remote exists (creates it if missing):
```bash
  git -C ~/tmp/tmw-fork remote get-url demo \
    || git -C ~/tmp/tmw-fork remote add demo git@github.com:anudeepreddy332/themachinist-website-fork.git
```
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
GIT_PUSH_ENABLED=true \
THEMACHINIST_REPO_PATH=$HOME/tmp/tmw-fork \
PUBLISH_REMOTE=demo \
NETLIFY_BASE_URL=https://tmw-demo-site.netlify.app \
API_BEARER_TOKEN=<the API_BEARER_TOKEN from your .env> \
uv run python main.py serve --port 8099
```
What each piece does:
- `GIT_PUSH_ENABLED=true` — git_node merges the article into your local fork clone.
- `THEMACHINIST_REPO_PATH` — the fork clone it merges into.
- `PUBLISH_REMOTE=demo` — the "Publish to live" button pushes to this remote.
- `NETLIFY_BASE_URL` — used to build the final live-article link.
- The server + web page run on port 8099. Leave this terminal open for the whole demo.

## 3. Open the demo
- Browser: http://localhost:8099
- Paste your API_BEARER_TOKEN into the token box at the top (saved in the browser after the first time).
- Type an article topic and press Generate.

## 4. Full approve → publish cycle (browser)
1. Watch the pipeline nodes stream live (retrieve, draft, verify, reflect).
2. GATE 1 (content): review the draft + grounding table. Press `a` to approve.
   `r` rejects, `c` requests changes (focuses the feedback box).
3. The article renders. GATE 2 (layout): click "Open preview in new tab" to view the HTML.
   Press `a` to approve, or `c` to request a layout change (the text stays frozen).
4. After gate-2 approval, a "🚀 Publish to live" button appears. Press `p` (or click it).
   This pushes the fork to GitHub; Netlify redeploys in ~1–2 minutes.
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