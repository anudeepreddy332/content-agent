# CHEATSHEET — Cloud demo (AWS EC2)

Run the demo from EC2 with a public HTTPS URL. The Docker image is pulled from Docker Hub
(`anudeepreddy332/content-agent:demo`) — nothing is built on the instance.

## 0. One-time facts
- SSH key: `~/Downloads/agent.pem`
- Login / dir: `ubuntu@<PUBLIC_IP>`, project at `~/content-agent`
- No Elastic IP is attached, so the public IP changes on every stop/start. You MUST update
  the demo domain after each start (step 2).

## 1. Start the instance and connect
```bash
# AWS console: EC2 → Instances → select → Instance state → Start instance.
# Wait ~60s, copy the new Public IPv4, then:
ssh -i ~/Downloads/agent.pem ubuntu@<PUBLIC_IP>
cd ~/content-agent
```

## 2. Update the demo domain to the new IP (REQUIRED after every restart)
```bash
NEW_DOMAIN="$(curl -s ifconfig.me | tr '.' '-').sslip.io"
echo "Demo URL will be: https://$NEW_DOMAIN"
grep -q '^DEMO_DOMAIN=' .env \
  && sed -i "s|^DEMO_DOMAIN=.*|DEMO_DOMAIN=$NEW_DOMAIN|" .env \
  || echo "DEMO_DOMAIN=$NEW_DOMAIN" >> .env
grep DEMO_DOMAIN .env
# sslip.io maps <dashed-ip>.sslip.io to your IP, so Caddy can issue an HTTPS cert for it.
```

## 3. Bring up the stack
```bash
F="-f docker-compose.prod.yml -f docker-compose.demo.yml -f docker-compose.image.yml"
docker compose $F pull        # pull the app image from Docker Hub + qdrant + caddy (no build)
docker compose $F up -d qdrant
docker compose $F up -d        # starts the API (app) and the HTTPS reverse proxy (caddy)
```

## 3b. First time only (or after `docker compose $F down -v` wipes Qdrant): ingest the KB
```bash
docker run --rm --network content-agent_default \
  -e QDRANT_URL=http://qdrant:6333 \
  -e QDRANT_COLLECTION=machinist_evergreen \
  -v ./kb/seed_docs:/app/kb/seed_docs:ro \
  anudeepreddy332/content-agent:demo \
  python scripts/ingest.py --source /app/kb/seed_docs/
# Skip on normal restarts — the Qdrant data volume persists across stop/start.
```

## 4. Verify health
```bash
docker compose $F ps                 # app, qdrant, caddy → Up / (healthy)
docker compose $F logs -f caddy      # watch the TLS cert get issued on first hit (~30s); Ctrl+C to stop
curl -s https://$NEW_DOMAIN/health   # expect {"status":"ok"} (give Caddy a minute first)
```

## 5. Run the demo (browser)
- Open `https://<YOUR_DEMO_URL>` — the value printed in step 2.
- Paste your API_BEARER_TOKEN at the top (saved in the browser after first use).
- Type a topic → Generate → watch nodes stream live.
- GATE 1 (content): `a` approve / `r` reject / `c` request changes.
- GATE 2 (layout): "Open preview in new tab" to view; `a` approve / `c` layout change.
- After gate-2 approval: press `p` (or the "🚀 Publish to live" button). The container pushes
  the fork to GitHub; Netlify redeploys in ~1–2 min; a clickable LIVE link appears.

## 6. Stop the instance (halt charges)
```bash
exit    # leave SSH
# AWS console: EC2 → select instance → Instance state → Stop instance.
# Stopped = no compute charges. EBS storage still costs a few cents/day; terminate to stop that too.
```

## 7. Restart from a stopped state — what persists vs not
PERSISTS (on the EBS disk): the repo, `.env`, `fork-clone/` (with its PAT), the Qdrant data
volume, and `outputs/`. So you never re-clone, re-ingest, or rebuild.
CHANGES: the public IP (→ redo step 2). The in-memory run registry is cleared (only matters
for a run that was paused mid-review across the restart).
TO RESTART: start instance → SSH in → step 2 (update DEMO_DOMAIN) → step 3 (pull + up). Skip
ingest. The image lives on Docker Hub, so there is never a build step on the box.