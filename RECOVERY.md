# content-agent — recovery & rollback runbook

## Roll back a published article (single command)
    ./scripts/rollback_publish.sh <slug>
    cd "$THEMACHINIST_REPO_PATH" && git push origin main      # the supervised push
Then confirm it's gone (Netlify redeploys ~1-3 min after push):
    curl -sI "$SITE_URL/<slug>.html" | head -1                # expect 404

The script reverts the article's --no-ff merge commit (`git revert -m 1`), which is
safe on a pushed branch — it adds an inverse commit, no history rewrite, and works for
both "merged" (new-file) and "tagged_and_merged" (changed-file) publishes. The pre-merge
tag v-YYYYMMDD-<slug> is shown for reference but is NOT required for rollback.

Tag-based rollback window: only the newest 5 `v-` tags are kept (git_node prunes the
rest). This limits the NAMED-TAG reference window, not the revert-based rollback above.

## Publishing posture (why a human is always in the loop)
git_node performs a LOCAL merge only — it never pushes to origin. The agent therefore
cannot publish to production autonomously; every live publish requires a deliberate
`git push origin main` by the operator. This is intentional and is the freeze's primary
publish safeguard. GIT_PUSH_ENABLED gates the LOCAL git operations, not the push.

## B4 durability limitation (carried from DEPLOY.md)
SqliteSaver checkpoints survive container restart (on the ./outputs volume); the API's
in-memory run REGISTRY does not. Drain awaiting_review runs before any restart/deploy.

## Service recovery
- App unhealthy: `docker compose -f docker-compose.prod.yml logs app`; look for the
  `api.warmup` startup line. Restart: `docker compose -f docker-compose.prod.yml up -d`.
  (Drain pending reviews first — see durability limitation.)
- Qdrant data intact across restarts via the qdrant_data named volume. If the collection
  is empty after a volume loss, re-ingest:
    docker compose -f docker-compose.prod.yml run --rm app python scripts/ingest.py --source kb/seed_docs/
- Full app rollback: `git checkout <prev tag> && docker compose -f docker-compose.prod.yml build && ... up -d`.

## Production verification (post-publish)
    SITE_URL=https://themachinist.org
    curl -sI "$SITE_URL/<slug>.html" | head -1        # 200 = live
    curl -s  "$SITE_URL/<slug>.html" | grep -qi "<phrase>" && echo OK