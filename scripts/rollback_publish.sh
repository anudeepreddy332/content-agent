#!/usr/bin/env bash
# Roll back the most recent publish of an article by reverting its --no-ff merge.
# Safe on a pushed branch: an inverse commit, no history rewrite. Handles BOTH paths:
#   - new-file publish  -> revert removes the file
#   - changed-file edit  -> revert RESTORES the previous version (file stays, old content)
# No manual file deletion: git revert updates the working tree correctly for both.
#
#   ./scripts/rollback_publish.sh <slug> [repo_path]
#
# repo_path defaults to $THEMACHINIST_REPO_PATH. Does NOT push — prints the push command
# so the operator stays the gate (same posture as publish).
set -euo pipefail

SLUG="${1:?usage: rollback_publish.sh <slug> [repo_path]}"
REPO="${2:-${THEMACHINIST_REPO_PATH:?set THEMACHINIST_REPO_PATH or pass repo_path}}"
FILE="${SLUG}.html"

cd "$REPO"
git checkout main >/dev/null 2>&1
[ -z "$(git status --porcelain)" ] || { echo "ABORT: working tree not clean"; exit 1; }

# Find the publish to undo. git_node makes a --no-ff merge of feature/article-<slug>,
# whose default message carries the branch name. We grep the MESSAGE, not a path filter:
# path-filtered --merges is simplified away by git's TREESAME rule (the B6 finding).
TARGET=$(git log -1 --merges --grep="feature/article-${SLUG}" --format=%H 2>/dev/null || true)
if [ -z "$TARGET" ]; then
    # Fallback: most recent commit that touched the file (e.g. a manual/squash publish).
    TARGET=$(git log -1 --format=%H -- "$FILE")
fi
[ -n "$TARGET" ] || { echo "ABORT: no commit found that published $FILE"; exit 1; }

PRE_TAG=$(git tag -l "v-*-${SLUG}" | sort | tail -1)

echo "Rollback plan for: $FILE"
echo "  commit to revert       : $TARGET  ($(git log -1 --format=%s "$TARGET"))"
echo "  pre-merge tag (if any) : ${PRE_TAG:-none (new-file publish)}"
echo "  commits since target   : $(git rev-list --count "${TARGET}..HEAD")"
echo
read -r -p "Type the slug to confirm revert: " CONFIRM
[ "$CONFIRM" = "$SLUG" ] || { echo "ABORT: confirmation mismatch"; exit 1; }

# Revert. -m 1 ONLY for a merge commit (has a 2nd parent); plain revert for a regular one.
if git rev-parse -q --verify "${TARGET}^2" >/dev/null 2>&1; then
    git revert -m 1 --no-edit "$TARGET"
else
    git revert --no-edit "$TARGET"
fi

echo
if [ -e "$FILE" ]; then
    echo "NOTE: $FILE is still present — this publish was a MODIFICATION, so the PREVIOUS"
    echo "version has been restored. This is correct; do not delete it. Verify its content."
else
    echo "$FILE removed — this was the article's initial publish."
fi
echo
echo "Reverted locally. Live rollback requires the supervised push:"
echo "    cd \"$REPO\" && git push origin main"
echo "Then verify (initial publish -> 404; modification -> previous version live):"
echo "    curl -sI \"\$SITE_URL/${FILE}\" | head -1"