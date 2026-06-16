#!/usr/bin/env bash
# Roll back a published article by reverting its --no-ff merge commit on main.
# Safe for a pushed branch: creates an inverse commit, no history rewrite.
# Works for both publish paths ("merged" new-file and "tagged_and_merged").
#
#   ./scripts/rollback_publish.sh <slug> [repo_path]
#
# repo_path defaults to $THEMACHINIST_REPO_PATH. Does NOT push — prints the exact
# push command so the operator stays the gate (same posture as publish).
set -euo pipefail

SLUG="${1:?usage: rollback_publish.sh <slug> [repo_path]}"
REPO="${2:-${THEMACHINIST_REPO_PATH:?set THEMACHINIST_REPO_PATH or pass repo_path}}"
FILE="${SLUG}.html"

cd "$REPO"
git checkout main >/dev/null 2>&1
[ -z "$(git status --porcelain)" ] || { echo "ABORT: working tree not clean"; exit 1; }

# Most recent --no-ff merge commit that touched this article on main.
MERGE=$(git log -1 --format=%H -- "$FILE")
[ -n "$MERGE" ] || { echo "ABORT: no merge commit found for $FILE"; exit 1; }

PRE_TAG=$(git tag -l "v-*-${SLUG}" | sort | tail -1)

echo "Rollback plan for: $FILE"
echo "  merge commit to revert : $MERGE  ($(git log -1 --format=%s "$MERGE"))"
echo "  pre-merge tag (if any) : ${PRE_TAG:-none (new-file publish)}"
echo "  commits since merge    : $(git rev-list --count "${MERGE}..HEAD")"
echo
read -r -p "Type the slug to confirm revert: " CONFIRM
[ "$CONFIRM" = "$SLUG" ] || { echo "ABORT: confirmation mismatch"; exit 1; }

git revert -m 1 --no-edit "$MERGE"
rm -f "$FILE"   # clean up the untracked file (git revert doesn't delete added files)
echo
echo "Reverted locally. Live rollback requires the supervised push:"
echo "    cd \"$REPO\" && git push origin main"
echo "Then verify the article is gone (expect 404):"
echo "    curl -sI \"\$SITE_URL/${FILE}\" | head -1"