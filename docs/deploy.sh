#!/usr/bin/env bash
#
# Build Sphinx docs and deploy to the gh-pages branch.
#
# Usage:  docs/deploy.sh          (from the repo root)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/docs/build/html"
WORKTREE_DIR="$(mktemp -d)"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    rm -rf "$WORKTREE_DIR"
}
trap cleanup EXIT

echo "==> Building Sphinx docs..."
make -C "$REPO_ROOT/docs" clean html

echo "==> Preparing gh-pages worktree..."
# Create the gh-pages branch as an orphan if it doesn't exist yet.
if ! git -C "$REPO_ROOT" rev-parse --verify gh-pages >/dev/null 2>&1; then
    echo "    Creating orphan gh-pages branch..."
    git -C "$REPO_ROOT" worktree add --detach "$WORKTREE_DIR"
    git -C "$WORKTREE_DIR" checkout --orphan gh-pages
    git -C "$WORKTREE_DIR" rm -rf . 2>/dev/null || true
    git -C "$WORKTREE_DIR" commit --allow-empty -m "Initialize gh-pages"
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
fi

git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" gh-pages

echo "==> Deploying docs..."
# Replace all existing content with the fresh build.
find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$BUILD_DIR"/. "$WORKTREE_DIR"

cd "$WORKTREE_DIR"
git add -A
if git diff --cached --quiet; then
    echo "==> No changes to deploy."
else
    git commit -m "Deploy docs"
    git push origin gh-pages
    echo "==> Deployed to gh-pages"
fi
