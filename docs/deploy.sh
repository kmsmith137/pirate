#!/usr/bin/env bash
#
# Build Sphinx docs and deploy to the gh-pages branch. The build is deployed on
# top of the published origin/gh-pages (fetched here), so the push fast-forwards;
# gh-pages is treated as a regenerated build artifact, not a hand-edited branch.
#
# Usage:  docs/deploy.sh          (from the repo root)

set -euo pipefail

die() { echo "deploy.sh: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/docs/build/html"
WORKTREE_DIR="$(mktemp -d)"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    rm -rf "$WORKTREE_DIR"
}
trap cleanup EXIT

echo "==> Building Sphinx docs (docs-clean for a fresh build; also builds notes/*.tex PDFs)..."
# 'docs' depends on 'tex', so the notes PDFs are built and copied into the source
# tree (conf.py) before Sphinx runs. 'docs-clean' first gives a from-scratch build.
# Run as two invocations, NOT 'make docs-clean docs': with -j the two independent
# goals could run concurrently (clean racing the build), and a single invocation
# also captures DOC_INPUTS (via $(shell find)) at parse time -- including the tex
# PDF that docs-clean then deletes, leaving a stale prerequisite with no rule. A
# separate second 'make' re-evaluates the file list after the clean. -j 32 speeds
# up the underlying 'lib' compile that 'docs' pulls in.
make -C "$REPO_ROOT" docs-clean
make -C "$REPO_ROOT" -j 32 docs

echo "==> Preparing gh-pages worktree..."
git -C "$REPO_ROOT" worktree prune   # drop stale entries from a crashed earlier run

# Deploy on top of the PUBLISHED gh-pages so the push fast-forwards. The earlier
# version only checked for a LOCAL gh-pages branch and, finding none, created a
# fresh orphan -- but on any clone without a local gh-pages that orphan shares no
# history with origin/gh-pages, so the push is rejected as non-fast-forward. So:
#   - remote HAS gh-pages -> point the local branch at it (gh-pages is a
#                            regenerated build branch, so dropping local-only deploy
#                            commits is intended; this also self-heals a local orphan);
#   - remote does NOT     -> create the orphan only if local lacks it too.
# Common case adds one network op (the fetch); with your key in an ssh-agent that
# is no extra prompt.
rc=0
git -C "$REPO_ROOT" fetch origin gh-pages 2>/dev/null || rc=$?
if [ "$rc" -eq 0 ]; then
    git -C "$REPO_ROOT" branch -f gh-pages FETCH_HEAD
else
    # fetch failed: tell "remote has no gh-pages yet" apart from a real error.
    # ls-remote --exit-code: 2 = no such ref, 0 = it exists, other = unreachable.
    lrc=0
    git -C "$REPO_ROOT" ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1 || lrc=$?
    if [ "$lrc" -eq 2 ]; then
        if ! git -C "$REPO_ROOT" rev-parse --verify gh-pages >/dev/null 2>&1; then
            echo "    No gh-pages on the remote or locally; creating orphan gh-pages branch..."
            git -C "$REPO_ROOT" worktree add --detach "$WORKTREE_DIR"
            git -C "$WORKTREE_DIR" checkout --orphan gh-pages
            git -C "$WORKTREE_DIR" rm -rf . 2>/dev/null || true
            git -C "$WORKTREE_DIR" commit --allow-empty -m "Initialize gh-pages"
            git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
        fi
    else
        die "could not fetch origin gh-pages (fetch rc=$rc, ls-remote rc=$lrc); check network/credentials and retry"
    fi
fi

git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" gh-pages

echo "==> Deploying docs..."
# Replace all existing content with the fresh build.
find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$BUILD_DIR"/. "$WORKTREE_DIR"
# Disable Jekyll so that GitHub Pages serves _static/ and other underscore dirs.
touch "$WORKTREE_DIR/.nojekyll"

cd "$WORKTREE_DIR"
git add -A
# Always commit (--allow-empty) and push, so 'git push' fires unconditionally
# (and thus always prompts for credentials). This matches the user expectation
# that running deploy.sh produces a deploy; otherwise an unmodified rerun is
# silent, which can leave the operator unsure whether anything happened.
git commit --allow-empty -m "Deploy docs"
git push origin gh-pages
echo "==> Deployed to gh-pages"
