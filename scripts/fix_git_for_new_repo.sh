#!/usr/bin/env bash
# Point this folder at a NEW GitHub repo (not sde-course).
set -euo pipefail
REPO_URL="${1:-git@github.com:kierandidi/ml-dl-course.git}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CURRENT="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$CURRENT" == *"sde-course"* ]]; then
  echo "WARNING: origin currently points at sde-course: $CURRENT"
  echo "This course must use its own repo, e.g. ml-dl-course"
  read -r -p "Remove .git and re-init? [y/N] " ans
  if [[ "${ans,,}" == "y" ]]; then
    rm -rf .git
    git init
    git branch -M main
  fi
fi

if ! git remote get-url origin &>/dev/null; then
  git remote add origin "$REPO_URL"
else
  git remote set-url origin "$REPO_URL"
fi

echo "origin -> $(git remote get-url origin)"
echo "Next: git add . && git commit -m 'ML & DL course' && git push -u origin main"
