#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
TYPST="${TYPST:-typst}"
mkdir -p "$ROOT/assets/slides"
for n in $(seq -w 1 10); do
  src="$ROOT/slides/days/day${n}.typ"
  out="$ROOT/assets/slides/day${n}.pdf"
  if [[ -f "$src" ]]; then
    echo "Compiling $src -> $out"
    "$TYPST" compile "$src" "$out" --root "$ROOT"
  fi
done
# smoke deck
if [[ -f "$ROOT/slides/_smoke.typ" ]]; then
  "$TYPST" compile "$ROOT/slides/_smoke.typ" "$ROOT/assets/slides/_smoke.pdf" --root "$ROOT"
fi
echo "All slides built."
