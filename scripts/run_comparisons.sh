#!/bin/bash
# Run the clean-vs-background comparison for every metric once BOTH sides'
# results exist in output/. Skips any metric whose inputs are not present yet.
# Canonical pair: the same-campaign 26.07.1 reproduction (clean26071 vs
# bkg26071). The full-statistics run_reproduction_26071.sh driver runs the same
# comparisons internally; this is the stand-alone re-compare helper. The legacy
# mixed-campaign ("clean","bkg_mixed") pair is retired (Type-1 26.02.0 purged;
# outputs archived) and is no longer the comparison target.
# Usage: scripts/run_comparisons.sh [max_attempts]  (repeat until --all-ok)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
OUT=output
DONE=0
SKIP=0
for m in acceptance efficiency resolution fake-rate pid-confusion; do
    clean="${OUT}/${m}_clean26071.json"
    bkg="${OUT}/${m}_bkg26071.json"
    if [ -f "$clean" ] && [ -f "$bkg" ]; then
        python -m trkperf compare --metric "$m" --clean "$clean" --bkg "$bkg" --out-dir "$OUT" \
            && DONE=$((DONE + 1))
    else
        echo "[compare/$m] waiting on inputs: $(basename "$clean") $(basename "$bkg")"
        SKIP=$((SKIP + 1))
    fi
done
echo "comparisons written: $DONE, waiting: $SKIP"
[ "$SKIP" -eq 0 ]