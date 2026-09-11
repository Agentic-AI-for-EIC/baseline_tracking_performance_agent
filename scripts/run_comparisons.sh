#!/bin/bash
# Run the clean-vs-bkg_mixed comparison for every metric once BOTH sides'
# results exist in output/. Skips any metric whose inputs are not present yet.
# Usage: scripts/run_comparisons.sh [max_attempts]  (repeat until --all-ok)
cd /home/wxie/eic/tracking_performance
OUT=output
DONE=0
SKIP=0
for m in acceptance efficiency resolution fake-rate pid-confusion; do
    clean="${OUT}/${m}_clean.json"
    bkg="${OUT}/${m}_bkg_mixed.json"
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