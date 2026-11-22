#!/bin/bash
# Run the clean-vs-background comparison for every tracking metric once BOTH
# sides' results exist in output/. Skips any comparison whose inputs are not
# present yet.
#
# Canonical pair: the same-campaign 26.07.1 reproduction (clean26071 vs
# bkg26071), canonical outputs named <metric>[-<region>]_26071_comparison.*
# to match scripts/run_reproduction_26071.sh (which runs the same comparisons
# internally; this is the stand-alone re-compare helper). The legacy
# mixed-campaign ("clean","bkg_mixed") pair is retired (Type-1 26.02.0
# purged; its bare <metric>_comparison.* files archived) and is no longer a
# comparison target.
#
# Usage: scripts/run_comparisons.sh   (exit 0 iff nothing was waiting)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
OUT=output
PY=${PYTHON:-python3}
DONE=0
SKIP=0

# metric + optional region (region only where the metric has region runs)
SPECS="acceptance: acceptance:backward acceptance:forward
       efficiency: efficiency:backward efficiency:forward
       resolution: fake-rate:"

for spec in $SPECS; do
    m=${spec%%:*}; r=${spec#*:}
    sfx=${r:+_$r}                       # output json: <m>[_<r>]_<tag>.json
    name=${m}${r:+-$r}                  # comparison stem: <m>[-<r>]_26071
    clean="${OUT}/${m}${sfx}_clean26071.json"
    bkg="${OUT}/${m}${sfx}_bkg26071.json"
    if [ -f "$clean" ] && [ -f "$bkg" ]; then
        $PY -m trkperf compare --metric "${name}_26071" \
            --clean "$clean" --bkg "$bkg" --out-dir "$OUT" \
            && DONE=$((DONE + 1))
    else
        echo "[compare/$name] waiting on inputs: $(basename "$clean") $(basename "$bkg")"
        SKIP=$((SKIP + 1))
    fi
done

# PID confusion matrix (truth-PID vs reco-PID): only once the PID chain has
# been re-run under the current definitions (deferred).
for tagpair in "pid-confusion_clean26071.json pid-confusion_bkg26071.json"; do
    set -- $tagpair
    if [ -f "$OUT/$1" ] && [ -f "$OUT/$2" ]; then
        $PY -m trkperf compare --metric "pid-confusion_26071" \
            --clean "$OUT/$1" --bkg "$OUT/$2" --out-dir "$OUT" && DONE=$((DONE + 1))
    else
        echo "[compare/pid-confusion] waiting on inputs (PID re-run deferred)"
        SKIP=$((SKIP + 1))
    fi
done

echo "comparisons written: $DONE, waiting: $SKIP"
[ "$SKIP" -eq 0 ]
