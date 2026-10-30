#!/bin/bash
# Re-run fake-rate (+ resolution) for both datasets under the NEW strict->0.5
# matching, same reco-side 4/3 cut as canonical, into output/_strict for diff.
set -uo pipefail
cd /local/scratch/a/wxie/eic/baseline_tracking_performance_agent
PY=/opt/local/bin/python3
OUT=output/_strict; mkdir -p "$OUT"
run () {
  $PY -m trkperf "$1" --file-list "$2" --dataset-tag "$3" --min-q2-tier 1 \
     --cache-dir "$4" --max-file-failures 10 --min-reco-hits 4 --min-reco-si-hits 3 \
     --out-dir "$OUT" --plots-dir "$OUT"
  echo "[strict] done $1/$3 (exit $?)"
}
run fake-rate filelists/clean26071_local.txt clean26071 cache/clean26071_files
run fake-rate filelists/bkg26071_local.txt  bkg26071  cache/bkg26071_files
run resolution filelists/clean26071_local.txt clean26071 cache/clean26071_files
run resolution filelists/bkg26071_local.txt  bkg26071  cache/bkg26071_files
echo "[strict] ALL DONE"
