#!/bin/bash
# Full fake-rate + resolution with the reco-side N_hit>=4 & N_si>=3 cut,
# written to output/reco_cut43 (canonical no-cut tables untouched).
# Resumable: every metric re-reads its flat branch-sets + the new hit-count
# tables from the on-disk cache, so a restart only processes the tail.
set -uo pipefail
cd /local/scratch/a/wxie/eic/baseline_tracking_performance_agent
PY=/opt/local/bin/python3
OUT=output/reco_cut43
mkdir -p "$OUT"

run () {
  local metric=$1 tag=$2 list=$3 cache=$4; shift 4
  echo "[reco_cut43] ===== $metric / $tag ====="
  $PY -m trkperf "$metric" --file-list "$list" --dataset-tag "$tag" \
     --min-q2-tier 1 --cache-dir "$cache" \
     --min-reco-hits 4 --min-reco-si-hits 3 --max-file-failures 20 \
     --out-dir "$OUT" --plots-dir "$OUT"
  echo "[reco_cut43] done $metric / $tag  (exit $?)"
}

for metric in fake-rate resolution; do
  run "$metric" clean26071 filelists/clean26071_local.txt cache/clean26071_files
  run "$metric" bkg26071  filelists/bkg26071_local.txt  cache/bkg26071_files
done
echo "[reco_cut43] ALL DONE"
