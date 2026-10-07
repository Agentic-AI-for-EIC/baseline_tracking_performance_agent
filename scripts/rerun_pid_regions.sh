#!/usr/bin/env bash
# Regenerate PID per-region tables/figures with the |eta|<1.5 boundary.
# Mirrors the original 26071 invocations (run_pid.sh): same models/scores
# (no retraining), same flags; only the boundary constant changed.
set -uo pipefail
cd /home/wxie/eic/baseline_tracking_performance_agent
export PYTHONUNBUFFERED=1
REGIONS="barrel,forward endcap,backward endcap"
CACHE=cache/pid_features
FAIL=0

for TAG in clean26071 bkg26071; do
  if [ "$TAG" = clean26071 ]; then MF=5; FL=filelists/clean26071_local.txt
  else MF=20; FL=filelists/bkg26071_local.txt; fi
  FEAT="output/pid-features_${TAG}.pkl"
  echo "[pid-regions] features $TAG"
  python3 -m pid features --file-list "$FL" --dataset-tag "$TAG" \
    --max-file-failures "$MF" --cache-dir "$CACHE" --out "$FEAT" --jobs 6 || FAIL=1
  for task in eid ehad hadpid pooled; do
    for lib in lightgbm xgboost sklearn_hgb; do
      S="output/models/${task}_${lib}_${TAG}/test_scores.pkl"
      [ -f "$S" ] || { echo "[pid-regions] SKIP missing $S"; continue; }
      echo "[pid-regions] evaluate $task/$lib/$TAG"
      python3 -m pid evaluate --task "$task" --model "$lib" --dataset-tag "$TAG" \
        --scores "$S" --by pt,eta --plot \
        --eta-region "$REGIONS" --features "$FEAT" \
        --max-file-failures "$MF" --cache-dir "$CACHE" || FAIL=1
    done
  done
done
for TAG in clean26071 bkg26071; do
  if [ "$TAG" = clean26071 ]; then MF=5; PDIR="output/plot_PID/plots_clean_pid_perform"
  else MF=20; PDIR="output/plot_PID/plots_bkgmix_pid_perform"; fi
  echo "[pid-regions] performance lightgbm/$TAG"
  python3 -m pid performance --dataset-tag "$TAG" --model lightgbm \
    --bin-source both --eta-region "$REGIONS" \
    --features "output/pid-features_${TAG}.pkl" \
    --max-file-failures "$MF" --cache-dir "$CACHE" --plots-dir "$PDIR" || FAIL=1
done
echo "[pid-regions] DONE fail=$FAIL"
