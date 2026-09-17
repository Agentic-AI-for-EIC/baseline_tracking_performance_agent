#!/bin/bash
# Detached PID feature extraction (+ optional training) for one dataset tag.
#
# Same two survival tricks as scripts/launch_bkg.sh, and for the same reason:
# sessions here run inside an Apptainer container that is torn down when the
# session ends, killing background children.
#   1. setsid + nohup + </dev/null  -> the job owns its session, attached to no tty
#   2. --cache-dir                  -> already-read files replay from local disk,
#      so a relaunch only pays for the unfinished tail
#
# Feature extraction reads ~20 collections per file, so on the overseas
# +background endpoint (which is ~5.6x slower than JLab and intermittently drops
# reads with "[ERROR] Operation expired") this is the long step.
#
# Usage:
#   pid/scripts/run_pid.sh <dataset_tag> <filelist> [max_failures] [cache_dir] [tasks] [seed]
#   pid/scripts/run_pid.sh status
#
# Examples:
#   pid/scripts/run_pid.sh clean     filelists/clean_150.txt 0
#   pid/scripts/run_pid.sh bkg_mixed filelists/bkg_200.txt   20 cache/pid_features "eid,ehad"
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
PY=${PYTHON:-python}

if [ "${1:-}" = "status" ]; then
  for pidfile in runs/pid_*.pid; do
    [ -e "$pidfile" ] || continue
    name="$(basename "$pidfile" .pid)"; pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then state=RUNNING; else state=stopped; fi
    printf '%-34s %-8s pid %-8s started %s\n' "$name" "$state" "$pid" \
      "$(cat "runs/${name}.start" 2>/dev/null || echo '?')"
    tail -2 "runs/${name}.log" 2>/dev/null | sed 's/^/      /'
  done
  exit 0
fi

TAG="${1:?dataset_tag: clean|bkg_mixed}"
FILELIST="${2:?filelist (one root:// URL or path per line)}"
MAXFAIL="${3:-0}"
CACHE="${4:-cache/pid_features}"
TASKS="${5:-eid,ehad,hadpid,pooled}"
SEED="${6:-1234}"
mkdir -p runs output cache
[ -f "$FILELIST" ] || { echo "no such filelist: $FILELIST" >&2; exit 1; }

FEATURES="output/pid-features_${TAG}.pkl"
LOG="runs/pid_features_${TAG}.log"

# features, then train/evaluate every task, all in one detached job: the training
# step is fast (minutes) and must not be left to a session that may not survive.
#
# QUOTING WARNING (a real bug lived here): the detached shell inherits only
# *exported* variables, so '$VAR' single-quoted references expand to EMPTY
# inside. Every outer-shell value below is therefore expanded by THIS shell
# (double quotes); only the inner loop variables are escaped (\$task/\$lib).
# --n-iter 24 (not the default 12): grid samples have the statistics for a
# wider hyperparameter search; smoke runs keep the default.
setsid nohup bash -c "
  set -x
  $PY -m pid features --file-list \"$FILELIST\" --dataset-tag \"$TAG\" \
      --max-file-failures \"$MAXFAIL\" --cache-dir \"$CACHE\" --out \"$FEATURES\" || exit 1
  for task in \$(echo \"$TASKS\" | tr ',' ' '); do
    for lib in lightgbm xgboost sklearn_hgb; do
      $PY -m pid train --task \"\$task\" --model \"\$lib\" --dataset-tag \"$TAG\" \
          --features \"$FEATURES\" --seed \"$SEED\" --n-iter 24 || echo \"[run_pid] train \$task/\$lib failed (statistics?)\"
      if [ -f \"output/models/\${task}_\${lib}_${TAG}/test_scores.pkl\" ]; then
        $PY -m pid evaluate --task \"\$task\" --model \"\$lib\" --dataset-tag \"$TAG\" \
            --scores \"output/models/\${task}_\${lib}_${TAG}/test_scores.pkl\" --by pt,eta --plot
        $PY -m pid importance --dir \"output/models/\${task}_\${lib}_${TAG}\" \
            --task \"\$task\" --model \"\$lib\" --dataset-tag \"$TAG\" --plot \
            || echo \"[run_pid] WARNING: importance \$task/\$lib failed - see log above\"
      fi
    done
  done
  $PY \"$HERE/pid/scripts/check_learners.py\" --dataset-tag \"$TAG\" --tasks \"$TASKS\"
  echo '[run_pid] done'
" > "$LOG" 2>&1 < /dev/null &

echo $! > "runs/pid_features_${TAG}.pid"
date +%Y-%m-%dT%H:%M:%S > "runs/pid_features_${TAG}.start"
echo "launched pid_features_${TAG} (pid $!) -> $LOG"
echo "follow with: tail -f $LOG    |    status: pid/scripts/run_pid.sh status"
echo
echo "when BOTH tags are done, compare them:"
echo "  python -m pid compare --task eid --model lightgbm --artifact vs_pt"
echo "  python -m pid compare --task eid --model lightgbm --artifact all"
