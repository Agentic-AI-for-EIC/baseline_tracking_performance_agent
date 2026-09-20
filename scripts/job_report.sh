#!/bin/bash
# Twice-daily grid-job status report (human-readable, one command).
#
# Usage: scripts/job_report.sh
#
# Covers both PID grid jobs (clean + bkg_mixed): process state, pipeline
# stage, feature progress, skipped files, per-task train outcomes and gate
# verdicts, errors/tracebacks, check-learners output, and deliverables
# manifest completeness. Read-only: never touches jobs, outputs, or the tree.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "===== PID grid jobs report: $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo
for TAG in clean bkg_mixed; do
    LOG="runs/pid_features_${TAG}.log"
    PIDFILE="runs/pid_features_${TAG}.pid"
    echo "--- ${TAG} ---"
    if [ ! -f "$LOG" ]; then
        echo "no log yet (job never launched?)"
        echo
        continue
    fi
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "state: RUNNING (pid $(cat "$PIDFILE"))"
    else
        echo "state: STOPPED (pid file: $(cat "$PIDFILE" 2>/dev/null || echo none))"
    fi
    echo "log lines: $(wc -l < "$LOG" 2>/dev/null || echo 0), done marker: $(grep -c '^\[run_pid\] done' "$LOG" 2>/dev/null)"
    prog=$(grep -o '\[features\] [0-9]*/[0-9]* files' "$LOG" 2>/dev/null | tail -1)
    echo "features progress: ${prog:-n/a}"
    drops=$(grep -c 'Operation expired\|vector_read properly' "$LOG" 2>/dev/null)
    echo "transient xrootd drops: ${drops:-0}"
    traces=$(grep -c '^Traceback' "$LOG" 2>/dev/null)
    echo "tracebacks: ${traces:-0}"
    echo "train outcomes:"
    grep -E "^\[train/[a-z]+/[a-z_]+/" "$LOG" 2>/dev/null \
        | sed 's/ (label-shuffle.*//' | head -20
    echo "gate FAILs:"
    grep "gate:FAIL" "$LOG" 2>/dev/null | sed 's/ | requires.*//' | head -10
    echo "train failures (caught by launcher):"
    grep "^\[run_pid\] train .* failed" "$LOG" 2>/dev/null | head -10
    echo "check-learners:"
    sed -n '/check-learners/,$p' "$LOG" 2>/dev/null | grep -E "PASS|WARNING|only [0-9]" | head -12
    echo
done

echo "--- deliverables manifests ---"
python3 - "$@" <<'PYEOF'
import json, os
for tag in ("clean", "bkg_mixed"):
    p = f"output/pid-deliverables_lightgbm_{tag}.json"
    if not os.path.exists(p):
        print(f"{tag}: no manifest yet"); continue
    d = json.load(open(p))
    rows = d["data"]
    missing = [f"{r['channel']}/{r['figure']}" for r in rows if not r["present"]]
    print(f"{tag}: {len(rows) - len(missing)}/{len(rows)} figures present", end="")
    print(("" if not missing else f"  MISSING: {missing[:6]}"))
PYEOF
echo "=== end report ==="
