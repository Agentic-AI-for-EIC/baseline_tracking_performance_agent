#!/bin/bash
# Hourly grid-job monitor for a single foreground wait.
#
# Usage: scripts/watch_jobs.sh [max_hours] [interval_seconds]
#
# Checks the clean PID job once per interval, appends a timestamped line to
# runs/monitor.log, and exits early when human attention is needed:
#   exit 0 - clean job finished ([run_pid] done marker present)
#   exit 2 - clean job dead WITHOUT the done marker, or a NEW traceback
#            appeared in its log (triage immediately)
#   exit 3 - timed out waiting (still running; re-arm or keep waiting)
# Read-only: never touches jobs, outputs, or the tree.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MAXH="${1:-5}"
INTERVAL="${2:-3600}"
LOG="runs/pid_features_clean.log"
PIDFILE="runs/pid_features_clean.pid"
MONLOG="runs/monitor.log"

deadline=$((SECONDS + MAXH * 3600))
last_traces=0
if [ -f "$LOG" ]; then
    last_traces=$(grep -c '^Traceback' "$LOG" 2>/dev/null)
    last_traces=${last_traces:-0}
fi

iter=0
while true; do
    iter=$((iter + 1))
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    alive="no"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        alive="yes"
    fi
    done_marker=0
    traces=0
    if [ -f "$LOG" ]; then
        done_marker=$(grep -c '^\[run_pid\] done' "$LOG" 2>/dev/null)
        done_marker=${done_marker:-0}
        traces=$(grep -c '^Traceback' "$LOG" 2>/dev/null)
        traces=${traces:-0}
        last_line=$(tail -1 "$LOG" 2>/dev/null | cut -c1-140)
    fi
    echo "[$ts] iter=$iter alive=$alive done=$done_marker traces=$traces last=[$last_line]" >> "$MONLOG"

    if [ "$done_marker" -ge 1 ]; then
        echo "CLEAN DONE (marker present, alive=$alive)"
        exit 0
    fi
    if [ "$alive" = "no" ]; then
        echo "CLEAN DEAD WITHOUT done marker (traces=$traces) - triage needed"
        exit 2
    fi
    if [ "$traces" -gt "$last_traces" ]; then
        echo "NEW TRACEBACK(S) in clean log ($last_traces -> $traces) - triage needed"
        exit 2
    fi
    last_traces=$traces
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "TIMEOUT after ${MAXH}h, job still running"
        exit 3
    fi
    sleep "$INTERVAL"
done
