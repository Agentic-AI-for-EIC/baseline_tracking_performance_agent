#!/bin/bash
# 10-minute monitor for the 26.07.1 reproduction pipeline.
#
# Usage: scripts/watch_repro_26071.sh [interval_seconds] [max_hours]
#
# Read-only. Appends one compact status line per interval to
# runs/repro_monitor.log and exits when the pipeline ends:
#   exit 0 - "[repro] DONE" marker present
#   exit 2 - driver gone WITHOUT the done marker (triage needed)
#   exit 3 - timed out waiting (still running; re-arm)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

INTERVAL="${1:-600}"
MAXH="${2:-16}"
RLOG="runs/reproduction_26071.log"
MONLOG="runs/repro_monitor.log"

deadline=$((SECONDS + MAXH * 3600))
# Baseline: a resumed log keeps FAIL markers of earlier, already-triaged
# attempts; only NEW failures/dones relative to startup are actionable.
base_fails=$(grep -c '^\[repro\] FAIL' "$RLOG" 2>/dev/null); base_fails=${base_fails:-0}
base_done=$(grep -c '^\[repro\] DONE' "$RLOG" 2>/dev/null); base_done=${base_done:-0}
iter=0
while true; do
    iter=$((iter + 1))
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    step=$(grep '^\[repro\]' "$RLOG" | tail -1 | sed 's/^\[repro\] //')
    done_n=$(grep -c '^\[repro\] DONE' "$RLOG")
    fails=$(grep -c '^\[repro\] FAIL' "$RLOG")
    driver=$(pgrep -f "bash scripts/run_reproduction_26071.sh" | head -1)
    child=$(pgrep -f "python3 -m trkperf" | head -1)
    if [ -n "$child" ]; then
        cinfo=$(ps -o etime=,time=,args= -p "$child" | awk '{printf "elapsed=%s cpu=%s [%s %s %s]", $1, $2, $4, $5, $6}')
        io=$(awk '/rchar/{printf "net+disk=%.1fGB", $2/1e9}' "/proc/$child/io" 2>/dev/null)
    else
        cinfo="no-trkperf-child"; io=""
    fi
    nout=$(ls output/ 2>/dev/null | grep -c 26071)
    cache=$(du -sh cache/clean26071_files cache/bkg26071_files 2>/dev/null | awk '{printf "%s:%s ", $2, $1}')
    (exec 3<>/dev/tcp/127.0.0.1/1294) 2>/dev/null && tun=tunnel=up || tun=tunnel=DOWN
    echo "[$ts] iter=$iter driver=${driver:-DEAD} step=[$step] $cinfo $io $tun out26071=$nout fails=$fails cache=$cache" >> "$MONLOG"

    if [ "$done_n" -gt "$base_done" ]; then echo "REPRO DONE"; exit 0; fi
    if [ "$fails" -gt "$base_fails" ]; then echo "REPRO FAIL seen (n=$fails) - triage"; exit 2; fi
    if [ -z "$driver" ] && [ "$done_n" -eq 0 ]; then
        echo "REPRO DRIVER DEAD without DONE marker - triage needed"; exit 2; fi
    if [ "$SECONDS" -ge "$deadline" ]; then echo "TIMEOUT after ${MAXH}h, still running"; exit 3; fi
    sleep "$INTERVAL"
done
