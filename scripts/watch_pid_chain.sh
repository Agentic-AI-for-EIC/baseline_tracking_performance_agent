#!/bin/bash
# 10-minute monitor for the PID 26.07.1 reproduction chain.
#
# Usage: scripts/watch_pid_chain.sh [interval_seconds] [max_hours]
#
# Read-only. One compact status line per interval appended to
# runs/pid_chain_monitor.log. Exits:
#   0 - "[chain] CHAIN DONE" present
#   2 - chain process gone without the done marker, or NEW tracebacks
#   3 - timed out waiting (re-arm)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
INTERVAL="${1:-600}"
MAXH="${2:-24}"

count_traces() {
    cat runs/pid_chain_26071.log runs/pid_features_clean26071.log \
        runs/pid_features_bkg26071.log 2>/dev/null | grep -c '^Traceback'
}
traces0=$(count_traces); traces0=${traces0:-0}
deadline=$((SECONDS + MAXH * 3600)); iter=0
while true; do
    iter=$((iter + 1)); ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    stage=$(grep '^\[chain\]' runs/pid_chain_26071.log 2>/dev/null | tail -1 | sed 's/^\[chain\] //')
    cpid=$(pgrep -f "pid_chain_26071" | head -1)
    jobsnap=$(ps -eo pid,etime,args --no-headers | awk '$3=="python" && $4=="-m" && $5=="pid" {
        out=$1" "$2" "$6;
        for(i=7;i<=NF;i++){if($i=="--task")out=out" task="$(i+1); if($i=="--model")out=out" model="$(i+1); if($i=="--dataset-tag")out=out" tag="$(i+1)}
        print out; exit}')
    job=""
    jobinfo=""
    if [ -n "$jobsnap" ]; then
        job=$(echo "$jobsnap" | awk '{print $1}')
        jobinfo=$(echo "$jobsnap" | cut -d' ' -f2-)
    fi
    feat=$(grep '^\[features\] .*files' runs/pid_features_clean26071.log runs/pid_features_bkg26071.log 2>/dev/null | tail -1 | sed 's/.*\[features\] //;s/ files//')
    gfail=$(grep -c '^\[gate:FAIL\]' runs/pid_features_clean26071.log runs/pid_features_bkg26071.log 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
    gpass=$(grep -h '^\[train\] artifacts' runs/pid_features_clean26071.log runs/pid_features_bkg26071.log 2>/dev/null | wc -l)
    done_n=$(grep -c '^\[chain\] CHAIN DONE' runs/pid_chain_26071.log 2>/dev/null); done_n=${done_n:-0}
    traces=$(count_traces)
    (exec 3<>/dev/tcp/127.0.0.1/1294) 2>/dev/null && tun=up || tun=DOWN
    echo "[$ts] iter=$iter chain=${cpid:-DEAD} stage=[$stage] job=[${jobinfo:-idle}] feat=$feat trained=$gpass gateFAIL=$gfail tunnel=$tun traces=$traces" >> runs/pid_chain_monitor.log
    [ "$done_n" -ge 1 ] && { echo "CHAIN DONE"; exit 0; }
    [ -z "$cpid" ] && { echo "CHAIN PROCESS DEAD without DONE marker - triage"; exit 2; }
    [ "$traces" -gt "$traces0" ] && { echo "NEW TRACEBACKS ($traces0 -> $traces) - triage"; exit 2; }
    [ "$SECONDS" -ge "$deadline" ] && { echo "TIMEOUT after ${MAXH}h, still running"; exit 3; }
    sleep "$INTERVAL"
done
