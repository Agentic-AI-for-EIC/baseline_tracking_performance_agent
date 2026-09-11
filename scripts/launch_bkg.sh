#!/bin/bash
# Launch all five +background (Type 2, bkg_mixed) metric runs as detached
# background jobs against the flaky overseas xrootd endpoint.
# Each job tolerates up to MAXFAIL failing files (skipped with warnings).
#
# Detachment note: these are multi-hour jobs and the per-session container is
# torn down when the session ends. setsid() puts each job in its own session
# (best-effort protection against process-group cleanup), and --cache-dir makes
# every relaunch free over files already read, so lost progress between
# sessions costs CPU only, not re-downloads.
#
# Usage: scripts/launch_bkg.sh [filelist] [max_failures] [cache_dir]
set -e
cd /home/wxie/eic/tracking_performance
FILELIST="${1:-filelists/bkg_200.txt}"
MAXFAIL="${2:-60}"
CACHE="${3:-cache/bkg_files}"
PY=python

launch() {
  local name="$1"; shift
  local log="runs/${name}.log"
  > "$log"
  setsid nohup "$PY" -m trkperf "$@" --max-file-failures "$MAXFAIL" \
    --cache-dir "$CACHE" \
    > "$log" 2>&1 < /dev/null &
  echo "$!" > "runs/${name}.pid"
  date +%Y-%m-%dT%H:%M:%S > "runs/${name}.start"
  echo "launched $name (pid $!) -> $log"
}

launch acceptance_bkg_mixed acceptance --file-list "$FILELIST" --dataset-tag bkg_mixed --min-q2-tier 1
launch efficiency_bkg_mixed efficiency --file-list "$FILELIST" --dataset-tag bkg_mixed --min-q2-tier 1
launch resolution_bkg_mixed resolution --file-list "$FILELIST" --dataset-tag bkg_mixed --min-q2-tier 1
launch fake-rate_bkg_mixed fake-rate --file-list "$FILELIST" --dataset-tag bkg_mixed --min-q2-tier 1
launch pid-confusion_bkg_mixed pid-confusion --file-list "$FILELIST" --dataset-tag bkg_mixed --min-q2-tier 1

echo "All bkg_mixed jobs launched (filelist=$FILELIST, max_file_failures=$MAXFAIL, cache_dir=$CACHE)."