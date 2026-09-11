#!/bin/bash
cd /home/wxie/eic/tracking_performance
METRIC="$1"; TAG="$2"; FILELIST="$3"
LOG="runs/${METRIC}_${TAG}.log"
# Write PID to a marker file so we can track it
echo $$ > "runs/${METRIC}_${TAG}.pid"
# Redirect stdout/stderr to log via shell-level redirect
python -m trkperf "$METRIC" --file-list "$FILELIST" --dataset-tag "$TAG" --min-q2-tier 1 \
  >> "$LOG" 2>&1
