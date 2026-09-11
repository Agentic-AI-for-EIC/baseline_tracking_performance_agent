#!/bin/bash
cd /home/wxie/eic/tracking_performance
METRIC="$1"; TAG="$2"; FILELIST="$3"
LOG="runs/${METRIC}_${TAG}.log"
exec python -m trkperf "$METRIC" --file-list "$FILELIST" --dataset-tag "$TAG" --min-q2-tier 1 \
  >> "$LOG" 2>&1
