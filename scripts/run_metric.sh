#!/bin/bash
# Usage: run_metric.sh <metric> <dataset_tag> <file_list> [max_file_failures] [extra args...]
# Runs a single trkperf metric on a file list, logging to runs/<metric>_<tag>.log
# Extra args (e.g. --region central --chunk-files 80 --cache-dir cache/x) are
# forwarded to `python -m trkperf` verbatim.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
METRIC="$1"; TAG="$2"; FILELIST="$3"; MAXFAIL="${4:-0}"; EXTRA=("${@:5}")
LOG="runs/${METRIC}_${TAG}.log"
exec 1>>"$LOG" 2>&1
python -m trkperf "$METRIC" --file-list "$FILELIST" --dataset-tag "$TAG" \
    --min-q2-tier 1 --max-file-failures "$MAXFAIL" "${EXTRA[@]}"
