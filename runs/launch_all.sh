#!/bin/bash
# Launch all metric runs as separate background jobs.
set -e
RUNROOT="/home/wxie/eic/tracking_performance"
cd "$RUNROOT"

launch() {
  local name="$1"; shift
  nohup python -m trkperf "$@" \
    > "runs/${name}.log" 2>&1 &
  echo "launched $name (pid $!) -> runs/${name}.log"
}

# Clean (Type 1: 150 files)
launch acceptance_clean acceptance --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch efficiency_clean efficiency --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch resolution_clean resolution --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch pidconfusion_clean pid-confusion --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1

# bkg_mixed (Type 2: 200 files, the statistics bottleneck)
launch acceptance_bkg acceptance --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1
launch efficiency_bkg efficiency --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1
launch resolution_bkg resolution --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1
launch pidconfusion_bkg pid-confusion --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1

# Fake-rate: clean (gun ~0 fakes) and bkg_mixed (real fakes) - bkg is the meaningful one
launch fakerate_clean fake-rate --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch fakerate_bkg fake-rate --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1

echo "All jobs launched."
