#!/bin/bash
set -e
cd /home/wxie/eic/tracking_performance

launch() {
  local name="$1"; shift
  nohup python -m trkperf "$@" \
    > "runs/${name}.log" 2>&1 &
  echo "launched $name (pid $!)"
}

# Fake-rate is light (no truth read, just reco+assoc), launch both immediately
launch fakerate_clean fake-rate --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch fakerate_bkg fake-rate --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1

# Clean jobs (reliable JLab endpoint): launch all 4 immediately
launch acceptance_clean acceptance --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch efficiency_clean efficiency --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch resolution_clean resolution --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1
launch pidconfusion_clean pid-confusion --file-list filelists/clean_150_clean.txt --dataset-tag clean --min-q2-tier 1

# Bkg jobs (flaky ASGC endpoint): delay 10 min to reduce endpoint pressure
echo "Waiting 10 min before launching bkg jobs to reduce ASGC endpoint pressure..."
sleep 600

launch acceptance_bkg acceptance --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1
launch efficiency_bkg efficiency --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1
launch resolution_bkg resolution --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1
launch pidconfusion_bkg pid-confusion --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed --min-q2-tier 1

echo "All jobs launched."
