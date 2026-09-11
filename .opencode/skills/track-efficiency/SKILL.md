---
name: track-efficiency
description: >-
  Compute ePIC craterlake track-finding efficiency — fraction of truth
  particles actually reconstructed as CentralCKFTracks, reported both
  within-acceptance (matched / in-acceptance truth) and absolute (matched /
  all generated truth), with binomial errors — binned in truth pT (log) and
  truth eta (0.5 bins) per species, then compared between the clean (Type 1)
  and +background (Type 2, bkg_mixed) datasets. Use when asked about
  reconstruction/tracking efficiency, matched fractions, or efficiency-loss
  under beam background.
---

# Track-finding efficiency

## What I do
Runs `python -m trkperf efficiency ...` on the clean and bkg_mixed datasets,
then the clean-vs-bkg comparison. Output is one row per (species, pt_bin,
eta_bin) with `n_truth`, `n_in_acceptance`, `n_matched`,
`efficiency_within_acceptance`, `efficiency_within_acceptance_err`,
`efficiency_absolute`, `efficiency_absolute_err`, `insufficient_stats`.

## When to use me
Any request about track-finding / reconstruction efficiency, matched-vs-truth
fractions, or how much efficiency is lost when beam background is overlaid.
For the file discovery / escalation / comparison / provenance details that
apply to every metric, load the `trkperf-grid-workflow` skill too.

## Metric definition
Efficiency uses truth<->reco matching via `CentralCKFTrackAssociations`
(weight >= 0.5, the standard majority-matching convention). Both numbers are measured
directly from data (matched / relevant denominator), not derived by
multiplying a separate acceptance table with a conditional efficiency — so
`efficiency_absolute` / acceptance need not exactly equal
`efficiency_within_acceptance`; cross-check against the `acceptance` metric.

- **within-acceptance**: matched & in-acceptance / in-acceptance truth.
- **absolute**: matched / all generated truth.

Acceptance uses the same >= 4-of-7 truth-hit-collection definition as the
`acceptance` metric (`ACCEPTANCE_MIN_LAYERS`); "layer" is a collection-level
proxy (documented simplification). Truth selection is primary-only.

## Procedure
Follow `trkperf-grid-workflow` for discovery/escalation/compare. The commands:

Type 1 (clean; JLab, strict):
```sh
python -m trkperf efficiency --file-list filelists/clean_150.txt \
    --dataset-tag clean --min-q2-tier 1
```
Type 2 (+background; flaky endpoint, tolerate skips and cache):
```sh
python -m trkperf efficiency --file-list filelists/bkg_200.txt \
    --dataset-tag bkg_mixed --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
Batch bkg launches: `scripts/launch_bkg.sh` (detached; cheap to relaunch
after session teardown via the cache).

Compare:
```sh
python -m trkperf compare --metric efficiency \
    --clean output/efficiency_clean.json \
    --bkg output/efficiency_bkg_mixed.json --out-dir output
```

Local, network-free smoke test first:
```sh
python -m trkperf efficiency --file data/dataset_small/signal/RECO/*.root \
    --dataset-tag clean --min-q2-tier 1
```

## Metric-specific rules
- Statistics floor: `n_in_acceptance >= 50` (the smaller denominator) per
  bin; below that the bin is `insufficient_stats`.
- **Memory**: the efficiency compute/merge is the heaviest of the five
  metrics (peaked at ~18 GB RSS on a 30 GB box, once OOM-killed). Do not
  launch it simultaneously with other heavy bkg jobs; the read cache means a
  kill/re-launch only redoes compute, not downloads.
- Outputs: `output/efficiency_{clean,bkg_mixed}.{json,md,root,png}`
  (absolute + within-acceptance vs pT plots) and
  `output/efficiency_comparison.{json,md}`.

## Definition of done
- Every quoted bin has >= 50 in-acceptance entries or is explicitly flagged
  insufficient with the escalation tier attempted.
- Trends are physically sensible; efficiency within acceptance should not be
  dramatically lower under background except in the expected noisy phase
  space. Before escalating to fix under-fed bins, sanity-check the
  `acceptance` comparison — a drift there flags campaign/geometry effects,
  not background.
- Always state dataset type(s), campaign, minQ2 tier(s), file counts,
  matching threshold (0.5), and the command used.