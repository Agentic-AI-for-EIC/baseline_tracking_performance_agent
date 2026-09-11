---
name: acceptance
description: >-
  Compute ePIC craterlake geometric tracking acceptance — the fraction of
  generated truth particles that could in principle be reconstructed, defined
  as >= 4 (ACCEPTANCE_MIN_LAYERS) of the 7 central-tracking truth-hit
  collections registering a hit — binned in truth pT (log) and truth eta (0.5
  bins) per species, then compared between the clean (Type 1) and +background
  (Type 2, bkg_mixed) datasets. The single metric that needs no truth<->reco
  matching, so it doubles as a cross-check for campaign/geometry drift. Use
  when asked about detector coverage, geometric/truth acceptance, or whether
  a particle could be tracked at all.
---

# Geometric acceptance

## What I do
Runs `python -m trkperf acceptance ...` on the clean and bkg_mixed datasets,
then the clean-vs-bkg comparison. Acceptance output is one row per
(species, pt_bin, eta_bin) with `n_generated`, `n_in_acceptance`,
`acceptance`, `acceptance_err`, `insufficient_stats`.

## When to use me
Any request about geometric acceptance, detector coverage, "how many
particles could be seen", or the campaign-drift cross-check. For the file
discovery / escalation / comparison / provenance details that apply to every
metric, load the `trkperf-grid-workflow` skill too.

## Metric definition
Acceptance = fraction of primary truth particles with hits in >= 4 of the 7
central-tracking truth-hit collections (`SiBarrelHits`, `VertexBarrelHits`,
`TrackerEndcapHits`, `MPGDBarrelHits`, `OuterMPGDBarrelHits`,
`BackwardMPGDEndcapHits`, `ForwardMPGDEndcapHits`). "Layer" here means
"distinct collection hit" — a collection-level proxy for a physical sensor
layer (documented simplification; decoding true layer numbers would need
subsystem cellID geometry). Purely truth/geometry-level: no matching, no
reconstruction — therefore background-independent and the built-in
cross-check for campaign/software/geometry drift between 26.02.0 and 26.07.1.

## Procedure
Follow `trkperf-grid-workflow` for discovery/escalation/compare. The commands:

Type 1 (clean; JLab, strict):
```sh
python -m trkperf acceptance --file-list filelists/clean_150.txt \
    --dataset-tag clean --min-q2-tier 1
```
Type 2 (+background; flaky endpoint, tolerate skips and cache):
```sh
python -m trkperf acceptance --file-list filelists/bkg_200.txt \
    --dataset-tag bkg_mixed --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
Batch bkg launches: `scripts/launch_bkg.sh` (detached; relaunch after a
session teardown is cheap thanks to the cache).

Compare:
```sh
python -m trkperf compare --metric acceptance \
    --clean output/acceptance_clean.json \
    --bkg output/acceptance_bkg_mixed.json --out-dir output
```

Local, network-free smoke test first:
```sh
python -m trkperf acceptance --file data/dataset_small/signal/RECO/*.root \
    --dataset-tag clean --min-q2-tier 1
```

## Metric-specific rules
- Statistics floor: `n_generated >= 50` per bin (`MIN_ENTRIES_PER_BIN`);
  below that the bin is reported as `insufficient_stats`, not a number.
- Read the result JSON's metadata for `n_files`, `min_q2_tier`,
  `dataset_tag` before quoting a number.
- Outputs: `output/acceptance_{clean,bkg_mixed}.{json,md,root,png}` and
  `output/acceptance_comparison.{json,md}`.

## Definition of done
- Every quoted bin has >= 50 generated entries and is explicitly flagged
  insufficient with the attempted escalation tier.
- No large clean-vs-bkg acceptance difference. A material difference means
  the 26.02.0 vs 26.07.1 campaigns differ geometrically/software-wise and the
  other comparison metrics must be read as "campaign difference" as well —
  do NOT escalate statistics expecting the drift to shrink.
- Always state dataset type(s), campaign, minQ2 tier(s), file counts,
  matching threshold, and the command used.