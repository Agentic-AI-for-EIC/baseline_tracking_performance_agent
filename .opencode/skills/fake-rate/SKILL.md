---
name: fake-rate
description: >-
  Compute ePIC craterlake fake-track rate — fraction of reconstructed
  CentralCKFTracks with no valid truth match (weight >= 0.5), binned in the
  track's OWN reconstructed pT (log) and eta (0.5 bins) — deliberately no
  species axis (a fake track has no true species) — then compared between the
  clean (Type 1) and +background (Type 2, bkg_mixed) datasets. Use when asked
  about fake tracks, ghost/background track rates, or reconstruction
  combinatorics under beam background. Warns that single-particle-gun samples
  are invalid for this metric.
---

# Fake rate

## What I do
Runs `python -m trkperf fake-rate ...` on the clean and bkg_mixed datasets,
then the clean-vs-bkg comparison. Output is one row per (pt_bin, eta_bin) —
**no species axis** — with `n_tracks`, `n_fake`, `fake_rate`,
`fake_rate_err`, `insufficient_stats`.

## When to use me
Any request about fake/ghost tracks, the fraction of reconstructed tracks
with no genuine truth match, or combinatorics/occupancy effects of beam
background on the reconstruction output. For the file discovery / escalation /
comparison / provenance details that apply to every metric, load the
`trkperf-grid-workflow` skill too.

## Metric definition
A reconstructed track is "fake" when no `CentralCKFTrackAssociations` match
to any truth particle has weight >= 0.5 (i.e. less than half its hits come
from a single truth particle). fake_rate = n_fake / n_tracks per
reconstructed-(pT, eta) bin, with binomial errors. Two deliberate
conventions:

- Binned in the **reconstructed** track's own (pT, eta) — a fake has no true
  momentum, so truth binning is impossible.
- **No species axis** — a fake track has no true species by construction.
  Never bin fake rate by species.

## Procedure
Follow `trkperf-grid-workflow` for discovery/escalation/compare. The commands:

Type 1 (clean; JLab, strict):
```sh
python -m trkperf fake-rate --file-list filelists/clean_150.txt \
    --dataset-tag clean --min-q2-tier 1
```
Type 2 (+background; flaky endpoint, tolerate skips and cache):
```sh
python -m trkperf fake-rate --file-list filelists/bkg_200.txt \
    --dataset-tag bkg_mixed --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
Batch bkg launches: `scripts/launch_bkg.sh` (detached; cheap to relaunch
after session teardown via the cache).

Compare:
```sh
python -m trkperf compare --metric fake-rate \
    --clean output/fake-rate_clean.json \
    --bkg output/fake-rate_bkg_mixed.json --out-dir output
```

Local, network-free smoke test first:
```sh
python -m trkperf fake-rate --file data/dataset_small/signal/RECO/*.root \
    --dataset-tag clean --min-q2-tier 1
```

## Metric-specific rules
- **NEVER use a single-particle-gun sample** (e.g. a single-track-gun test
  file) for fake rate: it has ~0 fakes by construction and is not a
  meaningful measurement. This metric is invalid without real DIS pileup +
  beam background.
- Statistics floor: `n_tracks >= 50` per bin; below that `insufficient_stats`.
- Small fake rates are best read on the log-y companion plot
  (`fake-rate_<tag>_vs_pt_logy.png`, ymin pinned at 1e-5).
- Outputs: `output/fake-rate_{clean,bkg_mixed}.{json,md,root,png}` (linear +
  log-y plots) and `output/fake-rate_comparison.{json,md}`.

## Definition of done
- Every quoted bin has >= 50 tracks or is explicitly flagged insufficient
  with the escalation tier attempted.
- The dataset/minQ2 tier(s) and file counts are named (fake rate is acutely
  sensitive to the background overlay and event topology).
- Trends are physically sensible; the clean sample should show a low,
  smoothly varying fake rate, and the bkg sample the expected occupancy
  increase. Always state dataset type(s), campaign, minQ2 tier(s), file
  counts, matching threshold (0.5), and the command used.