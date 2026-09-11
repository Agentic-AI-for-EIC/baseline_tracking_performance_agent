---
name: pid-confusion
description: >-
  Compute the ePIC craterlake PID confusion matrix — for each truth species
  that was successfully track-matched, what species the reconstruction's
  mass-hypothesis PID assigns, row-normalised (how often X is classified as
  Y) and column-adjacent efficiency/purity — binned in truth pT (log) and
  truth eta (0.5 bins), one row per (bin, truth_species, reco_species), then
  compared between the clean (Type 1) and +background (Type 2, bkg_mixed)
  datasets. Use when asked about particle identification, misidentification,
  π/K confusion, or PID purity under beam background.
---

# PID confusion matrix

## What I do
Runs `python -m trkperf pid-confusion ...` on the clean and bkg_mixed
datasets, then the clean-vs-bkg comparison. Output is one row per (pt_bin,
eta_bin, truth_species, reco_species) with `n_matchedinbin`,
`confusion_frac`, `confusion_frac_err`, `efficiency`, `efficiency_err`,
`n_truth_total_in_bin`, `insufficient_stats`.

## When to use me
Any request about particle identification, misidentification rates, π/K (or
other species) confusion, or PID purity/efficiency vs pT/eta, including how
beam background shifts those numbers. For the file discovery / escalation /
comparison / provenance details that apply to every metric, load the
`trkperf-grid-workflow` skill too.

## Metric definition
Uses the same single truth<->reco matched-pairs table as the other matched
metrics (`CentralCKFTrackAssociations`, weight >= 0.5). The reconstruction's
assigned species is `reco_pdg` mapped through `config.PDG_TO_SPECIES`; a
`reco_pdg` outside the tracked species set maps to `"unknown"`, so you can
see what fraction of tracks the reconstructor failed to classify within
`config.SPECIES`.

- **confusion_frac** (row / truth-normalised): n_ij / n_i — fraction of truth
  species i classified as reco species j. Answers "how often is π called K?".
- **efficiency** (column-adjacent / truth-share): n_i-per-bin / n_bin —
  denominator is truth-relative, so it also functions as a per-species
  matched-share for each bin.

PID is assigned from the mass-hypothesis track parameters; only matched
tracks enter, so this metric is conditional on track finding.

## Procedure
Follow `trkperf-grid-workflow` for discovery/escalation/compare. The commands:

Type 1 (clean; JLab, strict):
```sh
python -m trkperf pid-confusion --file-list filelists/clean_150.txt \
    --dataset-tag clean --min-q2-tier 1
```
Type 2 (+background; flaky endpoint, tolerate skips and cache):
```sh
python -m trkperf pid-confusion --file-list filelists/bkg_200.txt \
    --dataset-tag bkg_mixed --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
Batch bkg launches: `scripts/launch_bkg.sh` (detached; cheap to relaunch
after session teardown via the cache).

Compare:
```sh
python -m trkperf compare --metric pid-confusion \
    --clean output/pid-confusion_clean.json \
    --bkg output/pid-confusion_bkg_mixed.json --out-dir output
```

Local, network-free smoke test first:
```sh
python -m trkperf pid-confusion --file data/dataset_small/signal/RECO/*.root \
    --dataset-tag clean --min-q2-tier 1
```

## Metric-specific rules
- Statistics floor: `n_truth_total_in_bin >= 50` per bin; below that the row
  is `insufficient_stats`.
- The comparison matrix is the primary deliverable; JSON/markdown only (no
  single-group PNG — a two-species-axis matrix needs no group plot).
  ROOT TNuple is still written (`output/pid-confusion_<tag>.root`, tree
  `pid_confusion`).
- Be explicit that PID is conditional on track matching — it measures
  misidentification of already-found tracks, not absolute ID rates.
- Outputs: `output/pid-confusion_{clean,bkg_mixed}.{json,md,root}` and
  `output/pid-confusion_comparison.{json,md}`.

## Definition of done
- Every quoted bin/row has >= 50 truth entries in the bin or is explicitly
  flagged insufficient with the escalation tier attempted.
- Diagonal (true-to-itself) fractions dominate off-diagonal, and the bkg
  comparison shows a physically sensible increase in confusion only in the
  expected phase space.
- Always state dataset type(s), campaign, minQ2 tier(s), file counts,
  matching threshold (0.5), and the command used.