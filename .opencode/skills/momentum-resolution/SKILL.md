---
name: momentum-resolution
description: >-
  Compute ePIC craterlake momentum resolution: sigma of a ROOT Gaussian core
  fit (TH1F.Fit(TF1("gaus")), two-pass refit) to Delta(pT)/pT between
  reconstructed and true central tracks, for truth-matched pairs, binned in
  truth pT (log) and truth eta (0.5 bins) per species, then compared between
  the clean (Type 1) and +background (Type 2, bkg_mixed) datasets. Covers the
  R<1 physical-maximum convention, fit_converged handling, and the
  clean-vs-bkg comparison. Use when asked about momentum/pT resolution,
  resolution sigma vs pT, or a resolution-vs-background study.
---

# Momentum resolution

## What I do
Runs `python -m trkperf resolution ...` on the clean and bkg_mixed datasets,
then the clean-vs-bkg comparison. Output is one row per (species, pt_bin,
eta_bin) with `n_matched`, `mu`, `sigma`, `mu_err`, `sigma_err`, `chi2_ndf`,
`fit_converged`, `insufficient_stats`.

## When to use me
Any request about momentum/pT resolution, resolution-vs-pT trends, or the
impact of beam background on momentum reconstruction. For the file discovery /
escalation / comparison / provenance details that apply to every metric, load
the `trkperf-grid-workflow` skill too.

## Metric definition
Resolution is defined on **pT specifically**: Delta(pT)/pT =
(pT_reco - pT_truth) / pT_truth, consistent with the pT/eta binning used
everywhere. Only truth-matched pairs enter (weight >= 0.5 via
`CentralCKFTrackAssociations`). Per bin, the Delta(pT)/pT distribution is fit
to its Gaussian core with **ROOT** (`TH1F.Fit(TF1("gaus"))`, two-pass core
refit over +/- `RESOLUTION_FIT_CORE_SIGMA` sigma) — not scipy, so a human
re-fit in a ROOT session reproduces the numbers.

The reported sigma is physically bounded to [0, 1] on Delta(pT)/pT
(`RESOLUTION_MAX_SIGMA` = 1.0 means 100% resolution). A fit whose sigma pins
against the bound is reported `fit_converged=False`, never as a big number.

Reco momentum comes from `CentralCKFTrackParameters` (`qOverP`, `theta`,
`phi`) — never from `CentralCKFTracks.momentum.*`, which is all zeros in
every campaign file (AGENTS.md).

## Procedure
Follow `trkperf-grid-workflow` for discovery/escalation/compare. The commands:

Type 1 (clean; JLab, strict):
```sh
python -m trkperf resolution --file-list filelists/clean_150.txt \
    --dataset-tag clean --min-q2-tier 1
```
Type 2 (+background; flaky endpoint, tolerate skips and cache):
```sh
python -m trkperf resolution --file-list filelists/bkg_200.txt \
    --dataset-tag bkg_mixed --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
Batch bkg launches: `scripts/launch_bkg.sh` (detached; cheap to relaunch
after session teardown via the cache).

Compare:
```sh
python -m trkperf compare --metric resolution \
    --clean output/resolution_clean.json \
    --bkg output/resolution_bkg_mixed.json --out-dir output
```

Local, network-free smoke test first:
```sh
python -m trkperf resolution --file data/dataset_small/signal/RECO/*.root \
    --dataset-tag clean --min-q2-tier 1
```

## Metric-specific rules
- Statistics floor: `n_matched >= 50` per bin; below that the fit is skipped
  and the bin flagged `insufficient_stats` (mu/sigma NaN, fit_converged
  False).
- Check `chi2_ndf` for non-converged / poor fits: a chi2/ndf far from order 1
  is a red flag even when `fit_converged` is True.
- `sigma` must always lie in [0, 1]; if a bin's raw width exceeds that it is
  flagged non-converged, not reported as a huge number.
- Outputs: `output/resolution_{clean,bkg_mixed}.{json,md,root,png}` (the
  linear + log-y sigma-vs-pT plots) and `output/resolution_comparison.{json,md}`.

## Definition of done
- Every quoted bin has >= 50 matched entries or is explicitly flagged
  insufficient with the escalation tier attempted.
- Converged fits have chi2/ndf ~ 1 and sigma within [0, 1].
- Trends are physically sensible (resolution degrades roughly with 1/pT and
  worsens with background in the expected phase space); a wildly
  non-monotonic point is flagged, not silently reported.
- Always state dataset type(s), campaign, minQ2 tier(s), file counts,
  matching threshold (0.5), and the command used.