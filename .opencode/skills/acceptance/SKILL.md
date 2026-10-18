---
name: acceptance
description: >-
  Compute ePIC craterlake geometric tracking acceptance — the fraction of
  generated truth particles that could in principle be reconstructed, defined
  as >= 4 individual truth hits (ACCEPTANCE_MIN_HITS), of which >= 3 fall in
  the silicon tracker (ACCEPTANCE_MIN_SILICON_HITS) — the CKF seed requirement
  — over the region's 7 central-tracking truth-hit collections — binned in truth p (log) and truth eta (0.5
  bins) per species, then compared between the clean (Type 1) and +background
  (Type 2, bkg26071) datasets. The single metric that needs no truth<->reco
  matching, so it doubles as a cross-check for campaign/geometry drift. Use
  when asked about detector coverage, geometric/truth acceptance, or whether
  a particle could be tracked at all.
---

# Geometric acceptance

## What I do
Runs `python -m trkperf acceptance ...` on the clean26071 and bkg26071 reproduction datasets,
then the clean-vs-bkg comparison. Acceptance output is one row per
(species, p_bin, eta_bin) with `n_generated`, `n_in_acceptance`,
`acceptance`, `acceptance_err`, `insufficient_stats`.

## When to use me
Any request about geometric acceptance, detector coverage, "how many
particles could be seen", or the campaign-drift cross-check. For the file
discovery / escalation / comparison / provenance details that apply to every
metric, load the `trkperf-grid-workflow` skill too.

## Metric definition
Acceptance = fraction of primary truth particles with >= 4 **individual** truth
hits, of which >= 3 in the silicon tracker (`SiBarrelHits`, `VertexBarrelHits`,
`TrackerEndcapHits`), counted over the region's central-tracking truth-hit
collection set (`SiBarrelHits`, `VertexBarrelHits`, `TrackerEndcapHits`,
`MPGDBarrelHits`, `OuterMPGDBarrelHits`, `BackwardMPGDEndcapHits`,
`ForwardMPGDEndcapHits`). The >= 3 silicon floor is the CKF seed requirement;
hit counts come from the per-collection truth-hit entries, so no decoded cellID
layer number is needed. Purely truth/geometry-level: no matching, no
reconstruction — therefore background-independent. For the canonical same-campaign
(26.07.1) reproduction pair it is the built-in check that the clean and
+background sides share an identical detector/reconstruction configuration: a
material clean-vs-bkg acceptance difference flags a processing mismatch, not a
background effect. (For the retired mixed-campaign 26.02.0-vs-26.07.1 pair the
same test instead flagged campaign/software/geometry drift.)

## Procedure
Follow `trkperf-grid-workflow` for discovery/escalation/compare. The commands:

Type 1 (clean; local 26071 reproduction over the login00 xrootd tunnel, strict):
```sh
python -m trkperf acceptance --file-list filelists/clean26071_local.txt \
    --dataset-tag clean26071 --min-q2-tier 1
```
Type 2 (+background; same local tunnel; keep --cache-dir + tolerate transient tunnel drops):
```sh
python -m trkperf acceptance --file-list filelists/bkg26071_local.txt \
    --dataset-tag bkg26071 --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
Batch bkg launches: `scripts/launch_bkg.sh` (detached; relaunch after a
session teardown is cheap thanks to the cache).

Compare:
```sh
python -m trkperf compare --metric acceptance \
    --clean output/acceptance_clean26071.json \
    --bkg output/acceptance_bkg26071.json --out-dir output
```

Local, network-free smoke test first:
```sh
python -m trkperf acceptance --file data/dataset_small/signal/RECO/*.root \
    --dataset-tag clean26071 --min-q2-tier 1
```

## Metric-specific rules
- Statistics floor: `n_generated >= 50` per bin (`MIN_ENTRIES_PER_BIN`);
  below that the bin is reported as `insufficient_stats`, not a number.
- Read the result JSON's metadata for `n_files`, `min_q2_tier`,
  `dataset_tag` before quoting a number.
- Outputs: `output/acceptance_clean26071,bkg26071.{json,md,root,png}` and
  `output/acceptance_comparison.{json,md}`.

## Definition of done
- Every quoted bin has >= 50 generated entries and is explicitly flagged
  insufficient with the attempted escalation tier.
- No large clean-vs-bkg acceptance difference. For the same-campaign (26.07.1)
  reproduction pair a material difference flags a processing/configuration
  mismatch, so the other comparison metrics must be read with that caveat — do
  NOT escalate statistics expecting it to shrink. (For the retired mixed-campaign
  26.02.0-vs-26.07.1 pair it instead signalled campaign/geometry drift.)
- Always state dataset type(s), campaign, minQ2 tier(s), file counts,
  matching threshold, and the command used.