---
name: trkperf-grid-workflow
description: >-
  End-to-end procedure for running any trkperf tracking-performance metric on
  the ePIC craterlake grid: discover files with the rucio/xrootd MCP tools,
  run the clean (Type 1) and +background (Type 2, bkg_mixed) datasets with the
  right flags, escalate under-populated bins by file count then minQ2 tier,
  and write the clean-vs-bkg comparison with full provenance. Use before
  running acceptance, momentum resolution, track efficiency, fake rate, or PID
  confusion so dataset selection, commands, thresholds, elevation order, and
  reproducibility rules are applied consistently across all of them.
---

# trkperf grid-workflow

## What I do
Stands for the shared, metric-independent parts of every tracking-performance
run in this project:

1. Discover which grid files exist for the current tier via the **rucio MCP**
   tools (`list_dids`, `list_files`, `list_file_replicas`), and sanity-check a
   file with the **xrootd MCP** tools (`check_file_exists`,
   `get_file_info`) before a large run.
2. Run the metric on **both** dataset types: Type 1 clean and Type 2
   +background (`bkg_mixed`), using `python -m trkperf <metric> ...`.
3. Escalate statistics in the documented order when a bin is under-fed.
4. Run the clean-vs-bkg comparison and write JSON/markdown/plots/ROOT with
   provenance.
5. Apply the project's flaky-endpoint, detachment, and OOM conventions when
   launching long bkg jobs.

Never hardcode a file list beyond the two dataset templates in AGENTS.md;
build file lists with the rucio MCP tools and reuse the ones already under
`filelists/` where possible. Never write bespoke one-off I/O outside `trkperf`
(AGENTS.md).

## When to use me
Load this skill **before** running any metric (directly, or via the metric
skills `acceptance`, `momentum-resolution`, `track-efficiency`, `fake-rate`,
`pid-confusion`). Use it whenever a user asks to "run the analysis", "extend
statistics", "redo the comparison", or "check a bin" on either dataset.

## Datasets at a glance (from trkperf/config.py)

| Type | Dataset tag | Campaign | DID (template) | xrootd prefix | Files | Events/file |
|---|---|---|---|---|---|---|
| 1 clean | `clean` | 26.02.0 | `epic:/RECO/26.02.0/epic_craterlake/DIS/NC/10x100/minQ2=1` | `root://dtn-eic.jlab.org:1094//volatile/eic/EPIC/RECO/26.02.0/epic_craterlake/DIS/NC/10x100/minQ2=1` | 4,609 (minQ2=1) | 1,083 |
| 2 bkg | `bkg_mixed` | 26.07.1 | `epic:/RECO/26.07.1/epic_craterlake/Bkg_Exact1S_2us/GoldCt/10um/DIS/NC/10x100/minQ2={min_q2}` | `root://hpceph-xrootd.twgrid.org:1094//cephfs/epic/RECO/26.07.1/epic_craterlake/Bkg_Exact1S_2us/GoldCt/10um/DIS/NC/10x100/minQ2={min_q2}` | 1,463/902/547/551 for minQ2=1/10/100/1000 | ~99 |

Type 2 is the statistics bottleneck (~11x fewer events/file than Type 1):
matching file *counts* between types does not mean matching statistics.

## Procedure

### 1. Discover / pick the file list
- Reuse an existing list under `filelists/` if it covers the tier needed
  (e.g. `clean_150.txt`, `clean_full.txt`, `bkg_200.txt`,
  `bkg_full_minQ2_1.txt`, `bkg_full.txt`).
- Otherwise resolve paths with the `rucio` MCP server: `list_dids` on the DID
  above, `list_files` for the dataset, `list_file_replicas` to get the
  `root://` URLs. Save URLs one-per-line into a `filelists/*.txt`.
- For a quick network-free iteration use the local reference pair in
  `data/dataset_small/{signal,signal_BKG_mix}` (test files, not a real run).

### 2. Run the metric on Type 1 (clean)
JLab endpoint is reliable; keep `--max-file-failures` at 0:
```sh
python -m trkperf <metric> --file-list filelists/clean_150.txt \
    --dataset-tag clean --min-q2-tier 1
```
Metrics and their `--species` behavior (all but `fake-rate` take
`--species {e-|e+|pi+|pi-|K+|K-|proton|antiproton}`; default is all):
`acceptance`, `efficiency`, `resolution`, `fake-rate` (no species axis),
`pid-confusion`.

### 3. Run the metric on Type 2 (+background, bkg_mixed)
The overseas endpoint is flaky (`[ERROR] Operation expired` / socket
timeouts). Always:
- pass `--max-file-failures N` (5-20 is enough to ride out transient drops;
  the project historically used 60 for heavily degraded stretches);
- pass `--cache-dir` (no default — pass `cache/bkg_files`) so a job interrupted by
  container teardown re-start for free over already-read files.

```sh
python -m trkperf <metric> --file-list filelists/bkg_200.txt \
    --dataset-tag bkg_mixed --min-q2-tier 1 \
    --max-file-failures 20 --cache-dir cache/bkg_files
```
For the five-metric batch use `scripts/launch_bkg.sh [filelist] [max_failures]
[cache_dir]`; it detaches each job with `setsid`+`nohup` and writes
`runs/<metric>_bkg_mixed.{log,pid,start}`. Remember sessions tear down their
container when they end, so a multi-hour bkg job may need relaunching —
the cache makes each relaunch cheap.

### 4. Escalate statistics (only if a bin is under-fed)
Per AGENTS.md, a bin is only trusted at >= `MIN_ENTRIES_PER_BIN` (50). The
metric output flags `insufficient_stats`; if you need numbers where a bin is
flagged:
1. First add more files from the **current** minQ2 tier (Type 1: up to 4,609
   minQ2=1 files; Type 2: up to 1,463 minQ2=1 files).
2. Only when the whole tier is used, widen **Type 2 only** across minQ2 tiers
   `1 -> 10 -> 100 -> 1000` (`--min-q2-tier 10` etc., and rebuild the file list
   for that tier). Type 1 stays at minQ2=1 always.
3. With every result record which tier/files were actually used
   (`--min-q2-tier`, file list, file count).

Note the OOM risk on this 30 GB box: the efficiency compute/merge peaked at
~18 GB RSS; do not stack heavy bkg jobs simultaneously.

### 5. Compare clean vs bkg_mixed
Requires both sides' JSON in `output/` (`output/<metric>_clean.json` and
`output/<metric>_bkg_mixed.json`):
```sh
python -m trkperf compare --metric <metric> \
    --clean output/<metric>_clean.json \
    --bkg output/<metric>_bkg_mixed.json --out-dir output
```
or run all five via `scripts/run_comparisons.sh` (skips metrics whose inputs
are missing; repeat until it prints no "waiting"). Writes
`output/<metric>_comparison.{json,md}` with `*_ratio_bkg_over_clean[_err]` and
`*_diff_bkg_minus_clean[_err]` plus a combined `insufficient_stats` flag.

### 6. Report with provenance
Every result already writes `output/<metric>_<tag>.{json,md,root}` (+ PNGs).
Before quoting a number, always state: dataset type(s), campaign, minQ2
tier(s), file counts read vs skipped, matching threshold (0.5), which heavies
are `primary_only=True`, and the full `python -m trkperf` command. `report.py`
stamps `metric`/`dataset_tag`/`min_q2_tier`/`n_files` into each JSON's
metadata; re-read via `python -m trkperf plot --json output/<metric>_<tag>.json`
to re-render plots without touching the grid.

## Pitfalls / conventions
- Recovery of a truncated/failed run: relaunch with the same `--cache-dir`
  and file list; already-read files replay from local pickles.
- A timed-out MCP call means the server is busy — wait, retry once, then stop
  and report which tool/arguments failed. Do not silently reuse stale results.
- Clean-vs-bkg differences could reflect campaign/geometry drift (26.02.0 vs
  26.07.1), not only background. `acceptance` is background-independent by
  construction — a material clean-vs-bkg difference there flags drift; check
  it before escalating for an efficiency/fake-rate effect.
- A single-particle-gun sample has ~0 fakes by construction — never use it
  for the fake-rate metric.
- Run smoke tests locally before any grid access: `python -m unittest
  discover -s tests -v` (stdlib unittest; no pytest in this env).