# Tracking Performance Analysis — Project Plan
Location: /home/wxie/eic/baseline_tracking_performance_agent/

Status: implemented — metric package + clean (Type 1) results done; the
+background (Type 2) grid runs for all five metrics (200 files, minQ2=1) are
now COMPLETE and so are all five clean-vs-bkg comparisons (steps 12-14).
Statistics escalation (step 15) was evaluated and left as a documented
decision point. This file is the as-built record of the plan agreed across
the planning conversation, kept for documentation purposes (checklist below
tracks what has actually been carried out).

## 0. Goal
Measure ePIC craterlake tracking performance — geometric acceptance, momentum
resolution, track-finding efficiency, fake rate — as a function of transverse
momentum (pT), pseudorapidity (eta), and truth particle species, by matching
true GEANT tracks (MCParticles) to reconstructed tracks (CentralCKFTracks).
All four metrics are computed for BOTH dataset types (clean / +background);
the primary deliverable per metric is the comparison (ratio + difference)
between them, quantifying the impact of beam-induced background.

Tracking scope covers three detector regions — central, forward (hadron
endcap) and backward (electron endcap) — see §4.1. Acceptance and the
within-acceptance efficiency denominator are defined per region; resolution,
fake rate and pid-confusion are region-agnostic (binned in η everywhere, so
the endcap bins were always present but thin).

## 1. opencode setup
- Self-contained /home/wxie/eic/baseline_tracking_performance_agent/opencode.jsonc:
  - "model": "nrp/Qwen3"
  - "small_model": "nrp/gemma-small"
  - providers: NRP LLM (OpenAI-compatible) + Google, tokens via
    ~/.nrp-llm-token and ~/.gemini-token
  - mcp block: uproot/xrootd/rucio remote servers on 127.0.0.1:9101-9103
    (same eic-mcp instance already running for /home/wxie/eic)
- .github/copilot-instructions.md:
  one line, "Follow the project rules in AGENTS.md."
- Rationale: opencode discovers AGENTS.md/opencode.json by walking UP from
  cwd, not down — so this directory needs its own copies rather than relying
  on inheritance from /home/wxie/eic. Launch opencode with
  baseline_tracking_performance_agent itself as the root.

## 2. Project layout
```
/home/wxie/eic/baseline_tracking_performance_agent/
├── AGENTS.md
├── PLAN.md                             # this file
├── opencode.jsonc
├── .github/copilot-instructions.md
├── .opencode/skills/
│   ├── acceptance/SKILL.md
│   ├── momentum-resolution/SKILL.md
│   ├── track-efficiency/SKILL.md
│   ├── fake-rate/SKILL.md
│   ├── pid-confusion/SKILL.md
│   ├── pid-ml/SKILL.md
│   ├── pid-performance/SKILL.md
│   └── trkperf-grid-workflow/SKILL.md
├── data/dataset_small/                 # symlinks -> existing local files
│   ├── signal/{GEANT4,RECO}/...        # campaign 26.02.0, clean
│   └── signal_BKG_mix/{GEANT4,RECO}/...# campaign 26.07.1, +background
├── trkperf/                            # importable, modularized analysis package
│   ├── __init__.py
│   ├── config.py         # species PDG table; pT/eta bin edges; match-weight threshold;
│   │                      # acceptance N_min; dataset registry (paths + minQ2 tiers + escalation order)
│   ├── io.py              # open local path or root:// URL with uproot; generic jagged->flat
│   │                      # DataFrame helper; no physics
│   ├── truth.py           # MCParticles -> pT/eta/species; truth-hit layer counts for acceptance
│   ├── reco.py            # CentralCKFTracks -> reconstructed pT/eta/charge/pdg
│   ├── matching.py        # CentralCKFTrackAssociations -> shared truth<->reco matched-pairs table,
│   │                      # and the fake-track finder
│   ├── binning.py         # bin-edge assignment, Gaussian-core fit helper
│   ├── metrics/
│   │   ├── acceptance.py, resolution.py, efficiency.py, fake_rate.py
│   │   # (pid-confusion lives in pid/confusion.py below; the ML PID pipeline
│   │   # itself is the top-level pid/ package, documented in PLAN_pid.md)
│   ├── compare.py         # reads two same-metric JSON results (clean, bkg), aligns bins,
│   │                      # emits per-bin ratio + difference (+propagated uncertainty)
│   ├── pid/                # PID-confusion metric (consumes matching.py's
│   │                       # matched-pairs table) + pointer to the top-level
│   │                       # pid/ ML package
│   ├── report.py          # JSON + markdown-table + plot writers
│   └── cli.py              # `python -m trkperf <metric|compare|plot> ...`
├── tests/                   # unittest smoke tests against data/dataset_small (no network)
├── filelists/               # frozen file lists for grid runs (clean_150, bkg_200, full tiers)
├── scripts/                 # launch_bkg.sh, run_metric.sh, run_comparisons.sh, job_report.sh, watch_jobs.sh
├── runs/                    # per-job logs + pidfiles (not committed)
├── cache/                   # per-file read-cache pickles for flaky-endpoint runs (not committed)
├── pid/                     # ML particle-ID pipeline (see PLAN_pid.md)
└── output/                  # JSON/plots/tables (committed run artifacts)
```

## 3. Datasets

| Sample | Campaign | minQ2 bin(s) | Files available | Events/file | Total events | File size |
|---|---|---|---|---|---|---|
| Type 1: clean | 26.02.0 | 1 (fixed) | 4,609 | 1,083 | ~4.99M | 192.6 MB |
| Type 2: +background | 26.07.1 | 1 | 1,463 | 99 | ~145k | 546.9 MB |
| Type 2: +background | 26.07.1 | 10 | 902 | ~99* | ~89k* | ~547 MB* |
| Type 2: +background | 26.07.1 | 100 | 547 | ~99* | ~54k* | ~547 MB* |
| Type 2: +background | 26.07.1 | 1000 | 551 | ~99* | ~55k* | ~547 MB* |

(*minQ2=10/100/1000 event-count/size assumed ~ minQ2=1's; only minQ2=1 was
opened and verified with uproot.)

Confirmed real xrootd replicas (via `rucio replica list file`):
- Type 1: root://dtn-eic.jlab.org:1094//volatile/eic/EPIC/RECO/26.02.0/epic_craterlake/DIS/NC/10x100/minQ2=1/...
- Type 2: root://hpceph-xrootd.twgrid.org:1094//cephfs/epic/RECO/26.07.1/epic_craterlake/Bkg_Exact1S_2us/GoldCt/10um/DIS/NC/10x100/minQ2={1,10,100,1000}/...

Adaptive escalation strategy:
1. Start: Type 1 = 150 files (minQ2=1, ~162k events). Type 2 = minQ2=1 files,
   scaling up toward all 1,463 as needed (~145k events max there).
2. Type 2 is the statistics bottleneck (~11x fewer events/file than Type 1) —
   matching file COUNTS between types does not mean matching statistics.
3. If a (pT,eta,species) bin is still below the ~50-entry floor: (a) add more
   files from the current minQ2 tier first; (b) only once all files in that
    tier are used, add the next minQ2 tier (1 -> 10 -> 100 -> 1000) for Type 2
    only. Type 1 stays at minQ2=1 (matches how it was produced).
4. Every result records which tier was actually reached.

Caveat (methodological honesty): Type 1 (26.02.0) and Type 2 (26.07.1) are
different production campaigns, not the same campaign with background
toggled — a measured difference could in principle reflect campaign-level
software/geometry drift, not only the background overlay. `acceptance` is
purely truth/geometry-level and should be background-independent; treat it as
a built-in cross-check.

## 4. Data model (verified live against real files this session)
- MCParticles: .PDG, .generatorStatus, .momentum.x/y/z (double), .mass (double)
- CentralCKFTracks (edm4eic::TrackData): .momentum.x/y/z (float), .charge,
  .pdg, .chi2, .ndf, plus measurements/tracks/trajectory relations
- CentralCKFTrackParameters (edm4eic::TrackParametersData): .type, .surface,
  .loc.a/b, .theta, .phi, .qOverP, .time, .pdg, .covariance.covariance[21]
  (the ONLY source of reconstructed momentum here: CentralCKFTracks.momentum
  is all zeros in these productions, so p = 1/|qOverP| with px/py/pz from
  theta/phi - see trkperf/reco.py)
- CentralCKFTrackAssociations (edm4eic::MCRecoTrackParticleAssociationData):
  .weight + relations _..._rec (-> CentralCKFTracks), _..._sim (-> MCParticles)
- Truth tracking hits (edm4hep::SimTrackerHitData), each with a
  `_<Collection>_particle` relation to MCParticles: SiBarrelHits,
  VertexBarrelHits, TrackerEndcapHits, MPGDBarrelHits, OuterMPGDBarrelHits,
  BackwardMPGDEndcapHits, ForwardMPGDEndcapHits.

"Layer" for acceptance purposes = one of these 7 collections having >=1 hit
from the particle (a collection-level proxy; not a decoded physical sensor
layer number, which would need subsystem-specific cellID geometry decoding).

### 4.1 Tracking regions: central, forward (hadron endcap), backward (electron endcap)

There is no ForwardCKFTracks/BackwardCKFTracks collection in these
productions: endcap tracks are reconstructed as `CentralCKFTracks` (the CKF
extends through the endcap disks), so efficiency/resolution/fake-rate need no
new reconstruction chain — but the central ≥4/7 acceptance rule calls nearly
every endcap particle out-of-acceptance (backward e⁻: 0.002), which is
correct yet useless. Acceptance and the within-acceptance efficiency
denominator are therefore defined per region (`trkperf.config.TRACKING_REGIONS`,
`--region` on the acceptance/efficiency CLI):

| region | truth-hit collections (≥1 hit each) | N_min | measured joint acceptance (clean file) |
|---|---|---|---|
| central | the 7 collections above | 4 | unchanged behaviour |
| backward (electron endcap) | BackwardMPGDEndcapHits, TrackerEndcapHits, TOFEndcapHits | 2 | backward e⁻: ≥1 = 1.000, ≥2 = 0.968, ≥3 = 0.002 |
| forward (hadron endcap) | ForwardMPGDEndcapHits, TrackerEndcapHits, TOFEndcapHits, ForwardOffMTrackerHits, ForwardRomanPotHits | 2 | forward π/K/p: ≥1 = 0.519, ≥2 = 0.426, ≥3 = 0.271 |

Rationale: N_min = 2 is the stereo minimum (two independent measurements make
a segment); the joint fractions above are the measured justification, with
≥1/≥3 as brackets. TOF endcap hits count: TOF cluster hits are folded into
the track fit (26.07.1), so they are tracking-relevant space points, not
calorimetry. A collection whose branch is absent in a campaign degrades to 0
hits with a printed NOTE rather than aborting the run.

Explicitly out of scope (measured, not assumed):
- `B0TrackerCKFTracks` (+ its associations/parameters) exist but are EMPTY
  (0 entries/event) in these DIS files — B0 sees diffractive far-forward
  protons, absent here. B0 metrics need a diffractive sample, not code.
- `TaggerTrackerHits` is likewise empty in DIS (beam tagger) and excluded
  from the backward set for the same reason.
- No `SiEndcapTrackerHits` truth relation exists under that name, so silicon
  endcap disks enter only via `TrackerEndcapHits`.

Momentum resolution is defined on pT specifically: Delta(pT)/pT =
(pT_reco - pT_truth) / pT_truth, consistent with binning everything in pT/eta.

## 5. Skills (.opencode/skills/<name>/SKILL.md)
The four tracking skills (acceptance, momentum-resolution, track-efficiency,
fake-rate) plus pid-confusion share one workflow: discover files for the
current tier via `rucio` MCP -> run
`python -m trkperf <metric> --file-list ... --species ... --dataset-tag
{clean,bkg_mixed} --min-q2-tier ...` on Type 1 AND Type 2 -> escalate per the
order above if under-populated -> run `python -m trkperf compare --metric <metric>
--clean ... --bkg ...` -> report JSON/table/plot with provenance. Grouped
plots (eta regions barrel/forward/backward endcap × merged species e±/π±/K±
with p/pbar kept separate, recomputed from summed counts) re-render from any
result JSON with `python -m trkperf plot --json ... --grouped`.

## 6. Environment/tooling already confirmed working
- GitHub Copilot provider: authenticated (oauth); 17 models available.
- eic-mcp servers (uproot/xrootd/rucio) running on 127.0.0.1:9101-9103.
- Python environment has uproot, awkward, numpy, pandas, scipy, matplotlib
  (no pytest — tests use the stdlib `unittest` module instead).

## 7. Execution checklist (all steps below were carried out)
- [x] 1. Deleted stray /home/wxie/opencode.jsonc (accidental side effect from an
   earlier `eic-mcp config opencode` invocation).
- [x] 2. Verified exact branch schema for CentralCKFTrackParameters,
   CentralCKFTrackAssociations, CentralCKFTracks, and the truth-hit ->
   MCParticles relations directly against a real local file before writing
   any code that depends on them.
- [x] 3. Created the directory tree in section 2.
- [x] 4. Wrote opencode.jsonc, .github/copilot-instructions.md.
- [x] 5. Symlinked data/dataset_small/{signal,signal_BKG_mix}.
- [x] 6. Wrote AGENTS.md.
- [x] 7. Wrote this PLAN.md.
- [x] 8. Wrote the trkperf package (config.py, io.py, truth.py, reco.py,
   matching.py, binning.py, metrics/*.py, compare.py, pid/__init__.py,
   report.py, cli.py) with module/function docstrings and inline comments
   at every physics-choice constant.
- [x] 9. Wrote tests/ smoke suite (unittest) against data/dataset_small.
- [x] 10. Wrote the four SKILL.md files.
- [x] 11. Ran the tests locally to confirm the package works before any grid access.

## 8. Analysis execution state (as-built, beyond the original checklist)

Type 1 (clean) results for all five metrics are final and live in `output/`
(150 files, minQ2=1, matching threshold 0.5; per-metric JSON + markdown +
ROOT TNtuple for every metric, plus PNG plots for the four binnable metrics —
pid-confusion has no plots by design, its two-species-axis matrix is delivered
as JSON/markdown):
`acceptance_clean`, `efficiency_clean` (absolute + within-acceptance),
`resolution_clean`, `fake-rate_clean`, `pid-confusion_clean`.

Type 2 (+background, `bkg_mixed`) runs are the statistics bottleneck and the
overseas endpoint (hpceph-xrootd.twgrid.org) intermittently fails mid-read
with `[ERROR] Operation expired` / `Socket timeout`. Per AGENTS.md the Type 2
runs use `--max-file-failures` to skip transiently failing files; the launch
used a higher tolerance (60/200) than the AGENTS.md default (5-20) because the
endpoint was heavily degraded at launch time — every skip is printed to the
run log and the resulting bins are independently checked against
`MIN_ENTRIES_PER_BIN` (50) so under-fed bins are reported as insufficient
statistics rather than numbers.

Teardown discovery (this session): the background bkg jobs launched on
2026-09-03 16:30 (PIDs 2935712-2935716) all died unused shortly after the
session that launched them ended — this project's sessions run inside an
Apptainer container (`eic_xl-nightly.sif`) that is torn down when the
launching `opencode run`/`eic-shell` exits, taking background children with
it (`Terminating squashfuse_ll after timeout`). No `_bkg_mixed` output was
produced; nothing was recoverable. Two responses: (1) `scripts/launch_bkg.sh`
now launches each job with `setsid` + `nohup` + `</dev/null` so it is in its
own session and attached to no terminal; (2) `trkperf` gained a per-file
**read cache** (opt-in `--cache-dir`; bkg runs use `cache/bkg_files`) that pickles every
successfully read (file, branch-set) table and re-stamps `file_id` on load —
so a job interrupted mid-list restarts for free over the files it already
read (local pickle instead of the flaky network) and only downloads the
unfinished tail. Progress therefore accumulates across sessions instead of
being reset. Cache pickles live under the project dir (a /home host bind
mount), so they survive container teardown.

- [x] 12. Launched all five +background metric runs
   (`acceptance`, `efficiency`, `resolution`, `fake-rate`, `pid-confusion`)
   on `filelists/bkg_200.txt` (200 files, minQ2=1), detached, via
   `scripts/launch_bkg.sh` (logs `runs/<metric>_bkg_mixed.log`, PID files
   `runs/<metric>_bkg_mixed.pid`). Command shape:
   `python -m trkperf <metric> --file-list filelists/bkg_200.txt
   --dataset-tag bkg_mixed --min-q2-tier 1 --max-file-failures 60
   --cache-dir cache/bkg_files`.
   Each run writes `output/<metric>_bkg_mixed.{json,md,root}` (+ plots) on
   completion. Expected wall time is many hours (the bkg endpoint reads
   ~5.6x slower than JLab; ~3.6 min per readable 550 MB file for acceptance),
and the job may need re-launching across sessions per the teardown note
    above — the read cache makes each re-launch cheap.

    Follow-up (OOM fix): the first completed cycle showed the +background
    MCParticles tables are ~350 MB per file (~7M particles/file; a 30 GB box
    cannot hold 200 files' worth of frames). Per AGENTS.md ("extend the
    package"), `trkperf` therefore gained a per-file primary-only truth
    filter: `read_flat_multi(per_file_filter=...)` (applied AFTER the cache
    stash, so the cache stays RAW) and `read_truth_particles(primary_only=...)`
    (`generator_status == 1`); all four heavy metrics
    (`metrics/{acceptance,efficiency,resolution}.py`,
    `pid/confusion.py`) pass `primary_only=True`, which is semantically
    identical because each already did `truth.select_primary` immediately
    after reading. New tests in `tests/test_synthetic.py`
    (`TestPerFileFilter`): filter applied to returned rows, cache stashes RAW
    not filtered, and the truth wiring; `python -m unittest discover -s tests -v`
    passes 41/41 (re-verified 2026-09-20). `efficiency` still peaked at ~18 GB during its compute/merge
    (pd.concat on the full association tables) and was OOM-killed once (empty
    log) — it completed on the second launch once the earlier
    acceptance/comparison jobs had freed RAM (peak RSS 18.2 GB, all 200 files
    read from cache).
- [x] 13. Recorded the actual number of files read vs skipped for each bkg
   run, from each run log's `[io] WARNING: ... files skipped` banner:
   acceptance 200 read / 0 skipped (606 bins, 390 flagged insufficient);
   efficiency 200 / 0 on the final run (606 bins, 545 flagged; an earlier
   attempt OOM-killed during compute after reading 200/4 skipped);
   resolution 200 / 1 skipped (546 bins, 376 flagged); pid-confusion 200 / 3
   skipped (546 bins, 377 flagged); fake-rate 200 / 0 (110 bins, 34 flagged).
   This is recorded in the logs and is reproducible from
   `runs/<metric>_bkg_mixed.log`; each result's `max_file_failures=60`
   tolerance is documented in its JSON metadata.
- [x] 14. Ran the clean-vs-bkg comparison for every metric (all five complete,
   `output/<m>_comparison.{json,md}`):
   acceptance 706 bins compared, efficiency 706, resolution 666,
   pid-confusion 4138, fake-rate 118. Command shape as documented;
   `scripts/run_comparisons.sh` no longer skips anything.
- [x] 15. Checked every (pT, eta, species) bin against
   `MIN_ENTRIES_PER_BIN` (=50). Flagged insufficient-statistics bin counts per
   metric are listed in step 13 above (highest: efficiency 545/606). These
   are already reported as flagged rather than as numbers. ESCALATION — adding
   the remaining 1,263 minQ2=1 files and/or the minQ2 10/100/1000 tiers — has
   NOT been run; it remains an open decision (see step 17).
- [x] 16. Restored the local `data/dataset_small/` reference files (the
`{signal,signal_BKG_mix}` symlinks previously dangled — targets under
/home/wxie/eic/data/ no longer existed). Re-downloaded byte-exact via
`xrdcp` (clean 187,481,964 B from `dtn-eic.jlab.org`, campaign 26.02.0
hiDiv_1.0004; bkg 554,148,903 B from `hpceph-xrootd.twgrid.org`, campaign
26.07.1 hiDiv_1.2631) into /home/wxie/eic/data/dataset_small/{signal,
signal_BKG_mix}/RECO. `tests/test_local_data.py` (12 tests) passes again
network-free.

## 8.1 Endcap regions — implemented, local-pair results in hand

`--region {central,backward,forward}` on the acceptance/efficiency CLI
(`trkperf.config.TRACKING_REGIONS`; central keeps the legacy
`<metric>_<tag>` names, endcaps write `<metric>_<region>_<tag>`).
Resolution/fake-rate/pid-confusion are region-agnostic already. Local-pair
smoke (1 file each, `python -m trkperf {acceptance,efficiency} --region
{backward,forward}` + compares):

- backward e⁻ (pT ~1.07, η −2.75): acceptance 1.000 clean (n=575) and 1.000
  bkg (n=51) — vs 0.001 under the central rule; efficiency within 0.995/1.0.
- forward π⁺ (pT ~2.08, η +2.75): n=11 clean / 0 bkg on one file —
  insufficient statistics, as expected; endcap-hadron bins need grid files.
- `tests/test_synthetic.py::TestTrackingRegions` (3 tests) pins the
  registry, the region plumbing, and the unknown-region error.

Grid follow-up (not yet run): acceptance/efficiency `--region
backward/forward` on `filelists/clean_150.txt` (clean, strict) and
`filelists/bkg_200.txt` (`--max-file-failures 60 --cache-dir
cache/bkg_files`, as §8), then `trkperf compare --metric
acceptance-backward` (etc.). B0/tagger stay excluded (empty in DIS).

Incident 2026-09-23 (probe flake): the first `efficiency_forward_bkg_mixed`
run printed `[truth] NOTE: ForwardMPGDEndcapHits/TrackerEndcapHits/
TOFEndcapHits has no ... branch` and completed with the forward rule
degraded to OffM+RomanPot only — but a direct uproot probe showed all six
branches PRESENT in 26.07.1: the `_collection_present` single-file probe had
failed to OPEN file 0 (transient `[ERROR] Operation expired`) and mistaken
it for genuine absence. Those outputs were deleted as wrong, never plotted.
Fix: the probe now skips unopenable files (first successfully opened file
decides; all-unopenable returns True so reads fail loudly via the failure
budget), and `read_truth_hit_layer_counts` reports `found_collections`,
recorded in every acceptance/efficiency JSON's `collections_found` metadata.
`acceptance_backward/efficiency_backward_bkg_mixed` finished WITHOUT probe
NOTEs (12/17 files skipped, recorded) and are valid; `acceptance_forward`
aborted on budget (61/60) and both forward jobs were relaunched on the
fixed code. Covered by `TestCollectionProbe` (3 tests) + a
`found_collections` assertion on the local file.

Per-region plots: `trkperf plot --json ... --grouped [--eta-region
{barrel,"forward endcap","backward endcap"}]` draws species-group curves
for one detector region (or all three); pair the barrel plot with central
JSONs and each endcap plot with its `<region>`-rule JSON.

## 8.2 Eta-region-grouped plots (2026-09-22, no grid access)

The 16-eta-bin vs-pT plots were unreadable when overlaid (up to 120
series/panel). `trkperf` gained eta-region grouping for plotting only
(no recomputation, no new numbers):
- `report.eta_region()` + `report.SPECIES_GROUPS`: barrel (|eta| < 1),
  backward/forward endcap beyond; boundaries at +-1.0 so no 0.5-grid bin
  straddles. Species merge e+/e-, pi+/pi-, K+/K- pairs with proton and
  antiproton kept separate (beam-charge asymmetry).
- `report.aggregate_eta_species(df, value, err, count_cols)`: exact
  re-aggregation for count ratios (summed numerators/denominators, Wilson
  errors recomputed, 50-entry floor re-applied; a kept numerator column
  missing from older JSONs is recovered as sum(value*denom) with a loud
  NOTE); inverse-variance-weighted mean for resolution sigmas
  (approximate, non-converged fits excluded).
- `python -m trkperf plot --json ... --grouped --eta-region
  {barrel,"forward endcap","backward endcap"}` writes one plot per detector
  region (`<base>_<suffix>_grouped_<region>.png`, spaces slugged to
  underscores): 5 curves (e+/e-, pi+/pi-, K+/K- merged pairs + proton +
  antiproton separate), region-restricted bins only — never mixed-region
  points. Pair barrel plots with central JSONs, endcap plots with the
  matching `<region>`-rule JSONs (acceptance/efficiency); resolution and
  fake-rate have no region rule so all three region plots slice the single
  central JSON. Rendered 2026-09-23: acceptance 4 (clean x3 regions + bkg
  barrel), efficiency 8 (absolute + within x same), resolution 12, fake-rate
  12 (both datasets x3 regions x lin/logy); bkg endcap acceptance/efficiency
  plots pending the §8.1 grid jobs. Stale mixed-region `*_grouped.png`
  files were deleted.
- Covered by `tests/test_synthetic.py::TestEtaRegions` (7 tests).
- Interpretation caveat (verified on acceptance_clean): region curves at the
  barrel-endcap transition can be driven by a single edge bin — e.g. the
  backward e- point at pT 7.82 (acc 0.821, n=771) is entirely the eta=-1.25
  bin; deeper backward bins have no high-pT electrons. Stiff transition
  tracks cross many central layers, so acceptance there rises steeply with
  pT while the deep endcap stays ~0 under the central >=4/7 rule.

## 9. Outstanding decision (step 15 outcome)

- [ ] 17. Open decision: whether to deepen Type 2 statistics for the flagged
    bins (efficiency 545/606, acceptance 390/606, resolution 376/546,
    pid-confusion 377/546, fake-rate 34/110 are currently flagged
    insufficient). Discovery lists for the escalation are already staged
    (`filelists/clean_full.txt`, `filelists/bkg_full_minQ2_1.txt`,
    `filelists/bkg_full.txt`) but no escalation run has been launched.
    The wise escalation order is: (a) more minQ2=1 files — only
   200 of 1,463 used; then (b) minQ2 tiers 10 -> 100 -> 1000 for Type 2 only,
   as section 3 prescribes. Cost note: minQ2=1 files read ~3.5-7 min each at
   ~1-3 pickles/min and the compute peak is ~18 GB for efficiency — every
   additional tier is hours-days of wall time on this 30 GB box. Before
   escalating, sanity-check the flag-heavy bins against the acceptance
   cross-check: a large clean-vs-bkg difference in `acceptance` (which should
   be background-independent) would indicate campaign/geometry drift rather
   than a background effect — see section 3 caveat.
