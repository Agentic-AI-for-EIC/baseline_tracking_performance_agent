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

## 1. opencode setup
- Self-contained /home/wxie/eic/baseline_tracking_performance_agent/opencode.jsonc:
  - "model": "github-copilot/claude-sonnet-5"
  - "small_model": "github-copilot/claude-haiku-4.5"
  - mcp block: uproot/xrootd/rucio remote servers on 127.0.0.1:9101-9103
    (same eic-mcp instance already running for /home/wxie/eic)
- /home/wxie/eic/tracking_performance/.github/copilot-instructions.md:
  one line, "Follow the project rules in AGENTS.md."
- GitHub Copilot provider: already authenticated (oauth), no action needed.
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
│   └── fake-rate/SKILL.md
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
│   ├── compare.py         # reads two same-metric JSON results (clean, bkg), aligns bins,
│   │                      # emits per-bin ratio + difference (+propagated uncertainty)
│   ├── pid/                # empty placeholder package + docstring showing how a future
│   │                       # particle-ID module would consume matching.py's output
│   ├── report.py          # JSON + markdown-table + plot writers
│   └── cli.py              # `python -m trkperf <metric|compare|plot> ...`
├── tests/                   # unittest smoke tests against data/dataset_small (no network)
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

Momentum resolution is defined on pT specifically: Delta(pT)/pT =
(pT_reco - pT_truth) / pT_truth, consistent with binning everything in pT/eta.

## 5. Skills (.opencode/skills/<name>/SKILL.md)
All four: discover files for the current tier via `rucio` MCP -> run
`python -m trkperf <metric> --file-list ... --species ... --dataset-tag
{clean,bkg_mixed} --min-q2-tier ...` on Type 1 AND Type 2 -> escalate per the
order above if under-populated -> run `python -m trkperf compare --metric <metric>
--clean ... --bkg ...` -> report JSON/table/plot with provenance.

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
plots + ROOT TNtuple for every metric):
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
    passes 30/30. `efficiency` still peaked at ~18 GB during its compute/merge
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

## 9. Outstanding decision (step 15 outcome)

- [ ] 17. Open decision: whether to deepen Type 2 statistics for the flagged
   bins (efficiency 545/606, acceptance 390/606, resolution 376/546,
   pid-confusion 377/546, fake-rate 34/110 are currently flagged
   insufficient). The wise escalation order is: (a) more minQ2=1 files — only
   200 of 1,463 used; then (b) minQ2 tiers 10 -> 100 -> 1000 for Type 2 only,
   as section 3 prescribes. Cost note: minQ2=1 files read ~3.5-7 min each at
   ~1-3 pickles/min and the compute peak is ~18 GB for efficiency — every
   additional tier is hours-days of wall time on this 30 GB box. Before
   escalating, sanity-check the flag-heavy bins against the acceptance
   cross-check: a large clean-vs-bkg difference in `acceptance` (which should
   be background-independent) would indicate campaign/geometry drift rather
   than a background effect — see section 3 caveat.
