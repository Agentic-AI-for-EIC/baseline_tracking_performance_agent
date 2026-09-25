# AGENTS.md — Tracking-performance project

## What this project does
Measure ePIC craterlake tracking performance — geometric acceptance, momentum
resolution, track-finding efficiency, fake rate — as a function of transverse
momentum (pT), pseudorapidity (eta), and truth particle species, by matching
true GEANT tracks (MCParticles) to reconstructed tracks (CentralCKFTracks).
All four metrics are computed for both dataset types, clean and +background,
using identical binning/species/thresholds. The primary deliverable per
metric is the comparison (ratio and difference) between them, quantifying the
impact of beam-induced background on tracking performance.

## Environment
- Everything runs inside eic-shell; the MCP servers are started with `eic-mcp up`.
- A local reference pair lives under data/dataset_small/ (signal: clean;
  signal_BKG_mix: same process with beam-induced background) for fast,
  network-free iteration while developing/debugging.

## Tools and code
- `rucio` MCP server (list_dids, list_files, list_file_replicas): discover
  datasets/files and resolve root:// URLs. Never hardcode a file list beyond
  the two dataset templates below.
- `xrootd` MCP server (check_file_exists, get_file_info): verify a file before
  a large run.
- The actual analysis (truth/reco arrays, matching, metrics, fitting,
  comparison) is the `trkperf` Python package in this project, not uproot MCP
  kernels — run it directly (`python -m trkperf ...`) so it is fast,
  testable, and reusable outside any chat session. Use the uproot MCP tools
  only for quick ad hoc single-file checks.
- Do NOT write bespoke one-off file I/O outside trkperf; extend the package
  instead so the next person benefits from it too.
- The ML PID pipeline lives in `pid/` (`python -m pid <command>`) and **imports
  trkperf** for all ROOT I/O, truth/reco reading and matching — it adds PID schema,
  joins, features, models and metrics, never a second reader. Plan + as-built
  record: `PLAN_pid.md`; runbook: `pid/README.md`. Classifiers (lightgbm, xgboost,
  scikit-learn) come from `/home/wxie/.eic_python_pkgs`, already first on
  `PYTHONPATH` inside eic-shell — nothing is installed by this project.
  Performance quantification is `pid evaluate` (fixed-fake-rate merits, n-sigma),
  `pid performance` (max-significance working points + 13-figure manifest), and
  `pid compare` (clean-vs-bkg) — mechanics: the `pid-performance` skill. Grid
  runs go through `pid/scripts/run_pid.sh` (detached, cached, trains all three
  learner libraries, ends with the advisory `check_learners.py` agreement gate);
  smoke is `python -m pid all --relax-gates` on the local pair.

## When something fails
- Never install software and never bypass trkperf with a hand-rolled script.
- A timed-out MCP call means the server is BUSY, not broken. Wait, retry once,
  then stop and report which tool/arguments failed.
- Never reuse a cached earlier result as if fresh, and never let trkperf
  silently swallow an exception — surface the traceback.
- The overseas +background xrootd endpoint (hpceph-xrootd.twgrid.org) is
  flaky: it intermittently drops in-flight reads with
  `[ERROR] Operation expired`. `trkperf/io.py::read_flat_multi` retries each
  file once and, when `max_failures > 0`, skips persistently failing files
  with a printed warning instead of aborting the whole run. Use
  `--max-file-failures N` on bkg runs (a small N, e.g. 5-20, is enough to
  ride out transient drops while keeping the result trustworthy); keep clean
  (JLab) runs strict with the default 0. One failure budget is shared across
  all reads of a metric run (truth + hit collections + reco + associations
  share one skip set), so joined tables always cover the same files; skipped
  files are recorded in each output's metadata, not just the log.
- Sessions here run inside an Apptainer container that is torn down when the
  session ends, killing background jobs with it, so a multi-hour bkg run may
  need several relaunches to finish. Two mechanisms make that cheap:
  `scripts/launch_bkg.sh` detaches each job with `setsid`+`nohup`, and bkg
  runs pass `--cache-dir` (no default — bkg runs use `cache/bkg_files`) — `read_flat_multi`
  pickles every successfully read (file, branch-set) table and re-stamps
  `file_id` on load, so a relaunched run replays already-read files from
  local disk instead of the network and only downloads the unfinished tail.
  Always pass `--cache-dir` on bkg runs; the cache lives under /home (a host
  bind mount), so it survives container teardown.

## Datasets
- Type 1, clean: epic:/RECO/26.02.0/epic_craterlake/DIS/NC/10x100/minQ2=1
  (root://dtn-eic.jlab.org:1094//volatile/eic/EPIC/RECO/... ; 4,609 files,
  1,083 events/file). WARNING: as of 2026-09-25 the /volatile copies used
  here are PURGED (rucio shows tape-only replicas at JLAB-TAPE-SE); treat
  this path as unavailable.
- Type 2, +background: epic:/RECO/26.07.1/epic_craterlake/Bkg_Exact1S_2us/
  GoldCt/10um/DIS/NC/10x100/minQ2={1,10,100,1000}
  (root://hpceph-xrootd.twgrid.org:1094//cephfs/epic/RECO/... ; 1463/902/547/551
  files per bin respectively, ~99 events/file — ~11x fewer events/file than
  Type 1, so this sample is the statistics bottleneck, not the clean one).
- Reproduction pair (2026-09-25, the current primary comparison): tags
  `clean26071` + `bkg26071` — BOTH from campaign 26.07.1, local copies on
  gautschi.rcac.purdue.edu:/scratch/gautschi/wxie/eIC_data_small_set/{clean,
  bkg/reco} (300 files x 1409 events; 275 files x 99 events), streamed via a
  localhost-only xrootd daemon + ssh tunnel: start remote
  `/cvmfs/oasis.opensciencegrid.org/osg/modules/xrootd/4.2.1/bin/xrootd -p
  1294 -b 127.0.0.1 -l ~/xrdlog/xrootd.log /scratch/gautschi/wxie/
  eIC_data_small_set` then `ssh -N -L 1294:127.0.0.1:1294
  gautschi.rcac.purdue.edu`; filelists/clean26071_local.txt +
  bkg26071_local.txt carry root://localhost:1294//<absolute path> URLs (the
  daemon serves absolute paths). Driver: scripts/run_reproduction_26071.sh
  (sequential — 300-file efficiency is the memory peak; tunnel watchdog;
  resumable). This removes the campaign difference from the comparison.
- Escalation order when a bin lacks statistics: more files in the current
  minQ2 tier, then the next minQ2 tier (1 -> 10 -> 100 -> 1000), for Type 2
  only. Type 1 stays at minQ2=1 (matches how it was produced).
- Type 1 (26.02.0) and Type 2 (26.07.1) are different production campaigns —
  the comparison isolates "with vs without background" only to the extent the
  two campaigns share the same detector/reconstruction configuration.
  Acceptance (truth/geometry-level, should be background-independent) is a
  built-in cross-check: a material difference there flags a possible
  campaign/geometry effect rather than a background effect.
- A single-particle gun sample has ~0 fakes by construction and must never be
  used for the fake-rate metric.

## Data model
- Tree: `events`.
- Truth reference ("true GEANT tracks"): `MCParticles` — `.PDG`,
  `.generatorStatus`, `.momentum.x/y/z` (GeV), `.mass`. pT = p*sin(theta),
  eta = -ln(tan(theta/2)), always computed from the truth momentum vector,
  never from a dataset's nominal bin label.
- Central-tracking truth hits (for acceptance): `SiBarrelHits`,
  `VertexBarrelHits`, `TrackerEndcapHits`, `MPGDBarrelHits`,
  `OuterMPGDBarrelHits`, `BackwardMPGDEndcapHits`, `ForwardMPGDEndcapHits`.
  Each has a `_<Collection>_particle` relation back to the `MCParticles` entry
  that produced it (verified live on a real file). "Layer" in this project
  means "one of these 7 collections has >=1 hit from the particle" — a
  collection-level proxy for a physical sensor layer, not a decoded cellID
  layer number (that would need subsystem-specific geometry decoding, out of
  scope here). Document this simplification wherever acceptance is reported.
- Reconstructed tracks: `CentralCKFTracks` — `.charge`, `.pdg`, `.chi2`,
  `.ndf`. CRITICAL data-model fact (verified live on real files):
  `CentralCKFTracks.momentum.x/y/z` is *all zeros* in every campaign file we
  process — the momentum is NOT stored there. The actual perigee track
  parameters live in `CentralCKFTrackParameters` (`.qOverP`, `.theta`, `.phi`,
  populated) which aligns 1:1 with `CentralCKFTracks` by event + position
  (identical per-event counts; the `_CentralCKFTracks_tracks` relation linking
  them is empty in these files). `trkperf/reco.py` reads momentum from
  `CentralCKFTrackParameters` and reconstructs p = 1/|qOverP|, then
  px/py/pz from (theta, phi), feeding the same `add_kinematics` as the truth
  side. Keep this in mind for any new reco-momentum code: never read
  `CentralCKFTracks.momentum.*`.
- Truth matching: `CentralCKFTrackAssociations`
  (`edm4eic::MCRecoTrackParticleAssociationData`) — `.weight` plus relations
  `_CentralCKFTrackAssociations_rec` (-> CentralCKFTracks) and
  `_CentralCKFTrackAssociations_sim` (-> MCParticles), both `{index,
  collectionID}`. A match is valid at weight >= 0.5 (state explicitly if a
  run uses a different threshold). trkperf assumes a single `CentralCKFTracks`
  collection and a single `MCParticles` collection per file and joins on
  `.index` alone; revisit if that assumption ever stops holding.
- Species: keyed by `MCParticles.PDG`; `trkperf/config.py` is the single
  source of truth for which species are in scope.

## PID data model (verified live; full evidence in PLAN_pid.md §3)
Machine-learning PID (`pid/`) depends on which links actually work. These facts
were measured on both local reference files — re-run `python -m pid schema-check
--dataset <tag> --file <root>` to confirm them on a new production before trusting
any PID number.
- **Working joins into `CentralCKFTracks`** (collectionID `530999115`):
  `EcalEndcap{N,P}/HcalEndcapN/LFHCALTrackClusterMatches`, `CalorimeterTrackProjections`
  (~every track; per-surface projection points → ΔR matching), `DRICH{Gas,Aerogel}Tracks`
  (radiator path length), `ReconstructedChargedRealPIDParticles` (1:1 with tracks),
  `CentralCKFTracks.measurements → CentralTrackerMeasurements → *_RecHits.edep`.
  Resolve `collectionID → name` via `podio_metadata.events___CollectionTypeInfo`;
  never assume a collection name from its parent.
- **Broken / unusable — do NOT build features on these:** every
  `edm4hep::ParticleIDData` (`DIRC/DRICH/RICHEndcapN/CombinedTOF/RealPID ParticleIDs`)
  has `particle.index == -2` (null relation, empty `parameters`), so the
  pre-computed PID likelihoods cannot be attached to a track;
  `*TrackClusterMatches.weight` is 0.0 for every entry in both campaigns (recompute
  ΔR instead); the dRICH IRT `chargedParticle` collectionID (`1290518152`) is absent
  from the registry (dangling) → IRT photon/hypothesis info is event-level only.
- **Detector granularity limits:** the endcap ECALs are single-plane here
  (`EcalEndcapN/PRecHits.position.z` is one constant, `layer == -1`,
  `subdetectorEnergies` empty) → no longitudinal shower profile for them; transverse
  shapes must be computed from the cluster's hits (`Σ E_hit == E_cluster` exactly).
  Only barrel ScFi (12 layers), barrel imaging/presampler and LFHCAL (7) are layered.
  `TrackerHitData` has no `pathLength` and no per-track dE/dx collection exists →
  ionisation is an optional `edep` proxy, never Bethe-Bloch dE/dx.
- **Electrons and hadrons live in different legs of this sample** (measured): truth
  e⁻ mean η = −2.50 with 93.6 % matched to `EcalEndcapN`, truth π/K/p mean η ≈ +2.3
  with ~47 % matched to `EcalEndcapP`/LFHCAL. Hence leg-scoped PID models in
  `pid/config.py`; a single pooled classifier mostly learns the hemisphere.
- **Never model inputs:** truth-derived quantities (including the tempting
  `SiBarrelHits.eDep / pathLength`), `CentralCKFTracks.pdg`,
  `ReconstructedChargedRealPIDParticles.goodnessOfPID`, the signed track `charge`
  (a beam-charge tag in NC DIS), and track `p/pT/η` (the DIS flux shortcut — these
  are the *binning* variables of the performance tables instead). `pid.dataset`
  blocks them by explicit NEVER-model patterns (`mc_`/`true_`/`gen_`, weights,
  event/run/file ids), family and name rules, and selects only from an explicit
  allowlist — unlisted numeric columns are dropped with a loud warning, never
  silently trained. `pid.dataset.assert_no_leakage` (> 0.95 halts with
  `ValueError`) + the `pid train` gates fail the run if any survives. `pid/train` gates additionally require E/p to lead the SHAP ranking and
  the model to beat a balanced label-permutation control.

## Conventions
- Bin in truth pT (log-spaced) and truth eta (bins of ~0.5), per species.
- Acceptance: >= N_min=4 of the 7 central-tracking truth-hit collections have
  a hit (see the "Layer" note above).
- Momentum resolution: Delta(pT)/pT = (pT_reco - pT_truth) / pT_truth
  (pT specifically, consistent with the pT/eta binning used everywhere else
  in this project); Gaussian fit to the core, done with **ROOT**
  (`TH1F.Fit(TF1("gaus"))`, not scipy) — see the "ROOT" note below.
- Efficiency: report both within-acceptance (matched / in-acceptance truth)
  and absolute (matched / all generated truth) efficiency; Wilson score-interval
  errors (conservative half-width, never zero-width at eff = 1, NaN at n = 0).
- Fake rate: reconstructed tracks with no valid truth match / all
  reconstructed tracks, binned in the track's own (pT, eta) — not species (a
  fake has no true species). Always name the dataset/minQ2 tier(s) used.
- Every metric is run on both Type 1 and Type 2, then compared (ratio and
  difference, per bin, with propagated uncertainty) via `trkperf/compare.py`.
- Write results as JSON; render tables in markdown; save plots; and write a
  ROOT TNtuple — all under output/.
- Minimum entries per bin before trusting it: 50 (see
  `trkperf.config.MIN_ENTRIES_PER_BIN`).

## ROOT in trkperf
- The momentum-resolution Gaussian-core fit (`trkperf/binning.py::fit_gaussian_core`)
  is done with ROOT (`TH1F.Fit(TF1("gaus"))`, two-pass core refit), NOT scipy,
  so the numbers a user re-fits by hand in a ROOT session match trkperf's
  exactly and the fitted TF1 can be drawn with standard ROOT tools.
- **Physical maximum:** sigma is bound to [0, RESOLUTION_MAX_SIGMA] where
  `RESOLUTION_MAX_SIGMA = 1.0`. Since resolution is sigma on
  Delta(pT)/pT = (pT_reco - pT_truth)/pT_truth, a width of 1.0 means a 100%
  resolution — the natural physical ceiling. `TF1::SetParLimits` enforces it,
  and a fit that pins sigma against the bound (i.e. the data is not
  describable by any physical resolution) is reported `fit_converged=False`
  rather than as an unphysical huge number.
- Every metric writes a ROOT TNtuple alongside the JSON/markdown
  (`output/<metric>_<tag>.root`, tree name = the metric, e.g. `resolution`),
  so a human can analyse/adjust plots in a ROOT session. Species/code columns
  (species, truth_species, reco_species) become integer `*_code` branches
  (order = config.SPECIES; `-1` = unknown); the code legend plus run metadata
  are stored as TNamed objects in the same file (`species_code_<n> = <name>`,
  `meta_<key> = <value>`). String pt_bin/eta_bin intervals are dropped — their
  centres (pt_bin_center/eta_bin_center) carry the same info.
- `report.to_root(df, path, tree_name=..., meta=...)` is the writer; the CLI
  calls it automatically so every run produces the `.root` file.
- ROOT is a heavy dependency; `binning.py` and `report.py` import it lazily
  (`_root()` / `_import_root()`), so their non-fit / non-root parts work
  without PyROOT.

## Definition of done
- Every quoted bin has enough entries to trust it, or is explicitly reported
  as insufficient statistics with the escalation tier already attempted.
  Metric tables contain the full bin grid, so empty bins appear as explicit
  insufficient_stats rows, never silent gaps.
- Momentum-resolution fits: chi2/ndf of order 1.
- Momentum-resolution sigma is always within [0, 1] (the physical maximum on
  Delta(pT)/pT); a bin whose raw width exceeds that is flagged non-converged,
  never reported as a big number.
- Trends are physically sensible; a wildly non-monotonic point is flagged, not
  silently reported.
- Always state dataset(s), minQ2 tier(s), file counts, matching threshold, and
  the trkperf command used, so the run is reproducible. The CLI records all of
  these — plus layers/species selection, skipped files, and the command line —
  in each output's metadata automatically.

## Grid-job reporting (twice daily on request)
`scripts/job_report.sh` prints the full status of both PID grid jobs in one
command: process state, pipeline stage, feature progress, skipped files,
per-task train outcomes with gate verdicts, errors/tracebacks,
`check_learners.py` output, and deliverables-manifest completeness. It is
read-only (never touches jobs, outputs, or the tree). Lead every status report
with its output, then interpret: a `gate:FAIL` line names the failing gate and
its measured values; a missing `[run_pid] done` with a dead PID means relaunch
(feature cache makes it cheap); any `Traceback` names the crashed step.

## Extending this project (e.g. particle identification)
`trkperf/matching.py` is the single choke point producing a truth<->reco
matched-pairs table (truth PDG/pT/eta + reco track params + match weight). A
future `trkperf/pid/*.py` module should reuse that exact table, pull whatever
extra PID branches it needs via `trkperf/io.py`, and report through the same
`trkperf/binning.py` / `trkperf/report.py` helpers — no changes to existing
modules required. See `trkperf/pid/__init__.py` for a worked-through stub.
