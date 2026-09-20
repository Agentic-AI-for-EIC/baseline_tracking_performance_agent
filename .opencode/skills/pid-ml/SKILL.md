---
name: pid-ml
description: >-
  Run the machine-learning Particle Identification pipeline in `pid/` —
  electron ID (e vs π, targeting 10⁻³–10⁻⁴ pion rejection) in the backward leg,
  charged-hadron ID (π/K/p) in the forward leg, plus a pooled 4-class cross-check,
  with LightGBM / XGBoost / sklearn — feature table → grouped-by-file training with
  physics gates → efficiency-at-fixed-fake-rate, n-sigma, confusion, ROC,
  calibration, exact SHAP importances → clean-vs-+background comparison. Use when
  asked for ML PID, a BDT/DNN classifier, E/p or shower-shape based
  identification, pion-rejection power, feature importance, or PID performance
  under beam background. NOT for the reconstruction's own mass-hypothesis
  confusion matrix, which is the `pid-confusion` skill.
---

# ML PID pipeline

## What I do
`pid/` is the ML PID extension of this project. It **imports `trkperf`** for all
ROOT I/O, truth/reco reading and truth matching (AGENTS.md: no bespoke file I/O
outside the package), so PID and the tracking metrics share one matching
convention (`CentralCKFTrackAssociations.weight >= 0.5`) and are directly
comparable. Plan and verified data model: `PLAN_pid.md`, `AGENTS.md` → "PID data
model". Runbook: `pid/README.md`.

## When to use me
Electron ID, hadron ID, pion-rejection / fake-rate working points, n-sigma
separation, feature importance (gain / SHAP / permutation), or the effect of
beam-induced background on any of those. For the *reconstructor's* PID hypothesis
matrix use `pid-confusion`; that result is the baseline this pipeline is measured
against.

## Load the grid workflow first
`trkperf-grid-workflow` supplies file discovery (rucio/xrootd MCP), the two
dataset templates, the flaky-endpoint flags (`--max-file-failures`,
`--cache-dir`) and the minQ2 escalation order. Everything below assumes it.

## Procedure

### 0. Preflight (always, ~10 s) — verifies the ML stack *and* the links
```sh
python -m pid schema-check --dataset clean
python -m pid schema-check --dataset clean --file <one file>   # adds the link audit
python -m pid schema-check --dataset bkg_mixed --file <one file>
```
This reads `podio_metadata.events___CollectionTypeInfo` and checks every relation
the pipeline joins on. **Exit non-zero = a needed link broke or a new production
renamed something**: read the printed table, update `pid/schema.py`, and re-run
before spending hours on features. It also lists the *expected* dead ends (the
`edm4hep::ParticleIDData` likelihoods with `particle.index == -2`, the dRICH IRT
`chargedParticle` collectionID that is absent from the file, all-zero
`*TrackClusterMatches.weight`), which is how a future production that fixes one
gets noticed.

### 1. Smoke test on the local pair (no network)
```sh
python -m unittest discover -s pid/tests -v     # 255 tests
python -m pid all --dataset-tag clean --plot --relax-gates  # features→train→evaluate→importance (smoke checks plumbing; gates print but do not block — drop --relax-gates to enforce them)
```
One file ⇒ the pipeline prints `NOT a quotable performance number` (splits cannot
be by file) and the +background side refuses to train (~7 pions < the 30/class
floor). Both behaviours are correct; do not override them to get a number.

### 2. Features (the long step) — one row per reconstructed track
```sh
python -m pid features --file-list filelists/clean_150.txt --dataset-tag clean \
    --cache-dir cache/pid_features --out output/pid-features_clean.pkl
python -m pid features --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed \
    --max-file-failures 20 --cache-dir cache/pid_features \
    --out output/pid-features_bkg_mixed.pkl
```
`features` prints the median E/p per truth class per leg and **exits non-zero** if
the backward electron peak is not > 0.8 or the backward pion peak is not < 0.4 —
the physics sanity gate that catches a broken cluster↔track join. Measured on the
reference files: e 0.992 / π 0.195 (clean), 0.995 / 0.238 (bkg).

### 3. Train (gates are blocking by default)
```sh
for task in eid ehad hadpid pooled; do
  python -m pid train --task $task --model lightgbm --dataset-tag clean \
      --features output/pid-features_clean.pkl --n-iter 24
done
python -m pid train --task eid --model xgboost --dataset-tag clean \
    --features output/pid-features_clean.pkl     # twin-arm agreement check
```
Gates: train−test AUC gap < 0.02; the model must beat a **balanced
label-permutation control** (z ≥ 3 *and* ΔAUC ≥ 0.1 — with ~10 minority
candidates in test the control AUC fluctuates by ≈0.14, so a fixed tolerance about
0.5 is the wrong test); no single feature may correlate > 0.95 with the label
(`ValueError` halt — wide margin: strongest legitimate is 0.71); SHAP attribution
measured on held-out rows only, and electron tasks refuse to train without
measurable E/p; and
for the electron tasks **E/p must lead the SHAP attribution**. `hadpid`'s
attribution gate is advisory (no per-track Cherenkov/timing exists to lead with).
`--relax-gates` inspects without blocking — never quote a relaxed run silently.

### 4. Evaluate, importance, plots
```sh
python -m pid evaluate --task eid --model lightgbm --dataset-tag clean \
    --scores output/models/eid_lightgbm_clean/test_scores.pkl --by pt,eta --plot
python -m pid importance --dir output/models/eid_lightgbm_clean \
    --task eid --model lightgbm --dataset-tag clean --plot
```
Outputs `output/pid-<task>_<lib>_<tag>-{overall,vs_pt,vs_eta,confusion,roc,calibration}`
as JSON + markdown + ROOT TNtuple (text columns → `*_code` with a legend in the
metadata) plus PNGs in `output/plots/`. Quote efficiency at a fixed fake rate
**with** its Garwood interval and check `meets_target` / `fake_at_*_err_hi`:
with 20 background candidates a point estimate of 0 does not demonstrate 10⁻⁴.

### 4b. Maximum-significance working points (the performance package)

```sh
python -m pid performance --dataset-tag clean     --model lightgbm
python -m pid performance --dataset-tag bkg_mixed --model lightgbm
```

Channels `eid` (e/pi), `ehad` (e/hadron), `Kpi`, `pK`; hadron channels are reduced
to `P(sig)/(P(sig)+P(bkg))` over the two truth classes, so **each channel's c\*
lives in its own score space and cuts are not comparable across channels**.
Per channel this writes FOM(c) globally, **FOM(c) in every pT bin**
(`pid-<ch>_significance_vs_cut_by_pt.png` absolute + self-normalised, and
`..._panels.png` one panel per bin), c*(pT), efficiency / mis-ID / purity at c*,
2D (pT,eta) maps, rejection vs p, n-sigma vs p, the held-out confusion matrix and
the train-vs-test score check, plus `output/pid-working_points_<model>_<tag>`.

**State the binning basis.** `--bin-source reco|truth|both` partitions every binned
merit by the reconstructed track kinematics (default: what a cut is applied to) or by
the truth partner's (what the `trkperf` tracking metrics bin in, migration-free). The
basis is recorded in `variable`/`binning_basis`/`bin_column` in each table and in each
axis label, and truth-binned outputs carry a `*_truthpt` filename suffix. Both bases
partition the *same* held-out candidates, so their totals must agree exactly — if they
do not, rows are being dropped somewhere, not merely re-binned.

**Always read the `caution` column before quoting a row.** S/sqrt(S+B) tends to
sqrt(S) when the background is scarce, so with few candidates its maximum sits at
"accept everything"; the code labels that `degenerate` or `permissive optimum`
instead of presenting it as a selection, and reports
`rejection_lower_limit` rather than 1/0. The per-bin panels show the same
disease bin by bin - a grey "no curve" panel means that pT slice has too few
candidates of one species, which is the direct argument for how many grid files
to run. Weighting knobs (`--bkg-scale`, `--lumi-scale`) are recorded in every
output, because FOM is not scale-invariant: it grows as sqrt(luminosity), and the
optimum moves with the assumed composition.
For the full working-point mechanics — caution taxonomy, manifest verification,
bin-source accounting, error conventions — use the `pid-performance` skill.

### 5. Clean vs background
```sh
python -m pid compare --task eid --model lightgbm --artifact vs_pt
python -m pid compare --task eid --model lightgbm --artifact all
```
Requires both tags trained and evaluated; it refuses (naming the missing side)
rather than half-comparing. Also run the **cross-application** check (train
clean → evaluate bkg and vice versa) before attributing a difference to
background: the two campaigns differ in schema (PLAN_pid.md §3.3).
Row-identity rules, refused artifacts, and the grid launcher live in the
`pid-performance` skill.

## Pitfalls / conventions
- **Never add these as features**: truth-derived anything (including
  `SiBarrelHits.eDep/pathLength`, which exists and looks like dE/dx),
  `CentralCKFTracks.pdg`, `ReconstructedChargedRealPIDParticles.goodnessOfPID`,
  signed `charge` (beam-charge tag in NC DIS; ρ(label) = −0.74 measured), and
  `p/pT/η/φ` (DIS flux shortcut — with them in, `pt` outranks E/p in gain). They
  are blocked by explicit NEVER-model patterns (`mc_`/`true_`/`gen_`, weights,
  event/run/file ids) *and* by family/name rules in `pid/dataset.py`, and
  selection is fail-closed on an explicit allowlist — unlisted numeric columns
  are dropped with a loud warning, never silently trained; kinematics are
  the **binning** variables of the result tables instead. Opt-ins exist
  (`--use-charge`, `--include-kinematics`, `--include-event-level`) for labelled
  ablations only.
- **Event-level (`*_evt`) and `track_time*` are opt-in, not baseline**: occupancy
  out-ShAPed E/p when included, and event activity differs systematically between
  the campaigns, so a comparison built on them measures the overlay, not PID.
- **Missing detector responses stay `NaN`** (+ `has_*`). Never impute, never
  zero-fill: all three learners handle NaN natively, and fabricating E/p = 0
  invents a fake-rate tail.
- **Split by file, never by event** (`file_id` groups). A single input file falls
  back to an event split and says so in the log and in `report.json` (`cv_folds`).
- **Splits and search are single-threaded by default** (`config.N_JOBS = 1`):
  measured on this host, LightGBM *and* XGBoost deadlock at `n_jobs = 4`.
- The 26.07.1 production renames collections (`TOF*RecHits` → `TOF*ClusterHits`,
  adds `Cluster.radius/dispersion/principalAxes*`, drops
  `CentralTrackerMeasurements`): features resolve branches per campaign via
  `pid/schema.py`, and bkg-only fields are excluded from headline models so the
  arms stay identical.
- `python -m pid` is the only entry point; there is no `pid` console script. The
  tree cycles differ between campaigns (`events;5` vs `events;17`) — uproot picks
  the newest, and no code pins a cycle.
