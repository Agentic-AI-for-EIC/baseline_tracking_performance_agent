# `pid/` — ML Particle Identification pipeline

Runs inside `eic-shell`. Nothing is installed: the classifiers come from
`/home/wxie/.eic_python_pkgs`, which eic-shell already puts first on `PYTHONPATH`.
All file access goes through `trkperf` (`io`/`truth`/`reco`/`matching`), so PID and
the tracking metrics share one matching convention and one I/O path.

Design and the verified data-model facts behind it: **`../PLAN_pid.md`**.

---

## 0. Ten-second self-test (no network)

```bash
cd /home/wxie/eic/baseline_tracking_performance_agent
python -m pid schema-check --dataset clean \
    --file data/dataset_small/signal/RECO/*.root          # M0 gate: links + ML stack
python -m unittest discover -s pid/tests -v                # 255 tests, ~3.5 min first run
python -m pid all --dataset-tag clean --plot              # features -> train -> evaluate -> importance
```

`all` uses only the local reference file in `data/dataset_small/`, holds out by
event (not file) and therefore prints a `NOT a quotable performance number`
warning — that is deliberate, see §4.

## 1. Commands

| command | what it does | key outputs |
|---|---|---|
| `pid schema-check` | ML-stack preflight, dead-end registry, per-file relation audit (`podio_metadata` collectionIDs) | `output/pid-schema-audit_<tag>.csv` |
| `pid features` | per-track feature table over a file list (cached per file) | `output/pid-features_<tag>.pkl` |
| `pid train` | grouped CV, hyper-parameter search, gates, calibration | `output/models/<task>_<lib>_<tag>/{model.joblib,report.json,test_scores.pkl,all_scores.pkl,gain.csv}` |
| `pid performance` | maximum-significance working points (FOM(c), c*, c*(pT)) and the full differential package | see "Working points by maximum significance" |
| `pid evaluate` | AUC, efficiency at fixed fake rate, n-sigma, confusion, ROC, calibration | `output/pid-<task>_<lib>_<tag>-{overall,vs_pt,vs_eta,confusion,roc,calibration}.{json,md,root}` |
| `pid importance` | gain, **exact** SHAP, permutation + the physics check | `output/pid-<task>_<lib>_<tag>-importance-*.…` |
| `pid compare` | clean vs +background, via `trkperf.compare` | `output/pid-<task>_<lib>_<artifact>_comparison.{json,md}` |
| `pid all` | the whole chain for one dataset tag | all of the above |

### Separation power (n_sigma): do not fit the bounded score

`n_sigma` defaults to the closed-form **quantile** estimator
`2 * erfinv(2 * AUC - 1)` (exact for equal-width Gaussians, needs no fit). The
alternative `--nsigma-method logit` fits ROOT Gaussians to `ln(P/(1-P))`.
`--nsigma-method raw` reproduces the old behaviour and is **deprecated**: scores
live in [0, 1], so a Gaussian fitted to them measures the clip, not the tails
(measured: raw inflates n_sigma to 46 where quantile/logit agree at 16). Bins with
zero score overlap report `n_sigma_lower_limit` (95 % CL) instead of a point
estimate, drawn as upward triangles. Full derivation and validation:
**`PLAN_pid.md` §5c**.

### Working points by maximum significance

```sh
python -m pid performance --dataset-tag clean --model lightgbm
python -m pid performance --dataset-tag bkg_mixed --model lightgbm
python -m pid performance --dataset-tag clean --model lightgbm --channels Kpi,pK
python -m pid compare --task eid --model lightgbm --artifact vs_pt
```

Channels: `eid` (e/π), `ehad` (e/hadrons), `Kpi` (K/π), `pK` (p/K). A hadron channel
is reduced to its two hypotheses, `score = P(sig)/(P(sig)+P(bkg))` over truth in
{sig, bkg}, so **c\* lives in that channel's score space** — a K/π cut of 0.62 and an
e/π cut of 0.62 are not comparable numbers.

Figures written per channel into `output/plots/`:

| file | content |
|---|---|
| `pid-<ch>_significance_vs_cut.png` | FOM(c) over the whole channel + the optimum, and what the cut costs |
| `pid-<ch>_significance_vs_cut_by_pt.png` | FOM(c) in **every pT bin** (colour = bin): absolute and self-normalised, each c* starred |
| `pid-<ch>_significance_vs_cut_panels.png` | one panel per pT bin with c*, FOM, efficiency, fake rate; unusable bins drawn grey with the reason |
| `pid-<ch>_optimal_cut_vs_pt.png` | c*(pT) |
| `pid-<ch>_roc.png` | ROC in that channel's score space |
| `pid-<ch>_eff_vs_pt.png`, `_misid_vs_pt.png`, `_purity_vs_pt.png` | efficiency / fake rate / purity at c* |
| `pid-<ch>_eff_map_pt_vs_eta.png`, `_fake_map_pt_vs_eta.png` | 2D maps (masked where statistics do not support them) |
| `pid-eid_pion_rejection_vs_p.png`, `pid-<ch>_rejection_vs_p.png` | 1/fake vs p, log scale; 95 % CL **lower limits** where no background passed |
| `pid-hadpid_lightgbm_clean-nsigma_vs_p.png` | nσ(p) for K/π and p/K together (task-level names carry model_tag; per-tag plots live in `output/plots_<tag>/`) |
| `pid-hadpid_lightgbm_clean-confusion_matrix.png` | row-normalised multi-class confusion (held-out rows only) |
| `pid-<ch>_score_dist_train_vs_test.png` | overtraining check, log y, with a KS statistic per class |

Two extra switches: `--bin-source reco|truth|both` (see below) and
`--require-figures` (exit non-zero unless every deliverable exists on disk; the
manifest is `output/pid-deliverables_<model>_<tag>).

Summary table: `output/pid-working_points_<model>_<tag>.{json,md,root}` — one row per
channel with (c\*, peak FOM, efficiency, fake rate, purity, rejection and its lower
limit, and the `caution` text).

Definitions, formulas and traps for every quantity above (FOM, c\*,
`significance_eff`, Σw² vs Kish, efficiency / fake rate / purity / rejection and its
lower limit, nσ, AUC, the interval conventions and the statistics floors) are in
**`PLAN_pid.md` §5c**, and which reconstructed/truth information each one is computed
from (§5d, including the `--bin-source reco|truth|both` choice and the fact that all
merits are evaluated on held-out rows).

**Channel status.** `eid`/`ehad` are `headline`; `Kpi`/`pK` are **`exploratory`** —
calorimeter-only hadron ID is a baseline to be beaten, not a result, and each of
their figures gets a `*.png.note` sidecar saying so. `pid-deliverables_*` lists all
13 required figures per channel/basis with `present`, `shared_with_reco_run` and a
specific `reason` for anything absent.

**Read the cautions.** With a handful of background candidates, S/√(S+B) → √S and its
maximum migrates to "accept everything"; the code flags that as `degenerate` or
`permissive optimum` rather than reporting it as a working point, and reports
`rejection_lower_limit` instead of dividing by zero.

Tasks (`--task`): `eid` (e vs π, backward leg) · `ehad` (e vs hadrons) ·
`hadpid` (π/K/p, forward leg) · `pooled` (4-class, both legs, `leg` as a feature).
Learners (`--model`): `lightgbm` (default) · `xgboost` · `sklearn_hgb`.

## 2. Real grid runs

```bash
# clean (Type 1, JLab endpoint - keep it strict)
python -m pid features --file-list filelists/clean_150.txt --dataset-tag clean \
    --cache-dir cache/pid_features --out output/pid-features_clean.pkl
python -m pid train --task eid --model lightgbm --dataset-tag clean \
    --features output/pid-features_clean.pkl --n-iter 24
python -m pid evaluate --task eid --model lightgbm --dataset-tag clean \
    --scores output/models/eid_lightgbm_clean/test_scores.pkl --by pt,eta --plot

# +background (Type 2, flaky endpoint - tolerate failures, always cache)
python -m pid features --file-list filelists/bkg_200.txt --dataset-tag bkg_mixed \
    --max-file-failures 20 --cache-dir cache/pid_features \
    --out output/pid-features_bkg_mixed.pkl
python -m pid train --task eid --model lightgbm --dataset-tag bkg_mixed \
    --features output/pid-features_bkg_mixed.pkl
python -m pid compare --task eid --model lightgbm --artifact vs_pt
```

Feature extraction is the slow step on the overseas endpoint (it touches ~20
collections per file). `scripts/run_pid.sh` detaches it the same way
`scripts/launch_bkg.sh` does, and the per-file cache means a relaunch after a
container teardown replays finished files from disk.

## 3. Reading the outputs

Every JSON carries provenance (campaign, file counts, matching threshold 0.5,
ΔR window, excluded columns, library versions *including the shadowed numpy*,
seed, gate results). Every table also exists as a ROOT TNtuple with text columns
factorised to `*_code` plus a legend in the file metadata, so:

```bash
root -l 'output/pid-eid_lightgbm_clean-overall.root'
> pid_eid_overall->Draw("value:n_background","quantity_code==5")   # efficiency rows
```

`insufficient_stats` marks any bin below `MIN_ENTRIES_PER_BIN` (50); read
`meets_target` / `fake_at_*_err_hi` before quoting a rejection power.

## 4. Things this pipeline will refuse to do (and why)

* **Split by event.** Splits are by `file_id` (`StratifiedGroupKFold`). With one
  input file it falls back to an event split *and prints a warning* - a
  single-file AUC is a smoke test, not a result.
* **Train a class with < 30 rows.** `--allow-small-sample` overrides for tests.
  One +background file has 94 electrons but ~7 backward pions: the measured
  pion-side ceiling there is a fake rate of ~1/20, not 1e-4.
* **Use `charge` or `p`/`pT`/`eta` as inputs.** Signed charge is a beam-charge tag
  in NC DIS (e⁻ beam ⇒ the electron is always negative; measured correlation
  −0.74). Kinematics are the *flux* shortcut: with them in, `pt` outranks E/p in
  gain (measured: `0:pt 1:e_over_p_backward`); without them E/p leads both gain
  and SHAP. Both are re-enabled by `--use-charge` / `--include-kinematics` for
  explicit tagging studies.
* **Impute missing detector values.** Absent subsystems stay `NaN` (+ a `has_*`
  flag) for the boosters' native missing-handling. Zero-filling would invent an
  E/p = 0 population and fake a tail.
* **Trust the file's own PID summaries.** Every `edm4hep::ParticleIDData`
  collection here has a null `particle` relation (`index == -2`), and the dRICH
  IRT objects point at a collection that is not in the file - both detected at
  run time by `schema-check` against `podio_metadata`, not assumed.
* **Silently skip a broken link.** `schema-check` exits non-zero if a relation
  this pipeline depends on stops resolving, and lists the affected feature family.

## 5. Known feature gaps (measured, not assumed)

| wanted | status | reason |
|---|---|---|
| tracker dE/dx | **not available** | no per-track dE/dx collection; `TrackerHitData` has no `pathLength`. Reco-side `edep` proxy exists (`--enable-ionisation`, 21.6 % coverage in 26.02.0) |
| longitudinal shower profile in the endcaps | **not available** | `EcalEndcap{N,P}RecHits` are single-plane (one `position.z`, `layer == -1`); transverse shapes are computed from hits instead |
| per-hypothesis Cherenkov likelihood per track | **not available** | dangling `chargedParticle` collectionID; IRT quantities are aggregated event-level (`*_evt`) |
| TOF time per track | **campaign-asymmetric** | in 26.07.1 the TOF cluster hits are part of the track-fit measurement chain, in 26.02.0 they are not ⇒ excluded from the comparison model |
| per-layer barrel EMCal profile | available, but only in the backward / low-eta region | `EcalBarrelScFiRecHits` has 12 layers, `EcalBarrelImaging*` is the presampler; `EcalBarrelTrackClusterMatches` is empty in 26.02.0 |

## 6. Layout

```
pid/
  config.py schema.py links.py features.py dataset.py
  models/{base,lightgbm_model,xgboost_model,sklearn_hgb}.py
  train.py evaluate.py importance.py plots.py report.py compare.py
  significance.py performance.py
  cli.py __main__.py README.md scripts/run_pid.sh tests/
```
