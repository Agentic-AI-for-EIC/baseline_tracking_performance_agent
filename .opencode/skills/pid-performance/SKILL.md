---
name: pid-performance
description: >-
  Quantify PID performance in `pid/` — fixed-fake-rate merits (`pid evaluate`:
  efficiency, n_sigma, confusion, ROC, calibration), maximum-significance
  working points (`pid performance`: FOM(c), c*, c*(pT), 13-figure manifest),
  and the clean-vs-+background comparison (`pid compare`), plus the
  `run_pid.sh` grid workflow with the cross-learner agreement check. Use when
  asked for a working point, FOM, c*, significance scan, n_sigma, rejection
  limit, deliverables manifest, figure_notes, pid evaluate/compare, bin-source
  bases, or the PID grid runs. NOT for training classifiers, features, or
  gates (the `pid-ml` skill), nor for the reconstructor's own confusion matrix
  (the `pid-confusion` skill).
---

# PID performance quantification

## What I do
`pid evaluate` / `pid performance` / `pid compare` turn trained models into
quotable numbers. They run on **held-out test rows only**, bin in
reconstructed kinematics by default, and write JSON + markdown + ROOT TNtuple
under `output/` (text columns → `*_code` with a legend in the metadata).
Conventions: `PLAN_pid.md` §5c–§5e; runbook: `pid/README.md`.

## When to use me
Working points, FOM(c) scans, c* and c*(pT), n-sigma separation, rejection at a
fixed fake rate, confusion at a cut, calibration, the 13-figure deliverables
manifest, clean-vs-bkg PID comparison, or launching/reading the PID grid runs.
For training, features, gates, or SHAP mechanics use `pid-ml`; for the
reconstruction's mass-hypothesis matrix use `pid-confusion` (the baseline these
numbers are measured against).

## Load first
`pid-ml` for the trained models this layer consumes; `trkperf-grid-workflow`
for file discovery (rucio/xrootd MCP), the two dataset templates, the
flaky-endpoint flags, and the minQ2 escalation order. Everything below assumes
both.

## Procedure

### 0. Evaluate one task (fixed-fake-rate merits + n-sigma)
```sh
python -m pid evaluate --task eid --model lightgbm --dataset-tag clean \
    --scores output/models/eid_lightgbm_clean/test_scores.pkl --by pt,eta --plot
```
`output/pid-<task>_<lib>_<tag>-{overall,vs_pt,vs_eta,confusion,roc,calibration}`.
The `overall` row set is **stable across tags** (needed for comparison):
`auc`, `signal_peak`, `background_peak`, `n_sigma_lower_limit` (always present,
NaN when it does not apply), `n_sigma`, `efficiency` at 1e-2/1e-3/1e-4.
n-sigma is the fit-free quantile form `2*erfinv(2*AUC-1)` by default; a
saturated AUC=1 reports an exact order-statistics **lower limit**, refused
(NaN, "no overlap observed") when the sample cannot support one. Thin bins
(< 50) are drawn as crosses, never circles. Multiclass tables are quoted one
class at a time (`proba_<class>`); without `--signal` the first class is used
and said out loud. The multiclass `score` column is class-0 probability kept
for shape compatibility — nothing reads it. Confusion rows carry their
decision rule (`argmax` vs `thresholds_at_c*`); calibration exists for
eid/ehad only.

### 1. Maximum-significance working points (the performance package)
```sh
python -m pid performance --dataset-tag clean     --model lightgbm --bin-source both --require-figures
python -m pid performance --dataset-tag bkg_mixed --model lightgbm
```
Channels `eid` (e/π), `ehad` (e/hadron), `Kpi`, `pK`; hadron channels reduce to
`P(sig)/(P(sig)+P(bkg))`, so **each channel's c\* lives in its own score space
and cuts are never comparable across channels**. The global scan uses a
401-step grid, the per-bin curves/optima a 201-step grid — one grid constant,
so a table optimum and its drawn star cannot disagree on flat plateaus.

**Read the `caution` column before quoting any row.** `S/sqrt(S+B)` → sqrt(S)
when background is scarce: `degenerate` (accept everything — no working point
exists, though the optimum is still defined), `permissive optimum`
(fake > 0.5), `optimum at zero cut`, small-background-sample warnings. Per-bin
rows carry their own `caution`; empty bins are explicit NaN rows, never silent
gaps; rows with non-finite binning kinematics are filtered and counted.

**Manifest.** 13 required figures per channel per basis, checked **on disk** by
`verify_deliverables`; `--require-figures` turns any gap into a non-zero exit.
A missing figure carries a precise `reason`, including the `figure_notes`
sidecars (e.g. test_scores-only directories cannot draw train-vs-test). Three
score-level figures are written once under reco and counted as shared by the
truth run (the electron rejection figure travels under its physics alias
`pion_rejection_vs_p`).

**State the binning basis.** `--bin-source reco|truth|both` partitions by
reconstructed kinematics (what a cut is applied to) or the truth partner's
(what `trkperf` bins in, migration-free); truth outputs carry `*_truthpt`.
Both bases partition the *same* held-out candidates — totals must agree
exactly. Importance figures read the unwrapped `booster.joblib`, never the
calibrated `model.joblib`.

**Errors and scales.** Point estimates use the (possibly weighted) yields;
intervals and rejection limits are count-based — identical for class-constant
weights, approximate under `--bkg-scale` (which also reaches the per-class
cuts). `lumi_scale`/`bkg_scale` are recorded in every output because FOM is
not scale-invariant. Quoted at-c\* numbers are an in-sample optimum over the
threshold grid (mildly optimistic, equally on both tags — compare ratios, not
absolutes).

### 2. Clean vs +background
```sh
python -m pid compare --task eid --model lightgbm --artifact vs_pt
python -m pid compare --task eid --model lightgbm --artifact all   # overall,vs_pt,vs_eta
```
Row identity is declared per artifact in `pid/compare.py::JOIN_KEYS` (string
bin labels, never float centres; `overall` includes `target_fake_rate`, whose
NaN rows still join). `roc` is **refused** — curve points have no stable row
identity across tags, overlay the PNGs instead. Summaries carry `*_scored`
file/row counts. Requires both tags trained and evaluated; missing sides are
named, never half-compared. Also run the **cross-application** check (train
clean → evaluate bkg and vice versa) before attributing a difference to
background: the campaigns differ in schema (PLAN_pid.md §3.3).

### 3. Grid runs (detached; sessions die with the container)
```sh
pid/scripts/run_pid.sh clean     filelists/clean_150.txt 0
pid/scripts/run_pid.sh bkg_mixed filelists/bkg_200.txt   20 cache/pid_features "eid,ehad" [seed]
pid/scripts/run_pid.sh status
```
Signature: tag, filelist, max_failures, cache_dir, tasks, seed. One detached
job does features → train (all three learner libraries) → evaluate
(`--by pt,eta --plot`) → importance per task, then `check_learners.py`.
Quoting rule inside that script (a real bug lived here): the detached shell
inherits only *exported* variables, so every value must be expanded by the
launching shell — `'$VAR'` references expand to empty inside. The cache makes
relaunches free over finished files.

**`check_learners.py` is advisory, not blocking**: it compares held-out test
AUC across lightgbm/xgboost/sklearn_hgb per task against
`CROSS_LEARNER_MAX_AUC_SPREAD` (0.02). A WARN means the headline number depends
on the learner — do not quote one library's AUC as the task result; on a hard
task (hadpid) the spread can be statistics, not a bug.

## Pitfalls / conventions
- Statuses propagate everywhere (`headline` eid/ehad, `exploratory` Kpi/pK,
  `cross-check` pooled): an exploratory figure carries a `.note` sidecar, and
  quoting its n-sigma as a PID result is a category error.
- The `overall` confusion for binary tasks uses the 1e-4 working point and
  records its threshold; multiclass uses argmax unless per-class c\* cuts are
  supplied (then some candidates are UNASSIGNED by design).
- `python -m pid` is the only entry point; there is no `pid` console script.
