# ML Particle Identification pipeline — project plan (PLAN_pid.md)

Status: **delivered, tested, and run at grid scale** (M0–M5 machinery complete;
grid runs M6/M8 done — clean-150 + bkg-200 features, all learners trained,
`pid performance` manifests 104/104 clean + 26/26 bkg, clean-vs-bkg comparisons
written; §11 is the as-built record, §12 the remaining work). The plan
was agreed in the planning conversation; §3's data-model facts were verified live
against the two local reference files, not assumed. Headline physics numbers stay
gated: the `top_feature_is_physical` gate FAILs at grid scale (raw `ecal_*_E`
outranks E/p), so working points exist but are not quotable until adjudicated.

## 0. Goal
Extend this project from tracking performance into a **machine-learning Particle
Identification pipeline** on ePIC craterlake RECO simulation:

* Stage 1 — **electron ID** (e vs π), aiming at 10⁻³–10⁻⁴ pion rejection;
* Stage 2 — **charged-hadron ID** (π / K / p), reported as nσ separation vs (p, η);
* plus a **pooled 4-class** (e/π/K/p) arm as a cross-check against the
  hierarchical design;
* every metric produced for **both** dataset types (clean `Type 1`, `bkg_mixed`
  `Type 2`) and compared (ratio + difference), as the tracking metrics already are.

Truth labels come from the existing `trkperf/matching.py` choke point, so PID is
measured on the same truth↔reco association convention (weight ≥ 0.5) as the
efficiency/resolution/fake-rate results it accompanies.

## 1. Delivery contract — code a human can run and test
* **All** analysis code is Python under `/home/wxie/eic/baseline_tracking_performance_agent/pid/`.
* `pid/` imports `trkperf` for every piece of ROOT I/O, matching, binning and
  reporting (AGENTS.md: *no bespoke file I/O outside trkperf*). `pid/` adds the
  PID schema, joins, feature engineering, models and metrics — it does not
  re-implement reading.
* Every module is importable **and** CLI-invocable:
  `python -m pid {schema-check|features|train|evaluate|importance|performance|compare|all}`
  with the same flag vocabulary as `trkperf` (`--file`, `--file-list`,
  `--dataset-tag`, `--min-q2-tier`, `--max-file-failures`, `--cache-dir`, `--out-dir`).
* Each step has a **network-free smoke path** over `data/dataset_small/`
  (`--limit-files K`) so a human can run it in seconds.
* Deterministic: one `--seed` (default 1234) feeds every random consumer.
* Verification gates are **hard non-zero exits** — a broken step cannot be sailed
  past.
* Tests are stdlib `unittest` (no pytest in this environment):
  `python -m unittest discover -s pid/tests -v`.
* `pid/README.md` is the ordered copy-paste runbook; `pid/scripts/run_pid.sh`
  reproduces the detached + `--cache-dir` recipe used for the flaky bkg endpoint.

## 2. Environment (verified, not guessed)
| Item | Measured |
|---|---|
| `/home/wxie/.eic_python_pkgs` | already **first on `PYTHONPATH`** inside eic-shell → no installation needed, AGENTS.md's "never install" respected |
| classifiers | **lightgbm 4.7.0, xgboost 3.4.1, scikit-learn 1.9.1** (+ joblib, threadpoolctl) all import and fit |
| coexistence | ROOT 6.40 / uproot 5.7.1 / awkward 2.9.0 / pandas 3.0.3 import alongside; `trkperf.binning`'s ROOT Gaussian fit still works in the same interpreter |
| NaN handling | native, verified for LGBM, XGB and `HistGradientBoostingClassifier` |
| categorical | verified with pandas-3 `Categorical` (`categorical_feature=`, `enable_categorical=`) |
| multiclass | verified 4-class `predict_proba` |
| SHAP | `shap` **absent**; exact SHAP available from the trees themselves (`lgb…pred_contrib=True`, `xgb…pred_contribs=True`); plus gain and `sklearn.inspection.permutation_importance` |
| tuning | `optuna` **absent** → `RandomizedSearchCV` / `HalvingRandomSearchCV` |
| splitting | `StratifiedGroupKFold(groups=file_id)` → file-level splits (no event leakage) |
| calibration | `CalibratedClassifierCV` available |
| **no parquet** | `pyarrow`/`fastparquet` absent → feature tables are pickle/npz + a ROOT TNtuple |
| **numpy shadowing** | that dir ships numpy 2.4.6 over the container's 2.2.6 → versions stamped into every output; `assert_ml_env()` preflight |

## 3. Data model — verified live on both reference files
### 3.1 Working, populated joins (feature backbone)
| Path | Evidence |
|---|---|
| `_EcalEndcap{N,P}TrackClusterMatches_track → CentralCKFTracks` | `_…_track.collectionID == 530999115`, which `_CentralCKFTrackAssociations_rec` proves is `CentralCKFTracks`; cluster side cid `2822405817 == EcalEndcapPClusters` |
| `_CalorimeterTrackProjections_track → CentralCKFTracks` | same cid, **3.76 projections/event ≈ every track**; `_…_points` carries `position`, `momentum`, `theta`, `phi`, `pathlength`, `surface` at the calorimeter faces → geometric ΔR matching that does **not** depend on the (broken, see 3.2) built-in match weight |
| `_DRICH{Gas,Aerogel}Tracks_track → CentralCKFTracks` | same cid, 1.69/event; `DRICHGasTracks.length` = 734–1040 mm → per-track radiator path length |
| `ReconstructedChargedRealPIDParticles ↔ CentralCKFTracks` | `_…_tracks.collectionID == 530999115`, counts identical (397/397 clean, 492/492 bkg); `goodnessOfPID` ∈ [0,1] populated |
| `_EcalEndcap{N,P}Clusters_hits → EcalEndcap*RecHits` | populated (~8 hits/cluster; `energy`, `position.*`, `time`) → longitudinal + transverse shower shape from hits |
| **reco-side ionisation chain** `CentralCKFTracks.measurements → _CentralCKFTracks_measurements → CentralTrackerMeasurements → _CentralTrackerMeasurements_hits → *RecHits.edep` | all populated (26.35 measurements/track-event; hit cids `176177746 == MPGDBarrelRecHits`, `3971475192 == SiBarrelTrackerRecHits`, both in the registry) → a **leakage-free per-track dE/dx-style proxy** |
| `podio_metadata.events___CollectionTypeInfo{name,collectionID}` | 282 (clean) / 387 (bkg) registered collections → the authoritative **cid → collection-name registry**, read per file at run time |

### 3.2 Verified dead ends — these must NOT be used, and `pid/links.py` reports them
* Every `edm4hep::ParticleIDData` collection (`DIRCParticleIDs`,
  `DRICHParticleIDs`, `RICHEndcapNParticleIDs`, `CombinedTOFParticleIDs`,
  `ReconstructedChargedRealPIDParticleIDs`) **is populated (2.8–9.3 entries/event,
  PDG ±{11…2212}, likelihood ∈ [0,1]) but its `particle` relation is a null
  reference: `index == -2`, `collectionID == 0`, in both campaigns.** Their
  `parameters` vectors are empty (`begin == end == 0`). ⇒ the pre-computed
  per-hypothesis PID likelihoods cannot be attached to a track.
* `*TrackClusterMatches.weight == 0.0` for **every** entry in **both** campaigns
  ⇒ match quality must be recomputed (ΔR from projection/cluster direction).
* `DRICH{Gas,Aerogel}IrtCherenkovParticleID._chargedParticle` carries
  `collectionID == 1290518152`, which is **absent from
  `podio_metadata.events___CollectionTypeInfo`** (282 / 387 registered
  collections) ⇒ a reference to a collection that was never written; the IRT
  per-hypothesis weights (4 hypotheses per object, PDG 11…2212) are therefore
  **event-level only** in v1 and are excluded from per-track features.
  (`links.py` detects exactly this case by registry lookup and reports it.)
* `DRICH*IrtCherenkovParticleID.refractiveIndex` / `.photonEnergy` == 0;
  `EcalEndcapPClusters.subdetectorEnergies` empty and `energyError` == 0;
  `EcalEndcapPRecHits.layer == -1` in 26.02.0 ⇒ longitudinal profile from hit
  `position.z`, not `layer`.
* `EcalBarrelTrackClusterMatches` empty in 26.02.0.
* **No thickness-normalised dE/dx exists**: `edm4eic::TrackerHitData` (RecHits)
  stores `edep` but **no `pathLength`**, and no per-track `*Dedx*`/truncated-mean
  collection is written in either campaign. What *is* buildable is the
  **reco-side ionisation proxy** of 3.1 (`edep` per track per subsystem).
  The truth-side `eDep / pathLength` (available on `SiBarrelHits`,
  `VertexBarrelHits`, `MPGDBarrelHits`, …) is **never used as a feature** — it is
  derived from the truth particle itself and would leak the label; it is exposed
  only as a validation/diagnostic column, off by default.

### 3.3 Campaign schema skew (26.02.0 clean vs 26.07.1 bkg) — hence the alias table
clean 5,297 branches / bkg 6,945. bkg-only: `ClusterData.radius`, `.dispersion`,
`.principalAxesLengthsXYZ[3]`, `.principalAxesLengthsThetaPhi[2]`,
`*ExpectedClusters`, `*RemnantClusters`, `*TrackExpectedClusterMatches`,
`*TrackSplitMergeClusterMatches`, all `podio::LinkData` `*Links`,
`*ChargedCandidateParticlesAlpha`. clean-only: `TOFBarrel{Raw,}RecHits`,
`TOFEndcap{Raw,}RecHits`, `CentralTrackerMeasurements` (renamed in bkg to
`TOF{Barrel,Endcap}ClusterHits`, which exist in **both**). Consequences:
headline models use only the intersection; bkg-only shape fields are an
explicitly-labelled ablation arm; `schema.py` holds
`FEATURE → {campaign: branch}`; TTree cycle is never pinned (clean `events;5`,
bkg `events;17`).

### 3.4 Measured kinematics (300 clean events, matched primaries, p > 1 GeV)
| truth | n | mean η | EcalEndcapN | EcalEndcapP | LFHCAL | HcalEndcapN | HcalBarrel |
|---|---|---|---|---|---|---|---|
| π | 480 | +2.38 | 1.5 % | **49.4 %** | 37.7 % | 1.7 % | 3.8 % |
| **e** | 299 | **−2.50** | **93.6 %** | 0.7 % | 0.7 % | 65.9 % | – |
| K | 49 | +2.16 | 0 % | 46.9 % | 49.0 % | 2.0 % | 6.1 % |
| p | 59 | +2.32 | 0 % | 45.8 % | 39.0 % | 3.4 % | 5.1 % |

E/p medians (forward leg): π 0.249, K 0.276, p 0.321 — textbook MIP-like.
**Electrons live in the backward leg**, so e-ID uses `EcalEndcapN` + dRICH +
backward HCAL; a single pooled model would mostly learn hemisphere + flux rather
than ionisation/shower physics ⇒ leg-scoped models with a pooled cross-check arm.

## 4. Repository layout
```
pid/
  __init__.py  __main__.py  cli.py
  config.py     # classes, legs/regions, feature list, thresholds, stats floors, artifact paths
  schema.py     # FEATURE->{campaign:branch} alias table; assert_ml_env(); population census
  links.py      # podio_metadata cid->name registry; verified join table; broken-link reporter; DeltaR matcher
  features.py   # build_features(files, leg, ...) -> one row per candidate track
  dataset.py    # label policy, NaN policy, grouped splits, pickle/npz persistence
  models/{__init__,base,lightgbm_model,xgboost_model,sklearn_hgb}.py
  train.py      # grouped search + CV, weights, gates, calibration, artifacts
  evaluate.py   # ROC/AUC, eff@fixed-fake, n_sigma, per-(p,eta,species) tables, calibration
  importance.py # gain, exact SHAP, permutation importance, physical-sign check
  compare.py    # clean-vs-bkg via trkperf.compare on pid metric JSONs
  report.py     # reuses trkperf.report (json/md/TNtuple) + pid provenance
  significance.py # FOM(c)=S/sqrt(S+B): scan, optimum + tie-break, per-bin optima,
                  #  Clopper-Pearson limits, class/lumi weighting
  performance.py # channels (incl. 2-hypothesis reduction), eff/fake/purity at c*,
                  #  2D maps, n_sigma vs p, confusion, overtraining, pid-performance driver
  plots.py      # ROC, eff-vs-fake, n_sigma vs p, SHAP, FOM(c) per bin (overlay+panels),
                  #  2D maps, confusion heatmap, train-vs-test score dist
  tests/        # unittest suite, network-free
   scripts/run_pid.sh
   scripts/check_learners.py  # cross-learner AUC-agreement gate (advisory)
  README.md
```

## 5. Features (v1 — as built, every entry link-verified)

**Detector response (model inputs by default).**
*Calorimeter, per leg:* `e_over_p_<leg>`, `log_e_over_p_<leg>`,
`ecal_<leg>_E/n/t`, `ecal_<leg>_delta_r`, `ecal_<leg>_source` (bookkeeping),
transverse shapes computed from that cluster's hits (`rms_x`, `rms_y`, `rms_r`,
`core_frac`), the producer's unlabelled `shape_0..shape_6`, plus HCAL response
`e_hcal_<leg>`, `leakage_<leg>`, and the per-cluster composition diagnostic
`e_hit_over_e_clu` (bookkeeping, excluded). Shapes always follow the ΔR-winning
cluster — the same one `<prefix>_E` (hence E/p) is read from — never the file's
first built-in pair; geometric-only winners get real shapes, not NaNs.
*Cherenkov, per track:* `has_drich_gas/aerogel` (1 iff a radiator pathlength was
measured, not iff a link row exists), `drich_{gas,aerogel}_pathlength`
(734–1040 mm measured; verified link to `CentralCKFTracks`).
*Tracker:* `chi2`, `ndf`, `chi2_per_ndf`, `track_time`, `track_time_error`,
`proj_pathlength`.
*Event context (ablatable with `--no-event-level`):* `n_tracks_evt`,
`n_ecal*_clusters_evt`, `e_ecal*_sum_evt`, and the event-level dRICH IRT summary
`irt_{gas,aerogel}_{nobj,npe_tot,npe_max}_evt` (NaN when no photons, with
`has_irt_*_evt` presence flags — absent data is never coded 0.0). Tracks with
unmeasurable kinematics get `leg = NaN`, never a fallback hemisphere.
*Optional, off by default (`--enable-ionisation`):* reco-side `edep_{si,mpgd}_mean/n`
(21.6 % track coverage in 26.02.0; **not available in 26.07.1** — see 3.2).

**Deliberately not model inputs.**
*Track kinematics* `p, pT, eta, phi, px, py, pz` — measured, physical, but in this
sample they are the DIS **flux** shortcut: the scattered electron is the hardest
track in the backward leg, and with kinematics in the matrix the boosted trees rank
`pt` **first** and E/p second (measured: gain `0:pt 1:e_over_p_backward`). With them
out, `e_over_p_backward` leads both gain and SHAP. PID performance is therefore
*binned* in (p, pT, η) by `evaluate.binned_table`, and `--include-kinematics` runs
the explicit "full tagging" variant for comparison (`pid.config.KINEMATIC_COLUMNS`).
*Signed `charge`* — beam-configuration tag, not detector response (measured
correlation with the label: −0.74); `--use-charge` re-enables it.
*Producer PID hypotheses* (`CentralCKFTracks.pdg`,
`ReconstructedChargedRealPIDParticles.goodnessOfPID`) — label-adjacent; kept as
cut-based **baselines**, never inputs.
*Truth-side `eDep/pathLength`* — derived from the particle being classified, so it
would leak the label; only the reco-side proxy is offered.

**Missing values** stay `NaN` **plus** a `has_*` flag — never zero-filled, never
imputed (all three learners take a learned default direction for NaN).


## 5b. Working points by maximum significance (`pid performance`)

For each channel — `eid` (e/π), `ehad` (e/hadron), `Kpi` (K/π), `pK` (p/K) — the cut is
chosen to maximise

    FOM(c) = S(c) / sqrt(S(c) + B(c))

over a threshold grid, globally and per pT bin, with S/B the accepted weighted
yields. Hadron channels are reduced to two hypotheses,
`score = P(sig)/(P(sig)+P(bkg))` over truth in {sig, bkg}, so the reported c* is in
that channel's score space and is **not** comparable with another channel's cut.

Rules the implementation enforces, because the naive version of this analysis
reports fabrications:

* **FOM is not scale invariant.** It grows as √(luminosity) and its optimum moves
  with the assumed composition, so `--lumi-scale` / `--bkg-scale` are recorded in
  every output (`pid/significance.py`).
* **Weighted variances.** The companion `significance_eff` uses Σw² (the Poisson
  variance of a weighted sum), not Σw; with an up-weighted background the naive
  form is the optimistic one. Σw² and Kish's (Σw)²/Σw² are separate quantities and
  named separately.
* **No 1/0 rejection.** Where zero background candidates pass, the point estimate
  is NaN and the 95 % Clopper-Pearson **lower limit** is reported instead
  (`rejection_lower_limit`), drawn as limit markers rather than a curve reaching 10⁴.
* **Degenerate optima are labelled.** With few background candidates FOM → √S and
  its maximum migrates to "accept everything"; `caution` says `degenerate` or
  `permissive optimum` (both observed in this project's own smoke runs: p/K, K/π)
  rather than presenting it as a working point.
* **Ties broken towards the more rejective cut**, since FOM is flat over the
  plateau between the last accepted background candidate and the next signal one.
* **Per-bin optima never interpolate.** A pT slice lacking either species yields no
  curve and a grey panel with the reason, so the figure itself shows which slices
  need more files.

Figures per channel (in `output/plots/`, named as requested):
`significance_vs_cut`, `significance_vs_cut_by_pt` (FOM(c) for every pT bin, absolute
+ self-normalised), `significance_vs_cut_panels` (one panel per bin with c*, FOM, ε,
fake), `optimal_cut_vs_pt`, `roc`, `eff_vs_pt`, `misid_vs_pt`, `purity_vs_pt`,
`nsigma_vs_p`, `eff_map_pt_vs_eta`, `fake_map_pt_vs_eta`, `pion_rejection_vs_p` (e-ID),
`rejection_vs_p`, `score_dist_train_vs_test` (with a KS statistic per class), plus
task-level `pid-hadpid_lightgbm_clean-nsigma_vs_p` (K/π and p/K) and
`pid-hadpid_lightgbm_clean-confusion_matrix`
(held-out rows only). Task-level names carry model_tag, and per-tag channel
figures live in `output/plots_<tag>/` - without both, a bkg run silently
overwrites the clean figures under the same stem. The two task-level figures are extras outside the 13
per-channel required deliverables: the confusion matrix describes the multi-class
task and is undefined for the binary channels, so the manifest can neither require
nor sensibly miss it (see §5e). Every table also exists as JSON/markdown/ROOT
TNtuple, and
`output/pid-working_points_<model>_<tag>` summarises (c*, peak FOM, ε, fake, purity,
rejection + limit, caution) per channel. Binning: 13 uniform pT bins over
0.5–10 GeV/c and 14 momentum bins to 20 GeV/c (`config.PT_BINS_PERFORMANCE`,
`P_BINS_PERFORMANCE`) — deliberately *not* the log-spaced tracking bins.

## 5c. Definitions of every figure of merit and metric

Conventions are fixed here so a number cannot be quoted with an unstated
definition. `S` = accepted **signal** yield, `B` = accepted **background** yield,
`S_tot`/`B_tot` = the corresponding totals in the sample being measured; a yield is
always the sum of per-candidate weights (`w = 1` unless `--bkg-scale` /
`--lumi-scale` or balanced class weights are used, which are then recorded in the
output).

### Working-point merits

| name | definition | implemented in | read it as |
|---|---|---|---|
| **FOM(c)**, `significance` | `S(c) / sqrt(S(c) + B(c))` with acceptance `score >= c` | `pid/significance.py::scan_thresholds` | the requested maximum-significance merit. Not scale invariant: it grows as `sqrt(luminosity)` at fixed composition, so it is meaningless without the recorded `lumi_scale` |
| **c\*** , `threshold` | `argmax_c FOM(c)` on a 401-point uniform grid in [0,1] (`config.N_THRESHOLD_SCAN`) | `pid/significance.py::optimal_cut` | the cut **in that channel's score space**, i.e. a probability-like classifier output, never a GeV or a detector quantity. Cuts from different channels are not comparable |
| `significance_eff` | `S(c) / sqrt(S_eff(c) + B_eff(c))`, `S_eff = Σw_sig²`, `B_eff = Σw_bkg²` | same | the Poisson-corrected form: the variance of a *weighted* count is `Σw²`, not `Σw`. With an up-weighted background it is always ≤ `significance`; the gap is how much the weighting flatters the naive form |
| `S_eff`, `B_eff` | `Σ w²` (Poisson variance terms) — **not** Kish's `(Σw)²/Σw²` | `sum_of_squares` vs `kish_effective` | two different concepts kept deliberately separate and separately named; conflating them inflates weighted significance |
| tie-break | the **highest** tied `c` wins | `optimal_cut` (sort by `[significance desc, threshold desc]`) | FOM is flat between the last rejected background candidate and the next signal one; choosing the top of the plateau is the more rejective, conservative choice |
| `rejects_background` | `fake_rate(c*) < 0.5` | `optimal_cut` | a working point is only a selection if it rejects something |
| `caution` | `degenerate` (eff and fake both 1 ⇒ no separation), `permissive optimum` (fake > 0.5), `optimum at zero cut` (statistics-limited, since FOM → `sqrt(S)` when `B → 0`), `pinned at the top of the score range`, `< 5 background accepted` | `optimal_cut` | **always read this column.** Three of the four local channels currently carry a caution; presenting those c\* values as working points would be wrong |
| `c*(pT)` | the same maximisation inside each pT bin (13 uniform bins, 0.5–10 GeV/c) | `optimal_cut_in_bins`, `significance_curves_by_bin` | a bin without both species or with `< 20` candidates yields **no curve and no cut** (NaN + reason), never an interpolated or borrowed one |
| plateau fraction | share of the scanned range within 1e-9 of the maximum, annotated "flat top: N% of range" | `pid/plots.py::significance_vs_cut_panels` | how arbitrary the cut choice really is in that bin |

### Selection merits at a chosen cut

| name | definition | notes / trap |
|---|---|---|
| **efficiency** `ε` | `S(c)/S_tot` | signal acceptance. Weighted numerator and denominator; the interval is computed from **observed counts** (see below) |
| **fake rate** `f` | `B(c)/B_tot` | mis-ID rate = background acceptance. The denominator matters: quoting 1/10 = 10 % as "10⁻¹ rejection" when the true rate could be 1/2 is the classic error |
| **purity** | `S(c)/(S(c)+B(c))` | the analysis-facing quantity; strongly composition-dependent, so a purity claim must state the assumed flux (here: what the test sample contained, or the `bkg_scale` used) |
| **rejection** `R` | `1/f(c)` | defined only when `f > 0`; **NaN, never ∞, when nothing passes** |
| `rejection_limit` | `1 / f_upper`, `f_upper` = one-sided 95 % Clopper-Pearson limit on `k_bkg/n_bkg` | the honest quantity to quote from a background-empty tail. With 10 pions and 1 passing, `R = 10` but only `R > 2.5` is established; the figures draw these as limit markers |
| `meets_target`, `rejection_measurable` | achieved fake ≤ requested target; and `k_bkg > 0` | `efficiency at 1e-4` from 10 background candidates is not a measurement of anything |

### Cut-independent merits

| name | definition | implemented in |
|---|---|---|
| **AUC** | `P(score_sig > score_bkg)`, rank-based; `multi_class="ovr"` for multiclass | `pid/evaluate.py::auc` (sklearn `roc_auc_score`) |
| **n\_sigma** | **default (`quantile`)**: `2 * erfinv(2 * AUC - 1)` — exact for equal-width Gaussians since `AUC = Phi(d/sqrt(2))`, needs no fit, and is invariant under any monotone rescaling of the score. **Alternative (`logit`)**: `abs(mu_1 - mu_2) / sigma_pooled` from two **ROOT** single-Gaussian fits to `z = ln(P/(1-P))`. `raw` (fitting the bounded score) is **deprecated** — see the trap below | `pid/evaluate.py::nsigma_from_auc`, `measure_separation`, `logit`, `n_sigma` |
| peak fits | require ≥ 50 finite entries (`MIN_ENTRIES_PER_BIN`); `converged=False` with NaN parameters on failure — never a fabricated width | `fit_gaussian` |
| KS statistic | `ks_2samp(train scores, test scores)` per class, printed on the overtraining figure | `pid/performance.py::overtraining_check` |
| confusion matrix | row-normalised = efficiency view (fraction of true-i predicted as j); `col_fraction` = purity view; **held-out rows only** | `pid/performance.py::confusion_matrix` |

### Why the raw score is never fitted (the n\_sigma trap this replaced)

Boosted scores are probabilities confined to [0, 1]. A good classifier piles the
signal against 1 and the background against 0, so each distribution is truncated by
the *axis* rather than by its own tail; a Gaussian fitted to it returns a width set
by the clipping. Measured on the local K/π sample: both peaks at ~1.0 giving
n\_sigma = 0.0065, and on synthetic saturated scores the raw fit inflates n\_sigma to
**46** where the quantile and logit estimators agree at **16**. The fix is structural,
not cosmetic:

* `logit(P)` unbounds the score, so peak widths mean what they claim to;
* the **quantile** estimator needs no fit at all and therefore cannot diverge;
* `AUC = 1` returns NaN from `nsigma_from_auc` — a finite sample with zero score
  overlap has *no* point estimate of separation, so `nsigma_lower_limit` is reported
  instead (below);
* `auc`/`n_sigma` remain available **below** the fit floor: the 50-entry minimum gates
  the Gaussian fits, not the rank statistics (`fit_limited=True` records which
  happened), so 10 background candidates still yield an AUC — imprecise, but real.

### Saturated separation: `n_sigma_lower_limit`

With `n_sig x n_bkg` candidate pairs and no overlap, the 95 % CL lower limit `d_L`
solves `P(all background below all signal | d_L) = 0.05`. The probability is the
order-statistics integral

    P(d) = int_0^1 n_s (1-u)^(n_s-1) Phi(Phi^-1(u) + d)^n_b du

which is **not** the product of independent pair comparisons: independence
(`AUC^(n_s * n_b)`) *underestimates* P(no overlap) — at `d = 4.21`, where the
independence model itself reaches `P = 0.05`, the exact value is `P ≈ 0.47`, so
the approximation would set the 95 % CL limit at `d = 4.21` instead of the exact
`3.20` — a limit ~31 % stronger than the data supports. Validated in
`pid/tests/test_evaluate.py` against the exact combinatorial result at `d = 0`,
`P = 1/binom(n_s+n_b, n_s)`, and against Monte Carlo at the solved limit (observed
rate 0.0495/0.0508 against the nominal 0.05).
For this sample's held-out rows the separation did **not** saturate (AUC 0.9926 →
point estimate `n_sigma = 3.45`), so no limit is quoted. The limit rule is what
applies *if* it saturates: perfectly separated 204 electrons vs 10 pions would give
`n_sigma > 3.20 (95 % CL)` — a hypothetical for these counts, not a measurement.

### In-sample optimum (look-elsewhere note)

`c*` is chosen by maximising FOM over the ~400-step threshold grid on the same
held-out rows that the at-`c*` figures quote efficiency and fake rate from, so the
quoted numbers are an in-sample optimum — mildly optimistic, and compared fairly
across the two tags because both select the same way (the clean-vs-bkg ratios,
this project's headline, cancel it to first order). A N-fold re-split that removed
it is a `pid` roadmap item, not a current deliverable.

### Uncertainty conventions (one meaning per column name)

* `*_err_lo`, `*_err_hi` = **central 95 % Clopper-Pearson** interval on observed
  counts (`k` passing of `n`), i.e. `beta.ppf(0.025)`/`beta.ppf(0.975)`;
  implemented twice independently (`pid/evaluate.py::garwood`,
  `pid/performance.py::binomial_interval`) and cross-checked against
  `scipy.stats.binomtest(...).proportion_ci(method="exact")` in the tests.
  Unifying these was a real fix: the performance layer previously used one-sided
  bounds under the same column names as the evaluate layer's central interval.
* `*_upper_limit` (`fake_rate_upper_limit`) = **one-sided 95 %** limit, used only for
  limits (`pid/significance.py::garwood_upper`).
* Point estimates in the FOM layer use the (possibly weighted) yields; intervals
  and rejection limits are **count-based by construction** — identical for
  class-constant weights, approximate under a `--bkg-scale` rescaling (which also
  reaches the per-class cuts).
* `value_err` = half-width of the central interval, existing only so
  `trkperf.compare` can propagate errors with its `<name>`/`<name>_err` convention;
  the asymmetric columns are kept beside it and are the ones to quote.
* `insufficient_stats` uses **`min(n_signal, n_background) < MIN_ENTRIES_PER_BIN`**
  in both `pid/evaluate.binned_table` and `pid/performance.at_cut_table`: a bin with
  no background candidates has an undefined fake rate, purity and rejection, so it is
  flagged rather than reported as a row of silent NaNs.
* Confidence level everywhere: `config.REJECTION_CL = 0.95`.

### Statistics floors and flags (when a number is refused rather than reported)

| constant | value | governs |
|---|---|---|
| `MIN_ENTRIES_PER_BIN` (from `trkperf.config`) | 50 | `insufficient_stats` in every table; the minimum for a peak fit |
| `MIN_CANDIDATES_PER_WP_BIN` | 20 | whether a pT bin may have a c\* at all; masks 2D map cells |
| `MIN_ROWS_PER_CLASS` | 30 | refuses to **train** a class this thin |
| `n_thresholds` | 401 (global) / 201 (per-bin curves) | cut resolution; a 0.5-wide "flat top" is a property of the sample, not the grid |

### Training/evaluation gates (merits of the *pipeline*, enforced by `pid train`)

| gate | criterion | rationale |
|---|---|---|
| overfitting | `AUC_train − AUC_test ≤ 0.02` | file-grouped split; violation = memorisation |
| permutation control | real `AUC_test` beats the mean of **5 balanced label-permuted refits** by `z ≥ 3` **and** `ΔAUC ≥ 0.10` | the control's own spread (~0.14 with 10 test electrons) is why a fixed tolerance about 0.5 was the wrong test |
| physical attribution | `e_over_p` must lead the **SHAP** ranking for `eid`/`ehad`/`pooled` (measured on held-out rows; missing/unmeasurable E/p refuses training up front); top-3 by gain must be a detector-response family | gain counts usage, SHAP measures decision; the plan's claim is about the latter. Advisory for `hadpid`, where no single observable should dominate |
| leakage | no feature with `abs(corr(feature, label)) > 0.95` (`ValueError` halt); fail-closed allowlist (unlisted numeric columns dropped with a loud warning) + explicit NEVER-model patterns (`mc_`/`true_`/`gen_`, weights, event/run/file ids); single-class-measured columns reported as perfect correlation | `rec_idx`, a join artefact, once out-ranked E/p — hence the pattern rule, not a list |

### What to quote for which claim

* "the selection works" → **efficiency at a stated fake rate**, with
  `rejection_limit`, at the fixed-fake working point from `pid evaluate`
  (`config.FAKE_RATE_TARGETS`), **not** at c\*.
* "this is the best cut for this analysis" → **c\* and the FOM at c\*** from
  `pid performance`, with `caution`, `lumi_scale` and composition stated.
* "the detectors separate these species" → **AUC and n\_sigma**, which need no cut
  choice at all.
* "background degrades PID" → differences of the above between tags from
  `pid compare`, with the schema-skew caveats of §3.3, since a composition-driven
  FOM change is not a detector effect.

## 5d. How reconstructed tracks and truth particles enter every merit

The merits of §5c are computed on a table whose rows are **reconstructed track
candidates**, never truth particles. This section pins down exactly which
reconstructed quantity, which truth quantity and which pairing rule produces each
number, with the measured values from the clean reference file (1,083 events,
4,250 candidates) so nothing here is a description of intent.

### The row

| aspect | rule | measured / where |
|---|---|---|
| identity | one row per `(file_id, event, track_idx)`; `track_idx` is the position inside that event's `CentralCKFTracks` collection — the index space that `CentralCKFTrackAssociations._rec.index` and every `_…_track.index` relation address | `trkperf.io.read_flat`; 4,250 rows = 4,250 unique keys, no duplicates |
| not truth-keyed | a truth particle that produced **no** track contributes to nothing here (that population is `trkperf efficiency`/`acceptance`), so no PID merit is ever divided by a truth-particle count | `pid.dataset.prepare` |
| momentum | `p = 1/\|qOverP\|` and `px,py,pz` from `(theta, phi)` of `CentralCKFTrackParameters`; `p, pT, eta, phi` then via `trkperf.truth.add_kinematics` — the **same function** used for truth, so reco and truth quantities are directly comparable | `trkperf.reco`. `CentralCKFTracks.momentum.*` is all zeros in these files and is never read |
| track quality | `charge, chi2, ndf, time, timeError` from `CentralCKFTracks` | `pid.features` |
| truth partner | best association **per reconstructed track** (`_best_association` grouped on `file_id, event, rec_idx`), valid iff `weight >= config.MATCH_WEIGHT_THRESHOLD` (0.5); the particle is `MCParticles[sim_idx]` **of the same event** | `trkperf.matching.build_matched_tracks` |
| primary filter | truth rows are read with `primary_only=True` (`generatorStatus == 1`) | `trkperf.truth.read_truth_particles` |
| class | `config.CLASS_OF_ABS_PDG[abs(PDG)]` → `e, pi, K, p`; charge conjugates share a class because the detector response depends on \|PDG\|, only the curvature sign differs | `pid/features.build_features_one_file` |

### Who lands in S and who in B

`S`/`B` in every merit are counts of **rows** (candidates) of a given class that pass
the cut — so both numerator and denominator are reconstructed objects labelled by
their truth partner.

| population | clean file | in S/B? | why |
|---|---|---|---|
| truth-matched, class in the task | 4,166 | yes | the measurable PID sample |
| truth-matched, `truth_pdg` NaN | **72** | no | the best association points at a **non-primary** particle (decay product, δ-ray), which the primary filter has already removed → no class. Never counted as background |
| truth-matched, \|PDG\| = 13 (muon) | **4** | no | muons are not in `CLASS_OF_ABS_PDG` (0.09 % of matched tracks here; decay-in-flight, not a generator class in this sample) |
| unmatched (`is_fake`) | **8** | no (default) | `prepare(require_matched=True)`. With `require_matched=False` they carry `label = -1` for a future "fake tracks faking electrons" study (no CLI path yet — `train` hardcodes `require_matched=True`), and they are excluded from training either way |
| in the `eid` score table after all cuts | 1,100 rows (1,038 e / 62 π) | — | `train.py`; 214 held out (204 e / 10 π) |
| association weights actually observed | — | — | 0.60–1.00 over the whole channel (all ≥ 0.5 by the threshold); the 214 held-out rows of this smoke run all sit at 1.00 |

### Which rows the merits are computed over

Merits are evaluated on the **held-out rows**, never on the rows the model was fit
to — this is the difference between a PID performance number and a memorisation
measurement. `pid.train` writes both partitions into `all_scores.pkl` with a
`sample` column (`train`/`test`), `pid.evaluate` reads `test_scores.pkl`, and
`pid.performance` takes the test rows out of `all_scores.pkl` explicitly
(`test_only`); the train rows are used only by the overtraining figure
(`score_dist_train_vs_test` and its KS statistic). Consequences worth stating:

* on a one-file smoke run the held-out set is small by construction (here 214 rows,
  10 pions), so the S and B entering every merit are those 214 rows — which is why
  the local tables read "below the 50-entry floor";
* with several files the split is by `file_id`, so held-out rows come from files the
  model never saw (never from the same events);
* the multi-class confusion matrix is likewise built from held-out rows only
  (measured: 414 rows out-of-sample, K recalled at 27 %, against 72 % in-sample —
  a gap that only a train/test-mixed matrix would hide).

### Which input feeds which merit

| merit | score / cut applied to | class label from | binning column | weights |
|---|---|---|---|---|
| FOM(c), c\*, AUC, ROC, nσ, purity, fake, rejection | the classifier output for that channel (reco detector response only) | truth partner of the track, as above | `--bin-source` basis (see below) | per-row `w` from `flux_weights`/`lumi_scale`, default 1 |
| efficiency ε | `S(c)/S_tot`, S over truth-matched signal rows | same | same | same |
| confusion matrix | `argmax` of `proba_<class>` | `truth_class` | (none; global) | n/a — **held-out rows only** |
| kinematic acceptance / tracking efficiency (not a PID merit) | — | truth particle | truth `pT`, `eta` | n/a; lives in `trkperf`, keyed the other way round |

The truth particle contributes **labels and (optionally) the binning axis only**. It
never enters a score: `pid.dataset` blocks every `truth_*` column and every
truth-derived family from the feature matrix, which is what makes "S is the number of
truth-electron tracks passing c*" a legitimate statement rather than circularity.

### Binning basis: reconstructed vs truth (`--bin-source reco|truth|both`)

Every binned merit can be partitioned by the **reconstructed** `p, pT, eta` (default:
what a cut is actually applied to in an analysis) or by the truth partner's
`truth_p, truth_pt, truth_eta` (what `trkperf`'s tracking metrics bin in, per
AGENTS.md "bin in truth pT/eta", and free of resolution migration).

* **Same candidates, different partition.** Verified on the clean file (held-out
  rows): both bases give identical `sum(n_signal) = 204` and `sum(n_background) = 10`; 2 of 5 pT bins
  change population (max shift 2 tracks), and the per-bin efficiency changes by at
  most 0.0003 at this statistics. Only the migration moves, never the count.
* **Scale of the migration** (measured, clean file): median
  `|pT − truth_pt| / truth_pt = 1.5 %`, median `|eta − truth_eta| = 0.0017`.
* **The basis is recorded, never implied**: `variable`, `binning_basis` and
  `bin_column` columns in every table, `binning_basis` in every JSON `meta`, the
  x-axis label of every figure (`track p_T` vs `truth p_T`), and `*_truthpt` file
  suffixes when both bases are produced in one run.
* **A global merit does not depend on the basis** — c\*, global FOM, ε, fake and
  purity are computed over the whole channel, so `--bin-source both` reports them
  once per basis with identical values by construction. Only the differential tables
  and the per-bin c\*(pT) differ.
* Older score tables lack the truth columns; the code then names the missing column
  and says to re-train rather than silently falling back to the reco basis.

## 5e. Deliverables: channel status and the figure manifest

**Channel status.** Each entry in `config.CHANNELS` / `config.TASKS` carries a
`status`, propagated into every figure title, the `pid-working_points` table and
`performance.BIN_VARIABLES`-independent output:

| status | channels | meaning |
|---|---|---|
| `headline` | `eid`, `ehad` | electron ID is the result this pipeline is built to deliver |
| `exploratory` | `Kpi`, `pK` | **calorimeter-only hadron ID is a baseline, not a result.** With the dRICH IRT `chargedParticle` link dangling and no per-track dE/dx (§3.2), nothing in the feature set can separate same-momentum hadrons beyond leakage differences; measured K/π AUC 0.57 with a degenerate working point. Full π/K/p separation is deferred to §12.2, where the two link options are recorded. Quoting the n\_sigma it currently produces (0.35 for K vs rest) as a PID result would be a category error |
| `cross-check` | `pooled` task | exists to quantify the hemisphere shortcut, not to be quoted as a PID |

Each exploratory figure also gets a `pid-<channel>_<figure>.png.note` sidecar stating
`status: exploratory` and why, so a plot exported out of `output/plots/` still carries
its caveat, and the manifest table has a `status` column so a table of results cannot
silently mix a baseline with a measurement.

**Manifest.** `config.REQUIRED_FIGURES` lists the 13 deliverables per channel (the 9
requested plots plus the 4 working-point diagnostics). `pid.performance.
verify_deliverables` checks each one **on disk** — a plotting function returning a
path proves nothing — and `manifest_table` flattens the result to one row per
(channel, basis, figure) with `present` and `reason`, written to
`output/pid-deliverables_<model>_<tag>.{json,md,root}` and printed as
`[deliverables/eid] complete: 13/13 figures`. The task-level confusion figure is
deliberately *not* one of the 13: it is listed in `BASIS_INDEPENDENT_FIGURES`
because it is written once (task name, under the reco basis), but a per-channel
manifest can neither require nor sensibly miss a figure that is undefined for the
binary channels.

A gap is never silent, and the reason distinguishes the two ways one arises:

* `needs a valid working point c*; none exists because the FOM maximum is
  degenerate` — a statistics verdict (an exploratory hadron channel will show this);
* `per-bin curves disabled (--no-binned-curves)` — an operator choice;
* run-specific causes are named precisely when known (e.g. `needs all_scores.pkl
  ... retrain with the current pid.train`, `too few held-out candidates per class
  to compare train vs test distributions`), so a gap is actionable rather than a
  puzzle.

`--require-figures` turns any gap into a non-zero exit for CI-style use; it is off by
default because on small samples the gaps are the finding. The electron channel
satisfies `rejection_vs_p` through its physics-named alias `pion_rejection_vs_p`.

## 6. Metrics and outputs
(Every quantity is defined with its formula and traps in §5c; §5d states which
reconstructed and truth information each one is computed from.)
AUC (global + per (p,η) slice); **efficiency at fixed π fake rate 1e-2/1e-3/1e-4**
with Garwood intervals; **nσ(K/π)** and **nσ(e/π)** from the fit-free quantile
estimator `2*erfinv(2*AUC-1)` by default (the two-Gaussian logit-space fit is the
opt-in `--nsigma-method logit`; see §5c), binned in (p, η) using `trkperf.binning`'s
ROOT convention;
row- and column-normalised confusion matrices per slice; gain + exact SHAP +
permutation importance with a physical-sign check (E/p must push electron-like);
calibration reliability curve. Outputs: `output/pid-<what>_<tag>.{json,md,root}`
+ PNGs, `MIN_ENTRIES_PER_BIN = 50` flagging, and `trkperf/compare.py`-style
clean-vs-bkg ratio/difference tables.

## 7. Milestones and gates
| M | Deliverable | Human command | Gate |
|---|---|---|---|
| M0 | `schema.py`, `links.py`, `cli.py schema-check` | `python -m pid schema-check --dataset clean --file data/dataset_small/signal` | every admitted feature present **and** non-degenerate in both campaigns; unresolved cid families reported, not silently dropped; `assert_ml_env()` passes |
| M1 | `features.py`, `dataset.py`, `cli features` | `python -m pid features --file <local> --dataset-tag clean --limit-files 1` | rows == candidate tracks; backward-e median E/p > 0.8 and forward-π < 0.4; no all-NaN column kept; `--cache-dir` + `--max-file-failures` behave |
| M2 | cut-based baseline (as-built: fixed-fake-rate working points from `evaluate`) | `python -m pid evaluate --task eid --model lightgbm --dataset-tag clean --scores <test_scores.pkl>` | baseline AUC/nσ recorded as the bar ML must beat |
| M3 | Stage-1 e/π (LightGBM) | `python -m pid train --task eid --model lightgbm …` | held-out AUC; train−test gap < 0.02; **E/p top-1 by gain and SHAP (smoke scale; FAILs at grid scale — see the §11 grid-training note)**; label-shuffle control → AUC 0.50 ± 0.01 |
| M4 | Stage-2 π/K/p + pooled | `python -m pid train --task hadpid --model xgboost …` | 3 learners within ΔAUC 0.02 (checked automatically, advisory, by `pid/scripts/check_learners.py` at the end of every grid run); nσ(K/π) monotonic in p (manual check — not enforced in code) |
| M5 | physics report | `python -m pid evaluate … && python -m pid importance … && python -m pid performance …` | χ²/ndf ≈ 1, ≥50 entries/bin or flagged, provenance complete |
| M6 | background impact | `python -m pid scripts/run_pid.sh` bkg run, then `python -m pid compare …` | cross-application (clean→bkg, bkg→clean) reported; skew features excluded from headline arm |
| M7 | skills + docs | dry-run on 5 files | `.opencode/skills/pid-ml/`, `pid-performance/`; AGENTS.md gains §3 facts |
| M8 | statistics escalation | tier runs | each quoted fake rate at its stated precision, or reported statistic-limited |

## 8. Statistics budget
Backward pions ≈ 7 per 300 events ⇒ full clean (~5.0 M events) ≈ 1.2 × 10⁵, enough
to measure 10⁻⁴ rejection; a 200-file bkg run (~20 k events) ≈ 450, so **10⁻³ is
the bkg ceiling** unless the sample escalates to minQ2 = 100/1000 (harder, more
central hadron flux) per AGENTS.md's order. Fake rate is reported two ways: mis-ID
among truth-matched tracks, and over all reco tracks (includes combinatorial
background fakes).

## 9. Standing safeguards
Gun samples for efficiency calibration **only** (never for fake rate — AGENTS.md);
a leakage tripwire test failing if any feature correlates > 0.95 with the label;
`CentralCKFTracks.pdg` / `ReconstructedChargedRealPIDParticles.goodnessOfPID` kept
as baselines, never as model inputs; region-normalised
`E/p / median(E/p | leg, p-bin)` variant so campaign drift cannot masquerade as a
background effect; matching-threshold sensitivity (0.5 vs 0.8) as a systematic.

## 10. Defaults chosen (overridable)
1. Code in `pid/` importing `trkperf` (not inside `trkperf/`).
2. Leg-scoped models + pooled cross-check arm.
3. dE/dx out of v1 (absent from the data); IRT hypotheses excluded from per-track
   features until a real link exists.
4. Headline deliverable = ML PID on both datasets with the clean-vs-bkg comparison.

---

## 11. As-built record (this session)

**Delivered** (`pid/`: 21 files (7,855 lines) + 13 test modules (3,071 lines) = 10,926 lines; re-verified 2026-09-20):

| module | role |
|---|---|
| `config.py` | classes, legs, tasks, thresholds, gates, search spaces, paths, thread policy |
| `schema.py` | 97-entry `FEATURE → {campaign: branch}` table, 15 documented dead ends with measurements, `assert_ml_env()` |
| `links.py` | `podio_metadata` collectionID registry, relation audit, vectorised offset/relation unpacking, ΔR matcher |
| `features.py` | per-file feature builder (+ per-file disk cache, `max_failures`), shower shapes from hits, ionisation proxy, IRT event summary |
| `dataset.py` | label policy, NEVER-model patterns + fail-closed allowlist, NaN-preserving design matrix, file-grouped splits, leakage tripwire (> 0.95 halts) |
| `models/` | LightGBM / XGBoost / sklearn-HGB adapters: `estimator/gain/shap` (binary-exact SHAP; multiclass XGBoost averages attribution over classes via `as_sample_feature_matrix`, LightGBM keeps the first class block — documented asymmetry, diagnostic only) |
| `train.py` | grouped `RandomizedSearchCV` (groups reach the splitter), balanced class weights, E/p-presence precondition, test-set SHAP, permutation control, 4 gates, artifacts (`model.joblib` calibrated + `booster.joblib` unwrapped for importance) |
| `significance.py` | FOM(c) threshold scan (score-ordered weights), `optimal_cut` + per-bin c\* on one grid, exact per-row yields; limits count-based |
| `performance.py` | max-significance working-point package: global + per-bin c\*, at-cut tables, maps, nσ vs p, confusion, overtraining, 13-figure manifest with reasons |
| `evaluate.py` | ROOT Gaussian fits, nσ, Garwood intervals, working points, binned tables, confusion (with decision rule), ROC, calibration |
| `importance.py` | gain, exact SHAP, permutation, physics check |
| `plots.py`, `report.py`, `compare.py` | figures; JSON/MD/TNtuple via `trkperf.report` (text columns factorised); clean-vs-bkg via `trkperf.compare` |
| `cli.py` + `scripts/run_pid.sh` | `python -m pid {schema-check,features,train,evaluate,importance,performance,compare,all}`; detached cached grid launcher (trains all three libraries, `--seed`, advisory `check_learners.py` agreement gate) |

**Package changes (additive, existing behaviour untouched — `python -m unittest discover -s tests` 30/30 OK):**
`trkperf/io.py`: `read_nested`, `event_count`, `expand_by_offsets`.
`trkperf/matching.py`: `build_matched_tracks` — the track-keyed truth↔reco table PID needs (§4 of `PLAN.md` named `matching.py` as the single choke point; this keeps that true).
`trkperf/compare.py`: optional explicit `join_keys` (PID rows are keyed by
quantity/signal/target-fake-rate, not by `(pt_bin, eta_bin, species)`; auto-detection
found no keys and raised, and guessing would compare an electron row against a pion row).
`trkperf/pid/__init__.py`: pointer to the new `pid/` package.

**Second audit round (fixes, all covered by new tests):** `pid importance` crashed or went empty on calibrated models — `train` now persists the unwrapped booster as `booster.joblib` and importance prefers it; `python -m pid all` crashed on a hand-built Namespace missing `include_event_level`/`include_track_time`/`relax_gates`/`n_jobs` (all forwarded now, `--no-calibrate`/`--n-jobs` added); `--min-q2-tier` was silently ignored (forwarded + recorded); a leg-prefixed diagnostic escaped the leakage blocklist (substring block); missing calorimeter pathlengths were filled with theta values (NaN now); SHAP attribution matched by substring (exact observable match now); `optimal_cut_in_bins` misaligned weights with NaN inputs, dropped per-bin cautions, and used a different grid than the drawn curves (all fixed + threaded); per-class cuts ignored `bkg_scale`; `pid compare --artifact all` never found its inputs with default `--model ""` (stem rule shared with `compare_artifact`); ROC comparison was a cartesian product (refused with a message); comparison summaries read meta keys `evaluate` never wrote (`*_scored` now); `vs_*`/`calibration` joins moved from float centres to string bins; `at_cut_table` folded NaN kinematics into the last bin and skipped empty bins (filtered + explicit NaN rows); `run_pid.sh` expanded to nothing inside its detached shell (rewritten with outer-shell expansion + third learner + seed + advisory cross-learner check `pid/scripts/check_learners.py`); confusion `threshold=0.0` collapsed to NaN; TRK side: shared failure budget per metric run, Wilson errors, full-grid explicit insufficient rows, string-bin compare joins, `validate="one_to_one"` reco merge. Two audit claims were refuted by measurement (LightGBM `pred_contrib` works; NaN join keys do compare) and are recorded as such in the session log.

**Third audit round (new items only; round-2 items above):** the matcher never returned the winning cluster so shapes and E/p could describe different clusters (winner `cluster_idx` now, shapes joined on it — 9/1089 incoherent + 56 shape-NaN tracks fixed on the local file); `has_drich` lied on missing lengths; absent IRT photons coded 0.0 (NaN + flags now); NaN-eta tracks filed as `"central"` (NaN leg now); single-class-measured columns auto-passed the leakage tripwire (reported as perfect correlation now); exact attribution match, per-bin caution propagation, one-grid threading, `figure_notes` for both bases, alias-aware sharing, string-bin `pid compare` joins with one-sided-pair attribution, `at_cut` NaN filtering + explicit empty rows, `booster.joblib` loader preference; new `pid-performance` skill covers evaluate/working-points/compare mechanics.

**Fourth round (external compliance checklist):** feature selection inverted from
blocklist to fail-closed allowlist (curated from real matrices; unlisted numerics
dropped with a loud warning; `mc_`/`true_`/`gen_` + weights + ids in explicit
NEVER patterns, zero collisions on the 110-column table); tripwire 0.99 → 0.95
(measured max legit 0.71) halting with `ValueError`; electron tasks refuse training
without measurable E/p; gate SHAP and `pid importance` restricted to held-out
files/rows; fold-disjointness asserted on the splitter the search actually uses.
All verified against grid-scale data with zero matrix changes.

Follow-up found by the suite (not by review): the all-NaN drop in
`design_matrix` left `model_columns` naming dropped columns, so the gate's
SHAP indices mislabeled onto wrong features (forward junk "leading" electron
SHAP). Fixed at both layers — `model_columns` drops unmeasurable columns so
`feature_columns` is honest (eid: 52 → 30 real inputs), and the gate indexes
`X.columns` ground truth. Provably model-neutral for tree learners (a column
no split can use changes no tree); the full suite caught it, now guards it.

Deployed-note (2026-09-18): this round landed while the clean/bkg grid jobs were
training. No restart was needed: the allowlist reproduces the production matrices
exactly (52/52/52/53, zero warnings), the tightened tripwire cannot trip (max legit
0.71, no NaN-correlation columns in any task — all four gates PASS on real data),
E/p is present everywhere, and fitting is seeded/deterministic, so model weights,
scores and AUCs are identical under old and new code. Only gate-verdict internals
(mixed- vs held-out-row SHAP) can differ in close cases; pending `importance` runs
uniformly use the stricter held-out restriction.

**Grid-launch finding (critical):** the first multi-file training attempt (clean_150)
crashed every task with `n_splits=3 greater than the number of groups: 1` —
`cross_validate` built a `StratifiedGroupKFold` but `search.fit()` never received
`groups=`, so the splitter saw one pseudo-group. Single-file smoke always dodged it
via the `StratifiedKFold` branch, which means **grouped CV never actually ran before**:
all previously reported models are single-file smoke fits (unaffected), and no
multi-file number was ever published. Fixed by passing `groups=groups` (regression
test: 3 file groups complete a grouped search), verified live on the relaunched
clean run. `run_pid.sh` for bkg_mixed was still on features when found, so it was killed
and relaunched cleanly onto the fixed code (cache replayed, no state lost).

**Grid training, final state (2026-09-19):** all 12 clean models
(eid/ehad/hadpid/pooled × lightgbm/xgboost/sklearn_hgb) and all 6 bkg electron
models trained with grouped CV, evaluates and importance; `check_learners.py`
PASS on every task/tag (clean spreads: eid/ehad 0.0001, hadpid 0.0063, pooled
0.0022; bkg eid 0.0009, ehad 0.0000). The physics gate still FAILs on every
electron model at scale — gain and SHAP both rank raw `ecal_backward_E` (and
`chi2` on some arms) above E/p itself. Not a code bug (artifacts written,
failure loud and recorded): with 161k rows raw cluster energy competes with
E/p, and the gate's "E/p leads" premise may be too strict at grid scale, or it
may be flagging real shortcut learning. Do not quote any working point until
this is adjudicated; the gate exists for exactly this moment. Refuted this round:
NaN compare keys join fine; `test_scores` fallback is test-only by construction.

**Test suite:** `python -m unittest discover -s pid/tests` → **258 tests, 0 failures** (re-verified 2026-09-20), network-free (13 modules incl. `test_significance.py` with brute-force agreement of the FOM scan — including per-row weights travelling with their score — and closed-form Clopper-Pearson checks, `test_performance.py` with the two independent optimum-finding paths cross-checked against each other, `test_cli.py` pinning the `pid all` → `train` flag forwarding, `test_importance.py` pinning the booster-artifact preference, and `test_features.py` pinning the dRICH presence flag); `ruff check pid --select F,E9` clean; the existing `python -m unittest discover -s tests` (41) still passes. The offset/relation tests pin the event-boundary arithmetic (a wrong base silently reads another event's hits) and the fixtures' hit sums reproduce stored cluster energies exactly (`max|ΣE_hit − E_cluster| = 0.0` over 12,184 clusters / 198,335 hits).

**Measured on the local reference files** (one file each; event-split CV, so *not* quotable performance — see the warning the pipeline itself prints):

| check | clean 26.02.0 | bkg 26.07.1 |
|---|---|---|
| feature rows = candidate tracks | 4,250 | 492 |
| link audit | 11 relations usable, 3 documented dead ends | 10 usable, measurement chain reported as campaign-missing |
| median E/p, backward e / π | **0.992 / 0.195** | **0.995 / 0.238** |
| median E/p, forward π / K / p | 0.274 / 0.241 / 0.262 | 0.34 / 0.44 / 0.19 |
| `eid` (LightGBM) gates | 4/4 PASS: test AUC 0.9926, permutation control 0.395±0.137 (z = 4.4), **E/p leads gain *and* SHAP** | refused: 7 pions < floor 30 (correct); with `--allow-small-sample` AUC 0.5 and the gates fail loudly |
| `ehad` | 4/4 PASS: test AUC 0.9889, all 5 control permutations exactly 0.5, E/p leads | — |
| `hadpid` | test AUC 0.567 vs control max 0.544 (**z = 1.6, FAIL**) and train−test gap 0.13 (**FAIL**) — with calorimeter-only inputs, hadron ID is **not** demonstrated on this sample | — |
| `pooled` (with `leg`) | test AUC 0.786, but `leg` **leads SHAP** → attribution gate FAIL: the hemisphere shortcut, measured rather than asserted | — |
| `compare` | runs once both tags are evaluated; refuses (naming the file) when one side is missing | — |

**Grid sample actually run (2026-09-19; the one-file smoke figures below are kept
as the historical record):** clean-150 features = 627,688 rows; bkg-200 features =
92,502 rows (~19.8k events). Headline held-out evaluations: eid clean 161,195-row
matrix (AUC 0.9978 on all three learners), bkg eid 14,221 rows (AUC 0.9943–0.9947);
every JSON's `meta` carries the sample size (files, events, rows,
signal/background counts) plus campaign, matching threshold and ΔR window, so a
number cannot travel without its provenance. Nothing is quotable until the
physics-gate adjudication (§12.1) is resolved.

Two bugs the harness found in itself while building, both fixed structurally rather
than by patching the symptom: an association-table leftover (`rec_idx`) entered the
feature matrix and out-ShAPed E/p → bookkeeping is now blocked by name *pattern*
with a regression test; and `pid/compare` read the JSON key `metadata` where
`trkperf.report` writes `meta`, which would have silently emitted empty
provenance → covered by `test_compare_report`.

**Two findings that changed the design while building (both now defaults, with reasons in `pid/config.py`):**
1. **Kinematics are not model inputs.** With `p`/`pt` in the matrix the trees ranked `pt` above E/p (gain `0:pt 1:e_over_p_backward`) because the scattered electron is simply the hardest track — a DIS-flux shortcut. Kinematics are now the *binning* of every performance table; `--include-kinematics` runs the tagging variant for comparison.
2. **Event-level and `track_time*` columns are opt-in, not baseline.** With them in, an occupancy column out-ShAPed E/p (`0:e_ecal_n_sum_evt`), and event activity differs systematically between the two campaigns — so a clean-vs-bkg comparison built on them would measure the overlay rather than PID degradation. `CentralCKFTracks.time` spans 2991–2999 ns (a per-event timestamp in these productions).

**Binning basis made explicit (§5d).** The scored tables previously carried only the
reconstructed kinematics, so every binned merit was silently reco-binned while the
tracking metrics are truth-binned. `train.py` now writes `truth_p/truth_pt/truth_eta`
plus `assoc_weight/is_matched/is_fake` into the score tables, and
`performance.BIN_VARIABLES`/`--bin-source reco|truth|both` lets any binned merit be
partitioned by either basis, with `binning_basis`/`bin_column` recorded in every table
and the basis in every axis label. (Also documented there: merits are computed on
held-out rows only, and the one-file run shows why — 10 pions.)

Two convention collisions, found while writing the definitions of §5c and fixed so
one column name never means two things: `*_err_lo`/`*_err_hi` was a *one-sided* 95 %
bound in `pid/performance` but a *central* 95 % interval in `pid/evaluate` (now
central everywhere, cross-checked against `scipy.stats.binomtest(...).
proportion_ci(method="exact")`), and `insufficient_stats` used `max(n_signal,
n_background)` in the performance layer versus `min(...)` in the tracking-style
layer (now `min(...)` in both, so a background-free bin is flagged rather than
reported as a NaN row).

A third, uglier catch: `rec_idx` — a column left behind by the association merge — entered the feature matrix and outranked E/p on SHAP. The physics gate found it; the fix is structural (bookkeeping is now blocked by *name pattern*, `_idx/_row/_begin/_end`, so the next merge column cannot leak either), with a regression test.

**Gate design note:** the label-shuffle control is a **permutation distribution**, not a single number. With 10 electrons in the held-out set an uninformative model's AUC fluctuates by ≈0.137 (measured over 5 balanced permutations), so any fixed tolerance about 0.5 is wrong in both directions; the gate requires z ≥ 3 **and** an absolute AUC margin ≥ 0.1.

## 12. Remaining work
1. **Grid runs (M8)** — **done** (2026-09-19): `pid/scripts/run_pid.sh clean filelists/clean_150.txt 0` and `... bkg_mixed filelists/bkg_200.txt 20`, plus backfill jobs for the learners the first pass missed (hadpid/pooled xgboost, eid/ehad sklearn_hgb on both tags). All 12 clean + 6 bkg models trained with evaluates and importance; `pid performance` manifests 104/104 (clean) + 26/26 (bkg); `pid compare` eid/ehad written; `check_learners.py` PASS on every task/tag (spreads ≤ 0.0063). Open: physics-gate adjudication (E/p outranked at scale) and any minQ2 100/1000 escalation for the bkg pion-side ceiling near 10⁻³ (§8).
1b. **Per-region PID plots (2026-09-23, code done, grid blocked by endpoint outage).**
    New `pid/regions.py`: `attach()` adds `eta_region` (truth eta, same
    `|eta| = 1` boundary as tracking), `n_layers_hit` and `in_acceptance`
    to any score frame via `trkperf.truth` reads (shared io cache, no
    second reader), applying each region's own rule (barrel `>= 4/7`,
    endcaps `>= 2`). `pid evaluate` / `pid performance --eta-region`
    additionally write per-region tables + figures (`_<slug>` tag) at the
    channel's global cut; region outputs never enter the manifest; training
    untouched (leg-scoped). Proven on the local smoke chain (203/214
    backward-endcap electrons in acceptance, empty barrel honestly
    skipped) + 10 new tests. Grid commands ready (lightgbm first):
    `python -m pid performance --dataset-tag <tag> --model lightgbm
    --channels eid,ehad,Kpi,pK --bin-source both --eta-region "barrel,forward
    endcap,backward endcap"` (+ `--max-file-failures 20 --cache-dir
    cache/bkg_files` for bkg_mixed) and per-task `pid evaluate --plot` with
    the same `--eta-region`. BLOCKED 2026-09-23 ~20:00 UTC: the JLab
    endpoint 3011s every `/volatile/.../26.02.0` open (paths read fine hours
    earlier; egress verified OK) and the bkg endpoint expires — relaunch
    when reads succeed again. Also fixed en route: `trkperf.truth` empty
    counts now carry int64 dtypes (object-dtype empties crashed the attach
    merge), and region attach warns loudly on a degraded rule.
2. **Hadron ID needs per-track Cherenkov/timing** (and is currently labelled
   `exploratory` in `config.CHANNELS` for exactly this reason — the calorimeter-only
   K/π and p/K results are baselines to be beaten, not deliverables). Three concrete routes, in cost order: (a) decode `DRICH*RawHits` cellIDs against the compact geometry to attach photons to the extrapolated `DRICH*Tracks` segment (AGENTS.md currently scopes cellID decoding out — a deliberate decision to revisit); (b) recover the `*_ParticleIDs` likelihoods by establishing the entry order of those collections (their `particle` relation is null, but the *multiplicity* pattern — 4 hypotheses/particle — may be alignable; time-box it); (c) request a production where the IRT `chargedParticle` relation is written (cid `1290518152` is not in the file).
3. **Muon class**: measured 4 muons in 4,242 matched tracks (0.09 %) in one clean file — decay-in-flight background, not a generator class here. Keeping it out of `CLASS_OF_ABS_PDG`; revisit with a dedicated sample if e/μ separation is ever wanted.
4. ~~Skills + AGENTS.md~~ **done**: `.opencode/skills/pid-ml/SKILL.md` (the runbook, including which links to re-audit on a new production), `.opencode/skills/pid-performance/SKILL.md` (evaluate / working points / compare mechanics + grid launcher), and a "PID data model (verified live)" section in AGENTS.md, so the next session does not re-derive §3.
5. **Optional**: gun-sample efficiency calibration (never fake rate), `E/p` region normalisation for the campaign-drift cross-check, matching-threshold sensitivity at 0.8.
6. **Deferred by design (audited, not forgotten):** space-angle ΔR (unweighted dφ overweights near-beamline winners — documented in `links.delta_r`; changing it moves every match); file-grouped calibration CV splits (row-wise inside train files today); early stopping (gap gate carries that load instead); SHAP sign check (rank-only today); geometric pT bin centres (arithmetic mids today); `evaluate binned_table by="p"` reusing pT edges; duplicate per-basis CAUTION lines.
