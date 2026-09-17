"""Central configuration for the PID pipeline.

Single source of truth for every tunable, exactly like :mod:`trkperf.config`:
change a threshold here, not at the call sites.

Physics justification for each number lives in the comments and in
``PLAN_pid.md``; the *data-model* facts it depends on (which branch is populated,
which relation is broken) live in :mod:`pid.schema`, which was verified live
against the two reference files rather than assumed.
"""

from __future__ import annotations

import numpy as np

from trkperf import config as tk_config

# ---------------------------------------------------------------------------
# Reuse the tracking project's conventions wholesale where they already exist,
# so a PID result and an efficiency result from the same run are comparable.
# ---------------------------------------------------------------------------
MATCH_WEIGHT_THRESHOLD: float = tk_config.MATCH_WEIGHT_THRESHOLD  # 0.5
MIN_ENTRIES_PER_BIN: int = tk_config.MIN_ENTRIES_PER_BIN  # 50
PT_BIN_EDGES: np.ndarray = tk_config.PT_BIN_EDGES
ETA_BIN_EDGES: np.ndarray = tk_config.ETA_BIN_EDGES
DATASETS: dict = tk_config.DATASETS
LOCAL_REFERENCE_FILES: dict = tk_config.LOCAL_REFERENCE_FILES

# Campaign (production) per dataset tag. The PID schema is campaign-dependent
# (26.02.0 and 26.07.1 differ by ~1650 branches), so every branch lookup is
# keyed on this, never on "the file we happen to be reading".
CAMPAIGN_BY_DATASET_TAG: dict[str, str] = {
    "clean": "26.02.0",
    "bkg_mixed": "26.07.1",
}

# ---------------------------------------------------------------------------
# Species grouping: trkperf tracks charge-conjugate species separately (pi+/pi-),
# which is right for tracking performance but wrong for PID, where the detector
# response depends on |PDG| and the sign only flips the tracking curvature.
# Classes are therefore defined on abs(PDG).
# ---------------------------------------------------------------------------
CLASS_OF_ABS_PDG: dict[int, str] = {
    11: "e",
    211: "pi",
    321: "K",
    2212: "p",
}
CLASS_LABELS: tuple[str, ...] = ("e", "pi", "K", "p")

# Species names (trkperf keys) belonging to each class - used only to select
# truth rows, never as a model input.
CLASS_TO_SPECIES: dict[str, tuple[str, ...]] = {
    "e": ("e-", "e+"),
    "pi": ("pi+", "pi-"),
    "K": ("K+", "K-"),
    "p": ("proton", "antiproton"),
}

# ---------------------------------------------------------------------------
# Kinematic legs.
#
# MEASURED on the clean reference file (300 events, truth-matched primaries with
# p > 1 GeV): truth electrons have mean eta = -2.50 and 93.6 % of them have an
# EcalEndcapN (backward EMCal) track-cluster match, but only 0.7 % an
# EcalEndcapP one; truth pi/K/p have mean eta = +2.2..+2.4 and 46-49 %
# EcalEndcapP matches. Electrons and hadrons therefore live in different
# detector legs of this sample, and a single pooled classifier would mostly
# learn "which hemisphere" instead of shower physics. Hence leg-scoped models,
# with a deliberately pooled arm kept as a cross-check.
# ---------------------------------------------------------------------------
BACKWARD_ETA_MAX: float = -1.5
FORWARD_ETA_MIN: float = 1.5

LEGS: dict[str, dict] = {
    "backward": {
        "eta_min": -4.0,
        "eta_max": BACKWARD_ETA_MAX,
        # EMCal used for E/p in this leg (single-plane: see pid.schema notes).
        "ecal": "EcalEndcapN",
        "hcal": "HcalEndcapN",
        "cherenkov": ("DRICHGas", "DRICHAerogel"),
    },
    "forward": {
        "eta_min": FORWARD_ETA_MIN,
        "eta_max": 4.0,
        "ecal": "EcalEndcapP",
        "hcal": "LFHCAL",
        "cherenkov": ("DRICHGas", "DRICHAerogel", "DIRC"),
    },
    "all": {"eta_min": -4.0, "eta_max": 4.0, "ecal": None, "hcal": None, "cherenkov": ()},
}

# ---------------------------------------------------------------------------
# Tasks. Each names a leg, the classes in play, and whether it is binary.
#   eid    -> Stage 1: electron vs pion (the physics goal: e/pi separation).
#   ehad   -> Stage 1 variant: electron vs all hadrons (pi+K+p combined).
#   hadpid -> Stage 2: pi vs K vs p, multiclass.
#   pooled -> cross-check: 4 classes over both legs, with leg as a feature.
# ---------------------------------------------------------------------------
TASKS: dict[str, dict] = {
    "eid": {"leg": "backward", "classes": ("e", "pi"), "binary": True,
            "label": "e vs pi (backward leg)", "status": "headline", "status_note": ""},
    "ehad": {"leg": "backward", "classes": ("e", "hadron"), "binary": True,
             "label": "e vs hadron (backward leg)", "status": "headline", "status_note": ""},
    # status: "headline" results are quotable once the statistics allow; "exploratory"
    # means the calorimeter alone cannot do the job and the number is a baseline to
    # beat once Cherenkov/dE/dx information becomes track-attachable (PLAN_pid.md 12.2).
    "hadpid": {"leg": "forward", "classes": ("pi", "K", "p"), "binary": False,
               "label": "pi/K/p (forward leg)", "status": "exploratory",
               "status_note": ("calorimeter-only baseline; full K/pi/p separation deferred "
                               "until Cherenkov hit-matching or a dE/dx link is resolved")},
    "pooled": {"leg": "all", "classes": ("e", "pi", "K", "p"), "binary": False,
               "label": "pooled 4-class cross-check", "status": "cross-check",
               "status_note": "learns the hemisphere; see the leg-SHAP gate"},
}

# "hadron" is not a truth species, it is the union used by the ehad task.
HADRON_CLASSES: tuple[str, ...] = ("pi", "K", "p")

# ---------------------------------------------------------------------------
# Track selection for the PID sample.
# ---------------------------------------------------------------------------
# PID needs a reconstructed momentum to form E/p; below ~1 GeV/c the tracker
# resolution and the calorimeter threshold make the ratio meaningless, and the
# Cherenkov detectors are below threshold for most species anyway.
MIN_TRACK_P: float = 1.0
# Reject pathological E/p values (mismatches, noise clusters) rather than let
# them define a classifier tail. 10 covers a fully-contained EM shower plus
# scale error; larger values are mismatches.
E_OVER_P_MAX: float = 10.0

# ---------------------------------------------------------------------------
# Track<->cluster matching.
#
# The built-in `*TrackClusterMatches.weight` is 0.0 for EVERY entry in BOTH
# campaigns (verified), so match quality cannot be taken from the file; we
# recompute Delta R between the track direction and the cluster centroid and
# keep it as a feature. The window is generous on purpose: the endcap EMCal
# cells are ~few cm at ~2-3.5 m, i.e. ~10 mrad, and the projection error for a
# few-GeV track is larger than the cell size. Tightened/loosened in M1 by
# looking at the Delta R distribution's true-match peak.
# ---------------------------------------------------------------------------
DELTA_R_MAX: float = 0.3
# Minimum hits for a cluster to get computed shower-shape variables (a 1- or
# 2-hit cluster has no meaningful width; those rows get NaN + has_* = 0).
MIN_HITS_FOR_SHAPE: int = 3

# ---------------------------------------------------------------------------
# Cut-based (no ML) baseline, i.e. the number every model must beat.
# E/p > 0.8 is the classic EM-shower requirement; MIP hadrons sit near 0.25-0.35
# (measured medians: pi 0.249, K 0.276, p 0.321 in the forward leg).
# ---------------------------------------------------------------------------
BASELINE_E_OVER_P: float = 0.8
BASELINE_E_HCAL_OVER_E_ECAL: float = 0.2  # hadronic leakage veto for electrons

# Refuse to train when a class has fewer rows than this: below it a boosted
# tree's "performance" is noise (measured: one +background file gives 94 electrons
# but only ~2 pions after selection, and every gate fails), and the honest advice
# is to add files / escalate the minQ2 tier rather than fit it.
MIN_ROWS_PER_CLASS: int = 30

# ---------------------------------------------------------------------------
# Performance channels for the maximum-significance working-point study.
#
# `pair` channels are built from a multiclass score table by reducing it to the
# two species in question, score = P(sig) / (P(sig) + P(bkg)) over truth in
# {sig, bkg}: that is the optimal 2-hypothesis likelihood-ratio discriminant under
# a flat prior, and it keeps a K-vs-pi curve honest instead of letting the
# surviving proton and electron classes quietly change the background definition.
# ---------------------------------------------------------------------------
CHANNELS: dict[str, dict] = {
    "eid":  {"task": "eid",    "signal": "e", "background": "pi", "pair": False,
             "title": "e vs #pi (backward leg)", "status": "headline", "status_note": ""},
    "ehad": {"task": "ehad",   "signal": "e", "background": "hadron", "pair": False,
             "title": "e vs hadrons (backward leg)", "status": "headline",
             "status_note": ""},
    # The two hadron channels are EXPLORATORY, not headline results: with the
    # dRICH IRT `chargedParticle` link dangling and no per-track dE/dx (PLAN_pid.md
    # 3.2), the only inputs able to separate K from pi are calorimetric, and a
    # calorimeter cannot tell two same-momentum hadrons of different mass apart
    # beyond ~1/sqrt(E) leakage differences. Measured on the local sample: K/pi AUC
    # 0.57 with a degenerate working point, and 72 % K recall in-sample against 27 %
    # held out. Promote these to headline only once Cherenkov hit-matching or a
    # dE/dx link is resolved (PLAN_pid.md 12.2).
    "Kpi":  {"task": "hadpid", "signal": "K", "background": "pi", "pair": True,
             "title": "K vs #pi (forward leg)", "status": "exploratory",
             "status_note": "calorimeter-only: awaiting Cherenkov hit-matching or dE/dx"},
    "pK":   {"task": "hadpid", "signal": "p", "background": "K", "pair": True,
             "title": "p vs K (forward leg)", "status": "exploratory",
             "status_note": "calorimeter-only: awaiting Cherenkov hit-matching or dE/dx"},
}

#: pT binning for the performance suite: 13 uniform bins over 0.5-10 GeV/c, as
#: requested. Deliberately separate from trkperf's log-spaced tracking bins - these
#: figures are PID plots and must not silently re-use the tracking binning.
PT_BINS_PERFORMANCE: np.ndarray = np.linspace(0.5, 10.0, 14)
#: Momentum binning, extended to 20 GeV/c for the forward detectors (dRICH/LFHCAL)
#: where the hadron-ID reach lies at high p.
P_BINS_PERFORMANCE: np.ndarray = np.array(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0])
#: eta bins for the 2D acceptance/fake maps (detector-transition region is where
#: the cracks show, so keep them finer than the tracking default near |eta| ~ 1-3).
ETA_BINS_PERFORMANCE: np.ndarray = np.arange(-4.0, 4.01, 0.5)

#: Threshold grid on which FOM(c) is scanned (scores are probabilities in [0, 1]).
N_THRESHOLD_SCAN: int = 401
#: Coarser grid for the per-bin curve family (nbins x nthresholds rows, and the
#: panels only need enough resolution to place c* unambiguously).
N_THRESHOLD_SCAN_BINNED: int = 201
#: A pT slice with fewer accepted candidates than this cannot define a working
#: point; its c* is NaN and it is drawn as a gap, never interpolated across.
MIN_CANDIDATES_PER_WP_BIN: int = 20

#: Deliverable figures every channel must produce (PLAN_pid.md 5b/5e). `base`
#: is the file stem used inside output/plots/.
REQUIRED_FIGURES: tuple[tuple[str, str], ...] = (
    ("roc", "ROC"),
    ("eff_vs_pt", "signal efficiency at c* vs pT"),
    ("misid_vs_pt", "mis-ID fake rate at c* vs pT"),
    ("purity_vs_pt", "purity at c* vs pT"),
    ("eff_map_pt_vs_eta", "2D efficiency map at c*"),
    ("fake_map_pt_vs_eta", "2D fake-rate map at c*"),
    ("rejection_vs_p", "rejection 1/fake at c* vs p"),
    ("nsigma_vs_p", "n_sigma vs p (per-bin, at the channel working point)"),
    ("score_dist_train_vs_test", "train vs test score distributions (overtraining)"),
    ("significance_vs_cut", "FOM(c) and the optimum c*"),
    ("significance_vs_cut_by_pt", "FOM(c) in every pT bin"),
    ("significance_vs_cut_panels", "per-pT-bin FOM(c) panels with each c*"),
    ("optimal_cut_vs_pt", "c*(pT)"),
)

#: Deliverables that describe the channel's score, not a binned quantity, so they
#: are written once per channel even when both binning bases are produced. A
#: manifest must count these as satisfied, not missing. ``confusion_matrix`` is
#: listed here because it too is written exactly once (task-level, under the
#: reco basis, ``pid-<task>_confusion_matrix.png``) - but it is deliberately NOT
#: in REQUIRED_FIGURES: it describes the multi-class TASK (pi/K/p) and is
#: undefined for the binary channels, so a per-channel manifest can neither
#: require nor sensibly miss it.
BASIS_INDEPENDENT_FIGURES: tuple[str, ...] = (
    "significance_vs_cut", "roc", "score_dist_train_vs_test", "confusion_matrix",
)

#: Figures of merit. `s_over_sqrt` is the requested S/sqrt(S+B); the effective-count
#: variant (Kish) is reported next to it because with class weights the Poisson
#: variance of the *weighted* sums is sum(w^2), not sum(w), and quoting the naive
#: form with weights would overstate significance.
SIGNIFICANCE_MODES: tuple[str, ...] = ("s_over_sqrt", "s_over_sqrt_eff")
#: Default luminosity/flux scaling: 1.0 = the composition actually measured in the
#: test sample. Any other value must be stated with the resulting c* (FOM scales as
#: sqrt(L) at fixed composition, so the optimum cut itself shifts with exposure).
DEFAULT_LUMI_SCALE: float = 1.0
#: Confidence level for the rejection *lower limits* (a fake rate of 0/N is a
#: measurement of nothing; the limit is the honest statement).
REJECTION_CL: float = 0.95

# Working points at which electron efficiency is quoted.
FAKE_RATE_TARGETS: tuple[float, ...] = (1e-2, 1e-3, 1e-4)

# ---------------------------------------------------------------------------
# n-sigma separation.
#
# The naive approach - fit a single Gaussian to each class's raw BDT score and
# take |mu_1 - mu_2| / sigma_pooled - is INVALID here: boosted scores are
# probabilities confined to [0, 1], so a good classifier piles both classes
# against the bounds, the Gaussian's tail is truncated by the axis rather than
# by the distribution, and the fit either fails to converge (the local K/pi run
# returned n_sigma = 0.0065 with both peaks at ~1.0) or returns a width that is a
# property of the clipping. Two defensible estimators are implemented instead:
#
#   "quantile" (default)  n_sigma = 2 * erfinv(2 * AUC - 1)
#       Exact for two equal-variance Gaussians, because
#       AUC = Phi(d / sqrt(2)) with d = |mu_1 - mu_2| / sigma. Needs no fit at
#       all, so it cannot diverge; valid for any monotone rescaling of the score.
#       Verified against the pooled-sigma estimate to <0.1% for d = 0.5-5.
#   "logit"               the same pooled-sigma formula applied to
#                         z = ln(P / (1 - P)), which maps [0,1] onto the whole
#                         real line so the tails are no longer truncated.
#   "raw"                 the original fit, kept only for reproducing earlier
#                       numbers; it emits a warning because its result is
#                       dominated by the [0,1] boundaries.
# ---------------------------------------------------------------------------
NSIGMA_DEFINITION: str = ("quantile: 2*erfinv(2*AUC-1) [default, exact for equal-width "
                          "Gaussians]; logit: |mu_1-mu_2|/sqrt((sigma_1^2+sigma_2^2)/2) "
                          "on z=ln(P/(1-P)); raw: same on the bounded score (deprecated)")
NSIGMA_METHOD: str = "quantile"
NSIGMA_METHODS: tuple[str, ...] = ("quantile", "logit", "raw")
#: Probabilities are clipped into [eps, 1-eps] before the logit so a score that
#: hit the numerical bound contributes a large-but-finite z instead of +-inf.
LOGIT_EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# Model / training defaults.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 1234
# Feature families excluded from the model by default (see pid.schema notes and
# PLAN_pid.md 9): truth-derived (leakage) or producer-PID (label-adjacent).
LEAKAGE_BLOCKED_FAMILIES: tuple[str, ...] = ("truth", "producer_pid", "truth_dedx")
# Columns excluded from the feature set by default even though they are measured
# quantities.
#
# `charge`: in these NC DIS (10x100, e- beam) samples the scattered electron is
# always negative, so the sign of the track charge correlates with the truth label
# better than any calorimeter variable does (measured: -0.74). That is a property of
# the *beam configuration*, not of the detector response - a model leaning on it
# would be worthless under a positron beam. --use-charge re-enables it.
EXCLUDE_COLUMNS_DEFAULT: tuple[str, ...] = ("charge",)

# Event-level / non-per-track columns excluded from the headline model by default.
#
# Two independent reasons, both measured on the reference files:
#  * they are not per-track detector response, and the physics gate exists to
#    prove the calorimeter/Cherenkov information is doing the identifying - with
#    them in the matrix an event-occupancy column out-ShAPs E/p (measured SHAP
#    ranking: 0:e_ecal_n_sum_evt 1:track_time_error 2:e_over_p_backward);
#  * they differ systematically between the two campaigns by construction (a
#    +background event has far more activity), so a clean-vs-background comparison
#    built on them would measure the overlay, not the degradation of PID.
# Both are still BUILT into the feature table and available as ablations
# (`--include-event-level`, `--include-track-time`), which is the interesting
# study, not the baseline.
EVENT_LEVEL_SUFFIX = "_evt"
#: `CentralCKFTracks.time` spans only 2991.5-2999.3 ns with a ~3 us per-event
#: offset, i.e. it is an event timestamp in these productions, not a per-track
#: measurement - see pid.schema's track_time note.
TRACK_TIME_COLUMNS: tuple[str, ...] = ("track_time", "track_time_error")

# Track kinematics. Measured, physical - but in this sample they are also a flux
# shortcut: the scattered electron is the hardest track in the backward leg, so
# with p/pt/eta among the inputs the boosted trees rank pt FIRST and E/p second
# (measured: gain 0:pt 1:e_over_p_backward). That model would report excellent
# "PID" while mostly measuring the DIS kinematics.
#
# The standard remedy is what this project does: PID performance is *binned* in
# (p, pT, eta) - see pid.evaluate.binned_table - and the classifier is trained on
# detector response only, so a quoted efficiency at a fixed fake rate is a property
# of the calorimeter/Cherenkov information at a given momentum, not of the flux.
# --include-kinematics trains the "full tagging" variant for comparison.
KINEMATIC_COLUMNS: tuple[str, ...] = ("p", "pt", "eta", "phi", "px", "py", "pz")

# Thread count for the learners AND the hyper-parameter search.
#
# MEASURED on this host: LightGBM and XGBoost both *deadlock* (no progress, no
# error) at n_jobs=4 with these small tables, while n_jobs<=2 finish in seconds.
# The container reports 32 CPUs, so the limit is an OpenMP/thread-pool artefact,
# not a real core count. Single-threaded is also what makes a run reproducible;
# boost each only with --n-jobs after checking it completes on your sample.
N_JOBS: int = 1

MODEL_LIBRARY_DEFAULT: str = "lightgbm"
MODEL_LIBRARIES: tuple[str, ...] = ("lightgbm", "xgboost", "sklearn_hgb")

# Search spaces. Deliberately small: the bottleneck here is the *sample*, not
# the optimiser, and every trial re-reads a feature table from disk.
SEARCH_SPACES: dict[str, dict] = {
    "lightgbm": {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
        "feature_fraction": [0.7, 0.9, 1.0],
        "bagging_fraction": [0.7, 0.9, 1.0],
        "n_estimators": [200, 400, 800],
        "lambda_l2": [0.0, 1.0, 10.0],
    },
    "xgboost": {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [5, 20, 50],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "n_estimators": [200, 400, 800],
        "reg_lambda": [0.0, 1.0, 10.0],
    },
    "sklearn_hgb": {
        "max_depth": [None, 6, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_samples_leaf": [20, 50, 100],
        "max_leaf_nodes": [15, 31, 63],
        "l2_regularization": [0.0, 1.0, 10.0],
        "max_iter": [200, 400, 800],
    },
}
N_CV_SPLITS: int = 3
N_SEARCH_ITERATIONS: int = 12
TEST_SIZE: float = 0.25  # fraction of FILES held out (never events)

# Gates (PLAN_pid.md 7) - a violation is a hard failure, not a warning.
MAX_TRAIN_TEST_AUC_GAP: float = 0.02
# The balanced label-shuffle control is a small model on a small sample, so its
# AUC fluctuates by a few percent around chance; the real model must beat the
# control by at least this much for the result to count as a usable
# discriminator (z-score leg is MIN_CONTROL_Z_SCORE below).
MIN_SIGNAL_OVER_CONTROL_AUC: float = 0.10
# Number of balanced label permutations in the control, and the z-score the real
# held-out AUC must reach above that distribution (see pid.train.
# label_shuffle_control for why a fixed tolerance around 0.5 is the wrong test at
# these sample sizes).
N_CONTROL_PERMUTATIONS: int = 5
MIN_CONTROL_Z_SCORE: float = 3.0
LEAKAGE_MAX_LABEL_CORRELATION: float = 0.99
CROSS_LEARNER_MAX_AUC_SPREAD: float = 0.02

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "output"
MODEL_DIR: str = "output/models"
PLOT_DIR: str = "output/plots"
FEATURE_CACHE_DIR: str = "cache/pid_features"
# Calibration: BDT/boosted outputs are not probabilities; a post-hoc calibration
# is required before "efficiency at 1e-4 fake rate" means the same thing in the
# clean and the +background sample.
CALIBRATION_METHOD: str = "sigmoid"  # or "isotonic"
CALIBRATION_MAX_TRAIN: int = 200_000
