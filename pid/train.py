"""Train (and cross-validate) a PID classifier, enforcing the plan's gates.

Flow
----
1. load or build the feature table, :func:`pid.dataset.prepare` the task,
   :func:`pid.dataset.design_matrix` the inputs (NaN preserved, names kept),
2. :func:`pid.dataset.assert_no_leakage` *before* anything is fitted,
3. split by ``file_id``, then hyper-parameter search with
   ``StratifiedGroupKFold(groups=file_id)`` so no file ever appears in both a
   training fold and its own validation fold,
4. refit on train, score the held-out files,
5. calibration (boosted scores are not probabilities, and an "efficiency at
   1e-4 fake rate" working point is meaningless until the score scale means the
   same thing in the clean and the +background sample),
6. gates, then artifacts.

The gates are the reason this module exists as more than a 20-line fit:
over-fitting, a physics-inverted ranking, or a control sample that still scores
are exactly the ways a PID result becomes wrong while looking excellent.
``--relax-gates`` downgrades them to warnings so a human can inspect a surprising
result instead of being blocked, but the default is to stop.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import config, dataset, models, schema


#: Features that *must* dominate a physics-sensible PID model. Checked against the
#: leading gain feature (top-1); a failure means the classifier found a shortcut
#: (usually occupancy, an event-level variable, or a leaking column).
EXPECTED_TOP_FEATURES: dict[str, tuple[str, ...]] = {
    "eid": ("e_over_p", "log_e_over_p", "ecal_"),
    "ehad": ("e_over_p", "log_e_over_p", "ecal_"),
    "hadpid": ("shape_", "rms_", "irt_", "drich", "e_over_p", "leakage", "ecal_"),
    "pooled": ("e_over_p", "log_e_over_p", "ecal_", "leg"),
}


#: Feature that must lead the SHAP attribution for the electron tasks. `None` for
#: hadron ID, where no single observable is expected to dominate (separation comes
#: from Cherenkov + timing + shower shape together, and in this campaign the first
#: two are not track-attachable - see PLAN_pid.md 3.2).
PRIMARY_ATTRIBUTION: dict[str, str | None] = {
    "eid": "e_over_p", "ehad": "e_over_p", "pooled": "e_over_p", "hadpid": None,
}


def _attribution_matches(top_feature: str, primary: str) -> bool:
    """Does the leading SHAP feature identify the required observable?

    Compares on the observable name after stripping the leg suffix
    (``_backward``/``_forward``) and an optional ``log_`` prefix, so
    ``e_over_p_backward`` and ``log_e_over_p_forward`` both count as E/p while
    an unrelated column merely containing the substring does not.
    """
    name = str(top_feature)
    for leg in ("_backward", "_forward"):
        if name.endswith(leg):
            name = name[: -len(leg)]
            break
    if name.startswith("log_"):
        name = name[len("log_"):]
    return name == primary


def _require(feature: str, patterns: tuple[str, ...]) -> bool:
    return any(p in feature for p in patterns)


#: Columns every score table carries. The first block identifies the row (one
#: reconstructed ``CentralCKFTracks`` candidate, addressed by file_id/event/
#: track_idx), the next gives the RECONSTRUCTED kinematics a real analysis would
#: bin in, then the TRUTH partner's kinematics and the bookkeeping that shows how
#: the label was obtained (assoc_weight >= config.MATCH_WEIGHT_THRESHOLD, the
#: best association per reconstructed track, from CentralCKFTrackAssociations).
SCORED_COLUMNS = ("file_id", "event", "track_idx", "label", "truth_class", "leg",
                  "p", "pt", "eta", "truth_p", "truth_pt", "truth_eta",
                  "assoc_weight", "is_matched", "is_fake")


def missing_e_over_p_columns(prep: pd.DataFrame, task: str) -> list[str]:
    """E/p columns an electron task requires to be present and measurable.

    The SHAP gate requires E/p to LEAD, which is vacuous if E/p was never
    computed (e.g. a production without ECAL clusters leaves it all-NaN and
    the model quietly promotes a proxy). Returns the missing/unusable names
    (empty = usable E/p available); the caller refuses to train on a
    non-empty list.
    """
    if task not in ("eid", "ehad", "pooled"):
        return []
    legs = ("backward", "forward") if task == "pooled" else (config.TASKS[task]["leg"],)
    return [f"e_over_p_{leg}" for leg in legs
            if f"e_over_p_{leg}" not in prep.columns or not bool(
                pd.to_numeric(prep[f"e_over_p_{leg}"],
                              errors="coerce").notna().any())]


def load_or_build(*, features: str | None, files: list[str] | None, dataset_tag: str,
                  legs: tuple[str, ...], limit_files: int | None, max_failures: int,
                  cache_dir: str | None, enable_ionisation: bool):
    """``(feature_table, path_used)`` - read a cached table or build it from files."""
    if features:
        df = dataset.load_table(features)
        print(f"[train] feature table {features}: {len(df)} rows, {len(df.columns)} cols",
              file=sys.stderr)
        return df, features
    if not files:
        raise SystemExit("train needs --features <pickle> or a file list (--file/--file-list)")
    return dataset.load_table(_build_and_save(
        files, dataset_tag=dataset_tag, legs=legs, limit_files=limit_files,
        max_failures=max_failures, cache_dir=cache_dir, enable_ionisation=enable_ionisation))


def _build_and_save(files, *, dataset_tag, legs, limit_files, max_failures, cache_dir,
                    enable_ionisation) -> str:
    from . import features as pid_features

    df = pid_features.build_features(
        files, dataset_tag=dataset_tag, legs=legs, limit_files=limit_files,
        max_failures=max_failures, cache_dir=cache_dir or config.FEATURE_CACHE_DIR,
        enable_ionisation=enable_ionisation)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, f"pid-features_{dataset_tag}.pkl")
    dataset.save_table(df, out)
    skipped = df.attrs.get("skipped_files", [])
    print(f"[features] wrote {out}: {len(df)} rows from "
          f"{df.attrs.get('n_files', len(files))} files ({len(skipped)} skipped)",
          file=sys.stderr)
    if skipped:
        print(f"[features] skipped files: {skipped[:5]}"
              f"{' ...' if len(skipped) > 5 else ''}", file=sys.stderr)
    return out


def cross_validate(estimator_factory, X, y, groups, *, n_splits, n_iter, space, seed,
                   scoring: str, n_jobs: int = config.N_JOBS, sample_weight=None):
    """Randomised search over `space`, folds grouped by ``file_id``.

    Grid entries are discrete lists (from :data:`pid.config.SEARCH_SPACES`): the
    samples here are statistics-limited, not optimiser-limited, so a coarse
    explicit grid is more honest than a continuous distribution, and it keeps
    ``best_params_`` exactly reproducible. ``optuna`` is not installed in this
    environment, hence sklearn's search.
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

    n_groups = int(pd.Series(groups).nunique())
    if n_groups >= 2:
        cv = StratifiedGroupKFold(n_splits=max(2, min(n_splits, n_groups)), shuffle=True,
                                  random_state=seed)
        kind = "grouped_by_file"
    else:
        # A one-file table cannot be split by file. Fall back to event-level folds
        # and SAY SO, so a smoke-test score is never quoted as a leakage-safe one.
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
        kind = "single_file_event_split_NOT_leak_safe"
        print("[train] WARNING: only one input file - CV folds split events, not "
              "files. Fine for a smoke test, NOT a quotable performance number.",
              file=sys.stderr)
    search = RandomizedSearchCV(
        estimator_factory(), param_distributions=dict(space), n_iter=n_iter,
        scoring=scoring, cv=cv, random_state=seed, n_jobs=n_jobs, refit=True,
        error_score="raise")
    # Groups MUST ride along: without them a grouped splitter sees groups=None
    # (one pseudo-group) and every multi-file run dies in _iter_test_indices,
    # while single-file smoke silently passes via the StratifiedKFold branch.
    search.fit(X, y, groups=groups, sample_weight=sample_weight)
    search.cv_kind_ = kind
    return search


def _weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    """Per-row class weights (``balanced`` -> w_c = n / (K * n_c)).

    Reweighting rather than subsampling keeps every event: in the backward leg the
    sample is 94 % electrons, and subsampling to balance it would throw away the
    statistics the fake-rate measurement depends on, while unweighted training
    lets the majority peak dictate where the splits go.
    """
    if mode == "none":
        return None
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    per_class = dict(zip(classes.tolist(), counts.tolist()))
    k = len(classes)
    n = len(y)
    return np.array([n / (k * per_class[int(v)]) for v in y], dtype=float)


def label_shuffle_control(adapter, params, X, y, idx_train, idx_test, y_test, *,
                          seed: int, n_classes: int, n_permutations: int = 5) -> dict:
    """Permutation test: refit on balanced, permuted labels, several times.

    Two design points, both forced by the data rather than by taste.

    *Balanced first, then permuted.* Permuting the labels of a 94 %-electron
    sample leaves ~89 % of them unchanged, so a naive "control" is mostly a model
    trained on the real labels and scores ~0.73 - a false alarm that looks like
    leakage. Balancing the control draw makes agreement 50 % by construction.

    *A distribution, not a number.* With 10 electrons in the held-out set the AUC
    of a genuinely uninformative model fluctuates by ~0.09 (sigma ~ sqrt((n+1) /
    (12 n_sig n_bkg))), so any fixed tolerance around 0.5 is wrong in both
    directions. The gate therefore compares the real AUC against the spread of the
    permutation distribution and reports the resulting z-score.

    Returns ``{"auc_mean", "auc_std", "auc_max", "n_control", "z_vs_control",
    "note"}``.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed + 1)
    y_tr = np.asarray(y)[np.asarray(idx_train)]
    y_te = np.asarray(y_test)
    classes, counts = np.unique(y_tr, return_counts=True)
    n_each = int(counts.min())
    out = {"auc_mean": float("nan"), "auc_std": float("nan"), "auc_max": float("nan"),
           "n_control": 0, "n_per_class": n_each, "z_vs_control": float("nan"),
           "note": ""}
    if n_each < 5 or classes.size < 2:
        out["note"] = "a class is too small to balance; control could not be run"
        return out
    if int(np.unique(y_te).size) < 2:
        out["note"] = "held-out set has one class; control AUC undefined"
        return out

    pool = np.concatenate([rng.choice(np.where(y_tr == c)[0], size=n_each, replace=False)
                           for c in classes])
    scores = []
    for _ in range(int(n_permutations)):
        yc = rng.permutation(y_tr[pool])
        est = adapter.estimator(params, n_classes=n_classes, seed=seed)
        est.fit(X.iloc[np.asarray(idx_train)[pool]], yc)
        p = est.predict_proba(X.iloc[list(idx_test)])
        scores.append(float(roc_auc_score(y_te, p[:, 1]) if n_classes <= 2
                            else roc_auc_score(y_te, p, multi_class="ovr")))
    arr = np.asarray(scores, dtype=float)
    out.update({"auc_mean": float(arr.mean()), "auc_std": float(arr.std(ddof=1))
                if arr.size > 1 else 0.0, "auc_max": float(arr.max()),
                "n_control": int(arr.size), "aucs": [round(float(x), 4) for x in arr]})
    return out


def train(task: str, *, model: str = config.MODEL_LIBRARY_DEFAULT,
          features: str | None = None, files: list[str] | None = None,
          dataset_tag: str = "clean", legs: tuple[str, ...] = ("backward", "forward"),
          limit_files: int | None = None, max_failures: int = 0,
          cache_dir: str | None = config.FEATURE_CACHE_DIR,
          n_iter: int = config.N_SEARCH_ITERATIONS, n_splits: int = config.N_CV_SPLITS,
          seed: int = config.RANDOM_SEED, out_dir: str = config.MODEL_DIR,
          include_event_level: bool = False, enable_ionisation: bool = False,
          calibrate: bool = True, gates: bool = True, quiet: bool = False,
           n_jobs: int = config.N_JOBS, class_weight: str = "balanced",
           use_charge: bool = False, allow_small_sample: bool = False,
           include_kinematics: bool = False, include_track_time: bool = False,
           min_q2_tier: str | None = None) -> dict:
    """Train one task/model/dataset combination and write its artifacts.

    Returns the report dict (also written to ``out_dir/<name>/report.json``).
    """
    t0 = time.time()
    env = schema.assert_ml_env(strict=True)
    adapter = models.get_adapter(model)
    os.makedirs(out_dir, exist_ok=True)

    df, used_table = load_or_build(features=features, files=files, dataset_tag=dataset_tag,
                                   legs=legs, limit_files=limit_files,
                                   max_failures=max_failures, cache_dir=cache_dir,
                                   enable_ionisation=enable_ionisation)
    prep = dataset.prepare(df, task, require_matched=True)
    exclude = () if use_charge else config.EXCLUDE_COLUMNS_DEFAULT
    columns = dataset.model_columns(prep, task=task, include_event_level=include_event_level,
                                    include_ionisation=enable_ionisation, exclude=exclude,
                                    include_kinematics=include_kinematics,
                                    include_track_time=include_track_time)
    if not columns:
        raise SystemExit(f"train: no usable feature columns for task {task!r}")
    if task in ("eid", "ehad", "pooled"):
        missing = missing_e_over_p_columns(prep, task)
        if missing:
            raise SystemExit(
                f"pid.train: electron task {task!r} has no usable E/p column(s) "
                f"{missing} in this sample - train without an E/p check would "
                "certify a proxy. Fix the ECAL cluster join first.")
    X = dataset.design_matrix(prep, columns)
    y = prep["label"].to_numpy(dtype=int)
    groups = prep["file_id"].to_numpy()
    counts = dict(zip(*[list(x) for x in np.unique(y, return_counts=True)]))
    n_classes = len(counts)
    if n_classes < 2:
        raise SystemExit(f"train: task {task!r} has a single class in this sample "
                         f"({counts}); widen the file list before training.")
    rare = min(counts.values())
    if rare < config.MIN_ROWS_PER_CLASS and not allow_small_sample:
        raise SystemExit(
            f"pid.train: refusing to train {task!r} on this sample - the rarest class "
            f"has {rare} rows (floor {config.MIN_ROWS_PER_CLASS}). A boosted tree fit on "
            "that is noise, and the gates below cannot certify it. Escalate per "
            "PLAN_pid.md 8 / AGENTS.md: add files at the current minQ2 tier, then widen "
            "the +background sample to the next tier (1 -> 10 -> 100 -> 1000). "
            "Re-run with --allow-small-sample to fit it anyway (smoke tests only).")

    leak = dataset.assert_no_leakage(X, y)

    tr_idx, te_idx, test_groups = dataset.split_by_file(prep, seed=seed)
    scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
    factory = lambda: adapter.estimator({}, n_classes=n_classes, seed=seed)  # noqa: E731
    w_tr = _weights(y[tr_idx], class_weight)
    search = cross_validate(factory, X.iloc[tr_idx], y[tr_idx], groups[tr_idx],
                            n_splits=n_splits, n_iter=n_iter,
                            space={k: v for k, v in config.SEARCH_SPACES[model].items()},
                            seed=seed, scoring=scoring, n_jobs=n_jobs,
                            sample_weight=w_tr)
    # `best` is refit on the whole training set by RandomizedSearchCV (refit=True).
    # Importance and SHAP are taken from THIS model; the calibrated copy below is
    # used only for scores, because CalibratedClassifierCV wraps its own refits and
    # exposes no single booster to read gains from.
    best = search.best_estimator_

    # Calibration split: carved out of *training* files, so the held-out files
    # stay untouched by any part of the score transformation. The 3-fold CV
    # inside the calibrator splits ROWS, not file groups (sklearn has no
    # grouped calibrator): file-grouping is enforced at the outer split and
    # the search CV, while calibration only reshapes scores monotonically.
    min_class_in_cal = int(np.min(np.unique(y[tr_idx], return_counts=True)[1]) // 2)
    if calibrate and min_class_in_cal < 4 * 3:
        # A 3-fold CalibratedClassifierCV needs >= 3 examples of each class per
        # fold; on a small sample (or a rare class) it is better to skip
        # calibration and say so than to crash or silently mis-calibrate.
        print(f"[train] NOTE: skipping calibration - only ~{min_class_in_cal} rows per "
              "class available in a calibration fold. Scores are then raw booster "
              "outputs: fine for AUC/ROC/n-sigma, NOT for a quoted purity.",
              file=sys.stderr)
        calibrate = False
        calibrate_skip_reason = f"too few rows per class ({min_class_in_cal}) for 3-fold calibration"
    else:
        calibrate_skip_reason = None
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        # Calibrating a *fresh* estimator on a data subset, with the same weights
        # the main model saw: the base model itself is never refit here, so the
        # train AUC above still describes the model that produced the gains.
        sub = np.arange(len(tr_idx))
        rng = np.random.default_rng(seed)
        rng.shuffle(sub)
        half = sub[: max(50, len(sub) // 2)]
        sub2 = sub
        full_idx = tr_idx[half]
        cal = CalibratedClassifierCV(
            adapter.estimator(search.best_params_, n_classes=n_classes, seed=seed),
            method=config.CALIBRATION_METHOD, cv=3)
        cal.fit(X.iloc[full_idx], y[full_idx],
                sample_weight=(None if w_tr is None
                               else np.asarray(w_tr)[sub2[: max(50, len(sub) // 2)]]))
        final = cal
    else:
        final = best

    def _proba(idx):
        # `final` is used whole: sklearn names the fitted list
        # `calibrated_classifiers_` (plural), so there is no singular
        # attribute to unwrap, and unwrapping one fold's refit would not be
        # the calibrated model anyway.
        pos = 1 if n_classes <= 2 else None
        p = final.predict_proba(X.iloc[idx])
        return p[:, 1] if pos == 1 else p

    score_tr, score_te = _proba(tr_idx), _proba(te_idx)
    # Invert THIS task's label map: label order is the task's class list
    # (hadpid -> pi,K,p), NOT config.CLASS_LABELS positions, otherwise the saved
    # proba_<class> columns are labelled with the wrong species.
    lmap = prep.attrs["labels"]
    class_order = [cls for cls, _lab in sorted(lmap.items(), key=lambda kv: kv[1])]

    from sklearn.metrics import roc_auc_score

    def _auc(score, target):
        if n_classes > 2:
            return float(roc_auc_score(target, score, multi_class="ovr"))
        return float(roc_auc_score(target, score))

    auc_tr, auc_te = _auc(score_tr, y[tr_idx]), _auc(score_te, y[te_idx])

    # Control: shuffle the training labels, refit with the best parameters, and
    # require the held-out AUC to collapse to chance. A control that still scores
    # means the split leaks, not that the model is good.
    control = label_shuffle_control(adapter, search.best_params_, X, y, tr_idx, te_idx,
                                    y[te_idx], seed=seed, n_classes=n_classes,
                                    n_permutations=config.N_CONTROL_PERMUTATIONS)
    ctrl_auc = control["auc_mean"]
    ctrl_spread = control["auc_std"] if np.isfinite(control["auc_std"]) else 0.0
    # z-score of the real model against the permutation distribution.
    z = ((auc_te - ctrl_auc) / ctrl_spread) if ctrl_spread > 0 else float("inf")
    control["z_vs_control"] = float(z) if np.isfinite(z) else float("nan")

    gain = adapter.gain(best, columns)
    top = [str(c) for c in gain.head(3).index]
    # Gain counts how often a variable is *used*; SHAP measures how much it
    # *decides*. The plan's physics requirement ("E/p dominates electron
    # classification") is an attribution statement, so the leading SHAP feature is
    # the one that must be E/p for the electron tasks. Skimming a correlated proxy
    # (cluster energy, hit multiplicity) is legitimate but is not what we claim.
    shap_top: list[str] = []
    try:
        # Attribution is measured on HELD-OUT rows only: the gate asks what
        # drives decisions where it matters, and train rows would flatter
        # memorised splits.
        gate_rows = X.iloc[te_idx].head(min(2000, len(te_idx)))
        values = adapter.shap(best, gate_rows)
        mean_abs = np.nanmean(np.abs(values), axis=0)
        order = np.argsort(-np.nan_to_num(mean_abs))
        shap_top = [columns[i] for i in order[:3]]
    except NotImplementedError:
        shap_top = []  # sklearn HGB has no exact SHAP; gain ranking stands alone
    primary = PRIMARY_ATTRIBUTION.get(task)
    attribution_ok = (not primary) or (bool(shap_top) and _attribution_matches(shap_top[0], primary))
    physics_ok = (all(_require(c, EXPECTED_TOP_FEATURES.get(task, ())) for c in top[:1])
                  and attribution_ok)

    gate_results = {
        "train_test_auc_gap": {"value": auc_tr - auc_te, "limit": config.MAX_TRAIN_TEST_AUC_GAP,
                               "passed": bool((auc_tr - auc_te) <= config.MAX_TRAIN_TEST_AUC_GAP)},
        "label_shuffle_control": {
            "value": {"control_auc_mean": ctrl_auc, "control_auc_std": control["auc_std"],
                      "control_auc_max": control["auc_max"], "model_auc_test": auc_te,
                      "z": control["z_vs_control"]},
            "limit": {"min_z": config.MIN_CONTROL_Z_SCORE,
                      "min_auc_margin": config.MIN_SIGNAL_OVER_CONTROL_AUC},
            "evaluated": bool(control.get("n_control", 0) > 0),
            # The model must beat the permutation distribution by a margin that is
            # large both in sigma and in absolute AUC (a huge z with a 0.001 margin
            # is not a usable discriminator).
            "passed": bool(control.get("n_control", 0) > 0
                           and np.isfinite(ctrl_auc)
                           and (not np.isfinite(control["z_vs_control"])
                                or control["z_vs_control"] >= config.MIN_CONTROL_Z_SCORE)
                           and (auc_te - ctrl_auc) >= config.MIN_SIGNAL_OVER_CONTROL_AUC),
            "detail": control.get("note") or (
                f"{control['n_control']} balanced permutations, "
                f"{control.get('n_per_class', '?')}/class; AUCs {control.get('aucs')}"
                if control.get("n_control") else "not evaluated")},
        "top_feature_is_physical": {
            "value": {"gain_top3": top, "shap_top3": shap_top},
            "limit": {"gain_patterns": list(EXPECTED_TOP_FEATURES.get(task, ())),
                      "shap_must_lead": primary},
            "expected": list(EXPECTED_TOP_FEATURES.get(task, ())),
            "passed": bool(physics_ok),
            # A hadron-ID failure here is informative, not a bug: per-track
            # Cherenkov/timing are not linkable in these productions, so no single
            # observable is expected to dominate (PLAN_pid.md 12.2).
            "advisory": task == "hadpid"},
        "no_single_feature_label_correlation": {
            "value": float(leak.iloc[0]["corr_with_label"]),
            "limit": config.LEAKAGE_MAX_LABEL_CORRELATION,
            "passed": bool(abs(leak.iloc[0]["corr_with_label"])
                           <= config.LEAKAGE_MAX_LABEL_CORRELATION)},
    }
    # An advisory gate is reported loudly but does not block: hadron ID cannot
    # satisfy a "one detector response dominates" requirement in productions
    # where the per-track Cherenkov/timing links are dead (PLAN_pid.md 12.2), and
    # blocking there would hide the statistics gates that *are* meaningful.
    all_passed = all(v["passed"] for v in gate_results.values() if not v.get("advisory"))
    if not gate_results["label_shuffle_control"]["evaluated"]:
        print("[gate] WARNING: the permutation control could not be run ("
              f"{control.get('note', 'unknown reason')}); the result cannot be "
              "certified as leakage-free.", file=sys.stderr)

    stem = os.path.join(out_dir, f"{task}_{model}_{dataset_tag}")
    os.makedirs(stem, exist_ok=True)
    import joblib

    joblib.dump(final, os.path.join(stem, "model.joblib"))
    # Importance/SHAP must be read from the unwrapped booster: the calibrated
    # wrapper above refits on subsets and exposes no single booster, so
    # pid.importance cannot use model.joblib once calibration has run.
    joblib.dump(best, os.path.join(stem, "booster.joblib"))
    # Reconstructed kinematics (p, pt, eta) AND the truth partner's (truth_pt,
    # truth_eta) are both carried, because the two answer different questions and a
    # score table that cannot do the truth-binned version forces a retrain to find
    # out. e_over_p_* rides along so a suspicious working point can be traced back
    # to the detector quantity that produced it without reloading the features.
    scored = prep.iloc[te_idx][[c for c in SCORED_COLUMNS
                                if c in prep.columns] +
                               [c for c in ("e_over_p_backward", "e_over_p_forward")
                                if c in prep.columns]].copy()
    # Flat numeric columns only: a list-of-probabilities column cannot go into a
    # TNtuple/CSV, and evaluate() needs to slice by class anyway.
    if n_classes <= 2:
        scored["score"] = np.asarray(score_te, dtype=float)
    else:
        proba = np.asarray(score_te, dtype=float)
        for i, cls in enumerate(class_order):
            scored[f"proba_{cls}"] = proba[:, i]
        # Shape compatibility only: no single decision score exists for
        # multiclass (see the note in _with_scores below).
        scored["score"] = scored[f"proba_{class_order[0]}"]
    dataset.save_table(scored, os.path.join(stem, "test_scores.pkl"))

    # The overtraining check needs the *training* scores too, and they are only
    # meaningful alongside the test ones: a boosted tree run back over its own
    # training rows is optimistic by construction, so the comparison is a
    # diagnostic of how much of the separation is memorisation, not a performance
    # claim. Same row order as `scored`, plus a `sample` column.
    def _proba_rows(idx, model):
        p = model.predict_proba(X.iloc[idx])
        return np.asarray(p)

    ptr, pte = _proba_rows(tr_idx, final), _proba_rows(te_idx, final)
    def _with_scores(rows_idx, proba, label_text):
        t = prep.iloc[rows_idx][[c for c in SCORED_COLUMNS if c in prep.columns]].copy()
        t["sample"] = label_text
        if n_classes <= 2:
            t["score"] = proba[:, 1].astype(float)
        else:
            for i, cls in enumerate(class_order):
                t[f"proba_{cls}"] = proba[:, i].astype(float)
            # No single decision score exists for multiclass: keep the first
            # class's probability under the conventional name for shape
            # compatibility, but nothing reads it - evaluate() and
            # performance() always use the proba_<class> columns.
            t["score"] = proba[:, 0].astype(float)
        return t
    all_scores = pd.concat([_with_scores(tr_idx, ptr, "train"),
                            _with_scores(te_idx, pte, "test")], ignore_index=True)
    dataset.save_table(all_scores, os.path.join(stem, "all_scores.pkl"))
    pd.DataFrame({"column": gain.index, "gain": gain.to_numpy()}).to_csv(
        os.path.join(stem, "gain.csv"), index=False)

    report = {
        "task": task, "model": model, "dataset_tag": dataset_tag,
        "label_map": {str(k): int(v) for k, v in lmap.items()},
        "class_order": [str(c) for c in class_order],
        "campaign": schema.campaign_of(dataset_tag),
        "min_q2_tier": min_q2_tier,
        "n_files": int(pd.Series(groups).nunique()), "n_rows": int(len(prep)),
        "allow_small_sample": bool(allow_small_sample),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "n_features": len(columns), "classes": sorted(set(prep["truth_class"].dropna())),
        "test_files": [str(g) for g in test_groups],
        "auc_train": auc_tr, "auc_test": auc_te, "auc_label_shuffle": ctrl_auc,
        "auc_label_shuffle_std": control["auc_std"], "auc_label_shuffle_max": control["auc_max"],
        "control_z_score": control["z_vs_control"],
        "class_weight": class_weight, "label_shuffle_control": control,
        "excluded_columns": sorted(set(exclude) | (set() if include_kinematics
                                                   else set(config.KINEMATIC_COLUMNS))),
        "use_charge": bool(use_charge), "include_kinematics": bool(include_kinematics),
        "include_event_level": bool(include_event_level),
        "include_track_time": bool(include_track_time),
        "features_table": used_table, "artifact_dir": stem,
        "model_artifact": "model.joblib", "booster_artifact": "booster.joblib",
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in search.best_params_.items()},
        "cv_best_score": float(search.best_score_),
        "top_gain_features": top,
        "gates": gate_results, "gates_passed": all_passed,
        "cv_folds": getattr(search, "cv_kind_", "unknown"),
        "calibrated": bool(calibrate), "calibration_method": config.CALIBRATION_METHOD,
        "calibration_skipped_reason": calibrate_skip_reason,
        "seed": seed, "n_jobs": n_jobs, "env_versions": env,
        "wall_seconds": round(time.time() - t0, 1),
        "match_weight_threshold": config.MATCH_WEIGHT_THRESHOLD,
        "min_track_p": config.MIN_TRACK_P, "delta_r_max": config.DELTA_R_MAX,
        "feature_columns": columns,
    }
    with open(os.path.join(stem, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    if not quiet:
        print(f"[train/{task}/{model}/{dataset_tag}] {len(prep)} rows, {len(columns)} features, "
              f"{n_classes} classes -> AUC train {auc_tr:.4f} / test {auc_te:.4f} "
              f"(label-shuffle control {ctrl_auc:.4f}) in {report['wall_seconds']}s")
        print(f"[train] top gain features: {top}")
        for name, res in gate_results.items():
            flag = "PASS" if res["passed"] else "FAIL"
            print(f"[gate:{flag}] {name}: "
                  + json.dumps(res["value"], default=str)
                  + " | requires " + json.dumps(res.get("limit"), default=str)
                  + (f"  [{res['detail']}]" if res.get("detail") else ""))
        print(f"[train] artifacts -> {stem}/")

    if gates and not all_passed:
        failed = [k for k, v in gate_results.items() if not v["passed"]]
        raise SystemExit(
            f"pid.train: gates failed for {task}/{model}/{dataset_tag}: {failed}. "
            "Re-run with --relax-gates to inspect the result anyway, and record the "
            "failure if the number is ever quoted.")
    return report
