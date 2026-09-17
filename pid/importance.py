"""Feature importance: gain, exact SHAP, permutation - and the physics check.

Three independent rankings, because they answer different questions:

``gain``
    how much each feature contributed to the trees' split quality - sensitive to
    correlated duplicates (E/p and log E/p both score), so it is a screen, not a
    verdict;
``shap``
    **exact** TreeSHAP values from the booster itself (the ``shap`` package is not
    installed in this environment, but LightGBM and XGBoost compute TreeSHAP
    internally). Signed and additive, so it answers the question the plan asks:
    does a *larger* E/p push the score toward electron, and by how much, per
    feature, per class?
``permutation``
    model-agnostic accuracy drop when a column is shuffled within the held-out
    files; available for every learner including sklearn's HGB, which exposes no
    gain array worth trusting.

:func:`physics_check` is the gate: for the electron tasks E/p must lead the
ranking, because a PID that found some other route (occupancy, an event-level
variable, a leaking truth column) is not measuring what it claims to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, dataset, models


def gain_table(model, columns: list[str], *, model_name: str) -> pd.DataFrame:
    adapter = models.get_adapter(model_name)
    gain = adapter.gain(model, columns)
    total = float(np.nansum(gain.to_numpy()))
    return pd.DataFrame({
        "column": gain.index,
        "family": [dataset.family_of(c) for c in gain.index],
        "gain": gain.to_numpy(),
        "gain_fraction": (gain.to_numpy() / total) if total > 0 else np.nan,
        "rank": np.arange(1, len(gain) + 1),
    })


def shap_table(model, X: pd.DataFrame, *, model_name: str,
               max_rows: int = 5000, label: np.ndarray | None = None) -> pd.DataFrame:
    """Mean |SHAP| and signed mean SHAP per feature (exact, from the booster)."""
    adapter = models.get_adapter(model_name)
    try:
        values = adapter.shap(model, X.head(max_rows))
    except NotImplementedError as exc:
        print(f"[importance] {exc}")
        return pd.DataFrame(columns=["column", "mean_abs_shap", "mean_shap", "family"])
    cols = list(X.columns)
    return pd.DataFrame({
        "column": cols,
        "family": [dataset.family_of(c) for c in cols],
        "mean_abs_shap": np.nanmean(np.abs(values), axis=0),
        "mean_shap": np.nanmean(values, axis=0),
        "shap_std": np.nanstd(values, axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_table(model, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5,
                      seed: int = config.RANDOM_SEED, scoring: str = "roc_auc_ovr") -> pd.DataFrame:
    """Accuracy drop under column permutation (model-agnostic)."""
    from sklearn.inspection import permutation_importance

    if np.unique(y).size < 2:
        return pd.DataFrame(columns=["column", "family", "importance_mean", "importance_std"])
    multi = np.unique(y).size > 2
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=1,
        scoring=(scoring if multi else "roc_auc"))
    cols = list(X.columns)
    out = pd.DataFrame({
        "column": cols,
        "family": [dataset.family_of(c) for c in cols],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return out


def physics_check(tables: dict[str, pd.DataFrame], task: str) -> dict:
    """Does the ranking look like physics, or like a shortcut?

    Returns ``{"passed": bool, "top": {...}, "notes": [...]}``. The expected
    leading families are declared in :data:`pid.train.EXPECTED_TOP_FEATURES`;
    event-level columns are called out explicitly, since a PID result whose top
    variable is an occupancy proxy is measuring event multiplicity, not
    ionisation or shower shape.

    Deliberately rank-only: only the FIRST non-empty ranking (SHAP, then gain,
    then permutation) is inspected, and only the feature's identity is
    checked - not the sign of its effect. A high-E/p → hadron inversion with
    E/p on top would pass; the signed direction is left to the human reading
    the SHAP table (``mean_shap`` column).
    """
    from . import train as pid_train

    notes: list[str] = []
    best = None
    for kind in ("shap", "gain", "permutation"):
        table = tables.get(kind)
        if table is None or table.empty:
            continue
        top = table.iloc[0]
        best = {"kind": kind, "column": str(top["column"]), "family": str(top["family"])}
        patterns = pid_train.EXPECTED_TOP_FEATURES.get(task, ())
        if not any(p in str(top["column"]) for p in patterns):
            notes.append(
                f"{kind}: leading feature {top['column']!r} (family "
                f"{top['family']!r}) is not one of the expected PID observables "
                f"{list(patterns)}")
        if str(top["column"]).endswith("_evt") or str(top["column"]) == "n_tracks_evt":
            notes.append(f"{kind}: leading feature is event-level -> the model is "
                         "using occupancy, not per-track detector response")
        # Kinematics are excluded from the feature set by default precisely because
        # the DIS flux makes p/pT an easier discriminant than any detector response
        # (pid.config.KINEMATIC_COLUMNS); if they were re-enabled, say so loudly.
        kin = {"p", "pt", "eta", "phi", "px", "py", "pz"}
        if str(top["column"]) in kin:
            notes.append(f"{kind}: leading feature {top['column']!r} is track "
                         "kinematics, i.e. the flux shortcut - see "
                         "pid.config.KINEMATIC_COLUMNS")
        for row in tables[kind].head(3).itertuples():
            if str(getattr(row, "column", "")).endswith("_evt"):
                notes.append(f"{kind}: an event-level column is in the top 3 "
                             f"({row.column}); occupancy is contributing to the "
                             "decision (ablate with --no-event-level)")
        break
    if best is None:
        return {"passed": False, "top": None, "notes": ["no importance table available"]}
    return {"passed": len(notes) == 0, "top": best, "notes": notes}


def load_model_artifact(scores_or_model_dir: str):
    """Load the estimator gains/SHAP are read from, and which file it came from.

    Prefers ``booster.joblib`` (the unwrapped fitted booster written by
    :func:`pid.train.train`); falls back to ``model.joblib`` for artifact dirs
    predating it     (or uncalibrated runs, where both files coincide).
    """
    import joblib
    import os

    booster_path = os.path.join(scores_or_model_dir, "booster.joblib")
    if os.path.exists(booster_path):
        return joblib.load(booster_path), "booster.joblib"
    return joblib.load(os.path.join(scores_or_model_dir, "model.joblib")), "model.joblib"


def importance(scores_or_model_dir: str, *, model_name: str, task: str,
               kind: str = "all", out_dir: str = config.OUTPUT_DIR,
               dataset_tag: str = "clean", max_rows: int = 5000,
               write: bool = True) -> dict:
    """Compute the importance tables for a trained artifact directory.

    ``scores_or_model_dir`` is the directory written by :func:`pid.train.train`
    (containing ``model.joblib``, ``booster.joblib``, ``report.json`` and the
    feature-table path in the report). Gains and SHAP are read from
    ``booster.joblib`` - the unwrapped fitted booster - because ``model.joblib``
    is the calibrated wrapper once calibration has run, and a
    CalibratedClassifierCV exposes no single booster (its ``predict`` also
    accepts no SHAP kwargs). Re-reads the feature table used for training, so
    the numbers describe the same rows the model saw.
    """
    import json
    import os

    report_path = os.path.join(scores_or_model_dir, "report.json")
    with open(report_path) as fh:
        rep = json.load(fh)
    columns = rep["feature_columns"]
    features_path = rep.get("features_table")
    if not features_path or not os.path.exists(features_path):
        # Fall back to the conventional location written by pid.cli.
        guess = os.path.join(config.OUTPUT_DIR, f"pid-features_{dataset_tag}.pkl")
        if not os.path.exists(guess):
            raise SystemExit(
                f"pid.importance: cannot find the feature table used to train "
                f"{scores_or_model_dir} (report.json has features_table="
                f"{features_path!r}); pass the same --features used for training.")
        features_path = guess
    df = dataset.load_table(features_path)
    prep = dataset.prepare(df, task, require_matched=True)
    missing = [c for c in columns if c not in prep.columns]
    if missing:
        raise SystemExit(f"pid.importance: feature table is missing trained columns "
                         f"{missing[:6]} - rebuild features before computing importance")
    X = dataset.design_matrix(prep, columns)
    y = prep["label"].to_numpy(int)
    model, artifact = load_model_artifact(scores_or_model_dir)
    print(f"[importance] gains/SHAP read from {artifact} "
          f"({'unwrapped booster' if artifact == 'booster.joblib' else 'legacy fallback'})")

    tables: dict[str, pd.DataFrame] = {}
    if kind in ("all", "gain"):
        tables["gain"] = gain_table(model, columns, model_name=model_name)
    if kind in ("all", "shap"):
        tables["shap"] = shap_table(model, X, model_name=model_name, max_rows=max_rows)
    if kind in ("all", "permutation"):
        tables["permutation"] = permutation_table(model, X, y)
    check = physics_check(tables, task)

    if write:
        from . import report as pid_report

        stem = f"pid-{task}_{model_name}_{dataset_tag}-importance"
        for name, table in tables.items():
            if table.empty:
                continue
            pid_report.write(table.assign(importance_kind=name),
                             os.path.join(out_dir, stem + f"-{name}"),
                             tree_name=f"pid_{task}_importance",
                             meta={"importance_kind": name, "task": task,
                                   "model": model_name, "dataset_tag": dataset_tag,
                                   "physics_check_passed": check["passed"],
                                   "physics_check_notes": "; ".join(check["notes"])})
    return {"tables": tables, "check": check, "report": rep}
