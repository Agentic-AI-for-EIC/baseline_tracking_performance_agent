"""Turning a feature table into a training/evaluation dataset.

Three responsibilities, all of them safety-critical rather than cosmetic:

* **Column selection by family.** A truth column, a producer-PID column or the
  reconstructor's own mass hypothesis must never reach the model. Selection is
  therefore done by *family*, from a list, so a new column cannot leak by being
  forgotten at a call site - and :func:`assert_no_leakage` double-checks the
  resulting matrix against the labels.
* **Missing values are left missing.** LightGBM/XGBoost/HGB all learn a default
  direction for NaN; replacing it with 0.0 would invent a physically impossible
  E/p = 0 population and, worse, make "detector absent" and "detector measured
  nothing" the same input. Each subsystem also gets a ``has_*`` flag column.
* **Splitting by file, never by event.** Two tracks from the same DIS event share
  the event's kinematics and (in the +background sample) the same beam-gas
  overlay, and the same truth particle can produce several candidates; an
  event-level split would report an optimistically high AUC. ``file_id`` is the
  grouping.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from . import config

#: Columns that identify a row, carry truth information, or *are* the label.
#: Never model inputs - the reason :func:`model_columns` selects by exclusion.
KEY_COLUMNS = ("file_id", "event", "track_idx", "source_file", "label", "truth_pdg",
               "truth_class", "truth_species", "truth_pt", "truth_eta", "truth_phi",
               "truth_p", "truth_mass", "truth_generator_status", "assoc_weight",
               "is_matched", "is_fake", "pdg", "theta", "match_theta", "match_phi",
               "theta_out", "phi_out", "theta_in", "phi_in", "e_hit_over_e_clu")

#: Column-name prefixes that mark a family (used by :func:`family_of`).
#: The bare ``"p"`` is matched exactly, never by prefix: a startswith rule
#: would auto-admit any future ``pair_*``/``photon_*``/``pdg*`` column into
#: the tracker family.
FAMILY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("calorimeter", ("e_over_p", "log_e_over_p", "ecal_", "has_ecal", "shape_",
                      "rms_", "core_frac", "leakage", "e_hcal", "has_hcal",
                      "n_ecal", "e_ecal", "e_hit_over_e_clu", "has_shape")),
    ("cherenkov", ("drich", "irt_", "has_drich", "dirc")),
    ("ionisation", ("edep_", "has_ionisation")),
    ("timing", ("tof_", "track_time")),
    ("tracker", ("pt", "px", "py", "pz", "eta", "phi", "charge", "chi2", "ndf", "proj_",
                  "pathlength")),
)


def family_of(column: str) -> str:
    """Physics family a feature column belongs to (for reports and plots)."""
    if column in BOOKKEEPING_EXACT or column.endswith(BOOKKEEPING_SUFFIXES):
        return "bookkeeping"
    if column == "p":
        return "tracker"
    if column.endswith(config.EVENT_LEVEL_SUFFIX):
        return "event_context"
    for family, prefixes in FAMILY_PREFIXES:
        if column.startswith(prefixes):
            return family
    if column.startswith("n_tracks"):
        return "event_context"
    return "other"

#: Families never used as model input (see pid.schema.DEAD_LINKS / config).
BLOCKED_FAMILIES: tuple[str, ...] = config.LEAKAGE_BLOCKED_FAMILIES


#: Name patterns that mark *bookkeeping*, never physics. The join columns below
#: are artefacts of how the tables are merged (`rec_idx` from the association
#: table once leaked into a feature matrix and outranked E/p on SHAP, which is
#: meaningless - a track's position within its event's collection). Blocking by
#: pattern rather than by an enumerated list means the next join column added to
#: trkperf cannot silently become a feature either.
BOOKKEEPING_SUFFIXES: tuple[str, ...] = ("_idx", "_row", "_begin", "_end", "_codes",
                                         "_local", "_slot")
BOOKKEEPING_EXACT: frozenset[str] = frozenset(
    {"idx", "label", "hb", "he", "mb", "me", "cid", "n_hit_calc", "clu_sum_e",
     "clu_cx", "clu_cy", "theta_in", "phi_in", "theta_out", "phi_out",
     "slot", "hit_local", "obj_row"})


def _is_blocked(column: str) -> bool:
    # Explicit never-model patterns FIRST (PLAN: DEFINITIVE FEATURE
    # SEPARATION): Monte-Carlo truth, generator branches, weights and unique
    # identifiers are excluded by name, independent of every other rule, so a
    # renamed truth column cannot slip through on family mismatch.
    if column.startswith(config.NEVER_MODEL_PREFIXES) or column in config.NEVER_MODEL_EXACT:
        return True
    if column in KEY_COLUMNS or column in BOOKKEEPING_EXACT:
        return True
    if column.startswith(("truth_", "reco_", "proba_")):
        return True
    if column.endswith(BOOKKEEPING_SUFFIXES):
        return True
    if column.endswith("_matched"):
        # `<prefix>_matched` duplicates `<prefix>_has_*`; only the latter is the
        # documented missingness flag, so the matrix keeps one column per fact.
        return True
    # The cluster self-consistency diagnostic is declared never-an-input, but
    # features.py attaches it under a leg-prefixed name
    # (``<role>_<leg>_e_hit_over_e_clu``), which matches no exact/prefix/suffix
    # rule above. Block by substring so the diagnostic can never train.
    if "e_hit_over_e_clu" in column:
        return True
    return family_of(column) in BLOCKED_FAMILIES


def model_columns(df: pd.DataFrame, *, task: str,
                  include_ionisation: bool = False,
                  include_kinematics: bool = False,
                  include_event_level: bool = False,
                  include_track_time: bool = False,
                  exclude: tuple[str, ...] = config.EXCLUDE_COLUMNS_DEFAULT) -> list[str]:
    """Feature columns for `task`, honouring the block-list.

    ``include_event_level`` defaults to **False**: event-occupancy columns are an
    ablation, not the baseline (see :data:`pid.config.EVENT_LEVEL_SUFFIX` for the
    two measured reasons). Same for ``include_track_time``.

    ``include_kinematics=False`` (the default) drops p/pT/eta/phi/px/py/pz; see
    :data:`pid.config.KINEMATIC_COLUMNS` for why those are binning variables here
    rather than model inputs.

    Selection is fail-closed on top of the block-list: a column trains ONLY if
    it is named in the explicit allowlist (baseline plus whichever ablation
    flags are on). Anything else numeric-but-unlisted is dropped with a loud
    warning - adding a feature requires adding its name to
    :data:`pid.config.ALLOWED_BASELINE_COLUMNS` (or a flag-gated set) plus a
    test,     never silent auto-admission.
    """
    blocked = set(exclude)
    allowed = set(config.ALLOWED_BASELINE_COLUMNS)
    if include_kinematics:
        allowed |= set(config.KINEMATIC_COLUMNS)
    if include_track_time:
        allowed |= set(config.TRACK_TIME_COLUMNS)
    if include_event_level:
        allowed |= set(config.ALLOWED_EVENT_COLUMNS)
    if include_ionisation:
        allowed |= set(config.ALLOWED_IONISATION_COLUMNS)
    if "charge" not in blocked:
        # Explicit opt-in (--use-charge) re-admits the beam-charge tag; the
        # default exclude list keeps it out.
        allowed.add("charge")
    # Documented ablation opt-outs stay silent: these columns are excluded by
    # flag, not by surprise, so they must not trip the fail-closed warning.
    silent = set()
    if not include_kinematics:
        silent |= set(config.KINEMATIC_COLUMNS)
    if not include_track_time:
        silent |= set(config.TRACK_TIME_COLUMNS)
    if not include_event_level:
        silent |= {c for c in df.columns if c.endswith(config.EVENT_LEVEL_SUFFIX)}
    if not include_ionisation:
        silent |= {c for c in df.columns if family_of(c) == "ionisation"}
    cols: list[str] = []
    dropped: list[str] = []
    for c in df.columns:
        if _is_blocked(c) or c in blocked or c in silent:
            continue
        if c == "leg":  # admitted for the pooled task only, see below
            cols.append(c)
            continue
        if c not in allowed:
            # Strings could never train; warn only on numeric columns, where a
            # dropped name means a measurable quantity the model will not see.
            if pd.api.types.is_numeric_dtype(df[c]):
                dropped.append(c)
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if int(df[c].notna().sum()) == 0:
            # Unmeasurable here (e.g. opposite-leg calorimeters on a leg-scoped
            # task): not a usable feature, and keeping the name would misalign
            # every positional consumer downstream (SHAP indices).
            dropped.append(c)
            continue
        cols.append(c)
    if dropped:
        print(f"[dataset] WARNING: {len(dropped)} numeric column(s) excluded "
              f"(not allowlisted, or all-NaN here): {sorted(dropped)}",
              file=sys.stderr)
    # NOTE: no event-level / ionisation post-filters here on purpose - the
    # allowlist above already admits those families only under their flags,
    # so a second filter would be dead code hiding the single source of truth.
    if task != "pooled":
        # `leg` distinguishes the two hemispheres. Inside a single-leg task it is
        # constant; in a pooled task it is exactly the shortcut the model must not
        # be quietly rewarded for taking, so it is offered there only - and the
        # pooled arm's report is where the size of that shortcut gets measured.
        cols = [c for c in cols if c != "leg"]
    return sorted(set(cols))


def categorical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Columns to be treated as categorical (currently only ``leg``)."""
    return [c for c in columns if c == "leg"]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

#: Binary tasks: which truth class is the positive one, which classes are allowed.
#: e is always positive so that "electron efficiency" is the fraction with a high
#: score, matching how the fake-rate working points are quoted.
_BINARY_SPEC: dict[str, dict] = {
    "eid": {"positive": {"e"}, "allowed": {"e", "pi"}},
    "ehad": {"positive": {"e"}, "allowed": {"e", *config.HADRON_CLASSES}},
}


def label_map(task: str):
    """``{truth_class: label}`` for `task` (None for classes outside the task)."""
    if task in _BINARY_SPEC:
        spec = _BINARY_SPEC[task]
        return {c: (1 if c in spec["positive"] else 0) for c in spec["allowed"]}
    classes = [c for c in config.CLASS_LABELS if c in config.TASKS[task]["classes"]]
    return {c: i for i, c in enumerate(classes)}


def prepare(df: pd.DataFrame, task: str, *, require_matched: bool = True,
            min_p: float | None = None) -> pd.DataFrame:
    """Filter and label the rows that participate in `task`.

    Parameters
    ----------
    require_matched:
        ``True`` restricts to truth-matched tracks (a pure PID measurement).
        ``False`` keeps unmatched tracks too, for a future "fake tracks faking
        electrons" study (no train CLI path yet); those rows carry
        ``label = -1`` and are excluded from training by :func:`pid.train`.
    min_p:
        Reconstructed-momentum floor (default ``config.MIN_TRACK_P``) applied
        to the reco ``p`` column - not the truth momentum.
    """
    if task not in config.TASKS:
        raise ValueError(f"unknown PID task {task!r}; choices {sorted(config.TASKS)}")
    min_p = config.MIN_TRACK_P if min_p is None else min_p
    spec = config.TASKS[task]
    out = df[df["p"] >= min_p].copy()
    if spec["leg"] != "all":
        out = out[out["leg"] == spec["leg"]]
    if require_matched:
        out = out[out["is_matched"].fillna(False).astype(bool)]

    lmap = label_map(task)
    if require_matched:
        out = out[out["truth_class"].isin(lmap)].copy()
        out["label"] = out["truth_class"].map(lmap).astype(int)
    else:
        out["label"] = out["truth_class"].map(lmap).fillna(-1).astype(int)

    if "leg" in out.columns and out["leg"].dtype != "category":
        out["leg"] = out["leg"].astype("category")
    out.attrs.update({"task": task, "labels": lmap})
    return out


def split_by_file(df: pd.DataFrame, *, test_size: float = config.TEST_SIZE,
                  seed: int = config.RANDOM_SEED):
    """Train/test row index arrays, split on ``file_id`` (never on events).

    Returns ``(train_idx, test_idx, groups_used)``. Files are shuffled with a
    fixed seed; with a single file (the local smoke test) the split is by event
    parity instead, and that fallback is *reported* so a one-file run can never
    be mistaken for a leakage-safe evaluation.

    Splits are not stratified: a tiny-file test set can miss a class entirely,
    which surfaces downstream as a ``MIN_ROWS_PER_CLASS`` refusal, not here.
    """
    groups = df["file_id"].to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    if uniq.size >= 2:
        perm = rng.permutation(uniq)
        n_test = max(1, int(round(test_size * uniq.size)))
        test_files = set(perm[:n_test].tolist())
        is_test = np.array([g in test_files for g in groups])
        return np.where(~is_test)[0], np.where(is_test)[0], sorted(test_files)
    # single-file fallback: event parity (documented, not silent)
    warnings.warn("pid.dataset.split_by_file: single file - falling back to an "
                  "event-parity split, which is optimistic (same shower seen on "
                  "both sides). Fine for a smoke test, NOT a quotable evaluation.",
                  UserWarning, stacklevel=2)
    is_test = (df["event"].to_numpy() % 5) == 0
    return np.where(~is_test)[0], np.where(is_test)[0], ["<single-file: event split>"]


def design_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """    X as a DataFrame (names preserved; categorical kept as ``category``).

    DataFrames - not raw ndarrays - are passed to the learners on purpose: the
    column names then survive into gain and SHAP reports, and a misordered
    feature block becomes impossible to hide.
    """
    X = df[list(columns)].copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            continue
        before = int(pd.Series(X[c]).notna().sum())
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
        if int(X[c].notna().sum()) < before:
            raise ValueError(
                f"pid.dataset.design_matrix: column {c!r} holds non-numeric "
                "values that coerce to NaN - a string column must never "
                "silently become missing data")
    # All-NaN columns carry zero information no learner can split on, but they
    # crash learners without native-missing binning (observed: sklearn HGB
    # aborts inside its threshold search). Drop them loudly instead.
    empty = [c for c in X.columns if X[c].notna().sum() == 0]
    if empty:
        print(f"[dataset] NOTE: dropping {len(empty)} all-NaN column(s) with no "
              f"measurable values: {sorted(empty)}", file=sys.stderr)
        X = X.drop(columns=empty)
    for c in categorical_columns(df, columns):
        X[c] = X[c].astype("category")
    return X


def assert_no_leakage(X: pd.DataFrame, y: np.ndarray, *,
                      threshold: float = config.LEAKAGE_MAX_LABEL_CORRELATION) -> pd.DataFrame:
    """Fail if any single feature predicts the label almost perfectly.

    A |point-biserial correlation| above `threshold` on a boosted-tree input
    means something truth-derived slipped into the matrix (the classic mistakes
    are ``truth_*`` kinematics, the producer's ``*_ParticleIDs``, or
    ``CentralCKFTracks.pdg``). Raises ``ValueError`` and halts the caller -
    this gate must be impossible to sail past quietly. Returns the per-column
    correlation table so the offending column is named in the failure message.

    Empty by construction on a legitimate PID feature set; the largest genuine
    single-variable correlations here are ECAL energy at ~0.71 and E/p at ~0.6.
    """
    yv = np.asarray(y, dtype=float)
    if np.unique(yv[~np.isnan(yv)]).size < 2:
        raise ValueError("pid.dataset.assert_no_leakage: label has a single class")
    rows = []
    for c in X.columns:
        series = pd.Series(X[c])
        if str(series.dtype) == "category" or not pd.api.types.is_numeric_dtype(series):
            # Categorical inputs (e.g. `leg` in the pooled task) are compared via
            # their integer codes; a string->float conversion would raise here.
            v = (series.cat.codes if str(series.dtype) == "category"
                 else series.astype("category").cat.codes).to_numpy(dtype="float64")
        else:
            v = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
        ok = np.isfinite(v) & np.isfinite(yv)
        if ok.sum() < 10 or np.nanstd(v[ok]) == 0:
            rows.append({"column": c, "corr_with_label": 0.0})
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(np.corrcoef(v[ok], yv[ok])[0, 1])
        if not np.isfinite(corr):
            # The label is constant on the valid subset while it varies
            # globally: the column's *missingness pattern* predicts the label
            # perfectly (measured in only one class). That is a leak through
            # NaN placement, so report it as a perfect correlation - abs(NaN)
            # would otherwise sail under the threshold unnoticed.
            corr = 1.0
        rows.append({"column": c, "corr_with_label": corr})
    table = pd.DataFrame(rows).sort_values("corr_with_label",
                                          key=lambda s: s.abs(), ascending=False)
    top = table.iloc[0]
    if abs(top["corr_with_label"]) > threshold:
        raise ValueError(
            f"pid.dataset.assert_no_leakage: feature {top['column']!r} correlates "
            f"{top['corr_with_label']:+.3f} with the label (threshold "
            f"{threshold}). This is a leakage bug, not a good result - the column "
            "must be removed from the feature set (see pid.dataset.KEY_COLUMNS).")
    return table


# ---------------------------------------------------------------------------
# Persistence (no parquet engine exists in this environment)
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, path: str) -> str:
    import os

    from trkperf import io as tk_io  # noqa: F401 - import guard: features flow uses trkperf I/O
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_pickle(path)
    return path


def load_table(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Per-column NaN fraction / finite range - the M1 gate's evidence table."""
    cols = columns or list(df.columns)
    rows = []
    # Text columns (e.g. the match-source label) are not numeric features; they
    # are flagged as such so "empty" never fires just because to_numeric() cannot
    # parse a string.
    # pandas 3 stores strings as a `str` dtype (not object), so classify by
    # "is it numeric?" rather than by an explicit dtype list.
    text_cols = {c for c in cols if c in df.columns
                 and not (pd.api.types.is_numeric_dtype(df[c])
                          or pd.api.types.is_bool_dtype(df[c]))}
    for c in cols:
        if c not in df.columns:
            continue
        try:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        except (TypeError, ValueError):
            rows.append({"column": c, "family": family_of(c), "dtype": str(df[c].dtype),
                         "nan_fraction": float(df[c].isna().mean()), "min": np.nan,
                         "max": np.nan, "n_distinct": int(df[c].nunique())})
            continue
        finite = np.isfinite(v)
        rows.append({
            "column": c, "family": family_of(c), "dtype": str(df[c].dtype),
            "nan_fraction": float(1.0 - finite.mean()) if len(v) else np.nan,
            "min": float(np.nanmin(v)) if finite.any() else np.nan,
            "max": float(np.nanmax(v)) if finite.any() else np.nan,
            "n_distinct": int(pd.Series(df[c]).nunique()),
            "numeric": c not in text_cols and c not in ("",),
        })
    return pd.DataFrame(rows)


def empty_numeric_columns(table: pd.DataFrame, *, threshold: float = 0.999) -> pd.DataFrame:
    """Columns that are numeric features but effectively empty in this sample."""
    if table.empty or "numeric" not in table.columns:
        return table
    return table[table["numeric"] & (table["nan_fraction"] >= threshold)]
