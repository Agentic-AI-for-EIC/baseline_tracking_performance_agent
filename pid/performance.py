"""Maximum-significance PID performance package.

For each physics channel - electron ID (e/π, e/hadrons) and hadron ID (K/π, p/K) -
this module

1. finds the working point that maximises FOM(c) = S(c)/sqrt(S(c)+B(c)), globally
   and per pT bin,
2. evaluates efficiency, mis-ID (fake) rate, purity and rejection **at that cut**,
   differentially in pT and p and in 2D (pT, η),
3. adds the extended suite: Gaussian-equivalent nσ vs p, the multi-class confusion
   matrix, and the train-vs-test score comparison that exposes memorisation,
4. writes every table (JSON + markdown + ROOT TNtuple) and figure, plus one
   summary table listing (c*, peak FOM, efficiency, purity) per channel.

Channel scores
--------------
The binary tasks already hold a usable score. A hadron channel is *reduced* to its
two hypotheses, score = P(sig) / (P(sig) + P(bkg)) over truth in {sig, bkg}: the
optimal two-hypothesis discriminant under a flat prior, and the only way to make a
"K vs π" statement that is not quietly contaminated by the proton and electron
classes still sitting in the multiclass softmax. The consequence must be stated
when the number is quoted: **c* lives in that channel's score space**, so a K/π cut
of 0.62 and an e/π cut of 0.62 are not the same thing.

Why limits appear next to every background-driven number
--------------------------------------------------------
This sample has ~62 backward pions and ~160 kaons per file. A fake rate of 0/N is
not infinite rejection, it is an upper limit: :mod:`pid.significance` returns the
Clopper-Pearson limit alongside, and the figures plot the limit as the headline
quantity wherever the point estimate is undefined. Same for the FOM itself: with a
very small background, S/sqrt(S+B) -> sqrt(S) and its maximum migrates to "accept
everything", which is flagged as statistics-limited rather than reported as a
working point.
"""

from __future__ import annotations

import os

import sys

import numpy as np
import pandas as pd

from . import config, dataset, evaluate, plots, report, significance


# ---------------------------------------------------------------------------
# Binning basis: reconstructed vs truth kinematics
# ---------------------------------------------------------------------------
#: Every binned merit can be computed against either the RECONSTRUCTED track
#: kinematics (what an analysis actually has event by event, and what a cut is
#: applied to) or the TRUTH partner's (what the tracking metrics of this project
#: bin in, per AGENTS.md "bin in truth pT/eta", and free of resolution
#: migration). The two are NOT interchangeable: the measured difference in bin
#: populations for one clean file is ~3 % in the lowest pT bin, and the
#: convention is recorded in every table (`variable`) and figure (x-axis label).
BIN_VARIABLES: dict[str, dict] = {
    "pt":        {"column": "pt",        "edges": "PT_BINS_PERFORMANCE",
                  "label": "track $p_T$ [GeV]",        "basis": "reconstructed"},
    "p":         {"column": "p",         "edges": "P_BINS_PERFORMANCE",
                  "label": "track momentum $p$ [GeV]", "basis": "reconstructed"},
    "eta":       {"column": "eta",       "edges": "ETA_BINS_PERFORMANCE",
                  "label": "track $\\eta$",           "basis": "reconstructed"},
    "truth_pt":  {"column": "truth_pt",  "edges": "PT_BINS_PERFORMANCE",
                  "label": "truth $p_T$ [GeV]",        "basis": "truth"},
    "truth_p":   {"column": "truth_p",   "edges": "P_BINS_PERFORMANCE",
                  "label": "truth momentum $p$ [GeV]", "basis": "truth"},
    "truth_eta": {"column": "truth_eta", "edges": "ETA_BINS_PERFORMANCE",
                  "label": "truth $\\eta$",           "basis": "truth"},
}

#: basis name -> the three variables to bin in
BIN_BASES: dict[str, dict[str, str]] = {
    "reco":  {"pt": "pt", "p": "p", "eta": "eta"},
    "truth": {"pt": "truth_pt", "p": "truth_p", "eta": "truth_eta"},
}


def resolve_bin_variable(name: str, frame: pd.DataFrame) -> tuple[str, np.ndarray, str, str]:
    """``(column, edges, axis_label, basis)`` for a binning variable name.

    Fails with the remedy in the message: older score tables predate the truth
    columns, and a re-train is the fix, not a silent fallback to reco binning.
    """
    if name not in BIN_VARIABLES:
        raise KeyError(f"unknown binning variable {name!r}; choices {sorted(BIN_VARIABLES)}")
    spec = BIN_VARIABLES[name]
    column = spec["column"]
    if column not in frame.columns:
        raise KeyError(
            f"pid.performance: binning variable {name!r} needs column {column!r}, which "
            "this score table does not carry. Re-train with the current pid.train (it "
            "writes truth_pt/truth_p/truth_eta into test_scores.pkl/all_scores.pkl), or "
            "use the reconstructed basis (--bin-source reco).")
    return column, getattr(config, spec["edges"]), spec["label"], spec["basis"]


# ---------------------------------------------------------------------------
# Channel scores
# ---------------------------------------------------------------------------

def channel_frame(scores: pd.DataFrame, channel: str, *, model_dir: str = "") -> pd.DataFrame:
    """Reduce a score table to one channel: ``score`` (higher = more signal-like) and ``y``.

    Rows with non-finite scores are dropped up front (a NaN score would
    otherwise count as "failing" every cut) and counted in
    ``df.attrs["n_dropped_nonfinite"]``.

    Raises
    ------
    KeyError / ValueError
        Naming the columns that are missing, so a mis-wired channel fails loudly
        instead of silently evaluating the wrong hypothesis pair.
    """
    if channel not in config.CHANNELS:
        raise KeyError(f"unknown channel {channel!r}; choices {sorted(config.CHANNELS)}")
    spec = config.CHANNELS[channel]
    sig, bkg = spec["signal"], spec["background"]
    need = ["pt", "p", "eta", "truth_class"]
    missing = [c for c in need if c not in scores.columns]
    if missing:
        raise KeyError(f"pid.performance: score table for {channel} lacks columns {missing} "
                       f"(from {model_dir or 'memory'})")
    df = scores.copy()
    if spec["pair"]:
        for cls in (sig, bkg):
            if f"proba_{cls}" not in df.columns:
                raise KeyError(
                    f"pid.performance: channel {channel} needs proba_{cls}; the table has "
                    f"{[c for c in df.columns if c.startswith('proba_')]}. Train the "
                    f"{spec['task']} task on this sample first.")
        df = df[df["truth_class"].isin([sig, bkg])].copy()
        ps = df[f"proba_{sig}"].to_numpy(dtype=float)
        pb = df[f"proba_{bkg}"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["score"] = np.where(ps + pb > 0, ps / (ps + pb), np.nan)
        df["y"] = (df["truth_class"] == sig).astype(int)
    else:
        if "score" not in df.columns or "label" not in df.columns:
            raise KeyError(
                f"pid.performance: channel {channel} needs 'score' and 'label' columns; "
                f"got {sorted(df.columns)[:12]}...")
        df["y"] = df["label"].to_numpy(dtype=int)
        truth_bkg = {"pi": ("pi",), "hadron": config.HADRON_CLASSES}[bkg]
        keep = df["truth_class"].isin([sig, *truth_bkg])
        if not keep.all():
            df = df[keep].copy()
    df = df[np.isfinite(df["score"].to_numpy(dtype=float))].copy()
    df.attrs.update({"channel": channel, "signal": sig, "background": bkg,
                     "pair": bool(spec["pair"]), "title": spec["title"],
                     "n_dropped_nonfinite": int((~np.isfinite(
                         scores["score"].to_numpy(dtype=float))).sum())})
    return df


# ---------------------------------------------------------------------------
# Working points
# ---------------------------------------------------------------------------

def working_point(df: pd.DataFrame, channel: str, *, mode: str = "s_over_sqrt",
                  weights: np.ndarray | None = None,
                  lumi_scale: float = config.DEFAULT_LUMI_SCALE) -> dict:
    """Global maximum-significance cut for one channel."""
    sig, bkg = config.CHANNELS[channel]["signal"], config.CHANNELS[channel]["background"]
    table, opt = significance.significance_curve(
        df["score"].to_numpy(dtype=float), df["y"].to_numpy(),
        weights=weights, mode=mode, lumi_scale=lumi_scale, signal=1, background=0)
    opt.update({"channel": channel, "signal": sig, "background": bkg,
                "n_rows": int(len(df)), "score_space": "pair" if config.CHANNELS[channel]["pair"]
                else "task", "lumi_scale": float(lumi_scale),
                "weighted": weights is not None})
    return {"optimum": opt, "scan": table}


def working_points_vs_pt(df: pd.DataFrame, channel: str, *, bins=None,
                         weights: np.ndarray | None = None,
                         lumi_scale: float = config.DEFAULT_LUMI_SCALE,
                         mode: str = "s_over_sqrt", variable: str = "pt",
                         n_thresholds: int | None = None) -> pd.DataFrame:
    """c*(pT), bin by bin. `variable` selects the binning basis (see BIN_VARIABLES).

    `n_thresholds` fixes the scan grid to the one the per-bin FOM-curve
    figures draw: two optima from two grids can disagree on flat plateaus.
    The default is the same coarse grid the curves use
    (`config.N_THRESHOLD_SCAN_BINNED`), not the fine global-scan grid.
    """
    column, default_edges, _label, basis = resolve_bin_variable(variable, df)
    bins = default_edges if bins is None else bins
    finite = np.isfinite(df["score"].to_numpy(dtype=float))
    w = None if weights is None else np.asarray(weights, dtype=float)[finite]
    grid = np.linspace(0.0, 1.0, int(n_thresholds) if n_thresholds
                       else int(config.N_THRESHOLD_SCAN_BINNED))
    tab = significance.optimal_cut_in_bins(
        df["score"].to_numpy(dtype=float), df["y"].to_numpy(),
        df[column].to_numpy(dtype=float), bins=bins, weights=w, mode=mode,
        thresholds=grid, signal=1, background=0, lumi_scale=lumi_scale)
    tab.insert(0, "channel", channel)
    tab.insert(1, "binning_basis", basis)
    tab.insert(2, "bin_column", column)
    return tab


# ---------------------------------------------------------------------------
# Differential performance at a chosen cut
# ---------------------------------------------------------------------------

def at_cut_table(df: pd.DataFrame, channel: str, *, cut: float | None = None,
                 cut_per_bin: pd.DataFrame | None = None, variable: str = "pt",
                 bins: np.ndarray | None = None,
                 weights: np.ndarray | None = None) -> pd.DataFrame:
    """Efficiency, fake rate, purity and rejection per bin at a fixed (or per-bin) cut.

    ``cut`` applies one global threshold; ``cut_per_bin`` (from
    :func:`working_points_vs_pt`) applies each bin's own optimum. Pass both to see
    how much a per-bin working point buys - which is the whole argument for
    kinematic-dependent PID cuts, and can only be shown by comparing them.

    Rows with non-finite binning variable are dropped up front (a NaN momentum
    would otherwise fold into an edge bin via the clipped digitize below) and
    counted in the printed note. Values outside the bin edges fold into the
    edge bins by construction. Point estimates use the (possibly weighted)
    yields while the ``*_err_lo/_hi`` intervals and rejection limits are
    count-based - identical when weights are constant per class, approximate
    under a ``bkg_scale`` rescaling.
    """
    spec = config.CHANNELS[channel]
    out = df.copy()
    w = np.ones(len(out), dtype=float) if weights is None else np.asarray(weights, float)
    column, default_edges, axis_label, basis = resolve_bin_variable(variable, out)
    bins = default_edges if bins is None else np.asarray(bins)
    okx = np.isfinite(out[column].to_numpy(dtype=float))
    n_dropped = int((~okx).sum())
    if n_dropped:
        print(f"[performance/{channel}] NOTE: {n_dropped} rows with non-finite "
              f"{column} excluded from {variable} binning", file=sys.stderr)
        out = out.loc[okx].reset_index(drop=True)
        w = w[okx]
    idx = np.clip(np.digitize(out[column].to_numpy(dtype=float), bins[1:-1], right=True),
                  0, len(bins) - 2)
    out["_bin"] = idx
    per_bin_cut = None
    if cut_per_bin is not None and len(cut_per_bin):
        lookup = dict(zip(np.round(cut_per_bin["bin_center"].to_numpy(dtype=float), 6),
                          cut_per_bin["threshold"].to_numpy(dtype=float)))
        centers = np.array([0.5 * (bins[b] + bins[b + 1]) for b in range(len(bins) - 1)])
        per_bin_cut = np.array([lookup.get(round(float(c), 6), np.nan) for c in centers])

    rows = []
    nan_row = {"efficiency": np.nan, "efficiency_err_lo": np.nan,
               "efficiency_err_hi": np.nan, "fake_rate": np.nan,
               "fake_rate_err_lo": np.nan, "fake_rate_err_hi": np.nan,
               "purity": np.nan, "purity_err_lo": np.nan, "purity_err_hi": np.nan,
               "rejection": np.nan, "rejection_limit": np.nan,
               "significance": np.nan, "n_signal_pass": 0, "n_background_pass": 0}
    for b in range(len(bins) - 1):
        sel = (out["_bin"] == b).to_numpy()
        y = out["y"].to_numpy()[sel] if sel.any() else np.array([], dtype=int)
        sc = out["score"].to_numpy(dtype=float)[sel] if sel.any() else np.array([])
        ww = w[sel]
        c = (float(per_bin_cut[b]) if per_bin_cut is not None and np.isfinite(per_bin_cut[b])
             else (float(cut) if cut is not None else np.nan))
        n_sig, n_bkg = int((y == 1).sum()), int((y == 0).sum())
        row = {"channel": channel, "signal": spec["signal"], "background": spec["background"],
               "variable": variable, "binning_basis": basis, "bin_column": column,
               "bin": f"({bins[b]:g}, {bins[b + 1]:g}]",
               "bin_center": float(0.5 * (bins[b] + bins[b + 1])), "cut": c,
               "cut_source": ("per_bin_optimum" if per_bin_cut is not None and
                              np.isfinite(c) else "global" if np.isfinite(c) else "none"),
               "n_signal": n_sig, "n_background": n_bkg,
               "S": float(ww[y == 1].sum()), "B": float(ww[y == 0].sum())}
        if not sel.any():
            # An empty bin is an explicit NaN row, never a silent gap: figures
            # draw it as a hole and the manifest can tell it apart.
            row.update({**nan_row, "insufficient_stats": True,
                        "reason": "no candidates in this bin"})
            rows.append(row)
            continue
        if not np.isfinite(c):
            row.update({"efficiency": np.nan, "efficiency_err_lo": np.nan,
                        "efficiency_err_hi": np.nan, "fake_rate": np.nan,
                        "fake_rate_err_lo": np.nan, "fake_rate_err_hi": np.nan,
                        "purity": np.nan, "purity_err_lo": np.nan, "purity_err_hi": np.nan,
                        "rejection": np.nan, "rejection_limit": np.nan,
                        "significance": np.nan,
                        "insufficient_stats": True,
                        "reason": f"{n_sig + n_bkg} candidates < {config.MIN_CANDIDATES_PER_WP_BIN}"
                        if n_sig + n_bkg < config.MIN_CANDIDATES_PER_WP_BIN else "no cut defined"})
            rows.append(row)
            continue
        pass_sig = sc >= c
        k_sig, k_bkg = int((pass_sig & (y == 1)).sum()), int((pass_sig & (y == 0)).sum())
        S = float(ww[(y == 1) & pass_sig].sum())
        B = float(ww[(y == 0) & pass_sig].sum())
        Stot, Btot = float(ww[y == 1].sum()), float(ww[y == 0].sum())
        f_lo, f_up = binomial_interval(k_bkg, n_bkg)
        rej = significance.rejection_limits(k_bkg, n_bkg)
        rows.append({**row,
                     "n_signal_pass": k_sig, "n_background_pass": k_bkg,
                     "efficiency": S / Stot if Stot > 0 else np.nan,
                     "efficiency_err_lo": _lo(k_sig, n_sig),
                     "efficiency_err_hi": _hi(k_sig, n_sig),
                     "fake_rate": B / Btot if Btot > 0 else np.nan,
                     "fake_rate_err_lo": f_lo, "fake_rate_err_hi": f_up,
                     "purity": S / (S + B) if (S + B) > 0 else np.nan,
                     "purity_err_lo": _lo(k_sig, k_sig + k_bkg),
                     "purity_err_hi": _hi(k_sig, k_sig + k_bkg),
                     "rejection": rej["rejection"], "rejection_limit": rej["rejection_limit"],
                     "rejection_measurable": rej["measurable"],
                     "significance": (S / np.sqrt(S + B)) if (S + B) > 0 else np.nan,
                     # Same rule as pid.evaluate.binned_table: a bin is quotable only
                     # if BOTH classes clear the floor. A row with zero background
                     # candidates has no defined fake rate, purity or rejection, so it
                     # must never be marked as merely "fine with NaN columns".
                     "insufficient_stats": bool(min(n_sig, n_bkg) < config.MIN_ENTRIES_PER_BIN
                                                or (n_sig + n_bkg)
                                                < config.MIN_CANDIDATES_PER_WP_BIN),
                     "reason": ("" if min(n_sig, n_bkg) >= config.MIN_ENTRIES_PER_BIN
                                else ("no background candidates in this bin" if n_bkg == 0
                                      else f"min(n_signal, n_background) = {min(n_sig, n_bkg)}"
                                          f" < {config.MIN_ENTRIES_PER_BIN}"))})
    return pd.DataFrame(rows)


def binomial_interval(k: int, n: int, cl: float = config.REJECTION_CL) -> tuple[float, float]:
    """Central (two-sided) Clopper-Pearson interval for k successes in n trials.

    Deliberately the *central* interval, matching :func:`pid.evaluate.garwood`, so
    that a column named ``<x>_err_lo`` / ``<x>_err_hi`` means the same thing in
    every table of this project. One-sided 95 % upper limits are a different
    quantity and appear only under names that say so
    (``fake_rate_upper_limit``, ``rejection_limit`` - see
    :func:`pid.significance.garwood_upper`).
    """
    from scipy.stats import beta

    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(min(max(int(k), 0), int(n)))
    alpha = (1.0 - cl) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha, k + 1, n - k))
    return (lo, hi)


def _lo(k: int, n: int, cl: float = config.REJECTION_CL) -> float:
    """Lower half of the central Clopper-Pearson interval (see above)."""
    return binomial_interval(k, n, cl)[0]


def _hi(k: int, n: int, cl: float = config.REJECTION_CL) -> float:
    """Upper half of the central Clopper-Pearson interval."""
    return binomial_interval(k, n, cl)[1]


# ---------------------------------------------------------------------------
# Significance vs threshold, per kinematic bin (the working-point finder, drawn)
# ---------------------------------------------------------------------------

def significance_curves_by_bin(df: pd.DataFrame, channel: str, *, variable: str = "pt",
                               bins: np.ndarray | None = None,
                               weights: np.ndarray | None = None,
                               mode: str = "s_over_sqrt",
                               lumi_scale: float = config.DEFAULT_LUMI_SCALE,
                               n_thresholds: int | None = None) -> pd.DataFrame:
    """FOM(c) in every bin of `variable`, with each bin's own optimum flagged.

    Returns one row per (bin, threshold). Two views are needed and they answer
    different questions, so both are produced by the plotting layer from this one
    table: the *absolute* FOM shows which regions of phase space carry the
    statistical power at all, and each curve divided by its own maximum shows how
    the optimum migrates with momentum without being dominated by the busiest bin.

    A bin holding only one species yields no curve (`has_curve=False`), which the
    figures render as a labelled gap rather than an interpolated line.
    """
    column, default_edges, axis_label, basis = resolve_bin_variable(variable, df)
    edges = np.asarray(bins if bins is not None else default_edges, dtype=float)
    score = df["score"].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    x = df[column].to_numpy(dtype=float)
    w_all = None if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(score) & np.isfinite(x)
    score, y, x = score[ok], y[ok], x[ok]
    w_all = None if w_all is None else w_all[ok]
    grid = (np.linspace(0.0, 1.0, int(n_thresholds)) if n_thresholds
            else np.linspace(0.0, 1.0, int(config.N_THRESHOLD_SCAN_BINNED)))
    idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
    spec = config.CHANNELS[channel]
    rows: list[pd.DataFrame] = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n_sig, n_bkg = int((sel & (y == 1)).sum()), int((sel & (y == 0)).sum())
        meta = {"channel": channel, "signal": spec["signal"], "background": spec["background"],
                "variable": variable, "binning_basis": basis, "bin_column": column,
                "bin": f"({edges[b]:g}, {edges[b + 1]:g}]",
                "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
                "n_signal_bin": n_sig, "n_background_bin": n_bkg}
        if n_sig == 0 or n_bkg == 0 or int(sel.sum()) < config.MIN_CANDIDATES_PER_WP_BIN:
            rows.append(pd.DataFrame([{**meta, "threshold": np.nan, "S": np.nan, "B": np.nan,
                                       "significance": np.nan, "significance_eff": np.nan,
                                       "efficiency": np.nan, "fake_rate": np.nan,
                                       "purity": np.nan, "is_optimum": False,
                                       "has_curve": False,
                                       "reason": (f"only {n_sig} signal / {n_bkg} background "
                                                  f"candidates in this bin")}]))
            continue
        table = significance.scan_thresholds(score[sel], y[sel], thresholds=grid,
                                   weights=None if w_all is None else w_all[sel],
                                   lumi_scale=lumi_scale)
        opt = significance.optimal_cut(table, mode=mode, min_candidates=1)
        curve = table.assign(**meta, is_optimum=np.isclose(table["threshold"],
                                                           opt["threshold"], atol=1e-12),
                             has_curve=True,
                             optimum_threshold=opt["threshold"],
                             optimum_significance=opt["significance"],
                             optimum_efficiency=opt["efficiency"],
                             optimum_fake_rate=opt["fake_rate"],
                             optimum_purity=opt["purity"],
                             optimum_caution=opt["caution"],
                             reason="")
        rows.append(curve)
    out = pd.concat(rows, ignore_index=True)
    wanted = ["channel", "signal", "background", "variable", "binning_basis",
              "bin_column", "bin", "bin_center",
              "n_signal_bin", "n_background_bin", "threshold", "S", "B",
              "significance", "significance_eff", "efficiency", "fake_rate", "purity",
              "is_optimum", "has_curve", "optimum_threshold", "optimum_significance",
              "optimum_efficiency", "optimum_fake_rate", "optimum_purity",
              "optimum_caution", "reason"]
    return out[[c for c in wanted if c in out.columns]]


_OPTIMUM_COLUMNS = ["channel", "variable", "binning_basis", "bin", "bin_center", "n_signal_bin",
                    "n_background_bin", "threshold", "significance", "efficiency",
                    "fake_rate", "purity", "optimum_caution"]


def bin_optima_from_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """One row per bin: the cut that maximises FOM there (derived from `curves`)."""
    if curves.empty or "is_optimum" not in curves.columns:
        return pd.DataFrame(columns=_OPTIMUM_COLUMNS)
    marked = curves[curves["is_optimum"].astype(bool) & curves["has_curve"].astype(bool)]
    return marked[[c for c in _OPTIMUM_COLUMNS if c in marked.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2D maps
# ---------------------------------------------------------------------------

def map_tidy(m: dict) -> pd.DataFrame:
    """Long-format version of a 2D map: one row per (pT, eta) cell.

    The plotting layer needs the matrix, but the *record* of it must be queryable
    (a 2D block in a JSON is unreadable and cannot go into a TNtuple), so each cell
    becomes a row with its own statistics flags.
    """
    pt_c = 0.5 * (m["pt_edges"][:-1] + m["pt_edges"][1:])
    et_c = 0.5 * (m["eta_edges"][:-1] + m["eta_edges"][1:])
    rows = []
    for i, ptc in enumerate(pt_c):
        for j, etc in enumerate(et_c):
            value = float(m["value"][i, j])
            rows.append({"channel": m["channel"], "quantity": m["quantity"],
                         "cut": m["cut"], "x_variable": m.get("x_variable", "pt"),
                         "y_variable": m.get("y_variable", "eta"),
                         "pt_bin_center": float(ptc),
                         "eta_bin_center": float(etc), "n_candidates": float(m["counts"][i, j]),
                         m["quantity"]: value,
                         "insufficient_stats": bool(not np.isfinite(value))})
    return pd.DataFrame(rows)


def map_2d(df: pd.DataFrame, channel: str, *, cut: float, quantity: str = "efficiency",
           x_variable: str = "pt", y_variable: str = "eta",
           pt_bins=None, eta_bins=None) -> dict:
    """(pT, η) matrix of ``efficiency`` / ``fake_rate`` / ``purity`` / counts.

    Empty and statistics-starved cells are NaN (masked, never zero) so a
    calorimeter transition shows up as a *hole or a dip* against populated
    neighbours rather than as a fabricated 0 % efficiency.
    """
    x_column, x_edges_default, x_label, x_basis = resolve_bin_variable(x_variable, df)
    y_column, y_edges_default, y_label, y_basis = resolve_bin_variable(y_variable, df)
    pt_bins = x_edges_default if pt_bins is None else np.asarray(pt_bins)
    eta_bins = y_edges_default if eta_bins is None else np.asarray(eta_bins)
    y = df["y"].to_numpy()
    sc = df["score"].to_numpy(dtype=float)
    pt = df[x_column].to_numpy(dtype=float)
    eta = df[y_column].to_numpy(dtype=float)
    ok = np.isfinite(sc) & np.isfinite(pt) & np.isfinite(eta)
    y, sc, pt, eta = y[ok], sc[ok], pt[ok], eta[ok]
    ib = np.clip(np.digitize(pt, pt_bins[1:-1], right=True), 0, len(pt_bins) - 2)
    je = np.clip(np.digitize(eta, eta_bins[1:-1], right=True), 0, len(eta_bins) - 2)
    value = np.full((len(pt_bins) - 1, len(eta_bins) - 1), np.nan)
    counts = np.zeros_like(value, dtype=float)
    pass_flag = sc >= cut
    for i in range(len(pt_bins) - 1):
        for j in range(len(eta_bins) - 1):
            sel = (ib == i) & (je == j)
            n_sig, n_bkg = int((sel & (y == 1)).sum()), int((sel & (y == 0)).sum())
            counts[i, j] = n_sig + n_bkg
            if n_sig + n_bkg < config.MIN_CANDIDATES_PER_WP_BIN:
                continue
            k_sig = int((sel & pass_flag & (y == 1)).sum())
            k_bkg = int((sel & pass_flag & (y == 0)).sum())
            if quantity == "efficiency":
                value[i, j] = k_sig / n_sig if n_sig else np.nan
            elif quantity == "fake_rate":
                value[i, j] = k_bkg / n_bkg if n_bkg else np.nan
            elif quantity == "purity":
                value[i, j] = k_sig / (k_sig + k_bkg) if (k_sig + k_bkg) else np.nan
            elif quantity == "significance":
                value[i, j] = (k_sig / np.sqrt(k_sig + k_bkg)) if (k_sig + k_bkg) else np.nan
            else:
                raise ValueError(f"unknown map quantity {quantity!r}")
    return {"value": value, "counts": counts,
            "pt_edges": np.asarray(pt_bins), "eta_edges": np.asarray(eta_bins),
            "x_label": x_label, "y_label": y_label,
            "x_variable": x_variable, "y_variable": y_variable,
            "cut": float(cut), "quantity": quantity, "channel": channel,
            "masked_cells": int(np.isnan(value).sum()), "total_cells": int(value.size)}


# ---------------------------------------------------------------------------
# n-sigma vs momentum
# ---------------------------------------------------------------------------

def nsigma_vs_p(df: pd.DataFrame, channel: str, *, p_bins=None,
                variable: str = "p", nsigma_method: str | None = None) -> pd.DataFrame:
    """Separation power vs momentum, bin by bin, for one channel.

    The per-bin estimator is :func:`pid.evaluate.measure_separation`, so the
    default is the closed-form quantile estimator (no fit to diverge) with the
    logit-space Gaussian fit kept alongside as a cross-check. Both are reported
    per bin; a bin where neither converges yields NaN plus a reason, never a
    width borrowed from the [0,1] clip.
    """
    column, default_edges, axis_label, basis = resolve_bin_variable(variable, df)
    edges = default_edges if p_bins is None else np.asarray(p_bins)
    p = df[column].to_numpy(dtype=float)
    sc = df["score"].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    ok = np.isfinite(p) & np.isfinite(sc)
    p, sc, y = p[ok], sc[ok], y[ok]
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n_sig, n_bkg = int((sel & (y == 1)).sum()), int((sel & (y == 0)).sum())
        row = {"channel": channel, "signal": config.CHANNELS[channel]["signal"],
               "background": config.CHANNELS[channel]["background"],
               "variable": variable, "binning_basis": basis, "bin_column": column,
               "bin": f"({edges[b]:g}, {edges[b + 1]:g}]",
               "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
               "n_signal": n_sig, "n_background": n_bkg}
        if n_sig == 0 or n_bkg == 0:
            row.update({"auc": np.nan, "n_sigma": np.nan, "n_sigma_quantile": np.nan,
                        "n_sigma_logit": np.nan, "n_sigma_lower_limit": np.nan,
                        "nsigma_method": "", "saturated": False,
                        "mu_signal": np.nan, "sigma_signal": np.nan,
                        "mu_background": np.nan, "sigma_background": np.nan,
                        "converged": False, "insufficient_stats": True,
                        "reason": "one class has no candidates in this momentum bin"})
            rows.append(row)
            continue
        sep = evaluate.measure_separation(sc[sel & (y == 1)], sc[sel & (y == 0)],
                                          method=nsigma_method)
        fs, fb = sep.get("fit_signal") or {}, sep.get("fit_background") or {}
        row.update({"auc": sep["auc"], "n_sigma": sep["n_sigma"],
                    "n_sigma_quantile": sep["n_sigma_quantile"],
                    "n_sigma_logit": sep["n_sigma_logit"],
                    "n_sigma_lower_limit": sep.get("n_sigma_lower_limit", np.nan),
                    "nsigma_method": sep["nsigma_method"], "saturated": sep["saturated"],
                    "mu_signal": fs.get("mu", np.nan), "sigma_signal": fs.get("sigma", np.nan),
                    "mu_background": fb.get("mu", np.nan),
                    "sigma_background": fb.get("sigma", np.nan),
                    "converged": bool(np.isfinite(sep["n_sigma"])
                                      or np.isfinite(sep.get("n_sigma_lower_limit", np.nan))),
                    "insufficient_stats": bool(min(n_sig, n_bkg) < config.MIN_ENTRIES_PER_BIN
                                               or not np.isfinite(sep["n_sigma"])),
                    "reason": sep["reason"]})
        rows.append(row)
    return pd.DataFrame(rows)


#: Category for candidates that clear no class threshold under the at-working-point
#: decision rule (:func:`class_working_points`). Kept as a column rather than being
#: dropped: "the PID declines to identify this track" is a result, not missing data.
UNASSIGNED = "none"


def class_working_points(df: pd.DataFrame, classes: list[str], *,
                         mode: str = "s_over_sqrt",
                         lumi_scale: float = config.DEFAULT_LUMI_SCALE,
                         bkg_scale: float = 1.0) -> dict[str, dict]:
    """One-vs-rest maximum-significance cut for each class of a multiclass table.

    Needed because "apply the optimal threshold" is not defined for a softmax: the
    classes have different scores, different abundances and therefore different
    optimal cuts. Each class c gets its own c*_c from the scan of `proba_c`
    against everything that is not c, and the resulting decision rule is:

        predict the class with the highest probability AMONG those clearing c*_c,
        else leave the candidate UNASSIGNED.

    The unassigned category is the point: a PID selection in an analysis is allowed
    (and often required) to decline to identify a track, and a confusion matrix
    without that column silently invents a species for every low-quality candidate.
    """
    out: dict[str, dict] = {}
    for cls in classes:
        column = f"proba_{cls}"
        if column not in df.columns:
            continue
        truth = df["truth_class"].to_numpy()
        y = (truth == cls).astype(int)
        score = df[column].to_numpy(dtype=float)
        if y.sum() == 0 or y.sum() == len(y):
            out[cls] = {"threshold": float("nan"), "valid": False,
                        "reason": "class is not separable one-vs-rest in this sample"}
            continue
        w = None
        if bkg_scale != 1.0:
            # Same composition rule as the channel scans: rescale the
            # one-vs-rest background, recorded with the cut.
            w = np.ones(len(y), dtype=float)
            w[y == 0] = float(bkg_scale)
        table = significance.scan_thresholds(score, y, thresholds=None,
                                            lumi_scale=lumi_scale, weights=w)
        opt = significance.optimal_cut(table, mode=mode)
        out[cls] = {**opt, "score_column": column}
    return out


def _finite_or_nan(value) -> float:
    """A cut threshold where None/NaN means "no cut".

    A threshold of exactly 0.0 is a legitimate cut (accept everything above
    it); the ``or np.nan`` idiom would silently convert it to missing and the
    whole class would collapse to UNASSIGNED downstream.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def confusion_matrix(df: pd.DataFrame, cuts: dict[str, dict] | None = None) -> dict:
    """Multi-class confusion matrix from a `proba_*` table, held-out rows only.

    Parameters
    ----------
    cuts:
        Optional ``{class: {"threshold": c*_c}}`` from :func:`class_working_points`.
        When given, the decision rule is "assign the most probable class among
        those clearing **their own** optimal cut", and a candidate clearing none is
        placed in the :data:`UNASSIGNED` column. Without it the matrix uses the
        plain argmax rule, which is the comparison a reader expects - the two are
        reported side by side precisely because the difference between them *is*
        the cost of demanding a significance-optimal cut per class.

    Returns
    -------
    dict with ``matrix`` (row-normalised), ``counts``, ``truth_classes``,
    ``classes`` (the prediction categories), ``decision``, ``cuts``,
    ``n_rows_used``, ``n_rows_excluded``, ``excluded_classes``, ``n_truth``.

    Notes
    -----
    Rows whose truth species has no `proba_<class>` column are dropped, and that
    is reported: a matrix that quietly shrinks invites the reader to normalise by
    the wrong total.
    """
    empty = {"matrix": np.zeros((0, 0)), "counts": np.zeros((0, 0), dtype=int),
             "classes": [], "truth_classes": [], "n_truth": np.zeros(0, dtype=int),
             "decision": "argmax", "cuts": None, "n_rows_used": 0,
             "n_rows_excluded": 0, "excluded_classes": []}
    truth_classes = evaluate.class_order(df)
    if not truth_classes or len(df) == 0:
        return empty

    truth = np.asarray(df["truth_class"].to_numpy(), dtype=object)
    keep = [i for i, c in enumerate(truth_classes) if (truth == c).any()]
    truth_classes = [truth_classes[i] for i in keep]
    proba = np.column_stack([df[f"proba_{c}"].to_numpy(dtype=float) for c in truth_classes])
    proba = np.asarray(proba, dtype=float)

    if cuts:
        threshold = np.array([_finite_or_nan((cuts.get(c) or {}).get("threshold"))
                              for c in truth_classes], dtype=float)
        if not np.isfinite(threshold).any():
            raise ValueError(
                "pid.performance.confusion_matrix: none of the supplied class cuts is "
                "finite, so the threshold decision rule cannot be applied")
        cleared = proba >= np.where(np.isfinite(threshold), threshold, np.inf)
        masked = np.where(cleared, proba, -np.inf)
        pred = np.asarray(truth_classes, dtype=object)[np.argmax(masked, axis=1)]
        pred[~cleared.any(axis=1)] = UNASSIGNED
        pred_classes = [*truth_classes, UNASSIGNED]
        decision = "thresholds_at_c*"
    else:
        pred = np.asarray(truth_classes, dtype=object)[np.nanargmax(proba, axis=1)]
        pred_classes = list(truth_classes)
        decision = "argmax"

    in_scope = np.isin(truth, truth_classes)
    counts = np.zeros((len(truth_classes), len(pred_classes)), dtype=int)
    t_codes = np.asarray(pd.Categorical(pd.Series(truth[in_scope]),
                                        categories=truth_classes).codes, dtype=int)
    p_codes = np.asarray(pd.Categorical(pd.Series(pred[in_scope]),
                                        categories=pred_classes).codes, dtype=int)
    valid = (t_codes >= 0) & (p_codes >= 0)
    np.add.at(counts, (t_codes[valid], p_codes[valid]), 1)

    row_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_norm = np.where(row_totals > 0, counts / np.maximum(row_totals, 1), np.nan)

    n_used = int(counts.sum())
    excluded = sorted({str(c) for c in pd.Series(truth).dropna().unique()}
                      - set(truth_classes))
    if n_used != len(df):
        print(f"[performance] NOTE: confusion matrix ({decision}) uses {n_used}/"
              f"{len(df)} rows; excluded truth classes {excluded} "
              "(no matching proba_<class> column)", file=sys.stderr)
    return {"matrix": row_norm, "counts": counts, "classes": pred_classes,
            "truth_classes": truth_classes, "n_truth": counts.sum(axis=1),
            "decision": decision,
            "cuts": ({k: (v or {}).get("threshold") for k, v in cuts.items()} if cuts
                     else None),
            "n_rows_used": n_used, "n_rows_excluded": int(len(df) - n_used),
            "excluded_classes": excluded}


def _methods_used(nsigma_table) -> str:
    """Comma-joined n_sigma estimators actually used in a per-bin table."""
    if nsigma_table is None or not len(nsigma_table) or "nsigma_method" not in nsigma_table:
        return ""
    return ",".join(sorted({str(m) for m in nsigma_table["nsigma_method"].dropna()}))


def confusion_tidy(conf: dict) -> pd.DataFrame:
    """Long-format confusion matrix: one row per (true, predicted) cell."""
    rows = []
    counts, matrix = conf["counts"], conf["matrix"]
    for i, t in enumerate(conf["truth_classes"]):
        for j, p in enumerate(conf["classes"]):
            rows.append({"truth_class": t, "pred_class": p, "n": int(counts[i, j]),
                         "row_fraction": float(matrix[i, j]),
                         "n_truth": int(conf["n_truth"][i]),
                         "decision": conf.get("decision", "argmax"),
                         "insufficient_stats": bool(conf["n_truth"][i]
                                                    < config.MIN_ENTRIES_PER_BIN)})
    return pd.DataFrame(rows)


def manifest_table(manifests: list[dict]) -> pd.DataFrame:
    """Flatten manifest dicts into one row per (channel, basis, required figure).

    The in-memory manifest keeps its nested lists, but the artefact must survive
    `report.write`, whose ROOT branch rejects non-scalar cells - and a flat table
    is what a human actually reads when a figure is missing.
    """
    rows = []
    for mf in manifests:
        have = set(mf["present"])
        reasons = {m["figure"]: m["reason"] for m in mf["missing"]}
        shared = set(mf.get("shared_with_other_basis", []))
        for key, label in config.REQUIRED_FIGURES:
            suffix = "" if mf["binning_suffix"] == "reco" else mf["binning_suffix"]
            name = f"{key}{suffix}"
            is_shared = key in shared
            rows.append({"channel": mf["channel"], "status": mf["status"],
                         "binning_basis": mf["binning_suffix"], "figure": name,
                         "label": label,
                         "present": bool(is_shared or key in have or name in have),
                         "shared_with_reco_run": bool(is_shared),
                         "reason": ("" if is_shared else
                                    (reasons.get(key, "") or reasons.get(name, "")))})
    return pd.DataFrame(rows)


def verify_deliverables(written: dict[str, str], *, channel: str, status: str,
                        has_working_point: bool, binned_curves: bool = True,
                        suffix: str = "", shared: tuple[str, ...] = (),
                        notes: dict[str, str] | None = None) -> dict:
    """Check every required figure exists on disk; report what is missing and why.

    A missing figure is usually a statistics verdict, not a bug (no valid c*, a bin
    with one class empty), so the reason is required in the report rather than
    leaving a silent hole in the deliverables.
    """
    missing, present, shared_hits = [], [], []
    for key, label in config.REQUIRED_FIGURES:
        # the electron channel ships the rejection figure under the physics name
        # pion_rejection_vs_p; both satisfy the requirement (including the
        # shared-basis check below).
        alt = {"rejection_vs_p": "pion_rejection_vs_p"}.get(key)
        if suffix and key in config.BASIS_INDEPENDENT_FIGURES and (
                key in written or key in shared
                or (alt is not None and (alt in written or alt in shared))):
            # Written once per channel under the reconstructed basis, so it counts
            # as delivered here too - and is listed separately so the table never
            # implies this run itself drew it.
            shared_hits.append(key)
            present.append(key)
            continue
        path = written.get(key) or (written.get(alt) if alt else None)
        if not has_working_point and key in ("roc", "eff_vs_pt", "misid_vs_pt",
                                            "purity_vs_pt", "eff_map_pt_vs_eta",
                                            "fake_map_pt_vs_eta", "rejection_vs_p"):
            missing.append({"figure": key + suffix, "label": label,
                            "reason": "needs a valid working point c*; none exists "
                                      "because the FOM maximum is degenerate"})
            continue
        if not binned_curves and key.startswith("significance_vs_cut_"):
            missing.append({"figure": key + suffix, "label": label,
                            "reason": "per-bin curves disabled (--no-binned-curves)"})
            continue
        if path and os.path.exists(path):
            present.append(key)
        elif notes and key in notes:
            # a run-specific cause beats the generic advice below
            missing.append({"figure": key + suffix, "label": label,
                            "reason": notes[key]})
        else:
            missing.append({"figure": key + suffix, "label": label,
                            "reason": "not produced by this run - check the run log "
                                      "(empty table, one-class bin, or a skipped "
                                      "figure branch)"})
    return {"channel": channel, "status": status, "binning_suffix": suffix or "reco",
            "shared_with_other_basis": shared_hits,
            "n_required": len(config.REQUIRED_FIGURES), "n_present": len(present),
            "present": present, "missing": missing,
            "complete": bool(not missing)}


def overtraining_check(df_all: pd.DataFrame, channel: str, *, n_bins: int = 40) -> dict:
    """Train-vs-test score comparison per class (memorisation diagnostic)."""
    from scipy.stats import ks_2samp

    if "sample" not in df_all.columns:
        raise KeyError("pid.performance.overtraining_check: table has no 'sample' column "
                       "(retrain with pid.train to write all_scores.pkl)")
    out = {"channel": channel, "classes": {}, "ks": {}, "train_rows": 0, "test_rows": 0}
    # Count what is actually plotted: channel_frame filters pair channels to
    # their hypothesis classes, so raw table counts would overstate the
    # histograms' denominators.
    try:
        framed_all = channel_frame(df_all, channel, model_dir="all_scores")
    except KeyError:
        framed_all = None
    base = framed_all if framed_all is not None else df_all
    for label_text in ("train", "test"):
        sub = base[base["sample"] == label_text]
        out[f"{label_text}_rows"] = int(len(sub))
    for cls_value, cls_name in ((1, config.CHANNELS[channel]["signal"]),
                                (0, config.CHANNELS[channel]["background"])):
        per = {}
        for label_text in ("train", "test"):
            sub = df_all[(df_all["sample"] == label_text)]
            try:
                ch = channel_frame(sub, channel, model_dir="all_scores")
            except KeyError:
                continue
            s = ch.loc[ch["y"] == cls_value, "score"].to_numpy(dtype=float)
            per[label_text] = s
        if "train" in per and "test" in per and per["train"].size > 5 and per["test"].size > 5:
            ks = ks_2samp(per["train"], per["test"])
            out["ks"][cls_name] = {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)}
            bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
            for label_text in ("train", "test"):
                hist, _ = np.histogram(per[label_text], bins=bins)
                out["classes"].setdefault(cls_name, {})[label_text] = {
                    "counts": hist.astype(float).tolist(), "edges": bins.tolist(),
                    "n": int(per[label_text].size)}
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _model_dir(override: str, channel: str, model: str, tag: str, root: str = "") -> str:
    """Directory holding one task's artifacts.

    `override` points at a single task's directory; `root` relocates the whole
    model tree (used by the tests, and for comparing a run kept elsewhere) without
    silently falling back to the repository's own artifacts.
    """
    if override:
        return override
    base = root or config.MODEL_DIR
    return os.path.join(base, f"{config.CHANNELS[channel]['task']}_{model}_{tag}")


def run(*, channels: tuple[str, ...] = ("eid", "ehad", "Kpi", "pK"),
        dataset_tag: str, model: str = config.MODEL_LIBRARY_DEFAULT,
        out_dir: str = config.OUTPUT_DIR, plots_dir: str | None = None,
        model_root: str = "", scores_map: dict[str, str] | None = None,
        binned_curves: bool = True, n_thresholds: int | None = None,
        bin_source: str = "reco", nsigma_method: str | None = None,
        require_figures: bool = False,
        mode: str = "s_over_sqrt", lumi_scale: float = config.DEFAULT_LUMI_SCALE,
        bkg_scale: float = 1.0, write: bool = True, quiet: bool = False) -> dict:
    """Produce the whole maximum-significance performance package.

    ``bkg_scale`` realises the "realistic species fraction" option: it rescales the
    background yield relative to the measured composition (1.0 = as measured), and
    is recorded in every output because the optimum depends on it.
    """
    plots_dir = plots_dir or os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if bin_source not in ("reco", "truth", "both"):
        raise SystemExit(f"pid performance: --bin-source must be reco|truth|both, got {bin_source!r}")
    bases = ["reco", "truth"] if bin_source == "both" else [bin_source]
    summary, artefacts = [], {}
    seen_tasks: set[str] = set()
    #: n-sigma tables are collected per TASK so K/pi and p/K share one figure
    #: (that comparison is the point of the plot), rather than one per channel.
    nsigma_by_task: dict[str, list[pd.DataFrame]] = {}

    for channel in channels:
        directory = (scores_map or {}).get(channel) or _model_dir(
            "", channel, model, dataset_tag, root=model_root)
        all_scores_path = os.path.join(directory, "all_scores.pkl")
        test_scores_path = os.path.join(directory, "test_scores.pkl")
        path = all_scores_path if os.path.exists(all_scores_path) else test_scores_path
        if not os.path.exists(path):
            if not quiet:
                print(f"[performance/{channel}] SKIPPED: no score table under {directory} "
                      "(run `python -m pid train` for this channel)")
            continue
        table = dataset.load_table(path)
        test_only = table[table["sample"] == "test"] if "sample" in table.columns else table
        try:
            frame = channel_frame(test_only, channel, model_dir=directory)
        except (KeyError, ValueError) as exc:
            if not quiet:
                print(f"[performance/{channel}] SKIPPED: {exc}")
            continue
        if frame.empty:
            if not quiet:
                print(f"[performance/{channel}] SKIPPED: no rows for this hypothesis pair")
            continue

        weights = None if bkg_scale == 1.0 else significance.flux_weights(
            frame["y"].to_numpy(), signal=1, background=0, bkg_scale=bkg_scale)
        wp = working_point(frame, channel, mode=mode, weights=weights, lumi_scale=lumi_scale)
        opt, scan = wp["optimum"], wp["scan"]
        global_cut = opt["threshold"] if opt.get("valid") else None

        # Basis-independent artifacts first: the FOM scan and the optimum do not
        # depend on how the sample is binned.
        base_tables = {"significance_vs_cut": scan.assign(channel=channel,
                                                          binning_basis="none")}
        for basis in bases:
            names = BIN_BASES[basis]
            suffix = "" if basis == "reco" else "_truthpt"
            try:
                per_bin = working_points_vs_pt(frame, channel, weights=weights, mode=mode,
                                               lumi_scale=lumi_scale, variable=names["pt"],
                                               n_thresholds=n_thresholds)
                tables = {
                    **{k + suffix: v for k, v in base_tables.items()},
                    "optimal_cut_vs_pt" + suffix: per_bin,
                    "at_cut_global" + suffix: at_cut_table(frame, channel, cut=global_cut,
                                                           variable=names["pt"], weights=weights),
                    "at_cut_per_bin" + suffix: at_cut_table(frame, channel, cut_per_bin=per_bin,
                                                            variable=names["pt"], weights=weights),
                    "rejection_vs_p" + suffix: at_cut_table(frame, channel, cut=global_cut,
                                                            variable=names["p"], weights=weights),
                    "efficiency_vs_eta" + suffix: at_cut_table(frame, channel, cut=global_cut,
                                                               variable=names["eta"],
                                                               weights=weights),
                    "nsigma_vs_p" + suffix: nsigma_vs_p(frame, channel, variable=names["p"],
                                                        nsigma_method=nsigma_method),
                }
            except KeyError as exc:  # older score table without truth columns
                print(f"[performance/{channel}/{basis}] SKIPPED: {exc}", file=sys.stderr)
                continue
            if binned_curves:
                tables["significance_by_pt" + suffix] = significance_curves_by_bin(
                    frame, channel, variable=names["pt"], weights=weights, mode=mode,
                    lumi_scale=lumi_scale, n_thresholds=n_thresholds)
            maps = {}
            if global_cut is not None:
                for q in ("efficiency", "fake_rate"):
                    maps[q] = map_2d(frame, channel, cut=global_cut, quantity=q,
                                     x_variable=names["pt"], y_variable=names["eta"])

            # The multi-class confusion matrix describes the TASK (pi/K/p), not one
            # hypothesis pair, and uses held-out rows only: a matrix built from
            # train+test would fold in-sample memorisation into a performance table.
            conf = None
            spec = config.CHANNELS[channel]
            status = str(spec.get("status", "headline"))
            status_note = str(spec.get("status_note", ""))
            if spec["task"] == "hadpid" and spec["task"] not in seen_tasks:
                conf = confusion_matrix(test_only)
            seen_tasks.add(spec["task"])

            # Per-class one-vs-rest working points, so the confusion matrix can be
            # produced under the same "maximize significance" rule as everything
            # else instead of the threshold-free argmax.
            conf_at_cut = None
            if spec["task"] == "hadpid":
                try:
                    cls_cuts = class_working_points(test_only, evaluate.class_order(test_only),
                                                    mode=mode, lumi_scale=lumi_scale,
                                                    bkg_scale=bkg_scale)
                    conf_at_cut = confusion_matrix(test_only, cuts=cls_cuts)
                except (KeyError, ValueError) as exc:
                    print(f"[performance/{channel}] at-cut confusion unavailable: {exc}",
                          file=sys.stderr)

            over = None
            figure_notes: dict[str, str] = {}
            # The notes describe the artifact directory (same scores for both
            # bases), so they apply to the truth run too - not just reco.
            if not os.path.exists(all_scores_path):
                figure_notes["score_dist_train_vs_test"] = (
                    "needs all_scores.pkl (train and test rows); this artifact "
                    "directory only has test_scores.pkl - retrain with the current "
                    "pid.train to write both")
            if os.path.exists(all_scores_path) and basis == "reco":
                try:
                    over = overtraining_check(table, channel)
                except (KeyError, ValueError) as exc:  # pragma: no cover
                    figure_notes["score_dist_train_vs_test"] = (
                        f"overtraining check failed: {exc}")
                    print(f"[performance/{channel}] overtraining check unavailable: {exc}",
                          file=sys.stderr)
                if over is not None and not over.get("classes"):
                    figure_notes["score_dist_train_vs_test"] = (
                        "too few held-out candidates per class to compare train vs test "
                        "distributions (needs > 5 of each in both samples)")

            # keyed by (task, basis) so the truth-binned figure does not plot
            # alongside reconstructed-binned curves on the same axis.
            nsigma_by_task.setdefault((spec["task"], basis), []).append(
                tables["nsigma_vs_p" + suffix])

            if write:
                stem_base = os.path.join(out_dir, f"pid-{channel}_{model}_{dataset_tag}")
                for name, t in tables.items():
                    report.write(t, f"{stem_base}-{name}",
                                 tree_name=("pid_" + channel + "_" + name).replace("-", "_")[:31],
                                 meta={"channel": channel, "task": spec["task"],
                                       "model": model, "dataset_tag": dataset_tag,
                                       "significance_mode": mode, "lumi_scale": lumi_scale,
                                       "bkg_scale": bkg_scale,
                                       "binning_basis": basis,
                                       "optimal_cut": global_cut,
                                       "n_rows_channel": int(len(frame))})
                for q, m in maps.items():
                    long_name = ("eff_map_pt_vs_eta" if q == "efficiency"
                                 else "fake_map_pt_vs_eta") + suffix
                    report.write(map_tidy(m), f"{stem_base}-{long_name}",
                                 tree_name=("pid_" + channel + "_" + long_name).replace("-", "_")[:31],
                                 meta={"channel": channel, "model": model,
                                       "dataset_tag": dataset_tag, "quantity": q,
                                       "binning_basis": basis, "cut": float(m["cut"]),
                                       "masked_cells": m["masked_cells"],
                                       "total_cells": m["total_cells"]})
                if conf_at_cut is not None:
                    report.write(confusion_tidy(conf_at_cut),
                                 f"{stem_base}-confusion_matrix_atcut",
                                 tree_name="pid_confusion_atcut",
                                 meta={"channel": channel, "task": spec["task"],
                                       "model": model, "dataset_tag": dataset_tag,
                                       "decision": conf_at_cut.get("decision"),
                                       "class_cuts": conf_at_cut.get("cuts")})
                figures = plots.performance_package(
                    channel=channel, scan=scan, optimum=opt, per_bin=per_bin,
                    tables={k.replace(suffix, "") if k.endswith(suffix) and suffix else k: v
                            for k, v in tables.items()},
                    maps=maps, confusion=conf, overtraining=over, out_dir=plots_dir,
                    model=model, dataset_tag=dataset_tag, frame=frame, suffix=suffix,
                    write_basis_independent=(basis == "reco"), cut=global_cut,
                    confusion_at_cut=conf_at_cut)
                artefacts.setdefault(channel, {}).update(
                    {f"{basis}:{k}": v for k, v in tables.items()})
                artefacts[channel].setdefault("figures", {}).update(figures)
                artefacts[channel]["maps"] = maps
                # Basis-independent figures are written once per channel (under
                # the reconstructed run); later bases must count them as
                # satisfied rather than missing.
                shared = tuple(sorted(artefacts[channel].get("shared_figures", ())))
                manifest = verify_deliverables(
                    figures, channel=channel, status=status,
                    has_working_point=bool(global_cut is not None
                                           and np.isfinite(global_cut)),
                    binned_curves=binned_curves, suffix=suffix, shared=shared,
                    notes=figure_notes)
                if basis == "reco":
                    artefacts[channel]["shared_figures"] = tuple(figures.keys())
                artefacts[channel].setdefault("manifests", []).append(manifest)
                if not manifest["complete"]:
                    print(f"[deliverables/{channel}{suffix}] "
                          f"{manifest['n_present']}/{manifest['n_required']} present; missing: "
                          + "; ".join(f"{m['figure']} ({m['reason']})"
                                      for m in manifest["missing"]), file=sys.stderr)
                else:
                    extra = (f" ({len(manifest['shared_with_other_basis'])} shared with the "
                             f"reconstructed basis)") if manifest.get("shared_with_other_basis") else ""
                    print(f"[deliverables/{channel}{suffix}] complete: "
                          f"{manifest['n_required']}/{manifest['n_required']} figures{extra}")

            summary.append({**{k: opt.get(k) for k in ("threshold", "significance", "efficiency",
                                                       "fake_rate", "fake_rate_upper_limit",
                                                       "purity", "rejection",
                                                       "rejection_lower_limit",
                                                       "n_signal_total", "n_background_total",
                                                       "valid", "caution", "edge_flag")},
                            "rejects_background": opt.get("rejects_background", False),
                            "signal": opt["signal"], "background": opt["background"],
                            "channel": channel, "task": spec["task"],
                            "status": status, "status_note": status_note,
                            "nsigma_method": _methods_used(tables.get("nsigma_vs_p" + suffix)),
                            "score_space": opt["score_space"],
                            "binning_basis": basis,
                            "n_files": int(pd.Series(frame["file_id"]).nunique())
                            if "file_id" in frame else None,
                            "n_rows": int(len(frame)),
                            "n_bins_with_cut": int(per_bin["valid"].sum())
                            if "valid" in per_bin else 0,
                            "ks_signal": (over or {}).get("ks", {}).get(opt["signal"], {})
                            .get("statistic", np.nan),
                            "ks_background": (over or {}).get("ks", {}).get(opt["background"], {})
                            .get("statistic", np.nan)})


    # Task-level combined figures (written after the loop, once every channel of
    # the task has contributed its n-sigma table).
    if write:
        for (task, basis), tables_list in nsigma_by_task.items():
            usable = [t for t in tables_list if len(t)]
            if not usable:
                continue
            suffix = "" if basis == "reco" else "_truthpt"
            path = os.path.join(plots_dir, f"pid-{task}_nsigma_vs_p{suffix}.png")
            try:
                plots.nsigma_vs_p_table(usable, path,
                                        title=f"separation power vs momentum, {basis}-binned "
                                              f"[{model} / {dataset_tag}]")
            except SystemExit as exc:  # pragma: no cover - empty matrix guard
                print(f"[performance/{task}] n-sigma figure skipped: {exc}", file=sys.stderr)

    summary_df = pd.DataFrame(summary)
    if write and not summary_df.empty and artefacts:
        mt = manifest_table([m for ch in artefacts.values()
                             for m in ch.get("manifests", [])])
        report.write(mt, os.path.join(out_dir, f"pid-deliverables_{model}_{dataset_tag}"),
                     tree_name="pid_deliverables",
                     meta={"model": model, "dataset_tag": dataset_tag,
                           "n_required": len(config.REQUIRED_FIGURES),
                           "n_missing": int((~mt["present"].astype(bool)).sum())})
        if int((~mt["present"].astype(bool)).sum()):
            print(f"[deliverables] {int((~mt['present'].astype(bool)).sum())} of "
                  f"{len(mt)} required figure-slots missing; see "
                  f"pid-deliverables_{model}_{dataset_tag}.md for the reasons")
    incomplete = [m for ch in artefacts.values()
                  for mf in ch.get("manifests", []) for m in mf["missing"]]
    if require_figures and incomplete:
        raise SystemExit(
            "pid performance: --require-figures was set but "
            f"{len(incomplete)} deliverable figure(s) are missing: "
            + ", ".join(str(m["figure"]) for m in incomplete)
            + ". Each carries a reason (usually no valid c* or a disabled curve "
              "family); fix the sample or drop --require-figures.")
    if write and not summary_df.empty:
        report.write(summary_df, os.path.join(out_dir, f"pid-working_points_{model}_{dataset_tag}"),
                     tree_name="pid_working_points",
                     meta={"model": model, "dataset_tag": dataset_tag,
                           "significance_mode": mode, "lumi_scale": lumi_scale,
                           "bkg_scale": bkg_scale,
                           "channels": list(channels), "bin_source": bin_source,
                           "nsigma_method": nsigma_method or config.NSIGMA_METHOD,
                           "binning_bases_in_rows": bases,
                           "pt_bins": len(config.PT_BINS_PERFORMANCE) - 1,
                           "p_bins": len(config.P_BINS_PERFORMANCE) - 1})
    if not quiet and not summary_df.empty:
        cols = [c for c in ("channel", "status", "binning_basis", "threshold",
                            "significance", "efficiency", "fake_rate", "purity",
                            "rejection", "rejection_lower_limit", "n_background_total",
                            "caution") if c in summary_df.columns]
        print("\n[performance] maximum-significance working points")
        print(summary_df[cols].round(4).to_string(index=False))
    return {"summary": summary_df, "artefacts": artefacts}
