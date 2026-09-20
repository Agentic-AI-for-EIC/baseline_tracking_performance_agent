"""PID figures: ROC, efficiency vs fake rate, n-sigma vs p, importance.

Plain matplotlib, written to ``output/plots/`` next to the tables they come from,
so a figure and its numbers always correspond to the same JSON. Physics-style
axes (log fake-rate axis, pT on a log x axis) because the quantities span orders
of magnitude; every panel states the sample sizes it was made from, since a
1e-4 rejection drawn from 400 pions is a hint, not a measurement.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402


def _finish(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def roc(roc_table: pd.DataFrame, path: str, *, title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fx = np.clip(roc_table["fake_rate"].to_numpy(float), 1e-8, None)
    ax.plot(fx, roc_table["efficiency"], lw=1.6,
            label=f"{roc_table['signal'].iloc[0]} vs {roc_table['against'].iloc[0]}"
            if len(roc_table) else "")
    ax.set_xscale("log")
    ax.set_xlabel("fake rate (background acceptance)")
    ax.set_ylabel("efficiency")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    for target in config.FAKE_RATE_TARGETS:
        ax.axvline(target, ls=":", lw=0.8, color="grey")
        ax.text(target, 0.02, f"{target:g}", rotation=90, fontsize=6, color="grey",
                ha="right")
    if title:
        ax.set_title(title)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower right", fontsize=8)
    return _finish(fig, path)


def efficiency_vs_fake(overall: pd.DataFrame, path: str, *, title: str = "") -> str:
    """Efficiency at each quoted fake-rate working point (the headline figure)."""
    eff = overall[overall["quantity"] == "efficiency"].copy()
    if eff.empty:
        raise SystemExit(f"pid.plots.efficiency_vs_fake: no efficiency rows in {path}")
    x = eff["target_fake_rate"].to_numpy(float)
    y = eff["value"].to_numpy(float)
    # Missing bounds stay missing: nan_to_num would draw zero-length bars that
    # read as "exactly known" error bars.
    lo = y - eff["value_err_lo"].to_numpy(float)
    hi = eff["value_err_hi"].to_numpy(float) - y
    lo = np.where(np.isfinite(lo), lo, np.nan)
    hi = np.where(np.isfinite(hi), hi, np.nan)
    sig = str(eff["signal"].iloc[0])
    bkg = str(eff["against"].iloc[0])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel(f"{bkg} fake rate (working point)")
    ax.set_ylabel(f"{sig} efficiency")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    info = (f"n_sig = {int(eff['n_signal'].iloc[0])}, "
            f"n_bkg = {int(eff['n_background'].iloc[0])}")
    ax.set_title(title or f"{sig}/${bkg}$ working points  [{info}]")
    return _finish(fig, path)


#: Human-readable axis names for the binned-metric figures. The unit must follow
#: the slicing variable: labelling an eta histogram "eta [GeV]" (as a fixed [GeV]
#: suffix did) is simply wrong.
_VARIABLE_LABELS: dict[str, str] = {"pt": "reconstructed track $p_T$ [GeV]",
                                    "p": "reconstructed track momentum $p$ [GeV]",
                                    "eta": "reconstructed track $\\eta$",
                                    "truth_pt": "truth $p_T$ [GeV]",
                                    "truth_p": "truth momentum $p$ [GeV]",
                                    "truth_eta": "truth $\\eta$"}

#: What each plotted column means, spelled out on the axis so a reader is not
#: guessing whether "auc" is global or per-bin, and which pair it ranks.
_COLUMN_LABELS: dict[str, str] = {
    "auc": "AUC per bin = P(score$_{\\mathrm{signal}}$ > score$_{\\mathrm{background}}$)",
    "efficiency": "efficiency per bin",
    "n_sigma": "$n_\\sigma$ separation per bin",
}


def axis_labels(table: pd.DataFrame, column: str) -> tuple[str, str]:
    """``(xlabel, ylabel)`` for a binned-metric table (separated for testing)."""
    variable = str(table["variable"].iloc[0]) if len(table) and "variable" in table else ""
    xlabel = _VARIABLE_LABELS.get(variable, variable or "bin")
    signal = str(table["signal"].iloc[0]) if len(table) and "signal" in table else ""
    against = str(table["against"].iloc[0]) if len(table) and "against" in table else ""
    base = _COLUMN_LABELS.get(column, column)
    ylabel = f"{base}  [{signal} vs {against}]" if signal and against else base
    return xlabel, ylabel


def metric_vs_bin(variable_table: pd.DataFrame, path: str, *, column: str = "auc",
                  title: str = "") -> str:
    """`column` (AUC, efficiency, n-sigma, ...) vs bin centre.

    Bins with too few candidates to mean anything are drawn as open crosses and
    labelled "low stats" rather than dropped, because *where* the statistics run
    out is itself the result of a one-file test - and it is exactly the
    information that tells you how many grid files a quoted number needs.
    """
    table = variable_table.dropna(subset=[column])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    good = ~table["insufficient_stats"].astype(bool)
    for mask, marker, label, line in ((good, "o", "bins with enough stats", "-"),
                                      (~good, "x", "low statistics", "none")):
        sub = table[mask]
        if sub.empty:
            continue
        ax.plot(sub["bin_center"], sub[column], marker=marker, ls=line,
                ms=6 if bool(mask is good) else 8,
                alpha=0.9 if bool(mask is good) else 0.6, label=label)
    xlabel, ylabel = axis_labels(table, column)
    ax.set_xscale("log" if (len(table) and table["bin_center"].min() > 0) else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    if column == "auc":
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.text(0.02, 0.5, " chance", fontsize=7, color="grey", va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_ylim(0.45, 1.03)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="lower right")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def importance(tables: dict[str, pd.DataFrame], path: str, *, top: int = 15,
               title: str = "") -> str:
    kinds = [k for k in ("shap", "gain", "permutation") if k in tables and not tables[k].empty]
    if not kinds:
        raise SystemExit("pid.plots.importance: no importance tables to plot")
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.4 * len(kinds), 5.2), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        table = tables[kind].head(top)
        value_col = ("mean_abs_shap" if kind == "shap"
                     else "gain_fraction" if kind == "gain" else "importance_mean")
        names = [c if len(c) <= 26 else c[:24] + ".." for c in table["column"]]
        ax.barh(range(len(table)), table[value_col].to_numpy(float), color="0.35")
        ax.set_yticks(range(len(table)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(value_col, fontsize=8)
        ax.set_title(kind, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    return _finish(fig, path)


def score_distributions(fits: dict[str, dict], path: str, *, title: str = "") -> str:
    """Annotate the fitted discriminator peaks (what n-sigma is made from)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    plotted = 0
    for name, fit in fits.items():
        if not fit.get("converged"):
            continue
        plotted += 1
        mu, sg = fit["mu"], fit["sigma"]
        xs = np.linspace(mu - 4 * sg, mu + 4 * sg, 200)
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sg) ** 2), label=(
            f"{name}: $\\mu$={mu:.3f}, $\\sigma$={sg:.3f}, N={fit['n']}"))
        ax.axvline(mu, ls=":", lw=0.8)
    ax.set_xlabel("discriminator")
    ax.set_ylabel("a.u.")
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no converged fits", transform=ax.transAxes, ha="center",
                fontsize=9, color="0.4")
    if title:
        ax.set_title(title)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Maximum-significance performance figures
#
# Filenames follow the requested convention exactly (pid-<channel>_<figure>.png);
# the model and dataset tag live in each figure's title instead, so the same plot
# made for a different learner does not silently overwrite this one.
# ---------------------------------------------------------------------------

def significance_vs_cut(scan: pd.DataFrame, optimum: dict, path: str, *, title: str = "") -> str:
    """FOM(c) = S/sqrt(S+B) vs threshold, with the maximum marked.

    Both significance variants are drawn: the requested one, and the
    variance-corrected form using sum(w^2) under the root, which diverges
    from the naive form as soon as the classes are weighted differently.
    Where they differ, the naive form is overstating the reach.
    """
    tab = scan.dropna(subset=["significance"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax.plot(tab["threshold"], tab["significance"], lw=1.4, label="$S/\\sqrt{S+B}$")
    ax.plot(tab["threshold"], tab["significance_eff"], lw=1.0, ls="--",
            label="$S/\\sqrt{S_{eff}+B_{eff}}$")
    if optimum.get("valid"):
        c, f = optimum["threshold"], optimum["significance"]
        ax.axvline(c, color="0.4", ls=":", lw=1.0)
        ax.plot([c], [f], marker="*", ms=14, color="crimson", zorder=5,
                label=f"$c^*$ = {c:.3f}, FOM = {f:.1f}")
        ax.annotate(f"$c^*$={c:.3f}\nFOM={f:.2f}\n$\\varepsilon$={optimum['efficiency']:.3f}\n"
                    f"fake={optimum['fake_rate']:.2e}\npurity={optimum['purity']:.3f}",
                    xy=(c, f), xytext=(0.60, 0.45), fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("classifier score threshold $c$")
    ax.set_ylabel("significance $S/\\sqrt{S+B}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    if title:
        ax.set_title(title, fontsize=9)

    ax2.plot(tab["threshold"], tab["efficiency"], label="signal efficiency", lw=1.2)
    ax2.plot(tab["threshold"], np.clip(tab["fake_rate"], 1e-8, None), label="fake rate", lw=1.2)
    ax2.plot(tab["threshold"], tab["purity"], label="purity", lw=1.2)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-8, 1.5)
    ax2.set_xlabel("threshold $c$")
    ax2.set_ylabel("efficiency / fake rate / purity (log)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.set_title("what the cut costs and buys", fontsize=9)
    return _finish(fig, path)


def _bin_axis(ax, table: pd.DataFrame, column: str, *, ylabel: str, yerr_lo=None, yerr_hi=None,
              logy: bool = False, ylim=None, reference_lines=()):
    good = table[~table["insufficient_stats"].astype(bool)].dropna(subset=[column])
    thin = table[table["insufficient_stats"].astype(bool)].dropna(subset=[column])
    if yerr_lo is None:
        yerr_lo, yerr_hi = f"{column}_err_lo", f"{column}_err_hi"
    has_err = yerr_lo in good.columns and yerr_hi in good.columns
    lo = (good[column] - good[yerr_lo]).to_numpy(dtype=float) if has_err else None
    hi = (good[yerr_hi] - good[column]).to_numpy(dtype=float) if has_err else None
    if len(good):
        ax.errorbar(good["bin_center"], good[column], yerr=[np.abs(lo), np.abs(hi)],
                    fmt="o-", capsize=3, ms=5, label="bins with enough stats")
    if len(thin):
        ax.plot(thin["bin_center"], thin[column], "x", ms=8, alpha=0.55, ls="none",
                label="low statistics")
    for value, label in reference_lines:
        ax.axhline(value, ls="--", lw=0.7, color="grey")
        if label:
            ax.text(0.01, value, label, fontsize=6.5, color="grey", va="bottom",
                    transform=ax.get_yaxis_transform())
    ax.set_xlabel(_VARIABLE_LABELS.get(str(table["variable"].iloc[0]) if len(table) else "", "bin"))
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, which="both")


def efficiency_at_cut(table: pd.DataFrame, path: str, *, title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    _bin_axis(ax, table, "efficiency", ylabel="signal efficiency at $c^*$", ylim=(0, 1.05))
    ax.legend(fontsize=7.5, loc="lower left")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def misid_at_cut(table: pd.DataFrame, path: str, *, title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    _bin_axis(ax, table, "fake_rate", ylabel="mis-ID (fake) rate at $c^*$", logy=True,
              ylim=(1e-4, 2.0))
    ax.legend(fontsize=7.5, loc="upper right")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def purity_at_cut(table: pd.DataFrame, path: str, *, title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    _bin_axis(ax, table, "purity", ylabel="signal purity $S/(S+B)$ at $c^*$", ylim=(0, 1.05))
    ax.legend(fontsize=7.5, loc="lower left")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def rejection_vs_p(table: pd.DataFrame, path: str, *, title: str = "") -> str:
    """Rejection 1/fake vs momentum, log y from 10 to 10⁴.

    Where no background candidate passed, the point estimate is undefined and the
    95 % CL **lower limit** is drawn as an upward arrow instead. Reading those
    arrows as measurements (or plotting 1/0 = 10⁴) is the classic way to invent a
    rejection power that was never observed.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    t = table.dropna(subset=["rejection_limit"])
    measurable = t[t["rejection"].notna()]
    limits = t[t["rejection"].isna()]
    if len(measurable):
        ax.errorbar(measurable["bin_center"], measurable["rejection"],
                    yerr=[np.abs(measurable["rejection"] - measurable["rejection_limit"]),
                          np.zeros(len(measurable))],
                    fmt="o-", capsize=3, ms=5, label="$1/$fake (measured)")
    if len(limits):
        ax.plot(limits["bin_center"], limits["rejection_limit"], "^", ms=7, ls="none",
                color="darkorange",
                label="95 % CL lower limit only (0 fakes observed)")
    for decade in (1e2, 1e3, 1e4):
        ax.axhline(decade, ls=":", lw=0.7, color="grey")
        ax.text(0.01, decade, f" $10^{int(np.log10(decade))}$", fontsize=6.5, color="grey",
                va="bottom", transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(1.0, 1.5e4)
    ax.set_xlabel(_VARIABLE_LABELS.get(str(table["variable"].iloc[0]) if len(table) else "", "p"))
    ax.set_ylabel("pion/hadron rejection $1/fake$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def nsigma_vs_p_table(tables: list[pd.DataFrame], path: str, *, title: str = "") -> str:
    """nσ(p) for several hypothesis pairs on one axis (K/π, p/K, e/π).

    Two kinds of marker, because they mean different things: circles are point
    estimates from the AUC/logit estimator, upward triangles are **95 % CL lower
    limits** from bins where the two score samples do not overlap at all (AUC = 1).
    Plotting the limit as a point would claim a measurement; omitting the bin would
    hide it.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    colours = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    plotted = False
    for table, colour in zip(tables, colours):
        have_limits = "n_sigma_lower_limit" in table.columns
        thin = (table["insufficient_stats"].astype(bool)
                if "insufficient_stats" in table else
                pd.Series(False, index=table.index))
        limits = table[have_limits & table["n_sigma_lower_limit"].notna() & ~thin]
        # A point estimate from a bin with a handful of candidates is not a
        # measurement; it is drawn as a low-statistics cross, never as a circle
        # that a reader will compare against the 3-sigma line.
        ok = table[table["n_sigma"].notna() & ~thin]
        weak = table[table["n_sigma"].notna() & thin]
        failed = table[table["n_sigma"].isna()
                       & (have_limits & table["n_sigma_lower_limit"].isna())]
        if len(ok):
            ax.plot(ok["bin_center"], ok["n_sigma"], "o-", color=colour, ms=5,
                    label=f"{ok['signal'].iloc[0]} vs {ok['background'].iloc[0]}")
            plotted = True
        if len(limits):
            ax.plot(limits["bin_center"], limits["n_sigma_lower_limit"], "^", color=colour,
                    ms=8, ls="none",
                    label=f"{limits['signal'].iloc[0]} vs {limits['background'].iloc[0]}:"
                          " 95% CL lower limit (AUC=1)")
            plotted = True
        if len(weak):
            ax.plot(weak["bin_center"], weak["n_sigma"], "x", color=colour, ms=7,
                    ls="none", alpha=0.55)
        if len(failed):
            ax.plot(failed["bin_center"], np.full(len(failed), 0.105), "x", color=colour,
                    ms=7, ls="none", alpha=0.55)
    ax.axhline(3.0, ls="--", lw=0.8, color="grey")
    ax.text(0.01, 3.0, " 3σ", fontsize=7, color="grey", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.text(0.5, 0.105, "no estimate: too few candidates in that bin", transform=ax.transAxes,
            ha="center", fontsize=6, color="0.45", va="bottom")
    var0 = str(tables[0]["variable"].iloc[0]) if (len(tables[0]) and "variable" in tables[0]) else "p"
    ax.set_xlabel(_VARIABLE_LABELS.get(var0, f"{var0} [GeV]"))
    ax.set_ylabel("$n_\\sigma$ = $|\\mu_s-\\mu_b|/\\sqrt{(\\sigma_s^2+\\sigma_b^2)/2}$")
    ax.grid(alpha=0.3, which="both")
    if plotted:
        ax.legend(fontsize=7.5, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no momentum bin had enough candidates\nto fit both score peaks",
                transform=ax.transAxes, ha="center", fontsize=9, color="0.4")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def map_2d_figure(m: dict, path: str, *, title: str = "") -> str:
    """2D (pT, eta) map; cells without statistics are left white, not coloured zero."""
    value = np.ma.masked_invalid(np.asarray(m["value"], dtype=float))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mesh = ax.pcolormesh(m["pt_edges"], m["eta_edges"], value.T, cmap="viridis",
                         shading="flat", rasterized=True)
    if np.isfinite(value.compressed()).size:
        vmin = float(np.nanmin(value)) if np.isfinite(np.nanmin(value)) else 0.0
        vmax = float(np.nanmax(value)) if np.isfinite(np.nanmax(value)) else 1.0
        mesh.set_clim(vmin, vmax if vmax > vmin else vmin + 1e-6)
    ax.set_xscale("log")
    ax.set_xlabel(m.get("x_label", "track $p_T$ [GeV]"))
    ax.set_ylabel(m.get("y_label", "track $\\eta$"))
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(m["quantity"], fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Counts on top so a dip can be told apart from an empty cell - the difference
    # between "the detector crack kills efficiency" and "nothing was measured here".
    raw = np.asarray(m["value"], dtype=float)
    finite = raw[np.isfinite(raw)]
    median_ref = float(np.median(finite)) if finite.size else 0.0
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            n = m["counts"][i, j]
            if n >= 1 and np.isfinite(raw[i, j]):
                ax.text(0.5 * (m["pt_edges"][i] + m["pt_edges"][i + 1]),
                        0.5 * (m["eta_edges"][j] + m["eta_edges"][j + 1]), f"{int(n)}",
                        ha="center", va="center", fontsize=5,
                        color="white" if raw[i, j] < median_ref else "black")
    ax.set_title(title or f"{m['quantity']} at $c^*$={m['cut']:.3f}", fontsize=9)
    return _finish(fig, path)


def confusion_matrix_figure(conf: dict, path: str, *, title: str = "") -> str:
    """Row-normalised multi-class confusion matrix with counts annotated.

    Rows are truth species, columns the prediction categories - which include
    ``none`` when the matrix was built with per-class working points, so the
    reader can see how much of the sample the cuts decline to identify.
    """
    matrix, classes, counts = conf["matrix"], conf["classes"], conf["counts"]
    truth_classes = conf.get("truth_classes") or classes
    if not len(classes):
        raise SystemExit("pid.plots.confusion_matrix_figure: empty confusion matrix")
    fig, ax = plt.subplots(figsize=(1.9 + 1.25 * len(classes), 1.9 + 1.15 * len(classes)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), [f"pred {c}" for c in classes], fontsize=8)
    ax.set_yticks(range(len(truth_classes)), [f"true {c}" for c in truth_classes], fontsize=8)
    for i in range(len(truth_classes)):
        for j in range(len(classes)):
            v = matrix[i, j]
            ax.text(j, i, f"{'' if not np.isfinite(v) else f'{100 * v:.1f}'}%"
                    f"\n{counts[i, j]}", ha="center", va="center", fontsize=7,
                    color="black" if np.isfinite(v) and v < 0.55 else "white")
    ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(truth_classes), 1), minor=True)
    ax.grid(which="minor", color="0.8", lw=0.6)
    ax.tick_params(which="both", length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("fraction of true class",
                                                                fontsize=8)
    ax.set_title(title or "row-normalised confusion (per true species)", fontsize=9)
    return _finish(fig, path)


def score_dist_train_vs_test(over: dict, path: str, *, title: str = "",
                             cut: float | None = None) -> str:
    """Classifier score, training vs held-out sample, per class, log y.

    A large gap is memorisation: the training sample contains the very structures
    the model fitted. The KS statistic per class is printed on the figure so the
    judgement is quantitative, not eyeballed. Counts are drawn +0.3 (both
    histograms equally) so empty bins stay visible on the log y axis -
    display offset only, the legend quotes the true n.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), squeeze=False)
    for ax, (cls, per) in zip(axes[0], over["classes"].items()):
        edges = None
        for label_text, style in (("train", "-"), ("test", "--")):
            entry = per.get(label_text)
            if not entry:
                continue
            edges = np.asarray(entry["edges"], dtype=float)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, np.asarray(entry["counts"], dtype=float) + 0.3, style,
                    where="mid", ms=4, lw=1.2, label=f"{label_text} (n={entry['n']})")
        ks = over.get("ks", {}).get(cls, {})
        if cut is not None and np.isfinite(cut):
            ax.axvline(float(cut), ls="--", lw=0.9, color="crimson")
            ax.text(float(cut), ax.get_ylim()[1], f" $c^*$={float(cut):.3f}", fontsize=6.5,
                    color="crimson", ha="left", va="top")
        ax.set_yscale("log")
        ax.set_xlabel("classifier score")
        ax.set_ylabel("candidates (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)
        ax.set_title(f"{cls}" + (f"   KS = {ks['statistic']:.3f} "
                                 f"(p = {ks['p_value']:.2g})" if ks else ""), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=9)
    return _finish(fig, path)


def roc_from_scan(scan: pd.DataFrame, path: str, *, title: str = "",
                  operating_point: tuple | None = None) -> str:
    """ROC read off the threshold scan, with the c* operating point marked.

    The curve is cut-free (it *is* the cut scan), but the deliverable is meant to
    show where the quoted working point sits on it, so the (fake, eff) pair at c*
    is starred and annotated.
    """
    tab = scan.dropna(subset=["fake_rate", "efficiency"])
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    ax.plot(np.clip(tab["fake_rate"], 1e-8, None), tab["efficiency"], lw=1.5)
    if operating_point is not None:
        fake, eff, cut = operating_point
        if np.isfinite(fake) and np.isfinite(eff):
            ax.plot([max(float(fake), 1e-8)], [float(eff)], marker="*", ms=14,
                    color="crimson", mec="black", mew=0.5, ls="none", zorder=6,
                    label=f"$c^*$={float(cut):.3f}")
            ax.annotate(f"  $\\varepsilon$={float(eff):.3f}\n  fake={float(fake):.2e}",
                        xy=(max(float(fake), 1e-8), float(eff)), fontsize=7,
                        color="crimson", va="center")
            ax.legend(fontsize=7.5, loc="lower right")
    for target in config.FAKE_RATE_TARGETS:
        ax.axvline(target, ls=":", lw=0.7, color="grey")
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 1.2)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("background (fake) efficiency")
    ax.set_ylabel("signal efficiency")
    ax.grid(alpha=0.3, which="both")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def performance_package(*, channel: str, scan: pd.DataFrame, optimum: dict,
                        per_bin: pd.DataFrame, tables: dict, maps: dict,
                        confusion: dict | None, overtraining: dict | None,
                        out_dir: str, model: str, dataset_tag: str,
                        frame: pd.DataFrame | None = None,
                        suffix: str = "", write_basis_independent: bool = True,
                        cut: float | None = None,
                        confusion_at_cut: dict | None = None) -> dict[str, str]:
    """Write every deliverable figure for one channel; returns {key: path}.

    All differential figures are drawn **at the channel working point c\\***, which
    is passed in as `cut` and annotated wherever a threshold enters (ROC operating
    point, score-distribution line, per-pT-bin stars). n-sigma is the deliberate
    exception: it is a property of the score *distribution* (AUC / logit fits), so
    it does not depend on a cut - the working point is still recorded on the figure
    title so the package reads as one coherent selection.

    Channels whose `status` is ``exploratory`` get an explicit watermark: these are
    calorimeter-only results awaiting Cherenkov/dE/dx information, and a plot that
    looks like a headline number while being a baseline is how a result gets
    misquoted.
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = config.CHANNELS[channel]
    cap = f"[{model} / {dataset_tag}]"
    status = str(spec.get("status", "headline"))
    mark = "  EXPLORATORY" if status == "exploratory" else ""
    tag = f"{spec['title']} {cap}{mark}"
    written: dict[str, str] = {}

    def P(name: str) -> str:
        # `suffix` separates truth-binned figures from reconstructed-binned ones
        # when a run produces both bases.
        return os.path.join(out_dir, f"pid-{channel}_{name}{suffix}.png")

    def note(name: str, path: str) -> str:
        if status == "exploratory":
            _stamp_exploratory(name, path, spec.get("status_note", ""))
        return path

    if write_basis_independent:
        written["significance_vs_cut"] = note(
            "significance_vs_cut",
            significance_vs_cut(scan, optimum, P("significance_vs_cut"),
                                title=f"FOM(c) and the optimum {tag}"))
        if optimum.get("valid"):
            written["roc"] = note(
                "roc",
                roc_from_scan(scan, P("roc"), title=f"ROC in {channel} score space {tag}",
                              operating_point=(optimum.get("fake_rate"),
                                               optimum.get("efficiency"),
                                               optimum.get("threshold"))))

    if len(per_bin):
        written["optimal_cut_vs_pt"] = note(
            "optimal_cut_vs_pt",
            optimal_cut_vs_pt(per_bin, P("optimal_cut_vs_pt"), title=f"$c^*(p_T)$ {tag}",
                              global_cut=cut,
                              xlabel=("truth $p_T$ [GeV]" if suffix == "_truthpt"
                                      else "track $p_T$ [GeV]")))

    for name, key, func, extra in (
            ("eff_vs_pt", "at_cut_global", efficiency_at_cut, "signal efficiency at $c^*$"),
            ("misid_vs_pt", "at_cut_global", misid_at_cut, "mis-ID fake rate at $c^*$"),
            ("purity_vs_pt", "at_cut_global", purity_at_cut, "purity at $c^*$")):
        table = tables.get(key)
        if table is None or not len(table):
            continue
        col = {"eff_vs_pt": "efficiency", "misid_vs_pt": "fake_rate",
               "purity_vs_pt": "purity"}[name]
        if not table[col].notna().any():
            continue
        written[name] = note(name, func(table, P(name),
                                        title=f"{extra} {tag}"))

    rejection = tables.get("rejection_vs_p")
    if rejection is not None and len(rejection):
        fname = "pion_rejection_vs_p" if channel == "eid" else "rejection_vs_p"
        written[fname] = note(
            fname,
            rejection_vs_p(rejection, P(fname),
                           title=f"{'pion' if channel == 'eid' else spec['background']} "
                                 f"rejection vs $p$ at $c^*$ {tag}"))

    nsig = tables.get("nsigma_vs_p")
    if nsig is not None and len(nsig):
        method = str(nsig["nsigma_method"].dropna().iloc[0]) if (
            "nsigma_method" in nsig and nsig["nsigma_method"].notna().any()) else ""
        written["nsigma_vs_p"] = note(
            "nsigma_vs_p",
            nsigma_vs_p_table([nsig], P("nsigma_vs_p"),
                              title=f"n_sigma vs $p$ {tag}"
                                    + (f"  [{method}]" if method else "")))

    curves = tables.get("significance_by_pt")
    if curves is not None and len(curves):
        written["significance_vs_cut_by_pt"] = note(
            "significance_vs_cut_by_pt",
            significance_vs_cut_by_bin(curves, P("significance_vs_cut_by_pt"),
                                       global_cut=optimum.get("threshold"),
                                       title=f"FOM(c) per $p_T$ bin {tag}"))
        written["significance_vs_cut_panels"] = note(
            "significance_vs_cut_panels",
            significance_vs_cut_panels(curves, P("significance_vs_cut_panels"),
                                       title=f"per-bin working points {tag}"))

    for quantity, m in maps.items():
        fname = "eff_map_pt_vs_eta" if quantity == "efficiency" else "fake_map_pt_vs_eta"
        written[fname] = note(
            fname, map_2d_figure(m, P(fname),
                                 title=f"{quantity} in (pT, eta) at $c^*$ "
                                       f"={m['cut']:.3f} {tag}"))

    if overtraining and overtraining.get("classes"):
        written["score_dist_train_vs_test"] = note(
            "score_dist_train_vs_test",
            score_dist_train_vs_test(
                overtraining, P("score_dist_train_vs_test"), cut=cut,
                title=f"overtraining check {tag} (train n="
                      f"{overtraining['train_rows']}, test n={overtraining['test_rows']})"))

    if write_basis_independent and confusion is not None and len(confusion.get("classes", [])):
        written["confusion_matrix"] = confusion_matrix_figure(
            confusion, os.path.join(out_dir, f"pid-{spec['task']}_confusion_matrix.png"),
            title=f"confusion, {confusion.get('decision', 'argmax')} rule {cap}")
    if (write_basis_independent and confusion_at_cut is not None
            and len(confusion_at_cut.get("classes", []))):
        written["confusion_matrix_at_cut"] = confusion_matrix_figure(
            confusion_at_cut, os.path.join(out_dir,
                                 f"pid-{spec['task']}_confusion_matrix_atcut.png"),
            title=f"confusion at per-class $c^*$ {cap}")
    return written


def _stamp_exploratory(name: str, path: str, note_text: str) -> None:
    """Append the exploratory caveat to the figure's title (already drawn with it;
    this records it beside the file so the caveat travels with the artefact)."""
    sidecar = path + ".note"
    try:
        with open(sidecar, "w") as fh:
            fh.write(f"status: exploratory\nfigure: {name}\nwhy: {note_text}\n"
                     f"path: {path}\n")
    except OSError:  # pragma: no cover - a sidecar is a convenience
        pass


def optimal_cut_vs_pt(per_bin: pd.DataFrame, path: str, *, title: str = "",
                      global_cut: float | None = None, xlabel: str | None = None) -> str:
    """The optimal threshold as a function of pT, with unmeasurable bins as gaps.

    The dashed line is the channel's global working point when `global_cut`
    is given; otherwise (legacy callers) it falls back to the median of the
    plotted bin optima, honestly labelled as such.
    """
    t = per_bin.dropna(subset=["threshold"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    if len(t):
        good = t[~t.get("insufficient_stats", pd.Series(False, index=t.index)).astype(bool)]
        ax.step(good["bin_center"], good["threshold"], where="mid", lw=1.4,
                label="bin-by-bin $c^*$ (max FOM)")
        ax.errorbar(good["bin_center"], good["threshold"], fmt="o", ms=4, color="C0")
        if global_cut is not None and np.isfinite(global_cut):
            ax.axhline(float(global_cut), ls="--", lw=0.9, color="0.4",
                       label=f"global $c^*$ = {float(global_cut):.3f}")
        else:
            flat = good["threshold"].median()
            if np.isfinite(flat):
                ax.axhline(flat, ls="--", lw=0.9, color="0.4",
                           label=f"median bin $c^*$ = {flat:.3f}")
    ax.set_yscale("linear")
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel or "track $p_T$ [GeV]")
    ax.set_ylabel("optimal score cut $c^*$")
    ax.grid(alpha=0.3)
    if len(t):
        ax.set_xscale("log")
    ax.legend(fontsize=7.5, loc="upper right")
    if title:
        ax.set_title(title, fontsize=9)
    if not len(t):
        ax.text(0.5, 0.5, "no pT bin had both classes\nwith enough candidates",
                transform=ax.transAxes, ha="center", fontsize=9, color="0.4")
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Significance vs threshold, per kinematic bin (working-point determination)
# ---------------------------------------------------------------------------

def _bin_curve_groups(curves: pd.DataFrame):
    """``[(bin_center, sub-table-with-a-curve), ...]`` sorted by momentum."""
    out = []
    for centre, sub in curves.groupby("bin_center", sort=True):
        if bool(sub["has_curve"].iloc[0]) and sub["significance"].notna().any():
            out.append((float(centre), sub.sort_values("threshold")))
    return out


def significance_vs_cut_by_bin(curves: pd.DataFrame, path: str, *, title: str = "",
                              global_cut: float | None = None) -> str:
    """FOM(c) = S/sqrt(S+B) in every pT bin, on one axis (absolute) and one (shape).

    Left: the raw significance, so the vertical spread *is* the information about
    where the statistical power lives - a low-pT curve reaching FOM 12 and a
    high-pT curve topping out at 1.5 are not equally usable working points.

    Right: each curve divided by its own maximum, which removes the luminosity
    factor and exposes the thing that actually varies with momentum: the position
    of the optimum. Stars mark each bin's c*.
    """
    groups = _bin_curve_groups(curves)
    variable = str(curves["variable"].iloc[0]) if len(curves) else "pt"
    xlabel = _VARIABLE_LABELS.get(variable, f"{variable} bin")
    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    if not groups:
        for ax in (ax_abs, ax_norm):
            ax.text(0.5, 0.5, "no pT bin contained both classes\nwith enough candidates",
                    transform=ax.transAxes, ha="center", fontsize=9, color="0.4")
            ax.set_xlabel("threshold $c$")
        fig.suptitle(title or "FOM(c) per bin", fontsize=10)
        return _finish(fig, path)

    centres = np.array([c for c, _ in groups])
    span = float(np.ptp(centres))
    norm = plt.Normalize(float(centres.min()),
                         float(centres.max()) if span > 0 else float(centres.min()) + 1)
    cmap = matplotlib.colormaps["viridis"]
    for centre, sub in groups:
        colour = cmap(norm(centre))
        ax_abs.plot(sub["threshold"], sub["significance"], lw=1.1, color=colour,
                    label=f"{centre:.2f}")
        ax_norm.plot(sub["threshold"],
                     sub["significance"] / max(float(np.nanmax(sub["significance"])), 1e-12),
                     lw=1.1, color=colour)
        opt = sub[sub["is_optimum"].astype(bool)]
        if len(opt):
            xo = float(opt["threshold"].iloc[0])
            yo = float(opt["significance"].iloc[0])
            ax_abs.plot([xo], [yo], marker="*", ms=8, color=colour, mec="black", mew=0.5,
                        ls="none")
            ax_norm.plot([xo], [1.0], marker="*", ms=8, color=colour, mec="black", mew=0.5,
                         ls="none")
    if global_cut is not None and np.isfinite(global_cut):
        for ax in (ax_abs, ax_norm):
            ax.axvline(float(global_cut), ls="--", lw=0.9, color="crimson")
        ax_norm.text(float(global_cut), 1.01, " global $c^*$", fontsize=7, color="crimson")
    for ax, ylabel in ((ax_abs, "significance $S/\\sqrt{S+B}$"),
                       (ax_norm, "FOM normalised to its own maximum")):
        ax.set_xlabel(f"score threshold $c$   (colour = {xlabel} bin centre)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
    ax_norm.set_ylim(0.0, 1.06)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=[ax_abs, ax_norm], pad=0.015, fraction=0.035)
    cbar.set_label(f"{xlabel} bin centre", fontsize=8)
    cbar.set_ticks(list(centres[:: max(1, len(centres) // 6)]))
    cbar.ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    return _finish(fig, path)


def significance_vs_cut_panels(curves: pd.DataFrame, path: str, *, title: str = "",
                               ncols: int = 4) -> str:
    """One panel per pT bin: the full FOM(c) curve with c*, efficiency and purity at it.

    Drawn panel-by-panel because overlaying thirteen curves hides exactly what a
    working-point plot is for - whether a given bin's maximum is a sharp, stable
    choice or a flat plateau over which the cut is arbitrary. The plateau case is
    common with few background candidates, and the annotation makes it visible.
    """
    groups = _bin_curve_groups(curves)
    all_centres = sorted(curves["bin_center"].unique().tolist()) if len(curves) else []
    n_panels = max(len(all_centres), 1)
    ncols = int(max(2, min(ncols, n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    variable = str(curves["variable"].iloc[0]) if len(curves) else "pt"
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.75 * ncols, 2.35 * nrows),
                             squeeze=False, sharex=True)
    by_centre = dict(groups)
    for ax, centre in zip(axes.ravel(), all_centres):
        sub = curves[curves["bin_center"] == centre]
        row0 = sub.iloc[0]
        n_sig, n_bkg = int(row0["n_signal_bin"]), int(row0["n_background_bin"])
        if centre not in by_centre:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.55, "no curve", transform=ax.transAxes, ha="center",
                    fontsize=8, color="0.35")
            ax.text(0.5, 0.32, str(row0.get("reason", ""))[:34], transform=ax.transAxes,
                    ha="center", fontsize=5.5, color="0.4")
            ax.set_title(f"{variable} {centre:.2f} GeV\n"
                         f"{row0['signal']}: {n_sig}  {row0['background']}: {n_bkg}",
                         fontsize=7)
            continue
        g = by_centre[centre]
        ax.plot(g["threshold"], g["significance"], lw=1.2, color="tab:blue")
        opt = g[g["is_optimum"].astype(bool)]
        if len(opt):
            xo, yo = float(opt["threshold"].iloc[0]), float(opt["significance"].iloc[0])
            ax.axvline(xo, ls=":", lw=0.9, color="crimson")
            ax.plot([xo], [yo], marker="*", ms=12, color="crimson", mec="black", mew=0.4,
                    ls="none", zorder=5)
            ax.text(0.03, 0.92, f"$c^*$={xo:.3f}\nFOM={yo:.2f}\n"
                                f"$\\varepsilon$={float(opt['efficiency'].iloc[0]):.3f}\n"
                                f"fake={float(opt['fake_rate'].iloc[0]):.2e}",
                    transform=ax.transAxes, fontsize=6, va="top")
        peak = float(np.nanmax(g["significance"].to_numpy(dtype=float)))
        plateau = (g["significance"] >= peak - 1e-9).sum() / max(len(g), 1)
        if plateau > 0.05:
            ax.text(0.97, 0.05, f"flat top: {100 * plateau:.0f}% of range",
                    transform=ax.transAxes, ha="right", fontsize=5.5, color="0.4")
        # signal/background here are species NAMES; the counts are the numbers.
        ax.set_title(f"{variable} = {centre:.2f} GeV\n"
                     f"{row0['signal']}: {n_sig}   {row0['background']}: {n_bkg}",
                     fontsize=7)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(all_centres):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("threshold $c$", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("$S/\\sqrt{S+B}$", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    return _finish(fig, path)
