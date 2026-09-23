"""Output writers: JSON (machine-readable, what compare.py reads back),
markdown tables (human-readable), and simple diagnostic plots.

Kept separate from the metric modules so a metric function's return value
(a plain DataFrame) is easy to unit-test without touching the filesystem.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # headless - this project never assumes a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

#: Columns holding pandas Interval objects, which are not JSON-serialisable.
#: They are stringified for JSON/markdown output; the *_center columns carry
#: the numeric information forward for plotting and for compare.py's
#: bin-alignment join.
_INTERVAL_COLUMNS = ("pt_bin", "eta_bin")


def _import_root():
    """Lazy-import PyROOT (a heavy dependency) only when ROOT output is needed."""
    import ROOT

    ROOT.gROOT.SetBatch(True)  # headless
    return ROOT


def _json_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _INTERVAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def to_json(df: pd.DataFrame, path: str, meta: dict | None = None) -> None:
    """Write a metric result DataFrame to JSON as {"meta": ..., "data": [...]}.

    `meta` should record everything needed to reproduce the run: dataset
    tag, campaign, minQ2 tier(s), file count, matching threshold, etc. - see
    AGENTS.md's Definition of done.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("written_at_utc", datetime.now(timezone.utc).isoformat())

    # Round-trip through pandas' own JSON encoder first: it already knows
    # how to turn numpy int64/float64/bool_ and NaN into plain
    # JSON-safe/null values, which the stdlib json module does not.
    data = json.loads(_json_safe(df).to_json(orient="records"))
    payload = {"meta": meta, "data": data}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def read_json(path: str) -> tuple[pd.DataFrame, dict]:
    """Inverse of to_json: returns (DataFrame, meta dict)."""
    with open(path) as f:
        payload = json.load(f)
    return pd.DataFrame(payload["data"]), payload.get("meta", {})


def to_markdown_table(df: pd.DataFrame, path: str) -> None:
    """Write a metric result DataFrame as a markdown table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(_json_safe(df).to_markdown(index=False))
        f.write("\n")


#: Columns whose values are human-readable code names (species, etc.) that
#: must be encoded as an integer code in the ROOT output - TNtuple branches
#: can only hold floats. The mapping is derived at write time and recorded in
#: the file's metadata so a human can decode the codes when analysing by hand.
_ROOT_STRING_COLUMNS = ("species", "truth_species", "reco_species")


def to_root(df: pd.DataFrame, path: str, tree_name: str = "data", meta: dict | None = None):
    """Write a metric result DataFrame to a ROOT file as a TNtuple.

    Intended for *manual* follow-up analysis and plotting in a ROOT session:
    the same per-bin numbers that go into the JSON/markdown output are exposed
    as a flat tuple a human can query with ROOT cut expressions (e.g.
    ``Tree->Draw("sigma:pt_bin_center", "fit_converged && !insufficient_stats",
    "colz")`` for the resolution rows worth trusting).

    Just as with :func:`fit_gaussian_core`, the ROOT output is produced with
    the same ROOT libraries used elsewhere in the project, so the tuples a
    human draws match what trkperf computed.

    Species/code columns (species, truth_species, reco_species) are encoded as
    integer codes (0,1,2,...) matching the order in config.SPECIES, and the
    ``species_code`` -> name legend is stored as TNamed objects (``species_code_<n>``)
    alongside the ``meta_<key>`` provenance entries - not a RunInfo TTree
    (PyROOT does not transparently reflect Python-str reassignment into a
    char[] branch buffer, which silently drops the strings). All numeric/boolean columns become TNtuple
    branches (booleans as 0/1, NaN stays NaN). The ``pt_bin``/``eta_bin``
    string intervals are dropped - their centres (pt_bin_center /
    eta_bin_center) carry the same information and are kept.

    Parameters
    ----------
    df:
        A metric result DataFrame (one row per bin).
    path:
        Output ``.root`` file path.
    tree_name:
        Name of the TNtuple holding the per-bin data (default "data").
    meta:
        Provenance dict recorded in the ``RunInfo`` tree alongside the
        species legend, for reproducibility.
    """
    from . import config

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    species_names = list(config.SPECIES)
    name_to_code = {name: i for i, name in enumerate(species_names)}

    def code_for(name) -> float:
        if name is None or (isinstance(name, float) and np.isnan(name)):
            return -1.0  # sentinel: no/unknown species
        return float(name_to_code.get(str(name), -1.0))

    # Build the list of float branches: one per numeric column, plus an
    # integer code column for each species/string column.
    branches: list[tuple[str, list[float]]] = []
    branch_names: list[str] = []
    for col in df.columns:
        if col in _ROOT_STRING_COLUMNS:
            branch = f"{col}_code"
            branch_names.append(branch)
            branches.append((branch, df[col].map(code_for).astype(float).tolist()))
        elif col in ("pt_bin", "eta_bin"):
            continue  # centres carry the bin information
        else:
            try:
                values = df[col].astype(float).tolist()
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"trkperf.report.to_root: column {col!r} is neither numeric "
                    f"nor a registered string column {_ROOT_STRING_COLUMNS} - "
                    "add it there (with a code legend) or drop it before writing."
                ) from exc
            branch_names.append(col)
            branches.append((col, values))

    _root = _import_root()
    f = _root.TFile.Open(path, "RECREATE")
    nt = _root.TNtuple(tree_name, tree_name, ":".join(branch_names))
    n_rows = len(df)
    const = np.full((n_rows, len(branch_names)), np.nan, dtype=np.float32)
    arrays = {name: np.asarray(vals, dtype=np.float64) for name, (_, vals) in zip(branch_names, branches)}
    for j, name in enumerate(branch_names):
        const[:, j] = arrays[name]
    for i in range(n_rows):
        nt.Fill(const[i])
    nt.Write()

    # Companion metadata: provenance + the species-code legend, stored as
    # TNamed objects in the file directory so a human can decode the integer
    # species_code branches when analysing the tuple by hand. (A TTree of C
    # char-array branches is avoided here - PyROOT does not transparently
    # reflect Python-str reassignment into a char[] branch buffer, which
    # silently drops the strings.)
    for key, value in (meta or {}).items():
        f.WriteObject(_root.TNamed(f"meta_{key}", str(value)), f"meta_{key}")
    for name in species_names:
        code = name_to_code[name]
        f.WriteObject(_root.TNamed(f"species_code_{code}", name), f"species_code_{code}")
    f.WriteObject(_root.TNamed("species_code_-1", "unknown/no species"), "species_code_-1")
    f.Close()


#: Distinct matplotlib marker styles assigned to the distinct values of the
#: marker column (e.g. the ~16 eta bins) so the third dimension (here: which
#: eta region a point comes from) is readable from the symbol alone without
#: relying on color, which is reserved for the group column (species).
_MARKER_STYLES = [
    "o", "s", "^", "v", "D", "P", "X", "*", "H",
    "+", "x", "d", "1", "2", "3", "4",
]


#: eta-bin-center -> detector region. Bin centres fall at +-0.25, +-0.75,
#: +-1.25, ... so +-1.0 splits cleanly between bins: barrel |eta| < 1,
#: endcaps beyond.
def eta_region(center: float) -> str:
    """Detector region (barrel / forward endcap / backward endcap) for an eta-bin centre."""
    if not np.isfinite(center):
        return "unknown"
    if abs(center) < 1.0:
        return "barrel"
    return "forward endcap" if center > 0 else "backward endcap"


#: Species groups for the per-region grouped plots: charge-conjugate pairs
#: merge into one curve (e+/e- etc.); proton and antiproton stay separate
#: curves. Labels are plot-only; JSON tables keep the full species axis.
SPECIES_GROUPS: dict[str, str] = {
    "e-": "e+/e-", "e+": "e+/e-",
    "pi+": "pi+/pi-", "pi-": "pi+/pi-",
    "K+": "K+/K-", "K-": "K+/K-",
    "proton": "proton", "antiproton": "antiproton",
}


def aggregate_eta_species(
    df: pd.DataFrame, value_col: str, err_col: str | None,
    count_cols: tuple[str, str] | None = None,
    eta_region_filter: str | None = None,
) -> pd.DataFrame:
    """Aggregate a metric table over eta regions x species groups, recomputing
    the value from summed counts (exact for count-based metrics).

    Parameters
    ----------
    count_cols:
        ``(numerator, denominator)`` count columns to sum within each
        (species group, eta region, pT bin), e.g. ``("n_in_acceptance",
        "n_generated")``. The value becomes numerator/denominator with a
        Wilson error. If None (resolution sigmas), the value is the
        inverse-variance weighted mean over the group and `err_col` its
        propagated error instead.

    Returns a DataFrame with ``species_group``, ``eta_region``,
    ``pt_bin_center``, the value/err columns, summed ``n`` and a recomputed
    ``insufficient_stats`` flag (summed n < 50) - shaped so
    :func:`plot_metric_vs_pt` draws it with ``group_col="species_group"``
    and ``marker_col="eta_region"``.
    """
    from . import binning

    work = df.copy()
    work["species_group"] = work["species"].map(
        lambda s: SPECIES_GROUPS.get(s, s)) if "species" in work.columns else "all"
    work["eta_region"] = work["eta_bin_center"].map(eta_region)
    if eta_region_filter is not None:
        # One plot per detector region (barrel/forward/backward endcap):
        # keep only that region's bins. Pair with the region-correct rule
        # (central JSONs for barrel, <region>-rule JSONs for endcaps).
        work = work[work["eta_region"] == eta_region_filter]
        if work.empty:
            return pd.DataFrame(
                columns=["species_group", "eta_region", "pt_bin_center",
                         value_col, err_col or "err", "n", "insufficient_stats"])
    group_keys = ["species_group", "eta_region", "pt_bin_center"]
    rows = []
    for keys, sub in work.groupby(group_keys, observed=True):
        row = dict(zip(group_keys, keys))
        n = 0
        if count_cols is not None:
            num_col, den_col = count_cols
            n_total = float(sub[den_col].sum())
            n = n_total
            if num_col in sub.columns:
                k = float(sub[num_col].sum())
            else:
                # Older result files predate the kept numerator column: the
                # value IS numerator/denominator (NaN only where the
                # denominator is 0), so sum(value * denom) recovers it exactly.
                if not getattr(aggregate_eta_species, "_warned_recovery", False):
                    print("[report] NOTE: numerator column "
                          f"{num_col!r} absent - recovering it as "
                          "sum(value * denominator); exact wherever the value "
                          "is numerator/denominator", file=sys.stderr)
                    aggregate_eta_species._warned_recovery = True
                k = float((sub[value_col].fillna(0).astype(float)
                           * sub[den_col].fillna(0).astype(float)).sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                row[value_col] = k / n_total if n_total > 0 else np.nan
            if err_col:
                row[err_col] = float(binning.binomial_error(k, n_total))
        else:
            vals = sub[value_col].to_numpy(dtype=float)
            errs = (sub[err_col].to_numpy(dtype=float) if err_col and err_col in sub.columns
                    else np.full_like(vals, np.nan))
            ok = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
            if "fit_converged" in sub.columns:
                # A non-converged fit carries no meaningful width even when a
                # number is present: exclude it rather than averaging it in.
                ok = ok & (sub["fit_converged"].to_numpy() == True)  # noqa: E712
            if ok.sum() > 0:
                w = 1.0 / errs[ok]**2
                row[value_col] = float(np.sum(vals[ok] * w) / w.sum())
                if err_col:
                    row[err_col] = float(1.0 / np.sqrt(w.sum()))
            else:
                row[value_col] = np.nan
                if err_col:
                    row[err_col] = np.nan
            n = float(sub["n_matched"].sum()) if "n_matched" in sub.columns else 0
        row["n"] = n
        row["insufficient_stats"] = bool(n < config.MIN_ENTRIES_PER_BIN)
        rows.append(row)
    return pd.DataFrame(rows)


def _markers_for(values: list) -> dict:
    """Map each sorted `values` entry to a distinct marker style, cycling the
    built-in list if there are more values than styles."""
    return {v: _MARKER_STYLES[i % len(_MARKER_STYLES)] for i, v in enumerate(values)}


def plot_metric_vs_pt(
    df: pd.DataFrame,
    value_col: str,
    err_col: str | None,
    path: str,
    group_col: str = "species",
    x_col: str = "pt_bin_center",
    title: str | None = None,
    y_label: str | None = None,
    log_x: bool = True,
    log_y: bool = False,
    marker_col: str | None = None,
    group_color: str | None = None,
    ymin: float | None = None,
    marker_legend_loc: str = "upper right",
) -> None:
    """`value_col` vs `x_col`, one curve per distinct `group_col` value.

    Colours encode the group (species); if `marker_col` is given (e.g.
    ``eta_bin_center``), its distinct values are additionally encoded with
    **different marker symbols**, so each (species, eta-region) series is
    distinguishable within a single curve without merging eta bins together.
    All markers are drawn **open** (unfilled) so overlapping series remain
    readable on any background.

    If `group_color` is set (e.g. ``"black"`` for the fake-rate metric, whose
    rows carry no species), every series is drawn in that single colour and
    the colour legend is omitted, leaving the marker shapes alone to separate
    the series.

    Skips rows flagged insufficient_stats if that column is present, so thin
    bins do not visually masquerade as real measurements. With ``log_y=True``
    the y-axis is log-scaled (useful when values span orders of magnitude,
    e.g. very small fake rates), and rows with ``value_col <= 0`` are dropped
    from the plot — zero has no representation on a log axis; such bins are
    simply absent rather than drawn at an invented floor value. ``ymin`` sets
    the y-axis lower limit explicitly (e.g. 1e-5 so very small fake rates stay
    readable while the axis still spans the physical range); it only makes
    sense to use it with ``log_y=True``. Both legends sit OUTSIDE the axes
    (species top-right, markers bottom-right) so legend boxes never cover
    data points; ``marker_legend_loc`` is kept for signature compatibility
    only.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plot_df = df
    if "insufficient_stats" in df.columns:
        plot_df = df[~df["insufficient_stats"]]
    if log_y:
        plot_df = plot_df[plot_df[value_col] > 0]

    groups = [None] if group_col not in plot_df.columns else sorted(
        g for g in plot_df[group_col].dropna().unique()
    )
    markers = [None]
    if marker_col and marker_col in plot_df.columns:
        markers = sorted(m for m in plot_df[marker_col].dropna().unique())
        marker_map = _markers_for(markers)
        cmap = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
        color_map = {
            g: cmap(i) if cmap.N > 1 else cmap(0)
            for i, g in enumerate(groups)
        }
    else:
        marker_map = {}
        color_map = None

    if group_color is not None:
        color_map = None  # single-colour mode: no species colour legend

    fig, ax = plt.subplots(figsize=(7, 5))
    for group in groups:
        for m in markers:
            sub = plot_df if group is None else plot_df[plot_df[group_col] == group]
            if m is not None:
                sub = sub[sub[marker_col] == m]
            sub = sub.sort_values(x_col)
            if sub.empty:
                continue
            yerr = sub[err_col] if err_col else None
            color = group_color if group_color is not None else (
                color_map[group] if color_map is not None else None
            )
            ax.errorbar(
                sub[x_col], sub[value_col], yerr=yerr,
                marker=marker_map.get(m, "o"),
                color=color,
                fillstyle="none", mfc="none", mec=color,
                linestyle="-", lw=0.8, alpha=0.8, capsize=2, markersize=4,
            )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_xlabel("pT [GeV]")
    ax.set_ylabel(y_label or value_col)
    if title:
        ax.set_title(title)

    if color_map is not None and group_color is None:
        handles = [
            plt.Line2D([0], [0], color=color_map[g], marker="o", linestyle="-",
                       lw=1.2, ms=4, fillstyle="none", mfc="none")
            for g in groups
        ]
        labels = [str(g) if g is not None else "all" for g in groups]
        # Outside the axes (top right) so the box never covers data points.
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=7, title=group_col, borderaxespad=0.0)
    if marker_map:
        mhandles = [
            plt.Line2D([0], [0], color="#333333" if group_color is None else group_color,
                       marker=marker_map[m], linestyle="none", ms=5,
                       fillstyle="none", mfc="none")
            for m in markers
        ]
        mlabels = [
            f"{m}" if str(marker_col) == "eta_region"
            else (f"eta={m:+.2f}" if str(marker_col).startswith("eta") else f"{marker_col}={m}")
            for m in markers
        ]
        # Drawn at figure level, outside the axes (bottom right), so it
        # coexists with the species-colour legend and never covers data.
        # marker_legend_loc is kept for signature compatibility but the box
        # always sits outside the axes now.
        fig.legend(
            mhandles, mlabels, loc="lower left", bbox_to_anchor=(1.02, 0.0),
            bbox_transform=ax.transAxes, fontsize=7, title=marker_col,
            ncol=2 if len(markers) > 8 else 1, borderaxespad=0.0,
        )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
