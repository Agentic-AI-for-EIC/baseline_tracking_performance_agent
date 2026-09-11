"""Output writers: JSON (machine-readable, what compare.py reads back),
markdown tables (human-readable), and simple diagnostic plots.

Kept separate from the metric modules so a metric function's return value
(a plain DataFrame) is easy to unit-test without touching the filesystem.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # headless - this project never assumes a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    ``species_code`` -> name legend is stored in the accompanying ``RunInfo``
    TTree so it can be decoded. All numeric/boolean columns become TNtuple
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
            branch_names.append(col)
            branches.append((col, df[col].astype(float).tolist()))

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
    sense to use it with ``log_y=True``. The colour legend (species) is
    top-left and the marker-shape legend (eta bin) sits at
    ``marker_legend_loc`` (default top-right).
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
        ax.legend(handles, labels, loc="upper left", fontsize=7, title=group_col)
    if marker_map:
        mhandles = [
            plt.Line2D([0], [0], color="#333333" if group_color is None else group_color,
                       marker=marker_map[m], linestyle="none", ms=5,
                       fillstyle="none", mfc="none")
            for m in markers
        ]
        mlabels = [
            f"eta={m:+.2f}" if str(marker_col).startswith("eta") else f"{marker_col}={m}"
            for m in markers
        ]
        # Drawn at figure level (not ax.legend) so it coexists with the
        # species-colour legend: ax.legend would replace the previous legend.
        # The bbox anchor is derived from the requested corner so the legend
        # box sits at that corner (upper left for a colour legend, otherwise
        # `marker_legend_loc`).
        corner_anchor = {
            "upper right": (1.0, 1.0),
            "upper left": (0.0, 1.0),
            "lower right": (1.0, 0.0),
            "lower left": (0.0, 0.0),
        }
        anchor = corner_anchor.get(marker_legend_loc, (1.0, 1.0))
        fig.legend(
            mhandles, mlabels, loc=marker_legend_loc, bbox_to_anchor=anchor,
            bbox_transform=ax.transAxes, fontsize=7, title=marker_col,
            ncol=2 if len(markers) > 8 else 1,
        )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
