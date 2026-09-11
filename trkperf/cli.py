"""Command-line entry point: `python -m trkperf <metric|compare> ...`.

Discovery (which files exist on the grid) is NOT this CLI's job - per
AGENTS.md, use the rucio/xrootd MCP tools to build a file list, then pass it
here (via --file-list, a text file with one path/URL per line, and/or
repeated --file arguments) for the actual numeric processing.

Examples
--------
    python -m trkperf acceptance --file-list clean_files.txt \\
        --dataset-tag clean --min-q2-tier 1

    python -m trkperf fake-rate --file-list bkg_files.txt \\
        --dataset-tag bkg_mixed --min-q2-tier 1,10

    python -m trkperf compare --metric acceptance \\
        --clean output/acceptance_clean.json --bkg output/acceptance_bkg_mixed.json
"""

from __future__ import annotations

import argparse
import os
import sys

from . import compare as compare_module
from . import config, io as io_module, report
from .metrics import acceptance, efficiency, resolution
from .metrics import fake_rate as fake_rate_module
from .pid import confusion as pid_confusion_module

#: Metrics that take a --species argument and share the same
#: compute_fn(file_paths, species=...) shape.
_SPECIES_METRIC_FUNCTIONS = {
    "acceptance": acceptance.compute_acceptance,
    "resolution": resolution.compute_resolution,
    "efficiency": efficiency.compute_efficiency,
    "pid-confusion": pid_confusion_module.compute_pid_confusion,
}
FAKE_RATE_NAME = "fake-rate"

#: Which (value_col, err_col, output_suffix, [log_y], [ymin],
#: [marker_legend_loc]) to plot per metric. The optional 4th element makes
#: that plot use a log-scaled y-axis; the 5th pins the y-axis lower limit
#: (e.g. 1e-5 for fake rates so small values stay readable on the log axis);
#: the 6th places the marker-shape (eta-bin) legend box at a corner other
#: than the default upper right. Adding a metric here is optional -
#: _write_outputs falls back to no plot if a metric name is not listed.
_PLOT_SPEC: dict[str, list[tuple]] = {
    "acceptance": [("acceptance", "acceptance_err", "vs_pt")],
    "resolution": [
        ("sigma", "sigma_err", "sigma_vs_pt"),
        ("sigma", "sigma_err", "sigma_vs_pt_logy", True),
    ],
    "efficiency": [
        ("efficiency_absolute", "efficiency_absolute_err", "absolute_vs_pt"),
        (
            "efficiency_within_acceptance",
            "efficiency_within_acceptance_err",
            "within_acceptance_vs_pt",
        ),
    ],
    # The fake-rate linear plot plus a log-y companion, so very small fake
    # rates (well below the top of the linear axis) are readable. On the
    # linear axis the eta-bin legend sits upper-right; on the log axis the
    # range is pinned to ymin=1e-5 and the legend sits lower-right so it does
    # not cover the high-pT falloff in the top-right.
    FAKE_RATE_NAME: [
        ("fake_rate", "fake_rate_err", "vs_pt", False, None, "upper right"),
        ("fake_rate", "fake_rate_err", "vs_pt_logy", True, 1e-5, "lower right"),
    ],
    # pid-confusion: no single group axis works for a two-species-axis
    # matrix — JSON/markdown output is the primary deliverable here.
}


def _plot_specs(
    metric_name: str,
) -> list[tuple[str, str, str, bool, float | None, str]]:
    """Normalised (value_col, err_col, suffix, log_y, ymin,
    marker_legend_loc) list for a metric.

    Handles both 3/4-element and 5/6-element _PLOT_SPEC tuples (log_y
    defaults to False, ymin to None, marker_legend_loc to "upper right" for
    the shorter forms).
    """
    specs = []
    for entry in _PLOT_SPEC.get(metric_name, []):
        value_col, err_col, suffix, *rest = entry
        log_y = bool(rest[0]) if len(rest) > 0 else False
        ymin = rest[1] if len(rest) > 1 else None
        marker_loc = rest[2] if len(rest) > 2 else "upper right"
        specs.append((value_col, err_col, suffix, log_y, ymin, marker_loc))
    return specs


def _read_file_list(args: argparse.Namespace) -> list[str]:
    files = list(args.file or [])
    if args.file_list:
        with open(args.file_list) as f:
            files.extend(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )
    if not files:
        raise SystemExit("No input files given: use --file (repeatable) and/or --file-list.")
    return files


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--file", action="append", help="A file path or root:// URL (repeatable).")
    sub.add_argument("--file-list", help="Text file with one path/URL per line.")
    sub.add_argument(
        "--dataset-tag",
        required=True,
        choices=["clean", "bkg_mixed"],
        help="Which project dataset type this run is - used for output naming "
        "and recorded in the output's metadata for provenance.",
    )
    sub.add_argument(
        "--min-q2-tier",
        default=None,
        help="Which minQ2 tier(s) were used (e.g. '1' or '1,10') - recorded in "
        "metadata only, not enforced by this CLI.",
    )
    sub.add_argument("--out-dir", default="output")
    sub.add_argument(
        "--max-file-failures",
        type=int,
        default=0,
        help="Tolerate up to this many individual file I/O failures (skipping "
        "them with warnings) instead of aborting the run. Use >0 for flaky "
        "xrootd endpoints (e.g. the overseas +background server).",
    )
    sub.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache successfully read per-file tables (pickled, "
        "keyed on URL + branches). A relaunched run re-starts for free over "
        "already-processed files, so a multi-hour run interrupted by a "
        "container/process exit accumulates progress across restarts instead "
        "of re-downloading from the start. Cached tables are re-stamped with "
        "file_id on load, so the file-list order need not be stable.",
    )


def _write_outputs(
    df, metric_name: str, args: argparse.Namespace, n_files: int
) -> str:
    """Write JSON + markdown + ROOT TNtuple + (if configured) plots; return
    the output path stem so callers can build further filenames from it."""
    meta = {
        "metric": metric_name,
        "dataset_tag": args.dataset_tag,
        "min_q2_tier": args.min_q2_tier,
        "n_files": n_files,
    }
    base = os.path.join(args.out_dir, f"{metric_name}_{args.dataset_tag}")
    report.to_json(df, base + ".json", meta=meta)
    report.to_markdown_table(df, base + ".md")
    report.to_root(
        df, base + ".root", tree_name=metric_name.replace("-", "_"), meta=meta
    )

    n_flagged = int(df["insufficient_stats"].sum()) if "insufficient_stats" in df else None
    print(
        f"[{metric_name}/{args.dataset_tag}] wrote {base}.json, {base}.md and "
        f"{base}.root ({len(df)} bins, {n_flagged} flagged insufficient "
        f"statistics, from {n_files} files)"
    )

    for value_col, err_col, suffix, log_y, ymin, marker_loc in _plot_specs(metric_name):
        plot_path = f"{base}_{suffix}.png"
        group_col = "species" if metric_name != FAKE_RATE_NAME else "__no_species_axis__"
        group_color = "black" if metric_name == FAKE_RATE_NAME else None
        report.plot_metric_vs_pt(
            df, value_col, err_col, plot_path, group_col=group_col,
            marker_col="eta_bin_center", group_color=group_color, log_y=log_y,
            ymin=ymin, marker_legend_loc=marker_loc,
        )
        print(f"[{metric_name}/{args.dataset_tag}] wrote {plot_path}")

    return base


def _run_species_metric(metric_name: str, compute_fn):
    def run(args: argparse.Namespace):
        io_module.set_cache_dir(args.cache_dir)
        files = _read_file_list(args)
        df = compute_fn(files, species=args.species, max_failures=args.max_file_failures)
        _write_outputs(df, metric_name, args, n_files=len(files))
        return df

    return run


def _run_fake_rate(args: argparse.Namespace):
    io_module.set_cache_dir(args.cache_dir)
    files = _read_file_list(args)
    df = fake_rate_module.compute_fake_rate(
        files, max_failures=args.max_file_failures
    )
    _write_outputs(df, FAKE_RATE_NAME, args, n_files=len(files))
    return df


def _run_plot(args: argparse.Namespace):
    """Re-render the metric-vs-pT PNG(s) for an existing JSON result, without
    touching the grid — used to restyle a finished run's plots (e.g. after a
    plotting change). Reads the value/err column spec from _PLOT_SPEC so the
    output matches what the metric run itself would produce."""
    df, meta = report.read_json(args.json)
    metric_name = args.metric or meta.get("metric")
    if not metric_name:
        raise SystemExit("Cannot infer metric: pass --metric or use a JSON generated by a metric run.")
    base = args.json[:-5] if args.json.endswith(".json") else args.json
    group_col = "species" if metric_name != FAKE_RATE_NAME else "__no_species_axis__"
    n_plotted = 0
    for value_col, err_col, suffix, log_y, ymin, marker_loc in _plot_specs(metric_name):
        plot_path = f"{base}_{suffix}.png"
        group_col = "species" if metric_name != FAKE_RATE_NAME else "__no_species_axis__"
        group_color = "black" if metric_name == FAKE_RATE_NAME else None
        report.plot_metric_vs_pt(
            df, value_col, err_col, plot_path, group_col=group_col,
            marker_col="eta_bin_center", group_color=group_color, log_y=log_y,
            ymin=ymin, marker_legend_loc=marker_loc,
        )
        print(f"[plot/{metric_name}] wrote {plot_path}")
        n_plotted += 1
    if n_plotted == 0:
        print(f"[plot/{metric_name}] no plot configured for metric {metric_name!r} (JSON/markdown only)")
    return df


def _run_compare(args: argparse.Namespace):
    os.makedirs(args.out_dir, exist_ok=True)
    out_base = os.path.join(args.out_dir, f"{args.metric}_comparison")
    df = compare_module.compare_metric(args.clean, args.bkg, out_base + ".json", out_base + ".md")
    print(f"[compare/{args.metric}] wrote {out_base}.json and {out_base}.md ({len(df)} bins compared)")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trkperf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in _SPECIES_METRIC_FUNCTIONS.items():
        p = sub.add_parser(name, help=f"Compute the {name} metric.")
        _add_common_args(p)
        p.add_argument(
            "--species",
            nargs="*",
            default=None,
            choices=list(config.SPECIES),
            metavar="SPECIES",
            help="Restrict to these species (default: all species in config.SPECIES).",
        )
        p.set_defaults(func=_run_species_metric(name, fn))

    p = sub.add_parser(FAKE_RATE_NAME, help="Compute the fake-rate metric (no species axis).")
    _add_common_args(p)
    p.set_defaults(func=_run_fake_rate)

    p = sub.add_parser("compare", help="Compare a metric's clean vs bkg_mixed JSON results.")
    p.add_argument("--metric", required=True, help="Metric name, used only for output file naming.")
    p.add_argument("--clean", required=True, help="Path to the clean-run JSON output.")
    p.add_argument("--bkg", required=True, help="Path to the bkg_mixed-run JSON output.")
    p.add_argument("--out-dir", default="output")
    p.set_defaults(func=_run_compare)

    p = sub.add_parser(
        "plot",
        help="Re-render the metric-vs-pT PNG(s) for an existing JSON result "
        "(no grid access) - useful after a plotting change.",
    )
    p.add_argument("--json", required=True, help="Path to a metric-run JSON output.")
    p.add_argument(
        "--metric",
        default=None,
        help="Metric name for the plot spec; default: inferred from the JSON metadata.",
    )
    p.set_defaults(func=_run_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
