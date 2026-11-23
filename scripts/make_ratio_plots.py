#!/usr/bin/env python3
"""Regenerate the 13 bkg/clean RATIO plots in output/plot_tracking/.

These are produced outside `trkperf plot` (which handles base/grouped/window/
eta-slice figures), so the reproduction driver calls this script in its
auxiliary stage - without it, the ratio figures silently keep the previous
methodology (pre option-B / pre-N-weighting) after a regeneration sweep.

Set (from the CENTRAL efficiency/resolution tables; regions are eta slices
of the same table, matching how the tracking grouped plots slice them):
- efficiency_ratio_bkg_over_clean_{within_acceptance,absolute}_{allEta,barrel,
  backward_endcap,forward_endcap}_vs_p.png   (y in [0.5, 1.5])
- resolution_ratio_bkg_over_clean_{allEta,barrel,backward_endcap,
  forward_endcap,barrel_window_-1.00_+1.00}_vs_p.png   (y in [0.5, 5])

Usage: python3 scripts/make_ratio_plots.py [--plots-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trkperf import report  # noqa: E402

# (tag, value_col, err_col, count_cols, label) per ratio curve family;
# `tag` is the file-name component used by the committed plots.
FAMILIES = {
    "efficiency": [
        ("within_acceptance", "efficiency_within_acceptance",
         "efficiency_within_acceptance_err",
         ("n_matched_in_acceptance", "n_in_acceptance"), "efficiency (within acceptance)"),
        ("absolute", "efficiency_absolute", "efficiency_absolute_err",
         ("n_matched", "n_truth"), "efficiency (absolute)"),
    ],
    "resolution": [
        # tag "" keeps the established file names
        # (resolution_ratio_bkg_over_clean_<region>_vs_p.png)
        ("", "sigma", "sigma_err", None, "momentum resolution sigma"),
    ],
}
# family -> (ymin, ymax) as agreed for the deliverables
YLIM = {"efficiency": (0.5, 1.5), "resolution": (0.5, 5.0)}
# region key in filename -> (eta_region_filter, eta_window)
REGIONS = {
    "allEta": (None, None),
    "barrel": ("barrel", None),
    "backward_endcap": ("backward endcap", None),
    "forward_endcap": ("forward endcap", None),
}
RESOLUTION_EXTRA = {"barrel_window_-1.00_+1.00": ("barrel", (-1.0, 1.0))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plots-dir", default="output/plot_tracking")
    ap.add_argument("--out-dir", default="output",
                    help="directory holding the central metric JSONs")
    args = ap.parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)
    n = 0
    for family, values in FAMILIES.items():
        clean_p = os.path.join(args.out_dir, f"{family}_clean26071.json")
        bkg_p = os.path.join(args.out_dir, f"{family}_bkg26071.json")
        if not (os.path.exists(clean_p) and os.path.exists(bkg_p)):
            print(f"[ratio] SKIP {family}: missing {clean_p} or {bkg_p}")
            continue
        clean_df, _ = report.read_json(clean_p)
        bkg_df, _ = report.read_json(bkg_p)
        regions = dict(REGIONS)
        if family == "resolution":
            regions.update(RESOLUTION_EXTRA)
        ymin, ymax = YLIM[family]
        for tag, vcol, ecol, ccols, label in values:
            for rkey, (rfilter, rwindow) in regions.items():
                path = os.path.join(
                    args.plots_dir,
                    f"{family}_ratio_bkg_over_clean_"
                    + (f"{tag}_" if tag else "") + f"{rkey}_vs_p.png")
                title = (f"{label}  background/clean vs p "
                         f"[{rkey.replace('_', ' ')}]  "
                         f"(y in [{ymin},{ymax}])")
                report.plot_bkg_over_clean_ratio(
                    clean_df, bkg_df, vcol, ecol, path,
                    count_cols=ccols, eta_region_filter=rfilter,
                    eta_window=rwindow, ymin=ymin, ymax=ymax,
                    title=title, y_label=f"{label}  (bkg / clean)")
                print(f"[ratio] wrote {path}")
                n += 1
    print(f"[ratio] {n} plots")
    return 0 if n == 13 else 1


if __name__ == "__main__":
    raise SystemExit(main())
