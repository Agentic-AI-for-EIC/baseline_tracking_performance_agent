#!/usr/bin/env python3
"""Regenerate the 13 bkg/clean RATIO plots in output/plot_tracking/.

These are produced outside `trkperf plot` (which handles base/grouped/window/
eta-slice figures), so the reproduction driver calls this script in its
auxiliary stage - without it, the ratio figures silently keep the previous
methodology (pre option-B / pre-N-weighting) after a regeneration sweep.

Region rule sourcing (mirrors the driver's grouped-plot pairing):
- efficiency: each eta slice comes from the JSON of its OWN acceptance rule -
  barrel <- efficiency_<tag>.json (central rule), backward endcap <-
  efficiency_backward_<tag>.json, forward endcap <- efficiency_forward_<tag>.
  json. n_in_acceptance/n_matched_in_acceptance of the central table are
  evaluated with the CENTRAL collection set at every eta, so slicing it for
  an endcap curve would mix a different truth rule into the ratio (the
  region="auto" reco gate does not match it either). allEta = the three
  rule-correct slices concatenated. efficiency_absolute is rule-independent,
  so the same stitched frames serve it unchanged.
- resolution: no truth acceptance involved - the central table is correct for
  every eta slice, incl. the barrel |eta|<1.0 window.

Set:
- efficiency_ratio_bkg_over_clean_{within_acceptance,absolute}_{allEta,barrel,
  backward_endcap,forward_endcap}_vs_p.png   (y in [0.5, 1.5])
- resolution_ratio_bkg_over_clean_{allEta,barrel,backward_endcap,
  forward_endcap,barrel_window_-1.00_+1.00}_vs_p.png   (y in [0.5, 5])

Usage: python3 scripts/make_ratio_plots.py [--plots-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trkperf import report  # noqa: E402

# region filename-key -> eta_region label (report.eta_region vocabulary)
REGION_LABEL = {"barrel": "barrel", "backward_endcap": "backward endcap",
                "forward_endcap": "forward endcap"}
# efficiency region-key -> rule suffix in the JSON file name ("" = central)
EFF_RULE = {"barrel": "", "backward_endcap": "backward",
            "forward_endcap": "forward"}

EFF_VALUES = [
    ("within_acceptance", "efficiency_within_acceptance",
     "efficiency_within_acceptance_err",
     ("n_matched_in_acceptance", "n_in_acceptance"), "efficiency (within acceptance)"),
    ("absolute", "efficiency_absolute", "efficiency_absolute_err",
     ("n_matched", "n_truth"), "efficiency (absolute)"),
]
RES_VALUES = [
    # tag "" keeps the established file names
    ("", "sigma", "sigma_err", None, "momentum resolution sigma"),
]
RES_REGIONS = dict(REGION_LABEL, allEta=None)
YLIM = {"efficiency": (0.5, 1.5), "resolution": (0.5, 5.0)}
CL = {"efficiency": "Wilson 95% CL", "resolution": "1σ"}


def _load(path):
    df, _ = report.read_json(path)
    return df


def eff_stitched(kind: str, out_dir: str) -> pd.DataFrame:
    """Concatenated per-region efficiency rows, each from its OWN rule JSON."""
    frames = []
    for rkey, rule in EFF_RULE.items():
        name = f"efficiency{('_' + rule) if rule else ''}_{kind}26071.json"
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = _load(path)
        frames.append(df[df["eta_bin_center"].map(report.eta_region)
                         == REGION_LABEL[rkey]])
    return pd.concat(frames, ignore_index=True)


def _plot_ratio(family, clean_df, bkg_df, values, regions, plots_dir, out_dir):
    ymin, ymax = YLIM[family]
    n = 0
    for tag, vcol, ecol, ccols, label in values:
        for rkey, _ in regions.items():
            path = os.path.join(
                plots_dir,
                f"{family}_ratio_bkg_over_clean_"
                + (f"{tag}_" if tag else "") + f"{rkey}_vs_p.png")
            # eta slice filter; the barrel window is a sub-slice of barrel
            ef = REGION_LABEL.get(rkey)
            window = None
            if rkey.startswith("barrel_window"):
                ef, window = "barrel", (-1.0, 1.0)
            report.plot_bkg_over_clean_ratio(
                clean_df, bkg_df, vcol, ecol, path,
                count_cols=ccols, eta_region_filter=ef, eta_window=window,
                ymin=ymin, ymax=ymax, cl_hint=CL[family],
                title=f"{label}  background/clean vs p "
                      f"[{rkey.replace('_', ' ')}]  (y in [{ymin},{ymax}])",
                y_label=f"{label}  (bkg / clean)")
            print(f"[ratio] wrote {path}")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plots-dir", default="output/plot_tracking")
    ap.add_argument("--out-dir", default="output",
                    help="directory holding the metric JSONs")
    args = ap.parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)
    n = 0
    try:
        ec, eb = eff_stitched("clean", args.out_dir), eff_stitched("bkg", args.out_dir)
    except FileNotFoundError as exc:
        print(f"[ratio] SKIP efficiency: missing input {exc}")
        ec = eb = None
    if ec is not None:
        n += _plot_ratio("efficiency", ec, eb, EFF_VALUES,
                         dict(REGION_LABEL, allEta=None), args.plots_dir, args.out_dir)
    rp, bp = (os.path.join(args.out_dir, f"resolution_{t}26071.json")
              for t in ("clean", "bkg"))
    if os.path.exists(rp) and os.path.exists(bp):
        n += _plot_ratio("resolution", _load(rp), _load(bp), RES_VALUES,
                         {"allEta": None, **REGION_LABEL,
                          "barrel_window_-1.00_+1.00": None},
                         args.plots_dir, args.out_dir)
    else:
        print(f"[ratio] SKIP resolution: missing {rp} or {bp}")
    print(f"[ratio] {n} plots")
    return 0 if n == 13 else 1


if __name__ == "__main__":
    raise SystemExit(main())
