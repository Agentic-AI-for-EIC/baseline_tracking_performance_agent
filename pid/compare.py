"""Clean-vs-background comparison of PID results.

Delegates the arithmetic to :func:`trkperf.compare.compare_metric`, which already
defines the project's convention: per-bin ratio and difference with propagated
uncertainties, plus a combined ``insufficient_stats`` flag. This module only
locates the matching pair of PID artifacts and records the PID-specific
provenance (task, model library, campaign, gate states on both sides), so the
PID comparison is produced the same way as the five tracking comparisons and can
be read with the same tools.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from trkperf import compare as tk_compare

from . import config


def _read_json_meta(path: str) -> dict:
    """`trkperf.report.to_json` writes {"meta": ..., "data": [...]} (key: `meta`)."""
    with open(path) as fh:
        return json.load(fh).get("meta", {})


#: Row-identity columns per PID artifact. trkperf.compare auto-detects the
#: tracking metrics' (species, pt_bin_center, eta_bin_center); PID tables are keyed
#: differently, and guessing would silently compare an electron row against a
#: pion row, so each artifact declares its identity explicitly. String bin
#: labels are preferred over float centres: centres survive a JSON round-trip
#: exactly in practice, but labels are robust by construction.
JOIN_KEYS: dict[str, list[str]] = {
    "overall": ["quantity", "signal", "against", "target_fake_rate"],
    "vs_pt": ["variable", "signal", "against", "bin"],
    "vs_eta": ["variable", "signal", "against", "bin"],
    "vs_p": ["variable", "signal", "against", "bin"],
    "confusion": ["truth_class", "pred_class"],
    "calibration": ["bin"],
    "importance-gain": ["column"],
    "importance-shap": ["column"],
    "importance-permutation": ["column"],
}
#: Artifacts that must never go through the row-join comparison, with the
#: reason (checked before JOIN_KEYS so the generic "add keys" error cannot
#: invite a meaningless join).
_REFUSED_ARTIFACTS: dict[str, str] = {
    "roc": ("ROC curve points have no stable row identity across tags - the two "
            "operating-point grids differ, so a join is a cartesian product. "
            "Overlay the two pid-*-roc.png figures instead."),
}


def compare_artifact(name: str, *, task: str, model: str = "",
                     clean: str | None = None, bkg: str | None = None,
                     out_dir: str = config.OUTPUT_DIR) -> pd.DataFrame:
    """Compare ``output/pid-<task>_<model>_<tag>-<name>.json`` for both tags.

    ``clean``/``bkg`` override the two input paths; ``name`` is the artifact
    (``overall``, ``vs_pt``, ``vs_eta``, ``confusion``, ``roc``, an importance
    kind...).
    """
    stem = f"{task}_{model}_" if model else f"{task}_"
    clean = clean or os.path.join(out_dir, f"pid-{stem}clean-{name}.json")
    bkg = bkg or os.path.join(out_dir, f"pid-{stem}bkg_mixed-{name}.json")
    for path, role in ((clean, "clean"), (bkg, "background")):
        if not os.path.exists(path):
            raise SystemExit(
                f"pid.compare: missing the {role}-side input {path}. Run "
                f"`python -m pid evaluate --task {task} --dataset-tag <tag>` "
                "for that tag first (and, for a multiclass comparison, train "
                "that task on both samples).")
    pair = _read_json_meta(clean).get("dataset_tag"), _read_json_meta(bkg).get("dataset_tag")
    pair_tag = "" if pair == ("clean", "bkg_mixed") or pair == (None, None) \
        else f"_{pair[0]}-vs-{pair[1]}"
    out_stem = os.path.join(out_dir, f"pid-{stem}{name}{pair_tag}_comparison")
    if name in _REFUSED_ARTIFACTS:
        raise SystemExit(f"pid.compare: refusing artifact {name!r}: {_REFUSED_ARTIFACTS[name]}")
    keys = [k for k in JOIN_KEYS.get(name, []) ]
    if not keys:
        raise SystemExit(
            f"pid.compare: artifact {name!r} has no declared row-identity columns; "
            "add them to pid.compare.JOIN_KEYS so the two datasets are matched on "
            "the same physical row (guessing would compare, e.g., an electron row "
            "against a pion row).")
    df = tk_compare.compare_metric(clean, bkg, out_stem + ".json", out_stem + ".md",
                                   join_keys=keys)
    meta_clean, meta_bkg = _read_json_meta(clean), _read_json_meta(bkg)
    # Provenance labels name the ACTUAL tags compared (any pair sharing this
    # schema works: clean/bkg_mixed, clean26071/bkg26071, ...).
    clean_tag = pair[0] or "clean"
    bkg_tag = pair[1] or "bkg_mixed"

    def _meta_count(meta: dict, *names: str):
        for name in names:
            if meta.get(name) is not None:
                return meta.get(name)
        return None

    summary = {
        "artifact": name, "task": task, "model": model,
        "n_bins_compared": int(len(df)),
        clean_tag: {"campaign": meta_clean.get("campaign"),
                    "n_files": _meta_count(meta_clean, "n_files_scored", "n_files"),
                    "n_rows": _meta_count(meta_clean, "n_rows_scored", "n_rows"),
                    "insufficient": int(
                        df.get("insufficient_stats", pd.Series(dtype=bool)).sum())
                        if "insufficient_stats" in df else None},
        bkg_tag: {"campaign": meta_bkg.get("campaign"),
                  "n_files": _meta_count(meta_bkg, "n_files_scored", "n_files"),
                  "n_rows": _meta_count(meta_bkg, "n_rows_scored", "n_rows")},
    }
    with open(out_stem + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[compare/{name}] wrote {out_stem}.json and .md "
          f"({len(df)} rows compared) + {out_stem}.summary.json")
    return df


def compare_all(*, task: str, model: str = "", out_dir: str = config.OUTPUT_DIR,
                artifacts: tuple[str, ...] = ("overall", "vs_pt", "vs_eta")) -> dict:
    """Compare every clean/background tag pair that has both sides evaluated."""
    out = {}
    stem = f"{task}_{model}_" if model else f"{task}_"
    tag_pairs = config.DATASET_TAG_PAIRS
    for name in artifacts:
        for clean_tag, bkg_tag in tag_pairs:
            clean = os.path.join(out_dir, f"pid-{stem}{clean_tag}-{name}.json")
            bkg = os.path.join(out_dir, f"pid-{stem}{bkg_tag}-{name}.json")
            if os.path.exists(clean) and os.path.exists(bkg):
                out[f"{name}:{clean_tag}-vs-{bkg_tag}"] = compare_artifact(
                    name, task=task, model=model, clean=clean, bkg=bkg,
                    out_dir=out_dir)
        if not any(k.startswith(f"{name}:") for k in out):
            print(f"[compare] skipping {name}: no tag pair has both sides present yet")
    return out
