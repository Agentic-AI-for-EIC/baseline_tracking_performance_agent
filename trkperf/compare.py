"""Compare a metric's result between the clean and background-mixed
dataset types: per-bin ratio (bkg/clean) and difference (bkg-clean), with
simple propagated uncertainties.

This module is pure post-processing - it only reads the JSON that
report.to_json already wrote for two prior runs, so it works unchanged on
any future metric's output as long as that metric also used
report.to_json/read_json and follows the "value column + matching
<value>_err column" convention already used by every metric in metrics/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import report

#: Candidate bin-identifying columns to join clean vs bkg results on. Only
#: the ones actually present in both DataFrames are used (fake_rate has no
#: "species" column, for example - see AGENTS.md). String bin labels come
#: first: float centers survive a JSON round-trip exactly in practice, but
#: joining on the human-meaningful interval strings is robust by
#: construction, and near-tie floats can never silently misalign rows.
_JOIN_KEY_CANDIDATES = ("species", "truth_species", "reco_species", "pt_bin", "eta_bin",
                         "pt_bin_center", "eta_bin_center")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that returns NaN (not inf/warning) where the
    denominator is zero or either operand is already NaN."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator, np.nan, dtype=float)
    ok = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[ok] = numerator[ok] / denominator[ok]
    return result


def _detect_join_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    return [c for c in _JOIN_KEY_CANDIDATES if c in df_a.columns and c in df_b.columns]


def _detect_value_err_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Every `<name>_err` column paired with its `<name>` value column, for
    whichever names actually exist. Generic on purpose: a future metric
    module needs no changes here as long as it follows the same
    value/value_err naming convention already used throughout metrics/."""
    pairs = []
    for col in df.columns:
        if col.endswith("_err"):
            value_col = col[: -len("_err")]
            if value_col in df.columns:
                pairs.append((value_col, col))
    return pairs


def compare_metric(
    clean_json_path: str,
    bkg_json_path: str,
    out_json_path: str | None = None,
    out_md_path: str | None = None,
    join_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Load two same-metric result files and compute the clean-vs-background
    comparison.

    Parameters
    ----------
    join_keys:
        Columns identifying a row in both tables. Default: auto-detect from
        ``species / reco_species / pt_bin_center / eta_bin_center``.

    Returns
    -------
    Outer-joined DataFrame on the detected bin-identity columns, with
    `<value>_clean`, `<value>_bkg`, `<value>_ratio` (+`_err`),
    `<value>_diff` (+`_err`) for every value/err column pair detected in the
    clean-side result, plus a combined `insufficient_stats` flag (True if
    either side flagged that bin).
    """
    clean_df, clean_meta = report.read_json(clean_json_path)
    bkg_df, bkg_meta = report.read_json(bkg_json_path)

    # `join_keys` lets a caller with a non-tracking-shaped table (e.g. the PID
    # results, whose rows are keyed by quantity/signal/target fake rate rather
    # than by (pt_bin, eta_bin, species)) state its identity columns explicitly
    # instead of relying on the candidate list below.
    join_keys = list(join_keys) if join_keys else _detect_join_keys(clean_df, bkg_df)
    if not join_keys:
        raise ValueError(
            "compare_metric: no common bin-identity columns between the two "
            "results - are these really the same metric's output?"
        )
    # Detect value/err pairs on BOTH sides: a pair present on only one side
    # must still appear (with NaNs opposite) rather than silently vanish. A
    # one-sided pair keeps its UNSUFFIXED name through the merge, so attribute
    # it to its side explicitly.
    clean_pairs = set(_detect_value_err_pairs(clean_df))
    value_err_pairs = _detect_value_err_pairs(clean_df)
    for pair in _detect_value_err_pairs(bkg_df):
        if pair not in value_err_pairs:
            value_err_pairs.append(pair)

    merged = clean_df.merge(
        bkg_df, on=join_keys, how="outer", suffixes=("_clean", "_bkg")
    )
    for value_col, err_col in value_err_pairs:
        for stem in (value_col, err_col):
            clean_c, bkg_c = stem + "_clean", stem + "_bkg"
            if clean_c in merged.columns or bkg_c in merged.columns or stem not in merged.columns:
                continue
            if (value_col, err_col) in clean_pairs:
                merged = merged.rename(columns={stem: clean_c})
                merged[bkg_c] = np.nan
            else:
                merged = merged.rename(columns={stem: bkg_c})
                merged[clean_c] = np.nan

    for value_col, err_col in value_err_pairs:
        # Coerce first: JSON nulls arrive as None/object and must become NaN,
        # never a to_numpy(dtype=float) exception.
        clean_val = pd.to_numeric(merged[f"{value_col}_clean"], errors="coerce").to_numpy(dtype=float)
        bkg_val = pd.to_numeric(merged[f"{value_col}_bkg"], errors="coerce").to_numpy(dtype=float)
        clean_err = pd.to_numeric(merged[f"{err_col}_clean"], errors="coerce").to_numpy(dtype=float)
        bkg_err = pd.to_numeric(merged[f"{err_col}_bkg"], errors="coerce").to_numpy(dtype=float)

        ratio = _safe_divide(bkg_val, clean_val)
        # Standard error propagation for a ratio R = b/c:
        # (dR/R)^2 = (db/b)^2 + (dc/c)^2
        rel_err_sq = _safe_divide(bkg_err, bkg_val) ** 2 + _safe_divide(clean_err, clean_val) ** 2
        ratio_err = np.abs(ratio) * np.sqrt(rel_err_sq)
        # A well-defined ratio of exactly zero still carries an error: R = 0
        # comes from b = 0, so dR = db / |c| (the generic formula above gives
        # NaN through 0/0).
        zero = (ratio == 0) & np.isfinite(bkg_err) & np.isfinite(clean_val) & (clean_val != 0)
        ratio_err = np.where(
            zero, np.abs(bkg_err) / np.abs(np.where(zero, clean_val, 1.0)), ratio_err
        )

        diff = bkg_val - clean_val
        diff_err = np.sqrt(clean_err**2 + bkg_err**2)

        merged[f"{value_col}_ratio_bkg_over_clean"] = ratio
        merged[f"{value_col}_ratio_bkg_over_clean_err"] = ratio_err
        merged[f"{value_col}_diff_bkg_minus_clean"] = diff
        merged[f"{value_col}_diff_bkg_minus_clean_err"] = diff_err

    if "insufficient_stats_clean" in merged.columns or "insufficient_stats_bkg" in merged.columns:
        clean_flag = merged.get("insufficient_stats_clean", pd.Series(True, index=merged.index)).fillna(True)
        bkg_flag = merged.get("insufficient_stats_bkg", pd.Series(True, index=merged.index)).fillna(True)
        merged["insufficient_stats"] = clean_flag | bkg_flag

    meta = {
        "comparison": "bkg_mixed vs clean",
        "clean_source_meta": clean_meta,
        "bkg_source_meta": bkg_meta,
    }
    if out_json_path:
        report.to_json(merged, out_json_path, meta=meta)
    if out_md_path:
        report.to_markdown_table(merged, out_md_path)
    return merged
