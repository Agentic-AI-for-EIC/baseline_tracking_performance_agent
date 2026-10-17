"""Detector-region selection for PID performance plots.

The tracking half of this project measures acceptance/efficiency/resolution
per detector region (``trkperf``: barrel ``|eta| < 1.5``, forward/backward
endcaps beyond) with a region-correct truth-hit rule — every region requires
``>= 4`` total individual truth hits of which ``>= 3`` are silicon
(``trkperf.config.TRACKING_REGIONS``; the silicon core is the shared CKF seed
requirement). PID performance plots use
the same split: one plot per region, each drawn on the candidates that
satisfy that region's rule.

What this module does
---------------------
:func:`attach` adds three columns to a PID score frame (``test_scores.pkl`` /
``all_scores.pkl`` shape):

* ``eta_region`` — ``"barrel"`` / ``"forward endcap"`` / ``"backward endcap"``
  from the candidate's **truth** ``eta`` (geometry decides the rule,   exactly
  as in tracking; the same ``trkperf.report.eta_region`` function, so the
  ``|eta| = 1.5`` boundary cannot drift between the two halves), ``"unknown"``
  where the truth partner is missing;
* ``n_hits_total`` / ``n_hits_si`` — individual truth hits, total and in
  silicon, counted over the
  candidate's **own region rule** collections (NaN where unclassifiable);
* ``in_acceptance`` — ``>= ACCEPTANCE_MIN_HITS total hits AND >=
  ACCEPTANCE_MIN_SILICON_HITS of them silicon`` for that rule (CKF seed
  requirement; trkperf.config, 2026-10 criterion).

The truth-hit counts come from :mod:`trkperf.truth` (same reader, same
``{particle_idx: branch}`` column scheme, so the ``trkperf`` io cache is
shared, not duplicated). Files that fail the hit read cannot be classified:
their rows get ``in_acceptance = False`` and are counted in the returned
``info`` dict, never silently kept.

Deliberately NOT here
---------------------
* No model input: ``eta_region`` / ``in_acceptance`` / ``n_hits_total`` /
  ``n_hits_si`` are
  truth-derived and live in :data:`pid.dataset.KEY_COLUMNS` (never-model).
* No retraining: the classifiers keep their leg-scoped training samples;
  the region selection applies at *plot* time (``pid evaluate`` /
  ``pid performance --eta-region``), exactly like the tracking
  ``--eta-region`` plot filter.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from trkperf import config as tk_config
from trkperf import io as tk_io
from trkperf import report as tk_report
from trkperf import truth as tk_truth

#: Detector regions usable for PID region plots (same names and the same
#: |eta| = trkperf.config.BARREL_ETA_MAX (1.5) boundary as trkperf.report.eta_region).
ETA_REGIONS: tuple[str, ...] = ("barrel", "forward endcap", "backward endcap")

#: eta_region -> the trkperf tracking-region rule that defines its
#: acceptance (every region: ">= 4 total truth hits, >= 3 silicon").
REGION_RULE: dict[str, str] = {
    "barrel": "central",
    "forward endcap": "forward",
    "backward endcap": "backward",
}

#: Filename slug per region (spaces break nothing, but slugs read better and
#: match the tracking grouped-plot convention).
REGION_SLUGS: dict[str, str] = {
    "barrel": "barrel",
    "forward endcap": "forward_endcap",
    "backward endcap": "backward_endcap",
}


def describe_rule(region: str) -> str:
    """Short human description of the acceptance rule, for titles and meta."""
    rule = tk_config.TRACKING_REGIONS[REGION_RULE[region]]
    return (f">= {rule['min_hits']} hits (>= {rule['min_silicon_hits']} silicon) over "
            f"{len(rule['collections'])} {REGION_RULE[region]} collections")


def resolve_files(features: pd.DataFrame) -> list[str]:
    """Ordered file list matching the positional ``file_id`` convention.

    ``file_id`` is stamped positionally by ``trkperf.io.read_flat_multi``,
    so sorting the feature table's ``(file_id, source_file)`` pairs recovers
    the exact list order the ids refer to.
    """
    order = (features[["file_id", "source_file"]].drop_duplicates()
             .sort_values("file_id"))
    return order["source_file"].tolist()


def attach(frame: pd.DataFrame, *, features: pd.DataFrame,
           regions: tuple[str, ...] = ETA_REGIONS,
           max_failures: int = 0, cache_dir: str | None = None,
           ) -> tuple[pd.DataFrame, dict]:
    """Add ``eta_region`` / ``n_hits_total`` / ``n_hits_si`` / ``in_acceptance``
    to a score frame.

    Parameters
    ----------
    frame:
        Score table (``test_scores.pkl`` / ``all_scores.pkl`` shape) with
        ``file_id``, ``event``, ``track_idx`` and ``truth_eta`` columns.
    features:
        The feature table the scores were produced from (carries
        ``source_file`` and ``truth_idx``). Every score row must find its
        feature row, otherwise the file list the truth reads need cannot be
        trusted and the run refuses rather than attaching to the wrong file.
    regions:
        Subset of :data:`ETA_REGIONS` to classify (only these rules are
        read; rows outside them still get ``eta_region`` but no acceptance).
    max_failures, cache_dir:
        Passed to the truth-hit reads (flaky-endpoint flags, same meaning
        as in trkperf).

    Returns ``(frame_with_columns, info)``; ``info`` records the per-region
    rule, collections actually found, skipped files and unclassified rows
    for the caller's output metadata.
    """
    unknown = [r for r in regions if r not in ETA_REGIONS]
    if unknown:
        raise ValueError(f"unknown PID region(s) {unknown}; choices {list(ETA_REGIONS)}")
    for col in ("file_id", "event", "track_idx", "truth_eta"):
        if col not in frame.columns:
            raise KeyError(
                f"pid.regions.attach: score frame lacks {col!r} - re-train with "
                "the current pid.train (it writes truth kinematics into "
                "test_scores.pkl/all_scores.pkl), or drop --eta-region.")
    for col in ("file_id", "event", "track_idx", "source_file", "truth_idx"):
        if col not in features.columns:
            raise KeyError(
                f"pid.regions.attach: feature table lacks {col!r} - pass the "
                "table the model trained on via --features.")

    key = ["file_id", "event", "track_idx"]
    link = features[key + ["source_file", "truth_idx"]].copy()
    merged = frame.merge(link, on=key, how="left", validate="one_to_one",
                         indicator=True)
    n_missing = int((merged["_merge"] != "both").sum())
    if n_missing:
        raise SystemExit(
            f"pid.regions.attach: {n_missing}/{len(frame)} score rows find no "
            "feature row (file_id/event/track_idx) - the --features table does "
            "not match these scores. Pass the table the model trained on.")
    merged = merged.drop(columns=["_merge"])
    files = resolve_files(features)
    if cache_dir:
        tk_io.set_cache_dir(cache_dir)

    out = merged
    out["eta_region"] = out["truth_eta"].map(tk_report.eta_region)
    out["n_hits_total"] = np.nan
    out["n_hits_si"] = np.nan
    out["in_acceptance"] = False
    info: dict = {"regions": {}, "skipped_files": [],
                  "n_unclassified": 0, "n_files": len(files)}
    shared: dict = {}
    # Restrict the truth-hit reads to the particles these rows actually merge
    # onto: the raw background hit tables are millions of rows per file and
    # accumulating all of them OOMs (see trkperf.truth docstring).
    keep_particles = (merged[["file_id", "event", "truth_idx"]]
                      .dropna(subset=["truth_idx"])
                      .astype({"truth_idx": "int64"})
                      .rename(columns={"truth_idx": "idx"})
                      .drop_duplicates())
    for region in regions:
        rule_name = REGION_RULE[region]
        rule = tk_config.TRACKING_REGIONS[rule_name]
        found: list = []
        try:
            counts = tk_truth.read_truth_hit_counts(
                files, max_failures=max_failures, shared_failures=shared,
                collections=rule["collections"], found_collections=found,
                keep_particles=keep_particles)
        except RuntimeError as exc:
            raise SystemExit(f"pid.regions.attach: {exc}") from exc
        hit = counts.rename(columns={"idx": "truth_idx",
                                     "n_hits_total": f"_tot_{rule_name}",
                                     "n_hits_si": f"_si_{rule_name}"})
        out = out.merge(hit[["file_id", "event", "truth_idx",
                             f"_tot_{rule_name}", f"_si_{rule_name}"]],
                        on=["file_id", "event", "truth_idx"], how="left")
        # A matched particle with zero hits in every collection of this rule
        # has no count row (trkperf.truth contract) - that is a genuine 0,
        # not "unclassifiable". Fill it, but never for rows whose file was
        # skipped (no read happened) or whose truth partner is missing
        # (truth_idx NaN never joins).
        skipped_ids = {i for i, p in enumerate(files)
                       if p in set(shared.get("skipped", []))}
        tot_col, si_col = f"_tot_{rule_name}", f"_si_{rule_name}"
        has_truth = out["truth_idx"].notna()
        readable = ~out["file_id"].isin(skipped_ids)
        ok_rows = has_truth & readable
        out.loc[ok_rows, tot_col] = out.loc[ok_rows, tot_col].fillna(0)
        out.loc[ok_rows, si_col] = out.loc[ok_rows, si_col].fillna(0)
        sel = out["eta_region"] == region
        out.loc[sel, "n_hits_total"] = out.loc[sel, tot_col]
        out.loc[sel, "n_hits_si"] = out.loc[sel, si_col]
        out.loc[sel, "in_acceptance"] = (
            (out.loc[sel, tot_col].fillna(-1) >= rule["min_hits"])
            & (out.loc[sel, si_col].fillna(-1) >= rule["min_silicon_hits"]))
        out = out.drop(columns=[tot_col, si_col])
        if set(found) != set(rule["collections"]):
            missing = sorted(set(rule['collections']) - set(found))
            print(f"[regions] WARNING: {region} acceptance degraded "
                  f"({len(found)}/{len(rule['collections'])} collections read; "
                  f"missing {missing}) - recorded in output metadata, "
                  "do not quote these numbers as full-rule acceptance",
                  file=sys.stderr)
        info["regions"][region] = {
            "rule": rule_name, "min_hits": rule["min_hits"],
            "min_silicon_hits": rule["min_silicon_hits"],
            "collections": list(rule["collections"]),
            "collections_found": list(found),
            "n_rows": int(sel.sum()),
            "n_in_acceptance": int((sel & out["in_acceptance"]).sum()),
        }
    info["skipped_files"] = sorted(shared.get("skipped", []))
    # Unclassifiable = region inside the request but no count (unknown
    # region, unmatched truth, or a skipped file).
    need = out["eta_region"].isin(list(regions))
    info["n_unclassified"] = int(
        (need & out["n_hits_total"].isna()).sum())
    parts = "; ".join(
        f"{r}: {info['regions'][r]['n_in_acceptance']}/"
        f"{info['regions'][r]['n_rows']} in acceptance" for r in regions)
    print(f"[regions] attached {list(regions)} acceptance ({parts}; "
          f"{len(info['skipped_files'])} files skipped, "
          f"{info['n_unclassified']} rows unclassifiable)", file=sys.stderr)
    return out, info
