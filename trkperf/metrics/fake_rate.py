"""Fake rate: what fraction of reconstructed tracks have no genuine truth
match, binned in the *reconstructed* track's own (p, eta).

Deliberately no species axis here - a fake track has no true species by
construction (see AGENTS.md). Deliberately no acceptance dependence either -
fake rate is a statement about the reconstruction algorithm's response to
combinatorics/occupancy, not about truth-particle geometry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import binning, config, matching, reco


def compute_fake_rate(
    file_paths: list[str],
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
    max_failures: int = 0,
    min_reco_hits: int = 0,
    min_reco_si_hits: int = 0,
) -> pd.DataFrame:
    """Compute the fake-rate table for the given files.

    Callers MUST NOT pass a single-particle-gun file list here (see
    AGENTS.md: ~0 fakes by construction, not a meaningful measurement). This
    function does not itself check the file list's provenance - the SKILL.md
    procedure and the CLI are responsible for warning if given a gun sample.

    Optional reco-side quality cut (both default 0 = off): when
    ``min_reco_hits`` / ``min_reco_si_hits`` are set, a reconstructed track is
    only counted (fake or real) if it itself carries at least that many
    tracker hits (silicon + gaseous; TOF timing excluded) and at least that
    many silicon hits, per ``reco.read_reco_track_hit_counts``. Note CentralCKF
    already requires a >=3-silicon seed, so a 3/4 cut removes ~1% of tracks
    (near-tautological); a *stricter* cut is where it starts to bite.

    Returns
    -------
    One row per (p_bin, eta_bin):
        p_bin, eta_bin, p_bin_center, eta_bin_center, n_tracks, n_fake,
        fake_rate, fake_rate_err, insufficient_stats.
    """
    shared: dict = {}
    reco_df = reco.read_reco_tracks(
        file_paths, max_failures=max_failures, shared_failures=shared
    )
    assoc_df = matching.read_associations(
        file_paths, max_failures=max_failures, shared_failures=shared
    )

    flagged = matching.find_fake_tracks(reco_df, assoc_df, weight_threshold)

    n_reco_cut_removed = 0
    if min_reco_hits > 0 or min_reco_si_hits > 0:
        counts = reco.read_reco_track_hit_counts(
            file_paths, max_failures=max_failures, shared_failures=shared)
        flagged = flagged.merge(counts, on=["file_id", "event", "idx"], how="left")
        nh = flagged["n_reco_hits"].fillna(-1)
        ns = flagged["n_reco_si_hits"].fillna(-1)
        keep = (nh >= min_reco_hits) & (ns >= min_reco_si_hits)
        n_reco_cut_removed = int((~keep).sum())
        flagged = flagged[keep]

    flagged = binning.assign_bins(flagged)
    # Tracks with unmeasurable kinematics (qOverP == 0 or NaN angles) can never
    # be binned: count them separately instead of silently dropping them from
    # the denominator. Out-of-range but finite kinematics below are ordinary
    # phase-space exclusions, not this.
    n_unmeasurable = int(flagged["p"].isna().sum())
    if n_unmeasurable:
        print(
            f"[fake-rate] WARNING: {n_unmeasurable} reconstructed tracks have "
            "unmeasurable (NaN) p/eta and cannot be binned - counted in "
            "run_params, excluded from n_tracks.",
        )
    flagged = flagged.dropna(subset=["p_bin", "eta_bin"])

    group_cols = ["p_bin", "eta_bin", "p_bin_center", "eta_bin_center"]
    result = (
        flagged.groupby(group_cols, observed=True)
        .agg(n_tracks=("is_fake", "size"), n_fake=("is_fake", "sum"))
        .reset_index()
    )
    # Bins with zero entries are absent from the groupby; right-join the full
    # grid so they are explicitly reported as insufficient statistics.
    grid = binning.full_bin_grid(None)
    result = result.merge(
        grid, on=["p_bin", "eta_bin"], how="right", suffixes=("", "_grid")
    )
    result["p_bin_center"] = result["p_bin_center"].fillna(result.pop("p_bin_center_grid"))
    result["eta_bin_center"] = result["eta_bin_center"].fillna(result.pop("eta_bin_center_grid"))
    result["n_tracks"] = result["n_tracks"].fillna(0).astype(int)
    result["n_fake"] = result["n_fake"].fillna(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["fake_rate"] = result["n_fake"] / result["n_tracks"]
    result["fake_rate_err"] = binning.binomial_error(result["n_fake"], result["n_tracks"])
    result["insufficient_stats"] = result["n_tracks"] < config.MIN_ENTRIES_PER_BIN
    result.attrs["skipped_files"] = sorted(shared.get("skipped", []))
    result.attrs["run_params"] = {
        "weight_threshold": weight_threshold,
        "n_unmeasurable_tracks": n_unmeasurable,
        "min_reco_hits": min_reco_hits,
        "min_reco_si_hits": min_reco_si_hits,
        "n_reco_cut_removed": n_reco_cut_removed,
    }
    return result
