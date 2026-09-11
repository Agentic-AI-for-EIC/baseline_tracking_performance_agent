"""Fake rate: what fraction of reconstructed tracks have no genuine truth
match, binned in the *reconstructed* track's own (pT, eta).

Deliberately no species axis here - a fake track has no true species by
construction (see AGENTS.md). Deliberately no acceptance dependence either -
fake rate is a statement about the reconstruction algorithm's response to
combinatorics/occupancy, not about truth-particle geometry.
"""

from __future__ import annotations

import pandas as pd

from .. import binning, config, matching, reco


def compute_fake_rate(
    file_paths: list[str],
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
    max_failures: int = 0,
) -> pd.DataFrame:
    """Compute the fake-rate table for the given files.

    Callers MUST NOT pass a single-particle-gun file list here (see
    AGENTS.md: ~0 fakes by construction, not a meaningful measurement). This
    function does not itself check the file list's provenance - the SKILL.md
    procedure and the CLI are responsible for warning if given a gun sample.

    Returns
    -------
    One row per (pt_bin, eta_bin):
        pt_bin, eta_bin, pt_bin_center, eta_bin_center, n_tracks, n_fake,
        fake_rate, fake_rate_err, insufficient_stats.
    """
    reco_df = reco.read_reco_tracks(file_paths, max_failures=max_failures)
    assoc_df = matching.read_associations(file_paths, max_failures=max_failures)

    flagged = matching.find_fake_tracks(reco_df, assoc_df, weight_threshold)
    flagged = binning.assign_bins(flagged)
    flagged = flagged.dropna(subset=["pt_bin", "eta_bin"])

    group_cols = ["pt_bin", "eta_bin", "pt_bin_center", "eta_bin_center"]
    result = (
        flagged.groupby(group_cols, observed=True)
        .agg(n_tracks=("is_fake", "size"), n_fake=("is_fake", "sum"))
        .reset_index()
    )
    result["fake_rate"] = result["n_fake"] / result["n_tracks"]
    result["fake_rate_err"] = binning.binomial_error(result["n_fake"], result["n_tracks"])
    result["insufficient_stats"] = result["n_tracks"] < config.MIN_ENTRIES_PER_BIN
    return result
