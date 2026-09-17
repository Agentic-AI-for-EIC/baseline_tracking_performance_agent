"""Track-finding efficiency: what fraction of truth particles were actually
reconstructed (matched to a CentralCKFTracks entry), reported both within
geometric acceptance and absolute (over all generated truth particles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import binning, config, matching, reco, truth


def compute_efficiency(
    file_paths: list[str],
    species: list[str] | None = None,
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
    min_layers: int = config.ACCEPTANCE_MIN_LAYERS,
    max_failures: int = 0,
) -> pd.DataFrame:
    """Compute the track-finding-efficiency table for the given files.

    Returns
    -------
    One row per (species, pt_bin, eta_bin):
        species, pt_bin, eta_bin, pt_bin_center, eta_bin_center,
        n_truth (all generated), n_in_acceptance, n_matched,
        efficiency_within_acceptance, efficiency_within_acceptance_err,
        efficiency_absolute, efficiency_absolute_err, insufficient_stats.

    Both efficiencies are measured directly from data (matched / relevant
    denominator), not derived by multiplying a separately-computed
    acceptance table by a conditional efficiency - so
    efficiency_absolute / (n_in_acceptance / n_truth) need not exactly equal
    efficiency_within_acceptance for a given run; compare.py / report.py can
    cross-check this against metrics.acceptance's own output as a
    consistency check (see AGENTS.md).
    """
    shared: dict = {}
    truth_df = truth.read_truth_particles(
        file_paths, max_failures=max_failures, primary_only=True, shared_failures=shared
    )
    truth_df = truth.select_primary(truth_df, species=species)

    layer_counts = truth.read_truth_hit_layer_counts(
        file_paths, max_failures=max_failures, shared_failures=shared
    )
    truth_df = truth.add_acceptance_flag(truth_df, layer_counts, min_layers=min_layers)

    reco_df = reco.read_reco_tracks(file_paths, max_failures=max_failures, shared_failures=shared)
    assoc_df = matching.read_associations(
        file_paths, max_failures=max_failures, shared_failures=shared
    )
    pairs = matching.build_matched_pairs(truth_df, reco_df, assoc_df, weight_threshold)

    pairs = binning.assign_bins(pairs)
    pairs = pairs.dropna(subset=["pt_bin", "eta_bin"])

    # Precompute as a plain column (rather than a groupby-agg lambda that
    # reaches back into the outer frame) - simpler to read and avoids any
    # doubt about index alignment inside the aggregation.
    pairs["matched_in_acceptance"] = pairs["is_matched"] & pairs["in_acceptance"]

    group_cols = ["species", "pt_bin", "eta_bin", "pt_bin_center", "eta_bin_center"]
    grouped = pairs.groupby(group_cols, observed=True)

    result = grouped.agg(
        n_truth=("is_matched", "size"),
        n_in_acceptance=("in_acceptance", "sum"),
        n_matched=("is_matched", "sum"),
        n_matched_in_acceptance=("matched_in_acceptance", "sum"),
    ).reset_index()

    # Bins with zero entries are absent from the groupby; right-join the full
    # grid so they are explicitly reported as insufficient statistics.
    grid = binning.full_bin_grid(species if species is not None else list(config.SPECIES))
    result = result.merge(
        grid, on=["species", "pt_bin", "eta_bin"], how="right", suffixes=("", "_grid")
    )
    result["pt_bin_center"] = result["pt_bin_center"].fillna(result.pop("pt_bin_center_grid"))
    result["eta_bin_center"] = result["eta_bin_center"].fillna(result.pop("eta_bin_center_grid"))
    for col in ("n_truth", "n_in_acceptance", "n_matched", "n_matched_in_acceptance"):
        result[col] = result[col].fillna(0).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        result["efficiency_absolute"] = result["n_matched"] / result["n_truth"]
        result["efficiency_within_acceptance"] = (
            result["n_matched_in_acceptance"] / result["n_in_acceptance"]
        )
    result["efficiency_absolute_err"] = binning.binomial_error(
        result["n_matched"], result["n_truth"]
    )

    result["efficiency_within_acceptance_err"] = binning.binomial_error(
        result["n_matched_in_acceptance"], result["n_in_acceptance"]
    )

    # The statistics floor applies to whichever denominator is smaller
    # (within-acceptance, since n_in_acceptance <= n_truth always, so one
    # check covers both denominators).
    result["insufficient_stats"] = result["n_in_acceptance"] < config.MIN_ENTRIES_PER_BIN
    result.attrs["skipped_files"] = sorted(shared.get("skipped", []))
    result.attrs["run_params"] = {
        "weight_threshold": weight_threshold, "min_layers": min_layers, "species": species
    }
    return result.drop(columns=["n_matched_in_acceptance"])
