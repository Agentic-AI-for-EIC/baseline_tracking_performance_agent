"""PID confusion matrix: given a truth particle of species X that was
successfully track-matched, what species does the reconstruction's
mass-hypothesis PID assign it, and how often is that right?

Row-normalised fractions (truth-normalised): for each truth species, the
fraction classified as each reco species — answers "how often does the
reconstructor confuse π for K?" Column-normalised fractions
(reco-normalised): for each reco species, the fraction that truly came
from each truth species — answers "how pure is the reconstructed K
sample?"

No species-axis restriction on the output: a particle with a reco_pdg
code outside config.SPECIES is mapped to "unknown" so the user can see
exactly what fraction of tracks the reconstructor could not classify
within the tracked species set.
"""

from __future__ import annotations

import pandas as pd

from .. import binning, config, matching, reco, truth


def compute_pid_confusion(
    file_paths: list[str],
    species: list[str] | None = None,
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
    max_failures: int = 0,
) -> pd.DataFrame:
    """Compute the PID confusion-matrix table for the given files.

    Returns
    -------
    One row per (pt_bin, eta_bin, truth_species, reco_species):
        pt_bin, eta_bin, pt_bin_center, eta_bin_center,
        truth_species, reco_species,
        n_matchedinbin,                        # matches of this (truth, reco)
                                              # pair in this bin
        confusion_frac, confusion_frac_err,   # row-normalised: n_ij / n_i
        efficiency, efficiency_err,           # n_i / n_truth_total_in_bin
        n_truth_total_in_bin,                 # denominator for efficiency
        insufficient_stats.
    """
    shared: dict = {}
    truth_df = truth.read_truth_particles(
        file_paths, max_failures=max_failures, primary_only=True, shared_failures=shared
    )
    truth_df = truth.select_primary(truth_df, species=species)
    reco_df = reco.read_reco_tracks(file_paths, max_failures=max_failures, shared_failures=shared)
    assoc_df = matching.read_associations(
        file_paths, max_failures=max_failures, shared_failures=shared
    )

    pairs = matching.build_matched_pairs(truth_df, reco_df, assoc_df, weight_threshold)
    matched = pairs[pairs["is_matched"]].copy()
    if matched.empty:
        out = pd.DataFrame()
        out.attrs["skipped_files"] = sorted(shared.get("skipped", []))
        out.attrs["run_params"] = {"weight_threshold": weight_threshold, "species": species}
        return out

    matched["reco_species"] = matched["reco_pdg"].map(config.PDG_TO_SPECIES).fillna("unknown")

    matched = binning.assign_bins(matched)
    matched = matched.dropna(subset=["pt_bin", "eta_bin"])

    bin_cols = ["pt_bin", "eta_bin", "pt_bin_center", "eta_bin_center"]
    group_cols = [*bin_cols, "species", "reco_species"]

    counts = (
        matched.groupby(group_cols, observed=True)
        .size()
        .reset_index(name="n_matchedinbin")
    )

    # Total matched truth particles per (bin, truth species).
    truth_totals = (
        counts.groupby(bin_cols + ["species"], observed=True)["n_matchedinbin"]
        .sum()
        .reset_index(name="n_truth_total_in_bin")
    )

    merged = counts.merge(truth_totals, on=bin_cols + ["species"], how="left")
    merged["confusion_frac"] = merged["n_matchedinbin"] / merged["n_truth_total_in_bin"]
    merged["confusion_frac_err"] = binning.binomial_error(
        merged["n_matchedinbin"], merged["n_truth_total_in_bin"]
    )

    # Per-bin total of all matched truth particles (all species combined).
    bin_totals = (
        merged.groupby(bin_cols, observed=True)["n_matchedinbin"]
        .sum()
        .reset_index(name="n_bin_total")
    )
    merged = merged.merge(bin_totals, on=bin_cols, how="left")
    merged["efficiency"] = merged["n_truth_total_in_bin"] / merged["n_bin_total"]
    merged["efficiency_err"] = binning.binomial_error(
        merged["n_truth_total_in_bin"], merged["n_bin_total"]
    )

    merged["insufficient_stats"] = (
        merged["n_truth_total_in_bin"] < config.MIN_ENTRIES_PER_BIN
    )

    merged = merged.rename(columns={"species": "truth_species"})
    merged = merged.drop(columns=["n_bin_total"])
    merged.attrs["skipped_files"] = sorted(shared.get("skipped", []))
    merged.attrs["run_params"] = {"weight_threshold": weight_threshold, "species": species}
    return merged
