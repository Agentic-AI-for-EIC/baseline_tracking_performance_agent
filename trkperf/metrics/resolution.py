"""Momentum resolution: Delta(pT)/pT between the reconstructed and the true
GEANT track, for truth-matched pairs, Gaussian-fit per (species, pt_bin,
eta_bin).
"""

from __future__ import annotations

import pandas as pd

from .. import binning, config, matching, reco, truth


def compute_resolution(
    file_paths: list[str],
    species: list[str] | None = None,
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
    max_failures: int = 0,
) -> pd.DataFrame:
    """Compute the momentum-resolution table for the given files.

    Returns
    -------
    One row per (species, pt_bin, eta_bin):
        species, pt_bin, eta_bin, pt_bin_center, eta_bin_center, n_matched,
        mu, sigma, mu_err, sigma_err, chi2_ndf, insufficient_stats,
        fit_converged.

    `mu`/`sigma` are on Delta(pT)/pT = (pT_reco - pT_truth) / pT_truth (see
    AGENTS.md - this project defines resolution on pT specifically, not
    total momentum, for consistency with the pT/eta binning used
    everywhere).
    """
    truth_df = truth.read_truth_particles(file_paths, max_failures=max_failures, primary_only=True)
    truth_df = truth.select_primary(truth_df, species=species)

    reco_df = reco.read_reco_tracks(file_paths, max_failures=max_failures)
    assoc_df = matching.read_associations(file_paths, max_failures=max_failures)

    pairs = matching.build_matched_pairs(truth_df, reco_df, assoc_df, weight_threshold)
    matched = pairs[pairs["is_matched"]].copy()
    matched["delta_pt_over_pt"] = (matched["reco_pt"] - matched["pt"]) / matched["pt"]

    matched = binning.assign_bins(matched)
    matched = matched.dropna(subset=["pt_bin", "eta_bin"])

    rows = []
    group_cols = ["species", "pt_bin", "eta_bin", "pt_bin_center", "eta_bin_center"]
    for keys, group in matched.groupby(group_cols, observed=True):
        species_name, pt_bin, eta_bin, pt_center, eta_center = keys
        n_matched = len(group)
        insufficient = n_matched < config.MIN_ENTRIES_PER_BIN
        if insufficient:
            fit = binning.GaussianFitResult(n_matched, *([float("nan")] * 4), float("nan"), 0, False)
        else:
            fit = binning.fit_gaussian_core(group["delta_pt_over_pt"].to_numpy())
        rows.append(
            {
                "species": species_name,
                "pt_bin": pt_bin,
                "eta_bin": eta_bin,
                "pt_bin_center": pt_center,
                "eta_bin_center": eta_center,
                "n_matched": n_matched,
                "mu": fit.mu,
                "sigma": fit.sigma,
                "mu_err": fit.mu_err,
                "sigma_err": fit.sigma_err,
                "chi2_ndf": fit.chi2_over_ndf,
                "fit_converged": fit.converged,
                "insufficient_stats": insufficient,
            }
        )
    return pd.DataFrame(rows)
