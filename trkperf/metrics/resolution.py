"""Momentum resolution: Delta(p)/p between the reconstructed and the true
GEANT track, for truth-matched pairs, Gaussian-fit per (species, p_bin,
eta_bin). (Criterion change 2026-10: the project measures and bins in total
momentum p; this was Delta(pT)/pT before.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import binning, config, matching, reco, truth


def _chunk_residuals(
    file_paths: list[str],
    species: list[str] | None,
    weight_threshold: float,
    max_failures: int,
    shared: dict,
) -> pd.DataFrame:
    """Per-chunk stage of the resolution computation: read truth, reco and
    associations for one file-list chunk, build the matched pairs, and keep
    only the slim (bin keys, Delta(p)/p) frame the per-bin fits need.

    The wide truth/reco intermediates are dropped with the call frame, so
    peak memory stays proportional to one chunk. `shared` (skip-set +
    failure budget) is caller-owned and accumulates across chunks. The
    Delta(p)/p values (including any non-finite ones) are passed through
    exactly as the single pass would produce them - `binning.fit_gaussian_core`
    filters non-finite entries at fit time, so fitted sigmas are identical.
    """
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
    with np.errstate(divide="ignore", invalid="ignore"):
        matched["delta_p_over_p"] = (matched["reco_p"] - matched["p"]) / matched["p"]

    matched = binning.assign_bins(matched)
    matched = matched.dropna(subset=["p_bin", "eta_bin"])

    return matched[
        ["species", "p_bin", "eta_bin", "p_bin_center", "eta_bin_center",
         "delta_p_over_p"]
    ]


def compute_resolution(
    file_paths: list[str],
    species: list[str] | None = None,
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
    max_failures: int = 0,
    chunk_files: int = 0,
) -> pd.DataFrame:
    """Compute the momentum-resolution table for the given files.

    Returns
    -------
    One row per (species, p_bin, eta_bin):
        species, p_bin, eta_bin, p_bin_center, eta_bin_center, n_matched,
        mu, sigma, mu_err, sigma_err, chi2_ndf, insufficient_stats,
        fit_converged.

    `mu`/`sigma` are on Delta(p)/p = (p_reco - p_truth) / p_truth, with truth
    p defining the bin (2026-10 criterion change: the whole project bins and
    resolves in total momentum; this superseded the earlier Delta(pT)/pT
    definition).

    `chunk_files` bounds peak memory by processing the file list in
    consecutive groups of at most that many files (0/None = one pass, the
    historical behaviour). Gaussian-core fits are not additive, so chunks
    are merged at the residual level - the per-bin fit loop then sees
    exactly the same values as a single pass and returns identical numbers.
    """
    shared: dict = {}
    n_files = len(file_paths)
    if chunk_files and chunk_files > 0 and chunk_files < n_files:
        chunks = [file_paths[i:i + chunk_files] for i in range(0, n_files, chunk_files)]
    else:
        chunks = [list(file_paths)]
    matched = pd.concat(
        [
            _chunk_residuals(c, species, weight_threshold, max_failures, shared)
            for c in chunks
        ],
        ignore_index=True,
    )

    rows = []
    group_cols = ["species", "p_bin", "eta_bin", "p_bin_center", "eta_bin_center"]
    for keys, group in matched.groupby(group_cols, observed=True):
        species_name, p_bin, eta_bin, pt_center, eta_center = keys
        n_matched = len(group)
        insufficient = n_matched < config.MIN_ENTRIES_PER_BIN
        if insufficient:
            fit = binning.GaussianFitResult(n_matched, *([float("nan")] * 4), float("nan"), 0, False)
        else:
            fit = binning.fit_gaussian_core(group["delta_p_over_p"].to_numpy())
        rows.append(
            {
                "species": species_name,
                "p_bin": p_bin,
                "eta_bin": eta_bin,
                "p_bin_center": pt_center,
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
    result = pd.DataFrame(rows)
    # Bins with zero matches never enter the loop; right-join the full grid so
    # they are explicitly reported as insufficient statistics.
    grid = binning.full_bin_grid(species if species is not None else list(config.SPECIES))
    if result.empty:
        result = pd.DataFrame(
            {
                "species": grid["species"],
                "p_bin": grid["p_bin"],
                "eta_bin": grid["eta_bin"],
                "p_bin_center": grid["p_bin_center"],
                "eta_bin_center": grid["eta_bin_center"],
                "n_matched": 0,
                "mu": float("nan"),
                "sigma": float("nan"),
                "mu_err": float("nan"),
                "sigma_err": float("nan"),
                "chi2_ndf": float("nan"),
                "fit_converged": False,
                "insufficient_stats": True,
            }
        )
    else:
        result = result.merge(
            grid, on=["species", "p_bin", "eta_bin"], how="right", suffixes=("", "_grid")
        )
        result["p_bin_center"] = result["p_bin_center"].fillna(result.pop("p_bin_center_grid"))
        result["eta_bin_center"] = result["eta_bin_center"].fillna(result.pop("eta_bin_center_grid"))
    result["n_matched"] = result["n_matched"].fillna(0).astype(int)
    result["insufficient_stats"] = result["insufficient_stats"].fillna(True).astype(bool)
    result["fit_converged"] = result["fit_converged"].fillna(False).astype(bool)
    result.attrs["skipped_files"] = sorted(shared.get("skipped", []))
    result.attrs["run_params"] = {"weight_threshold": weight_threshold, "species": species,
                                  "chunk_files": chunk_files or 0, "n_chunks": len(chunks)}
    return result
