"""Geometric acceptance: what fraction of generated truth particles could in
principle be reconstructed (>= N truth-tracking layers hit), independent of
whether the reconstruction algorithm actually succeeded.

This is the only metric that needs no truth<->reco matching at all - it is
purely a property of the truth particle and the detector geometry it passed
through, which is also why it serves as a cross-check between the clean and
background-mixed samples (see AGENTS.md: acceptance should not depend on
beam background).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import binning, config, truth


def compute_acceptance(
    file_paths: list[str],
    species: list[str] | None = None,
    min_layers: int | None = None,
    max_failures: int = 0,
    region: str = "central",
) -> pd.DataFrame:
    """Compute the acceptance table for the given files.

    Parameters
    ----------
    file_paths:
        Local paths or root:// URLs to read (already resolved by the caller,
        e.g. via the rucio MCP server - this function does no discovery).
    species:
        Species names (keys of config.SPECIES) to include. None = every
        species trkperf knows about.
    min_layers:
        Override config.ACCEPTANCE_MIN_LAYERS if needed for a study of the
        threshold's own sensitivity.

    Returns
    -------
    One row per (species, pt_bin, eta_bin):
        species, pt_bin, eta_bin, pt_bin_center, eta_bin_center,
        n_generated, n_in_acceptance, acceptance, acceptance_err,
        insufficient_stats.

    `region` selects the detector region whose truth-hit collections define
    "could be reconstructed" (see config.TRACKING_REGIONS); an explicit
    `min_layers` overrides the region default.
    """
    if region not in config.TRACKING_REGIONS:
        raise ValueError(f"unknown tracking region {region!r}; "
                         f"choices {sorted(config.TRACKING_REGIONS)}")
    spec = config.TRACKING_REGIONS[region]
    if min_layers is None:
        min_layers = spec["min_layers"]
    truth_df = truth.read_truth_particles(
        file_paths, max_failures=max_failures, primary_only=True,
        shared_failures=(shared := {}),
    )
    truth_df = truth.select_primary(truth_df, species=species)

    layer_counts = truth.read_truth_hit_layer_counts(
        file_paths, max_failures=max_failures, shared_failures=shared,
        collections=spec["collections"], found_collections=(found := []),
    )
    truth_df = truth.add_acceptance_flag(truth_df, layer_counts, min_layers=min_layers)

    truth_df = binning.assign_bins(truth_df)
    truth_df = truth_df.dropna(subset=["pt_bin", "eta_bin"])

    grouped = truth_df.groupby(
        ["species", "pt_bin", "eta_bin", "pt_bin_center", "eta_bin_center"], observed=True
    )
    result = grouped.agg(
        n_generated=("in_acceptance", "size"),
        n_in_acceptance=("in_acceptance", "sum"),
    ).reset_index()

    # Bins with zero entries are absent from the groupby; right-join the full
    # grid so they are explicitly reported as insufficient statistics.
    grid = binning.full_bin_grid(species if species is not None else list(config.SPECIES))
    result = result.merge(
        grid, on=["species", "pt_bin", "eta_bin"], how="right", suffixes=("", "_grid")
    )
    result["pt_bin_center"] = result["pt_bin_center"].fillna(result.pop("pt_bin_center_grid"))
    result["eta_bin_center"] = result["eta_bin_center"].fillna(result.pop("eta_bin_center_grid"))
    result["n_generated"] = result["n_generated"].fillna(0).astype(int)
    result["n_in_acceptance"] = result["n_in_acceptance"].fillna(0).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        result["acceptance"] = result["n_in_acceptance"] / result["n_generated"]
    result["acceptance_err"] = binning.binomial_error(
        result["n_in_acceptance"], result["n_generated"]
    )
    result["insufficient_stats"] = result["n_generated"] < config.MIN_ENTRIES_PER_BIN
    result.attrs["skipped_files"] = sorted(shared.get("skipped", []))
    result.attrs["run_params"] = {"min_layers": min_layers, "species": species,
                                  "region": region,
                                  "collections": list(spec["collections"]),
                                  "collections_found": found}
    return result
