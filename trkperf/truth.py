"""Truth-level (MCParticles) kinematics and acceptance bookkeeping.

Everything in this project is measured *relative to* the truth GEANT
particle: this module is where that reference is built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, io

#: Branch paths for the MCParticles collection members this project needs.
#: Keep this in one place so a future addition (e.g. vertex position) only
#: needs to touch this dict, not every call site.
_MCPARTICLES_COLUMNS = {
    "pdg": "MCParticles/MCParticles.PDG",
    "generator_status": "MCParticles/MCParticles.generatorStatus",
    "px": "MCParticles/MCParticles.momentum.x",
    "py": "MCParticles/MCParticles.momentum.y",
    "pz": "MCParticles/MCParticles.momentum.z",
    "mass": "MCParticles/MCParticles.mass",
}


def add_kinematics(df: pd.DataFrame, px="px", py="py", pz="pz", prefix="") -> pd.DataFrame:
    """Add p, pt, eta, phi columns computed from Cartesian momentum components.

    Public on purpose: reco.py imports this exact function rather than
    reimplementing the same formulas, so the exact same formulas (and the
    same edge-case handling) are used on both the truth and reco side of
    every comparison - a resolution measured between two *different* eta
    definitions would be meaningless.
    """
    p = np.sqrt(df[px] ** 2 + df[py] ** 2 + df[pz] ** 2)
    pt = np.sqrt(df[px] ** 2 + df[py] ** 2)
    # theta undefined at p == 0 (should not occur for a real particle, but
    # guard against division by zero rather than let it raise or emit NaN
    # silently further downstream).
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.clip(df[pz] / p, -1.0, 1.0))
        eta = -np.log(np.tan(theta / 2.0))
    phi = np.arctan2(df[py], df[px])

    df[f"{prefix}p"] = p
    df[f"{prefix}pt"] = pt
    df[f"{prefix}eta"] = eta
    df[f"{prefix}phi"] = phi
    return df


def read_truth_particles(
    file_paths: list[str], *, max_failures: int = 0, primary_only: bool = False,
    shared_failures: dict | None = None,
) -> pd.DataFrame:
    """Read MCParticles from every file and compute pT/eta/phi/p.

    Parameters
    ----------
    primary_only:
        Drop non-primary particles (``generator_status != 1``) per-file,
        *before* accumulation, instead of keeping every beam-background
        particle in memory. The +background sample carries millions of
        background MCParticles per file, so the unfiltered table for even a
        few dozen files far exceeds this machine's RAM; every metric in this
        package down-selects to primaries immediately afterwards
        (``select_primary``), so ``primary_only=True`` gives identical bins
        at a fraction of the peak memory. The read cache always holds the
        RAW table regardless of this flag (see
        ``io.read_flat_multi.per_file_filter``), so no cached data is ever
        lost for a consumer that needs the full ancestor list.

    Returns
    -------
    DataFrame with columns: file_id, event, idx, pdg, generator_status,
    px, py, pz, mass, p, pt, eta, phi, species (name from
    config.PDG_TO_SPECIES, or None if the PDG code is not one we track).
    """
    filter_ = None
    if primary_only:
        filter_ = lambda df: df[df["generator_status"] == 1].copy()
    df = io.read_flat_multi(
        file_paths, _MCPARTICLES_COLUMNS, max_failures=max_failures, per_file_filter=filter_,
        shared_failures=shared_failures,
    )
    df = add_kinematics(df)
    df["species"] = df["pdg"].map(config.PDG_TO_SPECIES)
    return df


def select_primary(truth_df: pd.DataFrame, species: list[str] | None = None) -> pd.DataFrame:
    """Keep only generator-level primary particles (generatorStatus == 1).

    Parameters
    ----------
    species:
        Optional list of species names (keys of config.SPECIES) to keep.
        None means "every species trkperf knows about" (i.e. drop rows whose
        PDG code did not map to a known species at all).
    """
    out = truth_df[truth_df["generator_status"] == 1]
    if species is None:
        out = out[out["species"].notna()]
    else:
        out = out[out["species"].isin(species)]
    return out


def _collection_present(file_paths: list[str], branch: str, n_probe_files: int = 5) -> bool:
    """Does this campaign actually carry this branch (metadata-cheap probe)?

    A campaign entry is not proof (cf. pid.features._available): probe with a
    one-entry read so a never-written collection is skipped without spending
    the shared failure budget file by file.

    Robustness matters here because a wrong answer silently zeroes a whole
    collection for the run: files that fail to OPEN (transient xrootd drops)
    are skipped in favour of the next file — only a successfully opened file
    whose key list lacks the branch counts as genuine absence. If no file
    opens at all, return True so the normal read path fails loudly through
    the shared failure budget instead of this probe quietly zeroing data.
    """
    for path in (file_paths or [])[:n_probe_files]:
        try:
            tree = io.open_tree(path)
        except Exception:  # noqa: BLE001 - transient open failure; try next file
            continue
        try:
            tree.arrays([branch], library="ak", entry_start=0, entry_stop=1)
            return True
        except Exception:  # noqa: BLE001 - file opened but branch absent
            return False
    return True


def _keep_particle_filter(keep_particles: pd.DataFrame):
    """Build a ``read_flat_multi`` per-file filter keeping only hit rows of
    particles listed in `keep_particles` (columns file_id, event, idx).

    The (event, idx) pair is packed into one int64 key per file so the
    membership test is a single ``np.isin`` over the raw table. Safe because
    event and MCParticles-position are far below 2**31 per event.
    """
    keys = ((keep_particles["event"].astype("int64").to_numpy() << 32)
            | keep_particles["idx"].astype("int64").to_numpy())
    staged: dict = {}
    for fid, key in zip(keep_particles["file_id"].to_numpy(), keys):
        staged.setdefault(int(fid), []).append(key)
    by_file = {fid: np.unique(np.asarray(v, dtype="int64"))
               for fid, v in staged.items()}

    def filt(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        allowed = by_file.get(int(df["file_id"].iloc[0]))
        if allowed is None:
            return df.iloc[0:0]
        row_keys = ((df["event"].astype("int64").to_numpy() << 32)
                    | df["particle_idx"].astype("int64").to_numpy())
        return df[np.isin(row_keys, allowed)]

    return filt


def read_truth_hit_layer_counts(file_paths: list[str], *, max_failures: int = 0,
                                 shared_failures: dict | None = None,
                                 collections: tuple[str, ...] | None = None,
                                 found_collections: list | None = None,
                                 keep_particles: pd.DataFrame | None = None,
                                 ) -> pd.DataFrame:
    """Count, per truth particle, how many central-tracking truth-hit
    collections registered >= 1 hit from it (see AGENTS.md / config.py for
    why "collection" is used as a proxy for "layer").

    `collections` selects the detector region (see
    ``config.TRACKING_REGIONS``); None means the central seven. A collection
    whose branch is absent in a campaign degrades to 0 hits with a printed
    NOTE rather than aborting the run. Every collection actually read is
    appended to `found_collections` (when given) so the caller can record in
    its output metadata which collections the numbers rest on — a run that
    silently degraded must be distinguishable from a full one after the fact.

    Returns
    -------
    DataFrame with columns file_id, event, idx (idx = the MCParticles
    position this row's counts refer to), n_layers_hit.

    Particles with zero hits in every collection do not appear in the
    intermediate per-collection tables (there is nothing to count), so
    callers must left-merge this onto the full truth-particle table and
    fill missing values with 0 - see acceptance.compute_acceptance for the
    canonical example.

    `keep_particles` (optional DataFrame with file_id, event, idx — e.g. the
    truth-primary table the caller is about to merge onto) restricts the
    accumulation to those particles, per file, BEFORE concatenation. On the
    +background sample a raw truth-hit collection holds millions of
    beam-background particles per file; hoarding every one of their rows for
    a run's full file list is what OOM-killed the first 26.07.1 bkg
    acceptance. Every consumer of this function left-merges onto exactly
    this particle set (acceptance/efficiency via add_acceptance_flag,
    pid.regions onto matched truth), so the filtered accumulation is
    bit-identical to the full one. The read cache still holds the RAW
    per-file tables (io.read_flat_multi stashes before filtering), so a
    consumer that needs the full ancestor list loses nothing.
    """
    import sys

    if collections is None:
        collections = config.CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS
    filt = _keep_particle_filter(keep_particles) if keep_particles is not None else None
    frames = []
    for collection in collections:
        # Campaign-wide absence (a collection a production never wrote) must
        # NOT consume the shared failure budget: probing the first file keeps
        # a missing branch from failing every file one by one and shrinking
        # the whole run's sample. Per-file I/O failures below still do.
        branch = f"_{collection}_particle/_{collection}_particle.index"
        if not _collection_present(file_paths, branch):
            print(f"[truth] NOTE: {collection} has no {branch} branch in this "
                  "campaign; counting 0 hits from it in this run", file=sys.stderr)
            continue
        # Each truth hit has exactly one `particle` relation (OneToOne) back
        # to the MCParticles entry that produced it. Name the requested
        # value column "particle_idx" (NOT "idx") - read_flat already uses
        # "idx" for the hit's own bookkeeping position, and re-using that
        # name would produce two columns both called "idx" after the rename
        # below (pandas allows duplicate column names; groupby then breaks,
        # since `df["idx"]` returns a 2-column DataFrame, not a Series).
        columns = {"particle_idx": branch}
        try:
            hits = io.read_flat_multi(file_paths, columns, max_failures=max_failures,
                                      shared_failures=shared_failures,
                                      per_file_filter=filt)
        except Exception as exc:  # noqa: BLE001 - unreadable collection
            print(f"[truth] NOTE: {collection} unreadable ({exc.__class__.__name__}); "
                  "counting 0 hits from it in this run", file=sys.stderr)
            continue
        hits["collection"] = collection
        # Select only the columns we need FIRST (dropping the hit's own,
        # now-irrelevant bookkeeping "idx"), then rename - so there is only
        # ever one "idx" column, meaning "position within MCParticles".
        hits = hits[["file_id", "event", "particle_idx", "collection"]].rename(
            columns={"particle_idx": "idx"}
        )
        frames.append(hits)
        if found_collections is not None:
            found_collections.append(collection)

    if not frames:
        # No collection readable: return the contract columns with their
        # real dtypes (object-dtype empties poison downstream merges with
        # "Invalid value '[]'" setitem errors instead of clean NaN counts).
        return pd.DataFrame({c: pd.Series(dtype=d) for c, d in
                             (("file_id", "int64"), ("event", "int64"),
                              ("idx", "int64"), ("n_layers_hit", "int64"))})

    all_hits = pd.concat(frames, ignore_index=True)
    counts = (
        all_hits.groupby(["file_id", "event", "idx"])["collection"]
        .nunique()
        .reset_index(name="n_layers_hit")
    )
    return counts


def add_acceptance_flag(
    truth_df: pd.DataFrame,
    layer_counts: pd.DataFrame,
    min_layers: int = config.ACCEPTANCE_MIN_LAYERS,
) -> pd.DataFrame:
    """Left-merge truth-hit layer counts onto a truth-particle table and add
    an `in_acceptance` boolean column.

    Particles absent from `layer_counts` (no hits in any tracked collection)
    are treated as n_layers_hit = 0, not dropped.
    """
    merged = truth_df.merge(
        layer_counts, on=["file_id", "event", "idx"], how="left"
    )
    merged["n_layers_hit"] = merged["n_layers_hit"].fillna(0).astype(int)
    merged["in_acceptance"] = merged["n_layers_hit"] >= min_layers
    return merged
