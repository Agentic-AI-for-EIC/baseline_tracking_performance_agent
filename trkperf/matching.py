"""Truth<->reconstructed matching (CentralCKFTrackAssociations).

This module is the single choke point of the whole project: it produces the
one matched-pairs table that resolution.py and efficiency.py both consume,
and the fake-track table that fake_rate.py consumes. A future
particle-identification module should also build on `build_matched_pairs`
rather than re-deriving truth<->reco correspondence its own way - see
trkperf/pid/__init__.py.
"""

from __future__ import annotations

import pandas as pd

from . import config, io

_ASSOCIATIONS_COLUMNS = {
    "weight": "CentralCKFTrackAssociations/CentralCKFTrackAssociations.weight",
    "rec_idx": "_CentralCKFTrackAssociations_rec/_CentralCKFTrackAssociations_rec.index",
    "sim_idx": "_CentralCKFTrackAssociations_sim/_CentralCKFTrackAssociations_sim.index",
}

_JOIN_KEYS = ["file_id", "event"]


def read_associations(file_paths: list[str], *, max_failures: int = 0) -> pd.DataFrame:
    """Read CentralCKFTrackAssociations from every file.

    Returns
    -------
    DataFrame with columns file_id, event, idx (the association's own
    position - not otherwise meaningful), weight, rec_idx (position within
    CentralCKFTracks), sim_idx (position within MCParticles).

    NOTE on collectionID: EDM4eic relations are technically {index,
    collectionID} pairs, since podio allows multiple collections of the
    same type. This project assumes exactly one CentralCKFTracks collection
    and one MCParticles collection per file (true for every campaign used
    here), so `rec_idx`/`sim_idx` alone are unambiguous and collectionID is
    not read. If a future dataset ever has multiple such collections, this
    assumption breaks silently - add a collectionID check here first.
    """
    return io.read_flat_multi(file_paths, _ASSOCIATIONS_COLUMNS, max_failures=max_failures)


def _best_association(assoc_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Keep only the highest-weight association per group_cols.

    Used both "per truth particle" (group_cols include sim_idx, for
    building matched pairs) and "per reconstructed track" (group_cols
    include rec_idx, for finding fakes) - a truth particle or a
    reconstructed track can appear in more than one association row (e.g. a
    track sharing a few hits with more than one truth particle); only the
    dominant one defines "is this a genuine match".
    """
    if assoc_df.empty:
        return assoc_df
    return (
        assoc_df.sort_values("weight", ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
    )


def build_matched_pairs(
    truth_df: pd.DataFrame,
    reco_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
) -> pd.DataFrame:
    """Build the truth<->reco matched-pairs table.

    Every row of `truth_df` appears exactly once in the output (this is a
    left join from the truth side), with an `is_matched` flag and, for
    matched rows, the corresponding reco.* columns filled in. This is
    intentional: efficiency needs the *denominator* to be every truth
    particle, matched or not.

    Parameters
    ----------
    truth_df, reco_df:
        Output of truth.read_truth_particles (or a filtered subset of it)
        and reco.read_reco_tracks respectively.
    assoc_df:
        Output of read_associations.
    weight_threshold:
        Minimum association weight to accept as a genuine match.

    Returns
    -------
    DataFrame = truth_df's columns, plus: assoc_weight (weight of the best
    association for this particle, NaN if none), is_matched (bool),
    reco_pt, reco_eta, reco_phi, reco_p, reco_charge, reco_pdg, reco_chi2,
    reco_ndf (all NaN where not matched).
    """
    best = _best_association(assoc_df, group_cols=[*_JOIN_KEYS, "sim_idx"])

    merged = truth_df.merge(
        best[[*_JOIN_KEYS, "sim_idx", "rec_idx", "weight"]],
        left_on=[*_JOIN_KEYS, "idx"],
        right_on=[*_JOIN_KEYS, "sim_idx"],
        how="left",
    ).rename(columns={"weight": "assoc_weight"})

    merged["is_matched"] = merged["assoc_weight"].fillna(0.0) >= weight_threshold

    reco_cols = ["pt", "eta", "phi", "p", "charge", "pdg", "chi2", "ndf"]
    reco_renamed = reco_df.rename(columns={c: f"reco_{c}" for c in reco_cols}).rename(
        columns={"idx": "rec_idx"}
    )
    merged = merged.merge(
        reco_renamed[[*_JOIN_KEYS, "rec_idx", *[f"reco_{c}" for c in reco_cols]]],
        on=[*_JOIN_KEYS, "rec_idx"],
        how="left",
    )
    # A row whose best association fell below threshold still has a
    # (spurious) rec_idx/reco_* filled in from the merge above - blank those
    # out so "is_matched == False" always implies "no reco_* values", never
    # a below-threshold match masquerading as a real one.
    reco_value_cols = [f"reco_{c}" for c in reco_cols]
    merged.loc[~merged["is_matched"], reco_value_cols] = pd.NA

    return merged


def find_fake_tracks(
    reco_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    weight_threshold: float = config.MATCH_WEIGHT_THRESHOLD,
) -> pd.DataFrame:
    """Flag reconstructed tracks with no valid truth match ("fakes").

    Every row of `reco_df` appears exactly once in the output (left join
    from the reco side) - the fake-rate denominator is every reconstructed
    track, matched or not.

    Returns
    -------
    DataFrame = reco_df's columns, plus: assoc_weight (best association
    weight for this track, NaN if none), is_fake (bool).
    """
    best = _best_association(assoc_df, group_cols=[*_JOIN_KEYS, "rec_idx"])

    merged = reco_df.merge(
        best[[*_JOIN_KEYS, "rec_idx", "weight"]],
        left_on=[*_JOIN_KEYS, "idx"],
        right_on=[*_JOIN_KEYS, "rec_idx"],
        how="left",
    ).rename(columns={"weight": "assoc_weight"})

    merged["is_fake"] = merged["assoc_weight"].fillna(0.0) < weight_threshold
    return merged
