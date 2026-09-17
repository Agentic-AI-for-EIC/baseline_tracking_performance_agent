"""Collection registries, verified joins, and the Delta R matcher.

Two jobs.

*Read the file's own link bookkeeping* (:func:`read_registry`,
:func:`audit_links`): EDM4eic relations are ``{index, collectionID}`` pairs, so
whether a relation is *usable* is only decidable against
``podio_metadata.events___CollectionTypeInfo`` - the authoritative list of
collections actually written to that file. That is how
``DRICH*IrtCherenkovParticleID.chargedParticle`` was caught pointing at a
collection that does not exist (see :data:`pid.schema.DEAD_LINKS`), and it is
re-checked on every run because a new production can break a different link.

*Do the matching that the reconstruction does not do for us*
(:func:`match_tracks_to_clusters`): the built-in ``*TrackClusterMatches`` give a
usable track<->cluster pair list, but their ``weight`` is identically 0.0 in both
campaigns, so the quality of each pair - and the recovery of pairs the
reconstructor missed for high-energy electrons - must come from geometry: Delta R
between the track direction (or its calorimeter projection) and the cluster
centroid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import awkward as ak

from . import config
from trkperf import io

# ---------------------------------------------------------------------------
# podio collectionID registry
# ---------------------------------------------------------------------------

_REGISTRY_COLUMNS = {
    "name": "events___CollectionTypeInfo.name",
    "collectionID": "events___CollectionTypeInfo.collectionID",
}


def read_registry(path_or_url: str, tree_name: str = "podio_metadata") -> dict[int, str]:
    """Return ``{collectionID: collection name}`` for the ``events`` frame.

    Raises
    ------
    RuntimeError
        If the file carries no ``podio_metadata`` tree: without the registry a
        relation's ``collectionID`` cannot be validated at all, and silently
        trusting it is exactly the failure mode this module exists to prevent.
    """
    try:
        tree = io.open_tree(path_or_url, tree_name=tree_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"pid.links.read_registry: cannot open '{tree_name}' in {path_or_url} "
            f"({exc.__class__.__name__}: {exc}); refusing to guess collection links."
        ) from exc
    arrays = tree.arrays(list(_REGISTRY_COLUMNS.values()), library="ak")
    names = arrays[_REGISTRY_COLUMNS["name"]][0].to_list()
    ids = arrays[_REGISTRY_COLUMNS["collectionID"]][0].to_list()
    return {int(i): str(n) for n, i in zip(names, ids)}


def relation_cids(path_or_url: str, relation: str, *, entry_stop: int | None = 50) -> dict:
    """Inspect one relation branch: cids seen and index range.

    `relation` is the FULL relation branch name, leading underscore included
    (e.g. ``"_EcalEndcapNClusters_hits"``), exactly as it appears in the tree and
    in :data:`AUDITED_RELATIONS`.

    Returns a dict with ``cids``, ``index_min``, ``index_max``, ``n``. Used by
    :func:`audit_links`, :func:`resolve_relation_collection` and
    ``pid schema-check``.
    """
    cols = {
        "index": f"{relation}/{relation}.index",
        "cid": f"{relation}/{relation}.collectionID",
    }
    a = io.read_nested(path_or_url, cols, entry_stop=entry_stop)
    idx = ak.to_numpy(ak.flatten(a["index"]))
    cid = ak.to_numpy(ak.flatten(a["cid"]))
    if idx.size == 0:
        return {"cids": [], "index_min": None, "index_max": None, "n": 0}
    return {
        "cids": sorted(set(int(c) for c in cid.tolist())),
        "index_min": int(idx.min()),
        "index_max": int(idx.max()),
        "n": int(idx.size),
    }


#: Relations whose target must be resolvable for the pipeline to consider a
#: feature family usable. ``expected`` is the collection name the link must
#: resolve to (None = "whatever it resolves to, as long as it is registered").
AUDITED_RELATIONS: dict[str, dict] = {
    "_EcalEndcapNTrackClusterMatches_track": {"expected": "CentralCKFTracks", "family": "calorimeter"},
    "_EcalEndcapNTrackClusterMatches_cluster": {"expected": "EcalEndcapNClusters", "family": "calorimeter"},
    "_EcalEndcapPTrackClusterMatches_track": {"expected": "CentralCKFTracks", "family": "calorimeter"},
    "_EcalEndcapPTrackClusterMatches_cluster": {"expected": "EcalEndcapPClusters", "family": "calorimeter"},
    "_HcalEndcapNTrackClusterMatches_track": {"expected": "CentralCKFTracks", "family": "calorimeter"},
    "_LFHCALTrackClusterMatches_track": {"expected": "CentralCKFTracks", "family": "calorimeter"},
    "_CalorimeterTrackProjections_track": {"expected": "CentralCKFTracks", "family": "projection"},
    "_DRICHGasTracks_track": {"expected": "CentralCKFTracks", "family": "cherenkov"},
    "_DRICHAerogelTracks_track": {"expected": "CentralCKFTracks", "family": "cherenkov"},
    # The ionisation chain is an OPTIONAL family (off by default) and its shape
    # differs by campaign; a failure here is reported, never a gate failure.
    "_CentralCKFTracks_measurements": {"expected": ("CentralTrackerMeasurements",
                                                    "CentralWithoutTOFTrackerMeasurements"),
                                       "family": "ionisation", "optional": True},
    "_ReconstructedChargedRealPIDParticles_tracks": {"expected": "CentralCKFTracks", "family": "baseline"},
    # Known-broken, kept in the audit so a future production that fixes them is
    # noticed instead of silently ignored:
    "_DIRCParticleIDs_particle": {"expected": None, "family": "producer_pid", "expect_broken": True},
    "_CombinedTOFParticleIDs_particle": {"expected": None, "family": "producer_pid", "expect_broken": True},
    "_DRICHGasIrtCherenkovParticleID_chargedParticle": {
        "expected": None, "family": "cherenkov", "expect_broken": True},
}


def resolve_relation_collection(path_or_url: str, relation: str,
                                *, entry_stop: int | None = 50):
    """Collections a ``_<Collection>_<relation>`` list points at.

    Returns ``(names, cids)`` where ``names`` is a single string when exactly one
    collection is referenced and a ``tuple`` when the relation spans several (as
    the track->measurement chain does in 26.07.1, which splits into
    ``CentralTrackerMeasurements`` and ``CentralWithoutTOFTrackerMeasurements``).
    ``(None, [])`` means the relation is empty or dangling.

    This is used instead of assuming a name from the parent's label: gathering
    against the wrong collection silently mixes objects from another event's
    indexing, which shows up only as a shower shape that does not add up.
    """
    registry = read_registry(path_or_url)
    obs = relation_cids(path_or_url, relation, entry_stop=entry_stop)
    if not obs["cids"] or obs["n"] == 0:
        return None, []
    names = [registry.get(c) for c in obs["cids"]]
    if any(n is None for n in names):
        return None, obs["cids"]
    if len(names) == 1:
        return names[0], obs["cids"]
    return tuple(names), obs["cids"]


def audit_links(path_or_url: str, campaign: str | None = None) -> pd.DataFrame:
    """Validate every relation in :data:`AUDITED_RELATIONS` against the file.

    Returns one row per relation with ``usable`` (bool), ``status`` (free text),
    and the measured evidence. A relation counts as usable only if the cids it
    carries are registered in the file *and*, where an expectation exists,
    resolve to the expected collection name.
    """
    registry = read_registry(path_or_url)
    rows = []
    for rel, spec in AUDITED_RELATIONS.items():
        obs = relation_cids(path_or_url, rel)
        names = [registry.get(c) for c in obs["cids"]]
        name = names[0] if len(names) == 1 and names[0] else (
            "+".join(n for n in names if n) if all(names) else None)
        if obs["n"] == 0:
            usable, status = False, "no entries (empty collection)"
        elif not obs["cids"]:
            usable, status = False, "no collectionID recorded"
        elif any(c not in registry for c in obs["cids"]):
            missing = [c for c in obs["cids"] if c not in registry]
            usable, status = False, (
                "dangling: collectionID(s) "
                f"{missing} are not in this file's collection registry"
            )
        elif obs["index_min"] is not None and obs["index_min"] < 0:
            usable, status = False, f"null reference (index_min={obs['index_min']})"
        elif spec["expected"] and (
                (isinstance(spec["expected"], tuple) and name not in spec["expected"]
                 and not set(str(name).split("+")) <= set(spec["expected"]))
                or (isinstance(spec["expected"], str) and name != spec["expected"])):
            usable, status = False, f"resolves to {name!r}, expected {spec['expected']!r}"
        else:
            usable, status = True, f"resolves to {name!r}"
        if spec.get("expect_broken") and not usable:
            status += " [expected broken: recorded in pid.schema.DEAD_LINKS]"
        if spec.get("optional") and not usable:
            status += " [optional family: unavailable in this campaign, feature stays off]"
        rows.append(
            {
                "relation": rel,
                "family": spec["family"],
                "n": obs["n"],
                "cids": ",".join(str(c) for c in obs["cids"]),
                "resolves_to": name,
                "index_min": obs["index_min"],
                "index_max": obs["index_max"],
                "usable": usable,
                "optional": bool(spec.get("optional")),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

_C = 299.792458  # mm/ns


def theta_phi_from_position(x, y, z):
    """Polar/azimuthal angle of a position vector (calorimeter centroid)."""
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    phi = np.arctan2(y, x)
    return theta, phi


def delta_r(theta_a, phi_a, theta_b, phi_b):
    """Detector Delta R = sqrt(dtheta^2 + dphi^2), phi wrapped to [-pi, pi].

    The unweighted dphi is exact only away from the beamline: at very small
    (or very large) theta a given dphi subtends almost no true opening angle,
    so near-beamline winners are chosen more by longitudinal than transverse
    separation. Documented rather than fixed - a space-angle metric would move
    every match and invalidate the trained models for a second-order gain.
    """
    dtheta = np.asarray(theta_a) - np.asarray(theta_b)
    dphi = np.asarray(phi_a) - np.asarray(phi_b)
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return np.sqrt(dtheta**2 + dphi**2)


def wrap_phi(phi):
    return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi


def velocity(beta_gamma):
    """beta from p/mass (both GeV); mass <= 0 or invalid -> NaN."""
    bg = np.asarray(beta_gamma, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = bg / np.sqrt(1.0 + bg**2)
    b[~np.isfinite(bg)] = np.nan
    return b


# ---------------------------------------------------------------------------
# Offset-vector unpacking (inline vectors and relation lists)
# ---------------------------------------------------------------------------

def per_event_flat(array):
    """Split one per-event awkward array into ``(flat_values, counts_per_event)``.

    ``flat_values`` is the event-major concatenation, i.e. the same row order
    :func:`trkperf.io.read_flat` produces. This is the only place the awkward ->
    numpy boundary is crossed for inline vectors and relation lists.
    """
    counts = np.asarray(ak.to_numpy(ak.num(array, axis=1)), dtype=np.int64)
    flat = ak.to_numpy(ak.flatten(array))
    return flat, counts


def _expand_offsets(flat_values, value_counts, parent_event, begin, end):
    """Internal: turn per-object ``[begin, end)`` ranges into gathered values.

    ``begin``/``end`` index into each *event's* own flat vector (that is how
    podio writes inline vectors and relation lists), so the per-event base
    offset must be re-added after flattening - the one step that, if skipped,
    produces values taken from the wrong event.

    Returns ``(parent_row, gathered_values)`` with ``parent_row`` enumerating
    parents in the order given (event-major, matching ``read_flat``).
    """
    parent_event = np.asarray(parent_event, dtype=np.int64)
    begin = np.asarray(begin, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    counts = np.maximum(end - begin, 0)
    total = int(counts.sum())
    n_parents = counts.size
    parent_row = np.repeat(np.arange(n_parents, dtype=np.int64), counts)
    if total == 0:
        return parent_row, np.zeros(0, dtype=flat_values.dtype)

    n_events = int(parent_event.max()) + 1 if n_parents else 0
    if value_counts.size < n_events:  # events with no values at all
        value_counts = np.concatenate(
            [value_counts, np.zeros(n_events - value_counts.size, dtype=np.int64)]
        )
    base = np.concatenate([[0], np.cumsum(value_counts[:-1])])
    if base.size < n_events:
        base = np.concatenate([base, np.zeros(n_events - base.size, dtype=np.int64)])

    grp_start = np.concatenate([[0], np.cumsum(counts[:-1])])
    within = np.arange(total, dtype=np.int64) - np.repeat(grp_start, counts)
    # base[parent_event] is per *parent*; every one of its `counts` values needs it
    position = np.repeat(base[parent_event], counts) + np.repeat(begin, counts) + within
    if position.size:
        lo, hi = int(position.min()), int(position.max())
        if lo < 0 or hi >= int(np.size(flat_values)):
            raise ValueError(
                "pid.links: inline-vector/relation ranges run outside the stored "
                f"array (need [{lo},{hi}], size {int(np.size(flat_values))}). The "
                "begin/end offsets of this collection do not match its flattened "
                "values - reading on would mix events, so the run stops here.")
    return parent_row, np.asarray(flat_values, dtype=flat_values.dtype)[position]


def relation_targets(flat_values, value_counts, parent_event, begin, end):
    """Resolve a ToMany relation into (parent_row, target_row_within_event).

    Use :func:`gather_target` to then read the referenced objects' quantities.
    """
    parent_row, local_index = _expand_offsets(
        flat_values, value_counts, parent_event, begin, end
    )
    return parent_row, np.asarray(local_index, dtype=np.int64)


def gather_target(parent_row, local_index, parent_event_of_row, target, target_columns,
                  *, key=("event", "idx")):
    """Attach the referenced collection's columns to each relation slot.

    Parameters
    ----------
    parent_row, local_index:
        From :func:`relation_targets`.
    parent_event_of_row:
        ``target_events[parent_row]``: the event each slot belongs to. Passing it
        explicitly is what keeps the arithmetic event-correct: ``begin``/``end``
        restart every event while a flattened collection does not.
    target:
        The referenced collection as a flat ``DataFrame`` from
        :func:`trkperf.io.read_flat` (event-major, with ``event`` and ``idx``).
    target_columns:
        Columns to carry out.

    Returns
    -------
    DataFrame with one row per slot: ``event``, ``target_idx``, and
    `target_columns`.

    Raises
    ------
    ValueError
        On any out-of-range index. A mis-decoded relation must stop the run
        rather than contribute wrong shower shapes (AGENTS.md: never let the
        package silently swallow a failure).
    """
    parent_row = np.asarray(parent_row, dtype=np.int64)
    local_index = np.asarray(local_index, dtype=np.int64)
    ev = np.asarray(parent_event_of_row, dtype=np.int64)
    if parent_row.size == 0:
        return pd.DataFrame(columns=["event", "target_idx", *target_columns])

    evs = target[key[0]].to_numpy(dtype=np.int64)
    n_events = int(evs.max()) + 1 if evs.size else 0
    counts = np.bincount(evs, minlength=max(n_events, int(ev.max()) + 1 if ev.size else 1))
    base = np.concatenate([[0], np.cumsum(counts[:-1])])
    if (local_index < 0).any():
        raise ValueError(
            f"pid.links.gather_target: {int((local_index < 0).sum())}/"
            f"{local_index.size} negative relation indices"
        )
    bad = local_index >= counts[ev]
    if bad.any():
        raise ValueError(
            f"pid.links.gather_target: {int(bad.sum())}/{local_index.size} relation "
            "indices exceed their event's collection size - the link is not usable"
        )
    pos = base[ev] + local_index
    out = pd.DataFrame({"event": ev, "target_idx": target[key[1]].to_numpy()[pos]})
    for c in target_columns:
        out[c] = target[c].to_numpy()[pos]
    return out


def inline_vector(flat_values, value_counts, parent_event, begin, end):
    """Gather a per-object inline vector (e.g. ``shapeParameters``).

    Returns ``(parent_row, values)`` - ``values`` are the numbers themselves, so
    reshape per-object with :func:`as_matrix` when the width is fixed.
    """
    return _expand_offsets(flat_values, value_counts, parent_event, begin, end)


def as_matrix(parent_row, values, n_parents, width, *, fill=np.nan):
    """Reshape ``(parent_row, values)`` of fixed `width` into ``(n_parents, width)``.

    Short/empty ranges stay `fill` rather than 0.0, so a missing shape parameter
    can never be read by a tree as a measured zero.
    """
    out = np.full((n_parents, width), fill, dtype=float)
    if parent_row.size == 0:
        return out
    counts = np.bincount(parent_row, minlength=n_parents)
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])
    col = np.arange(parent_row.size, dtype=np.int64) - np.repeat(starts, counts)
    keep = (col >= 0) & (col < width)
    out[parent_row[keep], col[keep]] = np.asarray(values, dtype=float)[keep]
    return out


# ---------------------------------------------------------------------------
# Track <-> cluster matching
# ---------------------------------------------------------------------------

def match_tracks_to_clusters(
    tracks: pd.DataFrame,
    clusters: pd.DataFrame,
    pairs: pd.DataFrame | None = None,
    *,
    delta_r_max: float = config.DELTA_R_MAX,
    detector: str = "",
    track_cols=("theta", "phi"),
    energy_col: str = "energy",
) -> pd.DataFrame:
    """Attach one cluster to each track, preferring the reconstruction's pairs.

    Parameters
    ----------
    tracks:
        Must contain ``event``, ``idx`` and the directions named by
        `track_cols` (theta/phi of the track at its extrapolation point).
    clusters:
        Must contain ``event``, ``idx``, ``energy`` and cluster directions
        ``theta``, ``phi`` (computed from the cluster position).
    pairs:
        Optional ``event``/``track_idx``/``cluster_idx`` table taken from
        ``*TrackClusterMatches``. Entries are used only if they survive the same
        Delta R test as the geometric fallback - their weight is 0.0 in these
        productions, so geometry, not the file, decides whether a pair is real.
    delta_r_max:
        Acceptance window.
    detector:
        Label used in the output columns (e.g. ``"ecal_N"``).

    Returns
    -------
    DataFrame indexed by (event, track idx) with ``<detector>_matched`` (bool),
    ``<detector>_delta_r``, ``<detector>_E``, ``<detector>_cluster_idx`` (the
    winning cluster's index, NaN when unmatched) and the source of the pair
    (``built_in`` / ``geometric`` / ``none``).
    """
    prefix = detector + "_" if detector else ""
    out = tracks[["event", "idx"]].copy()
    out[f"{prefix}matched"] = False
    out[f"{prefix}delta_r"] = np.nan
    out[f"{prefix}E"] = np.nan
    out[f"{prefix}source"] = "none"
    if tracks.empty or clusters.empty:
        return out

    trk = tracks[["event", "idx", track_cols[0], track_cols[1]]].rename(
        columns={track_cols[0]: "theta", track_cols[1]: "phi"})
    clu = clusters[["event", "idx", energy_col, "theta", "phi"]].rename(
        columns={energy_col: "energy"}).copy()
    if clu.empty:
        return out

    base = trk.merge(clu, on="event", how="inner", suffixes=("_t", "_c"))
    base["delta_r"] = delta_r(base["theta_t"], base["phi_t"], base["theta_c"], base["phi_c"])
    cand = base[base["delta_r"] <= delta_r_max].copy()
    cand["is_builtin"] = False
    if pairs is not None and len(pairs):
        key = set(zip(pairs["event"].to_numpy(),
                      pairs["track_idx"].to_numpy(),
                      pairs["cluster_idx"].to_numpy()))
        if key:
            mask = np.array([(int(e), int(a), int(b)) in key
                             for e, a, b in zip(cand["event"], cand["idx_t"], cand["idx_c"])])
            cand.loc[mask, "is_builtin"] = True

    if cand.empty:
        return out

    # Built-in pairs win ties (they encode the reconstruction's intent); then
    # the smallest Delta R.
    cand = cand.sort_values(["event", "idx_t", "is_builtin", "delta_r"],
                            ascending=[True, True, False, True])
    best = cand.drop_duplicates(subset=["event", "idx_t"], keep="first").copy()

    merged = out.merge(
        best[["event", "idx_t", "idx_c", "delta_r", "energy", "is_builtin"]].rename(
            columns={"idx_t": "idx"}),
        on=["event", "idx"], how="left",
    )
    merged[f"{prefix}matched"] = merged["delta_r"].notna()
    merged[f"{prefix}delta_r"] = merged["delta_r"]
    merged[f"{prefix}E"] = merged["energy"]
    merged[f"{prefix}cluster_idx"] = merged["idx_c"]
    merged[f"{prefix}source"] = np.select(
        [merged["delta_r"].isna(), merged["is_builtin"].fillna(False).astype(bool)],
        ["none", "built_in"], default="geometric")
    return merged.drop(columns=["delta_r", "energy", "is_builtin", "idx_c"])
