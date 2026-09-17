"""Build the per-track PID feature table.

One row per **reconstructed track candidate** (every ``CentralCKFTracks`` entry
of every file, matched to truth or not), carrying:

* the track's own kinematics (via :func:`trkperf.reco.read_reco_tracks`, i.e.
  momentum from ``CentralCKFTrackParameters`` - ``CentralCKFTracks.momentum`` is
  all zeros in these files),
* the truth label from :func:`trkperf.matching.build_matched_tracks`, so PID
  uses the identical weight>=0.5 matching convention as the efficiency metrics,
* calorimeter response (E/p, transverse shower shape, hadronic leakage) attached
  through the joins verified in :mod:`pid.links`,
* dRICH availability and per-track radiator path length,
* event-context occupancy (ablatable - this is the mechanism by which beam
  background is expected to degrade PID),
* ``has_*`` flags for every subsystem, with absent values left as NaN.

Why this module exists at all: the PID problem is "for each track find its
cluster, then that cluster's hits", i.e. a *join across collections*, which
generic flattening cannot express. Every individual array still comes from
:func:`trkperf.io.read_flat` / :func:`trkperf.io.read_nested` and every truth
decision from :mod:`trkperf.matching`, so nothing here duplicates file access or
matching (AGENTS.md).

Cache
-----
``cache_dir`` pickles the finished per-file table, keyed on file + campaign +
legs + ionisation switch + :data:`SCHEMA_VERSION`. A relaunch after a container
teardown therefore replays finished files from local disk, mirroring (one level
up, where the tables are far smaller) why :mod:`trkperf.io` caches raw tables.
"""

from __future__ import annotations

import hashlib
import os
import sys

import awkward as ak
import numpy as np
import pandas as pd

from trkperf import io, matching, reco, truth

from . import config, links, schema

#: Bump whenever the column set or a column's definition changes. Cached tables
#: written under an older version are keyed differently, so a stale feature
#: table can never be silently mixed into a new run.
SCHEMA_VERSION = "pid-v1"

#: Detector -> logical branch prefixes (see pid.schema.BRANCHES).
ECAL_PREFIX = {"EcalEndcapN": "ecal_N", "EcalEndcapP": "ecal_P"}
HCAL_PREFIX = {"HcalEndcapN": "hcal_N", "LFHCAL": "lfhcal"}
MATCH_KEYS = {
    "EcalEndcapN": ("match_ecal_N_track", "match_ecal_N_cluster"),
    "EcalEndcapP": ("match_ecal_P_track", "match_ecal_P_cluster"),
    "HcalEndcapN": ("match_hcal_N_track", "match_hcal_N_cluster"),
    "LFHCAL": ("match_lfhcal_track", "match_lfhcal_cluster"),
}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

#: Branches found absent in the file being read (printed once per key).
_MISSING_BRANCH_NOTES: set[str] = set()


def _missing(key: str, branch: str, why: str) -> None:
    marker = f"{key}|{why}"
    if marker not in _MISSING_BRANCH_NOTES:
        _MISSING_BRANCH_NOTES.add(marker)
        print(f"[features] NOTE: {key} unavailable ({why}); the dependent columns "
              f"stay NaN + has_* = 0 rather than being fabricated from "
              f"{branch!r}", file=sys.stderr)


def _available(tree, branch: str) -> bool:
    """Is this branch really in this file?

    A campaign entry in the schema is not proof: 26.07.1 registers
    ``CentralTrackerMeasurements`` in its collection table but writes no branches
    for it (the collection was renamed). ``tree.keys()`` cannot answer this,
    because uproot lists top-level branches (``CentralCKFTracks``) and not the
    dotted sub-branches (``CentralCKFTracks.measurements_begin``) that EDM4eic
    reads address - so probe with a one-entry read, which is metadata-cheap.
    """
    try:
        tree.arrays([branch], library="ak", entry_start=0, entry_stop=1)
        return True
    except Exception:  # noqa: BLE001 - absent/unreadable branch in this production
        return False


def _flat(tree, campaign, keys):
    """``read_flat`` over a set of logical names belonging to ONE collection.

    Returns an empty DataFrame (with a printed note) when any member is absent
    from this particular file, so an optional feature family degrades instead of
    raising part-way through a multi-hour grid run.
    """
    cols = {}
    for k in keys:
        b = schema.BRANCHES[k].for_campaign(campaign)
        if b is None:
            return pd.DataFrame()
        if not _available(tree, b):
            _missing(k, b, "branch absent from this file")
            return pd.DataFrame()
        cols[k] = b
    try:
        out = io.read_flat(tree, cols)
    except ValueError as exc:  # ragged members of different lengths => not one collection
        _missing(keys[0], str(list(cols.values())), f"unreadable as one collection ({exc})")
        return pd.DataFrame()
    if out.empty:
        return out
    # read_flat assumes every member has the same per-event length; when a
    # production writes one collection empty and its sibling full, that assumption
    # breaks silently, so check and degrade instead.
    counts = {c: len(out) for c in cols}
    if len(set(counts.values())) > 1:  # pragma: no cover - defensive
        _missing(keys[0], str(list(cols.values())), "member branches disagree in length")
        return pd.DataFrame()
    return out


def _nested(path, campaign, keys, **kw):
    cols = {}
    for k in keys:
        b = schema.BRANCHES[k].for_campaign(campaign)
        if b is None:
            return {}
        cols[k] = b
    try:
        return io.read_nested(path, cols, **kw)
    except Exception as exc:  # noqa: BLE001 - absent collection in this production
        _missing(keys[0], str(list(cols.values())), f"{type(exc).__name__}")
        return {}


def _branch(path, tree, campaign, logical):
    """Return ``(flat_values, counts_per_event)`` for one nested branch."""
    raw = _nested(path, campaign, [logical])
    if not raw:
        return None, None
    return links.per_event_flat(raw[logical])


# ---------------------------------------------------------------------------
# Calorimeter clusters
# ---------------------------------------------------------------------------

def _weighted_moments(parent, e, x, y, n_parents):
    """Energy-weighted centroid and widths of each cluster, vectorised.

    Weighted by hit energy (not the unweighted spread of hit positions): a MIP
    crossing the outer ECAL rows would otherwise inflate the width of a compact
    electromagnetic core.
    """
    sum_e = np.bincount(parent, weights=e, minlength=n_parents)
    nz = sum_e > 0
    cx = np.zeros(n_parents)
    cy = np.zeros(n_parents)
    cx[nz] = np.bincount(parent, weights=e * x, minlength=n_parents)[nz] / sum_e[nz]
    cy[nz] = np.bincount(parent, weights=e * y, minlength=n_parents)[nz] / sum_e[nz]
    dx = x - cx[parent]
    dy = y - cy[parent]
    rms_x = np.full(n_parents, np.nan)
    rms_y = np.full(n_parents, np.nan)
    rms_x[nz] = np.sqrt(np.maximum(
        np.bincount(parent, weights=e * dx * dx, minlength=n_parents)[nz] / sum_e[nz], 0.0))
    rms_y[nz] = np.sqrt(np.maximum(
        np.bincount(parent, weights=e * dy * dy, minlength=n_parents)[nz] / sum_e[nz], 0.0))
    return sum_e, cx, cy, rms_x, rms_y


def _read_hits_of_cluster_collection(path, tree, campaign, detector):
    """Read the cluster's hit collection, named by the file rather than assumed.

    The target of ``_<Clusters>_hits`` is resolved through the file's own
    collectionID registry, and its member branches are then read directly. This
    matters: ``EcalEndcapNClusters._hits`` points at a collection whose name
    cannot be guessed from the parent, and gathering against the wrong collection
    produces shapes that look plausible but are not the cluster's own hits (the
    symptom is exactly the ``sum(hit E) != cluster E`` warning below).
    """
    p = ECAL_PREFIX[detector]
    rel_key = schema.BRANCHES[f"{p}_hit_idx"].for_campaign(campaign)
    if rel_key is None:
        return pd.DataFrame()
    # schema stores "_X_hits.index"; the relation branch name is the part before it
    relation = rel_key.split("/")[0]
    relation = relation[:-len(".index")] if relation.endswith(".index") else relation
    name, _cid = links.resolve_relation_collection(path, relation, entry_stop=None)
    if name is None:
        return pd.DataFrame()
    wanted = {"energy": "hit_e", "position.x": "hit_x", "position.y": "hit_y",
              "position.z": "hit_z", "time": "hit_t"}
    cols = {}
    for suffix, out in wanted.items():
        branch = f"{name}.{suffix}"
        try:
            if branch not in tree.keys():
                continue
        except Exception:  # noqa: BLE001 - key lookup on some trees raises
            continue
        cols[out] = branch
    if not cols:
        return pd.DataFrame()
    return io.read_flat(tree, cols)


def build_cluster_table(path, tree, campaign, detector):
    """Per-cluster table for one calorimeter, with shapes computed from hits.

    The producer's own ``shapeParameters`` are appended unchanged as
    ``shape_0..shape_6`` by :func:`attach_shape_parameters` (they are undocumented
    in the installed EDM4eic headers, and only two of the seven correlate with
    anything reproducible - r = +0.93/+0.95 against the computed transverse
    width), while the headline shapes are recomputed here so their definitions
    are known and identical across campaigns.

    No longitudinal profile is computed for the endcap ECALs: they are
    single-plane in these productions (constant ``position.z``, ``layer == -1``),
    see :data:`pid.schema.DEAD_LINKS`.
    """
    if detector in ECAL_PREFIX:
        p = ECAL_PREFIX[detector]
        clu = _flat(tree, campaign, [f"{p}_energy", f"{p}_nhits", f"{p}_time",
                                     f"{p}_pos_x", f"{p}_pos_y", f"{p}_pos_z",
                                     f"{p}_hits_begin", f"{p}_hits_end"])
        hit_keys = [f"{p}_hit_energy", f"{p}_hit_x", f"{p}_hit_y", f"{p}_hit_z"]
        rel_key = f"{p}_hit_idx"
    else:
        p = HCAL_PREFIX[detector]
        clu = _flat(tree, campaign, [f"{p}_energy", f"{p}_nhits", f"{p}_time",
                                     f"{p}_pos_x", f"{p}_pos_y", f"{p}_pos_z"])
        hit_keys, rel_key = [], None
    if clu.empty:
        return pd.DataFrame()
    # read_flat names columns after the *logical* keys given to it, so rename
    # through those, not through the raw branch suffixes.
    ren = {f"{p}_energy": "E", f"{p}_nhits": "n", f"{p}_time": "t",
           f"{p}_pos_x": "x", f"{p}_pos_y": "y", f"{p}_pos_z": "z",
           f"{p}_hits_begin": "hb", f"{p}_hits_end": "he"}
    # HCAL clusters have no time branch in some productions; a missing column is
    # then simply absent rather than a silently all-NaN feature.
    ren = {k: v for k, v in ren.items() if k in clu.columns or k in (f"{p}_time",)}
    clu = clu.rename(columns={k: v for k, v in ren.items() if k in clu.columns})
    clu["theta"], clu["phi"] = links.theta_phi_from_position(
        clu["x"].to_numpy(), clu["y"].to_numpy(), clu["z"].to_numpy())
    if "t" not in clu.columns:
        clu["t"] = np.nan
        clu = clu.drop(columns=[c for c in (f"{p}_time",) if c in clu.columns])
    clu = clu.reset_index(drop=True)
    if "hb" not in clu.columns or not hit_keys:
        return clu

    hits = _read_hits_of_cluster_collection(path, tree, campaign, detector)
    flat_idx, counts = _branch(path, tree, campaign, rel_key)
    if hits.empty or flat_idx is None:
        return clu.assign(has_shape=0)

    pev = clu["event"].to_numpy(dtype=np.int64)
    prow, lidx = links.relation_targets(flat_idx, counts, pev,
                                        clu["hb"].to_numpy(), clu["he"].to_numpy())
    tab = links.gather_target(prow, lidx, pev[prow], hits,
                              [c for c in ("hit_e", "hit_x", "hit_y", "hit_z", "hit_t")
                               if c in hits.columns])
    if tab.empty:
        return clu.assign(has_shape=0)
    e_col, x_col, y_col = "hit_e", "hit_x", "hit_y"
    n_par = len(clu)
    sum_e, cx, cy, rms_x, rms_y = _weighted_moments(
        prow, tab[e_col].to_numpy(dtype=float), tab[x_col].to_numpy(dtype=float),
        tab[y_col].to_numpy(dtype=float), n_par)

    s = pd.Series(tab[e_col].to_numpy(dtype=float)).groupby(prow)
    max_e = np.full(n_par, np.nan)
    mx = s.max()
    max_e[mx.index.to_numpy()] = mx.to_numpy()
    n_hit = np.full(n_par, 0, dtype=np.int64)
    sz = s.size()
    n_hit[sz.index.to_numpy()] = sz.to_numpy()

    clu["clu_sum_e"] = sum_e
    clu["clu_cx"], clu["clu_cy"] = cx, cy
    clu["rms_x"], clu["rms_y"] = rms_x, rms_y
    clu["rms_r"] = np.sqrt(np.maximum((rms_x**2 + rms_y**2) / 2.0, 0.0))
    clu["n_hit_calc"] = n_hit
    with np.errstate(invalid="ignore", divide="ignore"):
        clu["core_frac"] = np.where(sum_e > 0, max_e / sum_e, np.nan)
    clu["has_shape"] = (n_hit >= config.MIN_HITS_FOR_SHAPE).astype(int)
    clu.loc[clu["has_shape"] == 0, ["rms_x", "rms_y", "rms_r", "core_frac"]] = np.nan

    # Consistency DIAGNOSTIC (not a gate): the energy-weighted moments above are
    # computed over the hits the cluster lists, which is self-consistent whether
    # or not it reproduces the stored total. It does not always: clusters built by
    # merging sub-clusters (EcalEndcapN in 26.02.0) store a total larger than
    # their own hit list, while the unmerged EcalEndcapP matches to 1e-8. The
    # widths are still the cluster's shape; only ``sum_e``/``E`` reveals the
    # composition, so it is recorded rather than discarded or fatal.
    with np.errstate(invalid="ignore", divide="ignore"):
        clu["e_hit_over_e_clu"] = np.where(clu["E"] > 0, sum_e / clu["E"].to_numpy(), np.nan)
    bad = np.abs(sum_e - clu["E"].to_numpy()) > 1e-3 * np.maximum(clu["E"].to_numpy(), 1.0)
    if len(clu) and bad.mean() > 0.05:
        print(f"[features] NOTE: {detector}: sum(hit E) != cluster E for "
              f"{100.0 * bad.mean():.1f}% of clusters (max deficit "
              f"{float(np.nanmax(np.abs(sum_e - clu['E'].to_numpy()))):.3g} GeV) - "
              f"merged sub-clusters; widths computed from the listed hits",
              file=sys.stderr)
    return clu.drop(columns=["hb", "he"])


def attach_shape_parameters(path, tree, campaign, detector, clu):
    """Append the producer's unlabelled ``shape_0..N`` columns to `clu`."""
    if clu.empty or detector not in ECAL_PREFIX:
        return clu
    p = ECAL_PREFIX[detector]
    vals, counts = _branch(path, tree, campaign, f"{p}_shape_values")
    if vals is None:
        return clu
    offs = _flat(tree, campaign, [f"{p}_shape_begin", f"{p}_shape_end"])
    if offs.empty or len(offs) != len(clu):
        return clu
    pev = clu["event"].to_numpy(dtype=np.int64)
    prow, gathered = links.inline_vector(
        vals, counts, pev, offs[f"{p}_shape_begin"].to_numpy(), offs[f"{p}_shape_end"].to_numpy())
    M = links.as_matrix(prow, gathered, len(clu), schema.N_ECAL_SHAPE_PARAMS)
    out = clu.reset_index(drop=True).copy()
    for i in range(schema.N_ECAL_SHAPE_PARAMS):
        out[f"shape_{i}"] = M[:, i]
    return out


def cluster_match_pairs(path, tree, campaign, detector):
    """The reconstruction's own track<->cluster pairs for one calorimeter."""
    tk, ck = MATCH_KEYS[detector]
    t = _flat(tree, campaign, [tk])
    c = _flat(tree, campaign, [ck])
    if t.empty or c.empty:
        return pd.DataFrame(columns=["event", "track_idx", "cluster_idx"])
    t = t.rename(columns={tk: "track_idx"})[["event", "idx", "track_idx"]]
    c = c.rename(columns={ck: "cluster_idx"})[["event", "idx", "cluster_idx"]]
    m = t.merge(c, on=["event", "idx"], how="inner")
    ok = (m["track_idx"] >= 0) & (m["cluster_idx"] >= 0)
    return m.loc[ok, ["event", "track_idx", "cluster_idx"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cherenkov
# ---------------------------------------------------------------------------

def drich_track_link(path, tree, campaign):
    """Per-track dRICH availability + radiator path length (link verified)."""
    frames = []
    for tag, trk_key, len_key in (("gas", "drich_gas_track", "drich_gas_length"),
                                  ("aerogel", "drich_aerogel_track", "drich_aerogel_length")):
        tr = _flat(tree, campaign, [trk_key])
        seg = _flat(tree, campaign, [len_key])
        if tr.empty or seg.empty:
            continue
        # `_DRICH*_track.index` is keyed by the TrackSegment's own `idx`, which is
        # exactly the `idx` of the DRICH*Tracks row that carries `length`.
        tr = tr.rename(columns={trk_key: "track_idx"})[["event", "idx", "track_idx"]]
        seg = seg.rename(columns={len_key: "length"})[["event", "idx", "length"]]
        m = tr.merge(seg, on=["event", "idx"], how="left")
        m[f"drich_{tag}_pathlength"] = m["length"]
        # The flag must follow the measurement, not the link row: a track with
        # a length entry but no radiator segment has no pathlength.
        m[f"has_drich_{tag}"] = m["length"].notna().astype(int)
        ok = m["track_idx"].to_numpy() >= 0
        frames.append(m.loc[ok, ["event", "track_idx", f"drich_{tag}_pathlength",
                                 f"has_drich_{tag}"]])
    if not frames:
        return pd.DataFrame(columns=["event", "track_idx"])
    out = frames[0]
    for f in frames[1:]:
        # Each radiator frame must already be 1:1 per track; without this an
        # unexpected 1:N would silently multiply track rows on the outer join.
        for name, frame in (("gas", out), ("aerogel", f)):
            dup = int(frame.duplicated(subset=["event", "track_idx"]).sum())
            if dup:
                raise ValueError(
                    f"pid.features.drich_track_link: {dup} duplicated "
                    f"(event, track_idx) rows in the {name} frame - refusing to fan out")
        out = out.merge(f, on=["event", "track_idx"], how="outer")
    return out


def irt_event_summary(path, campaign, entry_stop=None):
    """Per-event dRICH photon summary (NOT per track - see DEAD_LINKS).

    The IRT objects' ``chargedParticle`` relation points at a collection absent
    from the file, so these columns describe the event. They are suffixed
    ``_evt`` and listed in :func:`pid.dataset.model_columns` as event-level, so
    they cannot be mistaken for per-track observables when a plot is labelled.
    """
    raw = _nested(path, campaign, ["irt_gas_npe", "irt_aerogel_npe"], entry_stop=entry_stop)
    if not raw:
        return pd.DataFrame()
    key0 = "irt_gas_npe" if "irt_gas_npe" in raw else "irt_aerogel_npe"
    n_ev = int(ak.num(raw[key0], axis=0))
    out = {"event": np.arange(n_ev, dtype=np.int64)}
    for tag in ("gas", "aerogel"):
        k = f"irt_{tag}_npe"
        if k not in raw:
            continue
        v = raw[k]
        cnt = np.asarray(ak.to_numpy(ak.num(v, axis=1)), dtype=float)
        tot = np.asarray(ak.to_numpy(ak.sum(v, axis=1)), dtype=float)
        mx = np.asarray(ak.to_numpy(ak.fill_none(ak.max(v, axis=1), np.nan)), dtype=float)
        # Absent photons are missing data, not zero photons: 0.0 would read as
        # a measured "nothing seen here". The has_* flag carries presence.
        out[f"irt_{tag}_nobj_evt"] = cnt
        out[f"irt_{tag}_npe_tot_evt"] = np.where(cnt > 0, tot, np.nan)
        out[f"irt_{tag}_npe_max_evt"] = np.where(cnt > 0, mx, np.nan)
        out[f"has_irt_{tag}_evt"] = (cnt > 0).astype(int)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Track side
# ---------------------------------------------------------------------------

def _tree_base(path, campaign, max_failures=0):
    """One row per reconstructed track with its truth label (via trkperf)."""
    truth_df = truth.read_truth_particles([path], max_failures=max_failures, primary_only=True)
    reco_df = reco.read_reco_tracks([path], max_failures=max_failures)
    assoc = matching.read_associations([path], max_failures=max_failures)
    tracks = matching.build_matched_tracks(truth_df, reco_df, assoc)
    if tracks.empty:
        return tracks, {}
    extra = _flat(io.open_tree(path), campaign, ["track_time", "track_time_error"])
    if not extra.empty:
        tracks = tracks.merge(extra[["event", "idx", "track_time", "track_time_error"]],
                              on=["event", "idx"], how="left")
    return tracks, {"extra": extra}


def projection_direction(path, tree, campaign):
    """Track direction where its extrapolation crosses the calorimeter faces.

    ``CalorimeterTrackProjections`` links 1:1 to ``CentralCKFTracks`` (verified
    by :func:`pid.links.audit_links`) and holds one point per crossed surface, so
    the *last* point gives the direction the cluster should be matched against -
    far better than the perigee direction, which ignores the bending field. The
    first point and the total path length are kept as well.
    """
    empty = pd.DataFrame(columns=["event", "track_idx"])
    trk = _flat(tree, campaign, ["proj_track"])
    offs = _flat(tree, campaign, ["proj_points_begin", "proj_points_end"])
    if trk.empty or offs.empty:
        return empty
    trk = trk.rename(columns={"proj_track": "track_idx"})[["event", "idx", "track_idx"]]
    proj = offs.merge(trk, on=["event", "idx"], how="inner").reset_index(drop=True)
    if proj.empty or proj["track_idx"].isna().all():
        return empty
    proj["track_idx"] = proj["track_idx"].astype(np.int64)
    proj["obj_row"] = np.arange(len(proj), dtype=np.int64)

    theta, tcounts = _branch(path, tree, campaign, "proj_point_theta")
    phi, _ = _branch(path, tree, campaign, "proj_point_phi")
    plen, _ = _branch(path, tree, campaign, "proj_point_pathlength")
    if theta is None:
        return empty
    pev = proj["event"].to_numpy(dtype=np.int64)
    b = proj["proj_points_begin"].to_numpy()
    e = proj["proj_points_end"].to_numpy()
    prow, th = links.inline_vector(theta, tcounts, pev, b, e)
    _, ph = links.inline_vector(phi, tcounts, pev, b, e)
    if plen is None:
        # No pathlength branch in this campaign: NaN, never another column's
        # values under a false name (a tree would happily split on theta).
        plen = np.full_like(np.asarray(theta, dtype=float), np.nan)
    _, pl = links.inline_vector(plen, tcounts, pev, b, e)
    if prow.size == 0:
        return empty

    slots = pd.DataFrame({"obj_row": prow, "theta": th, "phi": ph, "pathlength": pl})
    first = slots.groupby("obj_row", sort=True).first()
    last = slots.groupby("obj_row", sort=True).last()[["theta", "phi"]].rename(
        columns={"theta": "theta_out", "phi": "phi_out"})
    out = first.join(last).reset_index().merge(
        proj[["obj_row", "event", "track_idx"]], on="obj_row", how="inner")
    return out[["event", "track_idx", "theta", "phi", "theta_out", "phi_out", "pathlength"]].rename(
        columns={"theta": "theta_in", "phi": "phi_in"})


def ionisation_proxy(path, tree, campaign):
    """Reco-side ionisation summary per track: mean hit ``edep`` per subsystem.

    A *proxy*, not dE/dx: ``TrackerHitData`` carries no ``pathLength`` in this
    datamodel, so no thickness normalisation is possible - see
    :data:`pid.schema.DEAD_LINKS`.

    Chain, hop by hop (each hop's ``collectionID`` is resolved through the file's
    own registry, so a production that rewires it is reported rather than
    misread)::

        CentralCKFTracks.measurements_begin/end  ->  _CentralCKFTracks_measurements
            ->  CentralTrackerMeasurements.hits_begin/end  ->  _..._hits
            ->  *_RecHits.edep

    Note the two-level offset arithmetic: a track's measurement list and a
    measurement's hit list are *both* flattened per event, which is why they are
    expanded with :func:`pid.links.inline_vector` rather than read as columns of
    one table (they have different per-event lengths by construction).

    Off by default (``--enable-ionisation``): coverage is partial (only the Si
    barrel and MPGD barrel RecHits are reachable in 26.02.0, and nothing is in
    26.07.1), so it is a cross-check rather than a headline feature.
    """
    empty = pd.DataFrame(columns=["event", "track_idx"])
    trk = _flat(tree, campaign, ["track_meas_begin", "track_meas_end"])
    meas = _flat(tree, campaign, ["meas_hits_begin", "meas_hits_end"])
    hit_idx, hit_counts = _branch(path, tree, campaign, "meas_hit_idx")
    hit_cids, _ = _branch(path, tree, campaign, "meas_hit_cid")
    trk_idx, trk_counts = _branch(path, tree, campaign, "track_meas_idx")
    if trk.empty or meas.empty or hit_idx is None or hit_cids is None or trk_idx is None:
        return empty
    trk = trk.rename(columns={"track_meas_begin": "mb", "track_meas_end": "me",
                              "idx": "track_idx"}).reset_index(drop=True)

    # hop 1: track row -> measurement (event-local index)
    track_row, meas_local = links.inline_vector(trk_idx, trk_counts,
                                                trk["event"].to_numpy(dtype=np.int64),
                                                trk["mb"].to_numpy(), trk["me"].to_numpy())
    if track_row.size == 0:
        return empty
    meas = meas.reset_index(drop=True)
    meas_event = meas["event"].to_numpy(dtype=np.int64)
    base = np.concatenate([[0], np.cumsum(np.bincount(
        meas_event, minlength=int(meas_event.max()) + 1)[:-1])])
    ev_of_slot = trk["event"].to_numpy(dtype=np.int64)[track_row]
    meas_pos = base[ev_of_slot] + np.asarray(meas_local, dtype=np.int64)
    bad = np.asarray(meas_local, dtype=np.int64) >= np.bincount(
        meas_event, minlength=base.size)[ev_of_slot]
    if bad.any():
        print(f"[features] NOTE: ionisation - {int(bad.sum())}/{len(bad)} track "
              "measurement references fall outside their event's measurement "
              "collection; those slots are dropped", file=sys.stderr)
        keep = ~bad
        track_row, meas_pos, ev_of_slot = track_row[keep], meas_pos[keep], ev_of_slot[keep]
        if track_row.size == 0:
            return empty

    # hop 2: measurement -> hit, using the measurement's own offsets
    mrow, hit_local = links.inline_vector(hit_idx, hit_counts, meas_event[meas_pos],
                                          meas["meas_hits_begin"].to_numpy()[meas_pos],
                                          meas["meas_hits_end"].to_numpy()[meas_pos])
    _, cid_vals = links.inline_vector(hit_cids, hit_counts, meas_event[meas_pos],
                                      meas["meas_hits_begin"].to_numpy()[meas_pos],
                                      meas["meas_hits_end"].to_numpy()[meas_pos])
    if mrow.size == 0:
        return empty
    slots = pd.DataFrame({"slot": mrow, "hit_local": np.asarray(hit_local, dtype=np.int64),
                          "cid": np.asarray(cid_vals, dtype=np.int64)})
    slots["track_row"] = track_row[slots["slot"].to_numpy()]
    slots["event"] = ev_of_slot[slots["slot"].to_numpy()]

    registry = {v: int(k) for k, v in links.read_registry(path).items()}
    result = trk[["event", "track_idx"]].copy()
    for coll, key in (("SiBarrelTrackerRecHits", "si_hit_edep"),
                      ("MPGDBarrelRecHits", "mpgd_hit_edep")):
        col = "si" if coll.startswith("Si") else "mpgd"
        mean_name, n_name = f"edep_{col}_mean", f"edep_{col}_n"
        vals = np.full(len(result), np.nan)
        nn = np.zeros(len(result), dtype=np.int64)
        want = registry.get(coll)
        hits = _flat(tree, campaign, [key])
        if want is not None and not hits.empty:
            sel = slots["cid"].to_numpy() == want
            if sel.any():
                sub = slots.loc[sel]
                gathered = links.gather_target(sub["slot"].to_numpy(),
                                               sub["hit_local"].to_numpy(),
                                               sub["event"].to_numpy(), hits, [key])
                gathered["track_row"] = sub["track_row"].to_numpy()
                grp = gathered.groupby("track_row")[key]
                mean, cnt = grp.mean(), grp.size()
                rows = mean.index.to_numpy()
                vals[rows] = mean.to_numpy()
                nn[rows] = cnt.to_numpy()
        result[mean_name] = vals
        result[n_name] = nn
    num = [c for c in result.columns if c.endswith("_mean") and c.startswith("edep_")]
    result["has_ionisation"] = result[num].notna().any(axis=1).astype(int) if num else 0
    return result


def build_features_one_file(path, campaign, *, legs=("backward", "forward"),
                            max_failures=0, enable_ionisation=False):
    """Feature table for one ROOT file (see :func:`build_features`)."""
    base, _ = _tree_base(path, campaign, max_failures=max_failures)
    if base.empty:
        return pd.DataFrame()
    tree = io.open_tree(path)

    feats = base.rename(columns={"idx": "track_idx"}).copy()
    if "pz" in feats.columns:
        with np.errstate(invalid="ignore", divide="ignore"):
            feats["theta"] = np.arccos(np.clip(feats["pz"] / feats["p"], -1.0, 1.0))
    else:
        feats["theta"] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["chi2_per_ndf"] = np.where(feats["ndf"] > 0, feats["chi2"] / feats["ndf"], np.nan)
    for c in ("track_time", "track_time_error"):
        if c not in feats.columns:
            feats[c] = np.nan
    # Class = species family on |PDG| (config.CLASS_OF_ABS_PDG); NaN for tracks
    # with no truth match - those are kept, they are the combinatorial background
    # a working PID must reject.
    feats["truth_class"] = feats["truth_pdg"].map(
        lambda p: config.CLASS_OF_ABS_PDG.get(abs(int(p))) if pd.notna(p) else None)

    proj = projection_direction(path, tree, campaign)
    if not proj.empty:
        p = proj.drop_duplicates(subset=["event", "track_idx"])
        feats = feats.merge(p[["event", "track_idx", "theta_out", "phi_out", "pathlength"]]
                            .rename(columns={"pathlength": "proj_pathlength"}),
                            on=["event", "track_idx"], how="left")
        feats["match_theta"] = feats["theta_out"].fillna(feats["theta"])
        feats["match_phi"] = feats["phi_out"].fillna(feats["phi"])
    else:
        feats["match_theta"] = feats["theta"]
        feats["match_phi"] = feats["phi"]

    detectors = []
    for leg in legs:
        for role in ("ecal", "hcal"):
            d = config.LEGS[leg][role]
            if d and d not in detectors:
                detectors.append(d)

    calo = {}
    for det in detectors:
        clu = build_cluster_table(path, tree, campaign, det)
        if clu.empty:
            continue
        clu = attach_shape_parameters(path, tree, campaign, det, clu)
        # `idx` from read_flat IS the in-event cluster index that the match
        # relations refer to, so it - not a positional row number - is the key.
        calo[det] = clu

    for leg in legs:
        for role, det in (("ecal", config.LEGS[leg]["ecal"]), ("hcal", config.LEGS[leg]["hcal"])):
            short = f"{role}_{leg}"
            if det is None or det not in calo:
                continue
            clu = calo[det]
            pairs = cluster_match_pairs(path, tree, campaign, det)
            trk_for_match = feats[["event", "track_idx", "match_theta", "match_phi"]].rename(
                columns={"track_idx": "idx"})
            clu_for_match = clu[["event", "idx", "E", "theta", "phi"]]
            m = links.match_tracks_to_clusters(trk_for_match, clu_for_match, pairs,
                                               detector=short,
                                               track_cols=("match_theta", "match_phi"),
                                               energy_col="E")
            cols = [c for c in m.columns if c.startswith(f"{short}_")
                      and not c.endswith("cluster_idx")]
            if not cols:
                continue
            feats = feats.merge(m[["event", "idx", *cols]].rename(columns={"idx": "track_idx"}),
                                on=["event", "track_idx"], how="left")

            # "E" is excluded: the matcher already produced <prefix>_E, and
            # merging it again would silently create _x/_y duplicate columns.
            shape_cols = [c for c in clu.columns if c in
                          ("rms_r", "rms_x", "rms_y", "core_frac", "t", "n",
                            "e_hit_over_e_clu", "has_shape")
                           or c.startswith("shape_")]
            if not shape_cols:
                continue
            # Shapes follow the WINNING cluster (the one <prefix>_E was read
            # from), never the file's first built-in pair: E/p and the shower
            # shape must describe the same cluster.
            win = m.loc[m[f"{short}_matched"],
                        ["event", "idx", f"{short}_cluster_idx"]].rename(
                            columns={"idx": "track_idx",
                                     f"{short}_cluster_idx": "cluster_idx"}).copy()
            win = win.dropna(subset=["cluster_idx"])
            if win.empty:
                continue
            win["cluster_idx"] = win["cluster_idx"].astype(int)
            keyed = win.merge(clu[["event", "idx", *shape_cols]].rename(
                columns={"idx": "cluster_idx"}),
                on=["event", "cluster_idx"], how="left")
            ren = {c: f"{short}_{c}" for c in shape_cols}
            feats = feats.merge(keyed[["event", "track_idx", *shape_cols]].rename(columns=ren),
                                on=["event", "track_idx"], how="left")

    for leg in legs:
        e_col = f"ecal_{leg}_matched"
        if e_col in feats.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                ep = np.where(feats["p"] > 0, feats[f"ecal_{leg}_E"] / feats["p"], np.nan)
            feats[f"e_over_p_{leg}"] = np.where(ep <= config.E_OVER_P_MAX, ep, np.nan)
            feats[f"log_e_over_p_{leg}"] = np.log10(np.clip(feats[f"e_over_p_{leg}"], 1e-3, None))
            feats[f"has_ecal_{leg}"] = feats[e_col].fillna(False).astype(int)
        if f"hcal_{leg}_matched" in feats.columns and f"ecal_{leg}_E" in feats.columns:
            feats[f"e_hcal_{leg}"] = feats[f"hcal_{leg}_E"]
            with np.errstate(invalid="ignore", divide="ignore"):
                feats[f"leakage_{leg}"] = np.where(
                    feats[f"ecal_{leg}_E"] > 0, feats[f"hcal_{leg}_E"] / feats[f"ecal_{leg}_E"],
                    np.nan)
            feats[f"has_hcal_{leg}"] = feats[f"hcal_{leg}_matched"].fillna(False).astype(int)

    drich = drich_track_link(path, tree, campaign)
    if not drich.empty:
        feats = feats.merge(drich, on=["event", "track_idx"], how="left")
    for c in ("has_drich_gas", "has_drich_aerogel"):
        feats[c] = feats.get(c, pd.Series(0, index=feats.index)).fillna(0).astype(int)
    if "drich_gas_pathlength" not in feats.columns:
        feats["drich_gas_pathlength"] = np.nan
    if "drich_aerogel_pathlength" not in feats.columns:
        feats["drich_aerogel_pathlength"] = np.nan

    evt = irt_event_summary(path, campaign)
    if not evt.empty:
        feats = feats.merge(evt, on="event", how="left")

    ctx = feats.groupby("event").agg(n_tracks_evt=("track_idx", "size")).reset_index()
    feats = feats.merge(ctx, on="event", how="left")
    for det, clu in calo.items():
        tag = det.replace("EcalEndcap", "ecal_").replace("HcalEndcap", "hcal_").replace(
            "LFHCAL", "lfhcal").lower()
        s = clu.groupby("event")["E"].agg(["size", "sum"]).reset_index().rename(
            columns={"size": f"n_{tag}_clusters_evt", "sum": f"e_{tag}_sum_evt"})
        feats = feats.merge(s, on="event", how="left")

    if enable_ionisation:
        ion = ionisation_proxy(path, tree, campaign)
        if not ion.empty:
            feats = feats.merge(ion, on=["event", "track_idx"], how="left")

    eta = feats["eta"].to_numpy(dtype=float)
    leg = np.select(
        [eta <= config.BACKWARD_ETA_MAX, eta >= config.FORWARD_ETA_MIN],
        ["backward", "forward"], default="central").astype(object)
    leg[~np.isfinite(eta)] = np.nan
    feats["leg"] = leg
    return feats


# ---------------------------------------------------------------------------
# Multi-file driver
# ---------------------------------------------------------------------------

def cache_file_for(cache_dir, path, campaign, legs, enable_ionisation):
    material = "|".join([SCHEMA_VERSION, os.path.abspath(path), campaign,
                         ",".join(legs), str(bool(enable_ionisation))])
    return os.path.join(cache_dir, hashlib.sha256(material.encode()).hexdigest() + ".pkl")


def build_features(files, *, dataset_tag, campaign=None, legs=("backward", "forward"),
                   max_failures=0, cache_dir=config.FEATURE_CACHE_DIR, limit_files=None,
                   enable_ionisation=False, progress=True) -> pd.DataFrame:
    """Feature table over many files, concatenated with per-file provenance.

    ``max_failures`` tolerates a flaky endpoint; every skip is printed and
    recorded in ``attrs`` so a partial sample can never be mistaken for a
    complete one. ``cache_dir`` makes a relaunch free over finished files.
    """
    campaign = campaign or schema.campaign_of(dataset_tag)
    files = list(files)
    if limit_files:
        files = files[:limit_files]
    frames, skipped, empty = [], [], []
    for i, path in enumerate(files):
        cp = cache_file_for(cache_dir, path, campaign, legs, enable_ionisation) if cache_dir else None
        if cp and os.path.exists(cp):
            try:
                df = pd.read_pickle(cp)
                df["file_id"] = i
                df["source_file"] = path
                frames.append(df.reset_index(drop=True))
                continue
            except Exception as exc:  # noqa: BLE001 - a bad cache entry is not fatal
                print(f"[features] WARNING: ignoring unusable cache {cp}: {exc}", file=sys.stderr)
        try:
            df = build_features_one_file(path, campaign, legs=legs, max_failures=max_failures,
                                         enable_ionisation=enable_ionisation)
        except Exception as exc:  # noqa: BLE001
            skipped.append(path)
            print(f"[features] WARNING: file failed ({len(skipped)}/{max_failures + 1} "
                  f"allowed): {path}\n  {exc.__class__.__name__}: {exc}", file=sys.stderr)
            if len(skipped) > max_failures:
                raise RuntimeError(
                    f"pid.features.build_features: {len(skipped)} files failed "
                    f"(max allowed = {max_failures})") from exc
            continue
        if df.empty:
            # Not a failure (no budget consumed, not listed as skipped), but
            # never silent: an empty file still shrinks the sample.
            print(f"[features] WARNING: {path} yielded zero candidate rows",
                  file=sys.stderr)
            empty.append(path)
            continue
        df["file_id"] = i
        df["source_file"] = path
        if cp:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{cp}.tmp.{os.getpid()}"
            df.to_pickle(tmp)
            os.replace(tmp, cp)
        frames.append(df)
        if progress and (i + 1) % 10 == 0:
            print(f"[features] {i + 1}/{len(files)} files", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update({"skipped_files": skipped, "empty_files": empty, "n_files": len(files),
                      "campaign": campaign, "schema_version": SCHEMA_VERSION})
    return out
