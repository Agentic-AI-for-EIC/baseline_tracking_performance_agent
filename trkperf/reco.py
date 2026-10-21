"""Reconstructed-track kinematics (CentralCKFTracks / CentralCKFTrackParameters).

IMPORTANT data-model note (verified live on real files):
`CentralCKFTracks.momentum.x/y/z` (the stored 3-momentum copy) is *all
zeros* in every campaign file this project processes. The actual perigee
track parameters live in the `CentralCKFTrackParameters` collection
(`qOverP`, `theta`, `phi`), which is populated and aligns 1:1 with
`CentralCKFTracks` by event + position (identical per-event counts; the
`_CentralCKFTracks_tracks` relation that would link them is empty in these
files). `reco.py` therefore reads the momentum from `CentralCKFTrackParameters`
and reconstructs Cartesian components, rather than trusting
`CentralCKFTracks.momentum`. The charge/chi2/ndf/pdg still come from
`CentralCKFTracks` (those are populated there).

p is recovered as p = 1/|qOverP|, then
px = p*sin(theta)*cos(phi), py = p*sin(theta)*sin(phi), pz = p*cos(theta),
and fed through the same `add_kinematics` used on the truth side so truth and
reco are never computed with different formulas.

The public output contract (columns px, py, pz, p, pt, eta, phi, charge,
pdg, chi2, ndf, plus the file_id/event/idx bookkeeping) is unchanged from
what the rest of trkperf expects - so matching.py, resolution.py,
efficiency.py, fake_rate.py, and pid/confusion.py need no changes.
"""

from __future__ import annotations

import hashlib
import os
import sys

import awkward as ak
import numpy as np
import pandas as pd

from . import config, io
from .truth import add_kinematics  # reuse, do not reimplement

#: populates new/changed code below
_TRACK_PARAM_COLUMNS = {
    "qOverP": "CentralCKFTrackParameters/CentralCKFTrackParameters.qOverP",
    "theta": "CentralCKFTrackParameters/CentralCKFTrackParameters.theta",
    "phi": "CentralCKFTrackParameters/CentralCKFTrackParameters.phi",
}

_CENTRALCKFTRACKS_COLUMNS = {
    "charge": "CentralCKFTracks/CentralCKFTracks.charge",
    "pdg": "CentralCKFTracks/CentralCKFTracks.pdg",
    "chi2": "CentralCKFTracks/CentralCKFTracks.chi2",
    "ndf": "CentralCKFTracks/CentralCKFTracks.ndf",
}

_JOIN_KEYS = ["file_id", "event", "idx"]


def read_reco_tracks(file_paths: list[str], *, max_failures: int = 0,
                     shared_failures: dict | None = None) -> pd.DataFrame:
    """Read reconstructed tracks and their perigee momentum from every file.

    Reads `CentralCKFTrackParameters` (qOverP/theta/phi - the only place the
    momentum is actually stored) plus the populated scalar members from
    `CentralCKFTracks` (charge/pdg/chi2/ndf), reconstructs px/py/pz from the
    parameters, and computes p/pt/eta/phi via the shared `add_kinematics`.

    Returns
    -------
    DataFrame with columns: file_id, event, idx, px, py, pz, charge, pdg,
    chi2, ndf, p, pt, eta, phi.

    Note: `idx` is the position within the collection for that (file_id,
    event) - the value referenced by `_CentralCKFTrackAssociations_rec.index`
    (see matching.py). We rely on CentralCKFTrackParameters aligning 1:1 with
    CentralCKFTracks by (event, idx); verified on real files (see module
    docstring).
    """
    # Read the perigee parameters and the populate scalar track members.
    params = io.read_flat_multi(file_paths, _TRACK_PARAM_COLUMNS, max_failures=max_failures,
                                shared_failures=shared_failures)
    tracks = io.read_flat_multi(file_paths, _CENTRALCKFTRACKS_COLUMNS, max_failures=max_failures,
                                shared_failures=shared_failures)

    if params.empty:
        return pd.DataFrame(
            columns=[*_JOIN_KEYS, "px", "py", "pz", "charge", "pdg", "chi2", "ndf",
                     "p", "pt", "eta", "phi"]
        )

    merged = params.merge(tracks, on=_JOIN_KEYS, how="left", validate="one_to_one")

    with np.errstate(divide="ignore", invalid="ignore"):
        # qOverP = q/p, so |p| = 1/|qOverP|; sign of charge from the track
        # member. Guard against qOverP == 0 (p undefined -> NaN, not a crash).
        qov = np.asarray(merged["qOverP"], dtype=float)
        abs_p = np.where(np.abs(qov) > 0, 1.0 / np.abs(qov), np.nan)
        theta = np.asarray(merged["theta"], dtype=float)
        phi = np.asarray(merged["phi"], dtype=float)

        merged["px"] = abs_p * np.sin(theta) * np.cos(phi)
        merged["py"] = abs_p * np.sin(theta) * np.sin(phi)
        merged["pz"] = abs_p * np.cos(theta)

    merged = add_kinematics(merged)
    # Keep the same column contract as before; qOverP/theta/phi are internal.
    out = merged[[*_JOIN_KEYS, "px", "py", "pz", "charge", "pdg", "chi2", "ndf",
                  "p", "pt", "eta", "phi"]]
    return out


# ---------------------------------------------------------------------------
# Reconstructed-track hit composition (for an optional reco-side quality cut).
# ---------------------------------------------------------------------------

#: Branches describing the CKF track -> measurement relation.
_MEAS_REL = "_CentralCKFTracks_measurements/_CentralCKFTracks_measurements.{field}"


def _registry_for(path: str) -> dict[int, str]:
    """collectionID -> name, read from this file's podio_metadata registry."""
    with io.uproot.open(path) as f:
        arr = f["podio_metadata/events___CollectionTypeInfo"].array(library="ak")
    return {int(c): n for c, n in zip(
        arr[0]["events___CollectionTypeInfo.collectionID"],
        arr[0]["events___CollectionTypeInfo.name"])}


def _hits_of_measurement_collection(tree, coll: str):
    """Return (begin, end, hit_collID) awkward arrays for a measurement
    collection ``coll`` whose entries each wrap >=1 hit, or ``None`` if the
    collection is not a measurement container (no ``.hits_begin`` member)."""
    try:
        begin = tree[f"{coll}/{coll}.hits_begin"].array(library="ak")
        end = tree[f"{coll}/{coll}.hits_end"].array(library="ak")
        hit_cid = tree[f"_{coll}_hits/_{coll}_hits.collectionID"].array(library="ak")
    except Exception:  # noqa: BLE001 - not a measurement collection
        return None
    return begin, end, hit_cid


def _hit_counts_one_file(path: str) -> pd.DataFrame:
    """Per (file_id=0, event, idx) reconstructed-track tracker/silicon hit counts.

    Walks CentralCKFTracks.measurements -> its measurement collection -> the
    underlying *RecHits collection (resolved via the file's podio registry),
    and also handles measurement objects that are already hits (e.g. TOF),
    which are excluded from the tracker total by config.is_tracking_reco_hit.

    podio stores the track->measurement relation as ONE flat ObjectID list per
    event; a track's measurements are the slice [measurements_begin[t] :
    measurements_end[t]] of that list. Each ObjectID has an .index (position in
    its target collection) and a .collectionID (which collection). If the
    target wraps hits (has ``.hits_begin``) we deref to the underlying hit's
    collection; otherwise the target *is* the hit.
    """
    tree = io.open_tree(path)
    id2name = _registry_for(path)
    name2id = {v: k for k, v in id2name.items()}

    meas_begin = tree["CentralCKFTracks/CentralCKFTracks.measurements_begin"].array(library="ak")
    meas_end = tree["CentralCKFTracks/CentralCKFTracks.measurements_end"].array(library="ak")
    rel_index = tree[_MEAS_REL.format(field="index")].array(library="ak")
    rel_cid = tree[_MEAS_REL.format(field="collectionID")].array(library="ak")
    n_events = len(meas_begin)

    # Which object collections are hit-wrapping measurement containers?
    obj_names = {id2name.get(int(c), "") for c in ak.flatten(rel_cid).to_list() if True}
    obj_names.discard("")
    wrapped: dict[str, tuple] = {}          # name -> (begin, end, hit_cid) ak arrays
    meas_coll_ids: set[int] = set()
    for name in obj_names:
        w = _hits_of_measurement_collection(tree, name)
        if w is not None:
            wrapped[name] = w
            meas_coll_ids.add(name2id[name])

    rows = []
    for ev in range(n_events):
        mb = np.asarray(meas_begin[ev], dtype=np.int64)
        me = np.asarray(meas_end[ev], dtype=np.int64)
        ri = np.asarray(rel_index[ev], dtype=np.int64)
        rc = np.asarray(rel_cid[ev], dtype=np.int64)
        # per-event, per-measurement-collection underlying-hit offsets / cids
        wdata = {}
        for name, (b, e, hcid) in wrapped.items():
            wdata[name] = (np.asarray(b[ev], dtype=np.int64),
                           np.asarray(e[ev], dtype=np.int64),
                           np.asarray(hcid[ev], dtype=np.int64))
        for t in range(len(mb)):
            cnt_h = cnt_s = 0
            for j in range(int(mb[t]), int(me[t])):
                ocid = int(rc[j])
                if ocid in meas_coll_ids:
                    oname = id2name.get(ocid, "")
                    if oname not in wdata:
                        continue
                    b_ev, e_ev, hcid_ev = wdata[oname]
                    m_local = int(ri[j])
                    if m_local >= len(b_ev):
                        continue
                    lo, hi = int(b_ev[m_local]), int(e_ev[m_local])
                    hit_names = [id2name.get(int(h), "") for h in hcid_ev[lo:hi]]
                else:
                    hit_names = [id2name.get(ocid, "")]   # direct hit object
                for hn in hit_names:
                    if config.is_tracking_reco_hit(hn):
                        cnt_h += 1
                        if config.is_silicon_reco_hit(hn):
                            cnt_s += 1
            rows.append((0, ev, t, cnt_h, cnt_s))
    return pd.DataFrame(rows, columns=["file_id", "event", "idx",
                                        "n_reco_hits", "n_reco_si_hits"])


def _reco_hit_cache_path(cache_dir: str, path: str) -> str:
    key = hashlib.sha256(f"reco_track_hits\x00{os.path.abspath(path)}".encode()).hexdigest()
    return os.path.join(cache_dir, key + ".pkl")


def read_reco_track_hit_counts(file_paths: list[str], *, max_failures: int = 0,
                               shared_failures: dict | None = None,
                               cache_dir: str | None = None) -> pd.DataFrame:
    """Reconstructed-tracker hit composition per track.

    Returns a DataFrame with columns file_id, event, idx (position within
    ``CentralCKFTracks``, the value ``_CentralCKFTrackAssociations_rec.index``
    references), ``n_reco_hits`` (tracker space-point hits on the track,
    silicon + gaseous, TOF timing excluded) and ``n_reco_si_hits`` (the silicon
    subset). Used by the optional reco-side ``--min-reco-hits`` /
    ``--min-reco-si-hits`` quality cut; like :func:`read_reco_tracks` the file
    read is retried once and persistent failures respect the shared budget.
    """
    cache_dir = cache_dir if cache_dir is not None else io._CACHE_DIR
    frames = []
    skipped: list[str] = []
    if shared_failures is not None:
        skipped = shared_failures.setdefault("skipped", [])
    for file_id, path in enumerate(file_paths):
        if path in skipped:
            continue
        cpath = _reco_hit_cache_path(cache_dir, path) if cache_dir else None
        if cpath and os.path.exists(cpath):
            try:
                df = pd.read_pickle(cpath)
                df["file_id"] = file_id
                frames.append(df.reset_index(drop=True))
                continue
            except Exception as exc:  # noqa: BLE001 - best-effort cache
                print(f"[reco] WARNING: ignoring bad hit cache {cpath}: {exc}",
                      file=sys.stderr)
        last_exc = None
        for attempt in range(2):
            try:
                df = _hit_counts_one_file(path)
                df["file_id"] = file_id
                if cpath:
                    try:
                        tmp = cpath + ".tmp"
                        df.to_pickle(tmp); os.replace(tmp, cpath)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[reco] WARNING: could not stash hit cache {cpath}: {exc}",
                              file=sys.stderr)
                frames.append(df.reset_index(drop=True))
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc is not None:
            skipped.append(path)
            print(f"[reco] WARNING: skipping hit-count read ({len(skipped)} failed, "
                  f"max={max_failures}): {path}\n  {last_exc}", file=sys.stderr)
            if len(skipped) > max_failures:
                raise RuntimeError(
                    f"trkperf.reco.read_reco_track_hit_counts: {len(skipped)} files "
                    f"failed (max allowed = {max_failures})") from last_exc
    if not frames:
        return pd.DataFrame(columns=["file_id", "event", "idx",
                                     "n_reco_hits", "n_reco_si_hits"])
    return pd.concat(frames, ignore_index=True)
