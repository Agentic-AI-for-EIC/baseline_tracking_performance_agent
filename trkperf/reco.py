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

import numpy as np
import pandas as pd

from . import io
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


def read_reco_tracks(file_paths: list[str], *, max_failures: int = 0) -> pd.DataFrame:
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
    params = io.read_flat_multi(file_paths, _TRACK_PARAM_COLUMNS, max_failures=max_failures)
    tracks = io.read_flat_multi(file_paths, _CENTRALCKFTRACKS_COLUMNS, max_failures=max_failures)

    if params.empty:
        return pd.DataFrame(
            columns=[*_JOIN_KEYS, "px", "py", "pz", "charge", "pdg", "chi2", "ndf",
                     "p", "pt", "eta", "phi"]
        )

    merged = params.merge(tracks, on=_JOIN_KEYS, how="left")

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
