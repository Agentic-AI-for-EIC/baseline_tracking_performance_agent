"""PID branch schema: what is readable, what is a lie, and what is missing.

Everything in this module was established by *reading the reference files*, not
by trusting a datamodel document. Three facts drive the design:

1. Campaigns differ. 26.02.0 (clean) has 5,297 branches, 26.07.1 (``bkg_mixed``)
   has 6,945. Some collections were renamed (``TOFBarrelRecHits`` ->
   ``TOFBarrelClusterHits``), some fields exist only in the newer one
   (``Cluster.radius`` / ``dispersion`` / ``principalAxes*``). A branch name can
   therefore never be hardcoded - it is looked up per campaign below.

2. Some collections are present but *unusable*, and a naive reader would
   silently train on garbage. Each is recorded in :data:`DEAD_LINKS` with the
   measurement that condemns it, and :mod:`pid.links` refuses to join on them.

3. The calorimeters that matter for electrons are single-plane in these
   productions (``EcalEndcap{N,P}RecHits.position.z`` is one constant,
   ``layer == -1``), so a *longitudinal* depth profile is only computable where
   the detector is genuinely layered (barrel ScFi: 12 layers, LFHCAL: 7, barrel
   imaging/presampler: 4-6). Transverse granularity is real in both endcaps
   (65x64 and 205x156 distinct x/y values), so transverse shapes are computed
   from hits rather than trusted from the producer's unlabelled shape vector.

ML environment
--------------
The classifiers are not in the container's own site-packages; they live in
``/home/wxie/.eic_python_pkgs``, which eic-shell puts first on ``PYTHONPATH``.
Nothing is installed by this project (AGENTS.md). :func:`assert_ml_env` turns
"that directory vanished" into an immediate, explained failure instead of a
confusing ``ModuleNotFoundError`` deep inside a multi-hour run.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass

#: Directory that provides lightgbm / xgboost / scikit-learn in this environment.
ML_SITE_PACKAGES: str = "/home/wxie/.eic_python_pkgs"

#: package -> (import name, minimum version this project is written against)
ML_REQUIREMENTS: dict[str, str] = {
    "lightgbm": "4.0",
    "xgboost": "2.0",
    "sklearn": "1.3",
}


def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in v.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def assert_ml_env(strict: bool = True) -> dict[str, str]:
    """Check the classifier stack is importable; return the versions found.

    With ``strict=False`` this only reports and returns what it found, which is
    what the ``schema-check`` command uses so a human sees the whole picture at
    once instead of one error per run.

    Raises
    ------
    RuntimeError
        If a required package is missing or too old and ``strict`` is set. The
        message names :data:`ML_SITE_PACKAGES` and the ``PYTHONPATH`` remedy,
        because the fix is an environment variable, never a package install.
    """
    found: dict[str, str] = {}
    problems: list[str] = []

    if ML_SITE_PACKAGES not in sys.path and os.path.isdir(ML_SITE_PACKAGES):
        problems.append(
            f"{ML_SITE_PACKAGES} exists but is not on sys.path "
            "(start the session inside eic-shell, or export "
            f'PYTHONPATH="{ML_SITE_PACKAGES}:$PYTHONPATH")'
        )

    for pkg, minimum in ML_REQUIREMENTS.items():
        try:
            mod = importlib.import_module(pkg)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            problems.append(f"{pkg} not importable ({exc.__class__.__name__}: {exc})")
            continue
        ver = getattr(mod, "__version__", "0")
        found[pkg] = ver
        if _version_tuple(ver) < _version_tuple(minimum):
            problems.append(f"{pkg} {ver} is older than the supported {minimum}")

    # numpy is shadowed by the ML stack (that directory ships its own numpy);
    # stamp the resolved version so a container change is visible in provenance.
    try:
        found["numpy"] = importlib.import_module("numpy").__version__
        found["numpy_path"] = importlib.import_module("numpy").__file__ or "?"
    except Exception:  # pragma: no cover - numpy is always present here
        pass
    for extra in ("pandas", "awkward", "uproot", "torch", "ROOT"):
        try:
            found[extra] = importlib.import_module(extra).__version__
        except Exception:
            found[extra] = "missing"

    if problems and strict:
        raise RuntimeError(
            "pid.schema.assert_ml_env: the machine-learning stack required by "
            "this pipeline is not usable in this interpreter.\n  - "
            + "\n  - ".join(problems)
            + "\nNothing is installed by this project; make sure the session is "
              "inside eic-shell so that PYTHONPATH includes "
              f"{ML_SITE_PACKAGES}."
        )
    found["problems"] = "; ".join(problems) if problems else "none"
    return found


# ---------------------------------------------------------------------------
# Branch alias table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Branch:
    """One logical input, with its branch name per campaign.

    ``campaigns`` maps campaign -> full podio branch path. A campaign absent
    from the dict means the branch does not exist there (not "was not looked
    up"): every entry below was checked against both reference files.
    """

    logical: str
    campaigns: dict[str, str]
    note: str = ""

    def for_campaign(self, campaign: str) -> str | None:
        return self.campaigns.get(campaign)

    def available(self, campaign: str) -> bool:
        return campaign in self.campaigns


CLEAN, BKG = "26.02.0", "26.07.1"


def both(branch: str, note: str = "", *, clean: str | None = None, bkg: str | None = None) -> Branch:
    """A branch present under the same name in both campaigns."""
    return Branch(
        logical="",
        campaigns={CLEAN: clean or branch, BKG: bkg or branch},
        note=note,
    )


# Collection-level groups. Each dict member is read with
# ``trkperf.io.read_flat`` (same collection => same per-event length, safe to
# flatten together) or ``trkperf.io.read_nested`` (offset vectors / relations).
BRANCHES: dict[str, Branch] = {
    # ---- reconstructed track (kinematics come from CentralCKF.reco's own
    # ---- CentralCKFTrackParameters route; CentralCKFTracks.momentum is all
    # ---- zeros in these files, see trkperf/reco.py) ------------------------
    "track_time": both("CentralCKFTracks.time",
                       "fit time; ~3 us offset makes it event-scoped, so it is "
                       "used only relative to other tracks of the same event"),
    "track_time_error": both("CentralCKFTracks.timeError"),
    # ---- ECAL clusters (one per candidate) --------------------------------
    "ecal_N_energy": both("EcalEndcapNClusters.energy"),
    "ecal_N_nhits": both("EcalEndcapNClusters.nhits"),
    "ecal_N_time": both("EcalEndcapNClusters.time"),
    "ecal_N_pos_x": both("EcalEndcapNClusters.position.x"),
    "ecal_N_pos_y": both("EcalEndcapNClusters.position.y"),
    "ecal_N_pos_z": both("EcalEndcapNClusters.position.z"),
    "ecal_N_shape_begin": both("EcalEndcapNClusters.shapeParameters_begin"),
    "ecal_N_shape_end": both("EcalEndcapNClusters.shapeParameters_end"),
    "ecal_N_shape_values": both("_EcalEndcapNClusters_shapeParameters"),
    "ecal_N_hits_begin": both("EcalEndcapNClusters.hits_begin"),
    "ecal_N_hits_end": both("EcalEndcapNClusters.hits_end"),
    "ecal_N_hit_idx": both("_EcalEndcapNClusters_hits.index"),
    "ecal_P_energy": both("EcalEndcapPClusters.energy"),
    "ecal_P_nhits": both("EcalEndcapPClusters.nhits"),
    "ecal_P_time": both("EcalEndcapPClusters.time"),
    "ecal_P_pos_x": both("EcalEndcapPClusters.position.x"),
    "ecal_P_pos_y": both("EcalEndcapPClusters.position.y"),
    "ecal_P_pos_z": both("EcalEndcapPClusters.position.z"),
    "ecal_P_shape_begin": both("EcalEndcapPClusters.shapeParameters_begin"),
    "ecal_P_shape_end": both("EcalEndcapPClusters.shapeParameters_end"),
    "ecal_P_shape_values": both("_EcalEndcapPClusters_shapeParameters"),
    "ecal_P_hits_begin": both("EcalEndcapPClusters.hits_begin"),
    "ecal_P_hits_end": both("EcalEndcapPClusters.hits_end"),
    "ecal_P_hit_idx": both("_EcalEndcapPClusters_hits.index"),
    # ---- ECAL clusters: fields that only exist in the newer campaign ------
    "ecal_P_radius": Branch(
        "ecal_P_radius", {BKG: "EcalEndcapPClusters.radius"},
        "bkg-only producer shape; excluded from the headline model "
        "(campaign-skew rule, PLAN_pid.md 3.3)"),
    "ecal_P_dispersion": Branch(
        "ecal_P_dispersion", {BKG: "EcalEndcapPClusters.dispersion"}, "bkg-only"),
    # ---- ECAL hits (transverse shower shape; single-plane in z) -----------
    "ecal_N_hit_energy": both("EcalEndcapNRecHits.energy"),
    "ecal_N_hit_x": both("EcalEndcapNRecHits.position.x"),
    "ecal_N_hit_y": both("EcalEndcapNRecHits.position.y"),
    "ecal_N_hit_z": both("EcalEndcapNRecHits.position.z"),
    "ecal_N_hit_time": both("EcalEndcapNRecHits.time"),
    "ecal_P_hit_energy": both("EcalEndcapPRecHits.energy"),
    "ecal_P_hit_x": both("EcalEndcapPRecHits.position.x"),
    "ecal_P_hit_y": both("EcalEndcapPRecHits.position.y"),
    "ecal_P_hit_z": both("EcalEndcapPRecHits.position.z"),
    "ecal_P_hit_time": both("EcalEndcapPRecHits.time"),
    # ---- HCAL (hadronic leakage: electron veto / pion-MIP tag) ------------
    "hcal_N_energy": both("HcalEndcapNClusters.energy"),
    "hcal_N_time": both("HcalEndcapNClusters.time"),
    "hcal_N_nhits": both("HcalEndcapNClusters.nhits"),
    "hcal_N_pos_x": both("HcalEndcapNClusters.position.x"),
    "hcal_N_pos_y": both("HcalEndcapNClusters.position.y"),
    "hcal_N_pos_z": both("HcalEndcapNClusters.position.z"),
    "lfhcal_energy": both("LFHCALClusters.energy"),
    "lfhcal_time": both("LFHCALClusters.time"),
    "lfhcal_nhits": both("LFHCALClusters.nhits"),
    "lfhcal_pos_x": both("LFHCALClusters.position.x"),
    "lfhcal_pos_y": both("LFHCALClusters.position.y"),
    "lfhcal_pos_z": both("LFHCALClusters.position.z"),
    # ---- barrel EMCal: the only genuinely longitudinally-segmented EM calo -
    "ecalbarrel_scfi_energy": both("EcalBarrelScFiClusters.energy"),
    "ecalbarrel_scfi_nhits": both("EcalBarrelScFiClusters.nhits"),
    "ecalbarrel_scfi_pos_x": both("EcalBarrelScFiClusters.position.x"),
    "ecalbarrel_scfi_pos_y": both("EcalBarrelScFiClusters.position.y"),
    "ecalbarrel_scfi_pos_z": both("EcalBarrelScFiClusters.position.z"),
    "ecalbarrel_scfi_hit_energy": both("EcalBarrelScFiRecHits.energy"),
    "ecalbarrel_scfi_hit_layer": both("EcalBarrelScFiRecHits.layer"),
    "ecalbarrel_scfi_hit_r": both("EcalBarrelScFiRecHits.position.z"),
    "ecalbarrel_imaging_energy": both("EcalBarrelImagingClusters.energy"),
    "ecalbarrel_imaging_pos_x": both("EcalBarrelImagingClusters.position.x"),
    "ecalbarrel_imaging_pos_y": both("EcalBarrelImagingClusters.position.y"),
    "ecalbarrel_imaging_pos_z": both("EcalBarrelImagingClusters.position.z"),
    # ---- track<->cluster matches (link verified, weight is not) ----------
    "match_ecal_N_track": both("_EcalEndcapNTrackClusterMatches_track.index"),
    "match_ecal_N_cluster": both("_EcalEndcapNTrackClusterMatches_cluster.index"),
    "match_ecal_P_track": both("_EcalEndcapPTrackClusterMatches_track.index"),
    "match_ecal_P_cluster": both("_EcalEndcapPTrackClusterMatches_cluster.index"),
    "match_hcal_N_track": both("_HcalEndcapNTrackClusterMatches_track.index"),
    "match_hcal_N_cluster": both("_HcalEndcapNTrackClusterMatches_cluster.index"),
    "match_lfhcal_track": both("_LFHCALTrackClusterMatches_track.index"),
    "match_lfhcal_cluster": both("_LFHCALTrackClusterMatches_cluster.index"),
    # ---- track extrapolation to the calorimeter faces --------------------
    "proj_track": both("_CalorimeterTrackProjections_track.index",
                       "verified -> CentralCKFTracks; ~all tracks have one"),
    "proj_point_theta": both("_CalorimeterTrackProjections_points.theta"),
    "proj_point_phi": both("_CalorimeterTrackProjections_points.phi"),
    "proj_point_x": both("_CalorimeterTrackProjections_points.position.x"),
    "proj_point_y": both("_CalorimeterTrackProjections_points.position.y"),
    "proj_point_z": both("_CalorimeterTrackProjections_points.position.z"),
    "proj_point_pathlength": both("_CalorimeterTrackProjections_points.pathlength"),
    "proj_point_surface": both("_CalorimeterTrackProjections_points.surface"),
    "proj_points_begin": both("CalorimeterTrackProjections.points_begin"),
    "proj_points_end": both("CalorimeterTrackProjections.points_end"),
    # ---- dRICH per-track extrapolation (link verified) -------------------
    "drich_gas_track": both("_DRICHGasTracks_track.index",
                            "verified -> CentralCKFTracks"),
    "drich_gas_length": both("DRICHGasTracks.length"),
    "drich_aerogel_track": both("_DRICHAerogelTracks_track.index"),
    "drich_aerogel_length": both("DRICHAerogelTracks.length"),
    # ---- Cherenkov photon counting (event-level only, see DEAD_LINKS) ----
    "irt_gas_npe": both("DRICHGasIrtCherenkovParticleID.npe"),
    "irt_aerogel_npe": both("DRICHAerogelIrtCherenkovParticleID.npe"),
    # ---- recoil/ionisation chain (reco-side, leakage-free) ---------------
    "track_meas_begin": both("CentralCKFTracks.measurements_begin"),
    "track_meas_end": both("CentralCKFTracks.measurements_end"),
    "track_meas_idx": both("_CentralCKFTracks_measurements.index"),
    "meas_hits_begin": both("CentralTrackerMeasurements.hits_begin"),
    "meas_hits_end": both("CentralTrackerMeasurements.hits_end"),
    "meas_hit_idx": both("_CentralTrackerMeasurements_hits.index"),
    "meas_hit_cid": both("_CentralTrackerMeasurements_hits.collectionID"),
    "si_hit_edep": both("SiBarrelTrackerRecHits.edep"),
    "mpgd_hit_edep": both("MPGDBarrelRecHits.edep"),
    # ---- TOF (timing exists, but see DEAD_LINKS: not track-attachable) ---
    "tof_barrel_time": both("TOFBarrelClusterHits.time"),
    "tof_endcap_time": Branch("tof_endcap_time", {BKG: "TOFEndcapClusterHits.time"},
                              "bkg-only under this name"),
}

for _k, _b in BRANCHES.items():
    object.__setattr__(_b, "logical", _k)


# ---------------------------------------------------------------------------
# Verified dead ends.
#
# Each entry records what was measured, so a future maintainer who "fixes" the
# code by re-enabling one of these can see exactly why it is not enabled.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeadLink:
    name: str
    reason: str
    evidence: str = ""
    family: str = "producer_pid"


DEAD_LINKS: tuple[DeadLink, ...] = (
    DeadLink(
        "DIRCParticleIDs",
        "edm4hep::ParticleIDData.particle relation is a null reference, so the "
        "per-hypothesis likelihood cannot be attached to any track",
        "_DIRCParticleIDs_particle.index == -2 and .collectionID == 0 for all "
        "entries in both campaigns; parameters_begin == parameters_end (empty)",
    ),
    DeadLink("DRICHParticleIDs", "as DIRCParticleIDs",
             "_DRICHParticleIDs_particle.index == -2 in both campaigns"),
    DeadLink("RICHEndcapNParticleIDs", "as DIRCParticleIDs",
             "_RICHEndcapNParticleIDs_particle.index == -2"),
    DeadLink("CombinedTOFParticleIDs", "as DIRCParticleIDs",
             "_CombinedTOFParticleIDs_particle.index == -2; 9.2 entries/event"),
    DeadLink("ReconstructedChargedRealPIDParticleIDs", "as DIRCParticleIDs",
             "_..._particle.index == -2"),
    DeadLink(
        "DRICHGasIrtCherenkovParticleID.chargedParticle",
        "relation points at a collection that was never written to the file",
        "collectionID 1290518152 is absent from "
        "podio_metadata.events___CollectionTypeInfo (282 registered "
        "collections in 26.02.0, 387 in 26.07.1); hypotheses themselves are "
        "populated (4 per object, PDG 11..2212), so they are aggregated "
        "event-level only",
        family="cherenkov",
    ),
    DeadLink(
        "DRICHAerogelIrtCherenkovParticleID.chargedParticle",
        "same dangling collectionID as the gas IRT object",
        "collectionID 1290518152 not in the registry",
        family="cherenkov",
    ),
    DeadLink(
        "*TrackClusterMatches.weight",
        "match weight is 0.0 for every entry -> carries no information",
        "min == max == 0.0 over all matches in both campaigns; Delta R is "
        "recomputed instead (pid.links.match_clusters)",
        family="calorimeter",
    ),
    DeadLink(
        "EcalEndcap{N,P}RecHits depth",
        "the two endcap ECALs are single-plane in these productions, so no "
        "longitudinal shower profile exists for them",
        "EcalEndcapNRecHits.position.z has exactly 1 distinct value "
        "(-1850.13 mm) and EcalEndcapPRecHits likewise (3522.28 mm); "
        "layer == -1; subdetectorEnergies_begin == _end; energyError == 0",
        family="calorimeter",
    ),
    DeadLink(
        "EcalBarrelTrackClusterMatches",
        "empty in the clean campaign -> no barrel E/p path there",
        "0 matches in 26.02.0 (EcalBarrelClusters: 0.06/event)",
        family="calorimeter",
    ),
    DeadLink(
        "CentralTrackerMeasurements in 26.07.1",
        "the clean campaign's measurement collection is not written in the newer "
        "one (renamed to CentralWithoutTOFTrackerMeasurements, with the TOF cluster "
        "hits folded into the track fit), so the optional reco-side ionisation "
        "proxy is only complete in 26.02.0 and pid.features degrades it to NaN here",
        "audited: _CentralCKFTracks_measurements resolves to TOFBarrelClusterHits + "
        "TOFEndcapClusterHits + CentralWithoutTOFTrackerMeasurements (3 targets) and "
        "CentralTrackerMeasurements.* branches are absent from the tree",
        family="ionisation",
    ),
    DeadLink(
        "dE/dx (thickness-normalised)",
        "no per-track dE/dx collection exists, and TrackerHitData has no "
        "pathLength field, so Bethe-Bloch normalisation is impossible",
        "branch scan finds no *Dedx*/*Trunc* collection; SimTrackerHit "
        "pathLength exists only on TRUTH hits",
        family="ionisation",
    ),
    DeadLink(
        "SiBarrelHits.eDep / pathLength as a feature",
        "truth-derived: computing ionisation from the truth particle's own "
        "hits and then classifying that same truth particle leaks the label",
        "kept available as a validation column only (family 'truth_dedx', "
        "blocked by config.LEAKAGE_BLOCKED_FAMILIES)",
        family="truth_dedx",
    ),
    DeadLink(
        "TOF hit -> track association (26.02.0)",
        "TOF cluster hits are 2D measurements (surface + local coordinates), so "
        "attaching one to a track needs cellID/surface decoding, which AGENTS.md "
        "puts out of scope; in the clean campaign the track->measurement chain "
        "does not contain them",
        "audited: _CentralCKFTracks_measurements -> CentralTrackerMeasurements "
        "only. TOFBarrelClusterHits.time IS populated (~7/event) but the "
        "ParticleID summary built from it is dead (above)",
        family="timing",
    ),
    DeadLink(
        "TOF timing in the CLEAN campaign (campaign-asymmetric)",
        "in 26.07.1 the track->measurement chain DOES include the TOF cluster "
        "hits, i.e. TOF is part of the central track fit there, so a per-track "
        "timing feature is reachable in the +background campaign but not in "
        "26.02.0 - therefore it cannot enter the headline clean-vs-bkg model "
        "without breaking the comparison",
        "audited: _CentralCKFTracks_measurements -> TOFBarrelClusterHits + "
        "TOFEndcapClusterHits + CentralWithoutTOFTrackerMeasurements (3 targets) "
        "in 26.07.1; CentralTrackerMeasurements (1 target) in 26.02.0. Recorded "
        "as a follow-up item, not a v1 feature",
        family="timing",
    ),
)


@dataclass
class FeatureSpec:
    """Catalogue entry: one model column, its family, and how to describe it."""

    column: str
    family: str
    description: str
    #: campaigns in which the underlying branch exists (None = both)
    campaigns: tuple[str, ...] | None = None


#: Number of unlabelled floats the producer stores per endcap-ECAL cluster.
#: Correlated against quantities computed from hits: p0 and p1 track the
#: computed transverse width (r = +0.93 / +0.95), the rest are unidentified.
#: They are carried as ``shape_i`` (explicitly labelled "producer, unlabelled")
#: and the headline shower shapes are recomputed from hits instead, so this
#: project never depends on an undocumented index.
N_ECAL_SHAPE_PARAMS = 7


def branches_available(campaign: str, keys: list[str] | None = None) -> dict[str, str]:
    """``{logical: branch}`` for the branches that exist in `campaign`."""
    keys = list(BRANCHES) if keys is None else keys
    out: dict[str, str] = {}
    for k in keys:
        b = BRANCHES[k].for_campaign(campaign)
        if b:
            out[k] = b
    return out


def missing_branches(campaign: str, keys: list[str]) -> list[str]:
    """Keys whose branch does not exist in `campaign` (for explicit reporting)."""
    return [k for k in keys if not BRANCHES[k].available(campaign)]


def campaign_of(dataset_tag: str) -> str:
    """Map a trkperf dataset tag to its production campaign."""
    from . import config

    try:
        return config.CAMPAIGN_BY_DATASET_TAG[dataset_tag]
    except KeyError as exc:  # pragma: no cover - argparse restricts the choices
        raise ValueError(
            f"unknown dataset tag {dataset_tag!r}; expected one of "
            f"{sorted(config.CAMPAIGN_BY_DATASET_TAG)}"
        ) from exc


def dead_link_report() -> str:
    """Human-readable summary of :data:`DEAD_LINKS` (used by ``schema-check``)."""
    lines = []
    for d in DEAD_LINKS:
        lines.append(f"  [{d.family}] {d.name}\n      why : {d.reason}\n      meas: {d.evidence}")
    return "\n".join(lines)
