"""Central configuration for trkperf.

Every tunable number or path template used anywhere in this package lives
here. If you need to change a threshold, a bin edge, or where the data comes
from, change it in this file — the rest of the package reads from here, it
does not hardcode its own copies.

See AGENTS.md at the project root for the physics justification behind each
choice below.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Species in scope, keyed by name -> PDG code (PDG convention: particle and
# antiparticle have opposite-sign codes). Extend this dict to add a species;
# every metric module iterates over it, nothing else needs to change.
# ---------------------------------------------------------------------------
SPECIES: dict[str, int] = {
    "e-": 11,
    "e+": -11,
    "pi+": 211,
    "pi-": -211,
    "K+": 321,
    "K-": -321,
    "proton": 2212,
    "antiproton": -2212,
}

# Reverse lookup (PDG code -> species name), built once at import time.
PDG_TO_SPECIES: dict[int, str] = {pdg: name for name, pdg in SPECIES.items()}

# ---------------------------------------------------------------------------
# Binning. pT is log-spaced because tracking performance varies over orders
# of magnitude in momentum; eta is linear because detector acceptance is
# roughly uniform in bin width across the covered range.
# ---------------------------------------------------------------------------
# 9 edges -> 8 bins, spanning the momenta actually produced in these
# campaigns (roughly 0.1-20 GeV for DIS final-state particles at 10x100).
PT_BIN_EDGES: np.ndarray = np.logspace(np.log10(0.1), np.log10(20.0), 9)

# 0.5-wide eta bins spanning the central + far-forward/backward acceptance.
ETA_BIN_EDGES: np.ndarray = np.arange(-4.0, 4.01, 0.5)

# ---------------------------------------------------------------------------
# Truth-reco matching (CentralCKFTrackAssociations.weight).
# ---------------------------------------------------------------------------
# A track/particle pair is considered a genuine match if the association
# weight (the fraction of the track's hits that came from that truth
# particle) is at least this value. 0.5 = "most of the track's hits are from
# this particle" - the standard majority-matching convention.
MATCH_WEIGHT_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Acceptance: minimum number of the 7 central-tracking truth-hit collections
# (see CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS) that must register >= 1 hit
# from a truth particle for it to be counted "in acceptance" (i.e. it could
# in principle be reconstructed as a 3D track). 4 is the minimal requirement
# for an unambiguous 3D space-point track fit (fewer than 4 hits cannot
# over-constrain a helix in the presence of multiple scattering).
#
# NOTE: "collection" here is a proxy for "layer". A truth particle can leave
# several hits within the *same* collection (e.g. multiple SiBarrelHits) if
# that subsystem has several physical sensor layers; decoding the true
# per-layer count would require subsystem-specific cellID bit-unpacking
# against the detector's compact geometry description, which this project
# deliberately does not depend on. Counting "distinct collections hit" is a
# simpler, geometry-independent stand-in. Revisit if this proxy turns out to
# be too coarse (e.g. if SiBarrel alone has >= 4 physical layers, this
# definition would call a particle "in acceptance" from Si hits alone).
ACCEPTANCE_MIN_LAYERS: int = 4

CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS: tuple[str, ...] = (
    "SiBarrelHits",
    "VertexBarrelHits",
    "TrackerEndcapHits",
    "MPGDBarrelHits",
    "OuterMPGDBarrelHits",
    "BackwardMPGDEndcapHits",
    "ForwardMPGDEndcapHits",
)

# ---------------------------------------------------------------------------
# Tracking regions (see PLAN.md 4.1). The central CKF reconstructs endcap
# tracks, so no new reconstruction chain is needed - but the central >=4/7
# acceptance rule calls nearly every endcap particle out of acceptance.
# Each region therefore defines its own truth-hit collections and N_min
# (2 = the stereo minimum: two independent measurements make a segment),
# with the measured joint fractions in PLAN.md as justification.
# ---------------------------------------------------------------------------
TRACKING_REGIONS: dict[str, dict] = {
    "central": {
        "collections": CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS,
        "min_layers": ACCEPTANCE_MIN_LAYERS,  # 4
    },
    "backward": {
        "collections": (
            "BackwardMPGDEndcapHits",
            "TrackerEndcapHits",
            "TOFEndcapHits",
        ),
        "min_layers": 2,
    },
    "forward": {
        # Endcap disks only: 2 MPGD disks (one collection) + 5 Si disks
        # (in TrackerEndcapHits) + FTOF. The far-forward spectrometer
        # stations (ForwardOffM, Roman Pots) are deliberately NOT part of
        # endcap acceptance - they are a separate detector system.
        "collections": (
            "ForwardMPGDEndcapHits",
            "TrackerEndcapHits",
            "TOFEndcapHits",
        ),
        "min_layers": 2,
    },
}

# ---------------------------------------------------------------------------
# Statistics floor: a (pT, eta[, species]) bin with fewer raw entries than
# this is reported as "insufficient statistics" rather than a number, per
# AGENTS.md's Definition of done.
# ---------------------------------------------------------------------------
MIN_ENTRIES_PER_BIN: int = 50

# ---------------------------------------------------------------------------
# Momentum-resolution fit: the Gaussian core is fit over +/- this many
# "reasonable" multiples of a first-pass robust width estimate, to stay
# clear of non-Gaussian tails (bremsstrahlung, hard scattering, mis-fits)
# without needing a hand-tuned window per bin.
# ---------------------------------------------------------------------------
RESOLUTION_FIT_HIST_RANGE: tuple[float, float] = (-0.5, 0.5)  # in Delta(pT)/pT
RESOLUTION_FIT_HIST_BINS: int = 100
RESOLUTION_FIT_CORE_SIGMA: float = 3.0  # refit window = +/- N * first-pass sigma
# Physical upper bound on the reported resolution: sigma on
# Delta(pT)/pT = (pT_reco - pT_truth) / pT_truth. A width of 1.0 means a 100%
# resolution; anything >= that is unphysical (the reconstructed pT is
# uncorrelated with truth / a diverged fit) and must be flagged rather than
# reported as a real measurement. The ROOT Gaussian fit constrains sigma to
# [0, RESOLUTION_MAX_SIGMA]; a fit hitting the upper bound is treated as
# non-converged.
RESOLUTION_MAX_SIGMA: float = 1.0

# ---------------------------------------------------------------------------
# Dataset registry. Two sample "types": clean (Type 1) and the beam-induced
# background overlay (Type 2). File paths are resolved at run time via the
# rucio MCP server (list_dids / list_files / list_file_replicas) - what's
# recorded here is the *registry* of where to look and in what order to
# escalate statistics, not a literal file list.
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "clean": {
        "label": "Type 1: clean (no beam background)",
        "campaign": "26.02.0",
        "did": "epic:/RECO/26.02.0/epic_craterlake/DIS/NC/10x100/minQ2=1",
        "xrootd_prefix": "root://dtn-eic.jlab.org:1094//volatile/eic/EPIC/RECO/26.02.0"
        "/epic_craterlake/DIS/NC/10x100/minQ2=1",
        # Only one minQ2 tier exists/was requested for this sample - it does
        # not escalate across minQ2, only across file count within minQ2=1.
        "min_q2_tiers": [1],
        "files_available": {1: 4609},
        "events_per_file": {1: 1083},
        "start_n_files": 150,
    },
    "bkg_mixed": {
        "label": "Type 2: +beam-induced background (Bkg_Exact1S_2us/GoldCt/10um)",
        "campaign": "26.07.1",
        "did_template": (
            "epic:/RECO/26.07.1/epic_craterlake/Bkg_Exact1S_2us/GoldCt/10um"
            "/DIS/NC/10x100/minQ2={min_q2}"
        ),
        "xrootd_prefix_template": (
            "root://hpceph-xrootd.twgrid.org:1094//cephfs/epic/RECO/26.07.1"
            "/epic_craterlake/Bkg_Exact1S_2us/GoldCt/10um/DIS/NC/10x100/minQ2={min_q2}"
        ),
        # Escalation order: exhaust minQ2=1 first (it is what Type 1 uses),
        # then widen to the harder bins only if still statistics-limited.
        "min_q2_tiers": [1, 10, 100, 1000],
        "files_available": {1: 1463, 10: 902, 100: 547, 1000: 551},
        # Only minQ2=1 events/file was actually verified with uproot; the
        # other tiers are assumed similar (background overlay dominates the
        # per-event data volume regardless of the hard-scatter minQ2).
        "events_per_file": {1: 99, 10: 99, 100: 99, 1000: 99},
        "start_n_files": {1: 200, 10: 0, 100: 0, 1000: 0},
    },
    # ------------------------------------------------------------------
    # Same-campaign reproduction pair (2026-09-25): BOTH sides from
    # campaign 26.07.1, served from a local copy on gautschi.rcac.purdue
    # (.edu) through an ssh-tunnelled xrootd daemon (root://localhost:1294//
    # <absolute scratch path>; daemon bound to 127.0.0.1 only, tunnel with
    # `ssh -N -L 1294:127.0.0.1:1294 gautschi.rcac.purdue.edu`). This removes
    # the 26.02.0-vs-26.07.1 production difference from the clean-vs-
    # background comparison entirely. Replaces the purged JLab /volatile
    # Type 1 sample (26.02.0 is now tape-only).
    # ------------------------------------------------------------------
    "clean26071": {
        "label": "Type 1 reproduction: clean, SAME campaign 26.07.1 (local ssh copy)",
        "campaign": "26.07.1",
        "did": "epic:/RECO/26.07.1/epic_craterlake/DIS/NC/10x100/minQ2=1",
        "xrootd_prefix": (
            "root://localhost:1294//scratch/gautschi/wxie/eIC_data_small_set/clean"),
        "min_q2_tiers": [1],
        "files_available": {1: 300},
        "events_per_file": {1: 1409},
        "start_n_files": 300,
    },
    "bkg26071": {
        "label": "Type 2 reproduction: +background, campaign 26.07.1 (local ssh copy)",
        "campaign": "26.07.1",
        "did": ("epic:/RECO/26.07.1/epic_craterlake/Bkg_Exact1S_2us/GoldCt/10um"
                "/DIS/NC/10x100/minQ2=1"),
        "xrootd_prefix": (
            "root://localhost:1294//scratch/gautschi/wxie/eIC_data_small_set/bkg/reco"),
        "min_q2_tiers": [1],
        "files_available": {1: 275},
        "events_per_file": {1: 99},
        "start_n_files": 275,
    },
}

# Local, network-free reference files (symlinked into data/dataset_small/)
# for fast development/debugging before touching the grid.
LOCAL_REFERENCE_FILES: dict[str, str] = {
    "clean": "data/dataset_small/signal/RECO/"
    "pythia8NCDIS_10x100_minQ2=1_beamEffects_xAngle=-0.025_hiDiv_1.0004.eicrecon.edm4eic.root",
    "bkg_mixed": "data/dataset_small/signal_BKG_mix/RECO/"
    "pythia8NCDIS_10x100_minQ2=1_beamEffects_xAngle=-0.025_hiDiv_1.2631.eicrecon.edm4eic.root",
}
