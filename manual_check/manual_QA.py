"""manual_QA.py -- hand-check tracking performance on ONE file, in plain code.

Usage:
    python3 manual_QA.py <file.root> [outdir]

What it does (the same physics as trkperf, with all packaging stripped):
  1. reads truth particles, truth-hit links, reco tracks and associations,
  2. splits truth particles into 3 detector regions by truth eta,
  3. applies each region's own acceptance rule
         barrel          (|eta| < 1.5) : >= 4 of the 7 central collections
         backward endcap (eta < -1.5)  : >= 2 of {BackwardMPGD, TrackerEndcap, TOF}
         forward endcap  (eta > +1.5)  : >= 2 of {ForwardMPGD,  TrackerEndcap, TOF}
  4. matches truth <-> reco at association weight >= 0.5 (best match wins),
  5. draws one plot per region: acceptance & efficiency vs pT (the
     resolution core width + tail fraction are in the figure title); prints per-region totals and
     a file-wide fake rate (no plot - a fake has no truth species/region).

Conventions deliberately identical to trkperf: primaries only
(generatorStatus == 1), 8 species, (lo, hi] bins over pT (0.1, 20] and
eta (-4, 4], region by eta-bin centre, match at weight >= 0.5 with the
single best association winning. Deliberately simpler: 7 wide pT bins,
a robust MAD core width instead of the ROOT Gaussian-core fit, per-bin points hidden
below 5 entries (trkperf uses 50), with the bin's n= printed in grey.

Only dependency besides the standard scientific stack: uproot.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed; write PNGs straight to disk
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import uproot

# --------------------------------------------------------------------------
# Binning and cuts (kept tiny on purpose: this is a QA macro, not a paper)
# --------------------------------------------------------------------------
# pT bins cover the WHOLE analysis phase space (0.1-20 GeV, same as trkperf),
# so every plotted particle lands in the bin it belongs to - nothing is
# folded into an edge bin.
PT_EDGES = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])  # GeV
PT_CENTERS = np.sqrt(PT_EDGES[:-1] * PT_EDGES[1:])               # log centres
MATCH_WEIGHT = 0.5  # a match is genuine if >= half the track hits agree
# Phase space shared with trkperf: pT in (0.1, 20], eta in (-4, 4]; rows
# outside are dropped, not clipped (trkperf/binning.py::assign_bins does
# the same via pd.cut -> NaN).
PT_MIN, PT_MAX = PT_EDGES[0], PT_EDGES[-1]

# Region of a particle = region of its 0.5-wide eta BIN CENTRE (|centre| <
# BARREL_ETA_MAX is barrel), identical to trkperf.report.eta_region.
# BARREL_ETA_MAX mirrors trkperf.config.BARREL_ETA_MAX (kept as a literal so
# this macro stays standalone - change both together). It sits on a bin
# edge, so no bin straddles the split.
BARREL_ETA_MAX = 1.5
# Bin edges are (lo, hi], so a particle exactly at eta = -1.5 is backward and
# exactly at +1.5 is barrel: every particle has exactly one region (no gap,
# no double count) and boundaries cannot drift from the production
# convention.
ETA_EDGES = np.arange(-4.0, 4.01, 0.5)


def eta_bin_centre(eta):
    """Centre of the (lo, hi] eta bin; NaN outside [-4, 4]."""
    # searchsorted(side='left')-1 reproduces pd.cut's (lo, hi] exactly,
    # including x == an edge value; digitize(right=True) misplaces those.
    i = np.searchsorted(ETA_EDGES, eta, side='left') - 1
    centre = 0.5 * (ETA_EDGES[:-1] + ETA_EDGES[1:])
    ok = (i >= 0) & (i < len(centre))
    return np.where(ok, centre[np.clip(i, 0, len(centre) - 1)], np.nan)


# region -> (eta selector, hit collections, min hits)
REGIONS = {
    "barrel": (
        lambda eta: np.abs(eta_bin_centre(eta)) < BARREL_ETA_MAX,
        ["SiBarrelHits", "VertexBarrelHits", "TrackerEndcapHits",
         "MPGDBarrelHits", "OuterMPGDBarrelHits",
         "BackwardMPGDEndcapHits", "ForwardMPGDEndcapHits"],
        4,
    ),
    "backward endcap": (
        lambda eta: eta_bin_centre(eta) < -BARREL_ETA_MAX,
        ["BackwardMPGDEndcapHits", "TrackerEndcapHits", "TOFEndcapHits"],
        2,
    ),
    "forward endcap": (
        lambda eta: eta_bin_centre(eta) > BARREL_ETA_MAX,
        ["ForwardMPGDEndcapHits", "TrackerEndcapHits", "TOFEndcapHits"],
        2,
    ),
}

# truth species we care about (PDG -> name); everything else is ignored
WANT_PDG = {11: "e-", -11: "e+", 211: "pi+", -211: "pi-",
            321: "K+", -321: "K-", 2212: "proton", -2212: "antiproton"}


def main(path, outdir):
    tree = uproot.open(path)["events"]

    # --- 1. truth particles ------------------------------------------------
    # generatorStatus == 1  <=>  primary particle from the generator
    mc = tree.arrays(["MCParticles/MCParticles.PDG",
                      "MCParticles/MCParticles.generatorStatus",
                      "MCParticles/MCParticles.momentum.x",
                      "MCParticles/MCParticles.momentum.y",
                      "MCParticles/MCParticles.momentum.z"], library="np")
    n_ev = len(mc["MCParticles/MCParticles.PDG"])
    # flatten to one row per particle, remembering (event, index) for joins
    rows = []  # (event, idx, pdg, pt, eta)
    for ev in range(n_ev):
        pdg = mc["MCParticles/MCParticles.PDG"][ev]
        gen = mc["MCParticles/MCParticles.generatorStatus"][ev]
        px = mc["MCParticles/MCParticles.momentum.x"][ev]
        py = mc["MCParticles/MCParticles.momentum.y"][ev]
        pz = mc["MCParticles/MCParticles.momentum.z"][ev]
        for i, (p, g) in enumerate(zip(pdg, gen)):
            if g != 1 or p not in WANT_PDG:
                continue
            pt = float(np.hypot(px[i], py[i]))
            ptot = float(np.sqrt(px[i] ** 2 + py[i] ** 2 + pz[i] ** 2))
            if ptot <= 0:
                continue
            theta = np.arccos(np.clip(pz[i] / ptot, -1.0, 1.0))
            eta = -np.log(np.tan(theta / 2.0))
            rows.append((ev, i, p, pt, float(eta)))
    truth = np.array(rows, dtype=[("event", int), ("idx", int), ("pdg", int),
                                 ("pt", float), ("eta", float)])
    print(f"primaries of interest: {len(truth)} in {n_ev} events")

    # --- 2. truth-hit layer counts -----------------------------------------
    # _<Collection>_particle.index lists, per hit, which MCParticles entry
    # made it.  Counting distinct collections per particle = "layers hit".
    hit_by_particle = {}  # (event, idx) -> set(collections)
    missing = set()  # collections absent/unreadable in this file
    for coll in {c for _, cols, _ in REGIONS.values() for c in cols}:
        branch = f"_{coll}_particle/_{coll}_particle.index"
        try:
            data = tree.arrays(branch, library="ak")
            # uproot hands back one record per event keyed by branch name:
            # take the field to get the plain jagged int array
            arr = data[branch] if branch in data.fields else data
        except Exception as exc:  # campaign without this collection: 0 hits
            print(f"  NOTE {coll} unreadable ({type(exc).__name__}); counts 0")
            missing.add(coll)
            continue
        # jagged per-event index lists (plain ints, or {index, ...}
        # records on some productions - take ["index"] then)
        for ev, items in enumerate(ak.to_list(arr)):
            for it in items or []:
                j = it["index"] if isinstance(it, dict) else it
                hit_by_particle.setdefault((ev, int(j)), set()).add(coll)

    # --- 3. reco tracks ----------------------------------------------------
    # NOTE: CentralCKFTracks.momentum is all zeros in these files; the real
    # momentum lives in CentralCKFTrackParameters (qOverP, theta, phi).
    par = tree.arrays(["CentralCKFTrackParameters/CentralCKFTrackParameters.qOverP",
                       "CentralCKFTrackParameters/CentralCKFTrackParameters.theta",
                       "CentralCKFTrackParameters/CentralCKFTrackParameters.phi"],
                      library="np")
    reco_pt = {}  # (event, track idx) -> (pt, eta); finite values only
    for ev in range(n_ev):
        q = np.asarray(par["CentralCKFTrackParameters/CentralCKFTrackParameters.qOverP"][ev],
                       dtype=float)
        th = np.asarray(par["CentralCKFTrackParameters/CentralCKFTrackParameters.theta"][ev],
                        dtype=float)
        p = np.where(np.abs(q) > 0, 1.0 / np.abs(q), np.nan)
        pt = p * np.sin(th)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = -np.log(np.tan(th / 2.0))
        for i, (v, e) in enumerate(zip(pt, eta)):
            if np.isfinite(v) and np.isfinite(e):
                reco_pt[(ev, i)] = (float(v), float(e))

    # --- 4. associations ---------------------------------------------------
    # best match per truth particle AND per reco track (highest weight wins;
    # both directions are needed: efficiency matches truth->reco, the fake
    # rate matches reco->truth - exactly like trkperf/matching.py)
    best = {}  # (event, sim_idx) -> (weight, rec_idx), highest weight wins
    best_track = {}  # (event, rec_idx) -> weight, highest weight wins
    try:
        a = tree.arrays(["CentralCKFTrackAssociations/CentralCKFTrackAssociations.weight",
                         "_CentralCKFTrackAssociations_rec/_CentralCKFTrackAssociations_rec.index",
                         "_CentralCKFTrackAssociations_sim/_CentralCKFTrackAssociations_sim.index"],
                        library="np")
        for ev in range(n_ev):
            w = a["CentralCKFTrackAssociations/CentralCKFTrackAssociations.weight"][ev]
            r = a["_CentralCKFTrackAssociations_rec/_CentralCKFTrackAssociations_rec.index"][ev]
            s = a["_CentralCKFTrackAssociations_sim/_CentralCKFTrackAssociations_sim.index"][ev]
            for ww, rr, ss in zip(w, r, s):
                key = (ev, int(ss))
                if key not in best or ww > best[key][0]:
                    best[key] = (float(ww), int(rr))
                tkey = (ev, int(rr))
                if tkey not in best_track or ww > best_track[tkey]:
                    best_track[tkey] = float(ww)
    except Exception as exc:
        # Not recoverable: without associations every truth particle looks
        # unmatched (efficiency 0) and every track fake (fake rate 100%).
        raise SystemExit(
            "manual_QA: cannot read CentralCKFTrackAssociations "
            f"({type(exc).__name__}: {exc}). Efficiency/fake rate would be "
            "meaningless - check the file/campaign.") from exc

    # --- 5. per-region numbers ----------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    for region, (in_region, collections, n_min) in REGIONS.items():
        sel = in_region(truth["eta"])
        t = truth[sel]
        # phase-space cut shared with trkperf: drop out-of-range rows
        # (lo, hi] like pd.cut: pT == PT_MIN is OUTSIDE (trkperf drops it)
        t = t[(t["pt"] > PT_MIN) & (t["pt"] <= PT_MAX)
              & (t["eta"] > ETA_EDGES[0]) & (t["eta"] <= ETA_EDGES[-1])]
        # acceptance: >= n_min distinct collections hit (restricted to
        # this region's own collection list for an exact rule count)
        n_hit = np.array([len(hit_by_particle.get((e, i), set()) & set(collections))
                          for e, i in zip(t["event"], t["idx"])])
        in_acc = n_hit >= n_min
        # matching: best association weight >= threshold (reco kinematics
        # only needed for the residual, not for the match decision itself)
        matched = np.zeros(len(t), dtype=bool)
        delta = np.full(len(t), np.nan)  # (reco - truth)/truth pT
        for k, (e, i, p, pt, _) in enumerate(zip(
                t["event"], t["idx"], t["pdg"], t["pt"], t["eta"])):
            w_r = best.get((int(e), int(i)))
            if w_r is not None and w_r[0] >= MATCH_WEIGHT:
                matched[k] = True
                info = reco_pt.get((int(e), w_r[1]))
                if info is not None and pt > 0:
                    delta[k] = (info[0] - pt) / pt
        # bin in pT
        # (lo, hi] bins exactly like pd.cut (see eta_bin_centre). Rows are
        # already inside (PT_MIN, PT_MAX] by the cut above, so every row has
        # a valid bin 0..6; the assert guards that invariant loudly instead
        # of letting an out-of-range row wrap to bin -1 (searchsorted gives
        # index 0, minus 1) and silently join the last bin.
        b = np.searchsorted(PT_EDGES, t["pt"], side='left') - 1
        assert b.min(initial=0) >= 0 and b.max(initial=0) < len(PT_CENTERS), \
            "pT binning invariant broken: rows outside (PT_MIN, PT_MAX] reached the binner"
        acc, eff = np.full_like(PT_CENTERS, np.nan), np.full_like(PT_CENTERS, np.nan)
        n_bin = np.array([(b == j).sum() for j in range(len(PT_CENTERS))])
        for j in range(len(PT_CENTERS)):
            m = b == j
            if m.sum() >= 5:  # tiny-statistics guard for a 1-file QA
                acc[j] = in_acc[m].mean()
                in_bin = in_acc[m]
                eff[j] = ((matched[m] & in_bin).sum() / in_bin.sum()
                          if in_bin.sum() else np.nan)
        # Resolution = robust CORE width (1.4826 x median-absolute-deviation
        # = sigma for a Gaussian), not the plain RMS: residuals have heavy
        # tails (mis-reconstruction, brems), and a plain RMS of the forward
        # region is ~10x the core width. tail_frac = share beyond 3 sigma.
        # (Named `resid`, not `d`: the loop above already uses that name for
        # the per-bin in-acceptance mask.)
        resid = delta[matched & np.isfinite(delta)]
        if len(resid) >= 5:
            med = np.median(resid)
            res_core = float(1.4826 * np.median(np.abs(resid - med)))
            tail_frac = float(np.mean(np.abs(resid - med) > 3 * res_core)) if res_core > 0 else float("nan")
        else:
            res_core = tail_frac = float("nan")
        # Console summary uses GLOBAL fractions (sums over all bins), not the
        # mean of per-bin points: bins hold wildly different statistics, so a
        # mean-of-bins would mis-weight them. The figure keeps the per-bin view.
        acc_all = float(in_acc.mean()) if len(t) else float("nan")
        eff_all = (float((matched & in_acc).sum() / in_acc.sum())
                   if in_acc.sum() else float("nan"))

        # --- plot: one figure per region ------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(PT_CENTERS, acc, "o-", label="acceptance")
        ax.plot(PT_CENTERS, eff, "s-", label="efficiency (in acceptance)")
        ax.set_xscale("log")
        # full analysis range, so hidden (low-statistics) bins stay on-axis
        ax.set_xlim(PT_MIN, PT_MAX)
        ax.set_xlabel("truth pT [GeV]")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        # Say WHY a pT bin has no point: fewer than 5 particles (hidden on
        # purpose), shown as a grey tick with its count along the bottom.
        for j, nj in enumerate(n_bin):
            if nj < 5:
                ax.annotate(f"n={nj}", (PT_CENTERS[j], -0.03), ha="center",
                            fontsize=7, color="0.45")
        ax.legend(fontsize=8, loc="center right")
        ax.set_title(f"{region}: acceptance & efficiency vs pT\n"
                     f"n={len(t)} primaries, "
                     f"core resolution (MAD)={res_core:.3f}, "
                     f"{100 * tail_frac:.0f}% beyond 3 sigma",
                     fontsize=9)
        fig.savefig(os.path.join(outdir, f"qa_{region.replace(' ', '_')}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        lost = sorted(set(collections) & missing)
        if lost:
            print(f"  WARNING {region}: acceptance rule degraded, missing {lost} "
                  "(counted as 0 hits)")
        print(f"{region:15s} n={len(t):5d} acc={acc_all:.3f} "
              f"eff={eff_all:.3f} core_res={res_core:.3f} tail={tail_frac:.3f} "
              f"matched={int(matched.sum())}")

    # --- file-wide fake rate (no truth species by construction) --------------
    # a track is fake if its OWN best association is missing/below threshold.
    # Same phase-space rule as trkperf: only binnable tracks count (finite
    # pT in range); unbinnable ones are reported separately, not dropped
    # silently into the denominator.
    binnable = {k: v for k, v in reco_pt.items()
                if PT_MIN < v[0] <= PT_MAX
                and ETA_EDGES[0] < v[1] <= ETA_EDGES[-1]}
    n_tracks = len(binnable)
    n_fake = sum(1 for key in binnable
                 if best_track.get(key, 0.0) < MATCH_WEIGHT)
    n_unbinnable = len(reco_pt) - n_tracks
    print(f"fake rate: {n_fake}/{n_tracks} = "
          f"{(n_fake / n_tracks if n_tracks else float('nan')):.4f} "
          f"({n_unbinnable} unbinnable tracks excluded)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 manual_QA.py <file.root> [outdir]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "qa_out")
