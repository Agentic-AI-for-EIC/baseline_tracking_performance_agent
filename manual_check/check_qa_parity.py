"""Cross-check manual_QA.py vs trkperf on N files: per-region acceptance,
within-acceptance efficiency, matched, and global fake rate.
Usage: python check_qa_parity.py <n_files> <clean|bkg>"""
import re, sys, subprocess
import numpy as np, pandas as pd
from trkperf import truth, reco, matching, binning, report, config, io

io.set_cache_dir(f"cache/{sys.argv[2] if len(sys.argv)>2 else 'clean'}26071_files")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
kind = sys.argv[2] if len(sys.argv) > 2 else "clean"
files = [l.strip() for l in open(f"filelists/{kind}26071_local.txt") if l.strip()][:N]

REG = {"barrel": "central", "forward endcap": "forward", "backward endcap": "backward"}

def trkperf_one(f):
    sh = {}
    t = truth.select_primary(truth.read_truth_particles([f], primary_only=True, shared_failures=sh))
    t = binning.assign_bins(t)                       # adds eta_bin_center / p_bin_center
    t = t.dropna(subset=["p_bin", "eta_bin"])        # same phase-space cut as manual_QA (0.1,100],(-4,4]
    rc = reco.read_reco_tracks([f], shared_failures=sh)
    a = matching.read_associations([f], shared_failures=sh)
    pairs = matching.build_matched_pairs(t, rc, a, config.MATCH_WEIGHT_THRESHOLD)
    # also expose the manual_QA convention (best-per-particle >= 0.5) to isolate the threshold change
    bs = matching._best_association(a, ["file_id", "event", "sim_idx"])
    matched_ge = bs.assign(is_m=True)[["file_id", "event", "sim_idx", "weight"]]
    matched_ge["is_m"] = matched_ge["weight"] >= config.MATCH_WEIGHT_THRESHOLD
    # per-region acceptance + within-acceptance efficiency (partition particles by eta bin centre)
    out = {}
    keep = t[["file_id", "event", "idx"]]
    for region, rulename in REG.items():
        cols = config.TRACKING_REGIONS[rulename]["collections"]
        cnt = truth.read_truth_hit_counts([f], collections=cols, keep_particles=keep,
                                          shared_failures=sh)
        fl = truth.add_acceptance_flag(t, cnt)
        sub = fl[fl["eta_bin_center"].map(report.eta_region) == region]
        n_gen = len(sub); n_acc = int(sub["in_acceptance"].sum())
        m = sub.merge(pairs[["file_id", "event", "idx", "is_matched"]],
                      on=["file_id", "event", "idx"], how="left")
        n_macc = int((m["in_acceptance"] & m["is_matched"].fillna(False)).sum())
        mg = sub.merge(matched_ge[["file_id", "event", "sim_idx", "is_m"]],
                       left_on=["file_id", "event", "idx"], right_on=["file_id", "event", "sim_idx"],
                       how="left")
        n_macc_ge = int((mg["in_acceptance"] & mg["is_m"].fillna(False)).sum())
        out[region] = dict(n=n_gen, acc=n_acc, eff_macc=n_macc, eff_macc_ge=n_macc_ge)
    # fake rate over binnable tracks
    ff = matching.find_fake_tracks(rc, a, config.MATCH_WEIGHT_THRESHOLD)
    ff = binning.assign_bins(ff).dropna(subset=["p_bin", "eta_bin"])
    out["fake"] = dict(n_tracks=int(len(ff)), n_fake=int(ff["is_fake"].sum()),
                       n_fake_ge=int((ff["assoc_weight"].fillna(0.0) < config.MATCH_WEIGHT_THRESHOLD).sum()))
    return out

def qa_one(f):
    res = subprocess.run([sys.executable, "manual_check/manual_QA.py", f, "/tmp/qa_parity", "0"],
                         capture_output=True, text=True)
    out = {}
    for m in re.finditer(r"(barrel|forward endcap|backward endcap)\s+n=\s*(\d+)\s+acc=([\d.naif]+)\s+eff=([\d.naif]+).*matched=(\d+)", res.stdout):
        reg, n, acc, eff, matched = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
        acc = float(acc) if acc[0].isdigit() else 0.0
        eff = float(eff) if eff[0].isdigit() else 0.0
        n_acc = int(round(acc * n)); n_macc = int(round(eff * n_acc))
        out[reg] = dict(n=n, acc=n_acc, eff_macc=n_macc)
    mf = re.search(r"fake rate:\s*(\d+)/(\d+)", res.stdout)
    out["fake"] = dict(n_fake=int(mf.group(1)), n_tracks=int(mf.group(2))) if mf else None
    return out

def agg(lst, key, sub):
    return sum(d[key][sub] for d in lst if d.get(key))

T = [trkperf_one(f) for f in files]
Q = [qa_one(f) for f in files]

print(f"\n{'metric':<34}{'trkperf':>12}{'manual_QA':>12}{'Δ':>10}")
for reg in REG:
    tn, tan, tme = agg(T, reg, "n"), agg(T, reg, "acc"), agg(T, reg, "eff_macc")
    tmg = agg(T, reg, "eff_macc_ge")
    qn, qan, qme = agg(Q, reg, "n"), agg(Q, reg, "acc"), agg(Q, reg, "eff_macc")
    print(f"{'acceptance '+reg:<34}{tan/max(tn,1):>12.6f}{qan/max(qn,1):>12.6f}{tan/max(tn,1)-qan/max(qn,1):>10.2e}")
    print(f"{'  (n_gen / n_inacc)      '+reg:<34}{f'{tn}/{tan}':>12}{f'{qn}/{qan}':>12}")
    print(f"{'eff_within_acc '+reg:<34}{tme/max(tan,1):>12.6f}{qme/max(qan,1):>12.6f}{tme/max(tan,1)-qme/max(qan,1):>10.2e}")
    print(f"{'  eff n_macc strict/GE vs QA '+reg:<34}{f'{tme}/{tmg}':>12}{f'{qme}':>12}   {'OK' if tmg==qme and tan==qan and tn==qn else 'DIFF'}")
tft, tff, qft, qff = agg(T,"fake","n_tracks"), agg(T,"fake","n_fake"), agg(Q,"fake","n_tracks"), agg(Q,"fake","n_fake")
tfge = agg(T,"fake","n_fake_ge")
print(f"{'fake rate (global)':<34}{tff/max(tft,1):>12.6f}{qff/max(qft,1):>12.6f}{tff/max(tft,1)-qff/max(qft,1):>10.2e}")
print(f"{'  (n_fake / n_tracks)':<34}{f'{tff}/{tft}':>12}{f'{qff}/{qft}':>12}")
print(f"{'  trkperf n_fake at >=0.5 (as manual_QA)':<34}{tfge:>12}   -> matches manual_QA? {tfge==qff}")
