"""Cross-check manual_QA.py vs trkperf on N files - INTEGER-exact.

manual_QA.py prints raw integer lines ("QA-INTS region=... n_gen=...
n_in_acc=... n_match=... n_match_in_acc=..." and
"QA-INTS fake n_tracks=... n_fake=... n_cut_removed=..."), so this tool
compares counts, never parsed floats.

Method per dataset:
- trkperf side: fake-rate (one table, auto per-track regions) and
  efficiency central + backward + forward region runs, aggregated over the
  region's eta bins; all with the shipped defaults (strict >0.5 matching,
  reco 4/3 cut).
- manual_QA side: run per file, sum the QA-INTS lines.

Usage: PYTHONPATH=<repo root> python3 manual_check/check_qa_parity.py \
           <n_files> <clean|bkg>
"""
import re
import subprocess
import sys

import pandas as pd

from trkperf import report

QA_INTS_REGION = re.compile(
    r"QA-INTS region=(\S+) n_gen=(\d+) n_in_acc=(\d+) n_match=(\d+) "
    r"n_match_in_acc=(\d+)")
QA_INTS_FAKE = re.compile(
    r"QA-INTS fake n_tracks=(\d+) n_fake=(\d+) n_cut_removed=(\d+)")

REGIONS = [("barrel", "central", "efficiency"),
           ("backward_endcap", "backward", "efficiency_backward"),
           ("forward_endcap", "forward", "efficiency_forward")]


def qa_ints(files):
    """Sum manual_QA integer lines over files."""
    acc = {}
    for f in files:
        res = subprocess.run([sys.executable, "manual_check/manual_QA.py", f,
                              "/tmp/qa_parity", "0"],
                             capture_output=True, text=True, check=True)
        for m in QA_INTS_REGION.finditer(res.stdout):
            d = acc.setdefault(("region", m.group(1)), dict.fromkeys(
                ["n_gen", "n_in_acc", "n_match", "n_match_in_acc"], 0))
            for k in d:
                d[k] += int(m.group({"n_gen": 2, "n_in_acc": 3, "n_match": 4,
                                     "n_match_in_acc": 5}[k]))
        for m in QA_INTS_FAKE.finditer(res.stdout):
            d = acc.setdefault(("fake",), dict.fromkeys(
                ["n_tracks", "n_fake", "n_cut_removed"], 0))
            for k in d:
                d[k] += int(m.group({"n_tracks": 1, "n_fake": 2,
                                     "n_cut_removed": 3}[k]))
    return acc


def trkperf_totals(files, tag):
    """Same numbers from trkperf's own pipeline (defaults = strict + 4/3)."""
    from trkperf.metrics import efficiency as EFF
    from trkperf.metrics import fake_rate as FR
    out = {}
    lst = list(files)
    fk = FR.compute_fake_rate(lst, max_failures=20)
    fk["R"] = fk["eta_bin_center"].map(report.eta_region)
    for name, r in [("barrel", "barrel"), ("backward_endcap", "backward endcap"),
                    ("forward_endcap", "forward endcap")]:
        s = fk[fk["R"] == r]
        out[("fake", name)] = {"n_tracks": int(s.n_tracks.sum()),
                               "n_fake": int(s.n_fake.sum())}
    for qa_name, rule, fname in REGIONS:
        eff = EFF.compute_efficiency(
            lst, max_failures=20,
            region=rule if rule != "central" else "central")
        eff["R"] = eff["eta_bin_center"].map(report.eta_region)
        s = eff[eff["R"] == ("barrel" if qa_name == "barrel" else
                             ("backward endcap" if qa_name == "backward_endcap"
                              else "forward endcap"))]
        out[("region", qa_name)] = {
            "n_gen": int(s.n_truth.sum()),
            "n_in_acc": int(s.n_in_acceptance.sum()),
            "n_match": int(s.n_matched.sum()),
            "n_match_in_acc": int(s.n_matched_in_acceptance.sum())}
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    kind = sys.argv[2] if len(sys.argv) > 2 else "clean"
    from trkperf import io
    io.set_cache_dir(f"cache/{kind}26071_files")
    files = [l.strip() for l in open(f"filelists/{kind}26071_local.txt")
             if l.strip()][:n]
    q = qa_ints(files)
    t = trkperf_totals(files, kind)
    bad = 0
    print(f"{'metric':<44}{'trkperf':>10}{'manual_QA':>10}")
    for key in sorted(set(q) & set(t)):
        qq, tt = q[key], t[key]
        for col in tt:
            a, b = tt[col], qq.get(col, None)
            mark = "OK " if a == b else "DIFF"
            if a != b and col != "n_cut_removed":   # manual-only diagnostic
                bad += 1
            print(f"{mark} {'/'.join(key):<38}{a:>12}{str(b):>12}")
    missing = set(q) ^ set(t)
    if missing:
        print("unmatched keys:", missing)
        bad += 1
    print("\nRESULT:", "INTEGER-EXACT PARITY" if bad == 0 else f"{bad} MISMATCHES")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
