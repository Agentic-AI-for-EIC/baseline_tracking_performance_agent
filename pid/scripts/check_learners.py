#!/usr/bin/env python
"""Cross-learner agreement check (the M4 gate), run at the end of run_pid.sh.

For each task, compare the held-out test AUC across the trained model
libraries. A spread wider than ``CROSS_LEARNER_MAX_AUC_SPREAD`` means the
headline number depends on which learner you ask.

Advisory by design (exit 0 always): on a hard task the spread can be
statistics rather than a bug, and the artifacts are all valid either way -
but the warning is loud so nobody quotes a learner-dependent number as if
it were a detector measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    sys.path.insert(0, os.getcwd())
    from pid import config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-tag", required=True)
    ap.add_argument("--tasks", required=True, help="Comma list, e.g. eid,ehad,hadpid,pooled.")
    ap.add_argument("--model-dir", default=config.MODEL_DIR)
    args = ap.parse_args()

    limit = config.CROSS_LEARNER_MAX_AUC_SPREAD
    print(f"[check-learners] tag={args.dataset_tag} "
          f"agreement limit (max-min test AUC): {limit}")
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        aucs: dict[str, float] = {}
        for lib in config.MODEL_LIBRARIES:
            rep = os.path.join(args.model_dir, f"{task}_{lib}_{args.dataset_tag}", "report.json")
            if not os.path.exists(rep):
                print(f"[check-learners] {task}: {lib} has no report.json (not trained?) - skipped")
                continue
            with open(rep) as fh:
                aucs[lib] = float(json.load(fh)["auc_test"])
        if len(aucs) < 2:
            print(f"[check-learners] {task}: only {len(aucs)} librar(y/ies) trained - "
                  "no agreement to check")
            continue
        spread = max(aucs.values()) - min(aucs.values())
        detail = ", ".join(f"{lib}={v:.4f}" for lib, v in sorted(aucs.items()))
        if spread <= limit:
            print(f"[check-learners] PASS {task}: {detail} (spread {spread:.4f})")
        else:
            print(f"[check-learners] WARNING {task}: {detail} (spread {spread:.4f} > "
                  f"{limit}) - the headline AUC depends on the learner; do not quote "
                  "one library's number as the task result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
