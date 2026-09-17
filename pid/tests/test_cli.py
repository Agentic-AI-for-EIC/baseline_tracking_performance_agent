"""CLI plumbing: `pid all` must forward every flag `cmd_train` reads.

Regression: `cmd_all` hand-builds an argparse.Namespace that once omitted
`include_event_level`/`include_track_time`/`relax_gates` (AttributeError) and
hardcoded `max_file_failures=0` / `no_calibrate=True`, so `python -m pid all`
either crashed or silently ignored its own flags.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from pid.cli import build_parser, cmd_all

#: Every attribute `cmd_train` reads off its args (pid/cli.py::cmd_train).
#: Keep in sync when cmd_train gains a flag - cmd_all hand-builds this
#: Namespace, and a missing attribute is an AttributeError at runtime.
TRAIN_ARGS = (
    "task", "model", "features", "file", "file_list", "dataset_tag", "legs",
    "limit_files", "max_file_failures", "cache_dir", "out_dir", "model_dir",
    "n_iter", "n_splits", "seed", "n_jobs", "include_event_level",
    "enable_ionisation", "no_calibrate", "relax_gates", "class_weight",
    "use_charge", "allow_small_sample", "include_kinematics",
    "include_track_time", "min_q2_tier",
)


class TestAllForwardsToTrain(unittest.TestCase):
    def test_all_namespace_satisfies_cmd_train(self):
        captured = {}
        args = build_parser().parse_args([
            "all", "--dataset-tag", "clean", "--tasks", "eid", "--models", "lightgbm",
            "--max-file-failures", "7", "--include-event-level",
            "--include-track-time", "--no-calibrate", "--relax-gates",
        ])
        with mock.patch("pid.features.build_features",
                         return_value=pd.DataFrame({"a": [1.0]})), \
             mock.patch("pid.dataset.save_table"), \
             mock.patch("pid.cli.cmd_train",
                        side_effect=lambda ns: captured.setdefault("train", ns)), \
             mock.patch("pid.cli.cmd_evaluate"), \
             mock.patch("pid.cli.cmd_importance"):
            self.assertEqual(cmd_all(args), 0)
        ns = captured["train"]
        for attr in TRAIN_ARGS:
            self.assertTrue(hasattr(ns, attr), f"cmd_all forgot {attr!r}")
        # ... and the values are the parser's, not hardcoded constants.
        self.assertEqual(ns.max_file_failures, 7)
        self.assertTrue(ns.include_event_level)
        self.assertTrue(ns.include_track_time)
        self.assertTrue(ns.no_calibrate)
        self.assertTrue(ns.relax_gates)
        self.assertIsNone(ns.min_q2_tier)  # local reference file has no tier


if __name__ == "__main__":
    unittest.main()
