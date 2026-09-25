"""Detector-region acceptance for PID plots (pid.regions).

Unit tests use mocked truth-hit counts (no files); the end-to-end test runs
the real reader chain on the local reference file (network-free).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from pid import dataset, regions
from trkperf import config as tk_config

_CLEAN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "data/dataset_small/signal/RECO")


def _first(path_dir):
    if not os.path.isdir(path_dir):
        return None
    for name in sorted(os.listdir(path_dir)):
        if name.endswith(".root"):
            return os.path.join(path_dir, name)
    return None


CLEAN_FILE = _first(_CLEAN)


def _frame(rows):
    """Minimal score-frame-shaped table."""
    return pd.DataFrame(rows)


def _features(rows):
    return pd.DataFrame(rows)


def _counts(n, files=("f0",)):
    out = []
    for fi, _ in enumerate(files):
        for ev in range(2):
            for idx in range(4):
                out.append({"file_id": fi, "event": ev, "idx": idx,
                            "n_layers_hit": n})
    return pd.DataFrame(out)


class TestRegionMaps(unittest.TestCase):
    def test_slugs_cover_all_regions(self):
        self.assertEqual(set(regions.REGION_SLUGS), set(regions.ETA_REGIONS))

    def test_rules_match_the_tracking_convention(self):
        self.assertEqual(regions.REGION_RULE["barrel"], "central")
        self.assertEqual(regions.REGION_RULE["forward endcap"], "forward")
        self.assertEqual(regions.REGION_RULE["backward endcap"], "backward")
        central = tk_config.TRACKING_REGIONS["central"]
        self.assertEqual(central["min_layers"], 4)
        for end in ("forward", "backward"):
            self.assertEqual(tk_config.TRACKING_REGIONS[end]["min_layers"], 2)

    def test_unknown_region_refused(self):
        with self.assertRaises(ValueError):
            regions.attach(_frame([]), features=_features([]),
                           regions=("middle",))

    def test_resolve_files_follows_file_id_order(self):
        feats = _features([{"file_id": 1, "source_file": "b"},
                           {"file_id": 0, "source_file": "a"}])
        self.assertEqual(regions.resolve_files(feats), ["a", "b"])


class TestAttach(unittest.TestCase):
    def _run(self, frame_rows, feat_rows, counts_n, regions_wanted,
             files=("f0",)):
        from pid import regions as R

        feats = _features(feat_rows)
        with mock.patch("trkperf.truth.read_truth_hit_layer_counts",
                        side_effect=lambda files_, **kw: _counts(counts_n, files_)):
            with mock.patch("trkperf.io.set_cache_dir"):
                return R.attach(_frame(frame_rows), features=feats,
                                regions=regions_wanted)

    def _link(self, truth_idx=0, truth_eta=0.25, file_id=0, event=0, track_idx=0):
        return {"file_id": file_id, "event": event, "track_idx": track_idx,
                "source_file": "f0", "truth_idx": truth_idx}

    def test_barrel_needs_four_endcap_needs_two(self):
        # Same hit count (3) passes the endcap rule but fails the barrel one.
        frame = _frame([
            {"file_id": 0, "event": 0, "track_idx": 0, "truth_eta": 0.25},
            {"file_id": 0, "event": 0, "track_idx": 1, "truth_eta": -2.5},
        ])
        feats = _features([
            {**self._link(track_idx=0, truth_eta=0.25), "truth_idx": 0},
            {**self._link(track_idx=1, truth_eta=-2.5), "truth_idx": 1},
        ])
        from pid import regions as R
        with mock.patch("trkperf.truth.read_truth_hit_layer_counts",
                        side_effect=lambda files_, **kw: _counts(3, files_)):
            with mock.patch("trkperf.io.set_cache_dir"):
                out, info = R.attach(frame, features=feats,
                                     regions=("barrel", "backward endcap"))
        barrel = out[out["track_idx"] == 0].iloc[0]
        back = out[out["track_idx"] == 1].iloc[0]
        self.assertEqual(barrel["eta_region"], "barrel")
        self.assertEqual(back["eta_region"], "backward endcap")
        self.assertEqual(barrel["n_layers_hit"], 3)
        self.assertFalse(barrel["in_acceptance"])   # 3 < 4
        self.assertTrue(back["in_acceptance"])      # 3 >= 2
        self.assertEqual(info["regions"]["barrel"]["n_in_acceptance"], 0)
        self.assertEqual(info["regions"]["backward endcap"]["n_in_acceptance"], 1)

    def test_unmatched_truth_is_unclassifiable(self):
        frame = _frame([{"file_id": 0, "event": 0, "track_idx": 0,
                         "truth_eta": np.nan}])
        feats = _features([{**self._link(truth_eta=np.nan), "truth_idx": np.nan}])
        out, info = self._run(frame.to_dict("records"), feats.to_dict("records"),
                              7, ("barrel",))
        self.assertEqual(out.iloc[0]["eta_region"], "unknown")
        self.assertFalse(out.iloc[0]["in_acceptance"])
        self.assertEqual(info["n_unclassified"], 0)  # unknown region, not requested

    def test_zero_hit_particle_is_zero_not_unclassifiable(self):
        # A matched particle absent from the count table has 0 hits in the
        # rule's collections (trkperf.truth contract): n_layers_hit must be 0
        # (fails acceptance honestly), not NaN "unclassifiable".
        frame = _frame([
            {"file_id": 0, "event": 0, "track_idx": 0, "truth_eta": -2.5},
            {"file_id": 0, "event": 0, "track_idx": 1, "truth_eta": -2.5},
        ])
        feats = _features([
            {**self._link(track_idx=0), "truth_idx": 0},
            {**self._link(track_idx=1), "truth_idx": 1},
        ])
        counts = pd.DataFrame([{"file_id": 0, "event": 0, "idx": 0,
                                "n_layers_hit": 3}])
        from pid import regions as R
        with mock.patch("trkperf.truth.read_truth_hit_layer_counts",
                        return_value=counts):
            with mock.patch("trkperf.io.set_cache_dir"):
                out, info = R.attach(frame, features=feats,
                                     regions=("backward endcap",))
        zero = out[out["track_idx"] == 1].iloc[0]
        self.assertEqual(zero["n_layers_hit"], 0)
        self.assertFalse(zero["in_acceptance"])
        self.assertEqual(info["n_unclassified"], 0)

    def test_mismatched_features_refuse(self):
        frame = _frame([{"file_id": 0, "event": 0, "track_idx": 9,
                         "truth_eta": 0.25}])
        feats = _features([self._link(track_idx=0)])
        with self.assertRaises(SystemExit):
            self._run(frame.to_dict("records"), feats.to_dict("records"),
                      7, ("barrel",))

    def test_region_columns_never_train(self):
        self.assertIn("eta_region", dataset.KEY_COLUMNS)
        self.assertIn("in_acceptance", dataset.KEY_COLUMNS)
        self.assertIn("n_layers_hit", dataset.KEY_COLUMNS)


class TestRegionPackage(unittest.TestCase):
    def test_region_summary_uses_global_cut(self):
        from pid import performance as perf

        rng = np.random.default_rng(7)
        n = 120
        y = np.r_[np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)]
        score = np.clip(np.r_[rng.normal(0.75, 0.1, n // 2),
                              rng.normal(0.45, 0.1, n // 2)], 0, 1)
        frame = pd.DataFrame({
            "score": score, "y": y,
            "truth_class": ["e"] * (n // 2) + ["pi"] * (n // 2),
            "pt": np.full(n, 2.0), "p": np.full(n, 3.0),
            "eta": np.full(n, -2.0), "file_id": np.zeros(n, dtype=int),
            "eta_region": ["backward endcap"] * n,
            "in_acceptance": [True] * n})
        spec = {"signal": "e", "background": "pi", "task": "eid",
                "title": "e vs pi", "pair": False}
        opt = {"threshold": 0.6, "valid": True, "caution": "",
               "signal": "e", "background": "pi", "score_space": "task",
               "rejects_background": True, "edge_flag": False}
        summary: list = []
        perf._run_region_package(
            channel="eid", basis="reco", suffix="", frame=frame,
            region_info={"regions": {
                "backward endcap": {"rule": "backward", "min_layers": 2,
                                    "collections": ["a", "b", "c"],
                                    "collections_found": ["a", "b", "c"],
                                    "n_rows": n, "n_in_acceptance": n}}},
            eta_regions=("backward endcap",), spec=spec, status="headline",
            status_note="", model="lightgbm", dataset_tag="clean",
            out_dir="/tmp/nowhere", plots_dir="/tmp/nowhere",
            names={"pt": "pt", "p": "p", "eta": "eta"},
            global_cut=0.6, opt=opt, bkg_scale=1.0,
            mode="s_over_sqrt", lumi_scale=1.0, binned_curves=False,
            n_thresholds=None, nsigma_method=None, write=False, quiet=True,
            summary=summary)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["eta_region"], "backward endcap")
        self.assertEqual(row["threshold"], 0.6)
        self.assertTrue(row["valid"])
        self.assertIn("Nhit >= 2", row["acceptance_rule"])
        self.assertEqual(row["n_rows"], n)


class TestAttachLocalFile(unittest.TestCase):
    def test_real_hit_reads_classify_local_tracks(self):
        if not CLEAN_FILE:
            raise unittest.SkipTest("data/dataset_small/signal reference file missing")
        from trkperf import matching, reco, truth

        files = [CLEAN_FILE]
        truth_df = truth.read_truth_particles(files)
        reco_df = reco.read_reco_tracks(files)
        assoc_df = matching.read_associations(files)
        tracks = matching.build_matched_tracks(truth_df, reco_df, assoc_df)
        tracks = tracks[tracks["is_matched"]].head(200).copy()
        frame = tracks.rename(columns={"idx": "track_idx"})[
            ["file_id", "event", "track_idx", "truth_eta"]].copy()
        feats = tracks.rename(columns={"idx": "track_idx"})[
            ["file_id", "event", "track_idx", "truth_idx"]].copy()
        feats["source_file"] = CLEAN_FILE
        out, info = regions.attach(frame, features=feats,
                                   regions=("barrel", "backward endcap",
                                            "forward endcap"))
        self.assertTrue(set(out["eta_region"]) <= set(regions.ETA_REGIONS) | {"unknown"})
        # The local clean file carries every central collection: nothing in
        # the barrel may silently degrade.
        central_found = set(info["regions"]["barrel"]["collections_found"])
        self.assertEqual(central_found,
                         set(tk_config.CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS))
        classified = out[out["n_layers_hit"].notna()]
        self.assertGreater(len(classified), 0)
        self.assertTrue(((classified["n_layers_hit"] >= 0)
                         & (classified["n_layers_hit"] <= 7)).all())
