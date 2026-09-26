"""Unit tests for the per-file feature builders (mocked ROOT reads)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from pid import features
from pid import features as pid_features


class TestDrichTrackLink(unittest.TestCase):
    def test_has_flag_follows_length_not_link_row(self):
        # A track with a link row but no radiator segment has no pathlength:
        # has_drich must be 0, not 1.
        tr = pd.DataFrame({"event": [0, 0], "idx": [0, 1], "drich_gas_track": [3, 4]})
        seg = pd.DataFrame({"event": [0], "idx": [0], "drich_gas_length": [2.5]})

        def fake_flat(tree, campaign, keys):
            key = keys[0]
            if key == "drich_gas_track":
                return tr
            if key == "drich_gas_length":
                return seg
            return pd.DataFrame(columns=["event", "idx", key])

        with mock.patch.object(features, "_flat", side_effect=fake_flat):
            out = features.drich_track_link("P", "T", "C")
        by_track = out.set_index("track_idx")
        row = by_track.loc[4]
        self.assertEqual(int(row["has_drich_gas"]), 0)
        self.assertTrue(pd.isna(row["drich_gas_pathlength"]))
        row0 = by_track.loc[3]
        self.assertEqual(int(row0["has_drich_gas"]), 1)
        self.assertAlmostEqual(float(row0["drich_gas_pathlength"]), 2.5)


class TestBuildFeaturesJobs(unittest.TestCase):
    """The parallel (jobs>1) path must return the same ordered, file-stamped
    table as the serial one, keep the empty-file warning semantics and honour
    the shared max_failures budget."""

    def setUp(self):
        import pandas as pd

        self.pd = pd

        def fake_one_file(path, campaign, *, legs, max_failures, enable_ionisation):
            if path.startswith("BAD"):
                raise OSError(f"flaky {path}")
            n = {"A": 3, "B": 2, "C": 0, "D": 1}[path]
            if n == 0:
                return pd.DataFrame()
            return pd.DataFrame({"event": range(n), "x": [float(len(path))] * n})

        patcher = mock.patch.object(pid_features, "build_features_one_file",
                                    side_effect=fake_one_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, files, jobs, tmp, max_failures=0):
        return pid_features.build_features(files, dataset_tag="clean", jobs=jobs,
                                           max_failures=max_failures, cache_dir=tmp,
                                           progress=False)

    def test_parallel_equals_serial_order_and_ids(self):
        files = ["A", "B", "D", "C"]
        with tempfile.TemporaryDirectory() as s, tempfile.TemporaryDirectory() as p:
            serial = self._run(files, 1, s)
            par = self._run(files, 4, p)
        self.pd.testing.assert_frame_equal(serial.reset_index(drop=True),
                                           par.reset_index(drop=True))
        # file_id stamps the file-list position in both paths...
        self.assertEqual(serial["file_id"].tolist(), par["file_id"].tolist())
        self.assertEqual(list(dict.fromkeys(par["source_file"])), ["A", "B", "D"])
        # ...and the per-file caches are interchangeable between the two.
        with tempfile.TemporaryDirectory() as s2:
            again = self._run(files, 1, s)  # second serial pass: all cache hits
            self.pd.testing.assert_frame_equal(serial.reset_index(drop=True),
                                               again.reset_index(drop=True))

    def test_failures_budget_shared_across_workers(self):
        files = ["A", "BAD1", "B", "BAD2"]
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(files, 4, tmp, max_failures=5)
            self.assertEqual(sorted(out.attrs["skipped_files"]), ["BAD1", "BAD2"])
            self.assertEqual(out.attrs["empty_files"], [])
            with self.assertRaises(RuntimeError):
                self._run(files, 4, tmp, max_failures=1)

    def test_empty_files_recorded_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(["A", "C"], 2, tmp)
            self.assertEqual(out.attrs["empty_files"], ["C"])
            self.assertNotIn("C", set(out["source_file"]))
            self.assertNotIn("C", [os.path.basename(p) for p in out["source_file"]])


if __name__ == "__main__":
    unittest.main()
