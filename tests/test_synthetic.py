"""Unit tests using synthetic data - no files, no network required.

These pin down the pure-logic pieces (the Gaussian fit, binomial error,
the clean-vs-background comparison join) independently of any real EDM4eic
file, so they run instantly and should never be skipped.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from trkperf import binning, compare, io, report


class TestGaussianFit(unittest.TestCase):
    def test_recovers_known_mu_sigma(self):
        rng = np.random.default_rng(seed=42)
        true_mu, true_sigma = 0.01, 0.05
        values = rng.normal(true_mu, true_sigma, size=20000)
        fit = binning.fit_gaussian_core(values)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.mu, true_mu, delta=0.02)
        self.assertAlmostEqual(fit.sigma, true_sigma, delta=0.02)
        self.assertGreater(fit.chi2_over_ndf, 0)

    def test_reports_not_converged_on_too_few_entries(self):
        fit = binning.fit_gaussian_core(np.array([0.01, 0.02]))
        self.assertFalse(fit.converged)

    def test_rejects_sigma_beyond_physical_maximum(self):
        # Resolution is sigma on (pT_reco - pT_truth)/pT_truth; sigma = 1.0
        # is a 100% resolution, the physical maximum. A distribution whose
        # true width far exceeds that (here sigma = 3.0 -> a residual larger
        # than the momentum itself, unphysical / a diverged fit) must be
        # flagged non-converged, NOT reported as a huge number.
        rng = np.random.default_rng(seed=7)
        values = rng.normal(0.0, 3.0, size=20000)
        fit = binning.fit_gaussian_core(values)
        self.assertFalse(fit.converged)
        self.assertTrue(np.isnan(fit.sigma))

    def test_wide_sigma_is_capped_at_config_max(self):
        # A moderately-but-not-insanely wide residual (sigma ~ 0.4) is within
        # [0, RESOLUTION_MAX_SIGMA], so it fits and the returned sigma
        # respects the upper bound rather than ever exceeding 1.0.
        rng = np.random.default_rng(seed=3)
        values = rng.normal(0.0, 0.4, size=20000)
        fit = binning.fit_gaussian_core(values)
        self.assertLessEqual(fit.sigma, 1.0 + 1e-9)


class TestBinomialError(unittest.TestCase):
    def test_zero_denominator_is_zero_not_nan_or_inf(self):
        err = binning.binomial_error(0, 0)
        self.assertEqual(float(err), 0.0)

    def test_matches_known_formula(self):
        err = binning.binomial_error(50, 100)
        self.assertAlmostEqual(float(err), float(np.sqrt(0.5 * 0.5 / 100)), places=9)


class TestCompareMetric(unittest.TestCase):
    def test_ratio_and_difference_on_synthetic_acceptance(self):
        clean = pd.DataFrame(
            {
                "species": ["pi+", "pi+"],
                "pt_bin_center": [1.0, 2.0],
                "eta_bin_center": [0.0, 0.0],
                "acceptance": [0.8, 0.9],
                "acceptance_err": [0.02, 0.01],
                "insufficient_stats": [False, False],
            }
        )
        bkg = clean.copy()
        bkg["acceptance"] = [0.4, 0.6]
        bkg["acceptance_err"] = [0.03, 0.02]

        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))

        merged = merged.sort_values("pt_bin_center")
        np.testing.assert_allclose(
            merged["acceptance_ratio_bkg_over_clean"].to_numpy(), [0.4 / 0.8, 0.6 / 0.9]
        )
        np.testing.assert_allclose(
            merged["acceptance_diff_bkg_minus_clean"].to_numpy(), [0.4 - 0.8, 0.6 - 0.9]
        )
        self.assertFalse(merged["insufficient_stats"].any())

    def test_zero_clean_value_gives_nan_ratio_not_a_crash(self):
        clean = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.0],
                "fake_rate_err": [0.0],
            }
        )
        bkg = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.1],
                "fake_rate_err": [0.02],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))

        self.assertTrue(np.isnan(merged["fake_rate_ratio_bkg_over_clean"].iloc[0]))
        self.assertAlmostEqual(merged["fake_rate_diff_bkg_minus_clean"].iloc[0], 0.1)

    def test_compare_pid_confusion_joins_on_reco_species(self):
        clean = pd.DataFrame(
            {
                "species": ["pi+", "pi+", "K+", "K+"],
                "reco_species": ["pi+", "K+", "pi+", "K+"],
                "pt_bin_center": [1.0, 1.0, 1.0, 1.0],
                "eta_bin_center": [0.0, 0.0, 0.0, 0.0],
                "confusion_frac": [0.9, 0.1, 0.2, 0.8],
                "confusion_frac_err": [0.03, 0.03, 0.04, 0.04],
                "insufficient_stats": [False, False, False, False],
            }
        )
        bkg = clean.copy()
        bkg["confusion_frac"] = [0.85, 0.15, 0.25, 0.75]

        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))

        merged = merged.sort_values(["species", "reco_species"]).reset_index(drop=True)
        # The outer join may reorder rows; match expected values to the
        # actual (species, reco_species) ordering produced by the merge.
        expected_map = {
            ("K+", "K+"): 0.75 / 0.8,
            ("K+", "pi+"): 0.25 / 0.2,
            ("pi+", "K+"): 0.15 / 0.1,
            ("pi+", "pi+"): 0.85 / 0.9,
        }
        expected = [
            expected_map[(row["species"], row["reco_species"])]
            for _, row in merged.iterrows()
        ]
        np.testing.assert_allclose(
            merged["confusion_frac_ratio_bkg_over_clean"].to_numpy(),
            np.array(expected),
        )


class TestReadFlatMultiFailures(unittest.TestCase):
    """`io.read_flat_multi(max_failures=...)` skips flaky files instead of
    aborting the whole run - this is what makes the +background grid runs
    robust against the overseas xrootd endpoint dropping connections."""

    def test_zero_failures_is_strict(self):
        # By default any file I/O failure is fatal.
        with mock.patch.object(io, "open_tree", side_effect=OSError("boom")):
            with self.assertRaises(RuntimeError):
                io.read_flat_multi(
                    ["root://fake/file.root"], {"x": "T/x"}, max_failures=0
                )

    def test_tolerated_failure_is_skipped_not_fatal(self):
        # A failure within max_failures is skipped; good files still come back.
        good = pd.DataFrame({"file_id": [0], "event": [0], "idx": [0], "x": [1.0]})

        def fake_open(path, tree_name="events"):
            if path == "BAD":
                raise OSError("transient network error")
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with mock.patch.object(io, "open_tree", side_effect=fake_open), \
             mock.patch.object(io, "read_flat", side_effect=fake_read):
            result = io.read_flat_multi(
                ["BAD", "GOOD"], {"x": "T/x"}, max_failures=1
            )

        self.assertEqual(len(result), 1)  # only the good file's row

    def test_exceeding_max_failures_raises(self):
        good = pd.DataFrame({"file_id": [0], "event": [0], "idx": [0], "x": [1.0]})

        def fake_open(path, tree_name="events"):
            raise OSError("transient network error")

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with mock.patch.object(io, "open_tree", side_effect=fake_open), \
             mock.patch.object(io, "read_flat", side_effect=fake_read):
            with self.assertRaises(RuntimeError):
                io.read_flat_multi(
                    ["A", "B", "C"], {"x": "T/x"}, max_failures=1
                )


class TestPerFileFilter(unittest.TestCase):
    """`read_flat_multi(per_file_filter=...)` shrinks the accumulated table
    per file, but the cache stash must always hold the RAW (unfiltered)
    data - otherwise a filtered run would permanently poison the shared
    content-addressed cache for any consumer that needs the full table."""

    def test_filter_is_applied_to_returned_rows(self):
        import trkperf.io as io

        good = pd.DataFrame(
            {"file_id": [0, 0], "event": [0, 0], "idx": [0, 1], "x": [1.0, 2.0]}
        )

        def fake_open(path, tree_name="events"):
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with mock.patch.object(io, "open_tree", side_effect=fake_open), \
             mock.patch.object(io, "read_flat", side_effect=fake_read):
            result = io.read_flat_multi(
                ["A"], {"x": "T/x"}, per_file_filter=lambda df: df[df["x"] > 1.0]
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result["x"].iloc[0], 2.0)

    def test_cache_stashes_raw_not_filtered(self):
        import trkperf.io as io

        good = pd.DataFrame(
            {"file_id": [0, 0], "event": [0, 0], "idx": [0, 1], "x": [1.0, 2.0]}
        )
        calls = {"count": 0}

        def fake_open(path, tree_name="events"):
            calls["count"] += 1
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(io, "open_tree", side_effect=fake_open), \
                 mock.patch.object(io, "read_flat", side_effect=fake_read):
                filtered = io.read_flat_multi(
                    ["A"], {"x": "T/x"}, cache_dir=tmp,
                    per_file_filter=lambda df: df[df["x"] > 1.0],
                )
            self.assertEqual(len(filtered), 1)
            # Second read with NO filter must still see the RAW 2-row table:
            # the stash was written unfiltered.
            with mock.patch.object(io, "open_tree", side_effect=fake_open), \
                 mock.patch.object(io, "read_flat", side_effect=fake_read):
                raw = io.read_flat_multi(["A"], {"x": "T/x"}, cache_dir=tmp)
            self.assertEqual(len(raw), 2, "cache must hold the raw table")

    def test_read_truth_partcles_primary_only(self):
        from trkperf import truth

        table = pd.DataFrame(
            {
                "file_id": [0, 0, 0],
                "event": [0, 0, 0],
                "idx": [0, 1, 2],
                "pdg": [211, 22, 211],
                "generator_status": [1, 3, 1],
                "px": [1.0, 2.0, 3.0],
                "py": [0.0, 1.0, 0.0],
                "pz": [0.0, 0.0, 0.0],
                "mass": [0.139, 0.0, 0.139],
            }
        )
        with mock.patch.object(truth.io, "read_flat_multi", return_value=table) as mocked:
            truth.read_truth_particles(["A"], primary_only=True)
        # primary_only must be realised by passing a per-file filter to io,
        # so the memory win happens inside read_flat_multi's accumulation.
        filt = mocked.call_args.kwargs["per_file_filter"]
        kept = filt(table.copy())
        self.assertEqual(len(kept), 2)  # the two generator_status == 1 rows
        self.assertEqual(sorted(kept["idx"]), [0, 2])
        # The default (primary_only=False) must pass no filter at all.
        with mock.patch.object(truth.io, "read_flat_multi", return_value=table) as mocked2:
            truth.read_truth_particles(["A"])
        self.assertIsNone(mocked2.call_args.kwargs.get("per_file_filter"))


class TestReadCache(unittest.TestCase):
    """The per-file read cache makes a relaunched network run free over
    already-processed files: first read stashes a pickle, a second read of the
    same file (even with a different list position) loads the stash and never
    touches the network."""

    def test_cached_file_is_not_reopened_and_file_id_is_re_stamped(self):
        import trkperf.io as io

        good = pd.DataFrame({"file_id": [0], "event": [0], "idx": [0], "x": [1.0]})
        calls = {"count": 0}

        def fake_open(path, tree_name="events"):
            calls["count"] += 1
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(io, "open_tree", side_effect=fake_open), \
                 mock.patch.object(io, "read_flat", side_effect=fake_read):
                first = io.read_flat_multi(
                    ["A"], {"x": "T/x"}, cache_dir=tmp, max_failures=0
                )
                second = io.read_flat_multi(
                    ["A"], {"x": "T/x"}, cache_dir=tmp, max_failures=0
                )

        self.assertEqual(calls["count"], 1, "second read re-opened the network file")
        self.assertEqual(len(second), 1)
        # file_id is re-stamped to the position in THIS run's list.
        self.assertEqual(int(second["file_id"].iloc[0]), 0)

    def test_cache_is_keyed_on_columns(self):
        import trkperf.io as io

        good = pd.DataFrame({"file_id": [0], "event": [0], "idx": [0], "x": [1.0]})
        calls = {"count": 0}

        def fake_open(path, tree_name="events"):
            calls["count"] += 1
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(io, "open_tree", side_effect=fake_open), \
                 mock.patch.object(io, "read_flat", side_effect=fake_read):
                io.read_flat_multi(["A"], {"x": "T/x"}, cache_dir=tmp)
                # Same file, different branch set -> must not reuse the stash.
                io.read_flat_multi(["A"], {"y": "T/y"}, cache_dir=tmp)

        self.assertEqual(calls["count"], 2, "different branch sets must not share a cache entry")

    def test_process_wide_cache_dir_via_set_cache_dir(self):
        import trkperf.io as io

        good = pd.DataFrame({"file_id": [9], "event": [0], "idx": [0], "x": [1.0]})
        calls = {"count": 0}

        def fake_open(path, tree_name="events"):
            calls["count"] += 1
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        with tempfile.TemporaryDirectory() as tmp:
            io.set_cache_dir(tmp)
            try:
                with mock.patch.object(io, "open_tree", side_effect=fake_open), \
                     mock.patch.object(io, "read_flat", side_effect=fake_read):
                    io.read_flat_multi(["A"], {"x": "T/x"})
                    io.read_flat_multi(["A"], {"x": "T/x"})
            finally:
                io.set_cache_dir(None)

        self.assertEqual(calls["count"], 1, "set_cache_dir default was not honoured")


if __name__ == "__main__":
    unittest.main()
