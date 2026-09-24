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
    def test_zero_denominator_is_nan(self):
        # No data -> no error bar, not a zero-width one.
        err = binning.binomial_error(0, 0)
        self.assertTrue(np.isnan(float(err)))

    def test_matches_wilson_score_interval(self):
        # Wilson 95 % for 50/100: center 0.5, half-width 0.09617.
        err = binning.binomial_error(50, 100)
        self.assertAlmostEqual(float(err), 0.09617, places=4)

    def test_perfect_and_empty_rates_still_have_width(self):
        # The normal approximation returns exactly 0 at k=0 and k=n - the
        # most-quoted bins (efficiency ~1, fake rate ~0) must not report
        # zero-width errors.
        self.assertGreater(float(binning.binomial_error(20, 20)), 0.0)
        between = float(binning.binomial_error(0, 20))
        self.assertGreater(between, 0.0)
        self.assertLess(between, 1.0)


class TestFullBinGrid(unittest.TestCase):
    def test_grid_covers_every_species_bin_combination(self):
        from trkperf import config

        grid = binning.full_bin_grid(["pi+", "K+"])
        n_pt = len(config.PT_BIN_EDGES) - 1
        n_eta = len(config.ETA_BIN_EDGES) - 1
        self.assertEqual(len(grid), 2 * n_pt * n_eta)
        self.assertTrue((grid["pt_bin_center"] > 0).all())

    def test_grid_bins_match_assign_bins_categories(self):
        df = pd.DataFrame({"pt": [0.5, 5.0], "eta": [0.1, -2.0]})
        binned = binning.assign_bins(df)
        grid = binning.full_bin_grid(None)
        self.assertEqual(
            list(grid["pt_bin"].cat.categories),
            list(binned["pt_bin"].cat.categories),
        )
        self.assertEqual(
            list(grid["eta_bin"].cat.categories),
            list(binned["eta_bin"].cat.categories),
        )
        self.assertNotIn("species", grid.columns)


class TestEmptyInputGrids(unittest.TestCase):
    """Metrics over zero rows still return the full schema on the full grid.

    Empty bins must be explicitly reported (NaN values, insufficient_stats),
    never a missing table or a missing column that breaks report/compare.
    """

    def test_acceptance_empty_input_returns_full_flagged_grid(self):
        from trkperf import config
        from trkperf.metrics import acceptance

        empty_truth = pd.DataFrame({
            "file_id": pd.Series(dtype="int64"), "event": pd.Series(dtype="int64"),
            "idx": pd.Series(dtype="int64"), "generator_status": pd.Series(dtype="int64"),
            "species": pd.Series(dtype="object"), "pt": pd.Series(dtype="float64"),
            "eta": pd.Series(dtype="float64"),
        })
        empty_layers = pd.DataFrame({
            "file_id": pd.Series(dtype="int64"), "event": pd.Series(dtype="int64"),
            "idx": pd.Series(dtype="int64"), "n_layers_hit": pd.Series(dtype="int64"),
        })
        with mock.patch("trkperf.truth.read_truth_particles", return_value=empty_truth), \
             mock.patch("trkperf.truth.read_truth_hit_layer_counts", return_value=empty_layers):
            result = acceptance.compute_acceptance(["unreadable.root"])
        n_pt = len(config.PT_BIN_EDGES) - 1
        n_eta = len(config.ETA_BIN_EDGES) - 1
        self.assertEqual(len(result), len(config.SPECIES) * n_pt * n_eta)
        self.assertTrue(result["insufficient_stats"].all())
        self.assertTrue((result["n_generated"] == 0).all())
        self.assertTrue(result["acceptance"].isna().all())
        self.assertIn("skipped_files", result.attrs)
        self.assertEqual(result.attrs["run_params"]["min_layers"], config.ACCEPTANCE_MIN_LAYERS)

    def test_resolution_empty_matches_keep_schema(self):
        from trkperf import config
        from trkperf.metrics import resolution

        empty_pairs = pd.DataFrame({
            "species": pd.Series(dtype="object"), "is_matched": pd.Series(dtype="bool"),
            "reco_pt": pd.Series(dtype="float64"), "pt": pd.Series(dtype="float64"),
            "eta": pd.Series(dtype="float64"),
        })
        with mock.patch("trkperf.truth.read_truth_particles", return_value=empty_pairs), \
             mock.patch("trkperf.truth.select_primary", return_value=empty_pairs), \
             mock.patch("trkperf.reco.read_reco_tracks", return_value=empty_pairs), \
             mock.patch("trkperf.matching.read_associations", return_value=empty_pairs), \
             mock.patch("trkperf.matching.build_matched_pairs", return_value=empty_pairs):
            result = resolution.compute_resolution(["unreadable.root"])
        n_pt = len(config.PT_BIN_EDGES) - 1
        n_eta = len(config.ETA_BIN_EDGES) - 1
        self.assertEqual(len(result), len(config.SPECIES) * n_pt * n_eta)
        for col in ("mu", "sigma", "mu_err", "sigma_err", "chi2_ndf"):
            self.assertIn(col, result.columns)
            self.assertTrue(result[col].isna().all())
        self.assertFalse(result["fit_converged"].any())
        self.assertTrue(result["insufficient_stats"].all())


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

    def test_compare_pid_confusion_joins_on_truth_species(self):
        # Regression: the join keys omitted truth_species, so a clean
        # (pi+ -> K+) row joined a bkg (pi- -> K+) row in the same bin and
        # the "comparison" compared different truth species.
        cols = {
            "pt_bin_center": [1.0],
            "eta_bin_center": [0.0],
            "reco_species": ["K+"],
            "confusion_frac_err": [0.03],
            "insufficient_stats": [False],
        }
        clean = pd.DataFrame({**cols, "truth_species": ["pi+"],
                              "confusion_frac": [0.1]})
        bkg = pd.DataFrame({**cols, "truth_species": ["pi-"],
                            "confusion_frac": [0.2]})
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))
        # Different truth species must NOT join: two rows, NaN opposite.
        self.assertEqual(len(merged), 2)
        self.assertTrue(
            merged["confusion_frac_ratio_bkg_over_clean"].isna().all())

    def test_joins_on_string_bin_labels(self):
        # String interval labels are the primary join keys: float centers
        # must never be the thing holding two rows together.
        clean = pd.DataFrame(
            {
                "species": ["pi+"],
                "pt_bin": ["(0.5, 1.0]"],
                "eta_bin": ["(0.0, 0.5]"],
                "pt_bin_center": [0.75],
                "eta_bin_center": [0.25],
                "acceptance": [0.8],
                "acceptance_err": [0.02],
                "insufficient_stats": [False],
            }
        )
        bkg = clean.copy()
        bkg["acceptance"] = [0.4]
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(
            merged["acceptance_ratio_bkg_over_clean"].iloc[0], 0.5
        )

    def test_zero_bkg_value_has_finite_ratio_error(self):
        # R = 0 is well-defined (b = 0); its error is db/|c|, not NaN.
        clean = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.5],
                "fake_rate_err": [0.05],
            }
        )
        bkg = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.0],
                "fake_rate_err": [0.02],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))
        self.assertEqual(merged["fake_rate_ratio_bkg_over_clean"].iloc[0], 0.0)
        self.assertAlmostEqual(
            merged["fake_rate_ratio_bkg_over_clean_err"].iloc[0], 0.02 / 0.5
        )

    def test_null_errors_are_coerced_not_fatal(self):
        # JSON nulls arrive as None/object: they must become NaN, never a
        # to_numpy(dtype=float) exception.
        clean = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.5],
                "fake_rate_err": [None],
            }
        )
        bkg = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.4],
                "fake_rate_err": [0.03],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))
        self.assertAlmostEqual(
            merged["fake_rate_ratio_bkg_over_clean"].iloc[0], 0.8
        )

    def test_bkg_only_pair_still_appears(self):
        # A value/err pair present on only one side must show up with NaNs
        # opposite, never silently vanish from the comparison.
        clean = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.5],
                "fake_rate_err": [0.05],
            }
        )
        bkg = pd.DataFrame(
            {
                "pt_bin_center": [1.0],
                "eta_bin_center": [0.0],
                "fake_rate": [0.4],
                "fake_rate_err": [0.03],
                "purity": [0.9],
                "purity_err": [0.01],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            clean_path = Path(tmp) / "clean.json"
            bkg_path = Path(tmp) / "bkg.json"
            report.to_json(clean, str(clean_path))
            report.to_json(bkg, str(bkg_path))
            merged = compare.compare_metric(str(clean_path), str(bkg_path))
        self.assertIn("purity_diff_bkg_minus_clean", merged.columns)
        self.assertTrue(merged["purity_clean"].isna().all())
        self.assertTrue(merged["purity_diff_bkg_minus_clean"].isna().all())
        self.assertAlmostEqual(merged["purity_bkg"].iloc[0], 0.9)


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

    def test_shared_budget_covers_all_reads_of_a_run(self):
        # Two reads sharing one budget: the first failure is tolerated, the
        # second one (shared total 2 > 1) aborts - budgets must not reset.
        good = pd.DataFrame({"file_id": [0], "event": [0], "idx": [0], "x": [1.0]})

        def fake_open(path, tree_name="events"):
            if path.startswith("BAD"):
                raise OSError("transient network error")
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        shared: dict = {}
        with mock.patch.object(io, "open_tree", side_effect=fake_open), \
             mock.patch.object(io, "read_flat", side_effect=fake_read):
            first = io.read_flat_multi(
                ["BAD", "GOOD"], {"x": "T/x"}, max_failures=1, shared_failures=shared
            )
            self.assertEqual(len(first), 1)
            self.assertEqual(shared["skipped"], ["BAD"])
            with self.assertRaises(RuntimeError):
                io.read_flat_multi(
                    ["BAD-again", "GOOD"], {"x": "T/x"}, max_failures=1,
                    shared_failures=shared,
                )

    def test_shared_skip_set_keeps_tables_aligned(self):
        # A path that failed once is skipped immediately by later reads of the
        # same run: every joined table covers the same file subset.
        good = pd.DataFrame({"file_id": [0], "event": [0], "idx": [0], "x": [1.0]})
        calls = []

        def fake_open(path, tree_name="events"):
            calls.append(path)
            if path == "BAD":
                raise OSError("transient network error")
            return mock.Mock()

        def fake_read(tree, columns, file_id=0):
            return good.copy()

        shared: dict = {}
        with mock.patch.object(io, "open_tree", side_effect=fake_open), \
             mock.patch.object(io, "read_flat", side_effect=fake_read):
            io.read_flat_multi(
                ["BAD", "GOOD"], {"x": "T/x"}, max_failures=5, shared_failures=shared
            )
            n_calls_after_first = len(calls)
            second = io.read_flat_multi(
                ["BAD", "GOOD"], {"x": "T/x"}, max_failures=5, shared_failures=shared
            )
        # The second read never retried BAD (only GOOD was opened again).
        self.assertEqual(calls.count("BAD"), n_calls_after_first - 1)
        self.assertEqual(len(second), 1)


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
                first = io.read_flat_multi(  # noqa: F841 - warms the cache; assertions use `second`
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


class TestTrackingRegions(unittest.TestCase):
    """Endcap acceptance regions use their own collections/threshold (PLAN.md 4.1)."""

    def test_region_registry_is_sane(self):
        from trkperf import config

        for region, spec in config.TRACKING_REGIONS.items():
            self.assertTrue(len(spec["collections"]) >= 2, region)
            self.assertGreaterEqual(spec["min_layers"], 1, region)
            self.assertLessEqual(spec["min_layers"], len(spec["collections"]), region)
        self.assertEqual(config.TRACKING_REGIONS["central"]["min_layers"],
                         config.ACCEPTANCE_MIN_LAYERS)

    def test_unknown_region_raises(self):
        from trkperf.metrics import acceptance, efficiency

        with self.assertRaises(ValueError):
            acceptance.compute_acceptance([], region="nosecone")
        with self.assertRaises(ValueError):
            efficiency.compute_efficiency([], region="nosecone")

    def test_backward_acceptance_counts_endcap_hits(self):
        # Two backward electrons: one with BackwardMPGD+TrackerEndcap hits
        # (in), one with nothing (out). Central rule would reject both.
        from trkperf import config
        from trkperf.metrics import acceptance

        truth = pd.DataFrame({
            "file_id": [0, 0], "event": [0, 0], "idx": [0, 1],
            "generator_status": [1, 1], "species": ["e-", "e-"],
            "pt": [1.0, 1.0], "eta": [-2.75, -2.75],
        })
        layers = pd.DataFrame({
            "file_id": [0], "event": [0], "idx": [0],
            "n_layers_hit": [2],
        })
        with mock.patch("trkperf.truth.read_truth_particles", return_value=truth), \
             mock.patch("trkperf.truth.read_truth_hit_layer_counts",
                        return_value=layers) as rl:
            result = acceptance.compute_acceptance(["f.root"], region="backward")
        _, kwargs = rl.call_args
        self.assertEqual(list(kwargs["collections"]),
                         list(config.TRACKING_REGIONS["backward"]["collections"]))
        got = result[(result["species"] == "e-") & (result["n_generated"] > 0)]
        self.assertEqual(int(got["n_generated"].iloc[0]), 2)
        self.assertEqual(int(got["n_in_acceptance"].iloc[0]), 1)
        self.assertEqual(result.attrs["run_params"]["region"], "backward")


class TestEtaRegions(unittest.TestCase):
    def _ratio_df(self):
        # Two eta bins in barrel (-0.75, +0.75), one backward:
        # regional acceptance must equal summed counts, never mean of ratios.
        return pd.DataFrame([
            {"species": "e-", "pt_bin_center": 1.0, "eta_bin_center": -0.75,
             "acceptance": 0.5, "acceptance_err": 0.1, "n_generated": 100,
             "n_in_acceptance": 50, "insufficient_stats": False},
            {"species": "e-", "pt_bin_center": 1.0, "eta_bin_center": 0.75,
             "acceptance": 1.0, "acceptance_err": 0.0, "n_generated": 300,
             "n_in_acceptance": 300, "insufficient_stats": False},
            {"species": "e-", "pt_bin_center": 1.0, "eta_bin_center": -2.75,
             "acceptance": 0.0, "acceptance_err": 0.0, "n_generated": 1000,
             "n_in_acceptance": 0, "insufficient_stats": False},
        ])

    def test_region_assignment(self):
        from trkperf import report

        self.assertEqual(report.eta_region(-3.75), "backward endcap")
        self.assertEqual(report.eta_region(-1.25), "backward endcap")
        self.assertEqual(report.eta_region(-0.75), "barrel")
        self.assertEqual(report.eta_region(0.75), "barrel")
        self.assertEqual(report.eta_region(1.25), "forward endcap")
        self.assertEqual(report.eta_region(3.75), "forward endcap")
        self.assertEqual(report.eta_region(float("nan")), "unknown")

    def test_species_groups(self):
        from trkperf import report

        groups = report.SPECIES_GROUPS
        self.assertEqual(groups["e-"], groups["e+"])
        self.assertEqual(groups["pi+"], groups["pi-"])
        self.assertEqual(groups["K+"], groups["K-"])
        # Proton and antiproton stay separate (beam-charge asymmetry).
        self.assertNotEqual(groups["proton"], groups["antiproton"])
        # charge-conjugate merge is exact: (30 + 70) / (100 + 100) = 0.5
        df = pd.DataFrame([
            {"species": "e-", "pt_bin_center": 1.0, "eta_bin_center": 0.25,
             "acceptance": 0.3, "acceptance_err": 0.05, "n_generated": 100,
             "n_in_acceptance": 30, "insufficient_stats": False},
            {"species": "e+", "pt_bin_center": 1.0, "eta_bin_center": 0.25,
             "acceptance": 0.7, "acceptance_err": 0.05, "n_generated": 100,
             "n_in_acceptance": 70, "insufficient_stats": False},
        ])
        agg = report.aggregate_eta_species(
            df, "acceptance", "acceptance_err",
            count_cols=("n_in_acceptance", "n_generated"))
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg.iloc[0]["species_group"], groups["e-"])
        self.assertEqual(agg.iloc[0]["eta_region"], "barrel")
        self.assertAlmostEqual(agg.iloc[0]["acceptance"], 0.5)

    def test_ratio_recovered_exactly(self):
        agg = report.aggregate_eta_species(
            self._ratio_df(), "acceptance", "acceptance_err",
            count_cols=("n_in_acceptance", "n_generated"))
        # barrel: (50 + 300) / (100 + 300) = 0.875, NOT mean(0.5, 1.0) = 0.75
        barrel = agg[(agg["species_group"] == "e+/e-") & (agg["eta_region"] == "barrel")].iloc[0]
        self.assertAlmostEqual(barrel["acceptance"], 0.875)
        self.assertEqual(barrel["n"], 400.0)
        self.assertFalse(barrel["insufficient_stats"])
        # backward: 0 / 1000
        back = agg[(agg["species_group"] == "e+/e-") & (agg["eta_region"] == "backward endcap")].iloc[0]
        self.assertEqual(back["acceptance"], 0.0)
        self.assertEqual(set(agg["eta_region"]), {"barrel", "backward endcap"})

    def test_floor_reapplied(self):
        df = self._ratio_df()
        df.loc[df["eta_bin_center"] == 0.75, "n_generated"] = 10
        df.loc[df["eta_bin_center"] == 0.75, "n_in_acceptance"] = 10
        agg = report.aggregate_eta_species(
            df, "acceptance", "acceptance_err",
            count_cols=("n_in_acceptance", "n_generated"))
        barrel = agg[agg["eta_region"] == "barrel"].iloc[0]
        # (50 + 10) / (100 + 10) exact, n=110 passes the floor
        self.assertAlmostEqual(barrel["acceptance"], 60.0 / 110.0)
        df.loc[df["eta_bin_center"] == -0.75, "n_generated"] = 5
        agg2 = report.aggregate_eta_species(
            df, "acceptance", "acceptance_err",
            count_cols=("n_in_acceptance", "n_generated"))
        barrel2 = agg2[agg2["eta_region"] == "barrel"].iloc[0]
        self.assertTrue(barrel2["insufficient_stats"])  # n = 15 < 50

    def test_sigma_inverse_variance(self):
        df = pd.DataFrame([
            {"species": "pi+", "pt_bin_center": 1.0, "eta_bin_center": -0.25,
             "sigma": 0.10, "sigma_err": 0.01, "n_matched": 200,
             "fit_converged": True, "insufficient_stats": False},
            {"species": "pi+", "pt_bin_center": 1.0, "eta_bin_center": 0.25,
             "sigma": 0.20, "sigma_err": 0.02, "n_matched": 200,
             "fit_converged": True, "insufficient_stats": False},
            {"species": "pi+", "pt_bin_center": 1.0, "eta_bin_center": 0.75,
             "sigma": 9.99, "sigma_err": 0.001, "n_matched": 200,
             "fit_converged": False, "insufficient_stats": False},
        ])
        agg = report.aggregate_eta_species(df, "sigma", "sigma_err", count_cols=None)
        row = agg.iloc[0]
        # w = 1/0.01^2 : 1/0.02^2 = 4:1 -> (4*0.10 + 1*0.20)/5 = 0.12; the
        # non-converged row is excluded even though its tiny error would
        # otherwise dominate the weights.
        self.assertAlmostEqual(row["sigma"], 0.12, places=6)
        self.assertAlmostEqual(row["sigma_err"], 1.0 / np.sqrt(10000 + 2500), places=9)
        self.assertEqual(row["n"], 600.0)

    def test_eta_region_filter_keeps_one_region(self):
        agg = report.aggregate_eta_species(
            self._ratio_df(), "acceptance", "acceptance_err",
            count_cols=("n_in_acceptance", "n_generated"),
            eta_region_filter="barrel")
        self.assertTrue((agg["eta_region"] == "barrel").all())
        self.assertEqual(len(agg), 1)
        self.assertAlmostEqual(agg.iloc[0]["acceptance"], 0.875)


class TestCollectionProbe(unittest.TestCase):
    """_collection_present must never mistake a transient open failure for a
    genuinely absent branch (regression: on the flaky 26.07.1 endpoint a
    failed probe zeroed whole endcap collections for a run)."""

    def _tree(self, present: bool = True):
        tree = mock.MagicMock()
        if present:
            tree.arrays.return_value = {"b": mock.MagicMock()}
        else:
            tree.arrays.side_effect = ValueError("bad branch")
        return tree

    def test_transient_open_failure_falls_through_to_next_file(self):
        from trkperf import truth

        with mock.patch("trkperf.truth.io.open_tree",
                        side_effect=[RuntimeError("expired"), self._tree(True)]):
            self.assertTrue(truth._collection_present(["f0", "f1"], "b"))

    def test_genuine_absence_detected_on_opened_file(self):
        from trkperf import truth

        with mock.patch("trkperf.truth.io.open_tree",
                        return_value=self._tree(False)):
            self.assertFalse(truth._collection_present(["f0"], "b"))

    def test_no_file_opens_fails_loud_downstream(self):
        from trkperf import truth

        with mock.patch("trkperf.truth.io.open_tree",
                        side_effect=RuntimeError("expired")):
            # True: let the normal read path raise through the shared
            # failure budget rather than silently zeroing the collection.
            self.assertTrue(truth._collection_present(["f0", "f1"], "b"))


if __name__ == "__main__":
    unittest.main()
