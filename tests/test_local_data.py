"""Integration smoke tests: run the real trkperf pipeline against the small
local reference files (no network needed - these are the already-downloaded
signal / signal_BKG_mix files described in AGENTS.md).

A couple of local files cannot populate every pT/eta/species bin, so these
tests deliberately do not check specific numeric values - they check that
the pipeline runs end-to-end without error, produces the expected columns,
and every ratio/rate column stays within its physically valid range. That
is enough to catch a broken join, a renamed branch, or a typo'd column name
before it reaches a multi-hour grid run.
"""

from __future__ import annotations

import pathlib
import unittest

from trkperf import config, matching, reco, truth
from trkperf.metrics import acceptance, efficiency, fake_rate, resolution
from trkperf.pid import confusion

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _local_file(tag: str) -> str:
    return str(_PROJECT_ROOT / config.LOCAL_REFERENCE_FILES[tag])


class TestTruthAndReco(unittest.TestCase):
    def test_read_truth_particles_has_expected_columns_and_nonneg_pt(self):
        df = truth.read_truth_particles([_local_file("clean")])
        self.assertGreater(len(df), 0)
        for col in ("pdg", "generator_status", "pt", "eta", "phi", "p", "species"):
            self.assertIn(col, df.columns)
        self.assertTrue((df["pt"] >= 0).all())

    def test_select_primary_filters_generator_status_and_known_species(self):
        df = truth.read_truth_particles([_local_file("clean")])
        primary = truth.select_primary(df)
        self.assertTrue((primary["generator_status"] == 1).all())
        self.assertTrue(primary["species"].notna().all())

    def test_truth_hit_layer_counts_are_within_valid_range(self):
        found: list = []
        counts = truth.read_truth_hit_layer_counts([_local_file("clean")], found_collections=found)
        self.assertGreater(len(counts), 0)
        n_collections = len(config.CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS)
        # Every row here came from at least one hit, and can hit at most
        # once per distinct collection (nunique), so 1 <= n <= n_collections.
        self.assertTrue((counts["n_layers_hit"] >= 1).all())
        self.assertTrue((counts["n_layers_hit"] <= n_collections).all())
        # The local clean file carries every central collection: nothing may
        # silently degrade to 0 here (cf. the 26.07.1 probe flake, where a
        # transient open failure zeroed whole collections for a run).
        self.assertEqual(set(found), set(config.CENTRAL_TRACKING_TRUTH_HIT_COLLECTIONS))

    def test_read_reco_tracks_has_nonneg_pt(self):
        df = reco.read_reco_tracks([_local_file("clean")])
        self.assertGreater(len(df), 0)
        self.assertTrue((df["pt"] >= 0).all())

    def test_reco_momentum_is_populated_not_all_zero(self):
        # Regression test: CentralCKFTracks.momentum.* is all zeros in these
        # RECO files; reco.py must pull momentum from CentralCKFTrackParameters
        # (qOverP/theta/phi). If reco pt is ~0 everywhere the physics is
        # silently meaningless, so assert it is genuinely populated.
        df = reco.read_reco_tracks([_local_file("clean")])
        self.assertGreater((df["pt"] > 0).sum(), 0,
                           "reco pt is all zero - momentum not populated")
        self.assertGreater((df["eta"].abs() < 100).sum(), 0)


class TestMatching(unittest.TestCase):
    def setUp(self):
        files = [_local_file("clean")]
        self.truth_df = truth.select_primary(truth.read_truth_particles(files))
        self.reco_df = reco.read_reco_tracks(files)
        self.assoc_df = matching.read_associations(files)

    def test_build_matched_pairs_is_a_left_join_from_truth(self):
        pairs = matching.build_matched_pairs(self.truth_df, self.reco_df, self.assoc_df)
        self.assertEqual(len(pairs), len(self.truth_df))
        self.assertIn("is_matched", pairs.columns)
        matched = pairs[pairs["is_matched"]]
        unmatched = pairs[~pairs["is_matched"]]
        self.assertTrue(matched["reco_pt"].notna().all())
        self.assertTrue(unmatched["reco_pt"].isna().all())

    def test_find_fake_tracks_is_a_left_join_from_reco(self):
        flagged = matching.find_fake_tracks(self.reco_df, self.assoc_df)
        self.assertEqual(len(flagged), len(self.reco_df))
        self.assertIn("is_fake", flagged.columns)


class TestMetricsEndToEnd(unittest.TestCase):
    def _assert_fraction_column(self, df, col):
        self.assertGreater(len(df), 0)
        valid = df[col].dropna()
        self.assertTrue(((valid >= 0) & (valid <= 1)).all(), f"{col} out of [0, 1] range")

    def test_compute_acceptance(self):
        df = acceptance.compute_acceptance([_local_file("clean")])
        self._assert_fraction_column(df, "acceptance")

    def test_compute_efficiency(self):
        df = efficiency.compute_efficiency([_local_file("clean")])
        self._assert_fraction_column(df, "efficiency_absolute")
        self._assert_fraction_column(df, "efficiency_within_acceptance")

    def test_compute_fake_rate_on_bkg_mixed_sample(self):
        # Deliberately the +background file: a single-particle-gun sample
        # would show ~0 fakes by construction (see AGENTS.md), but this
        # small local file is a real multi-particle DIS event, so it is a
        # legitimate (if low-statistics) smoke test of the fake-rate path.
        df = fake_rate.compute_fake_rate([_local_file("bkg_mixed")])
        self._assert_fraction_column(df, "fake_rate")

    def test_compute_resolution_sigma_nonnegative_where_converged(self):
        df = resolution.compute_resolution([_local_file("clean")])
        converged = df[df["fit_converged"]]
        if len(converged):
            self.assertTrue((converged["sigma"] >= 0).all())

    def test_compute_pid_confusion(self):
        df = confusion.compute_pid_confusion([_local_file("clean")])
        self.assertGreater(len(df), 0)
        for col in ("truth_species", "reco_species", "confusion_frac", "confusion_frac_err",
                     "efficiency", "efficiency_err", "n_matchedinbin", "n_truth_total_in_bin"):
            self.assertIn(col, df.columns)
        # Row-normalised fractions must be in [0, 1].
        valid = df["confusion_frac"].dropna()
        self.assertTrue(((valid >= 0) & (valid <= 1)).all())
        # Full (bin x truth x reco) grid: 128 bins x 8 truth species x
        # 9 reco hypotheses (8 + "unknown") - zero-count pairs are explicit
        # rows, never silent gaps.
        self.assertEqual(len(df), 128 * 8 * 9)
        zero = df[df["n_matchedinbin"] == 0]
        self.assertGreater(len(zero), 0)
        # A zero pair where the truth species IS present is an exact 0.0,
        # not NaN; where it is absent the fraction is undefined (NaN).
        zero_present = zero[zero["n_truth_total_in_bin"] > 0]
        self.assertTrue((zero_present["confusion_frac"] == 0.0).all())
        zero_absent = zero[zero["n_truth_total_in_bin"] == 0]
        self.assertTrue(zero_absent["confusion_frac"].isna().all())
        self.assertTrue(zero_absent["insufficient_stats"].all())
        # Row normalisation preserved: fractions sum to 1 per (bin, truth)
        # wherever the truth species has any matches.
        present = df[df["n_truth_total_in_bin"] > 0]
        sums = present.groupby(
            ["pt_bin", "eta_bin", "truth_species"], observed=True)["confusion_frac"].sum()
        self.assertTrue(((sums - 1.0).abs() < 1e-9).all())
        # truth_species column should be populated (non-null).
        self.assertTrue(df["truth_species"].notna().all())


if __name__ == "__main__":
    unittest.main()
