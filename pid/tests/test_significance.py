"""Analytic tests for the maximum-significance machinery.

Every assertion here has a hand-checkable expected value: FOM scans are compared
against a brute-force loop, limits against the closed-form Clopper-Pearson result,
scalings against their analytic power laws. A working-point finder that is
"self-consistent but wrong" is the most dangerous kind of analysis bug, because
its output looks like a measurement.
"""

from __future__ import annotations

import unittest

import numpy as np

from pid import config, significance as sg


def two_gaussians(n=4000, seed=0, mu_sig=0.8, mu_bkg=0.3, sigma=0.1):
    rng = np.random.default_rng(seed)
    s = rng.normal(mu_sig, sigma, n)
    b = rng.normal(mu_bkg, sigma, n)
    return np.clip(np.r_[s, b], 0, 1), np.r_[np.ones(n, int), np.zeros(n, int)]


class TestScanAgreesWithBruteForce(unittest.TestCase):
    def test_fom_efficiency_fake_and_purity(self):
        score, y = two_gaussians()
        table = sg.scan_thresholds(score, y, thresholds=np.linspace(0, 1, 51))
        Nt, Nb = int((y == 1).sum()), int((y == 0).sum())
        for _, row in table.iterrows():
            c = row["threshold"]
            keep = score >= c
            S, B = int((keep & (y == 1)).sum()), int((keep & (y == 0)).sum())
            self.assertEqual(int(row["n_signal"]), S)
            self.assertEqual(int(row["n_background"]), B)
            expected = S / np.sqrt(S + B) if S + B else np.nan
            if np.isfinite(expected):
                self.assertAlmostEqual(float(row["significance"]), expected, places=9)
                self.assertAlmostEqual(float(row["efficiency"]), S / Nt, places=9)
                self.assertAlmostEqual(float(row["fake_rate"]), B / Nb, places=9)
                self.assertAlmostEqual(float(row["purity"]), S / (S + B), places=9)

    def test_per_row_weights_follow_the_score_ranking(self):
        # Weights that vary *within* a class catch the subtlest failure mode of a
        # cumsum scan: if the weights are sorted by their own magnitude instead
        # of travelling with their score, every weighted yield is wrong while
        # the counts stay right, and the table still looks like a measurement.
        score, y = two_gaussians(n=400, seed=3)
        rng = np.random.default_rng(11)
        w = rng.uniform(0.2, 5.0, size=score.size)
        table = sg.scan_thresholds(score, y, weights=w, thresholds=np.linspace(0, 1, 41))
        Nt = float(w[y == 1].sum())
        Nb = float(w[y == 0].sum())
        for _, row in table.iterrows():
            c = row["threshold"]
            keep = score >= c
            S = float(w[keep & (y == 1)].sum())
            B = float(w[keep & (y == 0)].sum())
            self.assertAlmostEqual(float(row["S"]), S, places=9)
            self.assertAlmostEqual(float(row["B"]), B, places=9)
            self.assertAlmostEqual(float(row["S_eff"]),
                                   float(np.sum(w[keep & (y == 1)] ** 2)), places=9)
            self.assertAlmostEqual(float(row["efficiency"]), S / Nt, places=9)
            self.assertAlmostEqual(float(row["fake_rate"]), B / Nb, places=9)
            self.assertAlmostEqual(float(row["purity"]), S / (S + B), places=9)

    def test_per_bin_scan_aligns_weights_with_nan_inputs(self):
        # Regression: weights were indexed with the post-filter bin mask while
        # still full-length -> IndexError whenever any score/x was NaN.
        rng = np.random.default_rng(5)
        score = np.clip(rng.normal(0.5, 0.2, 200), 0, 1)
        y = (rng.random(200) < 0.4).astype(int)
        x = rng.uniform(0.5, 10, 200)
        x[::17] = np.nan
        w = rng.uniform(0.5, 2.0, 200)
        table = sg.optimal_cut_in_bins(score, y, x, weights=w,
                                       bins=np.array([0.5, 3.0, 10.0]),
                                       min_candidates=5)
        row = table.iloc[0]
        self.assertTrue(np.isfinite(row["threshold"]))
        sel = np.isfinite(x) & (x <= 3.0)
        keep = score[sel] >= float(row["threshold"])
        expected_eff = (w[sel][keep & (y[sel] == 1)].sum()
                        / w[sel][y[sel] == 1].sum())
        self.assertAlmostEqual(float(row["efficiency"]), expected_eff, places=9)

    def test_degenerate_bin_carries_caution(self):
        # A constant score cannot separate anything: the optimum accepts
        # everything, and the row must say so instead of quoting a cut.
        score = np.full(60, 0.7)
        y = np.r_[np.ones(30, dtype=int), np.zeros(30, dtype=int)]
        x = np.linspace(0.6, 0.9, 60)
        table = sg.optimal_cut_in_bins(score, y, x, bins=np.array([0.0, 1.0]),
                                       min_candidates=5)
        self.assertEqual(len(table), 1)
        self.assertIn("degenerate", str(table.iloc[0]["caution"]))

    def test_optimum_matches_the_brute_force_maximum(self):
        score, y = two_gaussians()
        table = sg.scan_thresholds(score, y)
        opt = sg.optimal_cut(table)
        grid = np.linspace(0, 1, 20001)
        fom = np.array([
            (lambda k: k[k == 1].size / np.sqrt(k.size) if k.size else -1.0)(y[sc >= c])
            for sc in [score] for c in grid])
        self.assertAlmostEqual(opt["significance"], float(np.nanmax(fom)), places=6)

    def test_ties_are_broken_towards_the_more_rejective_cut(self):
        score, y = two_gaussians(n=300)
        table = sg.scan_thresholds(score, y)
        opt = sg.optimal_cut(table)
        best = table.loc[table["significance"].idxmax(), "significance"]
        tied = table[np.isclose(table["significance"], best, rtol=0, atol=1e-12)]
        self.assertGreaterEqual(opt["threshold"], float(tied["threshold"].max()) - 1e-9)

    def test_fom_is_bounded_by_the_square_root_of_the_signal(self):
        score, y = two_gaussians(n=900)
        table = sg.scan_thresholds(score, y)
        n_sig = int((y == 1).sum())
        self.assertLessEqual(float(table["significance"].max()), np.sqrt(n_sig) + 1e-9)

    def test_separating_case_rejects_background(self):
        score, y = two_gaussians(mu_sig=0.8, mu_bkg=0.2)
        _, opt = sg.significance_curve(score, y)
        self.assertLess(opt["fake_rate"], 0.05)
        self.assertGreater(opt["efficiency"], 0.8)
        self.assertTrue(opt["rejects_background"])
        self.assertEqual(opt["caution"], "")


class TestScalings(unittest.TestCase):
    def test_lumi_scale_multiplies_fom_by_sqrt_L(self):
        score, y = two_gaussians(n=1200)
        t1 = sg.scan_thresholds(score, y, lumi_scale=1.0)
        t9 = sg.scan_thresholds(score, y, lumi_scale=9.0)
        ratio = t9["significance"] / t1["significance"]
        self.assertTrue(np.allclose(ratio.dropna(), 3.0, atol=1e-9))

    def test_uniform_scaling_does_not_move_the_optimum(self):
        score, y = two_gaussians(n=1500)
        _, o1 = sg.significance_curve(score, y, lumi_scale=1.0)
        _, o4 = sg.significance_curve(score, y, lumi_scale=4.0)
        self.assertAlmostEqual(o1["threshold"], o4["threshold"], places=6)

    def test_up_weighted_background_moves_the_cut_up(self):
        score, y = two_gaussians(n=1500)
        w = sg.flux_weights(y, signal=1, background=0, bkg_scale=50.0)
        plain = sg.optimal_cut(sg.scan_thresholds(score, y))
        weighted = sg.optimal_cut(sg.scan_thresholds(score, y, weights=w))
        self.assertGreater(weighted["threshold"], plain["threshold"])

    def test_variance_form_is_the_pessimistic_one_wherever_background_passes(self):
        # At the optimum of a strongly up-weighted background nothing is accepted,
        # so the two forms coincide; the difference must be checked where B > 0.
        score, y = two_gaussians(n=1500)
        w = sg.flux_weights(y, signal=1, background=0, bkg_scale=50.0)
        table = sg.scan_thresholds(score, y, weights=w)
        row = table[table["n_background"] > 0].iloc[0]
        self.assertGreater(float(row["significance"]), float(row["significance_eff"]))
        # and the variance form is exactly S / sqrt(Sum w^2)
        S = float(row["S"]); v = float(row["S_eff"]) + float(row["B_eff"])
        self.assertAlmostEqual(float(row["significance_eff"]), S / np.sqrt(v), places=9)


class TestLimitsAndRejection(unittest.TestCase):
    def test_garwood_upper_zero_events_matches_closed_form(self):
        n = 20
        exact = 1.0 - 0.05 ** (1.0 / n)
        self.assertAlmostEqual(sg.garwood_upper(0, n), exact, places=9)

    def test_rejection_limit_when_no_background_passes(self):
        r = sg.rejection_limits(0, 20)
        self.assertTrue(np.isnan(r["rejection"]), "1/0 must not be reported as a number")
        self.assertFalse(r["measurable"])
        self.assertAlmostEqual(r["rejection_limit"], 1.0 / (1.0 - 0.05 ** (1 / 20)), places=6)

    def test_rejection_point_estimate_and_limit_are_ordered(self):
        for k, n in ((1, 100), (2, 20), (10, 1000), (5, 5)):
            r = sg.rejection_limits(k, n)
            self.assertLessEqual(r["rejection_limit"], r["rejection"] + 1e-12,
                                 f"k={k},n={n}: the limit cannot exceed the estimate")

    def test_limit_tightens_with_statistics(self):
        weak = sg.rejection_limits(0, 20)
        strong = sg.rejection_limits(0, 2000)
        self.assertGreater(strong["rejection_limit"], 10 * weak["rejection_limit"])

    def test_empty_sample_returns_nan_not_crash(self):
        r = sg.rejection_limits(0, 0)
        self.assertTrue(np.isnan(r["rejection"]))
        self.assertFalse(r["measurable"])


class TestCautions(unittest.TestCase):
    def test_no_separation_is_reported_as_no_working_point(self):
        rng = np.random.default_rng(3)
        score = rng.uniform(0, 1, 2000)
        y = np.r_[np.ones(1000, int), np.zeros(1000, int)]
        _, opt = sg.significance_curve(score, y)
        self.assertIn("degenerate", opt["caution"])
        self.assertFalse(opt["rejects_background"])

    def test_tiny_background_sample_is_flagged_as_statistics_limited(self):
        score, y = two_gaussians(n=2000)
        keep = np.r_[np.ones(2000, bool), (y[2000:] > 0.5)[:20]]
        s = np.r_[score[:2000], score[2000:][:20]]
        yy = np.r_[np.ones(2000, int), np.zeros(min(20, len(keep)), int)]
        _, opt = sg.significance_curve(s, yy)
        self.assertTrue(opt["caution"])
        self.assertIn("background", opt["caution"].lower())

    def test_missing_class_raises(self):
        with self.assertRaises(ValueError):
            sg.scan_thresholds(np.linspace(0, 1, 50), np.ones(50, int))

    def test_unknown_mode_raises(self):
        table = sg.scan_thresholds(*two_gaussians(n=200))
        with self.assertRaises(ValueError):
            sg.optimal_cut(table, mode="make_it_up")


class TestPerBinOptima(unittest.TestCase):
    def test_bins_without_both_classes_are_nan_with_a_reason(self):
        score, y = two_gaussians(n=800)
        x = np.r_[np.full(800, 2.0), np.full(800, 6.0)]
        x[y == 0] = 2.0  # background only in the low bin
        tab = sg.optimal_cut_in_bins(score, y, x, bins=np.array([0.5, 4.0, 8.0, 12.0]))
        self.assertTrue(np.isnan(tab.loc[tab["bin_center"] == 10.0, "threshold"].iloc[0]))
        self.assertFalse(bool(tab.loc[tab["bin_center"] == 10.0, "valid"].iloc[0]))
        self.assertIn("candidates", tab.loc[tab["bin_center"] == 10.0, "reason"].iloc[0].lower())

    def test_bin_candidate_counts_are_conserved(self):
        score, y = two_gaussians(n=500)
        x = np.random.default_rng(5).uniform(0.5, 10.0, len(score))
        bins = config.PT_BINS_PERFORMANCE
        tab = sg.optimal_cut_in_bins(score, y, x, bins=bins)
        idx = np.clip(np.digitize(x, bins[1:-1], right=True), 0, len(bins) - 2)
        counts = np.bincount(idx, minlength=len(bins) - 1)
        self.assertEqual(int(tab["n_signal_bin"].sum() + tab["n_background_bin"].sum()),
                         int(counts.sum()))

    def test_min_candidates_is_respected(self):
        score, y = two_gaussians(n=60)
        x = np.full(len(score), 3.0)
        tab = sg.optimal_cut_in_bins(score, y, x, bins=np.array([0.5, 2.0, 10.0]),
                                     min_candidates=10_000)
        self.assertFalse(bool(tab["valid"].any()))


class TestEffectiveCounts(unittest.TestCase):
    def test_kish_effective_equals_count_for_unit_weights(self):
        self.assertEqual(sg.kish_effective(np.ones(37)), 37.0)

    def test_kish_effective_shrinks_under_unequal_weights(self):
        w = np.r_[np.full(10, 10.0), np.full(90, 0.1)]
        self.assertLess(sg.kish_effective(w), float(np.nansum(w)))
        self.assertAlmostEqual(sg.kish_effective(np.ones(37)), 37.0, places=9)

    def test_sum_of_squares_is_the_variance_not_an_effective_count(self):
        # The two concepts must not be conflated: with unequal weights Sigma w^2
        # grows while Kish's n_eff shrinks.
        w = np.r_[np.full(10, 10.0), np.full(90, 0.1)]
        self.assertAlmostEqual(sg.sum_of_squares(w), 10 * 100.0 + 90 * 0.01, places=9)
        self.assertLess(sg.kish_effective(w), 20.0)

    def test_scan_reports_effective_totals(self):
        score, y = two_gaussians(n=400)
        w = sg.flux_weights(y, signal=1, background=0, bkg_scale=3.0)
        table = sg.scan_thresholds(score, y, weights=w, thresholds=[0.0, 0.5])
        zero_cut = table[table["threshold"] == 0.0].iloc[0]
        self.assertAlmostEqual(float(zero_cut["B"]), 3.0 * 400, places=6)
        self.assertAlmostEqual(float(zero_cut["B_eff"]), 9.0 * 400, places=6)  # 50^2 x n


if __name__ == "__main__":
    unittest.main()
