"""Metric correctness: intervals, ROOT fits, working points, tables."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pid import config, evaluate


def score_sample(n=4000, seed=1, separation=1.5):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)]
    mu_sig, mu_bkg, sigma = 0.8, 0.8 - separation * 0.1, 0.1
    s = np.r_[rng.normal(mu_sig, sigma, n // 2), rng.normal(mu_bkg, sigma, n // 2)]
    return y, np.clip(s, 0.0, 1.0), mu_sig, mu_bkg, sigma


class TestIntervals(unittest.TestCase):
    def test_garwood_brackets_the_point_estimate(self):
        lo, hi = evaluate.garwood(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertLess(lo, hi)

    def test_garwood_edge_cases(self):
        lo, hi = evaluate.garwood(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        lo, hi = evaluate.garwood(100, 100)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 1.0)
        self.assertTrue(np.isnan(evaluate.garwood(1, 0)[0]))

    def test_interval_shrinks_with_statistics(self):
        wide = evaluate.garwood(5, 100)
        narrow = evaluate.garwood(50, 1000)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])


class TestRootFits(unittest.TestCase):
    def test_recovers_mean_and_sigma(self):
        rng = np.random.default_rng(0)
        v = rng.normal(0.37, 0.052, 5000)
        fit = evaluate.fit_gaussian(v)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["mu"], 0.37, delta=0.01)
        self.assertAlmostEqual(fit["sigma"], 0.052, delta=0.01)
        self.assertGreater(fit["chi2"] / max(fit["ndf"], 1), 0.0)

    def test_too_few_entries_is_reported_not_fabricated(self):
        fit = evaluate.fit_gaussian(np.array([0.1, 0.2, 0.3]))
        self.assertFalse(fit["converged"])
        self.assertTrue(np.isnan(fit["mu"]))
        self.assertLess(fit["n"], config.MIN_ENTRIES_PER_BIN)

    def test_constant_input_does_not_converge_to_a_fake_width(self):
        fit = evaluate.fit_gaussian(np.full(300, 0.5))
        self.assertFalse(fit["converged"])

    def test_n_sigma_matches_separation_over_resolution(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.8, 0.1, 4000)
        b = rng.normal(0.5, 0.1, 4000)
        value = evaluate.n_sigma(evaluate.fit_gaussian(a), evaluate.fit_gaussian(b))
        self.assertAlmostEqual(value, 3.0, delta=0.35)

    def test_n_sigma_is_nan_when_a_fit_failed(self):
        bad = {"mu": np.nan, "sigma": np.nan, "converged": False}
        good = evaluate.fit_gaussian(np.random.default_rng(4).normal(0, 1, 500))
        self.assertTrue(np.isnan(evaluate.n_sigma(bad, good)))


class TestWorkingPoints(unittest.TestCase):
    def setUp(self):
        self.y, self.score, self.mu_sig, self.mu_bkg, self.sigma = score_sample()

    def test_auc_matches_the_analytic_two_gaussian_value(self):
        """For two equal-width Gaussians AUC = Phi(Δmu / (sigma*sqrt(2)))."""
        from scipy.stats import norm

        expected = float(norm.cdf((self.mu_sig - self.mu_bkg) / (self.sigma * np.sqrt(2))))
        value = evaluate.auc(self.y, self.score)
        self.assertLess(abs(value - expected), 0.02)
        self.assertLessEqual(value, 1.0)

    def test_efficiency_is_monotone_in_the_fake_rate_target(self):
        effs = [evaluate.working_point(self.y, self.score, t)["efficiency"]
                for t in (1e-4, 1e-3, 1e-2, 1e-1)]
        for low, high in zip(effs, effs[1:]):
            self.assertLessEqual(low, high + 1e-9)

    def test_achieved_fake_rate_never_exceeds_the_target(self):
        for target in (1e-4, 1e-3, 1e-2):
            wp = evaluate.working_point(self.y, self.score, target)
            self.assertLessEqual(wp["fake_rate"], target + 1e-12)
            self.assertTrue(wp["meets_target"])

    def test_small_background_sample_cannot_PROVE_a_tiny_fake_rate(self):
        """The point estimate can be 0/20, but the interval must say otherwise.

        With 20 background candidates a perfect ranking accepts none of them, so
        ``fake_rate`` is 0.0 and the target looks met - the honest statement is the
        upper limit on the true rate, which is ~0.16 at 95 % CL. This test pins
        that behaviour: a working point may never be quoted from the point
        estimate alone.
        """
        y = np.r_[np.ones(200, dtype=int), np.zeros(20, dtype=int)]
        rng = np.random.default_rng(5)
        score = np.r_[rng.normal(0.8, 0.1, 200), rng.normal(0.2, 0.05, 20)]
        wp = evaluate.working_point(y, score, 1e-4)
        self.assertEqual(wp["n_background"], 20)
        self.assertLessEqual(wp["fake_rate"], 1e-4)          # point estimate "meets" it
        upper = wp["fake_rate_err"][1]                        # ... but the limit does not
        self.assertGreater(upper, 1e-2)
        self.assertGreater(wp["fake_rate_err"][1], wp["fake_rate"])
        self.assertEqual(wp["n_background_pass"], 0)

    def test_missing_class_returns_nan_not_zero(self):
        wp = evaluate.working_point(np.ones(50, dtype=int), np.linspace(0, 1, 50), 1e-3)
        self.assertTrue(np.isnan(wp["efficiency"]))
        self.assertEqual(wp["n_background"], 0)

    def test_efficiency_at_threshold_is_consistent_with_manual_counting(self):
        threshold = 0.75
        res = evaluate.efficiency_at_threshold(self.y, self.score, threshold, 1e-2)
        pass_sig = int(((self.score >= threshold) & (self.y == 1)).sum())
        self.assertEqual(res["n_signal_pass"], pass_sig)
        self.assertAlmostEqual(res["efficiency"], pass_sig / int((self.y == 1).sum()))
        pass_bkg = int(((self.score >= threshold) & (self.y == 0)).sum())
        self.assertAlmostEqual(res["fake_rate"], pass_bkg / int((self.y == 0).sum()))

    def test_auc_with_only_one_class_is_nan(self):
        self.assertTrue(np.isnan(evaluate.auc(np.ones(30, dtype=int), np.arange(30) / 30)))


class TestTables(unittest.TestCase):
    def _binary_frame(self):
        y, score, *_ = score_sample()
        return pd.DataFrame({"label": y, "score": score,
                             "truth_class": np.where(y == 1, "e", "pi"),
                             "pt": np.random.default_rng(7).uniform(1, 15, len(y)),
                             "eta": np.where(y == 1, -2.4, 2.4), "p": 5.0})

    def test_overall_table_contains_every_quoted_quantity(self):
        table = evaluate.overall_table(self._binary_frame(), task="eid")
        quantities = set(table["quantity"])
        self.assertIn("auc", quantities)
        self.assertIn("n_sigma", quantities)
        effs = table[table["quantity"] == "efficiency"]
        self.assertEqual(len(effs), len(config.FAKE_RATE_TARGETS))
        self.assertTrue(set(effs["target_fake_rate"]) == set(config.FAKE_RATE_TARGETS))
        self.assertTrue((effs["value_err_lo"] <= effs["value"]).all())
        self.assertTrue((effs["value"] <= effs["value_err_hi"]).all())

    def test_lower_limit_row_exists_whether_or_not_it_applies(self):
        # `pid compare` keys on the quantity column: if the row set depended on
        # whether a tag happened to saturate, clean vs bkg would silently
        # compare different quantities. The row must always exist.
        table = evaluate.overall_table(self._binary_frame(), task="eid")
        row = table[table["quantity"] == "n_sigma_lower_limit"]
        self.assertEqual(len(row), 1)
        self.assertTrue(np.isnan(float(row["value"].iloc[0])))
        self.assertTrue(bool(row["insufficient_stats"].iloc[0]))
        self.assertTrue(row["nsigma_method"].iloc[0].startswith("not applicable"))
        # and the same row carries a number when the separation saturates
        n_sig, n_bkg = 60, 20
        sat = pd.DataFrame({
            "label": np.r_[np.ones(n_sig, int), np.zeros(n_bkg, int)],
            "score": np.r_[np.linspace(0.90, 0.99, n_sig), np.linspace(0.01, 0.10, n_bkg)],
            "truth_class": ["e"] * n_sig + ["pi"] * n_bkg,
            "pt": 5.0, "eta": 0.0, "p": 5.0})
        srow = evaluate.overall_table(sat, task="eid")
        srow = srow[srow["quantity"] == "n_sigma_lower_limit"]
        self.assertEqual(len(srow), 1)
        self.assertTrue(np.isfinite(float(srow["value"].iloc[0])))
        self.assertTrue(srow["nsigma_method"].iloc[0].startswith("lower limit"))

    def test_binned_table_flags_thin_bins(self):
        frame = self._binary_frame()
        table = evaluate.binned_table(frame, task="eid", by="pt")
        self.assertTrue({"insufficient_stats", "n_signal", "n_background", "auc"} <= set(table.columns))
        thin = table[table["insufficient_stats"]]
        self.assertTrue((thin[["n_signal", "n_background"]].min(axis=1)
                         < config.MIN_ENTRIES_PER_BIN).all())
        self.assertEqual(table["working_point"].unique().tolist(), ["per_bin"])
        fixed = evaluate.binned_table(frame, task="eid", by="pt", threshold=0.7)
        self.assertEqual(fixed["working_point"].unique().tolist(), ["global"])

    def test_binned_table_rejects_unknown_variable(self):
        with self.assertRaises(ValueError):
            evaluate.binned_table(self._binary_frame(), task="eid", by="charge")

    def test_confusion_rows_normalise_to_one(self):
        table = evaluate.confusion_table(self._binary_frame(), task="eid", threshold=0.7)
        for _, group in table.groupby("truth_class"):
            self.assertAlmostEqual(float(group["row_fraction"].sum()), 1.0, places=6)
        self.assertEqual(set(table["truth_class"]), {"e", "pi"})

    def test_roc_table_is_monotone(self):
        table = evaluate.roc_table(self._binary_frame(), task="eid")
        self.assertTrue((np.diff(table["fake_rate"].to_numpy()) >= -1e-12).all())
        self.assertTrue((np.diff(table["efficiency"].to_numpy()) >= -1e-12).all())

    def test_calibration_recovers_a_perfect_predictor(self):
        rng = np.random.default_rng(9)
        p = rng.uniform(0, 1, 6000)
        y = (rng.uniform(0, 1, 6000) < p).astype(int)
        table = evaluate.calibration_table(pd.DataFrame({"score": p, "label": y}))
        self.assertLess(float(np.abs(table["pulled"] - table["observed"]).mean()), 0.06)

    def test_signal_background_uses_task_labels_for_binary(self):
        frame = self._binary_frame()
        y, score, sig, bkg = evaluate.signal_background(frame, "eid")
        self.assertEqual((sig, bkg), ("e", "pi"))
        self.assertEqual(int(y.sum()), int((frame["truth_class"] == "e").sum()))
        _, _, sig2, bkg2 = evaluate.signal_background(frame, "ehad")
        self.assertEqual((sig2, bkg2), ("e", "hadron"))

    def test_multiclass_uses_proba_columns_in_the_stored_class_order(self):
        # Regression test: the class order must come from the task's label map.
        frame = pd.DataFrame({"truth_class": ["pi", "K", "p"] * 40,
                              "label": ([0, 1, 2] * 40),
                              "proba_pi": 0.8, "proba_K": 0.1, "proba_p": 0.1,
                              "pt": np.linspace(1, 10, 120), "eta": 2.4, "p": 5.0})
        y, score, sig, bkg = evaluate.signal_background(frame, "hadpid")
        self.assertEqual(sig, "pi")
        self.assertEqual(set(bkg.split("+")), {"K", "p"})
        self.assertEqual(int(y.sum()), 40)
        self.assertTrue(np.allclose(score, 0.8))
        y_k, _, sig_k, _ = evaluate.signal_background(frame, "hadpid", signal="K")
        self.assertEqual(sig_k, "K")
        self.assertEqual(int(y_k.sum()), 40)

    def test_multiclass_without_proba_columns_fails_clearly(self):
        with self.assertRaises(ValueError):
            evaluate.signal_background(pd.DataFrame({"truth_class": ["pi"]}), "hadpid")


if __name__ == "__main__":
    unittest.main()


class TestSeparationEstimators(unittest.TestCase):
    """n_sigma must not come from fitting a Gaussian to a bounded [0,1] score."""

    def test_quantile_estimator_values(self):
        self.assertAlmostEqual(evaluate.nsigma_from_auc(0.5), 0.0, places=12)
        # AUC = Phi(d/sqrt2) at d=3 -> Phi(2.1213) = 0.98305
        self.assertAlmostEqual(evaluate.nsigma_from_auc(0.98305), 3.0, delta=0.01)
        self.assertTrue(np.isnan(evaluate.nsigma_from_auc(1.0)))
        self.assertTrue(np.isnan(evaluate.nsigma_from_auc(0.0)))
        self.assertLess(evaluate.nsigma_from_auc(0.3), 0.0,
                        "an inverted score must stay visibly negative")
        aucs = np.linspace(0.55, 0.999, 40)
        values = [evaluate.nsigma_from_auc(a) for a in aucs]
        self.assertTrue(all(x < y for x, y in zip(values, values[1:])))

    def test_quantile_matches_the_pooled_sigma_definition(self):
        rng = np.random.default_rng(11)
        n = 30000
        for d in (0.5, 1.0, 2.0, 3.0):
            s = rng.normal(0.0, 1.0, n)
            b = rng.normal(-d, 1.0, n)
            y = np.r_[np.ones(n, int), np.zeros(n, int)]
            auc_value = evaluate.auc(y, np.r_[s, b])
            fs = evaluate.fit_gaussian(s)
            fb = evaluate.fit_gaussian(b)
            self.assertAlmostEqual(evaluate.nsigma_from_auc(auc_value),
                                   evaluate.n_sigma(fs, fb), delta=0.12,
                                   msg=f"d={d}: the two estimators must agree")

    def test_bounded_scores_break_the_raw_fit_but_not_the_others(self):
        # A strong classifier piles both classes against 0 and 1; the raw Gaussian
        # then measures the clip, which is the failure this change removes.
        rng = np.random.default_rng(12)
        s = np.clip(rng.normal(0.98, 0.05, 3000), 0, 1)
        b = np.clip(rng.normal(0.02, 0.05, 3000), 0, 1)
        quantile = evaluate.measure_separation(s, b, method="quantile")
        raw = evaluate.measure_separation(s, b, method="raw")
        self.assertTrue(np.isfinite(quantile["n_sigma"]))
        self.assertIn("DEPRECATED", raw["space"])
        self.assertIn("clip", raw["reason"])
        self.assertGreater(raw["n_sigma"], 1.5 * quantile["n_sigma"],
                           "the raw fit should visibly inflate separation when "
                           "the scores saturate - which is why it is not the default")

    def test_auc_is_reported_even_below_the_fit_floor(self):
        rng = np.random.default_rng(13)
        s = np.clip(rng.normal(0.8, 0.1, 200), 0, 1)
        b = np.clip(rng.normal(0.4, 0.1, 12), 0, 1)   # fewer than MIN_ENTRIES_PER_BIN
        res = evaluate.measure_separation(s, b)
        self.assertTrue(np.isfinite(res["auc"]))
        self.assertTrue(np.isfinite(res["n_sigma"]))
        self.assertTrue(res["fit_limited"])
        self.assertTrue(np.isnan(res["n_sigma_logit"]))
        self.assertIn("peak fits", res["reason"])

    def test_perfect_separation_yields_a_lower_limit_not_a_number(self):
        s = np.linspace(0.51, 0.99, 120)
        b = np.linspace(0.01, 0.49, 40)
        res = evaluate.measure_separation(s, b)
        self.assertEqual(res["auc"], 1.0)
        self.assertTrue(res["saturated"])
        self.assertTrue(np.isnan(res["n_sigma"]))
        self.assertTrue(np.isfinite(res["n_sigma_lower_limit"]))
        self.assertGreater(res["n_sigma_lower_limit"], 0.0)
        self.assertIn("95 % CL", res["reason"])

    def test_no_overlap_probability_matches_the_exact_combinatorics(self):
        from math import comb

        for ns, nb in ((40, 15), (60, 20)):
            exact = 1.0 / comb(ns + nb, ns)
            numeric = evaluate._p_no_overlap(0.0, ns, nb)
            self.assertGreater(numeric, 0.0)
            self.assertAlmostEqual(np.log(numeric), np.log(exact), places=3,
                                   msg=f"ns={ns}, nb={nb}")

    def test_lower_limit_is_calibrated(self):
        """The solved limit must really give 5 % perfect-separation experiments."""
        rng = np.random.default_rng(17)
        ns, nb, trials = 60, 20, 40000
        d_limit = evaluate.nsigma_lower_limit(ns, nb)
        s = rng.normal(0.0, 1.0, (trials, ns))
        b = rng.normal(-d_limit, 1.0, (trials, nb))
        rate = float((b.max(axis=1) < s.min(axis=1)).mean())
        err = np.sqrt(0.05 * 0.95 / trials)
        self.assertLess(abs(rate - 0.05) / err, 3.0,
                        f"observed {rate:.4f} vs nominal 0.05 ({err:.4f})")

    def test_limit_is_refused_when_the_sample_cannot_support_one(self):
        # With 1 x 1 candidates, P(no overlap) = 1/2 even at d = 0, so observing
        # perfect separation excludes nothing: no limit exists and none is invented.
        self.assertTrue(np.isnan(evaluate.nsigma_lower_limit(1, 1)))
        self.assertTrue(np.isnan(evaluate.nsigma_lower_limit(3, 3)))
        self.assertTrue(np.isfinite(evaluate.nsigma_lower_limit(60, 20)))
        res = evaluate.measure_separation(np.array([0.9, 0.95, 0.97]),
                                          np.array([0.1, 0.2, 0.05]))
        # too few to fit, too few to separate at all -> the reason must not print "nan"
        self.assertFalse(res["saturated"] and np.isfinite(res["n_sigma_lower_limit"]))
        self.assertNotIn("nan", res["reason"].lower().replace("n_sigma", ""))

    def test_limit_message_states_the_number_only_when_it_exists(self):
        res = evaluate.measure_separation(np.linspace(0.51, 0.99, 120),
                                          np.linspace(0.01, 0.49, 40))
        self.assertIn("n_sigma >", res["reason"])
        self.assertTrue(np.isfinite(res["n_sigma_lower_limit"]))
        # 2 x 2: P(no overlap | d=0) = 1/binom(4,2) = 0.167 > alpha, so perfect
        # separation here excludes nothing and no limit may be quoted.
        tiny = evaluate.measure_separation(np.linspace(0.51, 0.99, 2),
                                           np.linspace(0.01, 0.49, 2))
        self.assertIn("too small to set any lower limit", tiny["reason"])
        self.assertTrue(np.isnan(tiny["n_sigma_lower_limit"]))
        # 4 x 3 does admit a limit, but it is weak and must say so
        weak = evaluate.measure_separation(np.linspace(0.51, 0.99, 4),
                                           np.linspace(0.01, 0.49, 3))
        self.assertIn("n_sigma >", weak["reason"])
        self.assertLess(weak["n_sigma_lower_limit"], 1.0)
        self.assertIn("weak", weak["reason"])

    def test_limit_grows_with_the_smaller_sample(self):
        small = evaluate.nsigma_lower_limit(60, 5)
        large = evaluate.nsigma_lower_limit(60, 600)
        self.assertLess(small, large, "more background candidates must tighten the limit")
        self.assertTrue(np.isnan(evaluate.nsigma_lower_limit(0, 10)))

    def test_overall_table_reports_the_estimator_used(self):
        rng = np.random.default_rng(14)
        n = 600
        y = np.r_[np.ones(n, int), np.zeros(n, int)]
        score = np.r_[rng.normal(0.8, 0.1, n), rng.normal(0.35, 0.1, n)]
        df = pd.DataFrame({"label": y, "score": np.clip(score, 0, 1),
                           "truth_class": np.where(y == 1, "e", "pi"),
                           "pt": rng.uniform(0.5, 10, 2 * n), "eta": -2.4, "p": 5.0})
        table = evaluate.overall_table(df, task="eid")
        ns = table[table["quantity"] == "n_sigma"].iloc[0]
        self.assertIn("nsigma_method", table.columns)
        self.assertEqual(str(ns["nsigma_method"]), config.NSIGMA_METHOD)
        self.assertTrue(np.isfinite(float(ns["value"])))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            evaluate.measure_separation(np.linspace(0, 1, 60), np.linspace(0, 1, 60),
                                        method="by_vibes")


class TestLogitTransform(unittest.TestCase):
    def test_maps_unit_interval_to_the_reals(self):
        z = evaluate.logit(np.array([0.0, 0.5, 1.0]))
        self.assertLess(float(z[0]), -10.0)
        self.assertAlmostEqual(float(z[1]), 0.0, places=12)
        self.assertGreater(float(z[2]), 10.0)

    def test_is_monotone_and_finite(self):
        p = np.linspace(0.0, 1.0, 101)
        z = evaluate.logit(p)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(bool(np.all(np.diff(z) >= 0)))

    def test_epsilon_is_configurable(self):
        # a smaller epsilon lets a saturated score reach a LARGER |z|
        self.assertGreater(float(evaluate.logit(np.array([1.0]), eps=1e-12)[0]),
                           float(evaluate.logit(np.array([1.0]), eps=1e-3)[0]))
        self.assertAlmostEqual(float(evaluate.logit(np.array([0.5]), eps=1e-12)[0]), 0.0,
                               places=12)
