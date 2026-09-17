"""Every learner: NaN handling, ranking, SHAP availability, multiclass."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pid import models


def make_problem(n=800, seed=3, multiclass=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "e_over_p": rng.normal(0.3, 0.1, n),
        "p": rng.uniform(1, 20, n),
        "pt": rng.uniform(1, 15, n),
        "shape_0": rng.normal(30, 10, n),
        "drich_gas_pathlength": rng.uniform(0, 1, n),
        "irt_gas_npe_tot_evt": rng.uniform(0, 50, n),
    })
    signal = rng.uniform(0, 1, n)
    # e_over_p and shape_0 carry the class information, p is a nuisance.
    y = np.where(signal < 0.33, 0, np.where(signal < 0.66, 1, 2)) if multiclass else \
        (X["e_over_p"] + 0.4 * X["shape_0"] / 30.0 + 0.15 * rng.normal(scale=0.1, size=n) > 0.55).astype(int)
    if not multiclass:
        X.loc[X["e_over_p"] > 0.5, "shape_0"] += 15.0
    miss = rng.random(X.shape) < 0.2
    X = X.mask(miss)
    return X, np.asarray(y)


class TestAllAdapters(unittest.TestCase):
    LIBS = ("lightgbm", "xgboost", "sklearn_hgb")

    def test_registry_contents(self):
        self.assertEqual(set(models.available()), set(self.LIBS))
        with self.assertRaises(ValueError):
            models.get_adapter("nonexistent")

    def test_binary_fit_predict_and_nan_survives(self):
        X, y = make_problem()
        for lib in self.LIBS:
            with self.subTest(lib=lib):
                adapter = models.get_adapter(lib)
                est = adapter.estimator({"n_estimators": 60} if lib != "sklearn_hgb"
                                        else {"max_iter": 60}, n_classes=2, seed=0)
                est.fit(X, y)
                proba = est.predict_proba(X)
                self.assertEqual(proba.shape, (len(X), 2))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
                # A candidate whose every subsystem is absent (a real case: a
                # central track outside every calorimeter's acceptance).
                all_nan = X.iloc[[0]].astype(float) * np.nan
                proba_nan = est.predict_proba(all_nan)
                self.assertEqual(proba_nan.shape, (1, 2))
                self.assertTrue(np.isfinite(proba_nan).all())

    def test_gain_ranks_the_informative_feature_first(self):
        X, y = make_problem()
        for lib in self.LIBS:
            with self.subTest(lib=lib):
                adapter = models.get_adapter(lib)
                est = adapter.estimator({"n_estimators": 150} if lib != "sklearn_hgb"
                                        else {"max_iter": 150}, n_classes=2, seed=0)
                est.fit(X, y)
                gain = adapter.gain(est, list(X.columns))
                self.assertEqual(len(gain), X.shape[1])
                self.assertIn(str(gain.index[0]), ("e_over_p", "shape_0"),
                              f"{lib}: gain put {gain.index[0]} on top")

    def test_exact_shap_shape_and_sign(self):
        X, y = make_problem()
        for lib in ("lightgbm", "xgboost"):
            with self.subTest(lib=lib):
                adapter = models.get_adapter(lib)
                est = adapter.estimator({"n_estimators": 80}, n_classes=2, seed=0)
                est.fit(X, y)
                values = adapter.shap(est, X)
                self.assertEqual(values.shape, (len(X), X.shape[1]))
                self.assertTrue(np.isfinite(values).all())
                # e_over_p drives the class positively
                col = list(X.columns).index("e_over_p")
                obs = np.nan_to_num(X["e_over_p"].to_numpy())
                corr = np.corrcoef(obs, values[:, col])[0, 1]
                self.assertGreater(corr, 0.0, f"{lib}: SHAP sign inverted for e_over_p")

    def test_sklearn_hgb_declines_shap_explicitly(self):
        X, y = make_problem()
        adapter = models.get_adapter("sklearn_hgb")
        est = adapter.estimator({"max_iter": 30}, n_classes=2, seed=0)
        est.fit(X, y)
        with self.assertRaises(NotImplementedError) as ctx:
            adapter.shap(est, X)
        self.assertIn("permutation", str(ctx.exception))

    def test_multiclass_probability_shape(self):
        X, y = make_problem(multiclass=True)
        for lib in self.LIBS:
            with self.subTest(lib=lib):
                adapter = models.get_adapter(lib)
                est = adapter.estimator({"n_estimators": 60} if lib != "sklearn_hgb"
                                        else {"max_iter": 60}, n_classes=3, seed=0)
                est.fit(X, y)
                proba = est.predict_proba(X)
                self.assertEqual(proba.shape, (len(X), 3))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_determinism_for_a_fixed_seed(self):
        X, y = make_problem()
        scores = []
        for _ in range(2):
            est = models.get_adapter("lightgbm").estimator({"n_estimators": 40},
                                                           n_classes=2, seed=11)
            est.fit(X, y)
            scores.append(est.predict_proba(X)[:, 1])
        np.testing.assert_allclose(scores[0], scores[1], rtol=0, atol=1e-12)


class TestObjectiveHelpers(unittest.TestCase):
    def test_objective_names(self):
        from pid.models.base import objective

        self.assertEqual(objective(2), ("binary", "auc"))
        self.assertEqual(objective(4), ("multiclass", "mlogloss"))


class TestAttributionMatch(unittest.TestCase):
    def test_leg_suffix_and_log_prefix_count_as_the_observable(self):
        from pid.train import _attribution_matches

        self.assertTrue(_attribution_matches("e_over_p", "e_over_p"))
        self.assertTrue(_attribution_matches("e_over_p_backward", "e_over_p"))
        self.assertTrue(_attribution_matches("log_e_over_p_forward", "e_over_p"))

    def test_mere_substring_does_not_count(self):
        # Regression: `primary in shap_top[0]` let any column containing the
        # substring pass as the required observable.
        from pid.train import _attribution_matches

        self.assertFalse(_attribution_matches("some_e_over_purity", "e_over_p"))
        self.assertFalse(_attribution_matches("ecal_backward_E", "e_over_p"))


if __name__ == "__main__":
    unittest.main()
