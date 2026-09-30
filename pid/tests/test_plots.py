"""Figure labelling: units must follow the slicing variable, and the y axis must
state what the plotted quantity is (a reader should not have to guess whether
"auc" is global or per-bin, or which species pair it ranks)."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from pid import plots


def _table(variable="pt"):
    return pd.DataFrame({
        "variable": [variable] * 3, "signal": ["e"] * 3, "against": ["pi"] * 3,
        "bin_center": [0.5, 2.0, 8.0], "auc": [0.7, 0.99, 0.95],
        "insufficient_stats": [True, False, False], "working_point": ["per_bin"] * 3,
    })


class TestAxisLabels(unittest.TestCase):
    def test_momentum_variables_carry_gev(self):
        for variable in ("pt", "p"):
            xlabel = plots.axis_labels(_table(variable), "auc")[0]
            self.assertIn("GeV", xlabel, variable)

    def test_eta_is_not_labelled_with_units(self):
        xlabel = plots.axis_labels(_table("eta"), "auc")[0]
        self.assertNotIn("GeV", xlabel)
        self.assertIn("eta", xlabel)

    def test_ylabel_defines_auc_and_names_the_pair(self):
        _, ylabel = plots.axis_labels(_table("pt"), "auc")
        self.assertIn("AUC", ylabel)
        self.assertIn("e vs pi", ylabel)

    def test_unknown_column_falls_back_to_its_name(self):
        _, ylabel = plots.axis_labels(_table("pt"), "some_new_metric")
        self.assertIn("some_new_metric", ylabel)


class TestMetricVsBin(unittest.TestCase):
    def test_writes_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "auc_vs_pt.png")
            out = plots.metric_vs_bin(_table("pt"), path, column="auc", title="x")
            self.assertEqual(out, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)

    def test_all_nan_column_does_not_crash(self):
        table = _table("pt")
        table["auc"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.png")
            plots.metric_vs_bin(table, path, column="auc")   # empty but valid figure
            self.assertTrue(os.path.exists(path))

    def test_linear_x_axis_for_signed_bins(self):
        # eta bins straddle zero, so a log x scale would silently drop them.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "eta.png")
            table = _table("eta")
            table["bin_center"] = [-2.5, -1.5, 2.5]
            plots.metric_vs_bin(table, path, column="auc")
            self.assertTrue(os.path.exists(path))


class TestScoreDistDensity(unittest.TestCase):
    def _over(self):
        edges = [i / 10 for i in range(11)]
        return {"classes": {
            "e": {"train": {"counts": [0, 1, 3, 6] + [0] * 5 + [50],
                            "edges": edges, "n": 60},
                  "test": {"counts": [0, 0, 1, 2] + [0] * 5 + [17],
                           "edges": edges, "n": 20}},
            "pi": {"train": {"counts": [40, 5] + [0] * 8, "edges": edges, "n": 45},
                   "test": {"counts": [12, 2] + [0] * 8, "edges": edges, "n": 14}}},
                "ks": {}}

    def test_density_mode_writes_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dens.png")
            plots.score_dist_train_vs_test(self._over(), path, density=True)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)

    def test_count_mode_unchanged(self):
        # default path keeps the log-counts rendering (with its +0.3 offset).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "counts.png")
            plots.score_dist_train_vs_test(self._over(), path)
            self.assertTrue(os.path.exists(path))

    def test_peak_mode_normalises_maximum_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "peak.png")
            plots.score_dist_train_vs_test(self._over(), path, peak=True)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)


class TestRocOverlay(unittest.TestCase):
    """Cross-learner overlay: one ROC per library, or the CLI refuses."""

    def _roc(self, boost: float = 0.0):
        fx = [0.0, 1e-4, 1e-3, 1e-1, 1.0]
        eff = [0.0, 0.4 + boost, 0.7 + boost, 0.99, 1.0]
        return pd.DataFrame({"fake_rate": fx, "efficiency": eff,
                             "threshold": [1, .5, .3, .1, 0],
                             "signal": "e", "against": "pi"})

    def test_writes_figure_with_one_curve_per_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "overlay.png")
            out = plots.roc_multi({"lightgbm": self._roc(), "xgboost": self._roc(0.02)},
                                  path, title="t",
                                  aucs={"lightgbm": 0.91, "xgboost": 0.93})
            self.assertEqual(out, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)

    def test_cmd_overlay_end_to_end(self):
        import argparse

        import numpy as np

        from pid import cli
        with tempfile.TemporaryDirectory() as tmp:
            mdir = os.path.join(tmp, "models")
            odir = os.path.join(tmp, "out")
            for lib, shift in (("lightgbm", 0.0), ("xgboost", 0.1)):
                d = os.path.join(mdir, f"eid_{lib}_clean")
                os.makedirs(d)
                rng = np.random.default_rng(7)
                label = rng.integers(0, 2, 300)
                score = label * 0.5 + shift + rng.normal(0, 0.4, 300)
                pd.DataFrame({"label": label, "score": score}).to_pickle(
                    os.path.join(d, "test_scores.pkl"))
            rc = cli.cmd_overlay(argparse.Namespace(
                task="eid", dataset_tag="clean",
                models="lightgbm,xgboost,sklearn_hgb", signal=None,
                model_dir=mdir, out_dir=odir))
            self.assertEqual(rc, 0)  # the missing third learner is skipped, not fatal
            fig = os.path.join(odir, "plot_PID", "plot_model_eval", "clean",
                               "pid-eid_overlay_clean-roc.png")
            self.assertTrue(os.path.exists(fig))

    def test_cmd_overlay_refuses_fewer_than_two_learners(self):
        import argparse

        from pid import cli
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cli.cmd_overlay(argparse.Namespace(
                    task="eid", dataset_tag="clean",
                    models="lightgbm,xgboost,sklearn_hgb", signal=None,
                    model_dir=os.path.join(tmp, "nothing"),
                    out_dir=os.path.join(tmp, "out")))


if __name__ == "__main__":
    unittest.main()
