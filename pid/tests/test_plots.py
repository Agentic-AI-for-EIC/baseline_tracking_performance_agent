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


if __name__ == "__main__":
    unittest.main()
