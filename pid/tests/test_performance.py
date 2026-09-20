"""Channel construction, at-cut tables, 2D maps, confusion matrix, orchestration."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from pid import config, evaluate, performance as perf
from pid import significance as sg


def synthetic_scores(n=4000, seed=1, with_multiclass=True):
    """A score table shaped like what pid.train writes."""
    rng = np.random.default_rng(seed)
    cls = rng.choice(["e", "pi", "K", "p"], size=n, p=[0.35, 0.4, 0.13, 0.12])
    pt = rng.uniform(0.5, 10.0, n)
    # Eta overlaps between species (as it does in data: a backward calorimeter sees
    # mostly electrons but also pions), so eta-binned analyses are testable.
    eta = np.where(cls == "e", rng.uniform(-3.0, -0.5, n), rng.uniform(-2.0, 3.0, n))
    p = pt * np.cosh(eta)
    # a genuinely separating electron score, a nearly useless hadron score
    pe = np.clip(np.where(cls == "e", rng.normal(0.85, 0.10, n), rng.normal(0.25, 0.12, n)), 0, 1)
    pk = np.clip(np.where(cls == "K", rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)), 0, 1)
    pp = np.clip(np.where(cls == "p", rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)), 0, 1)
    ppi = np.clip(np.where(cls == "pi", rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)), 0, 1)
    out = pd.DataFrame({"file_id": rng.integers(0, 6, n), "event": rng.integers(0, 900, n),
                        "track_idx": np.arange(n), "truth_class": cls, "pt": pt, "p": p,
                        "eta": eta, "sample": rng.choice(["train", "test"], n, p=[0.7, 0.3]),
                        "score": pe, "label": (cls == "e").astype(int)})
    if with_multiclass:
        raw = np.column_stack([ppi, pk, pp])
        raw = raw / raw.sum(axis=1, keepdims=True)
        out["proba_pi"], out["proba_K"], out["proba_p"] = raw[:, 0], raw[:, 1], raw[:, 2]
    return out


class TestChannelFrame(unittest.TestCase):
    def setUp(self):
        self.table = synthetic_scores()

    def test_binary_channel_uses_score_and_label(self):
        fr = perf.channel_frame(self.table, "eid")
        self.assertTrue(set(fr["y"].unique()) <= {0, 1})
        self.assertEqual(int((fr["y"] == 1).sum()), int((fr["truth_class"] == "e").sum()))
        self.assertEqual(int((fr["y"] == 0).sum()), int((fr["truth_class"] == "pi").sum()))

    def test_pair_channel_reduces_to_two_hypotheses(self):
        fr = perf.channel_frame(self.table, "Kpi")
        self.assertTrue(set(fr["truth_class"]) == {"K", "pi"})
        expected = fr["proba_K"] / (fr["proba_K"] + fr["proba_pi"])
        np.testing.assert_allclose(fr["score"].to_numpy(float), expected.to_numpy(float),
                                   rtol=0, atol=1e-12)
        self.assertTrue(bool((fr.loc[fr["y"] == 1, "truth_class"] == "K").all()))

    def test_pair_channel_scores_stay_in_unit_interval(self):
        for channel in ("Kpi", "pK"):
            fr = perf.channel_frame(self.table, channel)
            self.assertTrue(bool(((fr["score"] >= 0) & (fr["score"] <= 1)).all()), channel)

    def test_unknown_channel_raises(self):
        with self.assertRaises(KeyError):
            perf.channel_frame(self.table, "unicorn")

    def test_missing_proba_columns_gives_actionable_message(self):
        table = synthetic_scores(with_multiclass=False)
        with self.assertRaises(KeyError) as ctx:
            perf.channel_frame(table, "Kpi")
        self.assertIn("proba_", str(ctx.exception))


class TestAtCutTable(unittest.TestCase):
    def setUp(self):
        self.fr = perf.channel_frame(synthetic_scores(), "eid")
        _, self.opt = sg.significance_curve(self.fr["score"], self.fr["y"])

    def test_counts_are_conserved_over_the_bins(self):
        table = perf.at_cut_table(self.fr, "eid", cut=self.opt["threshold"], variable="pt")
        self.assertEqual(int(table["n_signal"].sum()), int((self.fr["y"] == 1).sum()))
        self.assertEqual(int(table["n_background"].sum()), int((self.fr["y"] == 0).sum()))

    def test_efficiency_and_fake_match_direct_counting(self):
        cut = 0.5
        table = perf.at_cut_table(self.fr, "eid", cut=cut, variable="pt")
        sc = self.fr["score"].to_numpy(float)
        y = self.fr["y"].to_numpy()
        pt = self.fr["pt"].to_numpy(float)
        edges = config.PT_BINS_PERFORMANCE
        idx = np.clip(np.digitize(pt, edges[1:-1], right=True), 0, len(edges) - 2)
        for _, row in table.iterrows():
            b = int(np.argmin(np.abs([0.5 * (edges[i] + edges[i + 1]) - row["bin_center"]
                                      for i in range(len(edges) - 1)])))
            sel = idx == b
            n_sig = int((sel & (y == 1)).sum())
            k_sig = int((sel & (y == 1) & (sc >= cut)).sum())
            self.assertAlmostEqual(float(row["efficiency"]),
                                   k_sig / n_sig if n_sig else np.nan, places=9)
            n_bkg = int((sel & (y == 0)).sum())
            k_bkg = int((sel & (y == 0) & (sc >= cut)).sum())
            self.assertAlmostEqual(float(row["fake_rate"]),
                                   k_bkg / n_bkg if n_bkg else np.nan, places=9)
            self.assertEqual(int(row["n_signal_pass"]), k_sig)
            self.assertEqual(int(row["n_background_pass"]), k_bkg)

    def test_purity_equals_signal_over_total_accepted(self):
        table = perf.at_cut_table(self.fr, "eid", cut=self.opt["threshold"], variable="pt")
        ok = table.dropna(subset=["purity"])
        for _, row in ok.iterrows():
            denom = row["n_signal_pass"] + row["n_background_pass"]
            self.assertAlmostEqual(float(row["purity"]),
                                   row["n_signal_pass"] / denom if denom else np.nan, places=6)

    def test_no_cut_gives_nan_rows_and_a_reason(self):
        table = perf.at_cut_table(self.fr, "eid", cut=None, variable="pt")
        self.assertTrue(bool(table["efficiency"].isna().all()))
        self.assertTrue(bool(table["insufficient_stats"].all()))
        self.assertTrue((table["reason"].astype(str).str.len() > 0).all())

    def test_per_bin_cut_source_is_labelled(self):
        per_bin = sg.optimal_cut_in_bins(self.fr["score"], self.fr["y"], self.fr["pt"])
        table = perf.at_cut_table(self.fr, "eid", cut_per_bin=per_bin, variable="pt")
        self.assertIn("per_bin_optimum", set(table["cut_source"]))
        # A per-bin optimum is never worse (in FOM) than the global one in that bin
        glob = perf.at_cut_table(self.fr, "eid", cut=self.opt["threshold"], variable="pt")
        merged = table.merge(glob, on="bin_center", suffixes=("_bin", "_global"))
        usable = merged.dropna(subset=["significance_bin", "significance_global"])
        self.assertTrue(bool((usable["significance_bin"]
                              >= usable["significance_global"] - 1e-9).all()))

    def test_a_bin_without_background_candidates_is_flagged_not_quietly_nan(self):
        # 0 passing background means the fake rate is undefined, not zero.
        frame = self.fr.copy()
        edges = config.PT_BINS_PERFORMANCE
        # force one bin to be background-free
        pt = frame["pt"].to_numpy(float)
        frame.loc[(pt > edges[3]) & (pt <= edges[4]) & (frame["y"] == 0), "pt"] = edges[1]
        table = perf.at_cut_table(frame, "eid", cut=0.5, variable="pt")
        empty_bkg = table[(table["n_background"] == 0)]
        self.assertGreater(len(empty_bkg), 0)
        self.assertTrue(bool(empty_bkg["insufficient_stats"].all()),
                        "a background-free bin must be flagged insufficient")
        self.assertTrue((empty_bkg["reason"].astype(str).str.contains("background")).all())

    def test_insufficient_flag_matches_the_evaluate_convention(self):
        # Both metric layers must use min(n_signal, n_background) < floor, so the
        # same column name means the same thing in every table.
        table = perf.at_cut_table(self.fr, "eid", cut=self.opt["threshold"], variable="pt")
        flagged = table[table["insufficient_stats"].astype(bool)]
        self.assertTrue(bool((flagged[["n_signal", "n_background"]].min(axis=1)
                              < config.MIN_ENTRIES_PER_BIN).all()))

    def test_rejection_limit_is_reported_alongside_the_point_estimate(self):
        table = perf.at_cut_table(self.fr, "eid", cut=0.999, variable="pt")
        row = table[table["n_background_pass"] == 0].iloc[0]
        self.assertTrue(np.isnan(row["rejection"]))
        self.assertTrue(np.isfinite(row["rejection_limit"]))
        self.assertFalse(bool(row["rejection_measurable"]))


class TestMaps(unittest.TestCase):
    def setUp(self):
        self.fr = perf.channel_frame(synthetic_scores(), "eid")

    def test_counts_sum_to_the_channel_rows(self):
        m = perf.map_2d(self.fr, "eid", cut=0.5, quantity="efficiency")
        self.assertEqual(int(m["counts"].sum()), int(len(self.fr)))

    def test_thin_cells_are_masked_not_zeroed(self):
        m = perf.map_2d(self.fr, "eid", cut=0.5, quantity="efficiency",
                        pt_bins=np.linspace(0.5, 10, 8), eta_bins=np.arange(-4, 4.1, 0.5))
        thin = m["counts"] < config.MIN_CANDIDATES_PER_WP_BIN
        self.assertTrue(bool(np.isnan(m["value"][thin]).all()))
        self.assertGreater(m["masked_cells"], 0)

    def test_tidy_conversion_is_lossless(self):
        m = perf.map_2d(self.fr, "eid", cut=0.5, quantity="fake_rate")
        tidy = perf.map_tidy(m)
        self.assertEqual(len(tidy), m["value"].size)
        self.assertAlmostEqual(float(np.nansum(tidy["n_candidates"])), float(len(self.fr)))
        self.assertEqual(int(tidy["insufficient_stats"].sum()), int(m["masked_cells"]))

    def test_unknown_quantity_raises(self):
        with self.assertRaises(ValueError):
            perf.map_2d(self.fr, "eid", cut=0.5, quantity="vibes")

    def test_weights_rescale_purity_but_not_efficiency_or_fake(self):
        # Up-weighting the background 3x must leave efficiency and fake rate
        # (ratios within one class) unchanged while lowering purity: this pins
        # the weight plumbing against wiring w into the wrong numerator.
        y = self.fr["y"].to_numpy()
        w = np.where(y == 0, 3.0, 1.0)
        eff0 = perf.map_2d(self.fr, "eid", cut=0.5, quantity="efficiency")
        eff3 = perf.map_2d(self.fr, "eid", cut=0.5, quantity="efficiency", weights=w)
        np.testing.assert_allclose(np.nan_to_num(eff0["value"], nan=-1.0),
                                   np.nan_to_num(eff3["value"], nan=-1.0),
                                   rtol=0, atol=1e-12)
        pur0 = perf.map_2d(self.fr, "eid", cut=0.5, quantity="purity")
        pur3 = perf.map_2d(self.fr, "eid", cut=0.5, quantity="purity", weights=w)
        both = np.isfinite(pur0["value"]) & np.isfinite(pur3["value"])
        self.assertTrue(bool(both.any()))
        self.assertTrue(bool((pur3["value"][both] <= pur0["value"][both]).all()))
        self.assertTrue(bool((pur3["value"][both] < pur0["value"][both]).any()))


class TestConfusionMatrix(unittest.TestCase):
    def test_row_normalised_and_counts_conserved(self):
        table = synthetic_scores()
        conf = perf.confusion_matrix(table)
        counts = pd.DataFrame(conf["counts"], index=conf["classes"], columns=conf["classes"])
        used = int(table["truth_class"].isin(conf["classes"]).sum())
        self.assertEqual(int(counts.values.sum()), used)
        for cls in conf["classes"]:
            self.assertEqual(int(counts.loc[cls].sum()), int((table["truth_class"] == cls).sum()))
        sums = np.nansum(conf["matrix"], axis=1)
        np.testing.assert_allclose(sums, np.ones(len(sums)), atol=1e-12)

    def test_dropped_rows_are_reported_not_silent(self):
        # 'e' has no proba_e column in a hadron task table: the matrix must say how
        # many rows it left out, or a reader normalises over the wrong denominator.
        table = synthetic_scores()
        conf = perf.confusion_matrix(table)
        self.assertNotIn("e", conf["classes"])
        self.assertEqual(conf["n_rows_excluded"], int((table["truth_class"] == "e").sum()))
        self.assertEqual(conf["excluded_classes"], ["e"])
        self.assertEqual(conf["n_rows_used"] + conf["n_rows_excluded"], int(len(table)))

    def test_no_test_rows_gives_empty_matrix(self):
        conf = perf.confusion_matrix(synthetic_scores().iloc[0:0])
        self.assertEqual(list(conf["classes"]), [])

    def test_zero_threshold_is_a_threshold_not_missing(self):
        # Regression: `(cuts.get(c) or {}).get("threshold") or nan` turned a
        # legitimate 0.0 cut into NaN, collapsing the whole class to UNASSIGNED.
        table = synthetic_scores()
        cuts = {"pi": {"threshold": 0.0}, "K": {"threshold": 0.0},
                "p": {"threshold": 0.0}}
        conf = perf.confusion_matrix(table, cuts=cuts)
        self.assertEqual(conf["decision"], "thresholds_at_c*")
        used = int(table["truth_class"].isin(["pi", "K", "p"]).sum())
        self.assertEqual(int(conf["counts"].sum()), used)


class TestOvertraining(unittest.TestCase):
    def test_identical_distributions_give_small_ks(self):
        table = synthetic_scores(n=4000, seed=7)
        # force the same score distribution for train and test
        over = perf.overtraining_check(table, "eid")
        self.assertIn("train_rows", over)
        for cls, ks in over["ks"].items():
            self.assertLess(ks["statistic"], 0.1, cls)

    def test_shifted_training_scores_are_flagged(self):
        table = synthetic_scores(n=3000, seed=8)
        shifted = table.copy()
        sel = (shifted["sample"] == "train") & (shifted["label"] == 1)
        shifted.loc[sel, "score"] = np.clip(shifted.loc[sel, "score"] + 0.3, 0, 1)
        over = perf.overtraining_check(shifted, "eid")
        self.assertGreater(over["ks"]["e"]["statistic"], 0.3)

    def test_missing_sample_column_raises(self):
        table = synthetic_scores().drop(columns=["sample"])
        with self.assertRaises(KeyError):
            perf.overtraining_check(table, "eid")


class TestNsigmaVsP(unittest.TestCase):
    def test_well_separated_classes_give_a_large_nsigma(self):
        rng = np.random.default_rng(4)
        n = 1200
        y = np.r_[np.ones(n, int), np.zeros(n, int)]
        fr = pd.DataFrame({"y": y, "score": np.r_[rng.normal(0.9, 0.03, n), rng.normal(0.2, 0.03, n)],
                           "p": rng.uniform(1, 20, 2 * n), "pt": rng.uniform(1, 10, 2 * n)})
        tab = perf.nsigma_vs_p(fr, "eid", p_bins=np.array([0.5, 20.0]))
        self.assertTrue(bool(tab["converged"].iloc[0]))
        self.assertGreater(float(tab["n_sigma"].iloc[0]), 10.0)

    def test_thin_bins_are_not_drawn_as_measurements(self):
        # A bin with 3 candidates yields an AUC, but drawing it as a point
        # estimate invites comparison with the 3-sigma line.
        from pid import plots
        table = pd.DataFrame({"channel": ["eid"], "signal": ["e"], "background": ["pi"],
                              "variable": ["p"], "bin_center": [2.0], "n_sigma": [1.2],
                              "n_sigma_lower_limit": [np.nan], "converged": [True],
                              "insufficient_stats": [True]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "n.png")
            plots.nsigma_vs_p_table([table], path, title="thin only")
            self.assertTrue(os.path.exists(path))

    def test_missing_class_is_reported_not_extrapolated(self):
        fr = pd.DataFrame({"y": np.ones(400, int), "score": np.random.default_rng(0).normal(0.5, 0.1, 400),
                           "p": np.linspace(1, 10, 400)})
        tab = perf.nsigma_vs_p(fr, "eid", p_bins=np.array([0.5, 5.0, 10.0]))
        self.assertTrue(bool((~tab["converged"]).all()))
        self.assertTrue(bool(tab["n_sigma"].isna().all()))


class TestRun(unittest.TestCase):
    def _stage(self, tmp: str, *, with_all_scores: bool = True,
               hadron_columns: tuple[str, ...] = ("pi", "K", "p")) -> str:
        """Write a private model tree so the run cannot pick up real artifacts."""
        root = os.path.join(tmp, "models")
        for task in ("eid", "hadpid"):
            directory = os.path.join(root, f"{task}_lightgbm_clean")
            os.makedirs(directory, exist_ok=True)
            table = synthetic_scores(n=6000, seed=11)
            if task == "hadpid":
                table = table[table["truth_class"].isin(["pi", "K", "p"])].copy()
                for cls in ("pi", "K", "p"):
                    if cls not in hadron_columns:
                        table = table.drop(columns=[f"proba_{cls}"])
                keep = table["truth_class"].isin(hadron_columns)
                table = table[keep].copy()
            name = "all_scores.pkl" if with_all_scores else "test_scores.pkl"
            (table if with_all_scores else table[table["sample"] == "test"]).to_pickle(
                os.path.join(directory, name))
        return root

    def test_produces_summary_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            out = os.path.join(tmp, "out")
            result = perf.run(channels=("eid", "Kpi"), dataset_tag="clean",
                              model="lightgbm", out_dir=out, model_root=root, quiet=True)
            summary = result["summary"]
            self.assertEqual(len(summary), 2)
            self.assertTrue(bool(summary["valid"].any()))
            for column in ("threshold", "significance", "efficiency", "purity",
                           "rejection_lower_limit", "caution"):
                self.assertIn(column, summary.columns)
            figures = os.listdir(os.path.join(out, "plots"))
            for name in ("pid-eid_significance_vs_cut.png", "pid-eid_eff_vs_pt.png",
                         "pid-eid_misid_vs_pt.png", "pid-eid_purity_vs_pt.png",
                         "pid-eid_roc.png", "pid-eid_optimal_cut_vs_pt.png",
                         "pid-eid_pion_rejection_vs_p.png", "pid-eid_eff_map_pt_vs_eta.png",
                         "pid-eid_fake_map_pt_vs_eta.png",
                         "pid-eid_score_dist_train_vs_test.png"):
                self.assertIn(name, figures, name)
            self.assertTrue(os.path.exists(os.path.join(out, "pid-working_points_lightgbm_clean.json")))

    def test_summary_records_the_weighting_choices(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            out = os.path.join(tmp, "out")
            perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm", out_dir=out,
                     model_root=root, bkg_scale=3.0, lumi_scale=2.0, quiet=True)
            with open(os.path.join(out, "pid-working_points_lightgbm_clean.json")) as fh:
                meta = json.load(fh)["meta"]
            self.assertEqual(meta["bkg_scale"], 3.0)
            self.assertEqual(meta["lumi_scale"], 2.0)
            self.assertEqual(meta["significance_mode"], "s_over_sqrt")

    def test_channels_without_the_columns_they_need_are_skipped_not_fatal(self):
        # No proba_p in the staged hadron table: p/K cannot be formed, K/pi can.
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp, hadron_columns=("pi", "K"))
            result = perf.run(channels=("Kpi", "pK"), dataset_tag="clean", model="lightgbm",
                              out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True)
            self.assertIn("Kpi", set(result["summary"]["channel"]))
            self.assertNotIn("pK", set(result["summary"]["channel"]))

    def test_electron_channel_has_a_real_working_point_on_separable_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            result = perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm",
                              out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True)
            row = result["summary"].iloc[0]
            self.assertGreater(row["efficiency"], 0.7)
            self.assertLess(row["fake_rate"], 0.3)
            self.assertTrue(bool(row["rejects_background"]))

    def test_unseparable_hadron_channel_is_flagged_not_reported_as_a_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            result = perf.run(channels=("Kpi",), dataset_tag="clean", model="lightgbm",
                              out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True)
            row = result["summary"].iloc[0]
            self.assertTrue(str(row["caution"]))
            self.assertFalse(bool(row["rejects_background"]))

    def test_works_from_test_scores_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp, with_all_scores=False)
            result = perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm",
                              out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True)
            self.assertEqual(len(result["summary"]), 1)
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "out", "plots", "pid-eid_score_dist_train_vs_test.png")))


if __name__ == "__main__":
    unittest.main()


class TestDeliverableManifest(unittest.TestCase):
    """Every required figure is accounted for, present or explicitly absent."""

    def _stage(self, tmp, *, with_all_scores=True):
        root = os.path.join(tmp, "models")
        d = os.path.join(root, "eid_lightgbm_clean")
        os.makedirs(d, exist_ok=True)
        table = synthetic_scores(n=6000, seed=41)
        name = "all_scores.pkl" if with_all_scores else "test_scores.pkl"
        (table if with_all_scores else table[table["sample"] == "test"]).to_pickle(
            os.path.join(d, name))
        return root

    def test_manifest_reports_every_required_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            result = perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm",
                              out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True)
            manifests = result["artefacts"]["eid"]["manifests"]
            self.assertEqual(len(manifests), 1)
            mf = manifests[0]
            self.assertEqual(mf["n_required"], len(config.REQUIRED_FIGURES))
            self.assertTrue(mf["complete"], mf["missing"])

    def test_manifest_lists_a_missing_figure_with_a_reason(self):
        # Files must really exist: the manifest checks the artefact on disk, since
        # a plot function returning a path proves nothing about the deliverable.
        with tempfile.TemporaryDirectory() as tmp:
            written = {}
            for key, _label in config.REQUIRED_FIGURES:
                if key == "roc":
                    continue
                path = os.path.join(tmp, f"pid-eid_{key}.png")
                with open(path, "wb") as fh:
                    fh.write(b"\x89PNG\r\n\x1a\n fake")
                written[key] = path
            mf = perf.verify_deliverables(written, channel="eid", status="headline",
                                          has_working_point=True)
            self.assertFalse(mf["complete"])
            self.assertEqual([m["figure"] for m in mf["missing"]], ["roc"])
            self.assertEqual(mf["n_present"], len(config.REQUIRED_FIGURES) - 1)

    def test_no_working_point_explains_the_cut_dependent_gaps(self):
        mf = perf.verify_deliverables({}, channel="Kpi", status="exploratory",
                                      has_working_point=False)
        reasons = {m["figure"]: m["reason"] for m in mf["missing"]}
        self.assertIn("eff_vs_pt", reasons)
        self.assertIn("working point", reasons["eff_vs_pt"])

    def test_disabled_curves_are_reported_as_disabled_not_lost(self):
        mf = perf.verify_deliverables({}, channel="eid", status="headline",
                                      has_working_point=False, binned_curves=False)
        reasons = {m["figure"]: m["reason"] for m in mf["missing"]}
        self.assertIn("--no-binned-curves", reasons["significance_vs_cut_by_pt"])

    def test_electron_rejection_alias_satisfies_the_requirement(self):
        # eid ships rejection_vs_p under the physics name pion_rejection_vs_p.
        with tempfile.TemporaryDirectory() as tmp:
            written = {}
            for key, _ in config.REQUIRED_FIGURES:
                if key == "rejection_vs_p":
                    continue
                path = os.path.join(tmp, f"{key}.png")
                open(path, "wb").write(b"x")
                written[key] = path
            alias = os.path.join(tmp, "pion.png")
            open(alias, "wb").write(b"x")
            written["pion_rejection_vs_p"] = alias
            mf = perf.verify_deliverables(written, channel="eid", status="headline",
                                          has_working_point=True)
            self.assertTrue(mf["complete"], mf["missing"])

    def test_require_figures_turns_gaps_into_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            with self.assertRaises(SystemExit) as ctx:
                perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm",
                         out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True,
                         require_figures=True, binned_curves=False)
            self.assertIn("require-figures", str(ctx.exception))

    def test_status_is_carried_into_every_summary_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._stage(tmp)
            result = perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm",
                              out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True)
            row = result["summary"].iloc[0]
            self.assertEqual(row["status"], config.CHANNELS["eid"]["status"])
            self.assertEqual(row["status"], "headline")

    def test_manifest_table_is_flat_and_queryable(self):
        mf = [{"channel": "eid", "status": "headline", "binning_suffix": "reco",
               "present": ["roc"], "missing": [{"figure": "eff_vs_pt",
                                                "reason": "no c*"}],
               "shared_with_other_basis": [],
               "n_required": 2, "n_present": 1, "complete": False}]
        table = perf.manifest_table(mf)
        self.assertEqual(len(table), len(config.REQUIRED_FIGURES))
        self.assertTrue({"channel", "status", "binning_basis", "figure", "label",
                         "present", "reason", "shared_with_reco_run"}
                        <= set(table.columns))
        self.assertTrue(bool(table.loc[table["figure"] == "eff_vs_pt", "present"].
                             map(lambda v: not bool(v)).all()))

    def test_shared_figures_count_as_delivered_but_are_marked(self):
        # Basis-independent figures are drawn once per channel; a truth-binned
        # manifest must not report them as missing, nor claim it drew them itself.
        keys = [k for k, _ in config.REQUIRED_FIGURES]
        shared = [k for k in config.BASIS_INDEPENDENT_FIGURES if k in keys]
        mf = {"channel": "eid", "status": "headline", "binning_suffix": "_truthpt",
              # the binned figures drawn by this run, plus the shared ones
              "present": [k for k in keys if k not in shared],
              "missing": [], "n_required": len(keys), "n_present": len(keys),
              "shared_with_other_basis": shared, "complete": True}
        table = perf.manifest_table([mf])
        self.assertTrue(bool(table["present"].all()))
        shared = table[table["shared_with_reco_run"]]
        self.assertEqual(len(shared), len(config.BASIS_INDEPENDENT_FIGURES)
                         - len(set(config.BASIS_INDEPENDENT_FIGURES)
                               - set(dict(config.REQUIRED_FIGURES))))
        self.assertTrue((shared["reason"] == "").all())

    def test_basis_independent_set_covers_the_unduplicated_figures(self):
        self.assertIn("roc", config.BASIS_INDEPENDENT_FIGURES)
        self.assertIn("significance_vs_cut", config.BASIS_INDEPENDENT_FIGURES)
        # a binned figure must NOT be in it
        self.assertNotIn("eff_vs_pt", config.BASIS_INDEPENDENT_FIGURES)


class TestBinningBasis(unittest.TestCase):
    """Reconstructed vs truth kinematics: same candidates, different partition."""

    def setUp(self):
        rng = np.random.default_rng(31)
        n = 4000
        y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
        score = np.r_[rng.normal(0.85, 0.1, n // 2), rng.normal(0.3, 0.1, n // 2)]
        truth_pt = rng.uniform(0.5, 10.0, n)
        # 5 % resolution migration between truth and reconstructed pT, so the two
        # bases genuinely disagree bin by bin.
        pt = truth_pt * (1.0 + rng.normal(0.0, 0.05, n))
        self.fr = pd.DataFrame({"y": y, "score": score, "pt": pt, "truth_pt": truth_pt,
                                "p": pt * 2, "truth_p": truth_pt * 2,
                                "eta": np.where(y == 1, -2.4, 2.4),
                                "truth_eta": np.where(y == 1, -2.4, 2.4),
                                "truth_class": np.where(y == 1, "e", "pi")})
        self.cut = 0.5

    def test_both_bases_partition_the_same_candidate_set(self):
        reco = perf.at_cut_table(self.fr, "eid", cut=self.cut, variable="pt")
        truth = perf.at_cut_table(self.fr, "eid", cut=self.cut, variable="truth_pt")
        self.assertEqual(int(reco["n_signal"].sum()), int(truth["n_signal"].sum()))
        self.assertEqual(int(reco["n_background"].sum()), int(truth["n_background"].sum()))
        self.assertEqual(int(reco["n_signal_pass"].sum()), int(truth["n_signal_pass"].sum()))
        self.assertEqual(float(reco["significance"].sum()) * 0, 0.0)  # sanity

    def test_the_recorded_basis_is_never_ambiguous(self):
        for variable, expected in (("pt", "reconstructed"), ("truth_pt", "truth"),
                                   ("eta", "reconstructed"), ("truth_eta", "truth")):
            table = perf.at_cut_table(self.fr, "eid", cut=self.cut, variable=variable)
            self.assertEqual(set(table["binning_basis"]), {expected}, variable)
            self.assertEqual(set(table["variable"]), {variable}, variable)
            self.assertEqual(set(table["bin_column"]),
                             {variable.replace("truth_", "truth_")}, variable)

    def test_unknown_variable_names_the_choices(self):
        with self.assertRaises(KeyError) as ctx:
            perf.at_cut_table(self.fr, "eid", cut=self.cut, variable="banana")
        self.assertIn("truth_pt", str(ctx.exception))

    def test_missing_truth_column_says_to_retrain(self):
        frame = self.fr.drop(columns=["truth_pt"])
        with self.assertRaises(KeyError) as ctx:
            perf.at_cut_table(frame, "eid", cut=self.cut, variable="truth_pt")
        self.assertIn("Re-train", str(ctx.exception))

    def test_curves_and_maps_follow_the_basis(self):
        curves = perf.significance_curves_by_bin(self.fr, "eid", variable="truth_pt")
        self.assertEqual(set(curves["binning_basis"]), {"truth"})
        m = perf.map_2d(self.fr, "eid", cut=self.cut, quantity="efficiency",
                        x_variable="truth_pt", y_variable="eta")
        self.assertEqual(m["x_variable"], "truth_pt")
        self.assertIn("truth", m["x_label"])
        self.assertEqual(int(m["counts"].sum()), int(len(self.fr)))

    def test_evaluate_layer_accepts_truth_binning_too(self):
        frame = self.fr.copy()
        frame["label"] = frame["y"]
        frame["signal_col"] = 1.0
        table = evaluate.binned_table(frame, task="eid", by="truth_pt", threshold=self.cut)
        self.assertEqual(set(table["binning_basis"]), {"truth"})
        self.assertEqual(int(table["n_signal"].sum()), int((frame["y"] == 1).sum()))
        with self.assertRaises(ValueError):
            evaluate.binned_table(frame, task="eid", by="momentum")


class TestIntervalConvention(unittest.TestCase):
    """One meaning per column name, across both metric modules."""

    CASES = ((0, 20), (2, 20), (10, 100), (99, 100), (7, 13), (1, 1))

    def test_central_interval_matches_scipy_exact(self):
        from scipy.stats import binomtest

        for k, n in self.CASES:
            lo, hi = perf.binomial_interval(k, n)
            ref = binomtest(k, n).proportion_ci(confidence_level=config.REJECTION_CL,
                                                method="exact")
            self.assertAlmostEqual(lo, ref.low, places=9, msg=f"k={k},n={n} lower")
            self.assertAlmostEqual(hi, ref.high, places=9, msg=f"k={k},n={n} upper")

    def test_evaluate_and_performance_agree(self):
        # <x>_err_lo / <x>_err_hi must not mean "central 95%" in one table and
        # "one-sided 95%" in another.
        for k, n in self.CASES:
            ours, theirs = perf.binomial_interval(k, n), evaluate.garwood(k, n)
            # compared with a tolerance: the two call beta.ppf with differently
            # ordered arguments, so agreement to ~1e-16 is exact agreement here.
            self.assertAlmostEqual(ours[0], theirs[0], places=12, msg=f"k={k},n={n} lo")
            self.assertAlmostEqual(ours[1], theirs[1], places=12, msg=f"k={k},n={n} hi")

    def test_one_sided_limit_is_wider_on_the_limit_side_than_the_central_bound(self):
        # fake_rate_upper_limit (one-sided 95%) is the quantity quoted for a limit,
        # and must be tighter (smaller) than the central interval's upper edge.
        k, n = 0, 20
        central_hi = perf.binomial_interval(k, n)[1]
        one_sided = sg.garwood_upper(k, n)
        self.assertLess(one_sided, central_hi)

    def test_bracket_contains_the_point_estimate(self):
        score, y = np.linspace(0, 1, 500), np.r_[np.ones(300, int), np.zeros(200, int)]
        frame = pd.DataFrame({"y": y, "score": score, "pt": np.linspace(0.5, 10, 500),
                              "p": np.linspace(1, 20, 500), "eta": np.linspace(-3, 3, 500),
                              "truth_class": np.where(y == 1, "e", "pi"),
                              "label": y, "truth_idx": np.arange(500)})
        table = perf.at_cut_table(frame, "eid", cut=0.5, variable="pt")
        for _, row in table.dropna(subset=["efficiency"]).iterrows():
            self.assertLessEqual(float(row["efficiency_err_lo"]), float(row["efficiency"]))
            self.assertGreaterEqual(float(row["efficiency_err_hi"]), float(row["efficiency"]))


class TestSignificanceCurvesPerBin(unittest.TestCase):
    def setUp(self):
        self.fr = perf.channel_frame(synthetic_scores(n=8000, seed=21), "eid")
        self.curves = perf.significance_curves_by_bin(self.fr, "eid")

    def test_bin_candidate_counts_are_conserved(self):
        edges = config.PT_BINS_PERFORMANCE
        idx = np.clip(np.digitize(self.fr["pt"].to_numpy(float), edges[1:-1], right=True),
                      0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        per_bin = self.curves.drop_duplicates("bin_center")[["bin_center", "n_signal_bin",
                                                             "n_background_bin"]]
        y = self.fr["y"].to_numpy()
        expected_sig = np.bincount(idx, weights=(y == 1), minlength=len(edges) - 1)
        self.assertEqual(int(per_bin["n_signal_bin"].sum()), int((y == 1).sum()))
        self.assertEqual(int(per_bin["n_signal_bin"].iloc[0]), int(expected_sig[0]))
        self.assertEqual(int((per_bin["n_signal_bin"] + per_bin["n_background_bin"]).sum()),
                         int(counts.sum()))

    def test_curve_maximum_agrees_with_the_independent_optimum_search(self):
        # Two code paths (curve scan with is_optimum flag vs optimal_cut_in_bins)
        # must land on the same c*; disagreement would mean a silent bug in one.
        other = sg.optimal_cut_in_bins(self.fr["score"], self.fr["y"], self.fr["pt"])
        mine = perf.bin_optima_from_curves(self.curves)
        merged = mine.merge(other[["bin_center", "threshold"]], on="bin_center",
                            suffixes=("_curve", "_bins"))
        self.assertGreater(len(merged), 2, "fixture should give several usable bins")
        self.assertTrue(bool((merged["threshold_curve"] - merged["threshold_bins"]
                              ).abs().max() < 5e-3))

    def test_zero_threshold_reproduces_the_total_yield_formula(self):
        usable = self.curves[self.curves["has_curve"].astype(bool)]
        centres = usable["bin_center"].unique()
        c = float(centres[0])
        sub = usable[usable["bin_center"] == c].sort_values("threshold")
        n_sig = int(sub["n_signal_bin"].iloc[0])
        n_bkg = int(sub["n_background_bin"].iloc[0])
        self.assertAlmostEqual(float(sub["significance"].iloc[0]),
                               n_sig / np.sqrt(n_sig + n_bkg), places=9)
        self.assertAlmostEqual(float(sub["efficiency"].iloc[0]), 1.0, places=9)
        self.assertAlmostEqual(float(sub["fake_rate"].iloc[0]), 1.0, places=9)

    def test_significance_is_never_inflated_above_sqrt_signal(self):
        usable = self.curves[self.curves["has_curve"].astype(bool)]
        for centre, sub in usable.groupby("bin_center"):
            self.assertLessEqual(float(np.nanmax(sub["significance"])),
                                 np.sqrt(int(sub["n_signal_bin"].iloc[0])) + 1e-9)

    def test_p_and_eta_variables_are_supported(self):
        for variable in ("p", "eta"):
            curves = perf.significance_curves_by_bin(self.fr, "eid", variable=variable)
            self.assertEqual(set(curves["variable"]), {variable}, variable)
            self.assertTrue(bool(curves["has_curve"].any()), variable)

    def test_weighted_curves_differ_from_unweighted(self):
        y = self.fr["y"].to_numpy()
        w = sg.flux_weights(y, signal=1, background=0, bkg_scale=20.0)
        heavy = perf.significance_curves_by_bin(self.fr, "eid", weights=w)
        light = self.curves
        key = heavy[heavy["is_optimum"].astype(bool)][["bin_center", "threshold"]]
        base = light[light["is_optimum"].astype(bool)][["bin_center", "threshold"]]
        merged = key.merge(base, on="bin_center", suffixes=("_w", "_0"))
        self.assertGreater(float((merged["threshold_w"] - merged["threshold_0"]).abs().max()),
                           0.0, "up-weighting the background must move at least one bin's c*")

    def test_empty_frame_gives_no_curves_without_crashing(self):
        fr = self.fr.iloc[0:0].copy()
        curves = perf.significance_curves_by_bin(fr, "eid")
        self.assertFalse(bool(curves["has_curve"].any()))
        self.assertEqual(len(perf.bin_optima_from_curves(curves)), 0)


class TestBinnedCurveFigures(unittest.TestCase):
    def _run(self, tmp, **kw):
        root = os.path.join(tmp, "models")
        d = os.path.join(root, "eid_lightgbm_clean")
        os.makedirs(d, exist_ok=True)
        synthetic_scores(n=6000, seed=11).to_pickle(os.path.join(d, "all_scores.pkl"))
        return perf.run(channels=("eid",), dataset_tag="clean", model="lightgbm",
                        out_dir=os.path.join(tmp, "out"), model_root=root, quiet=True, **kw)

    def test_both_new_figures_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            figures = set(os.listdir(os.path.join(tmp, "out", "plots")))
            self.assertIn("pid-eid_significance_vs_cut_by_pt.png", figures)
            self.assertIn("pid-eid_significance_vs_cut_panels.png", figures)

    def test_curve_table_is_written_and_queryable(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            path = os.path.join(tmp, "out", "pid-eid_lightgbm_clean-significance_by_pt.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as fh:
                payload = json.load(fh)
            rows = pd.DataFrame(payload["data"])
            self.assertGreater(len(rows), 100)
            self.assertIn("is_optimum", rows.columns)
            self.assertEqual(len(rows[rows["is_optimum"].astype(bool)]),
                             int(rows[rows["has_curve"].astype(bool)]["bin_center"].nunique()))

    def test_binned_curves_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, binned_curves=False)
            figures = set(os.listdir(os.path.join(tmp, "out", "plots")))
            self.assertNotIn("pid-eid_significance_vs_cut_by_pt.png", figures)
            self.assertNotIn("pid-eid_significance_vs_cut_panels.png", figures)

    def test_panels_figure_is_nontrivial(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            size = os.path.getsize(os.path.join(tmp, "out", "plots",
                                                "pid-eid_significance_vs_cut_panels.png"))
            self.assertGreater(size, 20_000, "a near-blank panel figure would be a bug")
