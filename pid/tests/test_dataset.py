"""Label policy, feature selection, leakage guards and splitting."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pid import config, dataset


def synthetic_table(n=600, seed=0):
    """A feature table shaped like the real one, with known truth."""
    rng = np.random.default_rng(seed)
    cls = rng.choice(["e", "pi", "K", "p"], size=n, p=[0.4, 0.4, 0.1, 0.1])
    # Eta (hence leg) is only CORRELATED with species, not determined by it - in
    # the real sample 1.5 % of tracks matched to the backward EMCal are pions, and
    # a fixture where leg == (cls == "e") makes every task single-class.
    eta = rng.choice([-2.4, 2.4], size=n, p=[0.45, 0.55])
    leg = np.where(eta < 0, "backward", "forward")
    ep = np.where(cls == "e", rng.normal(0.95, 0.08, n), rng.normal(0.28, 0.12, n))
    return pd.DataFrame({
        "file_id": rng.integers(0, 4, n), "event": rng.integers(0, 200, n),
        "track_idx": np.arange(n), "p": rng.uniform(1, 20, n),
        "pt": rng.uniform(1, 15, n), "eta": eta,
        "charge": np.where(cls == "e", -1.0, 1.0), "pdg": 0,
        "leg": leg, "truth_class": cls, "truth_pdg": 211, "truth_pt": 3.0,
        "truth_eta": 2.0, "assoc_weight": 1.0, "is_matched": True, "is_fake": False,
        "e_over_p_backward": ep, "log_e_over_p_backward": np.log10(np.clip(ep, 1e-3, None)),
        "e_over_p_forward": 0.3 + 0.01 * rng.normal(size=n),
        "ecal_backward_rms_r": rng.uniform(1, 5, n), "has_ecal_backward": 1,
        "shape_0": rng.uniform(0, 100, n), "irt_gas_npe_tot_evt": rng.uniform(0, 50, n),
        "n_tracks_evt": rng.integers(2, 20, n), "drich_gas_pathlength": rng.uniform(0, 1, n),
        "edep_si_mean": rng.uniform(0, 1, n), "has_ionisation": 1,
        # Present in the real tables; excluded by default because in these
        # productions it is a ~3 us per-event timestamp, not a per-track quantity.
        "track_time": rng.uniform(2991, 2999, n), "track_time_error": rng.uniform(5, 400, n),
        "theta": 1.0, "match_theta": 1.0, "match_phi": 0.0, "source_file": "x.root",
    })


class TestLabels(unittest.TestCase):
    def test_binary_label_is_e_positive(self):
        lmap = dataset.label_map("eid")
        self.assertEqual(lmap["e"], 1)
        self.assertEqual(lmap["pi"], 0)
        self.assertNotIn("K", lmap)

    def test_ehad_collapses_all_hadrons_to_zero(self):
        lmap = dataset.label_map("ehad")
        self.assertEqual({lmap[c] for c in ("pi", "K", "p")}, {0})
        self.assertEqual(lmap["e"], 1)

    def test_kinematics_are_opt_in(self):
        lean = dataset.model_columns(synthetic_table(), task="eid")
        with_kin = dataset.model_columns(synthetic_table(), task="eid",
                                         include_kinematics=True)
        for c in ("p", "pt", "eta"):
            self.assertNotIn(c, lean)
            self.assertIn(c, with_kin)

    def test_multiclass_order_is_the_task_class_list(self):
        # Regression: the label order must come from the task, not from
        # config.CLASS_LABELS positions, or proba_<class> columns get swapped.
        lmap = dataset.label_map("hadpid")
        self.assertEqual(sorted(lmap), ["K", "p", "pi"])
        self.assertEqual(lmap["pi"], 0)

    def test_prepare_filters_leg_momentum_and_class(self):
        df = synthetic_table()
        eid = dataset.prepare(df, "eid")
        self.assertTrue(set(eid["leg"]) == {"backward"})
        self.assertTrue((eid["p"] >= config.MIN_TRACK_P).all())
        self.assertTrue(set(eid["truth_class"]) <= {"e", "pi"})
        had = dataset.prepare(df, "hadpid")
        self.assertTrue(set(had["leg"]) == {"forward"})

    def test_unmatched_rows_only_survive_when_asked(self):
        df = synthetic_table()
        df.loc[df.index[:50], "is_matched"] = False
        matched = dataset.prepare(df, "eid", require_matched=True)
        self.assertFalse(matched["is_matched"].eq(False).any())
        both = dataset.prepare(df, "eid", require_matched=False)
        self.assertGreaterEqual(len(both), len(matched))


class TestFeatureSelection(unittest.TestCase):
    def test_truth_columns_are_never_features(self):
        cols = dataset.model_columns(synthetic_table(), task="eid")
        for c in cols:
            self.assertFalse(c.startswith("truth_"), c)
        for blocked in ("file_id", "event", "track_idx", "label", "assoc_weight",
                        "is_matched", "is_fake", "source_file"):
            self.assertNotIn(blocked, cols)

    def test_signed_charge_is_excluded_by_default(self):
        # Signed charge tags the beam, not the particle: in NC DIS (e- beam) the
        # scattered electron is always negative, so it would be learned as PID.
        cols = dataset.model_columns(synthetic_table(), task="eid")
        self.assertNotIn("charge", cols)
        with_charge = dataset.model_columns(synthetic_table(), task="eid", exclude=())
        self.assertIn("charge", with_charge)

    def test_event_level_columns_are_opt_in_not_default(self):
        # Default OFF: an occupancy column out-ShAPed E/p in the real sample, and
        # event activity differs systematically between clean and +background, so
        # it must never be part of the baseline comparison model.
        lean = dataset.model_columns(synthetic_table(), task="eid")
        full = dataset.model_columns(synthetic_table(), task="eid",
                                     include_event_level=True)
        self.assertNotIn("irt_gas_npe_tot_evt", lean)
        self.assertIn("irt_gas_npe_tot_evt", full)
        self.assertIn("e_over_p_backward", lean)
        self.assertLess(len(lean), len(full))

    def test_track_time_is_opt_in_too(self):
        # CentralCKFTracks.time spans 2991-2999 ns with a ~3 us per-event offset:
        # an event timestamp here, not a per-track measurement.
        lean = dataset.model_columns(synthetic_table(), task="eid")
        with_time = dataset.model_columns(synthetic_table(), task="eid",
                                          include_track_time=True)
        self.assertNotIn("track_time", lean)
        self.assertIn("track_time", with_time)

    def test_join_bookkeeping_never_becomes_a_feature(self):
        # Regression: `rec_idx` (a column left behind by the association merge)
        # once entered the matrix and outranked E/p on SHAP.
        df = synthetic_table()
        df["rec_idx"] = np.arange(len(df))
        df["some_new_join_row"] = 3
        df["cluster_begin"] = 1
        cols = dataset.model_columns(df, task="eid")
        for leak in ("rec_idx", "some_new_join_row", "cluster_begin", "p", "pt", "eta"):
            self.assertNotIn(leak, cols)

    def test_leg_prefixed_diagnostic_never_becomes_a_feature(self):
        # Regression: features.py attaches the cluster self-consistency check as
        # `<role>_<leg>_e_hit_over_e_clu`, which matched no exact/prefix/suffix
        # blocking rule and entered the matrix, although the base name is
        # declared never-an-input in KEY_COLUMNS.
        df = synthetic_table()
        df["cluster_N_e_hit_over_e_clu"] = 1.0
        df["cluster_P_e_hit_over_e_clu"] = 1.0
        cols = dataset.model_columns(df, task="eid")
        self.assertNotIn("cluster_N_e_hit_over_e_clu", cols)
        self.assertNotIn("cluster_P_e_hit_over_e_clu", cols)

    def test_single_class_measured_column_fails_loudly(self):
        # A column measured only for electrons has a missingness pattern that
        # perfectly predicts the label; corrcoef returns NaN there, which must
        # fail the gate rather than auto-pass under the threshold.
        n = 60
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": np.r_[rng.normal(size=n // 2), [np.nan] * (n // 2)]})
        y = np.r_[np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)]
        with self.assertRaises(AssertionError):
            dataset.assert_no_leakage(df, y)

    def test_ionisation_is_off_unless_requested(self):
        self.assertNotIn("edep_si_mean", dataset.model_columns(synthetic_table(), task="eid"))
        self.assertIn("edep_si_mean", dataset.model_columns(
            synthetic_table(), task="eid", include_ionisation=True))

    def test_leg_only_enters_the_pooled_task(self):
        self.assertNotIn("leg", dataset.model_columns(synthetic_table(), task="eid"))
        self.assertIn("leg", dataset.model_columns(synthetic_table(), task="pooled"))

    def test_column_order_is_deterministic(self):
        df = synthetic_table()
        a = dataset.model_columns(df, task="eid")
        b = dataset.model_columns(df.copy(), task="eid")
        self.assertEqual(a, sorted(a))
        self.assertEqual(a, b)

    def test_family_assignment(self):
        self.assertEqual(dataset.family_of("e_over_p_backward"), "calorimeter")
        self.assertEqual(dataset.family_of("drich_gas_pathlength"), "cherenkov")
        self.assertEqual(dataset.family_of("edep_si_mean"), "ionisation")
        self.assertEqual(dataset.family_of("n_tracks_evt"), "event_context")


class TestDesignMatrix(unittest.TestCase):
    def test_nan_is_preserved_never_imputed(self):
        df = synthetic_table()
        df.loc[df.index[:20], "e_over_p_backward"] = np.nan
        cols = dataset.model_columns(df, task="eid")
        X = dataset.design_matrix(df, cols)
        self.assertEqual(int(X["e_over_p_backward"].isna().sum()), 20)
        self.assertNotEqual(float(np.nanmin(X["e_over_p_backward"])), 0.0)

    def test_column_names_survive_into_the_matrix(self):
        df = synthetic_table()
        cols = dataset.model_columns(df, task="eid")
        self.assertEqual(list(dataset.design_matrix(df, cols).columns), cols)


class TestSplitting(unittest.TestCase):
    def test_multi_file_split_never_shares_a_file(self):
        df = synthetic_table()
        prep = dataset.prepare(df, "pooled")
        tr, te, test_groups = dataset.split_by_file(prep)
        files_tr = set(prep["file_id"].to_numpy()[tr])
        files_te = set(prep["file_id"].to_numpy()[te])
        self.assertFalse(files_tr & files_te, "a file appeared in train and test")
        self.assertEqual(files_te, set(test_groups))
        self.assertTrue(te.size and tr.size)

    def test_single_file_fallback_is_reported(self):
        df = synthetic_table()
        df["file_id"] = 0
        tr, te, groups = dataset.split_by_file(df)
        self.assertEqual(groups, ["<single-file: event split>"])
        # and events are not shared between sides
        self.assertFalse(set(df["event"].to_numpy()[tr]) & set(df["event"].to_numpy()[te]))

    def test_split_is_seeded_and_reproducible(self):
        df = synthetic_table()
        a = dataset.split_by_file(df, seed=7)[1]
        b = dataset.split_by_file(df, seed=7)[1]
        np.testing.assert_array_equal(a, b)


class TestLeakageGate(unittest.TestCase):
    def test_clean_matrix_passes(self):
        df = synthetic_table()
        prep = dataset.prepare(df, "eid")
        cols = dataset.model_columns(prep, task="eid")
        table = dataset.assert_no_leakage(dataset.design_matrix(prep, cols),
                                         prep["label"].to_numpy())
        self.assertLess(abs(float(table.iloc[0]["corr_with_label"])), 0.99)

    def test_injected_truth_column_is_caught(self):
        df = synthetic_table()
        prep = dataset.prepare(df, "eid")
        cols = dataset.model_columns(prep, task="eid")
        X = dataset.design_matrix(prep, cols)
        # A truth-derived column copied into the matrix must trip the gate.
        X["oops_truth_label"] = prep["label"].to_numpy(dtype=float)
        with self.assertRaises(AssertionError):
            dataset.assert_no_leakage(X, prep["label"].to_numpy())

    def test_single_class_input_is_rejected(self):
        df = synthetic_table()
        df["truth_class"] = "e"
        prep = df[df["p"] >= 1.0]
        with self.assertRaises(ValueError):
            dataset.assert_no_leakage(prep[["p"]], prep["p"].to_numpy() * 0)


class TestSummary(unittest.TestCase):
    def test_text_columns_are_not_reported_as_empty_features(self):
        df = synthetic_table()
        table = dataset.summary(df)
        empty = dataset.empty_numeric_columns(table)
        self.assertNotIn("leg", set(empty["column"]))
        self.assertNotIn("source_file", set(empty["column"]))

    def test_genuinely_empty_numeric_column_is_flagged(self):
        df = synthetic_table()
        df["all_nan_feature"] = np.nan
        table = dataset.summary(df)
        self.assertIn("all_nan_feature", set(dataset.empty_numeric_columns(table)["column"]))


if __name__ == "__main__":
    unittest.main()
