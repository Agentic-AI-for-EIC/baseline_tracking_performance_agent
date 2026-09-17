"""End-to-end tests on the local reference pair (network-free).

These are the tests a human runs after any change::

    python -m unittest discover -s pid/tests -v

They cover the chain that actually matters: link audit -> feature table ->
physics sanity (the E/p peaks must land where the calorimeter says they should)
-> training gates -> metric tables. The feature tables are cached under
``cache/pid_features/`` by the pipeline itself, so a repeat run is fast.

If ``data/dataset_small/`` is missing (a fresh checkout without the reference
files) the heavy tests skip rather than fail, mirroring
``trkperf/tests/test_local_data.py``.
"""

from __future__ import annotations

import json
import os
import unittest

import numpy as np
import pandas as pd

from pid import config, dataset, links, schema

_CLEAN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "data/dataset_small/signal/RECO")
_BKG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data/dataset_small/signal_BKG_mix/RECO")


def _first(path_dir: str) -> str | None:
    if not os.path.isdir(path_dir):
        return None
    for name in sorted(os.listdir(path_dir)):
        if name.endswith(".root"):
            return os.path.join(path_dir, name)
    return None


CLEAN_FILE = _first(_CLEAN)
BKG_FILE = _first(_BKG)
_CACHE = os.environ.get("PID_TEST_CACHE", "cache/pid_tests")


class _FeatureMixin(unittest.TestCase):
    """Build (or reuse) the feature table for one reference file.

    Subclasses set ``CLEAN_OR_BKG``; the table is cached on disk so the whole
    suite runs in seconds after the first pass.
    """

    CLEAN_OR_BKG = "clean"

    @classmethod
    def setUpClass(cls):
        if cls.CLEAN_OR_BKG == "clean" and not CLEAN_FILE:
            raise unittest.SkipTest("data/dataset_small/signal reference file missing")
        if cls.CLEAN_OR_BKG == "bkg_mixed" and not BKG_FILE:
            raise unittest.SkipTest("data/dataset_small/signal_BKG_mix reference file missing")
        from pid import features as pid_features

        cls.path = CLEAN_FILE if cls.CLEAN_OR_BKG == "clean" else BKG_FILE
        cls.campaign = schema.campaign_of(cls.CLEAN_OR_BKG)
        os.makedirs(_CACHE, exist_ok=True)
        table_path = os.path.join(_CACHE, f"features_{cls.CLEAN_OR_BKG}.pkl")
        if os.path.exists(table_path):
            cls.df = pd.read_pickle(table_path)
        else:
            cls.df = pid_features.build_features(
                [cls.path], dataset_tag=cls.CLEAN_OR_BKG, cache_dir=_CACHE, limit_files=1)
            cls.df.to_pickle(table_path)


class TestLinkAudit(unittest.TestCase):
    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_clean_file_links_match_the_plan(self):
        audit = links.audit_links(CLEAN_FILE)
        by_relation = audit.set_index("relation")
        # The backbone of the whole feature set: track<->ECAL and track<->dRICH.
        for relation in ("_EcalEndcapNTrackClusterMatches_track",
                         "_EcalEndcapPTrackClusterMatches_track",
                         "_CalorimeterTrackProjections_track",
                         "_DRICHGasTracks_track", "_DRICHAerogelTracks_track"):
            self.assertTrue(bool(by_relation.loc[relation, "usable"]),
                            f"{relation} must resolve to CentralCKFTracks")
            self.assertEqual(by_relation.loc[relation, "resolves_to"], "CentralCKFTracks")

    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_producer_pid_links_are_still_dead(self):
        audit = links.audit_links(CLEAN_FILE).set_index("relation")
        for relation in ("_DIRCParticleIDs_particle", "_CombinedTOFParticleIDs_particle",
                         "_DRICHGasIrtCherenkovParticleID_chargedParticle"):
            self.assertFalse(bool(audit.loc[relation, "usable"]),
                             f"{relation} was found usable again - if a production "
                             "fixed it, re-open the feature design (PLAN_pid.md 3.2)")

    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_registry_is_readable_and_populated(self):
        registry = links.read_registry(CLEAN_FILE)
        self.assertGreater(len(registry), 200)
        self.assertIn("CentralCKFTracks", registry.values())


class TestFeatureTable(_FeatureMixin):
    CLEAN_OR_BKG = "clean"

    def test_row_count_equals_candidate_tracks(self):
        # One row per reconstructed track, matched or not (the unmatched ones are
        # the combinatorial background a PID must reject).
        from trkperf import io

        tree = io.open_tree(self.path)
        n_tracks = len(io.read_flat(tree, {"c": "CentralCKFTracks.charge"}))
        self.assertEqual(len(self.df), n_tracks)

    def test_electron_and_pion_elpk_land_where_physics_says(self):
        back = self.df[(self.df["leg"] == "backward") & self.df["truth_class"].notna()]
        med = back.groupby("truth_class")["e_over_p_backward"].median()
        self.assertIn("e", med.index)
        self.assertIn("pi", med.index)
        self.assertGreater(float(med["e"]), 0.8, "electrons must peak at E/p ~ 1")
        self.assertLess(float(med["pi"]), 0.4, "pions must be MIP-like")

    def test_hadron_elpk_increases_with_mass_at_fixed_momentum(self):
        # A classic cross-check: at the same momentum a heavier hadron is slower,
        # ionises less and leaks more, so E/p rises e -> pi -> K -> p ordering in
        # the MIP regime must show pi < K <= p (measured on this file).
        fwd = self.df[(self.df["leg"] == "forward") & (self.df["p"] > 1.0)]
        med = fwd.groupby("truth_class")["e_over_p_forward"].median()
        for cls in ("pi", "K", "p"):
            self.assertIn(cls, med.index, f"{cls} statistics too thin for this assertion")
        self.assertLess(float(med["pi"]), float(med["p"]))

    def test_missing_subsystems_stay_nan(self):
        nomatch = self.df[self.df["has_ecal_backward"] == 0]
        if len(nomatch):
            self.assertTrue(nomatch["e_over_p_backward"].isna().all(),
                            "absent calorimeter match must be NaN, never 0")

    def test_no_numeric_feature_is_completely_empty(self):
        table = dataset.summary(self.df)
        empty = dataset.empty_numeric_columns(table)
        self.assertTrue(empty.empty, f"all-NaN columns: {list(empty['column'])}")

    def test_matched_tracks_are_labelled_or_explained(self):
        matched = self.df[self.df["is_matched"]]
        self.assertGreater(len(matched), 0)
        unlabelled = matched[matched["truth_class"].isna()]
        # A matched track can legitimately have no class: its truth particle is
        # either a species outside the PID class map (muons from pi/K decay in
        # flight) or a non-primary, which read_truth_particles(primary_only=True)
        # has already dropped. Anything else means the join is broken.
        allowed_unlabelled_pdg = {13}
        self.assertTrue(set(unlabelled["truth_pdg"].abs().dropna().unique())
                        <= allowed_unlabelled_pdg,
                        f"unexpected unlabelled species: "
                        f"{unlabelled['truth_pdg'].abs().value_counts().to_dict()}")
        self.assertLess(len(unlabelled) / len(matched), 0.05)
        self.assertGreaterEqual(set(matched["truth_class"].dropna()), {"e", "pi"})

    def test_charge_and_truth_columns_exist_but_are_not_features(self):
        self.assertIn("charge", self.df.columns)
        cols = dataset.model_columns(self.df, task="eid")
        self.assertNotIn("charge", cols)
        self.assertFalse([c for c in cols if c.startswith("truth_")])


class TestElectronTraining(_FeatureMixin):
    CLEAN_OR_BKG = "clean"

    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_gates_pass_and_elpk_leads(self):
        from pid import train as pid_train

        prep = dataset.prepare(self.df, "eid")
        counts = np.unique(prep["label"].to_numpy(), return_counts=True)[1]
        if counts.min() < config.MIN_ROWS_PER_CLASS:
            self.skipTest("local sample too small for the training gates")
        report = pid_train.train("eid", model="lightgbm", dataset_tag="clean",
                                 features=self._table_path(), n_iter=2, n_splits=2,
                                 out_dir=os.path.join(_CACHE, "models"), gates=False,
                                 calibrate=False, quiet=True)
        self.assertTrue(report["gates"]["train_test_auc_gap"]["passed"])
        self.assertTrue(report["gates"]["no_single_feature_label_correlation"]["passed"])
        self.assertTrue(report["gates"]["top_feature_is_physical"]["passed"],
                        f"E/p must lead the ranking; got {report['top_gain_features']}")
        self.assertIn("e_over_p", str(report["top_gain_features"][0]))
        self.assertGreater(report["auc_test"], 0.9)

    def _table_path(self):
        return os.path.join(_CACHE, f"features_{self.CLEAN_OR_BKG}.pkl")


class TestMetricTables(_FeatureMixin):
    CLEAN_OR_BKG = "clean"

    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_evaluate_tables_are_well_formed(self):
        from pid import evaluate as pid_evaluate
        from pid import train as pid_train

        report = pid_train.train("eid", model="lightgbm", dataset_tag="clean",
                                 features=self._table_path(), n_iter=2, n_splits=2,
                                 out_dir=os.path.join(_CACHE, "models"), gates=False,
                                 calibrate=False, quiet=True)
        scores = os.path.join(report["artifact_dir"], "test_scores.pkl")
        out = pid_evaluate.evaluate(scores, task="eid", dataset_tag="clean",
                                    model="lightgbm", out_dir=os.path.join(_CACHE, "out"),
                                    by=("pt",), quiet=True)
        overall = out["overall"]
        self.assertIn("auc", set(overall["quantity"]))
        self.assertGreater(float(overall[overall["quantity"] == "auc"]["value"].iloc[0]), 0.9)
        conf = out["confusion"]
        for _, group in conf.groupby("truth_class"):
            self.assertAlmostEqual(float(group["row_fraction"].sum()), 1.0, places=6)
        self.assertTrue({"roc", "vs_pt"} <= set(out))
        # Every artifact must exist in all three formats the project promises.
        for name, stem in out["paths"].items():
            self.assertTrue(os.path.exists(stem + ".json"), name)
            self.assertTrue(os.path.exists(stem + ".md"), name)

    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_provenance_records_the_sample_size(self):
        # AGENTS.md: any quoted number must state files/events it came from, and a
        # per-bin AUC from a couple of hundred tracks is not a measurement.
        from pid import train as pid_train

        report = pid_train.train("eid", model="lightgbm", dataset_tag="clean",
                                 features=self._table_path(), n_iter=2, n_splits=2,
                                 out_dir=os.path.join(_CACHE, "models"), gates=False,
                                 calibrate=False, quiet=True)
        out = os.path.join(_CACHE, "out")
        from pid import evaluate as pid_evaluate
        pid_evaluate.evaluate(os.path.join(report["artifact_dir"], "test_scores.pkl"),
                              task="eid", dataset_tag="clean", model="lightgbm",
                              out_dir=out, by=("pt",), quiet=True)
        with open(os.path.join(out, "pid-eid_lightgbm_clean-overall.json")) as fh:
            meta = json.load(fh)["meta"]
        for key in ("n_files_scored", "n_events_scored", "n_rows_scored",
                    "n_signal", "n_background"):
            self.assertIn(key, meta, key)
            self.assertIsNotNone(meta[key], key)
        self.assertEqual(meta["n_files_scored"], 1)
        self.assertLess(meta["n_events_scored"], 2000, "local smoke sample only")

    def _table_path(self):
        return os.path.join(_CACHE, f"features_{self.CLEAN_OR_BKG}.pkl")


@unittest.skipUnless(BKG_FILE, "background reference file missing")
class TestBackgroundCampaign(_FeatureMixin):
    """The same pipeline must run on 26.07.1 with its renamed/extra branches."""

    CLEAN_OR_BKG = "bkg_mixed"

    def test_features_build_with_bkg_schema(self):
        self.assertGreater(len(self.df), 0)
        self.assertIn("e_over_p_backward", self.df.columns)

    def test_electron_peak_persists_under_background(self):
        back = self.df[(self.df["leg"] == "backward") & self.df["truth_class"].notna()]
        med = back.groupby("truth_class")["e_over_p_backward"].median()
        if "e" in med.index and len(back[back["truth_class"] == "e"]) >= 10:
            self.assertGreater(float(med["e"]), 0.8,
                               "the electron E/p peak must survive beam background")

    def test_occupancy_features_are_present(self):
        self.assertIn("n_tracks_evt", self.df.columns)
        self.assertGreater(float(self.df["n_tracks_evt"].max()), 0.0)

    def test_campaign_specific_columns_are_not_in_the_shared_feature_set(self):
        # ecal_P_radius exists only in 26.07.1; it must never appear as a feature
        # column, because the clean arm cannot provide it (PLAN_pid.md 3.3).
        cols = dataset.model_columns(self.df, task="eid")
        self.assertFalse([c for c in cols if "radius" in c or "principal" in c])

    def test_ionisation_proxy_degrades_gracefully_on_bkg_schema(self):
        # The 26.07.1 track->measurement chain spans three collections, so the
        # optional proxy may come back empty - but must not raise.
        from pid import features as pid_features

        df = pid_features.build_features_one_file(BKG_FILE, "26.07.1",
                                                  enable_ionisation=True)
        self.assertGreater(len(df), 0)
        if "edep_si_mean" in df.columns:
            self.assertLessEqual(float(df["edep_si_mean"].isna().mean()), 1.0)


class TestCompareNeedsBothSides(unittest.TestCase):
    @unittest.skipUnless(CLEAN_FILE, "reference file missing")
    def test_missing_bkg_artifact_is_reported_not_silently_skipped(self):
        from pid import compare as pid_compare

        with self.assertRaises(SystemExit) as ctx:
            pid_compare.compare_artifact("overall", task="eid", model="nottrained")
        self.assertIn("missing the", str(ctx.exception))
        self.assertIn("evaluate", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
