"""Comparison join keys and the ROOT TNtuple text encoding."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


from pid import compare as pid_compare
from pid import config, report


def _jsonable(df: pd.DataFrame) -> list[dict]:
    """NaN -> null, as pandas' encoder does, so the fixture matches real files."""
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def _write_metric(path: str, df: pd.DataFrame, **meta) -> None:
    with open(path, "w") as fh:
        json.dump({"meta": meta, "data": _jsonable(df)}, fh)


class TestCompareJoinKeys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pid_compare_")
        self.clean = os.path.join(self.tmp, "pid-eid_lightgbm_clean-overall.json")
        self.bkg = os.path.join(self.tmp, "pid-eid_lightgbm_bkg_mixed-overall.json")
        rows = [{"quantity": "auc", "signal": "e", "against": "pi", "target_fake_rate": np.nan,
                 "value": 0.99, "value_err": 0.01, "n_signal": 200, "n_background": 12},
                {"quantity": "efficiency", "signal": "e", "against": "pi", "target_fake_rate": 1e-3,
                 "value": 0.90, "value_err": 0.03, "n_signal": 200, "n_background": 12}]
        _write_metric(self.clean, pd.DataFrame(rows), campaign="26.02.0", n_files=150)
        # Same rows, degraded values, in the background order.
        rows_b = [dict(rows[1], value=0.70, value_err=0.05), dict(rows[0], value=0.93, value_err=0.02)]
        _write_metric(self.bkg, pd.DataFrame(rows_b), campaign="26.07.1", n_files=200)

    def test_overall_artifact_joins_on_quantity_and_target(self):
        df = pid_compare.compare_artifact("overall", task="eid", model="lightgbm",
                                          clean=self.clean, bkg=self.bkg, out_dir=self.tmp)
        self.assertEqual(len(df), 2)
        # Rows must be matched to each other, not to the other quantity: the AUC
        # pair compares 0.99 -> 0.93, the efficiency pair 0.90 -> 0.70.
        auc_row = df[(df["quantity"] == "auc") & (df["target_fake_rate"].isna())]
        self.assertEqual(len(auc_row), 1)
        self.assertAlmostEqual(float(auc_row["value_clean"].iloc[0]), 0.99)
        self.assertAlmostEqual(float(auc_row["value_bkg"].iloc[0]), 0.93)
        eff_row = df[df["quantity"] == "efficiency"]
        self.assertAlmostEqual(float(eff_row["value_ratio_bkg_over_clean"].iloc[0]),
                               0.70 / 0.90, places=6)

    def test_summary_records_both_campaigns(self):
        pid_compare.compare_artifact("overall", task="eid", model="lightgbm",
                                     clean=self.clean, bkg=self.bkg, out_dir=self.tmp)
        with open(os.path.join(self.tmp, "pid-eid_lightgbm_overall_comparison.summary.json")) as fh:
            summary = json.load(fh)
        self.assertEqual(summary["clean"]["campaign"], "26.02.0")
        self.assertEqual(summary["bkg_mixed"]["campaign"], "26.07.1")

    def test_nondefault_pair_is_named_and_labelled_by_its_tags(self):
        rows = [{"quantity": "auc", "signal": "e", "against": "pi", "target_fake_rate": np.nan,
                 "value": 0.99, "value_err": 0.01, "n_signal": 200, "n_background": 12}]
        clean = os.path.join(self.tmp, "pid-eid_lightgbm_clean26071-overall.json")
        bkg = os.path.join(self.tmp, "pid-eid_lightgbm_bkg26071-overall.json")
        _write_metric(clean, pd.DataFrame(rows), campaign="26.07.1", dataset_tag="clean26071")
        _write_metric(bkg, pd.DataFrame(rows), campaign="26.07.1", dataset_tag="bkg26071")
        pid_compare.compare_artifact("overall", task="eid", model="lightgbm",
                                     clean=clean, bkg=bkg, out_dir=self.tmp)
        # ...the pair carries its own suffix (the legacy name stays reserved
        # for the default pair) and its summary uses the real tags.
        self.assertFalse(os.path.exists(
            os.path.join(self.tmp, "pid-eid_lightgbm_overall_comparison.json")))
        stem = os.path.join(self.tmp,
                            "pid-eid_lightgbm_overall_clean26071-vs-bkg26071_comparison")
        self.assertTrue(os.path.exists(stem + ".json"))
        with open(stem + ".summary.json") as fh:
            summary = json.load(fh)
        self.assertIn("clean26071", summary)
        self.assertIn("bkg26071", summary)

    def test_unknown_artifact_refuses_instead_of_guessing_keys(self):
        with self.assertRaises(SystemExit) as ctx:
            pid_compare.compare_artifact("mystery", task="eid", model="lightgbm",
                                         clean=self.clean, bkg=self.bkg, out_dir=self.tmp)
        self.assertIn("JOIN_KEYS", str(ctx.exception))

    def test_missing_other_side_names_the_missing_file(self):
        with self.assertRaises(SystemExit) as ctx:
            pid_compare.compare_artifact("overall", task="eid", model="nope",
                                         clean=self.clean,
                                         bkg=os.path.join(self.tmp, "does-not-exist.json"),
                                         out_dir=self.tmp)
        self.assertIn("missing the background-side input", str(ctx.exception))


class TestRootEncoding(unittest.TestCase):
    def test_text_columns_become_codes_with_a_legend(self):
        df = pd.DataFrame({"quantity": ["auc", "n_sigma", "efficiency"],
                           "signal": ["e", "e", "pi"],
                           "value": [0.9, 1.2, 0.8], "n": [10, 20, 30]})
        numeric, legend = report.encode_for_root(df)
        self.assertNotIn("quantity", numeric.columns)
        self.assertIn("quantity_code", numeric.columns)
        self.assertEqual(len(legend["quantity"]), 3)
        # codes must be stable for a given set of labels, and -1 for missing
        again, _ = report.encode_for_root(df)
        self.assertEqual(numeric["quantity_code"].tolist(), again["quantity_code"].tolist())
        self.assertEqual(numeric["n"].tolist(), [10.0, 20.0, 30.0])

    def test_nan_survives_the_encoding(self):
        df = pd.DataFrame({"value": [0.5, np.nan], "flag": [True, False]})
        numeric, _ = report.encode_for_root(df)
        self.assertTrue(np.isnan(float(numeric["value"].iloc[1])))
        self.assertEqual(numeric["flag"].tolist(), [1.0, 0.0])

    def test_write_produces_all_three_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "pid-x-overall")
            df = pd.DataFrame({"quantity": ["auc"], "value": [0.9], "value_err": [0.01],
                               "insufficient_stats": [False]})
            report.write(df, stem, tree_name="pid_x_overall", meta={"task": "eid"})
            for suffix in (".json", ".md", ".root"):
                self.assertTrue(os.path.exists(stem + suffix), suffix)
            with open(stem + ".json") as fh:
                meta = json.load(fh)["meta"]
            self.assertEqual(meta["task"], "eid")
            self.assertEqual(meta["schema_version"], "pid-v1")
            self.assertEqual(meta["match_weight_threshold"], config.MATCH_WEIGHT_THRESHOLD)
            self.assertIn("lightgbm", meta["libraries"])

    def test_provenance_stamp_carries_the_numpy_that_was_resolved(self):
        stamp = report.stamp()
        # The ML package directory shadows numpy; without this in the metadata a
        # container change would silently alter numeric behaviour.
        self.assertTrue(stamp["numpy_path"])
        self.assertIn("schema_version", stamp)


if __name__ == "__main__":
    unittest.main()
