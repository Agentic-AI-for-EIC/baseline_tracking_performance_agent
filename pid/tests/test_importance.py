"""`pid.importance` must read gains/SHAP from the unwrapped booster.

Regression: train saves the calibrated wrapper as `model.joblib`, whose
`predict` accepts no SHAP kwargs (TypeError) and which exposes no booster
(XGBoost `get_booster` AttributeError, LightGBM silently empty gains). The
unwrapped fit is therefore persisted as `booster.joblib`, and the loader below
prefers it.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import joblib
import pandas as pd

from pid.importance import load_model_artifact


class TestLoadModelArtifact(unittest.TestCase):
    def test_prefers_booster_joblib(self):
        with tempfile.TemporaryDirectory() as d:
            joblib.dump({"kind": "booster"}, os.path.join(d, "booster.joblib"))
            joblib.dump({"kind": "model"}, os.path.join(d, "model.joblib"))
            obj, name = load_model_artifact(d)
            self.assertEqual(name, "booster.joblib")
            self.assertEqual(obj, {"kind": "booster"})

    def test_falls_back_to_model_joblib(self):
        with tempfile.TemporaryDirectory() as d:
            joblib.dump({"kind": "model"}, os.path.join(d, "model.joblib"))
            obj, name = load_model_artifact(d)
            self.assertEqual(name, "model.joblib")
            self.assertEqual(obj, {"kind": "model"})


class TestRestrictToTestFiles(unittest.TestCase):
    def test_keeps_only_held_out_files(self):
        from pid.importance import restrict_to_test_files

        df = pd.DataFrame({"file_id": [0, 0, 1, 1, 2, 2], "x": range(6)})
        out, scope = restrict_to_test_files(df, ["1", "2"])
        self.assertEqual(len(out), 4)
        self.assertIn("held-out", scope)
        self.assertTrue((out["file_id"] != 0).all())

    def test_no_match_falls_back_loudly(self):
        from pid.importance import restrict_to_test_files

        df = pd.DataFrame({"file_id": [0, 0], "x": [1, 2]})
        out, scope = restrict_to_test_files(df, ["9"])
        self.assertEqual(len(out), 2)
        self.assertIn("all rows", scope)

    def test_missing_record_keeps_everything(self):
        from pid.importance import restrict_to_test_files

        df = pd.DataFrame({"file_id": [0, 1], "x": [1, 2]})
        out, scope = restrict_to_test_files(df, None)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
