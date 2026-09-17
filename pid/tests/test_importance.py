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


if __name__ == "__main__":
    unittest.main()
