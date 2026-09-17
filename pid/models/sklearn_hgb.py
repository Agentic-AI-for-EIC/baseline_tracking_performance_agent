"""scikit-learn HistGradientBoosting adapter (dependency-light third arm)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import ModelAdapter


class SklearnHGBAdapter(ModelAdapter):
    name = "sklearn_hgb"
    space = ("max_depth", "learning_rate", "min_samples_leaf", "max_leaf_nodes",
             "l2_regularization", "max_iter")

    def estimator(self, params: dict, *, n_classes: int, seed: int):
        from sklearn.ensemble import HistGradientBoostingClassifier

        merged = dict(random_state=seed, early_stopping=False, max_bins=255,
                      **({} if config.N_JOBS == 1 else {"n_jobs": config.N_JOBS}))
        merged.update({k: v for k, v in params.items() if v is not None})
        return HistGradientBoostingClassifier(**merged)

    def gain(self, estimator, columns: list[str]) -> pd.Series:
        """Total weighted split gain per feature, from the fitted trees.

        sklearn has no public importance API for HGB; ``_predictors`` carries the
        node arrays including ``sum_weighted_gain``, which is the same quantity
        LightGBM/XGBoost report as "gain". Reading a private attribute is
        acceptable here because this arm is a cross-check, and the access is
        guarded: if a future sklearn changes the layout, the adapter returns
        zeros and the caller falls back to permutation importance instead of
        crashing a finished training run.
        """
        out = pd.Series(0.0, index=list(columns), dtype=float)
        try:
            for layer in estimator._predictors:
                for tree in layer:
                    nodes = tree.nodes
                    if "feature_idx" not in nodes.dtype.names or "sum_weighted_gain" not in nodes.dtype.names:
                        return out * np.nan
                    for idx, gain in zip(nodes["feature_idx"], nodes["sum_weighted_gain"]):
                        # leaf marker is -2 in sklearn's node array
                        if 0 <= idx < len(columns):
                            out.iloc[int(idx)] += float(gain)
        except (AttributeError, IndexError):  # pragma: no cover - layout changed
            return out * np.nan
        return out.sort_values(ascending=False)

    def shap(self, estimator, X: pd.DataFrame) -> np.ndarray:
        """Not available: use permutation importance for this arm."""
        raise NotImplementedError(
            "pid.models.sklearn_hgb: HistGradientBoosting has no built-in TreeSHAP. "
            "Use `pid importance --kind permutation`, or the lightgbm/xgboost models, "
            "for SHAP values.")
