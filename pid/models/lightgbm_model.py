"""LightGBM adapter (the pipeline's primary learner)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import ModelAdapter


class LightGBMAdapter(ModelAdapter):
    name = "lightgbm"
    space = ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction",
             "bagging_fraction", "n_estimators", "lambda_l2")

    def estimator(self, params: dict, *, n_classes: int, seed: int):
        import lightgbm as lgb

        merged = dict(
            objective="binary" if n_classes <= 2 else "multiclass",
            n_jobs=config.N_JOBS,
            verbosity=-1,
            random_state=seed,
            importance_type="gain",
            # bagging must be enabled for bagging_fraction to have any effect
            bagging_freq=1 if "bagging_fraction" in params else 0,
        )
        if n_classes > 2:
            merged["num_class"] = n_classes
        merged.update({k: v for k, v in params.items() if v is not None})
        return lgb.LGBMClassifier(**merged)

    def gain(self, estimator, columns: list[str]) -> pd.Series:
        booster = getattr(estimator, "booster_", estimator)
        try:
            values = booster.feature_importance(importance_type="gain")
        except AttributeError:  # pragma: no cover - unfitted estimator
            return pd.Series(dtype=float)
        names = booster.feature_name()
        # LightGBM reports the names it was fitted with; map defensively so a
        # renamed/categorical column can never be attributed to the wrong feature.
        out = pd.Series(0.0, index=list(columns), dtype=float)
        for n, v in zip(names, np.asarray(values, dtype=float)):
            if n in out.index:
                out[n] = v
        return out.sort_values(ascending=False)

    def shap(self, estimator, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values, shape ``(n_samples, n_features)``.

        ``pred_contrib=True`` is LightGBM's built-in TreeSHAP; the last column it
        returns is the bias term and is dropped.
        """
        contrib = np.asarray(estimator.predict(X, pred_contrib=True))
        return contrib[:, : X.shape[1]]
