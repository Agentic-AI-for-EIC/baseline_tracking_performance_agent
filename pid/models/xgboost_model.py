"""XGBoost adapter (twin arm: same task, independent learner implementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    name = "xgboost"
    space = ("max_depth", "learning_rate", "min_child_weight", "subsample",
             "colsample_bytree", "n_estimators", "reg_lambda")

    def estimator(self, params: dict, *, n_classes: int, seed: int):
        import xgboost as xgb

        objective, eval_metric = ("binary:logistic", "auc") if n_classes <= 2 else (
            "multi:softprob", "mlogloss")
        merged = dict(
            objective=objective, eval_metric=eval_metric, tree_method="hist",
            n_jobs=config.N_JOBS, random_state=seed, verbosity=0, enable_categorical=True,
        )
        if n_classes > 2:
            merged["num_class"] = n_classes
        merged.update({k: v for k, v in params.items() if v is not None})
        return xgb.XGBClassifier(**merged)

    def gain(self, estimator, columns: list[str]) -> pd.Series:
        booster = estimator.get_booster()
        scores = booster.get_score(importance_type="gain")
        names = booster.feature_names
        out = pd.Series(0.0, index=list(columns), dtype=float)
        for key, value in scores.items():
            if names and key in names:
                col = columns[names.index(key)]
            elif key.lstrip("f").isdigit():
                idx = int(key[1:])
                col = columns[idx] if idx < len(columns) else None
            else:
                col = key if key in out.index else None
            if col is not None:
                out[col] += float(value)
        return out.sort_values(ascending=False)

    def shap(self, estimator, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP contributions (XGBoost's ``pred_contribs``).

        Binary: ``(n, F+1)`` with a trailing bias column, dropped here.
        Multiclass: measured ``(n, C, F+1)`` - drop each class's bias column,
        then average attribution mass over classes so one ranking covers the
        task. Anything else raises rather than silently misattribute.
        """
        import xgboost as xgb

        booster = estimator.get_booster()
        dm = xgb.DMatrix(X, missing=np.nan, feature_names=list(X.columns))
        contrib = np.asarray(booster.predict(dm, pred_contribs=True))
        n_features = X.shape[1]
        if contrib.ndim == 3:
            if contrib.shape[2] != n_features + 1:
                raise ValueError(
                    "pid.models.xgboost.shap: unexpected multiclass "
                    f"pred_contribs shape {contrib.shape} for {n_features} "
                    "features - refusing to guess the class axis")
            return contrib[:, :, :n_features].mean(axis=1)
        return contrib[:, :n_features]
