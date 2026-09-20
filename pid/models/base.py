"""Model adapters: one interface over LightGBM, XGBoost and sklearn HGB.

Every adapter exposes the same four things:

``estimator(params, n_classes, seed)``
    a scikit-learn-compatible classifier (``fit`` / ``predict_proba``),
``gain(estimator, columns)``
    the library's own split-gain importance,
``shap(estimator, X)``
    **exact** tree SHAP values (the ``shap`` package is not installed here, but
    both boosting libraries compute exact TreeSHAP internally), collapsed to
    one ``(n_samples, n_features)`` ranking by
    :func:`as_sample_feature_matrix`. Binary tasks only for gating: the
    LightGBM adapter keeps the first class block for multiclass objectives
    while the XGBoost adapter averages attribution over classes (documented
    asymmetry - see each adapter), so multiclass SHAP rankings are
    diagnostic, not gating (the hadpid train gate is gain-only by design -
    see PRIMARY_ATTRIBUTION).
``space``
    the hyper-parameter search grid.

Missing values are handed to the learners as ``NaN``: all three have native
missing-value handling (a learned default split direction), which is why
:func:`pid.dataset.design_matrix` never imputes.

No early stopping is configured anywhere: model selection is the grouped
``RandomizedSearchCV`` plus the train/test-AUC-gap gate in
:func:`pid.train.train`. An eval-set stop would need file-grouped splits to
stay honest, which is why the gap gate carries that load instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ModelAdapter:
    """Base class: adapters must supply the four methods above."""

    name: str = "base"
    #: columns of the search grid (see pid.config.SEARCH_SPACES)
    space: tuple[str, ...] = ()

    def estimator(self, params: dict, *, n_classes: int, seed: int):
        raise NotImplementedError

    def gain(self, estimator, columns: list[str]) -> pd.Series:
        raise NotImplementedError

    def shap(self, estimator, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    @staticmethod
    def _as_frame_importance(values, columns) -> pd.Series:
        s = pd.Series(values, index=list(columns[: len(values)]))
        return s.sort_values(ascending=False)


def objective(n_classes: int) -> tuple[str, str]:
    """``(objective, eval_metric)`` names for a given problem size."""
    if n_classes <= 2:
        return ("binary", "auc")
    return ("multiclass", "mlogloss")


def as_sample_feature_matrix(values) -> np.ndarray:
    """Collapse a SHAP array to ``(n_samples, n_features)``.

    Binary estimators return 2-D; multiclass ones may add a trailing class
    axis, over which attribution mass is averaged so one ranking covers the
    task. Callers must slice any bias column BEFORE calling (adapters do).
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 3:
        v = np.nanmean(v, axis=2)
    return v
