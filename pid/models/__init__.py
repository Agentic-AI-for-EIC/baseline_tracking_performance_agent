"""Model registry: ``get_adapter("lightgbm" | "xgboost" | "sklearn_hgb")``.

The three learners are deliberately independent implementations of the same
task. Agreement between them is the check that a performance number is a property
of the *features and sample*, not of one optimiser's quirks (PLAN_pid.md M4).
"""

from __future__ import annotations

from .base import ModelAdapter, as_sample_feature_matrix, objective
from .lightgbm_model import LightGBMAdapter
from .sklearn_hgb import SklearnHGBAdapter
from .xgboost_model import XGBoostAdapter

_ADAPTERS: dict[str, ModelAdapter] = {
    LightGBMAdapter.name: LightGBMAdapter(),
    XGBoostAdapter.name: XGBoostAdapter(),
    SklearnHGBAdapter.name: SklearnHGBAdapter(),
}


def get_adapter(name: str) -> ModelAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model library {name!r}; choices {sorted(_ADAPTERS)}") from exc


def available() -> list[str]:
    return sorted(_ADAPTERS)


__all__ = ["get_adapter", "available", "ModelAdapter", "objective",
           "as_sample_feature_matrix",
           "LightGBMAdapter", "XGBoostAdapter", "SklearnHGBAdapter"]
