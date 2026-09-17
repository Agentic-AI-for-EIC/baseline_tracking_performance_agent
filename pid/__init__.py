"""ML Particle Identification pipeline for ePIC craterlake.

This package is the PID extension of :mod:`trkperf`. It deliberately contains
**no** ROOT reading code of its own: every branch access goes through
:mod:`trkperf.io` / :mod:`trkperf.truth` / :mod:`trkperf.reco` /
:mod:`trkperf.matching` (AGENTS.md: "extend the package instead of writing
bespoke one-off file I/O"), so PID is measured against exactly the same
truth<->reco matching convention as the efficiency / resolution / fake-rate
results it is reported alongside.

Module map
----------
:mod:`pid.config`     every tunable number (classes, legs, thresholds, paths).
:mod:`pid.schema`     ``FEATURE -> {campaign: branch}`` alias table, the
                      verified dead-end registry, and the ML environment
                      preflight.
:mod:`pid.links`      collectionID registry from ``podio_metadata``, the
                      verified track<->detector joins, and the DeltaR matcher.
:mod:`pid.features`   per-file feature-table builder (+ disk cache).
:mod:`pid.dataset`    label policy, missing-value policy, leak-safe splits.
:mod:`pid.models`     LightGBM / XGBoost / sklearn HGB behind one interface.
:mod:`pid.train`      hyper-parameter search, cross-validation, artifacts.
:mod:`pid.evaluate`   ROC/AUC, efficiency at fixed fake rate, n-sigma,
                      confusion matrices, calibration.
:mod:`pid.importance` gain, exact SHAP, permutation importance.
:mod:`pid.compare`    clean-vs-bkg comparison via :mod:`trkperf.compare`.
:mod:`pid.cli`        ``python -m pid <command>``.
"""

from __future__ import annotations

__all__ = [
    "config",
    "schema",
    "links",
    "features",
    "dataset",
    "models",
    "train",
    "evaluate",
    "importance",
    "compare",
    "report",
    "plots",
    "cli",
]
