"""Particle-identification metrics.

confusion.py implements the PID confusion matrix: for each truth species,
the fraction classified as each reco species (row-normalised), binned in
(pT, eta). See confusion.py's docstring for details.

This subpackage is the *reconstruction's own* mass-hypothesis PID - it makes no
model decision. The machine-learning PID pipeline that PLAN.md §"Extending this
project" anticipated now lives in the top-level :mod:`pid` package (``pid/``,
``python -m pid``); it consumes :func:`trkperf.matching.build_matched_tracks`
exactly as promised here, and its results are meant to be read against
``pid-confusion`` as the baseline.
"""

from __future__ import annotations
