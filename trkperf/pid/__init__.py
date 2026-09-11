"""Particle-identification metrics.

confusion.py implements the PID confusion matrix: for each truth species,
the fraction classified as each reco species (row-normalised), binned in
(pT, eta). See confusion.py's docstring for details.
"""

from __future__ import annotations
