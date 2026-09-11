"""trkperf — ePIC craterlake tracking-performance analysis.

A small, modular package for measuring tracking performance (geometric
acceptance, momentum resolution, track-finding efficiency, fake rate) from
EDM4eic reconstruction output, and comparing a clean sample against a sample
with beam-induced background overlaid.

See AGENTS.md at the project root for the full data model, conventions, and
definition of done. This package implements exactly that specification.

Module map
----------
config.py     : all tunable constants in one place (species, bin edges,
                thresholds, dataset registry). Change behaviour here, not by
                editing the logic modules.
io.py         : generic uproot access + the jagged-array -> flat pandas
                DataFrame helper used by every other module. No physics.
truth.py      : MCParticles -> truth kinematics; truth-hit layer counts used
                by the acceptance metric.
reco.py       : CentralCKFTracks -> reconstructed kinematics.
matching.py   : CentralCKFTrackAssociations -> the truth<->reco matched-pairs
                table (the shared data contract every metric consumes) and
                the fake-track finder.
binning.py    : pT/eta bin assignment and the Gaussian-core fit helper.
metrics/      : one module per metric (acceptance, resolution, efficiency,
                fake_rate) — each takes a file list and returns a tidy
                DataFrame of results, independent of how it will be reported.
compare.py    : clean-vs-background comparison of two metric results.
report.py     : JSON / markdown-table / plot writers.
pid/          : empty on purpose — the extension point for a future
                particle-identification module (see pid/__init__.py).
cli.py        : `python -m trkperf <metric|compare> ...` command-line entry
                point gluing the above together.
"""

from __future__ import annotations

__version__ = "0.1.0"
