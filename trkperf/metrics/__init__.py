"""One module per tracking-performance metric.

Every metric module exposes a single `compute_<metric>(file_paths, ...)`
function that returns a tidy pandas DataFrame of per-bin results (one row
per (pt_bin, eta_bin[, species])), independent of how the result will be
reported (see report.py) or compared across dataset types (see compare.py).

Adding a new metric means adding a new file here with the same shape of
function - nothing in cli.py, report.py, or compare.py needs metric-specific
special-casing beyond registering the new function (see cli.py's
`METRIC_FUNCTIONS` dict).
"""

from __future__ import annotations
