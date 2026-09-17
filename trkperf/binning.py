"""Bin assignment and the Gaussian-core fit used by the resolution metric.

Binning and fitting are kept separate from the metric modules so both can be
tested/reused independently (e.g. a future PID module will want the same
pT/eta binning without needing anything from resolution.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def _root():
    """Lazy-import ROOT (a heavy dependency) on first use.

    Kept behind a function so binning.py's non-fit parts (bin assignment,
    binomial error) import and run without PyROOT, which is only needed by
    the resolution fit.
    """
    import ROOT

    ROOT.gROOT.SetBatch(True)  # headless: never pop up a canvas
    return ROOT


#: Monotonic id used to give every ROOT histogram/function a unique name.
#: gROOT caches objects by name; reusing the same name for a later histogram
#: in the same process silently returns the *stale* first object (classic ROOT
#: gotcha), so each fit pass needs its own name.
_ROOT_OBJECT_COUNTER = [0]


def _next_root_name(base: str) -> str:
    _ROOT_OBJECT_COUNTER[0] += 1
    return f"{base}_{_ROOT_OBJECT_COUNTER[0]}"


def assign_bins(
    df: pd.DataFrame,
    pt_col: str = "pt",
    eta_col: str = "eta",
    pt_edges: np.ndarray = config.PT_BIN_EDGES,
    eta_edges: np.ndarray = config.ETA_BIN_EDGES,
) -> pd.DataFrame:
    """Add categorical pt_bin/eta_bin columns, plus their numeric centers.

    Rows falling outside [pt_edges[0], pt_edges[-1]] or
    [eta_edges[0], eta_edges[-1]] get pt_bin/eta_bin = NaN (pandas' default
    pd.cut behaviour) - callers should drop these before computing a metric,
    they are out of the analysis' defined phase space, not a bug.
    """
    out = df.copy()
    out["pt_bin"] = pd.cut(out[pt_col], bins=pt_edges)
    out["eta_bin"] = pd.cut(out[eta_col], bins=eta_edges)
    # astype(float): on an empty frame Series.apply preserves the input's
    # categorical dtype, which would poison every downstream fillna/merge.
    out["pt_bin_center"] = out["pt_bin"].apply(
        lambda i: i.mid if pd.notna(i) else np.nan
    ).astype(float)
    out["eta_bin_center"] = out["eta_bin"].apply(
        lambda i: i.mid if pd.notna(i) else np.nan
    ).astype(float)
    return out


def _bin_categories(edges: np.ndarray) -> pd.IntervalIndex:
    """The exact categoricals ``assign_bins`` cuts with.

    ``pd.cut`` rounds breakpoints to ``precision=3`` decimals, so rebuilding
    categories from the raw edges (e.g. via ``from_breaks``) yields
    *different* intervals that would never merge-match. Deriving them through
    ``pd.cut`` itself keeps grid and result keys identical by construction.
    """
    return pd.cut(pd.Series([float(edges[0])]), bins=np.asarray(edges)).cat.categories


def full_bin_grid(species: list[str] | None) -> pd.DataFrame:
    """Cartesian species x pt x eta grid (categorical bins + centers).

    A groupby only yields observed bins, so bins with zero entries would be
    silently absent from a metric table. AGENTS.md requires those reported as
    insufficient statistics instead, so every metric right-joins its counts
    onto this grid. ``species=None`` means "no species axis" (fake rate).
    The categoricals are built from the same edges ``assign_bins`` cuts with,
    so the merge keys match exactly.
    """
    pt = _bin_categories(config.PT_BIN_EDGES)
    eta = _bin_categories(config.ETA_BIN_EDGES)
    recs = []
    specs: list = [None] if species is None else list(species)
    for s in specs:
        for pi in pt:
            for ei in eta:
                recs.append((s, pi, ei, pi.mid, ei.mid))
    grid = pd.DataFrame(
        recs, columns=["species", "pt_bin", "eta_bin", "pt_bin_center", "eta_bin_center"]
    )
    if species is None:
        grid = grid.drop(columns=["species"])
    grid["pt_bin"] = pd.Categorical(grid["pt_bin"], categories=pt)
    grid["eta_bin"] = pd.Categorical(grid["eta_bin"], categories=eta)
    return grid


def gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    """Plain Gaussian, no offset - the resolution fit is signal-only (a
    truth-matched residual distribution, not a peak over background).

    Kept for symmetry with the ROOT "gaus" TF1 formula
    (A * exp(-0.5*((x-mu)/sigma)^2)); the actual fit is done by ROOT.
    """
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


@dataclass
class GaussianFitResult:
    """Result of :func:`fit_gaussian_core`, including the diagnostics needed
    to judge whether the fit should be trusted (AGENTS.md Definition of
    done: chi2/ndf of order 1)."""

    n_entries: int
    mu: float
    sigma: float
    mu_err: float
    sigma_err: float
    chi2: float
    ndf: int
    converged: bool

    @property
    def chi2_over_ndf(self) -> float:
        return self.chi2 / self.ndf if self.ndf > 0 else float("nan")


def fit_gaussian_core(
    values: np.ndarray,
    hist_range: tuple[float, float] = config.RESOLUTION_FIT_HIST_RANGE,
    n_bins: int = config.RESOLUTION_FIT_HIST_BINS,
    core_sigma: float = config.RESOLUTION_FIT_CORE_SIGMA,
    max_sigma: float = config.RESOLUTION_MAX_SIGMA,
) -> GaussianFitResult:
    """Two-pass Gaussian fit to the core of a residual distribution, using ROOT.

    The fit is done by ROOT's ``TH1F.Fit(TF1("gaus"))`` (not scipy), so the
    numbers a user re-fits by hand in a ROOT session match trkperf's exactly,
    and the fitted TF1 can be drawn with the standard ROOT plotting tools.

    Pass 1: histogram `values` over the full `hist_range` and fit a
    Gaussian, to get a rough (mu, sigma).
    Pass 2: re-histogram and re-fit restricted to
    [mu - core_sigma*sigma, mu + core_sigma*sigma] from pass 1, so
    non-Gaussian tails (mis-reconstruction, bremsstrahlung, hard scattering)
    do not bias the reported width. The pass-2 result is what is returned.

    The Gaussian's sigma is constrained (via TF1::SetParLimits) to
    [0, max_sigma]. Since resolution is defined on
    Delta(pT)/pT = (pT_reco - pT_truth) / pT_truth, a width of 1.0 means a
    100% resolution - the natural physical maximum. A fit that hits the upper
    bound, or otherwise fails to converge, is reported as ``converged=False``
    (sigma/NaN) rather than an unphysical huge number.

    A `GaussianFitResult` is always returned, even on failure
    (`converged=False`, other fields NaN) - callers must check `converged`
    and `n_entries` before trusting `sigma`; this function never raises just
    because a fit didn't converge (that is an expected outcome for a
    low-statistics bin, not a bug).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_entries = len(values)

    ROOT = _root()

    def _fit(vals: np.ndarray, rng: tuple[float, float]) -> tuple | None:
        h = ROOT.TH1F(
            _next_root_name("h"), "h", n_bins, rng[0], rng[1]
        )
        for v in vals:
            h.Fill(float(v))
        if h.GetEntries() < 4:  # need more points than the 3 fit parameters
            return None
        f = ROOT.TF1(_next_root_name("f"), "gaus", rng[0], rng[1])
        # Only constrain sigma (the physical quantity of interest): lower bound
        # keeps it positive; upper bound = RESOLUTION_MAX_SIGMA (1.0) enforces
        # the physical maximum on Delta(pT)/pT. Leave the amplitude free.
        f.SetParLimits(2, 1e-9, float(max_sigma))
        # Seed the parameters (amplitude, mean, sigma) from the data so they
        # start inside their bounds - avoids ROOT's "bounds outside current
        # value" chatter and gives the minimiser a sensible starting point.
        peak_bin = int(h.GetMaximumBin())
        f.SetParameters(
            float(h.GetBinContent(peak_bin)),
            float(h.GetBinCenter(peak_bin)),
            min(float(h.GetRMS()), max_sigma),
        )
        try:
            res = h.Fit(f, "S Q R N")
        except Exception:
            return None
        # A failed/degenerate fit can return a null or empty TFitResultPtr.
        # Any attribute access on a null one raises ReferenceError (attempt to
        # access a null-pointer) outside PyROOT's normal fit-return handling -
        # guard the whole inspection so such a bin is reported non-converged
        # instead of crashing the run. `not res` from the old guard cannot be
        # relied on alone: a partially-initialised result can be truthy yet
        # still raise on member access.
        try:
            is_empty = res.IsEmpty()
            status = res.Status()
            popt = list(res.Parameters())
            perr = list(res.Errors())
        except ReferenceError:
            return None
        if is_empty or status != 0:
            return None
        # If the fit pinned sigma against the physical maximum, the data is
        # not describable by a physical resolution -> treat as failed.
        if popt[2] >= max_sigma * 0.999:
            return None
        return popt, perr, float(res.Chi2()), int(res.Ndf())

    pass1 = _fit(values, hist_range)
    if pass1 is None:
        return GaussianFitResult(n_entries, *([float("nan")] * 4), float("nan"), 0, False)

    popt1, _perr1, _chi2_1, _ndf_1 = pass1
    mu1, sigma1 = popt1[1], popt1[2]
    core_range = (mu1 - core_sigma * abs(sigma1), mu1 + core_sigma * abs(sigma1))
    pass2 = _fit(values, core_range)
    if pass2 is None:
        # Fall back to the full-range fit rather than reporting nothing.
        popt, perr, chi2, ndf = pass1
    else:
        popt, perr, chi2, ndf = pass2
    mu, sigma = popt[1], popt[2]

    sigma = abs(sigma)
    # ROOT's TF1 float parameters are in [0, max_sigma] by construction, and
    # a boundary-pinned sigma was already rejected above. As a final guard,
    # also reject a center that sits far outside the data's central mass with
    # a poor chi2 (a sign the fit collapsed onto a spurious blob).
    median = float(np.median(values))
    median_dist = abs(mu - median)
    far_and_bad_chi2 = (
        median_dist > 3.0 * (sigma + 1e-9)
        and (chi2 / ndf if ndf > 0 else 1.0) > 5.0
    )
    if far_and_bad_chi2 or not np.all(np.isfinite(perr)):
        return GaussianFitResult(
            n_entries=n_entries,
            mu=float("nan"),
            sigma=float("nan"),
            mu_err=float("nan"),
            sigma_err=float("nan"),
            chi2=float("nan"),
            ndf=0,
            converged=False,
        )

    return GaussianFitResult(
        n_entries=n_entries,
        mu=float(mu),
        sigma=float(sigma),
        mu_err=float(perr[1]),
        sigma_err=float(perr[2]),
        chi2=chi2,
        ndf=ndf,
        converged=True,
    )


def binomial_error(k: np.ndarray | int, n: np.ndarray | int) -> np.ndarray:
    """Wilson score-interval uncertainty on a ratio k/n (95 % CL).

    Used for acceptance/efficiency/fake-rate errors. A single conservative
    width ``max(value - lo, hi - value)`` is reported so the ``*_err`` column
    convention (and compare.py's symmetric propagation) keeps working; unlike
    the normal approximation this never collapses to zero at k=0 or k=n -
    exactly the most-quoted bins (efficiency ~1, fake rate ~0). NaN where
    n == 0. Closed form on purpose: this module must work without scipy.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    kk, nn = np.broadcast_arrays(k, n)
    safe_n = np.where(nn > 0, nn, 1.0)
    z = 1.96
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 + z * z / safe_n
        p = np.where(nn > 0, kk / safe_n, np.nan)
        center = (p + z * z / (2.0 * safe_n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / safe_n + z * z / (4.0 * safe_n**2)) / denom
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    return np.where(nn > 0, np.maximum(p - lo, hi - p), np.nan)
