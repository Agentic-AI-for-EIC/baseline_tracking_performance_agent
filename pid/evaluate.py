"""Physics-grade PID performance metrics.

What a tracking-PID result has to report, and how each number is obtained:

``efficiency at a fixed fake rate``
    The headline. The working point is read off the ROC of the discriminator
    (signal = the class of interest, background = the competing class/es) at the
    fake rates in :data:`pid.config.FAKE_RATE_TARGETS`. Both numbers carry Garwood
    (Clopper-Pearson) intervals, so a 1e-4 rejection measured from 400 pions is
    visibly as uncertain as it is.
``n_sigma(a, b)``
    Separation power between two species' discriminator peaks, from **ROOT**
    single-Gaussian fits - the same fitter as the momentum-resolution core fits,
    so a human can reproduce both in one ROOT session - combined as
    ``|mu_a - mu_b| / sqrt((sigma_a^2 + sigma_b^2)/2)``.
``confusion matrix``
    At the working point, row-normalised (efficiency view) and column-normalised
    (purity view). Unlike ``trkperf pid-confusion``, which tabulates the
    *reconstructor's* mass hypothesis, this tabulates this classifier's decision,
    so the two can be compared to see how much truth-level confusion is PID
    rather than tracking.
``calibration``
    Reliability table: pulled fraction vs observed fraction. Boosted scores are
    not probabilities, and without this the clean-vs-background comparison of a
    "1e-3 fake rate working point" would silently compare two different things.

Tables are written through :func:`pid.report.write` (JSON + markdown + ROOT
TNtuple under ``output/``), with ``insufficient_stats`` flagged wherever a bin
falls below :data:`pid.config.MIN_ENTRIES_PER_BIN`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from trkperf import binning

from . import config, dataset, report


# ---------------------------------------------------------------------------
# Fits and intervals
# ---------------------------------------------------------------------------

def fit_gaussian(values, *, n_bins: int = 60, rng: tuple | None = None) -> dict:
    """Single-Gaussian fit with ROOT (``TH1F.Fit(TF1("gaus"))``).

    Like :func:`trkperf.binning.fit_gaussian_core` but with a free mean
    (discriminator peaks are not at zero) and an explicit range instead of a
    core-refit window.

    Returns
    -------
    dict with ``mu``, ``sigma``, ``chi2``, ``ndf``, ``n``, ``converged``. A fit
    that cannot be made returns ``converged=False`` with NaN parameters - never a
    fabricated number (AGENTS.md).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    empty = {"mu": np.nan, "sigma": np.nan, "sigma_err": np.nan, "mu_err": np.nan,
             "chi2": np.nan, "ndf": 0, "n": int(v.size), "converged": False}
    if v.size < config.MIN_ENTRIES_PER_BIN:
        return empty
    ROOT = binning._root()
    if rng is None:
        lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
        if not hi > lo:
            lo, hi = float(v.min()) - 1e-6, float(v.max()) + 1e-6
        rng = (lo, hi)
    h = ROOT.TH1F("pid_gauss_fit", "pid_gauss_fit", int(n_bins), float(rng[0]), float(rng[1]))
    for x in v:
        h.Fill(float(x))
    if h.GetEffectiveEntries() < 1:
        return empty
    f = ROOT.TF1("pid_gaus", "gaus", float(rng[0]), float(rng[1]))
    f.SetParameters(h.GetMaximum() * 0.4, h.GetMean(), max(h.GetRMS(), 1e-6))
    f.SetParLimits(1, float(rng[0]), float(rng[1]))
    f.SetParLimits(2, 1e-6, float(rng[1] - rng[0]))
    try:
        h.Fit(f, "QNR")
        mu, mu_err = f.GetParameter(1), f.GetParError(1)
        sigma, sigma_err = f.GetParameter(2), f.GetParError(2)
        chi2, ndf = f.GetChisquare(), f.GetNDF()
    except Exception:  # noqa: BLE001 - a failed fit is a result, not a crash
        return empty
    converged = bool(np.isfinite(mu) and np.isfinite(sigma) and sigma > 0 and ndf > 0)
    return {"mu": float(mu), "sigma": float(abs(sigma)), "sigma_err": float(sigma_err),
            "mu_err": float(mu_err), "chi2": float(chi2), "ndf": int(ndf),
            "n": int(v.size), "converged": converged}


def n_sigma(fit_a: dict, fit_b: dict) -> float:
    """Separation in sigma between two fitted peaks."""
    if not (fit_a.get("converged") and fit_b.get("converged")):
        return float("nan")
    denom = np.sqrt((fit_a["sigma"] ** 2 + fit_b["sigma"] ** 2) / 2.0)
    if not denom > 0:
        return float("nan")
    return abs(fit_a["mu"] - fit_b["mu"]) / denom


def logit(p, *, eps: float = config.LOGIT_EPSILON) -> np.ndarray:
    """z = ln(P / (1 - P)), with P clipped into [eps, 1-eps].

    Boosted classifiers saturate: a well-separated electron score sits at
    0.999... and the pion score near 0.001, so on the probability scale both
    distributions are truncated by the axis rather than by their own tails. The
    logit maps them onto the real line, where a Gaussian width measures what it is
    supposed to measure.
    """
    v = np.asarray(p, dtype=float)
    v = np.clip(v, float(eps), 1.0 - float(eps))
    return np.log(v / (1.0 - v))


def nsigma_from_auc(auc_value: float) -> float:
    """n_sigma = 2 * erfinv(2 * AUC - 1).

    Exact for two equal-variance Gaussians, since AUC = Phi(d / sqrt(2)) where
    d = |mu_1 - mu_2| / sigma. Requires no fit (so it cannot diverge on the
    saturated scores that break the naive approach) and is invariant under any
    monotone rescaling of the discriminator.

    AUC of exactly 0 or 1 returns NaN: in a finite sample either means the two
    score distributions do not overlap at all, which is beyond this estimator's
    range rather than an infinite separation - see :func:`nsigma_lower_limit`.
    AUC below 0.5 keeps its sign: a negative n_sigma is an inverted score, i.e. a
    broken model, and must not be laundered into a small positive number.
    """
    from scipy.special import erfinv

    a = float(auc_value)
    if not np.isfinite(a) or a >= 1.0 or a <= 0.0:
        return float("nan")
    return float(2.0 * erfinv(2.0 * a - 1.0))


def _p_no_overlap(d: float, n_signal: int, n_background: int) -> float:
    """P(all background scores below all signal scores) for equal-width Gaussians.

    With S ~ N(0,1) and B ~ N(-d, 1), substituting u = Phi(x) turns the
    order-statistics integral into

        P = int_0^1 n_s (1-u)^(n_s-1) Phi(Phi^-1(u) + d)^n_b du

    which is exact - importantly **not** the product of independent per-pair
    probabilities, since pair comparisons share their scores. Independence
    (``AUC**(ns*nb)``) *underestimates* P(no overlap): at d = 4.21 - where the
    independence model itself reaches P = 0.05 - the exact value is P ~ 0.47,
    so solving the exact integral instead gives d_L = 3.20. The approximation
    would therefore have set the 95 % CL limit ~31 % stronger than the data
    supports.
    """
    from scipy.integrate import quad
    from scipy.stats import norm

    ns, nb = int(n_signal), int(n_background)
    if ns <= 0 or nb <= 0:
        return float("nan")

    def f(u):
        if not (0.0 < u < 1.0):
            return 0.0
        log_u = (np.log(ns) + (ns - 1) * np.log1p(-u)
                 + nb * np.log(max(norm.cdf(norm.ppf(u) + d), 1e-300)))
        return float(np.exp(log_u)) if log_u > -745.0 else 0.0

    value = quad(f, 0.0, 1.0, limit=800, epsabs=1e-14, epsrel=1e-10,
                 points=[1e-9, 1.0 - 1e-9])[0]
    if value > 1e-12:
        return float(value)
    # Underflow region (d ~ 0 with large samples): integrate in log space.
    v = np.linspace(1e-14, 1.0 - 1e-14, 60001)
    lv = (np.log(ns) + (ns - 1) * np.log1p(-v)
          + nb * np.log(np.clip(norm.cdf(norm.ppf(v) + d), 1e-300, None)))
    m = float(lv.max())
    return float(np.trapezoid(np.exp(lv - m), v) * np.exp(m))


def nsigma_lower_limit(n_signal: int, n_background: int,
                       cl: float = config.REJECTION_CL) -> float:
    """95 % CL lower limit on n_sigma when the two score samples do not overlap.

    P(no overlap) grows with the separation, so the 95 % CL lower limit d_L solves
    P(no overlap | d_L) = alpha = 1 - CL = 0.05: every d < d_L would have produced
    at least one overlapping pair in more than 95 % of experiments, and is therefore
    excluded by the observed perfect separation. Reported instead of a point
    estimate because AUC = 1 saturates the quantile estimator, and printing
    ``n_sigma = inf`` (or the huge artefact a bounded Gaussian fit produces) would
    be worse than useless.

    Validated against the exact combinatorial value at d = 0
    (P = 1 / binom(n_s + n_b, n_s)) and against Monte Carlo at the solved limit
    (see pid/tests/test_evaluate.py).
    """
    from scipy.optimize import brentq

    ns, nb = int(n_signal), int(n_background)
    if ns <= 0 or nb <= 0:
        return float("nan")
    if ns * nb > 4.0e6:  # pragma: no cover - guard against absurd integrals
        return float("nan")
    alpha = 1.0 - float(cl)
    # P(no overlap) grows with d from its d->0 value 1/binom(ns+nb, ns). If even
    # that floor exceeds alpha the observation carries no information about d
    # (e.g. one signal and one background candidate: P = 1/2 at d = 0), and NO
    # limit exists. Returning NaN here rather than extrapolating is the honest
    # answer, and the caller must say so instead of printing "n_sigma > nan".
    floor = _p_no_overlap(1e-6, ns, nb)
    if not np.isfinite(floor) or floor > alpha:
        return float("nan")
    try:
        return float(brentq(lambda d: _p_no_overlap(d, ns, nb) - alpha,
                            1e-6, 60.0, xtol=1e-5))
    except (ValueError, RuntimeError):  # pragma: no cover - bracketing failure
        return float("nan")


def measure_separation(score_signal, score_background, *, method: str | None = None,
                       n_bins: int = 60) -> dict:
    """The single entry point for "how well do these two classes separate".

    Parameters
    ----------
    score_signal, score_background:
        The two classes' discriminator values (probabilities in [0, 1]).
    method:
        ``"quantile"`` (default, from :data:`pid.config.NSIGMA_METHOD`) - closed
        form from the AUC; ``"logit"`` - pooled-sigma Gaussian fit to
        :func:`logit`(score); ``"raw"`` - the deprecated fit to the bounded score,
        kept only to reproduce earlier numbers.

    Returns
    -------
    dict with ``auc``, ``n_sigma`` (the chosen estimator), ``n_sigma_quantile``,
    ``n_sigma_logit``, ``n_sigma_raw``, the peak fits, ``nsigma_method``,
    ``saturated``, ``reason``, ``n_signal``, ``n_background``.

    Notes
    -----
    When the quantile estimator saturates (AUC = 1) but the logit fit still
    measures something, ``n_sigma`` falls back to the logit value and
    ``nsigma_method`` records that fallback, so the provenance of every number
    survives.
    """
    if method is None:
        method = config.NSIGMA_METHOD
    if method not in config.NSIGMA_METHODS:
        raise ValueError(f"unknown n_sigma method {method!r}; "
                         f"choices {config.NSIGMA_METHODS}")
    s = np.asarray(score_signal, dtype=float)
    b = np.asarray(score_background, dtype=float)
    s, b = s[np.isfinite(s)], b[np.isfinite(b)]
    out: dict = {"auc": float("nan"), "n_sigma": float("nan"),
                 "n_sigma_quantile": float("nan"), "n_sigma_logit": float("nan"),
                 "n_sigma_raw": float("nan"), "nsigma_method": method,
                 "saturated": False, "reason": "", "n_signal": int(s.size),
                 "n_background": int(b.size), "space": "logit(score)",
                 "fit_signal": {}, "fit_background": {}, "fit_limited": False,
                 "n_sigma_lower_limit": float("nan")}
    if s.size == 0 or b.size == 0:
        out["reason"] = (f"need candidates of both classes, have {s.size} signal / "
                         f"{b.size} background")
        return out

    # The statistic floors apply to the FIT-based estimators only. An AUC (and so
    # the quantile n_sigma from it) is well defined with a handful of background
    # candidates - imprecise, but a number that exists; refusing to report it
    # because a Gaussian peak fit would be meaningless conflates two different
    # limitations. fit_gaussian enforces its own MIN_ENTRIES_PER_BIN and simply
    # returns converged=False, which propagates as a NaN n_sigma_logit.
    thin = min(s.size, b.size) < config.MIN_ENTRIES_PER_BIN
    if thin:
        out["reason"] = (f"peak fits need >= {config.MIN_ENTRIES_PER_BIN} per class "
                         f"(have {s.size} / {b.size}): n_sigma from the Gaussian fits "
                         "is unavailable, the AUC-based estimate is still computed")
        out["fit_limited"] = True

    y = np.r_[np.ones(s.size, int), np.zeros(b.size, int)]
    out["auc"] = auc(y, np.r_[s, b])
    out["n_sigma_quantile"] = nsigma_from_auc(out["auc"])
    if out["auc"] == 0.0:
        # Perfectly INVERTED scores: the quantile estimator is NaN by design
        # (see nsigma_from_auc), so say so - otherwise a broken model reports
        # a bare NaN with no reason attached.
        out["reason"] = (out["reason"] + "; " if out["reason"] else "") + (
            "AUC = 0: scores rank backgrounds above signal (inverted model)")
    if not np.isfinite(out["n_sigma_quantile"]):
        out["saturated"] = bool(out["auc"] >= 1.0)
        if out["saturated"]:
            out["n_sigma_lower_limit"] = nsigma_lower_limit(s.size, b.size)
            if np.isfinite(out["n_sigma_lower_limit"]):
                out["reason"] = (
                    f"AUC = 1: no score overlap among {s.size} x {b.size} "
                    f"signal-background pairs, so the point estimate saturates; "
                    f"n_sigma > {out['n_sigma_lower_limit']:.2f} at 95 % CL"
                    + (" (weak: the sample barely excludes no separation)"
                       if out["n_sigma_lower_limit"] < 1.0 else "")
                    + " (pid.evaluate.nsigma_lower_limit)")
            else:
                out["reason"] = (
                    f"AUC = 1: perfect separation of {s.size} signal against {b.size} "
                    "background candidates, but this sample is too small to set any "
                    "lower limit on n_sigma (P(no overlap) at d = 0 already exceeds "
                    "the tail probability) - report 'no overlap observed', not a number")
        else:
            out["reason"] = "AUC undefined"

    zs, zb = logit(s), logit(b)
    if thin:
        # Still call the fitter so the reason (too few entries) is recorded, but do
        # not let a NaN width become the headline number.
        fs, fb = ({"mu": np.nan, "sigma": np.nan, "n": int(s.size), "converged": False,
                   "chi2": np.nan, "ndf": 0},
                  {"mu": np.nan, "sigma": np.nan, "n": int(b.size), "converged": False,
                   "chi2": np.nan, "ndf": 0})
    else:
        fs, fb = fit_gaussian(zs, n_bins=n_bins), fit_gaussian(zb, n_bins=n_bins)
    out["fit_signal"], out["fit_background"] = fs, fb
    out["n_sigma_logit"] = n_sigma(fs, fb)

    if method == "raw":
        rs, rb = fit_gaussian(s, n_bins=n_bins), fit_gaussian(b, n_bins=n_bins)
        out["n_sigma_raw"] = n_sigma(rs, rb)
        out["fit_signal_raw"], out["fit_background_raw"] = rs, rb
        out["space"] = "score (bounded [0,1]; DEPRECATED)"
        out["reason"] = (out["reason"] + "; " if out["reason"] else "") + (
            "method='raw' fits Gaussians to bounded scores, so the widths measure "
            "the [0,1] clip rather than the distribution tails")

    # Label the space by the estimator that actually produced the number: the
    # quantile form uses no space at all, and writing "logit(score)" beside it
    # would send a reader looking for a fit that never happened.
    primary = {"quantile": out["n_sigma_quantile"], "logit": out["n_sigma_logit"],
               "raw": out["n_sigma_raw"]}[method]
    if not np.isfinite(primary) and np.isfinite(out["n_sigma_logit"]):
        primary = out["n_sigma_logit"]
        out["nsigma_method"] = "logit(fallback: primary estimator undefined)"
    out["n_sigma"] = float(primary) if np.isfinite(primary) else float("nan")
    out["space"] = {"quantile": "rank-based (AUC; no fit)",
                    "logit": "logit(score)", "raw": out["space"]}[
                        method if method != "raw" else "raw"]
    if not np.isfinite(out["n_sigma"]) and "no separation estimator converged" not in out["reason"]:
        out["reason"] = (out["reason"] + "; " if out["reason"] else "") + (
            "no separation estimator converged")
    return out


def garwood(k: int, n: int) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 95 % interval for k successes out of n."""
    from scipy.stats import beta

    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(min(max(k, 0), n))
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Working points
# ---------------------------------------------------------------------------

def working_point(y, score, fake_target: float) -> dict:
    """Threshold and signal efficiency at (or just under) a fixed fake rate.

    The threshold chosen is the *least* selective ROC point whose measured
    background acceptance is still <= ``fake_target``, i.e. the best efficiency
    that actually meets the requirement on this sample. The achieved fake rate is
    reported alongside, because with few background candidates it lands below the
    target rather than on it.
    """
    from sklearn.metrics import roc_curve

    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score) & np.isfinite(y)
    y, score = y[ok], score[ok]
    n_sig, n_bkg = int((y == 1).sum()), int((y == 0).sum())
    base = {"threshold": np.nan, "efficiency": np.nan, "efficiency_err": (np.nan, np.nan),
            "fake_rate": np.nan, "fake_rate_err": (np.nan, np.nan), "n_signal": n_sig,
            "n_background": n_bkg, "target_fake_rate": fake_target, "meets_target": False}
    if n_sig == 0 or n_bkg == 0:
        return base
    fpr, tpr, thr = roc_curve(y, score)
    accept = np.where(fpr <= fake_target)[0]
    if accept.size == 0:
        # Unreachable for any target >= 0 (roc_curve always starts at fpr =
        # 0), kept so a future caller passing a negative target fails loudly
        # instead of indexing garbage.
        idx, achieved, eff = int(np.argmin(fpr)), float(fpr.min()), float(tpr[int(np.argmin(fpr))])
    else:
        idx = int(accept.max())  # largest fpr still within target -> best efficiency
        achieved, eff = float(fpr[idx]), float(tpr[idx])
    # Count acceptances directly at the chosen cut: inverting the ROC
    # fractions with round() can err by one, and the Garwood intervals below
    # must describe observed counts, not reconstructed ones.
    thr = float(thr[idx]) if idx < len(thr) else np.nan
    passed = score >= thr if np.isfinite(thr) else np.zeros_like(score, dtype=bool)
    n_sig_pass = int((passed & (y == 1)).sum())
    n_bkg_pass = int((passed & (y == 0)).sum())
    return {"threshold": thr,
            "efficiency": eff, "efficiency_err": garwood(n_sig_pass, n_sig),
            "fake_rate": achieved, "fake_rate_err": garwood(n_bkg_pass, n_bkg),
            "n_signal": n_sig, "n_background": n_bkg, "n_signal_pass": n_sig_pass,
            "n_background_pass": n_bkg_pass, "target_fake_rate": fake_target,
            "meets_target": bool(achieved <= fake_target)}


def efficiency_at_threshold(y, score, threshold: float, fake_target: float) -> dict:
    """Accept/efficiency of a *fixed* threshold (the same cut in every bin)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n_sig, n_bkg = int((y == 1).sum()), int((y == 0).sum())
    pass_sig = int(((score >= threshold) & (y == 1)).sum())
    pass_bkg = int(((score >= threshold) & (y == 0)).sum())
    eff = pass_sig / n_sig if n_sig else np.nan
    fake = pass_bkg / n_bkg if n_bkg else np.nan
    return {"threshold": float(threshold),
            "efficiency": float(eff) if np.isfinite(eff) else np.nan,
            "efficiency_err": garwood(pass_sig, n_sig),
            "fake_rate": float(fake) if np.isfinite(fake) else np.nan,
            "fake_rate_err": garwood(pass_bkg, n_bkg),
            "n_signal": n_sig, "n_background": n_bkg, "n_signal_pass": pass_sig,
            "n_background_pass": pass_bkg, "target_fake_rate": fake_target,
            "meets_target": bool(np.isfinite(fake) and fake <= fake_target)}


def auc(y, score) -> float:
    """Rank-based discrimination power (NaN when only one class is present)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if s.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Score / label extraction
# ---------------------------------------------------------------------------

def class_order(df: pd.DataFrame) -> list[str]:
    return [c for c in config.CLASS_LABELS if f"proba_{c}" in df.columns]


def signal_background(df: pd.DataFrame, task: str, signal: str | None = None):
    """``(y, score, signal_class, background_label)`` for one class of interest.

    Binary tasks carry the decision directly in ``label``/``score``. A multiclass
    table is evaluated one class at a time against all the others, which is the
    form PID performance is quoted in ("pion efficiency at 1e-3 kaon fake rate").
    """
    if task in ("eid", "ehad"):
        bkg_label = "hadron" if task == "ehad" else "pi"
        return df["label"].to_numpy(int), df["score"].to_numpy(float), "e", bkg_label
    order = class_order(df)
    if not order:
        raise ValueError(
            f"pid.evaluate: no proba_<class> score columns for task {task!r}; "
            "train it with a multiclass task first")
    sig = signal if signal in order else order[0]
    other = [c for c in order if c != sig]
    y = (df["truth_class"] == sig).astype(int).to_numpy()
    return y, df[f"proba_{sig}"].to_numpy(float), sig, "+".join(other)


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------

def _symmetric_err(lo, hi, value):
    """Half-width of an asymmetric interval, so the result plugs into
    ``trkperf.compare``'s ``<name>`` / ``<name>_err`` convention. The asymmetric
    ``*_err_lo`` / ``*_err_hi`` columns are kept alongside: a symmetric proxy is
    for error *propagation*, never for quoting an uncertainty."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    v = np.asarray(value, dtype=float)
    return np.where(np.isfinite(lo) & np.isfinite(hi) & np.isfinite(v),
                    np.abs((hi - lo) / 2.0), np.nan)


def overall_table(df, *, task, signal=None, targets=config.FAKE_RATE_TARGETS,
                  nsigma_method: str | None = None) -> pd.DataFrame:
    """AUC, both class peaks, n-sigma, efficiency at each working point."""
    y, score, sig, bkg = signal_background(df, task, signal)
    n_sig, n_bkg = int((y == 1).sum()), int((y == 0).sum())
    sep = measure_separation(score[y == 1], score[y == 0], method=nsigma_method)
    rows = [{"quantity": "auc", "signal": sig, "against": bkg, "value": sep["auc"],
             "value_err_lo": np.nan, "value_err_hi": np.nan, "value_err": np.nan,
             "space": "", "n_signal": n_sig, "n_background": n_bkg,
             "target_fake_rate": np.nan,
             "insufficient_stats": bool(min(n_sig, n_bkg) < config.MIN_ENTRIES_PER_BIN)}]
    for name, key in (("signal_peak", "fit_signal"), ("background_peak", "fit_background")):
        fit = sep.get(key) or {}
        rows.append({"quantity": name, "signal": sig, "against": bkg,
                     "value": fit.get("mu", np.nan), "value_err_lo": np.nan,
                     "value_err_hi": np.nan, "value_err": np.nan,
                     "space": sep["space"], "n_signal": fit.get("n", 0),
                     "n_background": np.nan, "target_fake_rate": np.nan,
                     "insufficient_stats": bool(not fit.get("converged", False))})
    # The lower-limit row is emitted unconditionally, NaN when it does not
    # apply: `pid compare` keys on (quantity, ...), and a data-dependent row set
    # would silently compare different quantities between clean and bkg tags.
    limit = sep.get("n_sigma_lower_limit", float("nan"))
    if sep["saturated"]:
        limit_label = ("lower limit (perfect separation observed)" if np.isfinite(limit)
                       else "perfect separation observed, sample too small for any limit")
    else:
        limit_label = "not applicable (AUC < 1: point estimate exists)"
    rows.append({"quantity": "n_sigma_lower_limit", "signal": sig, "against": bkg,
                 "value": limit, "value_err_lo": np.nan,
                 "value_err_hi": np.nan, "value_err": np.nan,
                 "space": "equal-width Gaussian limit", "n_signal": n_sig,
                 "n_background": n_bkg, "target_fake_rate": np.nan,
                 "nsigma_method": limit_label,
                 "insufficient_stats": bool(not np.isfinite(limit))})
    rows.append({"quantity": "n_sigma", "signal": sig, "against": bkg,
                 "value": sep["n_sigma"], "value_err_lo": np.nan, "value_err_hi": np.nan,
                 "value_err": np.nan, "space": sep["space"], "n_signal": n_sig,
                 "n_background": n_bkg, "target_fake_rate": np.nan,
                 "nsigma_method": sep["nsigma_method"],
                 "n_sigma_quantile": sep["n_sigma_quantile"],
                 "n_sigma_logit": sep["n_sigma_logit"],
                 "saturated": sep["saturated"],
                 "insufficient_stats": bool(not np.isfinite(sep["n_sigma"]))})
    for target in targets:
        wp = working_point(y, score, target)
        eloe, ehi = wp["efficiency_err"]
        rows.append({"quantity": "efficiency", "signal": sig, "against": bkg,
                     "value": wp["efficiency"], "value_err_lo": eloe, "value_err_hi": ehi,
                     "value_err": float(_symmetric_err(eloe, ehi, wp["efficiency"])[()])
                     if np.isfinite(eloe) and np.isfinite(ehi) else np.nan,
                     "space": "",
                     "n_signal": wp["n_signal"], "n_background": wp["n_background"],
                     "target_fake_rate": target, "threshold": wp["threshold"],
                     "achieved_fake_rate": wp["fake_rate"],
                     "meets_target": wp["meets_target"],
                     "insufficient_stats": bool(n_bkg < config.MIN_ENTRIES_PER_BIN)})
    out = pd.DataFrame(rows)
    out.attrs["separation"] = sep
    return out


def binned_table(df, *, task, by="pt", signal=None, targets=config.FAKE_RATE_TARGETS,
                 threshold=None) -> pd.DataFrame:
    """Efficiency / fake rate / AUC vs pT, p or eta, in the project's bin edges.

    ``threshold`` fixes one working point across all bins (the honest way to quote
    a cut-based performance map); ``None`` re-derives the cut per bin, and the
    ``working_point`` column records which was done so the two are never confused.
    """
    y, score, sig, bkg = signal_background(df, task, signal)
    work = df.copy()
    work["_y"] = y
    work["_score"] = score
    # truth_pt / truth_p / truth_eta are accepted so this table can be put
    # side-by-side with the tracking metrics, which bin in TRUTH kinematics per
    # AGENTS.md; the default basis (pt, p, eta) is the RECONSTRUCTED one, which is
    # what a working point is actually applied to. Rows are partitioned
    # differently, never counted differently: the candidate set is identical.
    truth_col = {"truth_pt": "pt", "truth_p": "p", "truth_eta": "eta"}.get(by, by)
    if truth_col not in ("pt", "p", "eta"):
        raise ValueError(f"pid.evaluate.binned_table: 'by' must be pt/p/eta or "
                         f"truth_pt/truth_p/truth_eta, got {by!r}")
    column = by
    if column not in work.columns:
        raise KeyError(
            f"pid.evaluate.binned_table: this score table has no {column!r} column; "
            "truth-binned tables need a score table written by the current pid.train.")
    edges = config.ETA_BIN_EDGES if truth_col == "eta" else config.PT_BIN_EDGES
    work["_bin"] = pd.cut(work[column], bins=edges)
    n_out_of_range = int(work["_bin"].isna().sum())
    if n_out_of_range:
        print(f"[evaluate/{task}] NOTE: {n_out_of_range} rows fall outside the "
              f"{by} bin edges and are excluded from the binned table (the "
              "global overall table still counts them)", file=sys.stderr)
    rows = []
    for b, sub in work.groupby("_bin", observed=True):
        yy, ss = sub["_y"].to_numpy(), sub["_score"].to_numpy()
        n_sig, n_bkg = int((yy == 1).sum()), int((yy == 0).sum())
        row = {"bin": str(b), "bin_center": float(b.mid), "variable": by,
               "binning_basis": "truth" if by.startswith("truth_") else "reconstructed",
               "signal": sig,
               "against": bkg, "n_signal": n_sig, "n_background": n_bkg,
               "auc": auc(yy, ss),
               "working_point": "per_bin" if threshold is None else "global"}
        for target in targets:
            res = (working_point(yy, ss, target) if threshold is None
                   else efficiency_at_threshold(yy, ss, threshold, target))
            lo, hi = res["efficiency_err"]
            flo, fhi = res["fake_rate_err"]
            row[f"eff_at_{target:g}"] = res["efficiency"]
            row[f"eff_at_{target:g}_err_lo"], row[f"eff_at_{target:g}_err_hi"] = lo, hi
            row[f"eff_at_{target:g}_err"] = float(_symmetric_err(lo, hi, res["efficiency"]))
            row[f"fake_at_{target:g}"] = res["fake_rate"]
            row[f"meets_target_at_{target:g}"] = res["meets_target"]
            row[f"fake_at_{target:g}_err_lo"], row[f"fake_at_{target:g}_err_hi"] = flo, fhi
            row[f"fake_at_{target:g}_err"] = float(_symmetric_err(flo, fhi, res["fake_rate"]))
            if threshold is None:
                row[f"threshold_at_{target:g}"] = res["threshold"]
        row["insufficient_stats"] = bool(min(n_sig, n_bkg) < config.MIN_ENTRIES_PER_BIN)
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_table(df, *, task, threshold=None) -> pd.DataFrame:
    """Per-species confusion at the working point, row- and column-normalised.

    Every row carries the decision rule that produced it (``decision`` plus,
    for the binary threshold rule, ``threshold``/``target_fake_rate``): a
    confusion matrix without its cut is unquotable.
    """
    order = class_order(df)
    decision, thr, target = "argmax", float("nan"), float("nan")
    if order:
        mat = np.column_stack([df[f"proba_{c}"].to_numpy(float) for c in order])
        pred = np.array([order[i] for i in np.nanargmax(mat, axis=1)], dtype=object)
        classes = order
    else:
        y, score, sig, bkg = signal_background(df, task)
        thr = threshold
        target = float(min(config.FAKE_RATE_TARGETS))
        if thr is None or not np.isfinite(thr):
            thr = working_point(y, score, target)["threshold"]
        pred = np.where(np.asarray(score) >= thr, sig, bkg).astype(object)
        classes = [sig, bkg]
        decision = f"threshold@{target:g}"
    truth = df["truth_class"].to_numpy()
    rows = []
    for t in classes:
        n_t = int((truth == t).sum())
        for p in classes:
            n = int(((truth == t) & (pred == p)).sum())
            lo, hi = garwood(n, n_t)
            n_p = int((pred == p).sum())
            rows.append({"truth_class": t, "pred_class": p, "n": n,
                         "row_fraction": (n / n_t) if n_t else np.nan,
                         "row_fraction_err_lo": lo, "row_fraction_err_hi": hi,
                         "row_fraction_err": float(_symmetric_err(lo, hi,
                                                    (n / n_t) if n_t else np.nan)),
                         "col_fraction": (n / n_p) if n_p else np.nan,
                         "n_truth": n_t, "n_pred": n_p,
                         "decision": decision, "threshold": thr,
                         "target_fake_rate": target,
                         "insufficient_stats": bool(n_t < config.MIN_ENTRIES_PER_BIN)})
    return pd.DataFrame(rows)


def calibration_table(df, *, score_col="score", y_col="label", n_bins=10) -> pd.DataFrame:
    """Reliability curve: does a pulled 0.7 mean 70 % electrons?"""
    p = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    columns = ["bin", "bin_center", "pulled", "observed", "observed_err_lo",
               "observed_err_hi", "n", "insufficient_stats"]
    rows = []
    if p.size == 0:
        return pd.DataFrame(columns=columns)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    for b in range(int(n_bins)):
        sel = idx == b
        n = int(sel.sum())
        k = int(y[sel].sum()) if n else 0
        lo, hi = garwood(k, n)
        # digitize(..., right=True) makes bins (lo, hi] (the first bin also
        # catches p == 0); label them that way instead of [lo, hi].
        lo_edge, hi_edge = edges[b], edges[b + 1]
        label = (f"[{lo_edge:.2f}, {hi_edge:.2f}]" if b == 0
                 else f"({lo_edge:.2f}, {hi_edge:.2f}]")
        rows.append({"bin": label,
                     "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
                     "pulled": float(p[sel].mean()) if n else np.nan,
                     "observed": float(k / n) if n else np.nan,
                     "observed_err_lo": lo, "observed_err_hi": hi, "n": n,
                     "insufficient_stats": bool(n < config.MIN_ENTRIES_PER_BIN)})
    return pd.DataFrame(rows)


def roc_table(df, *, task, signal=None, n_points: int = 200) -> pd.DataFrame:
    """ROC curve (fake rate vs efficiency) for plotting and for reading any cut."""
    from sklearn.metrics import roc_curve

    y, score, sig, bkg = signal_background(df, task, signal)
    ok = np.isfinite(score)
    fpr, tpr, thr = roc_curve(y[ok], score[ok])
    if fpr.size > n_points:
        keep = np.linspace(0, fpr.size - 1, n_points).astype(int)
        fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    return pd.DataFrame({"fake_rate": fpr, "efficiency": tpr, "threshold": thr[:len(fpr)],
                         "signal": sig, "against": bkg})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(scores_path: str, *, task: str, dataset_tag: str, model: str = "",
             signal=None, targets=config.FAKE_RATE_TARGETS, out_dir=config.OUTPUT_DIR,
             by=("pt", "eta"), quiet=False) -> dict:
    """Evaluate a saved score table and write every output file."""
    df = dataset.load_table(scores_path)
    tag = f"{task}_{model}_{dataset_tag}" if model else f"{task}_{dataset_tag}"
    if signal is None and task not in ("eid", "ehad"):
        # Multiclass tables are quoted one class at a time; without --signal
        # this defaults to the first class, which must be said out loud.
        try:
            first = class_order(df)[0]
        except (ValueError, IndexError):
            first = None
        print(f"[evaluate/{tag}] NOTE: no --signal given for multiclass task "
              f"{task!r}; quoting {first!r} vs rest", file=sys.stderr)
    tables = {"overall": overall_table(df, task=task, signal=signal, targets=targets)}
    for variable in by:
        tables[f"vs_{variable}"] = binned_table(df, task=task, by=variable,
                                                signal=signal, targets=targets)
    tables["confusion"] = confusion_table(df, task=task)
    tables["roc"] = roc_table(df, task=task, signal=signal)
    if task in ("eid", "ehad") and "score" in df.columns:
        tables["calibration"] = calibration_table(df)

    # Provenance that must survive into every table: how many files and events the
    # quoted numbers came from. AGENTS.md requires stating this alongside any
    # number, and an AUC is meaningless without knowing it came from 214 tracks.
    # Signal/background counts are only meaningful for binary (0/1-labelled)
    # tables; multiclass tables carry per-class counts in each row instead, so
    # reporting label==1/==0 here would silently quote one class pair.
    labels = set(pd.Series(df["label"]).dropna().unique().tolist()) if "label" in df else set()
    binary_labels = labels <= {0, 1}
    n_files = int(pd.Series(df["file_id"]).nunique()) if "file_id" in df else None
    n_events = int(pd.Series(df["event"]).nunique()) if "event" in df else None
    provenance = {"n_files_scored": n_files, "n_events_scored": n_events,
                  "n_rows_scored": int(len(df)),
                  "n_signal": int((df["label"] == 1).sum()) if binary_labels else None,
                  "n_background": int((df["label"] == 0).sum()) if binary_labels else None}

    paths = {}
    for name, table in tables.items():
        paths[name] = report.write(table, os.path.join(out_dir, f"pid-{tag}-{name}"),
                                   tree_name=f"pid_{task}_{name}",
                                   meta={"artifact": name, "task": task, "model": model,
                                         "dataset_tag": dataset_tag,
                                         "scores_table": scores_path, **provenance})
    if not quiet:
        print(f"[evaluate/{tag}] scored sample: {n_files} file(s), {n_events} event(s), "
              f"{len(df)} tracks"
              + (f" ({provenance['n_signal']} signal / {provenance['n_background']} background)"
                 if provenance["n_signal"] is not None else ""))
        for _, r in tables["overall"].iterrows():
            note = f" (target fake {r['target_fake_rate']:g})" if r["quantity"] == "efficiency" else ""
            # Distinguish "below the statistics floor" (the number exists but is
            # imprecise) from "could not be computed at all" (a Gaussian fit needs
            # >= MIN_ENTRIES_PER_BIN candidates; one file yields ~10 pions).
            if not np.isfinite(float(r["value"])) if r["value"] is not None else True:
                flag = "  [NOT COMPUTABLE: too few candidates to fit]"
            elif bool(r.get("insufficient_stats", False)):
                flag = (f"  [below the {config.MIN_ENTRIES_PER_BIN}-entry floor: "
                        "imprecise, do not quote alone]")
            else:
                flag = ""
            print(f"[evaluate/{tag}] {r['quantity']:16s} {r['value']}{note}{flag}")
    tables["paths"] = paths
    tables["tag"] = tag
    return tables
