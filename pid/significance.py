"""Maximum-significance working points.

The cut is chosen to maximise

    FOM(c) = S(c) / sqrt(S(c) + B(c))

with S/B the *accepted* weighted signal/background yields at threshold ``c``
(signal passes when score >= c). This is the standard HEP discovery-style figure
of merit for a PID selection: it interpolates correctly between "keep
everything" and "keep only the pure tail", unlike efficiency at a fixed fake
rate, which requires a background sample large enough to define that rate - and
in the backward leg of this sample there are ~60 pions per file, so 10⁻⁴ cannot
be evaluated at all (see :func:`rejection_limits`).

Two things this module refuses to do silently, both of which would otherwise
produce a defensible-looking but wrong number:

* **Claim a rejection power from an empty background.** With N background
  candidates and zero accepted, the point estimate is fake = 0 and 1/fake is
  infinite. The honest quantity is the Clopper-Pearson upper limit on the fake
  rate, hence a *lower limit* on rejection: 3 accepted-free pions out of 20 still
  only proves R > 1/0.14 ≈ 7 at 95 % CL. :func:`rejection_limits` returns both,
  and the plotting layer draws the limit, not the infinity.
* **Forget that FOM depends on normalisation.** S/sqrt(S+B) is not scale
  invariant: doubling the exposure (fixed composition) multiplies it by sqrt(2)
  and moves the optimum. Every returned optimum therefore carries the weights and
  ``lumi_scale`` it was computed with, and the caller must quote them with the
  cut (:data:`pid.config.DEFAULT_LUMI_SCALE`).

Weights are handled as effective counts (Kish) for the variance-based variant:
the Poisson variance of a weighted sum is Σw², not Σw, so with a background
rescaled by a factor f the naive formula would overstate the significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def garwood_upper(k: int, n: int, cl: float = config.REJECTION_CL) -> float:
    """One-sided Clopper-Pearson upper limit on a binomial rate of k/n."""
    from scipy.stats import beta

    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    alpha = 1.0 - cl
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def sum_of_squares(weights) -> float:
    """Σw² - the Poisson variance of a weighted count.

    This is what belongs under the square root of S/sqrt(S+B) once rows carry
    weights: for independent Poisson cells, Var[Σ w_i n_i] = Σ w_i², so a
    background rescaled by f contributes f² to the variance but only f to the
    yield. Using the naive sum instead would inflate the significance of any
    weighted result.
    """
    w = np.asarray(weights, dtype=float)
    return float(np.nansum(w * w))


def kish_effective(weights) -> float:
    """Kish's effective sample size (Σw)²/Σw² - a *count*, not a variance.

    Reported so a reader can see how many effectively-independent candidates a
    weighted channel really contains (a channel whose background is scaled by 100
    still has the statistical power of the candidates actually recorded, and this
    is the quantity that says so).
    """
    w = np.asarray(weights, dtype=float)
    total = float(np.nansum(w))
    ss = float(np.nansum(w * w))
    return (total * total / ss) if ss > 0 else 0.0


def _totals(weights: np.ndarray) -> dict:
    """Weighted yield, variance term and Kish count of one class."""
    return {"sum": float(np.nansum(weights)), "var": sum_of_squares(weights),
            "kish": kish_effective(weights)}


def scan_thresholds(score, y, *, weights=None, thresholds=None,
                    lumi_scale: float = config.DEFAULT_LUMI_SCALE,
                    signal=1, background=0) -> pd.DataFrame:
    """Scan FOM(c) over a threshold grid.

    Parameters
    ----------
    score, y:
        1-D arrays of equal length: discriminator (higher = more signal-like) and
        class label (``signal`` / ``background``).
    weights:
        Optional per-row weight (defaults to 1); arbitrary within-class variation
        is supported - each weight travels with its own score through the
        cumulative sums. Background-only rescaling - the realistic-flux case -
        is expressed by passing those weights in, so the weighting is visible in
        every number rather than hidden in a flag.
    thresholds:
        Grid of cuts. Default: ``config.N_THRESHOLD_SCAN`` uniform values over
        ``[0, 1]``, which is the right domain for a probability score.
    lumi_scale:
        Multiplies both classes (equal exposure change). Applied to the yields
        *and* their variances, so FOM scales as sqrt(lumi_scale).

    Returns
    -------
    DataFrame, one row per threshold, with the accepted yields and their
    efficiencies/fake rates, purities, both significance variants, and the
    rejection point estimate plus its 95 % CL lower limit.

    Point estimates use the (possibly weighted) yields; the confidence limits
    are count-based by construction (weights state an assumed mixture, they do
    not change how many candidates were recorded). With class-constant weights
    the two agree exactly; under a rescaling study the limits are approximate.
    """
    score = np.asarray(score, dtype=float)
    y = np.asarray(y)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    if weights is None:
        w = np.ones(score.shape, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[ok]

    is_sig = y == signal
    is_bkg = y == background
    if not is_sig.any() or not is_bkg.any():
        raise ValueError(
            "pid.significance.scan_thresholds: class "
            f"{signal!r} has {int(is_sig.sum())} and class {background!r} has "
            f"{int(is_bkg.sum())} entries - a working point needs both")

    w_sig_all, w_bkg_all = w[is_sig], w[is_bkg]
    tot_sig = _totals(w_sig_all * lumi_scale)
    tot_bkg = _totals(w_bkg_all * lumi_scale)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, int(config.N_THRESHOLD_SCAN))
    thresholds = np.asarray(sorted(float(t) for t in thresholds), dtype=float)

    # The cumulative sums must run down the *score* ranking: csig[ns] is the
    # yield of the ns rows with the highest scores, so each weight has to travel
    # with its own score. Sorting the weights by their own magnitude instead is
    # a silent bug for any per-row weighting (it only agrees when the weights
    # are constant within a class). Ties cannot corrupt the cumsum: ns counts
    # every row with score >= c, so a tied group is never split.
    sig_order = np.argsort(score[is_sig], kind="stable")[::-1]
    bkg_order = np.argsort(score[is_bkg], kind="stable")[::-1]
    sig_cut = score[is_sig][sig_order]
    bkg_cut = score[is_bkg][bkg_order]
    w_sig_desc = w_sig_all[sig_order]
    w_bkg_desc = w_bkg_all[bkg_order]
    # cumulative weighted yields as a function of descending score
    csig = np.concatenate([[0.0], np.cumsum(w_sig_desc)])
    cbkg = np.concatenate([[0.0], np.cumsum(w_bkg_desc)])

    rows = []
    for c in thresholds:
        ns = int(np.searchsorted(-sig_cut, -c, side="right"))   # count(score >= c)
        nb = int(np.searchsorted(-bkg_cut, -c, side="right"))
        s = float(csig[ns]) * lumi_scale
        b = float(cbkg[nb]) * lumi_scale
        s_eff = sum_of_squares(w_sig_desc[:ns]) * lumi_scale ** 2
        b_eff = sum_of_squares(w_bkg_desc[:nb]) * lumi_scale ** 2
        denom = s + b
        denom_eff = s_eff + b_eff
        fake = b / tot_bkg["sum"] if tot_bkg["sum"] > 0 else np.nan
        # The confidence limit is computed from the OBSERVED counts, not the
        # composition-weighted ones: the weights state an assumed mixture, they do
        # not change how many candidates this sample actually contained, and
        # scaling them would manufacture a tighter limit than was measured.
        n_bkg_tot = int(is_bkg.sum())
        upper = garwood_upper(nb, n_bkg_tot) if n_bkg_tot > 0 else np.nan
        rows.append({
            "threshold": float(c),
            "n_signal": int(ns), "n_background": int(nb),
            "S": s, "B": b,
            "S_eff": s_eff, "B_eff": b_eff,
            "n_signal_total": int(is_sig.sum()), "n_background_total": int(is_bkg.sum()),
            "efficiency": s / tot_sig["sum"] if tot_sig["sum"] > 0 else np.nan,
            "fake_rate": fake,
            "fake_rate_upper_limit": upper,
            "purity": s / denom if denom > 0 else np.nan,
            "significance": s / np.sqrt(denom) if denom > 0 else np.nan,
            "significance_eff": s / np.sqrt(denom_eff) if denom_eff > 0 else np.nan,
            "rejection": 1.0 / fake if (np.isfinite(fake) and fake > 0) else np.nan,
            "rejection_lower_limit": 1.0 / upper if (np.isfinite(upper) and upper > 0) else np.nan,
        })
    table = pd.DataFrame(rows)
    table.attrs.update({"tot_sig": tot_sig, "tot_bkg": tot_bkg,
                        "weighted": weights is not None, "lumi_scale": float(lumi_scale),
                        "note": "S_eff/B_eff are Poisson variance sums (Sigma w^2), "
                                "not Kish effective counts"})
    return table


def optimal_cut(table: pd.DataFrame, *, mode: str = "s_over_sqrt",
                min_candidates: int = 0) -> dict:
    """The threshold maximising FOM(c).

    Ties are broken towards the *higher* threshold (more background rejection,
    i.e. the more conservative selection), and rows with fewer accepted
    candidates than `min_candidates` are excluded so a one-event fluctuation at
    the extreme tail cannot be nominated as a working point.
    """
    if mode not in config.SIGNIFICANCE_MODES:
        raise ValueError(f"unknown significance mode {mode!r}; {config.SIGNIFICANCE_MODES}")
    col = "significance" if mode == "s_over_sqrt" else "significance_eff"
    cand = table.dropna(subset=[col]).copy()
    if min_candidates > 0:
        cand = cand[(cand["n_signal"] + cand["n_background"]) >= min_candidates]
    if cand.empty:
        return {"threshold": float("nan"), "significance": float("nan"),
                "mode": mode, "valid": False,
                "reason": "no threshold has both a defined FOM and enough candidates"}
    best = cand.sort_values([col, "threshold"], ascending=[False, False]).iloc[0]
    out = {
        "threshold": float(best["threshold"]),
        "significance": float(best[col]),
        "significance_s_over_sqrt": float(best["significance"]),
        "significance_eff": float(best["significance_eff"]),
        "efficiency": float(best["efficiency"]),
        "fake_rate": float(best["fake_rate"]),
        "fake_rate_upper_limit": float(best["fake_rate_upper_limit"]),
        "purity": float(best["purity"]),
        "n_signal": int(best["n_signal"]), "n_background": int(best["n_background"]),
        "n_signal_total": int(best["n_signal_total"]),
        "n_background_total": int(best["n_background_total"]),
        "rejection": float(best["rejection"]),
        "rejection_lower_limit": float(best["rejection_lower_limit"]),
        "mode": mode, "valid": True, "reason": "", "caution": "",
        "edge_flag": ("at_grid_low_edge" if best["threshold"] <= 0.0
                      else "at_grid_high_edge" if best["threshold"] >= 1.0
                      else "interior"),
    }
    # S/sqrt(S+B) approaches sqrt(S) when B is tiny, so with too few background
    # candidates the maximum sits at the *lowest* threshold: the "optimal working
    # point" is then just "keep everything", which is a statement about the sample
    # size, not about the classifier. Flag it instead of quietly reporting it.
    # A maximum is only a working point if it *rejects* something. Three distinct
    # failure shapes, all seen in this project's own smoke runs, must be told apart
    # because they have different remedies (more files vs a better model):
    cautions: list[str] = []
    if out["efficiency"] >= 1.0 - 1e-9 and out["fake_rate"] >= 1.0 - 1e-9:
        cautions.append(
            "degenerate: FOM is maximised by accepting every candidate (efficiency and "
            "fake rate both 1), i.e. the classifier does not separate these classes - "
            "no working point exists in this score space")
    elif out["fake_rate"] > 0.5:
        cautions.append(
            f"permissive optimum: fake rate {out['fake_rate']:.2f} at c*, so the FOM gain "
            "comes from keeping signal rather than rejecting background - the separation "
            "power is poor (compare the AUC before quoting this as a selection)")
    if out["threshold"] <= 0.0 and out["efficiency"] >= 1.0 - 1e-9:
        cautions.append(
            f"optimum at zero cut: with only {out['n_background_total']} background "
            "candidates FOM ~ sqrt(S) is maximised by no selection (statistics-limited)")
    if out["edge_flag"] == "at_grid_high_edge":
        cautions.append("optimum pinned at the top of the score range")
    if out["n_background"] < 5:
        cautions.append(
            f"only {out['n_background']} of {out['n_background_total']} background "
            "candidates accepted: fake rate and purity are limited by the sample, use "
            "fake_rate_upper_limit / rejection_lower_limit")
    out["caution"] = "; ".join(cautions)
    out["rejects_background"] = bool(out["valid"] and out["fake_rate"] < 0.5)
    return out


def significance_curve(score, y, **kw) -> tuple[pd.DataFrame, dict]:
    """``(scan table, optimum)`` in one call - the normal entry point."""
    mode = kw.pop("mode", "s_over_sqrt")
    table = scan_thresholds(score, y, **kw)
    return table, optimal_cut(table, mode=mode)


def optimal_cut_in_bins(score, y, x, *, bins=None, weights=None,
                        mode: str = "s_over_sqrt",
                        min_candidates: int = config.MIN_CANDIDATES_PER_WP_BIN,
                        thresholds=None,
                        **kw) -> pd.DataFrame:
    """Per-bin optimum c*(x) - usually x = pT.

    Bins that cannot support a working point return NaN with a `reason`, never a
    borrowed value from a neighbouring bin: a flat c*(pT) line drawn through
    unmeasurable bins is how a fabricated momentum dependence gets published.

    `thresholds` fixes the scan grid (callers plotting per-bin FOM curves must
    pass the same grid they draw, or the table optimum and the drawn star can
    disagree on flat plateaus).
    """
    score = np.asarray(score, dtype=float)
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(score) & np.isfinite(x)
    score, y, x = score[keep], y[keep], x[keep]
    # Weights travel with the same mask: indexing the unfiltered weights with
    # the post-filter bin mask is an IndexError whenever any input is NaN.
    w = None if weights is None else np.asarray(weights, dtype=float)[keep]
    if bins is None:
        bins = config.PT_BINS_PERFORMANCE
    idx = np.clip(np.digitize(x, np.asarray(bins)[1:-1], right=True), 0, len(bins) - 2)
    rows = []
    for b in range(len(bins) - 1):
        sel = idx == b
        row = {"bin": f"({bins[b]:g}, {bins[b + 1]:g}]", "bin_center": 0.5 * (bins[b] + bins[b + 1]),
               "n_signal_bin": int(((y == kw.get("signal", 1)) & sel).sum()),
               "n_background_bin": int(((y == kw.get("background", 0)) & sel).sum())}
        if sel.sum() < min_candidates:
            row.update({"threshold": np.nan, "significance": np.nan, "valid": False,
                        "reason": f"bin has {int(sel.sum())} candidates < {min_candidates}",
                        "caution": ""})
            rows.append(row)
            continue
        try:
            table = scan_thresholds(score[sel], y[sel],
                                    weights=None if w is None else w[sel],
                                    thresholds=thresholds,
                                    **kw)
        except ValueError as exc:  # one class absent in this bin
            row.update({"threshold": np.nan, "significance": np.nan, "valid": False,
                        "reason": str(exc), "caution": ""})
            rows.append(row)
            continue
        opt = optimal_cut(table, mode=mode, min_candidates=1)
        row.update({"threshold": opt["threshold"], "significance": opt["significance"],
                    "efficiency": opt["efficiency"], "fake_rate": opt["fake_rate"],
                    "purity": opt["purity"], "valid": opt["valid"], "reason": opt["reason"],
                    "caution": opt["caution"]})
        rows.append(row)
    return pd.DataFrame(rows)


def rejection_limits(n_accepted: int, n_total: int, *, cl: float = config.REJECTION_CL) -> dict:
    """Rejection point estimate *and* its lower limit for ``1/fake``.

    Returns ``{"fake_rate", "fake_rate_upper_limit", "rejection", "rejection_limit",
    "measurable"}``. ``measurable`` is False when the limit is so weak that the
    point estimate should not be shown as a number (zero accepted background).
    """
    if n_total <= 0:
        return {"fake_rate": np.nan, "fake_rate_upper_limit": np.nan, "rejection": np.nan,
                "rejection_limit": np.nan, "measurable": False}
    k = int(max(0, min(int(n_accepted), int(n_total))))
    fake = k / n_total
    upper = garwood_upper(k, n_total, cl=cl)
    return {"fake_rate": fake,
            "fake_rate_upper_limit": upper,
            "rejection": (1.0 / fake) if fake > 0 else np.nan,
            "rejection_limit": (1.0 / upper) if upper > 0 else np.inf,
            "measurable": bool(k > 0)}


def flux_weights(y, *, signal, background, bkg_scale: float = 1.0,
                 sig_scale: float = 1.0) -> np.ndarray:
    """Per-row weights realising a target species composition.

    The measured composition of a DIS sample is already realistic *for that
    sample*, so the default is unity weights. This exists for the studies where it
    genuinely matters - e.g. quoting a hadron-ID channel after rescaling the pion
    background to the abundance expected with an upgraded interaction rate - and it
    is recorded in the output metadata whenever used.
    """
    y = np.asarray(y)
    w = np.ones(y.shape, dtype=float)
    w[y == signal] = float(sig_scale)
    w[y == background] = float(bkg_scale)
    return w
