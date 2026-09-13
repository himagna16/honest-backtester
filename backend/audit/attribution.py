"""Factor attribution (audit check #9): is this alpha, or repackaged beta?

The first question any professional asks of a backtest is "strip out market,
size, value, momentum, profitability and investment — is there anything
left?". Total return, CAGR and Sharpe cannot answer it: a long-only equity
strategy in a bull-biased window beats cash on all three while carrying beta
0.95 and zero skill. The answer is a time-series OLS regression of the
strategy's EXCESS return on factor returns, where the intercept is alpha and
its HAC t-statistic is the verdict (Fama & French 2015).

Design choices worth justifying:

* **Model ladder, not one model.** We fit CAPM → FF3 → Carhart 4 → FF5 → FF5+MOM
  and show alpha decaying across them. The decay pattern IS the finding: alpha
  of 6% under CAPM that collapses to 0.4% under Carhart 4 means the strategy is
  momentum in a trenchcoat — and this DSL's `return` and `sma_cross` signals are
  momentum-shaped by construction, so it will happen often and we should be the
  one to say it. Showing only the flattering model is the commonest way a
  backtest misleads.
* **HAC (Newey–West) standard errors.** Strategy residuals are heteroskedastic
  and serially correlated; OLS errors understate them, so an OLS t-stat
  overstates significance. Both are reported — the gap between them is itself
  informative. Lag rule is stated in the payload, never hidden.
* **Hand-rolled numpy, no statsmodels.** statsmodels drags in scipy (tens of MB)
  which matters on a free-tier host. The HAC sandwich and the Student-t tail are
  ~120 lines here and add zero dependencies.
* **FF's own RF.** Excess returns subtract the RF column shipped with the
  factors, not ^IRX, so alpha is internally consistent with the factors it is
  estimated against. Forgetting this inflates alpha by the whole cash rate
  (~5%/yr in 2023-2025 — enough to manufacture alpha out of nothing).
* **Join on the accrual date.** The engine indexes returns by accrual date
  (signal at t, return earned at t+1); Ken French indexes by the day the return
  was earned. Joining on the signal date instead shifts the series one day and
  collapses market beta from ~0.95 toward ~0.05. The overlap and unit assertions
  below exist to catch exactly that class of silent corruption.
* **Refuse rather than render.** A near-singular design still returns
  coefficients from lstsq — they are just garbage with enormous standard errors.
  Preconditions (observation count, overlap, scaled condition number, R², residual
  autocorrelation, NaNs) each produce a structured Finding instead of a confident
  wrong number, mirroring metrics._check. A DEGENERATE fit — zero residual
  variance, so t and R² are NaN rather than small — takes the same refusal path:
  every `if x < bar` test is False for NaN, so letting NaN through routes it
  into the "cleared the bar" branch, which is the worst possible failure for
  this product. Degeneracy does not always arrive as a NaN, though: when the
  strategy is an exact linear combination of the factors, SSR is rounding dust
  rather than zero, so se(α) is a few ulps, R² is exactly 1.0, and t = α/se(α)
  is a constant divided by a rounding error — finite, plausible and meaningless.
  That path is caught by the scale-free residual-variance share
  (RESID_VAR_SHARE_FLOOR), which sets `alpha_testable` per model.
  **`alpha_testable` is checked where each verdict is RENDERED, not folded into
  the shared `alpha_significant` predicate.** Three separate verdicts read that
  t — the headline alpha finding, the ladder-decay finding and the benchmark
  spanning finding — and they need different untestable behaviour, so a single
  widened predicate cannot serve them: folding testability into
  `alpha_significant` silenced the alpha verdict correctly and, in the same
  move, turned the spanning verdict from silence into "no risk-adjusted value
  added" for a clone whose intercept is a real +5%/yr. A degenerate fit must
  reach the reader as an UNAVAILABLE result carrying its reason, never as a
  verdict in either direction — a false refutation misleads harder than a false
  pass because it looks rigorous.
* **ONE degeneracy predicate, asked at every site and every level.** Six rounds
  of this module each closed a named degeneracy case and left a sibling, because
  each site RE-DERIVED degeneracy from whatever quantity was in front of it. The
  policy that replaced the instances:

  1. `_untestable_kind_of` is the predicate — FIVE conditions, returning WHICH one
     fired as a name shared by the payload reason codes, the prose and the refusal
     evidence. `_alpha_is_testable` is `not` it and `_untestable_kind(model)` is it
     applied to a `regress` row, so the bool in the payload and the reason code at
     every renderer are the same code rather than two spellings of it. They were
     two spellings until round 10, and the claim in this docstring that they were
     one predicate was FALSE: the bool checked five conditions, the name checked
     three, and a fit with a healthy residual share and a non-finite t answered
     "untestable" at the entry and "testable" at every renderer that asks for the
     name. A false claim about which guard protects what is the same defect class
     as an ungated verdict. The predicate checks the relative SHARE (SSR/SST) *and*
     the absolute SCALE of the denominator (SSR/Σy²): a pure ratio is blind to a
     collapsing denominator, and a pure absolute floor needs recalibrating for
     every return scale. Both, and they cannot disagree because there is only one
     of them.
  2. ONE CHOKEPOINT DECIDES WHAT IS PUBLISHED, and the default is WITHHELD.
     Every numeric and boolean leaf of every block this module publishes passes
     through `_withhold_unsupported`, which nulls anything whose path is not in
     `HONEST_UNDER_DEGENERACY` — a table of point estimates and refusal evidence,
     each entry carrying the reason it stays honest when the fit is dust. Asking
     the predicate at each RENDERER was the previous policy and it was not enough:
     rounds 3-8 each gated the quantity in front of the author and shipped the
     sibling three lines below it — the t gated while the 95% CI and the p-value
     beside it published (a zero-width interval that EXCLUDED ZERO, with p = 0.0,
     under an interpretation reading "alpha is NOT TESTABLE"), then those gated
     while `appraisal_ratio` published alpha over a residual scale of rounding
     dust, while `alpha_significant: True` was exported beside
     `alpha_testable: False`. A verdict has more than one RENDERING — a t, an
     interval, a probability, a ratio, a boolean are the same claim five ways —
     and hand-enumerating the renderings is done by the same reasoning that just
     missed one. A new field is now gated BY CONSTRUCTION: to publish it on a
     degenerate fit its author has to add a path to that table and write down why.
     What is withheld is the PRECISION claim, never the point estimate: α itself,
     the loadings, and the residual scale are honest arithmetic and stay, because
     suppressing them would replace a fabricated number with a fabricated absence.
     `test_every_numeric_leaf_of_a_degenerate_report_is_null_unless_declared_honest`
     walks the payload structure rather than a list of known fields, so a field
     added outside the chokepoint fails it with no test edit; its prose twin walks
     every sentence the module can emit and forbids a formatted t, p, R², interval
     bound or ratio in any of them.
  3. Every LEVEL the arithmetic happens asks it: the daily ladder, the monthly
     resampling AND the rolling-window level, which until this round was the one
     that did not — `rolling_market_beta` fitted with a bare pinv, asked nothing,
     and published dust rounded to six places. It now fits with `ols_hac`, asks
     `_untestable_kind_of`, and publishes through the chokepoint like everything
     else. When the two ladders DISAGREE about the same input the payload says so
     (`monthly.disagrees_with_daily`) instead of publishing both in silence, and
     the note names a resampling MECHANISM only for the degeneracy kind that
     mechanism actually explains.
  4. Where a quantity has an ALGEBRAIC BOUND, the bound is asserted rather than
     calibrated: SSR ≤ SST is an identity for least squares with an intercept,
     so SSR/SST > 1 is proof of dust over dust and needs no threshold story.
     Likewise `_assert_variance_identities` at the site that renders variance
     shares — the two shares partition one total and the per-factor contributions
     sum to the factor share, both exactly. Unreachable is why they are free, not
     a reason to leave them unwritten.
* **Sign is part of the verdict.** Significance is |t| ≥ 2, but the verdict is
  not: a significantly NEGATIVE alpha means the strategy reliably loses to its
  own factor exposures, and it can never read as a pass.

Every number here is produced by this module. No model writes any of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from audit.findings import Finding
from config import settings

# ---------------------------------------------------------------- constants

FACTOR_FIXTURE = settings.fixtures_dir / "factors_ff5_daily.csv"

FACTOR_COLUMNS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
RF_COLUMN = "rf"

FACTOR_LABELS = {
    "mkt_rf": "market",
    "smb": "size (SMB)",
    "hml": "value (HML)",
    "rmw": "profitability (RMW)",
    "cma": "investment (CMA)",
    "mom": "momentum (MOM)",
    "bench_excess": "benchmark",
}

# The ladder, weakest model first. Rendering these as columns of one table and
# watching alpha decay left-to-right is the whole point (Fama & French 2015).
MODEL_LADDER: dict[str, tuple[str, tuple[str, ...]]] = {
    "capm": ("CAPM", ("mkt_rf",)),
    "ff3": ("Fama-French 3", ("mkt_rf", "smb", "hml")),
    "carhart4": ("Carhart 4", ("mkt_rf", "smb", "hml", "mom")),
    "ff5": ("Fama-French 5", ("mkt_rf", "smb", "hml", "rmw", "cma")),
    "ff6": ("Fama-French 5 + momentum", ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")),
}
HEADLINE_MODEL = "ff6"

TRADING_DAYS = 252
MONTHS_PER_YEAR = 12

# Newey-West (1987) Bartlett kernel lag truncation. Stated in the payload
# rather than hidden: one trading month for daily data, six months for monthly,
# both common conventions in the empirical literature.
HAC_LAGS_DAILY = 21
HAC_LAGS_MONTHLY = 6

# Preconditions. 252 daily observations (one year) is the floor at which a
# daily regression says anything; 36 months is the academic floor for the
# monthly robustness pass.
MIN_DAILY_OBS = 252
MIN_MONTHLY_OBS = 36
# Factor data lags the present by ~6 weeks (CRSP release cadence), so a recent
# backtest legitimately loses its tail. Warn below 0.95, refuse below 0.80.
COVERAGE_WARN = 0.95
COVERAGE_FLOOR = 0.80
# Belsley scaled condition index; >30 is the textbook multicollinearity bar.
CONDITION_BAR = 30.0
VIF_BAR = 5.0
# R^2 this high means the strategy IS a factor portfolio and the intercept is
# not meaningfully interpretable as skill.
R2_FACTOR_PORTFOLIO = 0.99
# Below THIS residual share of total variance (SSR/SST) the fit is exact to
# floating point and the intercept has no testable sampling distribution: the
# intercept is whatever constant the regressors cannot reproduce, se(alpha) is
# rounding dust, and their ratio carries no information about whether that
# constant differs from zero. Measured on `rf + 1.2*mkt_rf` over 2016-2024 on
# the committed fixture: alpha = 6.5e-19/day (1.6e-16/yr), se(alpha) = 2.8e-19,
# t = 2.32 -> "alpha survives the six-factor model", in the same payload that
# said R2 of 1.000 means the strategy IS a factor portfolio.
#
# Deliberately SCALE-FREE (a share of variance), not an absolute floor on
# se(alpha): an absolute floor has to be recalibrated for every return scale and
# silently mis-fires when it is not.
#
# WHERE THE BAR SITS, by measurement. Two quantities bracket it, both measured
# over 2016-2024 (2,264 sessions) on the committed fixture by
# test_exact_fit_residual_share_is_orders_of_magnitude_below_the_bar, which
# recomputes them rather than quoting them:
#   * an EXACT linear combination's SSR/SST is pure arithmetic noise: worst of
#     160 draws with |coefficients| spanning ~1 to 1e6 is 9.05e-30.
#   * a GENUINE residual of sd 1e-9 against a ~1e-2 daily return — seven orders
#     of magnitude below the signal, far tighter than any real strategy —
#     gives SSR/SST = 5.23e-15, i.e. 23.5x ABOVE the bar, so it stays testable.
# eps = 2.22e-16 sits ~13.4 orders of magnitude above the first and ~1.4 orders
# below the second. Anything in that gap works; eps is the one value in it that
# needs no calibration story of its own.
#
# What is NOT the justification (it was stated here and it was wrong by ~4x):
# eps is not where the reported R2 stops telling a tight fit from an exact one.
# That point is eps/4 = 5.55e-17 — `1.0 - s` rounds to exactly 1.0 for every
# s <= eps/4 and to 1.0 - eps/2 just above it, measured in
# test_r2_stops_distinguishing_a_tight_fit_at_eps_over_four. So this bar is 4x
# CONSERVATIVE in that direction: it refuses every fit whose R2 we cannot tell
# from 1.0, plus the narrow band (eps/4, eps] whose R2 differs from 1.0 by an
# ulp or two while its t is still dust over dust.
RESID_VAR_SHARE_FLOOR = float(np.finfo(float).eps)
RESID_AUTOCORR_BAR = 0.50
# Conventional significance, and the Harvey-Liu-Zhu (2016) bar for a newly
# discovered factor given ~300 already published.
T_CONVENTIONAL = 2.0
T_HARVEY_LIU_ZHU = 3.0

# The engine charges cost against TRADED NOTIONAL (Σ|Δw|), while the statistic
# it reports as `annual_turnover` is ONE-WAY turnover (0.5·Σ|Δw|). See
# dsl/conventions.py, rows "Turnover (the reported statistic)" and "Traded
# notional (the cost base)", implemented at engine/backtest.py
# (`turnover = 0.5 * notional`; `net = gross - notional * bps/1e4`). Any cost
# arithmetic in this module goes through this constant, never through a bare 2.
# test_engine_still_charges_cost_on_two_times_one_way_turnover (in
# backend/tests/test_attribution.py) re-derives the ratio from a real engine run,
# so if the engine's convention ever moves, that test fails here.
NOTIONAL_PER_ONE_WAY_TURNOVER = 2.0

ROLLING_WINDOW = 252   # 12 months of trading days
ROLLING_STEP = 21      # one point per month keeps the chart payload small

# Unit assertion: mean |daily market excess return| must land in a plausible
# band. A factor-of-100 error (percent left unconverted) trips this instantly.
MKT_ABS_MEAN_BAND = (0.002, 0.020)


# ------------------------------------------------------- Student-t tail

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz).

    Numerical Recipes in C, 2nd ed., §6.4 (`betacf`). Needed because the venv
    has no scipy and we refuse to approximate a p-value we can compute exactly.
    """
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b). Numerical Recipes §6.4."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided Student-t p-value: P(|T_df| > |t|) = I_{df/(df+t^2)}(df/2, 1/2).

    Abramowitz & Stegun 26.5.27 / Numerical Recipes §6.4. Exact (to continued-
    fraction tolerance), not a normal approximation — a t of 3.2 on 40 degrees
    of freedom is not the same claim as a z of 3.2 and the product says so.
    """
    if df <= 0 or not np.isfinite(t):
        return float("nan")
    return float(_betainc(df / 2.0, 0.5, df / (df + t * t)))


def t_hurdle_for_p(p_target: float, df: int) -> float:
    """Smallest |t| whose two-sided Student-t p-value is <= p_target.

    Bisection on the exact tail above; used to render the multiple-testing bar
    concretely ('at 40 variations tried, alpha needs t > 3.4; yours is 2.3').
    """
    if not (0.0 < p_target < 1.0) or df <= 0:
        return float("nan")
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > p_target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ------------------------------------------------------------ OLS + HAC

@dataclass
class OLSFit:
    """Raw regression output, pre-annualization. `coefs[0]` is the intercept."""
    coefs: np.ndarray
    se_ols: np.ndarray
    se_hac: np.ndarray
    resid: np.ndarray
    r2: float
    adj_r2: float
    # SSR/SST — the residual share of total variance. Carried separately from r2
    # because `1 - r2` cannot recover it: r2 saturates at exactly 1.0 once
    # SSR/SST drops below eps/2, which is the regime this number exists to name.
    resid_var_share: float
    # SSR/Σy² — the residual share of the UNCENTERED second moment. The centered
    # share above is a ratio, and a ratio only sees its NUMERATOR: it catches an
    # exact factor combination (SSR → dust against a healthy SST) and is blind
    # when the DENOMINATOR collapses. If y has no variation of its own — a
    # strategy that never traded, against a flat risk-free rate, so excess = −RF
    # is a nonzero CONSTANT — then SST is rounding dust as well, SSR/SST comes
    # out ABOVE 1 (measured 3507 on the committed fixture) and every guard that
    # asks "is the ratio small?" answers no. Σy² does not collapse with the
    # variation, so this share stays a true measure of how much of the dependent
    # variable the fit failed to reproduce, and it is dust in BOTH degenerate
    # regimes.
    resid_uncentered_share: float
    n_obs: int
    n_params: int
    condition_number: float
    hac_lags: int

    @property
    def df_resid(self) -> int:
        return self.n_obs - self.n_params


def _scaled_condition_number(X: np.ndarray) -> float:
    """Belsley scaled condition index: normalize each column to unit length
    before taking the condition number.

    Without the scaling the intercept column of ones dwarfs factor columns of
    magnitude ~0.01 and the condition number is enormous for every regression,
    collinear or not — the check would fire always and mean nothing.
    """
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    return float(np.linalg.cond(X / norms))


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> OLSFit:
    """OLS with Newey-West (1987) HAC standard errors.

    S_hat = Σ_t u_t² x_t x_t'
          + Σ_{l=1..L} w_l Σ_{t>l} u_t u_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')
    with Bartlett weights w_l = 1 - l/(L+1); then
    Var(β̂) = (X'X)^{-1} S_hat (X'X)^{-1} × T/(T-k) (small-sample correction).

    Newey & West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity
    and Autocorrelation Consistent Covariance Matrix", Econometrica 55(3),
    703-708. Implementation follows the Stata `newey` manual entry:
    https://www.stata.com/manuals/tsnewey.pdf
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(f"cannot fit {k} parameters on {n} observations")

    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    coefs = xtx_inv @ (X.T @ y)
    resid = y - X @ coefs

    sst = float(((y - y.mean()) ** 2).sum())
    uss = float((y ** 2).sum())
    ssr = float((resid ** 2).sum())
    resid_var_share = ssr / sst if sst > 0 else float("nan")
    # SST > 0 does not imply the variation is REAL: a constant y has SST of
    # rounding dust, which is > 0. Σy² is the denominator that survives that.
    resid_uncentered_share = ssr / uss if uss > 0 else float("nan")
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if sst > 0 else float("nan")

    # Classical OLS errors, for the side-by-side comparison.
    sigma2 = ssr / (n - k)
    se_ols = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0.0, None))

    # HAC sandwich.
    u_x = X * resid[:, None]                      # u_t x_t, shape (n, k)
    s_hat = u_x.T @ u_x
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)           # Bartlett
        gamma = u_x[lag:].T @ u_x[:-lag]
        s_hat = s_hat + w * (gamma + gamma.T)
    cov_hac = xtx_inv @ s_hat @ xtx_inv * (n / (n - k))
    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 0.0, None))

    return OLSFit(
        coefs=coefs, se_ols=se_ols, se_hac=se_hac, resid=resid,
        r2=r2, adj_r2=adj_r2, resid_var_share=resid_var_share,
        resid_uncentered_share=resid_uncentered_share, n_obs=n, n_params=k,
        condition_number=_scaled_condition_number(X), hac_lags=max_lag,
    )


def _lag1_autocorr(u: np.ndarray) -> float:
    if len(u) < 3:
        return float("nan")
    a, b = u[1:], u[:-1]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _vifs(frame: pd.DataFrame) -> dict[str, float]:
    """VIF_k = 1/(1 - R²_k) from regressing factor k on the other factors.

    HML, RMW and CMA are correlated by construction — Fama & French (2015)
    found HML redundant once RMW and CMA are present — so collinearity here is
    expected, not a bug. Flagging it tells the user that the individual
    loadings are unreliable even though the fit and the intercept are fine.
    """
    cols = list(frame.columns)
    out: dict[str, float] = {}
    if len(cols) < 2:
        return {c: 1.0 for c in cols}
    for col in cols:
        others = [c for c in cols if c != col]
        y = frame[col].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(dtype=float)
                                                     for c in others])
        fit_coefs = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        resid = y - X @ fit_coefs
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else 0.0
        out[col] = float(1.0 / max(1.0 - r2, 1e-12))
    return out


# -------------------------------------------------------------- factors

@dataclass
class FactorData:
    """The parsed factor panel plus everything needed to disclose its origin."""
    frame: pd.DataFrame
    synthetic: bool
    provenance: str
    path: str
    start: str
    end: str


def _read_header_comments(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open() as fh:
        for raw in fh:
            if not raw.startswith("#"):
                break
            lines.append(raw.lstrip("#").strip())
    return lines


def load_factors(path: Path | None = None) -> FactorData:
    """Load the committed daily factor fixture (decimal returns, accrual-dated).

    The fixture header declares `SYNTHETIC: yes|no`. That flag is load-bearing:
    if it ever says yes, every downstream payload and a dedicated audit finding
    say the attribution is illustrative only. Synthetic factor data must never
    be able to masquerade as Ken French's.
    """
    path = Path(path) if path is not None else FACTOR_FIXTURE
    if not path.exists():
        raise FileNotFoundError(f"factor fixture missing: {path}")

    header = _read_header_comments(path)
    synthetic = any(ln.lower().startswith("synthetic:") and "yes" in ln.lower()
                    for ln in header)
    provenance = " ".join(ln for ln in header if ln) or "(no provenance header)"

    frame = pd.read_csv(path, comment="#", index_col=0, parse_dates=True)
    frame.index.name = "date"
    missing = [c for c in (*FACTOR_COLUMNS, RF_COLUMN) if c not in frame.columns]
    if missing:
        raise ValueError(f"factor fixture {path.name} is missing columns: {missing}")
    frame = frame[[*FACTOR_COLUMNS, RF_COLUMN]].astype(float)

    if frame.isna().any().any():
        raise ValueError(
            f"factor fixture {path.name} contains NaN — refusing to attribute "
            f"against broken factor data"
        )
    # Units assertion (research: a factor-of-100 error is silent otherwise).
    mkt_abs = float(frame["mkt_rf"].abs().mean())
    lo, hi = MKT_ABS_MEAN_BAND
    if not (lo <= mkt_abs <= hi):
        raise ValueError(
            f"factor fixture {path.name}: mean |mkt_rf| is {mkt_abs:.5f}, outside "
            f"the plausible daily band [{lo}, {hi}] — the file is probably still "
            f"in percent rather than decimal"
        )
    return FactorData(
        frame=frame, synthetic=synthetic, provenance=provenance, path=str(path),
        start=str(frame.index[0].date()), end=str(frame.index[-1].date()),
    )


# ------------------------------------------------------- regression table

def _check_no_nan(series: pd.Series, what: str) -> pd.Series:
    """Same discipline as engine.metrics._check: a NaN here is an upstream
    alignment bug, and skipping it silently turns broken data into a confident
    wrong number.

    ±inf is the same bug and must take the same path. `isna()` is False for an
    infinity, so testing it alone let an inf through to the regression, where it
    produced NaN coefficients and came back out as `degenerate_regression` — a
    refusal, but the wrong one: it blames a series that never left cash for a
    series that is arithmetically broken.
    """
    bad = ~np.isfinite(series.to_numpy(dtype=float))
    if bad.any():
        raise ValueError(
            f"{what} contains {int(bad.sum())} NaN or non-finite value(s) — "
            f"refusing to run a factor regression on broken data"
        )
    return series


def _annualize_alpha(alpha_per_period: float, periods: int) -> tuple[float, float]:
    """→ (arithmetic, geometric). Arithmetic (periods × α) is the academic
    convention and keeps the t-stat interpretable — the t-statistic is invariant
    to annualization, so only the point estimate is scaled, never the ratio.
    The geometric figure is the one a client can compound."""
    arithmetic = alpha_per_period * periods
    geometric = (1.0 + alpha_per_period) ** periods - 1.0
    return float(arithmetic), float(geometric)


def _untestable_kind_of(alpha_t_hac: float, r2: float, resid_var_share: float,
                        resid_uncentered_share: float) -> str:
    """THE degeneracy predicate — the one condition set, and WHICH condition
    fired. Empty string means the intercept has a sampling distribution.

    Does this fit's intercept have one at all? FIVE ways it does not, and the
    first three were each added after a fabricated verdict reached a reader:

      1. t or R² came back NaN (SST exactly 0, zero residual variance);
      2. the fit is exact to floating point, so t is rounding-error-over-
         rounding-error and SSR/SST is dust (RESID_VAR_SHARE_FLOOR);
      3. SSR/SST > 1, which is ARITHMETICALLY IMPOSSIBLE for least squares with an
         intercept — SSR <= SST is an identity there — so a share above 1, and the
         negative R² that accompanies it, is proof the ratio is dust over dust
         rather than a measurement. When a quantity has an algebraic bound, assert
         the bound: an impossible value is the cheapest degeneracy detector there
         is;
      4. the UNCENTERED share SSR/Σy² is dust. 2 and 3 both read the same ratio,
         and a ratio cannot see its own DENOMINATOR collapse: for a strategy that
         never traded against a flat RF the excess return is a nonzero CONSTANT,
         SST is rounding dust, SSR/SST came out 3507 and every gated verdict fired
         with a t of -3.7e14. Σy² does not collapse with the variation;
      5. the shares are all legal and the T ITSELF is not finite — se(α) came back
         exactly 0 while SSR did not. No reachable input produces it, and it is
         here because it is the ONE condition the two halves of this module's
         "one predicate" used to disagree about (see below), not because it was
         ever seen.

    NaN fails every `>` comparison, so 1 is covered several times over.

    ONE FUNCTION, TWO RETURN SHAPES. This used to be two functions —
    `_alpha_is_testable` returning the bool and `_untestable_kind` returning the
    name — and the module docstring claimed they were one predicate. They were
    not: this one checked five conditions and the name-returning one checked
    three, so a fit with a healthy share 0.5 and a NaN t answered
    `alpha_testable = False` at the entry and `kind = ""` — "testable" — at every
    renderer that asks for the name, and `_untestable_reason` described it as
    "residual variance 5.0e-01 of total, at or below the 2.2e-16 floor", a
    sentence its own quoted number contradicts. A false claim about WHICH guard
    protects what is the same defect class as an ungated verdict, so the two are
    now literally the same code: `_alpha_is_testable` is `not this`, and
    `_untestable_kind(model)` is this applied to a model dict. Order is
    load-bearing and is documented on `_untestable_kind`.
    """
    if not np.isfinite(resid_var_share):
        return "no_variation"
    if resid_var_share <= RESID_VAR_SHARE_FLOOR:
        return "no_testable_intercept"
    uncentered_is_dust = not (np.isfinite(resid_uncentered_share)
                              and resid_uncentered_share > RESID_VAR_SHARE_FLOOR)
    if resid_var_share > 1.0 or uncentered_is_dust:
        return "no_dependent_variation"
    # Last, so that every reachable degeneracy keeps the more specific name it
    # already had: on all three reachable regimes the shares fire first and the t
    # is finite dust, not NaN.
    if not (np.isfinite(alpha_t_hac) and np.isfinite(r2)):
        return "no_intercept_standard_error"
    return ""


def _alpha_is_testable(alpha_t_hac: float, r2: float, resid_var_share: float,
                       resid_uncentered_share: float) -> bool:
    """The predicate in bool form — `not _untestable_kind_of(...)`, and nothing
    else, so the flag in the payload and the reason code every renderer reads can
    never disagree about the same fit.

    Kept as a named function because `alpha_testable` is what `regress` publishes
    and what every consumer checks, and because 3 and 4 above are REDUNDANT on
    every input reachable from the product — the flat-RF constant trips both at
    once — so no end-to-end test can say which one is load-bearing, and deleting
    either leaves the other quietly covering for it until an input arrives that
    only one of them sees.
    test_alpha_testable_needs_every_one_of_its_five_conditions drives them one at
    a time on constructed inputs. It is deliberately NOT the shared
    `alpha_significant` predicate: folding testability into a predicate three
    unrelated verdicts read is the mistake this module's docstring records.
    """
    return not _untestable_kind_of(alpha_t_hac, r2, resid_var_share,
                                   resid_uncentered_share)


# --------------------------------------------------------- THE CHOKEPOINT

def _finite(x: object) -> bool:
    """True iff `x` is a real number this module may compare against a bar.

    `np.isfinite(nan)` is False but `np.isfinite(None)` RAISES, and the
    chokepoint below leaves None where a number used to be, so every guard in
    this module asks this instead of calling `np.isfinite` on a payload value. A
    withheld quantity has to take the same branch a NaN one takes — never a
    TypeError, and above all never the "cleared the bar" branch, which is the
    original sin this whole effort has been unwinding.
    """
    if x is None or isinstance(x, bool):
        return False
    if not isinstance(x, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(x))


# READ THIS TABLE BEFORE ADDING A FIELD TO ANYTHING THIS MODULE PUBLISHES.
#
# It is an ALLOW-LIST, and that direction is the entire point. Every numeric and
# boolean leaf of every block this module publishes passes through
# `_withhold_unsupported`, which — on a fit with no testable intercept — nulls
# every leaf whose path is NOT listed here. A field added tomorrow is therefore
# withheld BY CONSTRUCTION, and its author has to come to this table and write
# down why the quantity stays honest when the fit is dust.
#
# Why the default had to be inverted: six consecutive rounds of this module each
# gated the named quantity and shipped a sibling, usually in the SAME function
# three lines below the new gate — the t was gated while the 95% CI and the
# p-value beside it were not; then those were gated and `appraisal_ratio`
# (alpha over a residual scale that is rounding dust) went on publishing from
# three lines further down; `alpha_significant` was a fourth rendering of the
# same verdict in the export. Every one of those was found by hand-enumerating
# the sites, by the same reasoning that had just missed one. Enumeration by hand
# does not converge; a default does.
#
# Paths are dotted; `*` matches exactly one segment (a dict key or a list index).
# Only two kinds of entry qualify:
#   (a) POINT ESTIMATES. The intercept IS whatever constant the regressors
#       cannot reproduce, and a loading IS the exposure; withholding either
#       would replace a fabricated number with a fabricated absence — which is
#       what `_untestable_reason` tells the same reader in words.
#   (b) THE REFUSAL'S OWN EVIDENCE: quantities whose dustiness IS the reason for
#       refusing. They are published only ever beside `alpha_testable: False`
#       and a reason code; no renderer quotes them (the prose walker in
#       test_attribution.py proves that over every sentence this module can
#       emit); and `_untestable_kind` reads four of them, so withholding them
#       would delete the reason code along with the number.
# A quantity that is neither — a ratio, a share, an interval, a probability, a
# significance boolean — belongs nowhere near this table.
HONEST_UNDER_DEGENERACY: dict[str, str] = {
    # ---- (a) point estimates
    "alpha_per_period": (
        "the point estimate: the intercept IS the constant the regressors "
        "cannot reproduce, whatever its standard error is"),
    "alpha_annual": "the same estimate annualized (periods x alpha, exact)",
    "alpha_annual_geometric": "the same estimate compounded, no precision claim",
    "betas.*.beta": (
        "the loading is a real exposure even when the fit is exact — the "
        "spanning beta of a clone IS 1.00 and the module quotes it"),
    "beta": "rolling-window market loading: the same point estimate, per window",
    # ---- (b) the refusal's own evidence
    "alpha_se_hac": (
        "the dust itself, reported as dust: it is the evidence for the refusal "
        "and the number the diagnostics quote when explaining it"),
    "alpha_t_hac": (
        "the division that was ATTEMPTED, published beside alpha_testable=False "
        "as the diagnostic for why there is no verdict; the prose walker proves "
        "no sentence quotes it, and `_untestable_kind` reads it"),
    "alpha_t_ols": "the same diagnostic on the classical standard error",
    "r2": (
        "the tell, not the claim: exactly 1.0 says the fit is exact and a "
        "NEGATIVE value is the arithmetically impossible one the refusal points "
        "at. Read by `_untestable_kind`; never quoted by a renderer"),
    "adj_r2": "the same tell with the degrees-of-freedom correction",
    "resid_var_share": (
        "SSR/SST, the quantity the floor is measured against — the refusal's "
        "primary evidence, read by `_untestable_kind` and quoted by "
        "`_untestable_reason` as the reason rather than as a result"),
    "resid_uncentered_share": (
        "SSR/Sum(y^2), the denominator-collapse evidence, same role"),
    "resid_vol_annual": (
        "the residual SCALE, a measurement of how much dust there is; it is "
        "the appraisal RATIO built on top of it that is not supported"),
    "betas.*.se_hac": "the loading's dust, same role as alpha_se_hac",
    "betas.*.vif": (
        "a property of the FACTOR matrix alone — the same value on any "
        "dependent variable, so this fit's degeneracy says nothing about it"),
    "vif.*": "the same VIFs, as the diagnostics block republishes them",
    "condition_number": (
        "a property of the design matrix alone, not of the dependent variable"),
    "factor_variance": (
        "x'Fx from the published loadings and the factor covariance: a property "
        "of the exposures, which the fit does support"),
    "specific_variance": (
        "the residual scale squared — the same measurement as resid_vol_annual"),
    # ---- structural counts and constants (not derived from this fit at all)
    "n_obs": "a count of the observations the fit actually used",
    "df_resid": "a count of residual degrees of freedom, n - k",
    "hac_lags": "the lag truncation actually used, a setting",
    "annualization": "periods per year, a constant of the frequency",
    "attempts": "the pipeline's attempt counter, an input",
    "available": (
        "the refusal FLAG itself: nulling it would withhold the statement that "
        "something was withheld"),
    "alpha_testable": "the predicate's own answer, for the same reason",
    "verdict.available": "the same flag inside a verdict sub-dict",
    "alpha_ci95_t_critical": (
        "t_{0.975,df}: a Student-t quantile of the residual degrees of freedom "
        "alone — the same number on any input with the same df, which is why "
        "`t_hurdle_bonferroni` survives a refused multiple-testing verdict too"),
    "t_hurdle_bonferroni": (
        "a property of the attempt count and df, not of this regression"),
    "t_hurdle_harvey_liu_zhu": "a constant from the literature (3.0)",
}

_ALLOW_PATTERNS: tuple[tuple[str, ...], ...] = tuple(
    tuple(p.split(".")) for p in HONEST_UNDER_DEGENERACY)


def _path_is_honest(path: tuple[str, ...]) -> bool:
    return any(
        len(pattern) == len(path)
        and all(seg == "*" or seg == got for seg, got in zip(pattern, path))
        for pattern in _ALLOW_PATTERNS
    )


def _strip_unsupported(value, path: tuple[str, ...], withheld: list[str]):
    if isinstance(value, dict):
        return {k: _strip_unsupported(v, path + (str(k),), withheld)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_unsupported(v, path + (str(i),), withheld)
                for i, v in enumerate(value)]
    if isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
        if not _path_is_honest(path):
            withheld.append(".".join(path) or "<root>")
            return None
    return value


def _withhold_unsupported(block: dict, untestable_kind: str, *,
                          record: bool = True) -> dict:
    """THE CHOKEPOINT. Every derived quantity this module publishes goes through
    here, and on an untestable fit everything not in `HONEST_UNDER_DEGENERACY`
    comes out None.

    `untestable_kind` is the ONE predicate's answer (`_untestable_kind_of`, or
    `_untestable_kind` on an assembled row) — empty string means the block is
    published as computed. So the decision of WHETHER to withhold and the reason
    code in the payload are the same value, and the decision of WHAT to withhold
    is one table instead of one `if` per field at each of a dozen sites.
    Withheld values are None rather than NaN: None survives JSON export as null,
    and a consumer that tests `lo > 0` on it RAISES instead of silently taking
    the False branch, which is exactly how a NaN t-statistic once walked through
    every `<` comparison in this module and came out the other side as a pass.

    `record=False` omits the `withheld` inventory, for blocks published one per
    chart point where the inventory would outweigh the data.
    """
    withheld: list[str] = []
    out = (_strip_unsupported(block, (), withheld) if untestable_kind
           else dict(block))
    if record:
        out["withheld"] = withheld
    return out


def regress(excess: pd.Series, factors: pd.DataFrame, columns: tuple[str, ...],
            periods_per_year: int, hac_lags: int, label: str) -> dict:
    """Fit one model of the ladder and return the full PM-facing table row.

    r_t − RF_t = α + Σ_k β_k f_kt + e_t  (Fama & French 2015, eq. 5 for FF5)

    Reported: annualized α with its 95% CI (exact Student-t critical value on
    the residual degrees of freedom, not 1.96), HAC and OLS t-stats, every loading
    with both t-stats and its VIF, R², adjusted R², annualized residual
    (idiosyncratic) volatility, and the appraisal ratio — α per unit of
    idiosyncratic risk, i.e. the Sharpe of the hedged residual, which is what
    actually determines how much capital a PM would allocate.

    EVERY one of those is computed unconditionally here and then published
    through `_withhold_unsupported`, the module's one chokepoint, which nulls
    whatever a degenerate fit cannot support. There is deliberately no per-field
    `if alpha_testable` left in this function: five rounds of those each gated
    the field in front of the author and left the next one three lines below.
    """
    X_cols = [factors[c].to_numpy(dtype=float) for c in columns]
    X = np.column_stack([np.ones(len(excess))] + X_cols)
    y = excess.to_numpy(dtype=float)
    fit = ols_hac(y, X, hac_lags)

    alpha = float(fit.coefs[0])
    alpha_se_hac = float(fit.se_hac[0])
    alpha_t_hac = alpha / alpha_se_hac if alpha_se_hac > 0 else float("nan")
    alpha_t_ols = alpha / float(fit.se_ols[0]) if fit.se_ols[0] > 0 else float("nan")
    alpha_arith, alpha_geom = _annualize_alpha(alpha, periods_per_year)
    # THE ONE PREDICATE, asked once per fit. `alpha_testable` is its bool form and
    # `untestable_kind` its name; the chokepoint at the bottom of this function
    # takes the same value, so the flag in the payload, the reason code in the
    # prose and the decision about WHICH numbers survive are one answer rather
    # than three derivations of it.
    kind = _untestable_kind_of(
        alpha_t_hac, fit.r2, fit.resid_var_share, fit.resid_uncentered_share)
    alpha_testable = not kind
    # The lag-1 residual autocorrelation is ALSO a claim, and `_lag1_autocorr`
    # has no dispersion test of its own: it refuses only an EXACTLY constant
    # residual (sd == 0), so on a degenerate fit it returns a tidy correlation of
    # rounding dust (measured +0.01 on residuals of sd 3.1e-17) which the
    # diagnostics block compares against a bar and the reliability PASS prints
    # verbatim. It is computed here and WITHHELD by the chokepoint, like every
    # other unsupported quantity, rather than by a test of its own.
    resid_autocorr1 = _lag1_autocorr(fit.resid)
    # Exact Student-t critical value on the regression's own residual degrees of
    # freedom, not the normal 1.96. The module refuses to approximate the
    # p-value; approximating the interval that reports the same uncertainty
    # would be incoherent, and at the 36-month monthly floor (df ≈ 29) 1.96
    # understates the half-width by ~4%.
    t_crit = t_hurdle_for_p(0.05, fit.df_resid)
    if not np.isfinite(t_crit):
        t_crit = 1.96
    # PRINCIPLE 2 at the two OTHER claims this one function publishes, and the
    # reason the gate is now a chokepoint instead of an `if` here. The t was
    # gated in round 8 and the residual autocorrelation beside it, and then three
    # lines later the same fit published a 95% INTERVAL and a p-VALUE — the same
    # significance verdict in interval form and in probability form — ungated;
    # and three lines below THAT, the appraisal ratio survived another whole round.
    # Measured on the live tree before round 8: a constant 0.1%/day strategy
    # over the fixture's 793-session flat-RF run published
    # `alpha_annual_ci95 = [+20.1600000000%, +20.1600000000%]` (width 2.1e-15)
    # with `alpha_p_hac = 0.0`, in the same payload whose interpretation reads
    # "alpha is NOT TESTABLE"; the exact benchmark clone published a spanning CI
    # of [+5.000000000000%, +5.000000000000%] beside a finding promising that "no
    # intercept or t-statistic is quoted". Systematic, not a corner: 794 of the
    # 1,770 degenerate fits on the test suite's first axis, and 111 of 111 on its
    # second, published a 95% CI that EXCLUDED ZERO — widths ~1e-17 to ~1e-14.
    # Both counts are pinned in test_attribution.py, so a change that makes the
    # gate vacuous shows up as a changed number rather than as silence.
    #
    # The point ESTIMATE is honest arithmetic — (0.001 − 0.0002) × 252 = 20.16%
    # is what the constant IS — and it stays. What the fit cannot support is the
    # PRECISION claim: that the constant is pinned to 1e-15 and differs from zero
    # with p = 0. So the estimate is published and the interval around it is
    # withheld, which is the same split `_untestable_reason` already puts in
    # words ("the intercept is whatever that constant is ... but its standard
    # error is rounding dust").
    #
    # The withheld shape is None, not NaN, and uniformly so: None survives JSON
    # export as null, and a consumer that tests `lo > 0` on it RAISES instead of
    # silently taking the False branch — which is the failure mode that put a NaN
    # t through every `<` comparison in this module and out the other side as a
    # pass. Formed unconditionally here; withheld at the bottom.
    ci_half = t_crit * alpha_se_hac * periods_per_year
    alpha_ci95 = [alpha_arith - ci_half, alpha_arith + ci_half]
    alpha_p_hac = t_two_sided_p(alpha_t_hac, fit.df_resid)

    # σ̂ = sqrt(SSR/(n−k)) — the regression's unbiased residual scale, the same
    # denominator the OLS standard errors use. resid.std(ddof=1) is sqrt(SSR/(n−1)),
    # which is biased LOW and therefore biases the appraisal ratio HIGH.
    ssr = float((fit.resid ** 2).sum())
    resid_sigma = math.sqrt(ssr / fit.df_resid) if fit.df_resid > 0 else float("nan")
    resid_vol_annual = float(resid_sigma * math.sqrt(periods_per_year))
    # α per unit of idiosyncratic risk. On a degenerate fit the NUMERATOR is a
    # real constant and the DENOMINATOR is rounding dust, so this ratio is the
    # most flattering fabrication the row can carry: 1.6e-16 over 4.9e-16 is an
    # appraisal ratio of 0.33, indistinguishable from a real one. It sat three
    # lines below round 8's gate for a full round. It is a RATIO, so it is not on
    # the honest-under-degeneracy table and the chokepoint nulls it.
    appraisal = (alpha_arith / resid_vol_annual) if resid_vol_annual > 0 else float("nan")

    vifs = _vifs(factors[list(columns)])
    betas: dict[str, dict] = {}
    for i, name in enumerate(columns, start=1):
        se_h = float(fit.se_hac[i])
        se_o = float(fit.se_ols[i])
        # PRINCIPLE 2 at the LOADING level. A degenerate fit's loadings are
        # estimated against the same dust denominator as its intercept — on the
        # flat-RF constant the six loading t's came out between -1e14 and +9e13 —
        # so a per-factor t is exactly as unsupported as alpha's, and it is the
        # number a reader uses to decide which exposure is "real". The LOADING
        # ITSELF stays (it is on the honest table): an exact factor combination
        # has genuinely reproducible betas, and `_absorbing_factor` deliberately
        # ranks by β×mean(f) rather than by |t|, so nothing inside this module
        # loses information. `betas.*.t_hac` and `.t_ols` are on no table, so the
        # chokepoint nulls them — including for a factor added to the ladder
        # tomorrow, whose row nobody will remember to gate.
        betas[name] = {
            "label": FACTOR_LABELS.get(name, name),
            "beta": float(fit.coefs[i]),
            "se_hac": se_h,
            "t_hac": float(fit.coefs[i] / se_h) if se_h > 0 else float("nan"),
            "t_ols": float(fit.coefs[i] / se_o) if se_o > 0 else float("nan"),
            "vif": vifs.get(name, float("nan")),
        }

    row = {
        "key": label,
        "factors": list(columns),
        "alpha_per_period": alpha,
        "alpha_se_hac": alpha_se_hac,
        "alpha_annual": alpha_arith,
        "alpha_annual_geometric": alpha_geom,
        "alpha_annual_ci95": alpha_ci95,
        "alpha_ci95_t_critical": float(t_crit),
        "alpha_t_hac": alpha_t_hac,
        "alpha_t_ols": alpha_t_ols,
        "alpha_p_hac": alpha_p_hac,
        "alpha_testable": alpha_testable,
        # The reason code, carried beside the flag so a consumer never has to
        # re-derive it — and the same string the chokepoint acted on.
        "untestable_kind": kind or None,
        # A statement about the t-statistic ALONE: |t| >= 2 (NaN is False), and
        # deliberately NOT conjoined with `alpha_testable` — three call sites read
        # it and they need DIFFERENT untestable behaviour, so widening the
        # predicate moves all three at once (that is how the spanning verdict once
        # flipped from silence to "no risk-adjusted value added" on a clone with a
        # real +5%/yr). What used to be wrong is that a row saying
        # `alpha_testable: False` published `alpha_significant: True` beside it —
        # a fourth RENDERING of the verdict, exported to JSON. It is a
        # significance boolean, so it is on no honest table and the chokepoint
        # nulls it: None is "no verdict exists", which is neither True nor False,
        # the same shape `survives_bonferroni` uses.
        "alpha_significant": bool(
            _finite(alpha_t_hac) and abs(alpha_t_hac) >= T_CONVENTIONAL
        ),
        "betas": betas,
        "r2": fit.r2,
        "adj_r2": fit.adj_r2,
        "resid_var_share": fit.resid_var_share,
        "resid_uncentered_share": fit.resid_uncentered_share,
        "resid_vol_annual": resid_vol_annual,
        "appraisal_ratio": appraisal,
        "n_obs": fit.n_obs,
        "df_resid": fit.df_resid,
        "condition_number": fit.condition_number,
        "resid_autocorr1": resid_autocorr1,
        "hac_lags": fit.hac_lags,
        "annualization": periods_per_year,
    }
    # THE CHOKEPOINT. Everything above is arithmetic; what a reader is allowed to
    # see is decided in exactly one place, from the one predicate's answer.
    return _withhold_unsupported(row, kind)


def rolling_market_beta(excess: pd.Series, factors: pd.DataFrame,
                        window: int = ROLLING_WINDOW,
                        step: int = ROLLING_STEP) -> list[dict]:
    """Rolling 12-month CAPM fit → one gated dict per window.

    A full-sample beta averages over regimes. A strategy whose market beta
    swings between 0.3 and 1.3 is a market-timing bet, not a stable exposure,
    and its full-sample alpha is close to meaningless because the linear model
    is misspecified. Showing the path lets a PM see regime dependence instead of
    taking one number on trust.

    THE THIRD LEVEL AT WHICH THIS MODULE FITS A REGRESSION, and until this round
    the only one that never asked the degeneracy predicate — which is what made
    the docstring's policy item 3 ("every LEVEL the arithmetic happens asks it")
    false. It was not a harmless omission: on an exact factor combination every
    window here is degenerate, and the rolling series was published with a
    `round(..., 6)` that prints dust as a tidy 0.0 with no reason code attached.
    Each window now fits with `ols_hac` (the same estimator as the ladder, so the
    window has a real standard error to ask about), asks `_untestable_kind_of`
    (the same predicate), and publishes through `_withhold_unsupported` (the same
    chokepoint). The two point estimates survive per the honest table; a derived
    quantity added to a window row tomorrow does not.
    """
    if len(excess) < window:
        return []
    mkt = factors["mkt_rf"].to_numpy(dtype=float)
    y = excess.to_numpy(dtype=float)
    out: list[dict] = []
    for end in range(window, len(y) + 1, step):
        lo = end - window
        Xw = np.column_stack([np.ones(window), mkt[lo:end]])
        fit = ols_hac(y[lo:end], Xw, HAC_LAGS_DAILY)
        a, se = float(fit.coefs[0]), float(fit.se_hac[0])
        kind = _untestable_kind_of(a / se if se > 0 else float("nan"), fit.r2,
                                   fit.resid_var_share, fit.resid_uncentered_share)
        out.append(_withhold_unsupported({
            "date": str(excess.index[end - 1].date()),
            "beta": round(float(fit.coefs[1]), 6),
            "alpha_annual": round(a * TRADING_DAYS, 6),
            "alpha_testable": not kind,
            "untestable_kind": kind or None,
        }, kind, record=False))
    return out


def _assert_variance_identities(factor_share: float, specific_share: float,
                                per_factor_share: dict[str, float]) -> None:
    """PRINCIPLE 4 at the site that renders SHARES: where a quantity has an
    ALGEBRAIC bound, assert the bound instead of calibrating a threshold for it.

    Two identities, neither reachable from any input:
      1. the two shares PARTITION one total, so they sum to exactly 1;
      2. x'Fx expands exactly into Σ_k x_k (Fx)_k, so the per-factor
         contributions sum to the factor share.
    Unreachable is the reason they are free, not a reason to skip them: an
    identity that fails is proof that the arithmetic above it is not the
    arithmetic the sentence beside it describes, which is the failure mode this
    module has spent six rounds on. A named function rather than two bare
    `assert`s at the call site so the bounds can be driven directly — an
    unexercised guard is a guess, by this module's own standard.
    """
    assert abs(factor_share + specific_share - 1.0) < 1e-9, (
        f"factor and specific shares must partition the total variance exactly: "
        f"{factor_share} + {specific_share} != 1")
    assert abs(sum(per_factor_share.values()) - factor_share) < 1e-9, (
        f"per-factor contributions must sum to the factor share: "
        f"{sum(per_factor_share.values())} != {factor_share}")


def risk_decomposition(model: dict, factors: pd.DataFrame,
                       periods_per_year: int) -> dict:
    """Split total variance into factor variance and specific variance.

    var(R_p) = x' F x + σ²_specific, where x is the loading vector and F the
    (annualized) factor covariance matrix — the honest small version of a
    Barra-style decomposition, built from the regression already fitted. The
    headline is one number: what share of this strategy's variance is systematic.
    Deliberately NOT called a commercial risk model; we have no fundamental data.

    A SITE THAT RENDERS A CLAIM, so it asks the module's one degeneracy
    predicate (`_untestable_kind`) rather than trusting `total > 0`. On the
    reachable degenerate input — a strategy that never traded over a flat-RF run
    — the loadings are dust and the residual scale is dust, `total` is ~1e-33
    which is finite and positive, and this published `available: True` with
    "1.6% of variance is systematic, 98.4% specific": dust over dust, to three
    significant figures, on shares that still sum to 1. It has no reader in this
    repo today, which is exactly why it would have been fabricating quietly.
    """
    names = model["factors"]
    x = np.array([model["betas"][n]["beta"] for n in names], dtype=float)
    F = np.cov(factors[names].to_numpy(dtype=float).T, ddof=1) * periods_per_year
    F = np.atleast_2d(F)
    factor_var = float(x @ F @ x)
    specific_var = float(model["resid_vol_annual"] ** 2)
    total = factor_var + specific_var
    # Two ways to have nothing to publish, and they carry DIFFERENT reason codes:
    # the fit has no testable intercept (the predicate's own name, so this
    # refusal, the spanning verdict, the multiple-testing verdict and the prose
    # all name the same condition), or the variance is literally zero.
    usable = _finite(total) and total > 0
    kind = _untestable_kind(model) or ("" if usable else "zero_total_variance")
    if usable:
        Fx = F @ x
        factor_share = factor_var / total
        specific_share = specific_var / total
        per_factor = {n: float(x[i] * Fx[i] / total) for i, n in enumerate(names)}
        total_vol = math.sqrt(total)
        _assert_variance_identities(factor_share, specific_share, per_factor)
    else:
        factor_share = specific_share = total_vol = float("nan")
        per_factor = {n: float("nan") for n in names}
    # One key set on both paths (integration contract): a consumer that reads
    # rd["factor_share"] must not KeyError on the refusal. The shares, the total
    # volatility and the per-factor contributions are RATIOS, so they are on no
    # honest table and the chokepoint nulls every one of them — including a
    # per-factor entry for a factor added tomorrow.
    return _withhold_unsupported({
        "available": not kind,
        "reason": kind or None,
        "factor_variance": factor_var if _finite(factor_var) else None,
        "specific_variance": specific_var if _finite(specific_var) else None,
        "factor_share": factor_share,
        "specific_share": specific_share,
        "total_vol_annual": total_vol,
        # contribution_k = x_k (Fx)_k / σ²_p; these sum to the factor share.
        "per_factor_share": per_factor,
    }, kind)


# ------------------------------------------------------------- findings

def _fmt_t(t: float | None) -> str:
    """`_finite`, not `np.isfinite`: a withheld quantity is None, and a formatter
    that raises on it would turn the refusal into a 500."""
    return f"{t:.2f}" if _finite(t) else "n/a"


def _rho_txt(rho: float | None) -> str:
    return f"{rho:+.2f}" if _finite(rho) else "n/a"


def _untestable_kind(model: dict) -> str:
    """WHICH way this fit has no testable intercept — one name, shared by the
    reason code in the payload, the prose and the refusal evidence so the three
    can never disagree. `_untestable_kind_of` applied to a `regress` row, so it
    is the SAME condition set as `alpha_testable` rather than a mirror of it: a
    mirror is what the two functions used to be, and the reflection was missing a
    condition.

    Reads the four quantities by name and does NOT default them — a row this is
    asked about that lacks one is a bug in the caller, and `.get(..., <default>)`
    would answer "testable" for it, which is the answer this module can least
    afford to guess at. All four are on `HONEST_UNDER_DEGENERACY` for exactly
    this reason: the chokepoint must not be able to withhold the evidence the
    reason code is derived from, or a gated row would come back "testable" and
    the refusal would lose its name one layer downstream.

    Order is load-bearing. The centered floor is tested BEFORE the
    denominator-collapse tests because an exact factor combination trips both:
    `rf + 1.2*mkt_rf` has SSR/SST = 9.3e-31 AND SSR/Σy² = 9.2e-31 (SST ≈ Σy²
    there, the series has real variation), and its honest description is "the
    regressors reproduce it up to a constant", not "it has no variation". The
    collapse cases are the ones where SST is dust while Σy² is not: SSR/SST above
    1, or a centered share that cleared the floor only because its denominator
    fell through it.
    """
    return _untestable_kind_of(
        model["alpha_t_hac"], model["r2"],
        model["resid_var_share"], model["resid_uncentered_share"])


def _untestable_reason(model: dict, regressors: str = "the factors") -> str:
    """Why this model's intercept has no sampling distribution, in one clause.

    Shared by the refusal Finding and the prose so the two can never disagree.
    The NaN branch keeps `nan` out of reader-facing text (non-negotiable: a
    rendered NaN is a data problem shown as a number).

    "UP TO A CONSTANT" is load-bearing and was wrong here. The earlier wording
    said the regressors "reproduce its excess return to floating-point
    precision", which is false for the case that matters most: an exact clone of
    the benchmark PLUS a real 5%/yr constant fits exactly with an intercept of
    exactly +5%/yr. The regressors reproduce the VARIATION, not the level; what
    is untestable is the constant's standard error, not the constant.
    """
    share = model["resid_var_share"]
    kind = _untestable_kind(model)
    if kind == "no_variation":
        return ("the regression produced no usable t-statistic for alpha — the "
                "excess return has no variation for the model to work on")
    if kind == "no_dependent_variation":
        # The DENOMINATOR case. Which tell fired is named, but NO number from this
        # fit is quoted: the finding and the prose both promise the reader that no
        # intercept, t-statistic or R² is quoted, and printing the impossible R²
        # here — however honest the intent — is the same message/payload
        # contradiction one layer up. The numbers live in the refusal's evidence.
        tell = (
            "the residual share of total variance came out ABOVE 1 and the R² "
            "NEGATIVE, both ARITHMETICALLY IMPOSSIBLE for a least-squares fit "
            "with an intercept, where SSR ≤ SST is an identity"
            if share > 1.0 else
            f"the residual share of the UNCENTERED second moment (SSR/Σy², the "
            f"one denominator that does not collapse with the variation) is at or "
            f"below the {RESID_VAR_SHARE_FLOOR:.1e} floor"
        )
        return (
            f"its excess return has no variation OF ITS OWN for {regressors} to "
            f"work on — it is a constant to floating-point precision, which is "
            f"what a strategy that never traded looks like against a flat "
            f"risk-free rate. Its total variance is rounding dust, so {tell}. "
            f"Dust divided by dust is not a measurement: the t-statistic, the R² "
            f"and the appraisal ratio this fit reports are rounding artefacts, "
            f"and no significance verdict exists in either direction"
        )
    if kind == "no_intercept_standard_error":
        # The condition the two halves of the predicate used to disagree about.
        # It needs its OWN clause for the same reason every other branch does:
        # falling through to the wording below would have described a fit whose
        # residual share is a healthy 0.5 as "residual variance 5.0e-01 of total,
        # at or below the 2.2e-16 floor" — a refusal contradicted by the number
        # printed inside it, which is the defect this function was rewritten to
        # fix. No number from the fit is quoted: the tell is a zero, and the
        # diagnostics carry it.
        return (
            f"the regression produced no usable t-statistic for alpha — its HAC "
            f"standard error came back exactly zero while {regressors} left real "
            f"residual variation behind, so the ratio that would be the "
            f"t-statistic is a division by zero rather than a measurement, and no "
            f"significance verdict exists in either direction"
        )
    return (
        f"its excess return is reproduced by {regressors} UP TO A CONSTANT, to "
        f"floating-point precision (residual variance {share:.1e} of total, at "
        f"or below the {RESID_VAR_SHARE_FLOOR:.1e} floor). The intercept is "
        f"whatever that constant is — it may be zero or it may be large — but "
        f"its standard error is rounding dust, so the t-statistic is not a "
        f"measurement of anything and no significance verdict exists in either "
        f"direction"
    )


def _ladder_disagreement_note(daily_kind: str, monthly_kind: str) -> str:
    """The daily and the monthly ladder reached OPPOSITE answers about the same
    strategy. Say so, and say which resampling mechanism produced it.

    A named function rather than prose inlined at its one call site, for the
    reason this round exists: WHICH degeneracy the daily fit has decides which
    mechanism survived the resampling, and only one of the four branches is
    reachable from the fixture. Naming the wrong mechanism would be a refusal
    that misstates its own reason — the defect `_untestable_reason` was rewritten
    to fix — and a branch no test can drive is a branch that rots. Here every
    branch is driven directly.

    Each mechanism is claimed for the ONE kind it explains, never for "any daily
    degeneracy". The nonlinear-compounding story is true of an exact linear
    combination and of nothing else: `elif daily_kind:` asserted it for all four
    kinds, so a daily fit with a zero intercept standard error (or none at all)
    would have been explained by curvature that has nothing to do with it. A
    mechanism named wrongly is worse than no mechanism named, because it is the
    part a reader will repeat.
    """
    if daily_kind == "no_dependent_variation":
        tell = ("compounding to calendar months makes the RISK-FREE leg vary with "
                "the number of trading days in the month (19-23), so a strategy "
                "that never traded acquires month-length variation that is a "
                "CALENDAR ARTEFACT rather than a result")
    elif daily_kind == "no_testable_intercept":
        tell = ("compounding is NONLINEAR, so an excess return that is an exact "
                "linear combination of the daily factors stops being one at "
                "monthly frequency, and the monthly residual is that curvature "
                "rather than a result")
    elif daily_kind:
        tell = (f"the daily fit's degeneracy is {daily_kind}, and NO mechanism is "
                f"named for it here: the two this note can explain — a constant "
                f"excess return acquiring month-length variation, and an exact "
                f"linear combination losing exactness under compounding — are "
                f"specific to the other two kinds, and asserting either one for "
                f"this fit would be a stated reason that is not the reason. What "
                f"the two ladders share on this input is the series, not the "
                f"measurement")
    else:
        tell = (f"the MONTHLY fit is the degenerate one ({monthly_kind}) — its "
                f"alpha, t-statistic and R² are rounding artefacts rather than "
                f"measurements, whatever the daily fit supports")
    return (
        f"The daily and monthly regressions DISAGREE about whether this strategy "
        f"has a testable intercept at all — the daily fit is "
        f"{'degenerate (' + daily_kind + ')' if daily_kind else 'well-posed'} and "
        f"the monthly fit is "
        f"{'well-posed' if daily_kind else 'degenerate (' + monthly_kind + ')'} — "
        f"because the monthly series is a RESAMPLING of the daily one and not the "
        f"same measurement: {tell}. The monthly pass is not a robustness check on "
        f"this input; read it as a property of the resampling."
    )


def _synthetic_finding(fd: FactorData) -> Finding | None:
    if not fd.synthetic:
        return None
    return Finding(
        check="factor_data_synthetic", severity="warning",
        name="Factor attribution runs on SYNTHETIC factors",
        message=(
            "The factor file shipped with this build is a clearly-labelled "
            "SYNTHETIC placeholder, not the Kenneth R. French data library. "
            "Every alpha, loading and t-statistic in this section is "
            "ILLUSTRATIVE ONLY and must not be read as a claim about this "
            "strategy's real factor exposure."
        ),
        evidence={"factor_source": fd.path, "synthetic": True},
        explainer="factor-attribution",
    )


def _unavailable(reason: str, detail: str, evidence: dict) -> tuple[dict, list[Finding]]:
    """Refusing is a finding, never a silence (non-negotiable #7)."""
    return (
        {"available": False, "reason": reason, "message": detail, **evidence},
        [Finding(
            check="factor_attribution", severity="note",
            name="Factor attribution not computed",
            message=detail,
            evidence={"reason": reason, **evidence},
            explainer="factor-attribution",
        )],
    )


# ----------------------------------------------------------- entry point

@dataclass
class _Joined:
    strategy: pd.Series
    rf: pd.Series
    excess: pd.Series
    factors: pd.DataFrame
    bench_excess: pd.Series | None = None
    coverage: float = 1.0
    notes: list[str] = field(default_factory=list)


def _join(strategy: pd.Series, factors: pd.DataFrame,
          bench: pd.Series | None) -> _Joined:
    """Inner-join on the ACCRUAL date and convert to excess returns using FF's
    own RF column.

    The engine indexes a return by the day it was EARNED, and so does Ken
    French. Joining on anything else (the signal date) silently shifts the
    series by one day and collapses market beta toward zero.
    """
    idx = strategy.index.intersection(factors.index)
    f = factors.loc[idx]
    rf = f[RF_COLUMN]
    strat = strategy.loc[idx]
    excess = (strat - rf).rename("excess")
    bench_excess = None
    notes: list[str] = []
    if bench is not None:
        # Index coverage is not enough: a benchmark that HAS every date but
        # carries NaN — or ±inf, which `isna()` reports as present — on some of
        # them produces a singular design and an LinAlgError out of pinv. All
        # three failures are the same failure — the benchmark does not supply a
        # usable return on every attribution day — so all three skip the
        # spanning test with a note rather than raising.
        if not idx.difference(bench.index).empty:
            notes.append(
                "benchmark series does not cover every attribution day; the "
                "spanning test was skipped"
            )
        elif (bench_bad := int((~np.isfinite(
                bench.loc[idx].to_numpy(dtype=float))).sum())):
            notes.append(
                f"benchmark series has {bench_bad} missing or non-finite "
                f"value(s) inside the attribution window; the "
                f"spanning test was skipped"
            )
        else:
            bench_excess = (bench.loc[idx] - rf).rename("bench_excess")
    coverage = len(idx) / max(len(strategy), 1)
    joined = pd.concat([excess, f], axis=1)
    if joined.isna().any().any():
        raise ValueError(
            "joined strategy/factor frame contains NaN — refusing to attribute"
        )
    return _Joined(strategy=strat, rf=rf, excess=excess, factors=f,
                   bench_excess=bench_excess, coverage=coverage, notes=notes)


def _to_monthly(daily: pd.Series, calendar: pd.DatetimeIndex | None = None) -> pd.Series:
    """Compound daily returns into calendar months, dropping partial months.

    FF monthly factors are full calendar months; a stub month at either end
    biases alpha, so partial months are dropped rather than kept. Note this
    compounds the DAILY factor files rather than fetching French's separately
    published monthly files — the two differ slightly because his monthly
    factors are built from monthly portfolio returns, which is why this pass is
    labelled robustness and the daily regression is the headline.

    "Partial" is measured against a TRADING calendar, not against calendar
    dates. Comparing a month's first calendar day to the first observation
    discards January whenever the 1st is a holiday or a weekend — i.e. almost
    always — and discards the final month unless the window happens to end
    exactly on a month end. `calendar` is the full factor index (Ken French
    publishes every NYSE session), so a month is complete iff the series
    contains every trading day that calendar has in it.
    """
    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    if len(monthly) == 0:
        return monthly
    cal = pd.DatetimeIndex(calendar) if calendar is not None else daily.index
    if len(cal) == 0:
        return monthly.iloc[0:0]
    cal_days = pd.Series(1, index=cal).resample("ME").sum()
    have_days = pd.Series(1, index=daily.index).resample("ME").sum()
    cal_first, cal_last = cal[0], cal[-1]
    keep = []
    for period_end in monthly.index:
        month_start = period_end.replace(day=1)
        expected = int(cal_days.get(period_end, 0))
        keep.append(
            expected > 0
            and int(have_days.get(period_end, 0)) == expected
            # The calendar itself must not be truncated inside this month,
            # or "every day the calendar has" would be a partial month too.
            and cal_first < month_start
            and cal_last > period_end
        )
    return monthly[pd.Series(keep, index=monthly.index)]


def attribution_report(
    strategy_returns: pd.Series,
    bench_returns: pd.Series | None = None,
    bench_label: str = "the benchmark",
    attempts: int = 1,
    factors: FactorData | None = None,
) -> tuple[dict, list[Finding]]:
    """Full factor attribution → (payload, findings).

    `strategy_returns` is the daily net-of-costs return series indexed by
    ACCRUAL date (the pipeline's net5). `bench_returns` is the same benchmark
    series the rest of the report is judged against; it drives the spanning
    test. `attempts` is the pipeline's multiple-testing counter — alpha is the
    number that most deserves the haircut, because the intercept of a factor
    regression is precisely what gets mined (Harvey, Liu & Zhu 2016).

    Never raises for ordinary data shortfalls: a precondition failure returns
    `available: False` plus a Finding saying why.
    """
    _check_no_nan(strategy_returns, "strategy return series")
    fd = factors if factors is not None else load_factors()
    findings: list[Finding] = []
    syn = _synthetic_finding(fd)
    if syn:
        findings.append(syn)

    factor_meta = {
        "synthetic": fd.synthetic,
        "path": fd.path,
        "provenance": fd.provenance,
        "factor_start": fd.start,
        "factor_end": fd.end,
    }
    bt_start = str(strategy_returns.index[0].date())
    bt_end = str(strategy_returns.index[-1].date())

    j = _join(strategy_returns, fd.frame, bench_returns)
    n = len(j.excess)

    if n < MIN_DAILY_OBS or j.coverage < COVERAGE_FLOOR:
        detail = (
            f"The strategy window ({bt_start} to {bt_end}) overlaps the factor "
            f"data ({fd.start} to {fd.end}) on only {n} trading day(s) "
            f"({j.coverage:.0%} of the backtest). Factor attribution needs at "
            f"least {MIN_DAILY_OBS} overlapping days and {COVERAGE_FLOOR:.0%} "
            f"coverage; alpha estimated on less than that is noise wearing a "
            f"decimal point, so no alpha is reported."
        )
        payload, refusal = _unavailable("insufficient_overlap", detail, {
            "n_obs": n, "coverage": j.coverage,
            "backtest_start": bt_start, "backtest_end": bt_end, **factor_meta,
        })
        return payload, findings + refusal

    att_start = str(j.excess.index[0].date())
    att_end = str(j.excess.index[-1].date())

    # ---- the ladder (daily is primary: this engine is a daily engine) ----
    models: dict[str, dict] = {}
    for key, (label, cols) in MODEL_LADDER.items():
        row = regress(j.excess, j.factors, cols, TRADING_DAYS, HAC_LAGS_DAILY, key)
        row["label"] = label
        models[key] = row
    headline = models[HEADLINE_MODEL]

    # ---- NaN discipline: a degenerate fit is a REFUSAL, never a verdict ----
    # If the excess return has no variance around the factor model (the engine
    # emits exactly this when nothing ever qualifies and the book sits in cash
    # at rf), then se(α) = 0, t = NaN and R² = NaN. Every downstream comparison
    # against a bar — `abs(t) < 2`, `r2 > 0.99` — is False for NaN, so the
    # ladder would route a degenerate regression into the "clears the bar" and
    # "well-posed" branches and print "n/a" and "nan%" beside them. Refuse.
    if not (np.isfinite(headline["alpha_t_hac"]) and np.isfinite(headline["r2"])):
        detail = (
            f"The six-factor regression over {att_start} to {att_end} is "
            f"degenerate: the strategy's excess return carries no variation the "
            f"model can work on (residual standard error "
            f"{headline['alpha_se_hac']:.2e}), so alpha's t-statistic and the R² "
            f"are undefined rather than merely small. This is what a series that "
            f"never left cash looks like. No alpha, t-statistic, R² or appraisal "
            f"ratio is reported, because every one of them would be a fabricated "
            f"number rather than a measured one."
        )
        payload, refusal = _unavailable("degenerate_regression", detail, {
            "n_obs": n, "coverage": j.coverage,
            "alpha_per_period": headline["alpha_per_period"],
            "alpha_se_hac": headline["alpha_se_hac"],
            "backtest_start": bt_start, "backtest_end": bt_end,
            "attribution_start": att_start, "attribution_end": att_end,
            **factor_meta,
        })
        return payload, findings + refusal

    # ---- monthly robustness pass (the literature's reporting frequency) ----
    monthly_models: dict[str, dict] = {}
    monthly_note: str | None = None
    # Compound the raw series and RF separately, then subtract: compounding an
    # already-differenced excess return is not the monthly excess return.
    cal = fd.frame.index
    m_excess = _to_monthly(j.strategy, cal) - _to_monthly(j.rf, cal)
    m_factors = pd.DataFrame({
        c: _to_monthly(j.factors[c], cal) for c in FACTOR_COLUMNS
    }).dropna()
    m_idx = m_excess.index.intersection(m_factors.index)
    if len(m_idx) >= MIN_MONTHLY_OBS:
        for key, (label, cols) in MODEL_LADDER.items():
            row = regress(m_excess.loc[m_idx], m_factors.loc[m_idx], cols,
                          MONTHS_PER_YEAR, HAC_LAGS_MONTHLY, key)
            row["label"] = label
            monthly_models[key] = row
    else:
        monthly_note = (
            f"Only {len(m_idx)} full calendar month(s) of overlap — below the "
            f"{MIN_MONTHLY_OBS}-month academic floor, so the monthly robustness "
            f"pass was not run. The daily regression is reported alone."
        )

    # The SAME predicate at the SAME level on both ladders, and they can DISAGREE
    # about the same input. On the reachable degenerate case — a strategy that
    # never traded over the fixture's 793-session flat-RF run — the daily
    # headline has no variation of its own and every daily verdict is refused,
    # while compounding to calendar months makes the RF leg vary with the NUMBER
    # OF TRADING DAYS IN THE MONTH (21 vs 22), which is a calendar artefact and
    # not strategy variation. That artificial variation clears the predicate, so
    # the monthly ff6 comes back testable with t = -104 and is published beside a
    # daily ladder that refuses to report a t at all. Silence about that is its
    # own fabrication. Both pass the predicate here, and the disagreement — in
    # either direction — is named where the monthly models are published.
    monthly_disagrees = bool(monthly_models) and (
        bool(_untestable_kind(headline))
        != bool(_untestable_kind(monthly_models[HEADLINE_MODEL]))
    )
    if monthly_disagrees:
        monthly_note = _ladder_disagreement_note(
            _untestable_kind(headline),
            _untestable_kind(monthly_models[HEADLINE_MODEL]))

    # ---- spanning test against the user's own benchmark ----
    spanning: dict | None = None
    if j.bench_excess is not None:
        sp_frame = pd.DataFrame({"bench_excess": j.bench_excess})
        spanning = regress(j.excess, sp_frame, ("bench_excess",),
                           TRADING_DAYS, HAC_LAGS_DAILY, "spanning")
        spanning["label"] = f"Spanning vs {bench_label}"
        beta_b = spanning["betas"]["bench_excess"]["beta"]
        spanning["replicating_portfolio"] = _replicating_portfolio(beta_b, bench_label)
        # The spanning verdict is gated on ITS OWN testability, here, where it is
        # rendered — not on the headline ladder's, and not through
        # `alpha_significant`. An exact clone of the benchmark is the single most
        # likely input to this test (it is what a user checks first), and it is
        # exactly the fit that has no sampling distribution: `strat = rf +
        # 1.0*(bench - rf) + 0.05/252` fits with zero residual variance, an
        # intercept of exactly +5%/yr and t = 4e14. Reading `alpha_significant`
        # alone there printed "+5.0%/yr ... NOT statistically distinguishable
        # from zero", a sentence that contradicts its own first half. The
        # `verdict` sub-dict carries the same key set on both paths so a consumer
        # that reads it cannot KeyError on the refusal (same contract as
        # risk_decomposition).
        if spanning["alpha_testable"]:
            spanning["verdict"] = {"available": True, "reason": None, "message": None}
            spanning["plain_english"] = (
                f"{spanning['replicating_portfolio']} reproduces this strategy's "
                f"market exposure; what it does not reproduce is "
                f"{spanning['alpha_annual']:+.1%}/yr "
                f"(t = {_fmt_t(spanning['alpha_t_hac'])}). "
                + (
                    "That residual is statistically distinguishable from zero."
                    if (spanning["alpha_significant"] and spanning["alpha_annual"] > 0)
                    else
                    "That residual is significantly NEGATIVE — on a risk-adjusted "
                    "basis the strategy is WORSE than the benchmark it is judged "
                    "against, not merely redundant with it."
                    if spanning["alpha_significant"] else
                    "That residual is NOT statistically distinguishable from zero — "
                    "on a risk-adjusted basis the strategy is redundant with the "
                    "benchmark it is judged against."
                )
            )
        else:
            why = _untestable_reason(spanning, f"{bench_label} alone")
            spanning["verdict"] = {
                "available": False,
                "reason": _untestable_kind(spanning),
                "message": why,
            }
            spanning["plain_english"] = (
                f"{spanning['replicating_portfolio']} reproduces this strategy's "
                f"market exposure. What it does not reproduce CANNOT BE TESTED "
                f"here: {why}. No intercept, t-statistic or R² is quoted and no "
                f"verdict is given in either direction — this fit supports "
                f"neither 'redundant with {bench_label}' nor 'adds value beyond "
                f"{bench_label}'."
            )

    # ---- multiple testing on the alpha t-stat ----
    # Third consumer of the same t-statistic, and it publishes a VERDICT BOOLEAN
    # (`survives_bonferroni`) of exactly the shape `alpha_significant` has. On a
    # degenerate fit the headline p is computed from dust-over-dust (t = 2.32 ->
    # p = 0.020 -> survives_bonferroni True), so the p-values and the boolean are
    # gated on the same per-render testability check as every other verdict. The
    # hurdles are properties of df and the attempt count, not of this fit, so
    # they stay populated.
    alpha_testable = headline["alpha_testable"]
    headline_kind = _untestable_kind(headline)
    df = headline["df_resid"]
    # Computed from the headline t UNCONDITIONALLY — the same arithmetic
    # `regress` does, on the same inputs, so on a testable fit this is bit-for-bit
    # `headline["alpha_p_hac"]` — and then withheld by the chokepoint. Reading the
    # row's already-withheld p instead would work today and would make this
    # block's gate untestable (a mutant that removes it would survive, because the
    # None would be inherited): one place DECIDES, every place COMPUTES.
    p_raw = t_two_sided_p(headline["alpha_t_hac"], df)
    p_bonf = min(1.0, attempts * p_raw) if _finite(p_raw) else None
    # `survives_bonferroni` is a VERDICT BOOLEAN, and gating it to a hard False on
    # an untestable fit is the false-refutation shape this module's docstring
    # argues against: False reads as "this alpha was tested and did not survive
    # multiple testing", which is a negative result, not a missing one. It is None
    # when no test exists, and the reason travels with it in `verdict` — the same
    # {available, reason, message} contract the spanning block and
    # risk_decomposition already publish, so a consumer can tell "did not survive"
    # from "was never testable" without re-deriving it.
    mt = _withhold_unsupported({
        "attempts": attempts,
        "alpha_testable": alpha_testable,
        "p_raw": p_raw,
        "p_bonferroni": p_bonf,
        "t_hurdle_bonferroni": t_hurdle_for_p(0.05 / max(attempts, 1), df),
        "t_hurdle_harvey_liu_zhu": T_HARVEY_LIU_ZHU,
        "survives_bonferroni": (bool(p_bonf < 0.05) if _finite(p_bonf) else None),
        "verdict": {
            "available": alpha_testable,
            "reason": headline_kind or None,
            "message": (None if alpha_testable else
                        f"No multiple-testing verdict exists for this fit: "
                        f"{_untestable_reason(headline)}. The Bonferroni hurdle "
                        f"below is a property of the attempt count and the degrees "
                        f"of freedom, not of this regression, so it is still "
                        f"reported; the p-values and the survival flag are not."),
        },
    }, headline_kind)

    # ---- diagnostics / reliability ----
    reliability_issues: list[str] = []
    if headline["condition_number"] > CONDITION_BAR:
        reliability_issues.append(
            f"the six factors are near-collinear (scaled condition number "
            f"{headline['condition_number']:.0f} > {CONDITION_BAR:.0f})"
        )
    high_vif = {n: b["vif"] for n, b in headline["betas"].items() if b["vif"] > VIF_BAR}
    if high_vif:
        reliability_issues.append(
            "individual loadings on "
            + ", ".join(f"{FACTOR_LABELS.get(n, n)} (VIF {v:.1f})"
                        for n, v in sorted(high_vif.items()))
            + " are unreliable even though the fit and alpha are fine"
        )
    # DEGENERACY: ASK THE PREDICATE, never re-derive it. This block used to key
    # on `r2 < 0.0 or resid_var_share > 1.0` — a THIRD derivation of degeneracy
    # (after `_alpha_is_testable` and `_untestable_kind`) and one that reads only
    # the RATIO. A fit refused by the predicate's condition 4 alone (the
    # UNCENTERED share is dust) has a perfectly legal ratio — measured share
    # 0.9998, R² +0.00024 — which is neither negative, nor above 1, nor above
    # 0.99, so no issue fired, `issues` came back EMPTY and factor_reliability
    # rendered its PASS, "the regression is well-posed and the loadings are
    # interpretable", on a fit whose t is 3.0e14 and whose residual sd is 3.1e-17
    # — in the same findings list as "[note] factor_alpha: Alpha could not be
    # tested". The entry point refused while the renderer fabricated. One
    # predicate, asked at every site that renders a claim.
    #
    # The prose comes from `_untestable_reason`, the same clause the refusal
    # Finding and the plain-English paragraph use, so the three cannot disagree
    # about WHY. The `r2 < 0.0` disjunct is gone rather than pinned: it was
    # redundant with `share > 1.0` on every input this module has ever been shown
    # (they are the same identity rearranged), and an unpinned branch is the
    # thing this module keeps being bitten by.
    kind = _untestable_kind(headline)
    if kind == "no_variation":
        reliability_issues.append(
            "the regression R² is undefined — the excess return has no "
            "variation for the factor model to explain"
        )
    elif kind:
        reliability_issues.append(
            "the six-factor fit has NO TESTABLE INTERCEPT: "
            + _untestable_reason(headline, "the six factors")
        )
    # Not an `elif` of the degeneracy branch: an EXACT factor combination is both
    # degenerate AND a factor portfolio, and the R² half of that is the useful
    # half. `NaN > x` is False, so the NaN case needs no guard here.
    #
    # But the NUMBER goes when the fit is degenerate. This sentence quoted
    # "R² of 1.000" in the same findings list as a note promising the reader that
    # "no alpha, t-statistic or R² is quoted" — the prose walker in the tests
    # found it, which is the point of walking the prose rather than listing the
    # sentences known to be risky. The QUALITATIVE half is the useful half and it
    # survives; a reader who wants the number can read `r2` beside
    # `alpha_testable: False` in the payload, where it is evidence rather than a
    # claim about explained variation.
    if _finite(headline["r2"]) and headline["r2"] > R2_FACTOR_PORTFOLIO:
        reliability_issues.append(
            "the factor model reproduces this strategy's excess return almost "
            "exactly, so the strategy IS a factor portfolio and its intercept "
            "carries almost no information — the R² is not quoted here because "
            "this fit has no testable intercept"
            if kind else
            f"R² of {headline['r2']:.3f} means the strategy IS a factor "
            f"portfolio and its intercept carries almost no information"
        )
    rho1 = headline["resid_autocorr1"]
    if _finite(rho1) and abs(rho1) > RESID_AUTOCORR_BAR:
        reliability_issues.append(
            f"residual lag-1 autocorrelation is {rho1:+.2f}, which usually means "
            f"stale pricing or a date-alignment bug rather than a real result"
        )

    payload = {
        "available": True,
        "frequency": "daily",
        "annualization": TRADING_DAYS,
        "annualization_note": (
            f"Alpha is annualized arithmetically ({TRADING_DAYS} × daily alpha), "
            f"the academic convention; the geometric equivalent is reported "
            f"alongside. The t-statistic is invariant to annualization — only "
            f"the point estimate is scaled."
        ),
        "hac_rule": (
            f"Newey-West (1987) HAC standard errors, Bartlett kernel, "
            f"L = {HAC_LAGS_DAILY} trading days (daily) / {HAC_LAGS_MONTHLY} "
            f"months (monthly). OLS standard errors are reported beside them."
        ),
        "factors": factor_meta,
        "window": {
            "backtest_start": bt_start, "backtest_end": bt_end,
            "attribution_start": att_start, "attribution_end": att_end,
            "coverage": j.coverage, "n_obs": n,
        },
        "notes": j.notes,
        "ladder": list(MODEL_LADDER.keys()),
        "headline_model": HEADLINE_MODEL,
        "models": models,
        "monthly": {"available": bool(monthly_models), "models": monthly_models,
                    "note": monthly_note,
                    # A machine-readable companion to the note above, so a
                    # consumer does not have to parse prose to find out that the
                    # two ladders contradict each other on this input.
                    "disagrees_with_daily": monthly_disagrees},
        "spanning": spanning,
        "rolling_market_beta": rolling_market_beta(j.excess, j.factors),
        "risk_decomposition": risk_decomposition(headline, j.factors, TRADING_DAYS),
        "multiple_testing": mt,
        # Through the chokepoint like everything else. Today this call withholds
        # nothing the row had not already withheld — `resid_autocorr1` arrives as
        # None — so it is defence in depth rather than a load-bearing gate, and
        # it is here for the diagnostic somebody adds next, which is the only
        # kind of field this module has ever got wrong.
        "diagnostics": _withhold_unsupported({
            "condition_number": headline["condition_number"],
            "resid_autocorr1": rho1,
            "vif": {n: b["vif"] for n, b in headline["betas"].items()},
            "issues": reliability_issues,
        }, kind),
        "interpretation": _interpretation(headline, mt, fd),
    }

    findings.extend(_findings(models, headline, mt, j, fd, payload, bench_label))
    return payload, findings


def _replicating_portfolio(beta: float, bench_label: str) -> str:
    """The leverage-adjusted framing a PM would say out loud. Phrase the
    above-1 case as borrowing rather than 'negative cash', which reads as a
    typo."""
    if beta > 1.0:
        return (f"{beta:.0%} {bench_label}, funded with {beta - 1:.0%} borrowed "
                f"at the risk-free rate,")
    if beta < 0.0:
        return f"a {abs(beta):.0%} SHORT position in {bench_label}"
    return f"A portfolio of {beta:.0%} {bench_label} and {1 - beta:.0%} cash"


def _interpretation(headline: dict, mt: dict, fd: FactorData) -> str:
    """The plain-English sentence. Deterministic template — this is the product,
    and no model is allowed anywhere near it."""
    names = ", ".join(FACTOR_LABELS.get(n, n) for n in headline["factors"])
    a, t = headline["alpha_annual"], headline["alpha_t_hac"]
    if not headline["alpha_testable"]:
        # Return early rather than fall through to the shared template: that
        # template quotes alpha, the t-statistic, R² and the appraisal ratio, and
        # on a degenerate fit every one of those is a rounding artefact. Printing
        # "+0.0%/yr with a HAC t-statistic of 2.32" beside the word NOT TESTABLE
        # still puts the fabricated number in front of the reader.
        base = (
            f"After controlling for {names}, alpha is NOT TESTABLE: "
            f"{_untestable_reason(headline)}. No alpha, t-statistic, R² or "
            f"appraisal ratio is quoted, because each would be a rounding error "
            f"reported as a measurement."
        )
        if fd.synthetic:
            base = "ILLUSTRATIVE ONLY (synthetic factors): " + base
        return base
    if abs(t) < T_CONVENTIONAL:
        verdict = "NOT statistically distinguishable from zero"
    elif a < 0:
        verdict = ("statistically BELOW zero, i.e. the strategy underperforms "
                   "its own factor exposures")
    else:
        verdict = "statistically distinguishable from zero"
    base = (
        f"After controlling for {names}, the remaining alpha is {a:+.1%}/yr "
        f"with a HAC t-statistic of {_fmt_t(t)} — {verdict}. The six-factor "
        f"model explains {headline['r2']:.0%} of this strategy's return "
        f"variation, leaving {headline['resid_vol_annual']:.1%} annualized "
        f"idiosyncratic volatility (appraisal ratio "
        f"{headline['appraisal_ratio']:.2f})."
    )
    hurdle = mt["t_hurdle_bonferroni"]
    if mt["attempts"] > 1 and np.isfinite(hurdle):
        base += (
            f" Across {mt['attempts']} strategy variants tested on this "
            f"universe, a Bonferroni-adjusted 5% bar needs t > {hurdle:.2f}."
        )
    if fd.synthetic:
        base = "ILLUSTRATIVE ONLY (synthetic factors): " + base
    return base


def _factor_contributions(headline: dict, factors: pd.DataFrame,
                          periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """Annualized return each non-market loading ABSORBED: β_k × mean(f_k) × P.

    This is the alpha decomposition itself — α_CAPM − α_headline equals the sum
    of these terms plus the market-beta shift — so it answers "which factor ate
    the alpha" in the units of the alpha.
    """
    out: dict[str, float] = {}
    for name, b in headline["betas"].items():
        if name == "mkt_rf" or name not in factors.columns:
            continue
        beta = b["beta"]
        mean_f = float(factors[name].mean())
        if not (np.isfinite(beta) and np.isfinite(mean_f)):
            continue
        out[name] = float(beta * mean_f * periods_per_year)
    return out


def _absorbing_factor(headline: dict, factors: pd.DataFrame,
                      decay: float,
                      periods_per_year: int = TRADING_DAYS
                      ) -> tuple[str, float, float] | None:
    """Which factor ate the CAPM alpha, ranked by CONTRIBUTION, not by t-stat.

    A t-statistic measures how precisely a loading is estimated, not how much
    return it explains: a huge, razor-sharp loading on a factor with zero mean
    return absorbs exactly nothing, and ranking by |t| names it anyway. Rank by
    β_k × mean(f_k) × P, signed in the direction the alpha actually moved.
    """
    contribs = _factor_contributions(headline, factors, periods_per_year)
    if not contribs:
        return None
    direction = 1.0 if decay >= 0 else -1.0
    name = max(contribs, key=lambda k: direction * contribs[k])
    return name, contribs[name], float(factors[name].mean() * periods_per_year)


def _findings(models: dict, headline: dict, mt: dict, j: _Joined,
              fd: FactorData, payload: dict, bench_label: str) -> list[Finding]:
    out: list[Finding] = []
    a, t = headline["alpha_annual"], headline["alpha_t_hac"]
    capm = models["capm"]

    # ---- the headline: does alpha survive the factors? ----
    # Order matters and the sign is not optional. `abs(t)` alone treats "beats
    # its factors by 3%/yr" and "loses to its factors by 3%/yr" as the same
    # result; the second must never be able to reach severity "pass".
    if not headline["alpha_testable"]:
        # Two ways in. The NaN way is unreachable from attribution_report (the
        # degenerate guard refuses the whole report first). The EXACT-FIT way is
        # reachable and used to end in severity "pass": SSR is rounding dust, so
        # t is finite and plausible while alpha is ~1e-16/yr — a fabricated pass
        # printed beside this same payload's "R² of 1.000 means the strategy IS a
        # factor portfolio". The t-statistic is deliberately not quoted below;
        # quoting it is the defect.
        share = headline["resid_var_share"]
        why = _untestable_reason(headline)
        out.append(Finding(
            check="factor_alpha", severity="note",
            name="Alpha could not be tested",
            message=(
                f"The six-factor regression has no testable intercept: {why}. No "
                f"verdict is given, and no alpha, t-statistic or R² is quoted. A "
                f"missing test is reported as missing, never as a pass."
            ),
            # This dict IS serialized into the exported report JSON, so it carries
            # NO number the message above says is not quoted. It used to ship the
            # fabricated alpha, t and R² beside a sentence promising none of them —
            # latent only because no renderer reads `evidence` today, and a payload
            # that contradicts its own message is the same defect one layer down.
            # The keys are kept, with None as the value, so a consumer that reads
            # them cannot KeyError and cannot mistake dust for a measurement.
            evidence={"alpha_annual": None, "alpha_t_hac": None, "r2": None,
                      "alpha_testable": False,
                      "untestable_reason": _untestable_kind(headline),
                      "resid_var_share": share,
                      "resid_uncentered_share": headline["resid_uncentered_share"],
                      "resid_var_share_floor": RESID_VAR_SHARE_FLOOR,
                      "withheld": ["alpha_annual", "alpha_t_hac", "r2"]},
            explainer="factor-attribution",
        ))
    elif abs(t) < T_CONVENTIONAL:
        out.append(Finding(
            check="factor_alpha", severity="warning",
            name="No alpha survives the factor model",
            message=(
                f"Regressed on the Fama-French five factors plus momentum, this "
                f"strategy's alpha is {a:+.1%}/yr with a HAC t-statistic of "
                f"{_fmt_t(t)} — not statistically distinguishable from zero. "
                f"{headline['r2']:.0%} of its return variation is explained by "
                f"exposures you can buy in an index fund "
                f"(market beta {headline['betas']['mkt_rf']['beta']:.2f}). "
                f"Whatever this strategy earned, it earned by taking known risks."
            ),
            evidence={"alpha_annual": a, "alpha_t_hac": t, "r2": headline["r2"],
                      "model": headline["label"]},
            explainer="factor-attribution",
        ))
    elif a < 0:
        out.append(Finding(
            check="factor_alpha", severity="warning",
            name="Alpha is significantly NEGATIVE",
            message=(
                f"Regressed on the Fama-French five factors plus momentum, this "
                f"strategy's alpha is {a:+.1%}/yr with a HAC t-statistic of "
                f"{_fmt_t(t)} — significantly BELOW zero, not above it. Holding "
                f"the same six exposures passively (market beta "
                f"{headline['betas']['mkt_rf']['beta']:.2f}) would have earned "
                f"about {abs(a):.1%}/yr MORE than this strategy did. Whatever the "
                f"headline return was, the active decisions subtracted from it; "
                f"this is the opposite of alpha and no multiple-testing "
                f"adjustment can turn it into a pass."
            ),
            evidence={"alpha_annual": a, "alpha_t_hac": t, "r2": headline["r2"],
                      "model": headline["label"], **mt},
            explainer="factor-attribution",
        ))
    elif not mt["survives_bonferroni"]:
        out.append(Finding(
            check="factor_alpha", severity="warning",
            name="Alpha is significant on its own, but not after multiple testing",
            message=(
                f"Alpha after the six-factor model is {a:+.1%}/yr with a HAC "
                f"t-statistic of {_fmt_t(t)}, which clears the conventional bar "
                f"of 2.0. It does not clear the bar for this session: "
                f"{mt['attempts']} variants have been tested on this universe, "
                f"so a Bonferroni-adjusted 5% threshold needs t > "
                f"{mt['t_hurdle_bonferroni']:.2f} (Bonferroni p = "
                f"{mt['p_bonferroni']:.3f}). Harvey, Liu and Zhu argue a newly "
                f"discovered factor needs t > {T_HARVEY_LIU_ZHU:.1f} even before "
                f"counting your own searching."
            ),
            evidence={"alpha_annual": a, "alpha_t_hac": t, **mt},
            explainer="factor-attribution",
        ))
    else:
        out.append(Finding(
            check="factor_alpha", severity="pass",
            name="Alpha survives the six-factor model",
            message=(
                f"Alpha after market, size, value, profitability, investment and "
                f"momentum is {a:+.1%}/yr with a HAC t-statistic of {_fmt_t(t)}, "
                f"still significant at a Bonferroni-adjusted bar for "
                f"{mt['attempts']} variant(s) tested (t > "
                f"{mt['t_hurdle_bonferroni']:.2f}). Rare — and still a statement "
                f"about the past, not a forecast."
            ),
            evidence={"alpha_annual": a, "alpha_t_hac": t, **mt},
            explainer="factor-attribution",
        ))

    # ---- the decay pattern across the ladder ----
    # Gated on BOTH rungs being testable, here where the finding is rendered.
    # np.isfinite alone is not the test: for an exact factor combination both t's
    # are finite and plausible while every number in the message below —
    # "+0.0%/yr (t = 2.64)" vs "+0.0%/yr (t = 0.53)", and the absorbed-by
    # attribution built from them — is one rounding error divided by another. A
    # scan of `rf + c*mkt_rf`, c in [0.05, 3.0) step 0.005, tripped the
    # finite-only gate on 215 of 590 values, i.e. this was not a corner.
    t_capm, t_head = capm["alpha_t_hac"], t
    decay_testable = bool(capm["alpha_testable"] and headline["alpha_testable"])
    if not decay_testable:
        if headline["alpha_testable"]:
            # The headline branch above gave a real verdict, so nothing has told
            # the reader the LADDER comparison is unavailable. Say it rather than
            # go quiet (a refusal is a finding, never a silence). When the
            # headline is untestable too, the factor_alpha note above already
            # covers it and a second note would be noise.
            #
            # Reaching this branch needs CAPM untestable while the headline is
            # testable. Algebraically that cannot happen — adding regressors
            # cannot raise SSR, and SST is unchanged, so the headline's residual
            # share is never the larger one. That argument is exact in algebra
            # and only approximate in floating point: two shares an ulp either
            # side of the floor can invert it. The guard is here for that, and
            # test_ladder_decay_refuses_when_only_one_rung_is_testable drives it
            # directly rather than waiting for the coincidence.
            out.append(Finding(
                check="factor_alpha_decay", severity="note",
                name="Alpha decay across the ladder could not be tested",
                message=(
                    f"The ladder comparison needs a testable intercept on BOTH "
                    f"rungs, and the {capm['label']} fit does not have one: "
                    f"{_untestable_reason(capm, 'the market factor')}. No decay "
                    f"figure and no t-statistics are quoted for that comparison."
                ),
                evidence={"capm_alpha_testable": capm["alpha_testable"],
                          "headline_alpha_testable": headline["alpha_testable"],
                          "capm_resid_var_share": capm["resid_var_share"],
                          "headline_resid_var_share": headline["resid_var_share"],
                          "resid_var_share_floor": RESID_VAR_SHARE_FLOOR},
                explainer="factor-attribution",
            ))
    elif (np.isfinite(t_capm) and np.isfinite(t_head) and abs(t_capm) >= T_CONVENTIONAL
            and abs(t_head) < abs(t_capm) / 2.0):
        decay = capm["alpha_annual"] - a
        picked = _absorbing_factor(headline, j.factors, decay)
        contributions = _factor_contributions(headline, j.factors)
        head = (
            f"CAPM alpha is {capm['alpha_annual']:+.1%}/yr (t = {_fmt_t(t_capm)}), "
            f"but adding size, value, profitability, investment and momentum cuts "
            f"it to {a:+.1%}/yr (t = {_fmt_t(t_head)})"
        )
        absorber = absorbed = None
        if picked:
            absorber, absorbed, factor_ret = picked
            beta = headline["betas"][absorber]
            head += (
                f". The loading that absorbed it is "
                f"{FACTOR_LABELS.get(absorber, absorber)}: a {beta['beta']:+.2f} "
                f"beta on a factor that itself returned {factor_ret:+.1%}/yr over "
                f"this window accounts for {absorbed:+.1%}/yr of the "
                f"{capm['alpha_annual']:+.1%}/yr CAPM alpha"
            )
        out.append(Finding(
            check="factor_alpha_decay", severity="warning",
            name="Alpha is repackaged style exposure",
            message=head + ". This is a known risk premium you can rent "
                           "cheaply, not skill.",
            evidence={"capm_alpha_annual": capm["alpha_annual"],
                      "capm_t": t_capm, "headline_alpha_annual": a,
                      "headline_t": t_head, "absorbed_by": absorber,
                      "absorbed_annual": absorbed,
                      "factor_contributions_annual": contributions},
            explainer="factor-attribution",
        ))

    # ---- spanning against the user's benchmark ----
    # Gated on the SPANNING fit's own testability, before either verdict branch.
    # Neither branch below has an untestable case, so reading `alpha_significant`
    # first routes a degenerate clone into the "no risk-adjusted value added"
    # branch — a false REFUTATION, which is worse than the false pass it replaced:
    # a false pass flatters, a false refutation actively misleads while looking
    # rigorous.
    sp = payload.get("spanning")
    if sp and not sp["alpha_testable"]:
        beta_b = sp["betas"]["bench_excess"]["beta"]
        out.append(Finding(
            check="factor_spanning", severity="note",
            name=f"Spanning against {bench_label} could not be tested",
            message=(
                f"Regressed on {bench_label} alone, this strategy's intercept has "
                f"no sampling distribution: {sp['verdict']['message']}. The beta "
                f"of {beta_b:.2f} this fit does support: "
                f"{sp['replicating_portfolio'][0].lower() + sp['replicating_portfolio'][1:]} "
                f"reproduces this strategy's market exposure. But NO verdict is "
                f"given in either direction, and no intercept or t-statistic is "
                f"quoted: this fit supports neither 'spanned by {bench_label}' "
                f"nor 'adds value beyond {bench_label}'. A missing test is "
                f"reported as missing, never as either answer."
            ),
            # The beta IS supported by this fit and is quoted in the message, so it
            # stays. The R² is not: this fit's R² is either exactly 1.0 by
            # construction or NEGATIVE and impossible, and the message promises the
            # reader no intercept or t-statistic — the exported evidence must not
            # carry what the sentence beside it says was withheld.
            evidence={"beta": beta_b, "r2": None,
                      "spanning_alpha_annual": None, "spanning_t": None,
                      "spanning_resid_var_share": sp["resid_var_share"],
                      "spanning_resid_uncentered_share": sp["resid_uncentered_share"],
                      "resid_var_share_floor": RESID_VAR_SHARE_FLOOR,
                      "withheld": ["spanning_alpha_annual", "spanning_t", "r2"],
                      "verdict": sp["verdict"]},
            explainer="factor-attribution",
        ))
    elif sp and sp["alpha_significant"] and sp["alpha_annual"] < 0:
        # A significantly negative spanning intercept is a LOUDER result than a
        # zero one, and the old gate (only fire when insignificant) stayed
        # silent on it.
        beta_b = sp["betas"]["bench_excess"]["beta"]
        out.append(Finding(
            check="factor_spanning", severity="warning",
            name=f"Beaten by {bench_label} on a risk-adjusted basis",
            message=(
                f"Regressed on {bench_label} alone, this strategy's intercept is "
                f"{sp['alpha_annual']:+.1%}/yr (t = {_fmt_t(sp['alpha_t_hac'])}) "
                f"with beta {beta_b:.2f} and R² {sp['r2']:.0%} — significantly "
                f"BELOW zero. Buying "
                f"{sp['replicating_portfolio'][0].lower() + sp['replicating_portfolio'][1:]} "
                f"and doing nothing else would have beaten this strategy by "
                f"about {abs(sp['alpha_annual']):.1%}/yr at the same market "
                f"exposure, before this strategy's own trading costs."
            ),
            evidence={"spanning_alpha_annual": sp["alpha_annual"],
                      "spanning_t": sp["alpha_t_hac"],
                      "beta": beta_b, "r2": sp["r2"]},
            explainer="factor-attribution",
        ))
    elif sp and not sp["alpha_significant"]:
        beta_b = sp["betas"]["bench_excess"]["beta"]
        out.append(Finding(
            check="factor_spanning", severity="warning",
            name=f"Spanned by {bench_label}",
            message=(
                f"Regressed on {bench_label} alone, this strategy's intercept is "
                f"{sp['alpha_annual']:+.1%}/yr (t = {_fmt_t(sp['alpha_t_hac'])}) "
                f"with beta {beta_b:.2f} and R² {sp['r2']:.0%}. That combination "
                f"says the strategy is "
                f"{sp['replicating_portfolio'][0].lower() + sp['replicating_portfolio'][1:]} "
                f"with no risk-adjusted value added — a result you could "
                f"reproduce with one trade and no turnover."
            ),
            evidence={"spanning_alpha_annual": sp["alpha_annual"],
                      "spanning_t": sp["alpha_t_hac"],
                      "beta": beta_b, "r2": sp["r2"]},
            explainer="factor-attribution",
        ))

    # ---- attribution window shorter than the backtest window ----
    w = payload["window"]
    gap_days = (pd.Timestamp(w["backtest_end"]) - pd.Timestamp(w["attribution_end"])).days
    if gap_days > 31 or j.coverage < COVERAGE_WARN:
        out.append(Finding(
            check="factor_window", severity="note",
            name="Attribution covers a shorter window than the backtest",
            message=(
                f"The backtest runs {w['backtest_start']} to {w['backtest_end']}, "
                f"but the factor library ends {fd.end}, so alpha and the loadings "
                f"are estimated on {w['attribution_start']} to "
                f"{w['attribution_end']} ({w['coverage']:.0%} of the backtest "
                f"days). The factor data lags the present by roughly six weeks; "
                f"the alpha and the CAGR therefore describe slightly different "
                f"periods."
            ),
            evidence={"coverage": j.coverage, "gap_days": gap_days, **w},
            explainer="factor-attribution",
        ))

    # ---- reliability preconditions ----
    issues = payload["diagnostics"]["issues"]
    if issues:
        out.append(Finding(
            check="factor_reliability", severity="warning",
            name="Factor regression diagnostics flag a caveat",
            message=(
                "The regression ran, but: " + "; ".join(issues) + ". Read the "
                "loadings with that in mind — HML, RMW and CMA are correlated "
                "by construction, so collinearity here is expected rather than a "
                "defect, but it does make individual betas unstable."
            ),
            evidence=payload["diagnostics"],
            explainer="factor-attribution",
        ))
    else:
        out.append(Finding(
            check="factor_reliability", severity="pass",
            name="Factor regression diagnostics",
            message=(
                f"{w['n_obs']} overlapping trading days, scaled condition number "
                f"{headline['condition_number']:.1f}, residual lag-1 "
                f"autocorrelation {_rho_txt(headline['resid_autocorr1'])} — the "
                f"regression is well-posed and the loadings are interpretable."
            ),
            evidence=payload["diagnostics"],
            explainer="factor-attribution",
        ))
    return out



# --------------------------------------------------------- cost crossover

def breakeven_cost_bps(
    alpha_annual_gross: float,
    *,
    annual_traded_notional: float | None = None,
    annual_turnover: float | None = None,
) -> float:
    """Cost rate (bps PER SIDE, PER DOLLAR TRADED) at which gross alpha hits zero.

    The unit is the engine's `cost_bps`, the same number that labels the 0/5/20
    columns — NOT a round trip and NOT a rate per unit of one-way turnover.
    dsl/conventions.py is the source of truth for both halves of this:

      "Costs": net_ret = gross_ret − traded_notional × cost_bps / 10,000
      "Traded notional": traded_notional = Σ|Δw| = 2 × one-way turnover

    So the annual cost drag at rate c is c/10⁴ × (annual traded notional), and
    the crossover is

        c* = alpha_annual_gross / annual_traded_notional × 10 000.

    Dividing by the reported one-way TURNOVER statistic instead — which is half
    the notional — reports a breakeven exactly 2× too high, i.e. tells a user
    their edge survives 673 bps when it actually dies at 343. That is why this
    function refuses to take a bare positional number: the caller has to name
    which of the two quantities it is handing over.

    Pass `annual_traded_notional = result.traded_notional.sum() / years`
    (preferred — no conversion factor anywhere), or `annual_turnover =
    result.annual_turnover`, which is converted here by the documented
    NOTIONAL_PER_ONE_WAY_TURNOVER.

    `alpha_annual_gross` must be the alpha of the GROSS (0 bps) return series.
    Feeding it an already-charged net series (e.g. net5) answers a different
    question: the additional cost the strategy can bear on top of the 5 bps it
    was already charged.

    Returns +inf when nothing was ever traded (no cost rate can kill it), and a
    NEGATIVE number when gross alpha is already ≤ 0 — that strategy has no
    breakeven cost; it is under water before the first commission.
    (Novy-Marx & Velikov 2016 frame the same crossover as a capacity question.)
    """
    if (annual_traded_notional is None) == (annual_turnover is None):
        raise ValueError(
            "breakeven_cost_bps takes exactly one of annual_traded_notional= "
            "(Σ|Δw| per year, the cost base) or annual_turnover= (the one-way "
            "0.5·Σ|Δw| statistic) — naming it is mandatory because the two "
            "differ by a factor of 2 and silently swapping them doubles the "
            "reported breakeven"
        )
    notional = (annual_traded_notional if annual_traded_notional is not None
                else annual_turnover * NOTIONAL_PER_ONE_WAY_TURNOVER)
    if not np.isfinite(alpha_annual_gross) or not np.isfinite(notional):
        return float("nan")
    if notional <= 0:
        return float("inf")
    return float(alpha_annual_gross / notional * 10_000.0)
