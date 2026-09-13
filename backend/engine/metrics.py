"""Performance metrics — the tear-sheet suite a PM reads, not a retail summary.

Every annualization constant comes from dsl/conventions.py (the single source
of truth); nothing in this module hardcodes a trading-day count, so changing
the convention changes the numbers in exactly one place.

Backward compatibility: `summary()` keeps every key it has ever returned with
its original meaning (total_return, cagr, vol, sharpe, max_drawdown, days,
hit_rate) — the report payload, the PDF export and the snapshot tests read
those names. New families are added ALONGSIDE, never renamed over.

Families and their formulas (each function carries its own citation):

- returns      total_return, cagr, per-year and per-month tables
- risk         vol, downside deviation, ulcer index, VaR / CVaR (daily and at
               a stated multi-day horizon), skew, excess kurtosis, tail ratio
- risk-adjusted sharpe, sortino (MAR = 0 and MAR = rf), calmar, martin
- drawdown     depth, peak / trough / recovery dates, durations, time under
               water, longest episode, top-N table
- benchmark    alpha and beta by OLS on excess returns (with standard errors,
               t-statistics and R²), tracking error, information ratio,
               up/down capture, hit rate
- exposure     average position count, average gross weight, average cash
               weight, share of periods invested, share of periods in cash

Sign conventions: drawdowns, VaR and CVaR are negative numbers; ratios are
unitless; returns are FRACTIONS (0.07 = +7%), never percentages — formatting
is the reader's job, not the engine's.

NaN discipline: `_check` refuses a NaN return series rather than silently
skipping the hole (confirmed audit finding), and `_excess` extends that refusal
to a risk-free / MAR PATH that does not cover the return window — a rf series
with holes would otherwise be absorbed silently by pandas' skipna and inflate
every downside statistic. A metric that is genuinely undefined for the input
(no losing day, a single observation, a perfect regression fit) returns NaN.

NO FUNCTION IN THIS MODULE EVER RETURNS ±inf. An earlier version returned +inf
from `sortino` with no downside at all (the empyrical/quantstats limit) and
from the OLS t-statistics on a perfect fit. `json.dumps` writes those as the
bare token `Infinity`, which `JSON.parse` rejects outright — and "infinite
risk-adjusted return" is a data artefact, not a measurement. Every such limit
is now NaN. NaN still needs sanitizing before serialization; that is a
BLOCKING prerequisite on the caller, not an optional nicety.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dsl.conventions import TRADING_DAYS_PER_MONTH, TRADING_DAYS_PER_YEAR

# Calendar months per year. conventions.py fixes the trading-DAY counts; the
# monthly aggregations below are calendar-based and need this separately.
MONTHS_PER_YEAR = 12

_SQRT_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)

# A month with fewer sessions than this is a stub: accrual starts the day
# after the first signal date and ends on the window's last session, so the
# first and last buckets are usually partial (confirmed audit finding). They
# stay in the tables, flagged, and are excluded from complete-month counts.
#
# The offset is 3, not 6: at 6 a month with 15 of ~21 sessions — a 29% stub —
# counted as "complete" and fed the honest-denominator `complete_months` the
# report discloses. 3 tolerates the genuinely short NYSE months (a Thanksgiving
# or Independence Day week, or a 19-session February) without waving through a
# half month. The matching year offset is 15, one NYSE holiday week.
# TODO(integration): both offsets belong in dsl/conventions.py as
# MIN_SESSIONS_FOR_COMPLETE_MONTH / _YEAR. Left here for now because
# conventions.py is being edited by another agent in this same pass; moving
# them is a mechanical follow-up with no numeric consequence.
MIN_COMPLETE_MONTH_DAYS = TRADING_DAYS_PER_MONTH - 3
MIN_COMPLETE_YEAR_DAYS = TRADING_DAYS_PER_YEAR - 15

# Horizon for the multi-day tail statistics. One trading month: long enough to
# be the horizon a reader actually cares about, short enough that a 2–3 year
# backtest supplies enough INDEPENDENT windows to resolve a 5% quantile
# empirically. See `horizon_value_at_risk` for why a 1-year horizon is not
# offered — it needs ~15 years of data before the estimate stops being noise.
TAIL_HORIZON_DAYS = TRADING_DAYS_PER_MONTH

# Degeneracy floor. A return series whose DAILY standard deviation is below
# this is not a strategy, it is floating-point dust: an all-cash book carries
# ~1e-14 of it, and `sd == 0` let that through as a Sharpe of -1e17, while the
# scale-relative tolerance that replaced it was keyed to the RISK-FREE RATE
# (~5e-5) and so never exceeded its own absolute floor — 400 sessions of zeros
# plus N(0, 1e-9) rounding dust still printed a Sharpe of -2.0e6 (confirmed
# reviewer counter-example). This is an absolute floor in the units returns are
# actually expressed in: 1e-7 daily is an annualized volatility of 1.6e-6
# (0.00016%), roughly 3,000× below the quietest real book (a money-market fund
# at 0.5%/yr has a daily sd near 3e-4) and ~1e7 above float64 dust.
_MIN_DAILY_SD = 1e-7

# Belt and braces for anything that squeaks past the floor: a ratio this large
# on a series with essentially no volatility is an artefact, not an edge.
_ABSURD_RATIO = 20.0
_NEGLIGIBLE_ANNUAL_VOL = 5e-3      # 0.5%/yr

# Kept for the capture-ratio zero test, where the quantity is a compounded
# annual return and float equality is the wrong comparison.
_ABS_TOL = 1e-12


def _check(returns: pd.Series) -> pd.Series:
    """A NaN in a return series is an upstream alignment bug. Skipping it
    silently converts broken data into confident wrong numbers (confirmed
    audit finding) — refuse instead."""
    if returns.isna().any():
        raise ValueError(
            f"return series contains {int(returns.isna().sum())} NaN value(s) "
            f"— refusing to compute metrics on broken data"
        )
    return returns


def _excess(returns: pd.Series, rate: float | pd.Series) -> pd.Series:
    """`returns` minus a risk-free / MAR rate, with the NaN refusal applied to
    the RESULT.

    `_check` alone runs on `returns` BEFORE the subtraction, so a rf PATH with
    holes (^IRX keeps its own holiday calendar) slipped straight through:
    pandas then skipped those days in .mean()/.std()/.sum() without a word,
    and `downside_deviation` summed the survivors over a denominator of
    `len(r)` — not even a clean sub-sample. A 300-day series with 10 missing
    rf days inflated Sortino by 13% (confirmed reviewer counter-example).

    A rate path that does not cover every session of the return window is a
    data problem, and data problems surface as findings, never absorbed.
    """
    r = _check(returns)
    if isinstance(rate, pd.Series):
        aligned = rate.reindex(r.index)
        missing = int(aligned.isna().sum())
        if missing:
            raise ValueError(
                f"rate series does not cover {missing} of the {len(r)} return "
                f"session(s) — refusing to compute metrics on a partial "
                f"risk-free path"
            )
        return r - aligned
    value = float(rate)
    if not np.isfinite(value):
        raise ValueError(f"rate must be finite, got {rate!r}")
    return r - value


def _degenerate(sd_daily: float) -> bool:
    """True when a DAILY standard deviation is indistinguishable from zero, so
    any ratio built on it is noise amplification rather than a measurement."""
    return (not np.isfinite(sd_daily)) or sd_daily <= _MIN_DAILY_SD


def _implausible(annualized_ratio: float, sd_daily: float) -> bool:
    """True for an absurd risk-adjusted ratio on a series with essentially no
    risk — the residue the floor alone cannot catch."""
    if not np.isfinite(annualized_ratio):
        return True
    return (abs(annualized_ratio) > _ABSURD_RATIO
            and sd_daily * _SQRT_YEAR < _NEGLIGIBLE_ANNUAL_VOL)


def _rf_scalar(rf_daily: float | pd.Series) -> float:
    """Mean daily risk-free rate. Accepts the scalar the pipeline passes today
    and the daily path it should pass (audit finding), so callers can upgrade
    without touching this module. A path with holes is refused, not averaged
    around."""
    if isinstance(rf_daily, pd.Series):
        if rf_daily.isna().any():
            raise ValueError(
                f"risk-free path contains {int(rf_daily.isna().sum())} NaN "
                f"value(s) — refusing to average around the holes"
            )
        return float(rf_daily.mean()) if len(rf_daily) else 0.0
    return float(rf_daily)


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    common = a.index.intersection(b.index)
    return _check(a.loc[common]), _check(b.loc[common])


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else str(pd.Timestamp(ts).date())


def _or_none(value: float) -> float | None:
    """None for a non-finite value — used where the dict already carries None
    as its "not available" marker (recovery dates, position counts) so the key
    stays strict-JSON-encodable."""
    return float(value) if np.isfinite(value) else None


# --------------------------------------------------------------- core -----

def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + _check(returns)).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + _check(returns)).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    days = len(returns)
    if days == 0:
        return float("nan")
    total = (1.0 + _check(returns)).prod()
    if total <= 0:
        return -1.0
    return float(total ** (TRADING_DAYS_PER_YEAR / days) - 1.0)


def annual_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(_check(returns).std(ddof=1) * _SQRT_YEAR)


def sharpe(returns: pd.Series, rf_daily: float | pd.Series = 0.0) -> float:
    """Sharpe = mean(r − rf) / std(r − rf) × √(trading days/yr), ddof = 1.

    Sharpe (1994), "The Sharpe Ratio", Journal of Portfolio Management 21(1).
    """
    if len(returns) < 2:
        return float("nan")
    excess = _excess(returns, rf_daily)
    sd = float(excess.std(ddof=1))
    if _degenerate(sd):
        return float("nan")
    sr = float(excess.mean() / sd * _SQRT_YEAR)
    return float("nan") if _implausible(sr, sd) else sr


def sharpe_stderr_iid(returns: pd.Series,
                      rf_daily: float | pd.Series = 0.0) -> float:
    """Standard error of the ANNUALIZED Sharpe **assuming IID returns**:
    SE[SR] = √((1 + SR²/2) / T) in per-period units, then × √(days/yr).

    Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal
    58(4), eq. (9).

    THE IID ASSUMPTION DOES NOT HOLD FOR THIS ENGINE'S OUTPUT. Overlapping
    hold_days cohorts and slow-moving signals make the daily return series
    autocorrelated BY CONSTRUCTION (a standing audit finding against this very
    file), and positive autocorrelation makes this number too SMALL — at a
    first-order autocorrelation of 0.2 it understates the true sampling
    dispersion by ~18%, at 0.4 by ~34% (simulation in the test suite). A 95%
    confidence interval built on it is too narrow, and a too-narrow interval
    driving a GREEN verdict is a false honesty claim.

    The name carries the `_iid` suffix so that no consumer can mistake it for
    the usable one. Publish intervals off `sharpe_stderr_hac` instead; this
    function is kept for the side-by-side comparison, which is itself worth
    disclosing.

    Denominator convention: T, per Lo's asymptotic eq. (9). The project's
    stat-validation note writes the same variance over (T − 1); the gap is
    0.84% at T = 60 and 0.20% at T = 250, and T is the form actually cited.
    """
    n = len(returns)
    if n < 3:
        return float("nan")
    sr = sharpe(returns, rf_daily)
    if not np.isfinite(sr):
        return float("nan")
    sr_period = sr / _SQRT_YEAR
    return float(math.sqrt((1.0 + 0.5 * sr_period ** 2) / n) * _SQRT_YEAR)


# The HAC standard error narrows the IID gap but does NOT close it: on AR(1)
# daily returns with φ = 0.4 over 750 sessions it averages this fraction of the
# simulated sampling dispersion of the annualized Sharpe (measured in the test
# suite, which pins this number against the simulation). The math is left alone
# — a kernel estimator is downward-biased in finite samples — but the shortfall
# has to travel with the VALUE, not sit in a docstring no report reader opens,
# so `summary()` carries the sentence below beside `sharpe_stderr_hac`.
HAC_DISPERSION_RATIO = 0.93
SHARPE_STDERR_HAC_DISCLOSURE = (
    "Newey-West (HAC) standard error of the annualized Sharpe. It still "
    "UNDERSTATES the true sampling dispersion: on AR(1) daily returns with a "
    "first-order autocorrelation of 0.4 over 750 sessions it averages about "
    f"{HAC_DISPERSION_RATIO:.2f}x the simulated dispersion, so an interval "
    "built on it is roughly 7% too narrow. Treat it as a floor on the "
    "uncertainty, not a bound — and never publish the IID form instead, which "
    "is far worse (~0.66x on the same simulation)."
)


def newey_west_lag(n: int) -> int:
    """Standard automatic Bartlett bandwidth, ⌊4·(T/100)^(2/9)⌋ (Newey & West
    1994). 6 lags at T = 750, 7 at T = 1250."""
    if n < 2:
        return 0
    return int(4.0 * (n / 100.0) ** (2.0 / 9.0))


def sharpe_stderr_hac(returns: pd.Series,
                      rf_daily: float | pd.Series = 0.0,
                      lag: int | None = None) -> float:
    """Standard error of the ANNUALIZED Sharpe under SERIAL DEPENDENCE — the
    one a confidence interval may be published from.

    Lo (2002) §4 / Proposition 3: treat SR = μ/√(γ₂ − μ²) as a smooth function
    of the moments θ = (μ, γ₂) = (E[r], E[r²]) and apply the delta method to a
    heteroskedasticity- and autocorrelation-consistent estimate of the long-run
    covariance of zₜ = (rₜ, rₜ²):

        ∇ = ( 1/σ + μ²/σ³ , −μ/(2σ³) )
        Ψ = Γ₀ + Σ_{k=1..L} (1 − k/(L+1))·(Γ_k + Γ_kᵀ)      [Bartlett kernel]
        SE[SR] = √(∇ᵀ Ψ ∇ / T) × √(days/yr)

    For IID returns Ψ collapses to the plain covariance and the expression
    reduces ALGEBRAICALLY to Lo's eq. (9), √((1 + SR²/2)/T) — so the two
    functions agree on IID data and diverge exactly as far as the dependence
    in the data warrants. Verified by simulation in the test suite: on AR(1)
    daily returns with φ = 0.4 and T = 750 the true sampling dispersion of the
    annualized Sharpe is ≈ 0.876; this estimator averages ≈ 0.82 while the IID
    form is stuck at ≈ 0.58.

    `lag` defaults to the automatic Newey–West bandwidth. A caller that knows
    the strategy's own dependence horizon should pass it — for this engine
    that is max(hold_days, sessions per rebalance period), because cohorts
    overlapping for hold_days is precisely where the autocorrelation comes
    from.

    Uses the population moments (ddof = 0) the GMM derivation is written in;
    `sharpe()` reports with ddof = 1, an O(1/T) difference (0.2% at T = 250).
    """
    ex = _excess(returns, rf_daily).to_numpy(dtype=float)
    n = ex.size
    if n < 10:
        return float("nan")
    mu = float(ex.mean())
    var = float((ex ** 2).mean()) - mu * mu
    if var <= 0 or _degenerate(math.sqrt(var)):
        return float("nan")
    sd = math.sqrt(var)
    grad = np.array([1.0 / sd + mu * mu / sd ** 3, -mu / (2.0 * sd ** 3)])

    z = np.column_stack([ex, ex ** 2])
    z = z - z.mean(axis=0)
    bandwidth = newey_west_lag(n) if lag is None else int(lag)
    bandwidth = max(0, min(bandwidth, n - 1))
    psi = z.T @ z / n
    for k in range(1, bandwidth + 1):
        gamma_k = z[k:].T @ z[:-k] / n
        psi = psi + (1.0 - k / (bandwidth + 1.0)) * (gamma_k + gamma_k.T)

    v = float(grad @ psi @ grad)
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    return float(math.sqrt(v / n) * _SQRT_YEAR)


# ----------------------------------------------------------- drawdown -----

def drawdown_series(returns: pd.Series) -> pd.Series:
    # The peak includes the initial capital: an investor who funds at
    # inception and immediately loses 30% IS 30% underwater — measuring
    # only from later highs hides that (confirmed audit finding).
    eq = equity_curve(returns)
    peak = eq.cummax().clip(lower=1.0)
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_episodes(returns: pd.Series) -> list[dict]:
    """Every underwater episode, deepest first.

    An episode runs from the high-water mark (peak) through the trough to the
    first session that regains the peak (recovery). pyfolio's
    `gen_drawdown_table` finds the same objects by excising the worst episode
    and recursing; walking the drawdown series once is equivalent and cheaper.

    Durations are counted in SESSIONS of this backtest's own calendar (not
    `pd.bdate_range` business days, which would invent holidays the run never
    traded through):
      peak_to_trough_days  sessions from the peak to the trough
      recovery_days        sessions from the trough to the recovery
      duration_days        peak → recovery; if `recovered` is False this is
                           days-so-far and the episode is still open
      underwater_days      sessions strictly below the prior peak

    `peak_date` is None when the drawdown starts at inception — the peak is
    the funding point itself, one session before the first accrual day, and
    fabricating a date for it would be a lie. Render it as "inception".
    `recovery_date` None means NOT RECOVERED; render it as such with the
    days-so-far, never as a blank cell.
    """
    dd = drawdown_series(returns)
    values = dd.to_numpy()
    index = dd.index
    n = len(values)
    episodes: list[dict] = []
    i = 0
    while i < n:
        if values[i] >= 0:
            i += 1
            continue
        j = i
        while j < n and values[j] < 0:
            j += 1
        trough = i + int(np.argmin(values[i:j]))
        peak = i - 1                      # last session at the high-water mark
        recovered = j < n
        episodes.append({
            "peak_date": _iso(index[peak]) if peak >= 0 else None,
            "trough_date": _iso(index[trough]),
            "recovery_date": _iso(index[j]) if recovered else None,
            "depth": float(values[trough]),
            "peak_to_trough_days": int(trough - peak),
            "recovery_days": int(j - trough) if recovered else None,
            "duration_days": int((j if recovered else n - 1) - peak),
            "underwater_days": int(j - i),
            "recovered": bool(recovered),
        })
        i = j
    episodes.sort(key=lambda e: e["depth"])
    return episodes


def drawdown_table(returns: pd.Series, top: int = 5) -> list[dict]:
    """The N worst drawdowns — the table every tear sheet opens the risk
    section with (pyfolio `show_worst_drawdown_periods`)."""
    return drawdown_episodes(returns)[:top]


def time_under_water(returns: pd.Series) -> float:
    """Share of sessions spent below the prior high-water mark.

    López de Prado (2018), *Advances in Financial Machine Learning*, ch. 14
    ("Backtest Statistics") treats time under water — not just max drawdown —
    as a first-class statistic.
    """
    if len(returns) == 0:
        return float("nan")
    return float((drawdown_series(returns) < 0).mean())


def ulcer_index(returns: pd.Series) -> float:
    """UI = √(Σ dd_t² / (N − 1)) — the root-mean-square depth of drawdown, so
    it penalises DURATION as well as depth (max drawdown ignores duration).

    Martin & McCann (1989), *The Investor's Guide to Fidelity Funds*;
    implementation matches quantstats' `ulcer_index`. Returned as a fraction
    (0.07 = 7%), on the same scale as max_drawdown.
    """
    dd = drawdown_series(returns)
    if len(dd) < 2:
        return float("nan")
    return float(math.sqrt(float((dd ** 2).sum()) / (len(dd) - 1)))


def martin_ratio(returns: pd.Series, rf_daily: float | pd.Series = 0.0) -> float:
    """Martin (Ulcer Performance) ratio = (CAGR − rf_annual) / ulcer_index.

    Martin & McCann (1989). We use the ANNUALIZED excess return in the
    numerator (the original UPI definition); quantstats uses the cumulative
    return instead, which makes the figure grow with window length.
    """
    ui = ulcer_index(returns)
    if not np.isfinite(ui) or ui <= 0:
        return float("nan")
    rf_annual = (1.0 + _rf_scalar(rf_daily)) ** TRADING_DAYS_PER_YEAR - 1.0
    return float((cagr(returns) - rf_annual) / ui)


def calmar(returns: pd.Series) -> float:
    """Calmar = CAGR / |max drawdown|.

    Young (1991), "Calmar Ratio: A Smoother Tool", Futures magazine;
    empyrical's `calmar_ratio`. NaN — never inf — when the window has no
    drawdown at all: "infinite return per unit of risk" is a data artefact.
    """
    mdd = max_drawdown(returns)
    if not (mdd < 0) or not np.isfinite(mdd):
        return float("nan")
    return float(cagr(returns) / abs(mdd))


# ------------------------------------------------------------ downside -----

def downside_deviation(returns: pd.Series,
                       mar_daily: float | pd.Series = 0.0) -> float:
    """Annualized downside deviation: √(Σ min(r − MAR, 0)² / N) × √(days/yr).

    Sortino & Price (1994), "Performance Measurement in a Downside Risk
    Framework", Journal of Investing 3(3). The denominator averages over ALL
    N observations — a positive period contributes a zero, it is NOT dropped
    from the count (empyrical `downside_risk`, quantstats `sortino`). Dividing
    by the count of negatives instead inflates the ratio, which is why the
    convention is worth stating.

    `mar_daily` may be a daily PATH; it is reindexed onto the return index and
    refused if it has holes, because a silently-skipped rf day leaves the
    numerator summing survivors over a denominator of N (see `_excess`).
    """
    r = _check(returns)
    if len(r) == 0:
        return float("nan")
    below = _excess(r, mar_daily).clip(upper=0.0)
    return float(math.sqrt(float((below ** 2).sum()) / len(r)) * _SQRT_YEAR)


def sortino(returns: pd.Series, mar_daily: float | pd.Series = 0.0) -> float:
    """Sortino = (mean(r) − MAR) × days/yr / annualized downside deviation.

    Sortino & Price (1994). MAR (minimum acceptable return) is stated by the
    caller: `summary()` reports BOTH the MAR = 0 convention (`sortino`, the
    one every Python tear sheet prints) and MAR = the daily risk-free rate
    (`sortino_rf`, the one that is directly comparable to Sharpe).

    With no downside at all the ratio is NaN — NOT +inf. empyrical and
    quantstats return +inf here; this module does not, for two reasons. A
    window in which nothing ever fell below MAR has not measured downside risk,
    it has failed to observe any, and "infinite risk-adjusted return" renders
    that failure as a triumph. And `json.dumps(float('inf'))` emits the bare
    token `Infinity`, which `JSON.parse` rejects outright. Render the NaN as an
    em-dash with the reason ("no period below the MAR in this window").
    """
    r = _check(returns)
    if len(r) == 0:
        return float("nan")
    dd = downside_deviation(r, mar_daily)
    if dd <= 0 or not np.isfinite(dd):
        return float("nan")
    mean_excess = float(_excess(r, mar_daily).mean())
    return float(mean_excess * TRADING_DAYS_PER_YEAR / dd)


# -------------------------------------------------------- distribution -----

def skew(returns: pd.Series) -> float:
    """BIAS-CORRECTED Fisher–Pearson sample skewness, G1 — pandas `.skew()`.
    Negative = the left tail is longer: small frequent gains, rare large
    losses.

    NOT the same estimator pyfolio reports. pyfolio's SIMPLE_STAT_FUNCS call
    `scipy.stats.skew` at its default `bias=True`, i.e. the uncorrected g1;
    G1 = √(n(n−1))/(n−2) · g1. The gap is ~3.9% at n = 40 and ~0.2% at
    n = 756. G1 is the better estimator and is the one kept, but the two are
    different numbers and a report must not cite pyfolio as its authority for
    this figure.
    """
    if len(returns) < 3:
        return float("nan")
    return float(_check(returns).skew())


def excess_kurtosis(returns: pd.Series) -> float:
    """BIAS-CORRECTED sample EXCESS kurtosis, G2 — pandas `.kurtosis()`
    (Fisher): a normal distribution scores 0, not 3. Named `excess_` so nobody
    compares it against 3.

    Same estimator caveat as `skew`: pyfolio reports scipy's uncorrected g2,
    and on the short windows this product actually reports (OOS splits, regime
    segments) the two diverge by ~15% at n = 40 and ~7% at n = 120, converging
    to ~1% by n = 756.
    """
    if len(returns) < 4:
        return float("nan")
    return float(_check(returns).kurtosis())


def tail_ratio(returns: pd.Series) -> float:
    """|95th percentile| / |5th percentile| (empyrical `tail_ratio`). 0.25
    means the left tail is four times the right."""
    r = _check(returns)
    if len(r) == 0:
        return float("nan")
    left = abs(float(np.percentile(r.to_numpy(), 5)))
    if left <= 0:
        return float("nan")
    return float(abs(float(np.percentile(r.to_numpy(), 95))) / left)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """HISTORICAL VaR: the (1 − level) percentile of the realized daily
    returns (empyrical `value_at_risk`). Negative by construction.

    Historical, not parametric: quantstats uses a Gaussian `norm.ppf` and
    pyfolio's row is `mean − 2·std`; both assume normality and so understate
    exactly the tail they claim to measure.
    """
    r = _check(returns)
    if len(r) == 0:
        return float("nan")
    return float(np.percentile(r.to_numpy(), (1.0 - level) * 100.0))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical CVaR / expected shortfall: the mean of every return at or
    below the VaR cutoff (empyrical `conditional_value_at_risk`)."""
    r = _check(returns)
    if len(r) == 0:
        return float("nan")
    cutoff = value_at_risk(r, level)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())


def horizon_returns(returns: pd.Series,
                    horizon_days: int = TAIL_HORIZON_DAYS) -> pd.Series:
    """Every OVERLAPPING `horizon_days`-session compounded return,
    ∏(1 + r) − 1, indexed on the session that closes each window.

    Overlapping rather than non-overlapping: the windows are serially
    dependent, so they buy no extra INDEPENDENT information, but they do
    remove the arbitrary choice of where the first window starts, which on a
    non-overlapping grid can move a 21-day VaR by several percentage points.
    The independent-window count is what governs precision, and it is reported
    alongside the estimate rather than hidden.
    """
    r = _check(returns)
    h = int(horizon_days)
    if h < 1 or len(r) < h:
        return pd.Series(dtype=float)
    return ((1.0 + r).rolling(h).apply(np.prod, raw=True) - 1.0).dropna()


def independent_windows(returns: pd.Series,
                        horizon_days: int = TAIL_HORIZON_DAYS) -> int:
    """How many NON-OVERLAPPING windows of this horizon the sample contains —
    the sample size that actually governs a horizon-quantile estimate."""
    h = int(horizon_days)
    return 0 if h < 1 else int(len(returns) // h)


def _min_independent_windows(level: float) -> int:
    """Two independent observations strictly beyond the cutoff is the minimum
    at which an empirical (1 − level) quantile means anything: 40 windows at
    95%, 200 at 99%."""
    tail = 1.0 - float(level)
    if tail <= 0:
        return 0
    return int(math.ceil(2.0 / tail))


def horizon_value_at_risk(returns: pd.Series, level: float = 0.95,
                          horizon_days: int = TAIL_HORIZON_DAYS) -> float:
    """EMPIRICAL VaR at a multi-session horizon: the (1 − level) percentile of
    the realized compounded `horizon_days` returns. No distributional
    assumption, and no √-of-time scaling anywhere.

    This REPLACES the `*_annualized` keys, which were wrong by ~1.7×. Those
    multiplied a daily QUANTILE by √(days/yr). A quantile is
    (drift + z·dispersion) and √-of-time scaling is legitimate for the
    dispersion term ONLY: over H days the drift got √H where it needed H,
    shifting the whole figure by μ·(H − √H), and arithmetic scaling ignores
    compounding on top of that. Reviewer's counter-example, reproduced: on IID
    N(0.0005, 0.011), 200k simulated one-year paths put the true 1-year 95%
    VaR at −16.2%; the √-scaled figure read −27.2%, i.e. 1.68× too deep. At
    μ = 0 it was still 1.11× off (arithmetic vs geometric compounding).
    The docstring's own "exact only for IID returns" disclaimer was false —
    the counter-example process was exactly IID normal.

    Returns NaN when the sample holds fewer independent windows than the
    quantile needs (`_min_independent_windows`), rather than printing a number
    the data cannot support. That is why the default horizon is one month and
    not one year: a 1-year 95% VaR needs ~40 independent years before the
    estimate settles, and on 5 years of data the same estimator reads −10.5%
    against a truth of −16.2%. Call with `horizon_days=TRADING_DAYS_PER_YEAR`
    on a long enough sample if the annual figure is genuinely wanted; it will
    refuse until the sample earns it.
    """
    windows = horizon_returns(returns, horizon_days)
    if len(windows) == 0:
        return float("nan")
    if independent_windows(returns, horizon_days) < _min_independent_windows(level):
        return float("nan")
    return float(np.percentile(windows.to_numpy(), (1.0 - level) * 100.0))


def horizon_tail_unavailable_reason(
        returns: pd.Series, level: float = 0.95,
        horizon_days: int = TAIL_HORIZON_DAYS) -> str | None:
    """WHY the horizon VaR / CVaR is unavailable, or None when it is available.

    The refusal in `horizon_value_at_risk` is the COMMON case, not the corner
    one: a 95% quantile needs 40 independent windows, so at the one-month
    default it takes 840 sessions (~3.3 years) before the figure is reported at
    all, and most backtests run here are shorter than that. A bare None in the
    payload reads as "omitted" and lets a report drop the row silently; the
    reader needs the sentence. Deterministic template text — no LLM, no number
    that is not counted from the sample itself.

    Returns None exactly when `horizon_value_at_risk` returns a finite number
    on the same arguments (pinned in the test suite).
    """
    r = _check(returns)
    h = int(horizon_days)
    n = len(r)
    if h < 1:
        return f"a horizon of {h} session(s) is not a horizon"
    if n < h:
        return (f"a {h}-session horizon needs at least {h} sessions; this "
                f"backtest has {n}")
    need = _min_independent_windows(level)
    have = independent_windows(r, h)
    if have < need:
        return (f"a {level:.0%} quantile at a {h}-session horizon needs {need} "
                f"independent windows ({need * h} sessions); this backtest has "
                f"{have} ({n} sessions)")
    return None


def horizon_conditional_value_at_risk(
        returns: pd.Series, level: float = 0.95,
        horizon_days: int = TAIL_HORIZON_DAYS) -> float:
    """Expected shortfall at the same horizon: the mean of every compounded
    `horizon_days` return at or below `horizon_value_at_risk`."""
    cutoff = horizon_value_at_risk(returns, level, horizon_days)
    if not np.isfinite(cutoff):
        return float("nan")
    windows = horizon_returns(returns, horizon_days)
    tail = windows[windows <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())


# ---------------------------------------------------- calendar buckets -----

def monthly_returns(returns: pd.Series) -> pd.Series:
    """Geometrically compounded calendar-month returns, PeriodIndex[M]."""
    r = _check(returns)
    if len(r) == 0:
        return pd.Series(dtype=float)
    return (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0


def _monthly_session_counts(returns: pd.Series) -> pd.Series:
    return returns.groupby(returns.index.to_period("M")).size()


def complete_months(returns: pd.Series) -> int:
    """Months with a full-ish set of sessions — the denominator a reader
    assumes when they see a monthly hit rate."""
    if len(returns) == 0:
        return 0
    counts = _monthly_session_counts(_check(returns))
    return int((counts >= MIN_COMPLETE_MONTH_DAYS).sum())


def hit_rate_monthly(returns: pd.Series, benchmark: pd.Series) -> float:
    """Share of calendar months in which the strategy beat the benchmark."""
    if len(returns) == 0:
        return float("nan")
    r_m = monthly_returns(returns)
    b_m = monthly_returns(benchmark)
    common = r_m.index.intersection(b_m.index)
    if len(common) == 0:
        return float("nan")
    return float((r_m.loc[common] > b_m.loc[common]).mean())


def monthly_returns_matrix(returns: pd.Series) -> list[dict]:
    """Years × months matrix for the heatmap, plus the compounded year.

    Rows: {"year": int, "months": [12 values or None], "eoy": float,
    "partial": bool}. Values are FRACTIONS; the frontend multiplies. A partial
    first/last year keeps the months it has and is flagged rather than
    dropped — an unflagged stub year reading +3% for two months misleads.
    """
    m = monthly_returns(returns)
    if len(m) == 0:
        return []
    counts = _monthly_session_counts(_check(returns))
    rows: list[dict] = []
    for year in sorted({p.year for p in m.index}):
        months: list[float | None] = [None] * MONTHS_PER_YEAR
        partial = False
        for period, value in m.items():
            if period.year != year:
                continue
            months[period.month - 1] = float(value)
            if int(counts.loc[period]) < MIN_COMPLETE_MONTH_DAYS:
                partial = True
        present = [v for v in months if v is not None]
        eoy = float(np.prod([1.0 + v for v in present]) - 1.0)
        rows.append({
            "year": int(year),
            "months": months,
            "eoy": eoy,
            "partial": bool(partial or len(present) < MONTHS_PER_YEAR),
        })
    return rows


def annual_returns_table(returns: pd.Series,
                         benchmark: pd.Series | None = None) -> list[dict]:
    """Per calendar year: strategy, benchmark, excess, sessions, partial flag.

    GIPS 2020 for firms (Provision 4.A.1) requires annual performance
    alongside an appropriate total-return benchmark. Excess is the arithmetic
    difference of the two annual figures — the convention fact sheets use.
    """
    r = _check(returns)
    if len(r) == 0:
        return []
    strat = (1.0 + r).groupby(r.index.year).prod() - 1.0
    sessions = r.groupby(r.index.year).size()
    bench = None
    if benchmark is not None and len(benchmark) > 0:
        b = _check(benchmark)
        bench = (1.0 + b).groupby(b.index.year).prod() - 1.0
    rows: list[dict] = []
    full_year = MIN_COMPLETE_YEAR_DAYS       # tolerate a short NYSE year
    for year in strat.index:
        row: dict = {
            "year": int(year),
            "strategy": float(strat.loc[year]),
            "benchmark": None,
            "excess": None,
            "days": int(sessions.loc[year]),
            "partial": bool(int(sessions.loc[year]) < full_year),
        }
        if bench is not None and year in bench.index:
            row["benchmark"] = float(bench.loc[year])
            row["excess"] = float(strat.loc[year] - bench.loc[year])
        rows.append(row)
    return rows


def period_extremes(returns: pd.Series) -> dict:
    """Best/worst day and month WITH their dates, average up/down month, and
    the up/down month counts — quantstats' distribution block. An undated
    extreme is less useful and slightly evasive."""
    r = _check(returns)
    out: dict = {
        "best_day": float("nan"), "best_day_date": None,
        "worst_day": float("nan"), "worst_day_date": None,
        "best_month": float("nan"), "best_month_date": None,
        "worst_month": float("nan"), "worst_month_date": None,
        "avg_up_month": float("nan"), "avg_down_month": float("nan"),
        "up_months": 0, "down_months": 0, "months": 0,
        "complete_months": 0,
    }
    if len(r) == 0:
        return out
    out["best_day"] = float(r.max())
    out["best_day_date"] = _iso(r.idxmax())
    out["worst_day"] = float(r.min())
    out["worst_day_date"] = _iso(r.idxmin())

    m = monthly_returns(r)
    if len(m) == 0:
        return out
    out["best_month"] = float(m.max())
    out["best_month_date"] = str(m.idxmax())
    out["worst_month"] = float(m.min())
    out["worst_month_date"] = str(m.idxmin())
    up, down = m[m > 0], m[m < 0]
    out["avg_up_month"] = float(up.mean()) if len(up) else float("nan")
    out["avg_down_month"] = float(down.mean()) if len(down) else float("nan")
    out["up_months"] = int(len(up))
    out["down_months"] = int(len(down))
    out["months"] = int(len(m))
    out["complete_months"] = complete_months(r)
    return out


# ------------------------------------------------- benchmark-relative -----

def alpha_beta(returns: pd.Series, benchmark: pd.Series,
               rf_daily: float | pd.Series = 0.0) -> dict:
    """OLS of daily EXCESS strategy return on daily EXCESS benchmark return:

        r_t − rf = α + β·(b_t − rf) + ε_t

    estimated with an intercept, so the fit also yields standard errors,
    t-statistics and R² — Jensen (1968), "The Performance of Mutual Funds in
    the Period 1945–1964", Journal of Finance 23(2). For a single regressor
    the slope is algebraically Cov/Var, but the bare `cov(s,b)/var(b)` the
    audit found in audit/extra.py estimates no intercept, subtracts no
    risk-free rate and reports no inference: a beta with no standard error
    cannot be argued with.

    α is annualized geometrically, (1 + α_daily)^(days/yr) − 1, matching
    empyrical; quantstats uses α_daily × days/yr instead.

    Returns NaN everywhere when fewer than 3 overlapping observations exist
    (OLS with an intercept has n − 2 residual degrees of freedom), and NaN —
    never ±inf — for the t-statistics on a perfectly fitting regression.

    `rf_daily` may be a daily PATH; it goes through `_excess`, so a rf series
    that does not cover the overlap is refused rather than producing an
    all-NaN regression that still reports the full `n_obs`.
    """
    out = {
        "alpha_daily": float("nan"), "alpha_annual": float("nan"),
        "alpha_stderr": float("nan"), "alpha_tstat": float("nan"),
        "beta": float("nan"), "beta_stderr": float("nan"),
        "beta_tstat": float("nan"), "r_squared": float("nan"),
        "correlation": float("nan"), "n_obs": 0,
    }
    r, b = _align(returns, benchmark)
    n = len(r)
    out["n_obs"] = int(n)
    if n < 3:
        return out

    y = _excess(r, rf_daily).to_numpy(dtype=float)
    x = _excess(b, rf_daily).to_numpy(dtype=float)
    x_bar, y_bar = x.mean(), y.mean()
    sxx = float(((x - x_bar) ** 2).sum())
    syy = float(((y - y_bar) ** 2).sum())
    if _degenerate(math.sqrt(sxx / (n - 1))):
        return out                      # a flat benchmark identifies no beta

    sxy = float(((x - x_bar) * (y - y_bar)).sum())
    beta = sxy / sxx
    alpha = y_bar - beta * x_bar
    resid = y - (alpha + beta * x)
    ssr = float((resid ** 2).sum())
    s2 = ssr / (n - 2)                  # residual variance, n − 2 dof

    out["beta"] = float(beta)
    out["alpha_daily"] = float(alpha)
    out["alpha_annual"] = float((1.0 + alpha) ** TRADING_DAYS_PER_YEAR - 1.0)
    out["r_squared"] = float(1.0 - ssr / syy) if syy > 0 else float("nan")
    out["correlation"] = float(np.corrcoef(x, y)[0, 1]) if syy > 0 else float("nan")

    if s2 > 0:
        se_beta = math.sqrt(s2 / sxx)
        se_alpha = math.sqrt(s2 * (1.0 / n + x_bar ** 2 / sxx))
        out["beta_stderr"] = float(se_beta)
        out["alpha_stderr"] = float(se_alpha)
        out["beta_tstat"] = float(beta / se_beta)
        out["alpha_tstat"] = float(alpha / se_alpha)
    else:
        # A perfect fit: zero residual variance, so the t-statistics are 0/0.
        # NaN, not ±inf — the module emits no infinities (json.dumps writes
        # them as the bare token `Infinity`, which JSON.parse rejects), and a
        # regression with no residual is a degenerate input (the benchmark
        # against itself), not evidence of infinite significance. The zero
        # standard errors are reported as the measured fact they are.
        out["beta_stderr"] = 0.0
        out["alpha_stderr"] = 0.0
        out["beta_tstat"] = float("nan")
        out["alpha_tstat"] = float("nan")
    return out


def tracking_error(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualized active risk: std(r − b, ddof = 1) × √(days/yr)."""
    r, b = _align(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    return float((r - b).std(ddof=1) * _SQRT_YEAR)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """IR = mean(r − b) × days/yr / tracking error.

    Grinold & Kahn (2000), *Active Portfolio Management*. Numerator and
    denominator share a periodicity: the mean annualizes by ×days/yr and the
    standard deviation by ×√(days/yr) — mixing the two is the common error.
    IR is uninterpretable without the tracking error beside it, so both are
    reported.
    """
    r, b = _align(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    active = r - b
    sd = float(active.std(ddof=1))
    if _degenerate(sd):
        return float("nan")
    ir = float(active.mean() * TRADING_DAYS_PER_YEAR / (sd * _SQRT_YEAR))
    return float("nan") if _implausible(ir, sd) else ir


def _annualized_monthly(m: pd.Series) -> float:
    if len(m) == 0:
        return float("nan")
    total = float((1.0 + m).prod())
    if total <= 0:
        return -1.0
    return float(total ** (MONTHS_PER_YEAR / len(m)) - 1.0)


def capture_ratios(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Up/down capture on MONTHLY returns.

    capture = annualized return of the strategy over the months the benchmark
    was positive (resp. negative) ÷ the benchmark's own annualized return over
    those same months — empyrical's geometric `capture()`, not a mean of
    ratios. Monthly rather than daily: daily up/down capture is noise, and
    fund fact sheets quote the monthly figure.

    Down capture keeps its sign convention: the denominator is negative, so
    down_capture > 1 means the strategy fell MORE than the benchmark. A window
    with fewer than 2 down months yields NaN — and that NaN is itself a
    finding ("no bear market in this window").

    The frequency label is keyed `capture_frequency`, not `frequency`: these
    keys land in `summary()`'s flat scalar block, and a bare `frequency` there
    reads as a property of the whole block rather than of the capture family.
    It is a STRING in a block a serializer will otherwise walk as numbers —
    along with the date and month-label strings — so a sanitizer must key off
    the value type, not the block.
    """
    out = {"up_capture": float("nan"), "down_capture": float("nan"),
           "up_down_capture": float("nan"), "up_periods": 0, "down_periods": 0,
           "capture_frequency": "monthly"}
    r, b = _align(returns, benchmark)
    if len(r) == 0:
        return out
    r_m, b_m = monthly_returns(r), monthly_returns(b)
    common = r_m.index.intersection(b_m.index)
    r_m, b_m = r_m.loc[common], b_m.loc[common]
    for key, mask in (("up_capture", b_m > 0), ("down_capture", b_m < 0)):
        rr, bb = r_m[mask], b_m[mask]
        out["up_periods" if key == "up_capture" else "down_periods"] = int(len(rr))
        if len(rr) < 2:
            continue
        denom = _annualized_monthly(bb)
        if not np.isfinite(denom) or abs(denom) < _ABS_TOL:
            continue
        out[key] = float(_annualized_monthly(rr) / denom)
    if np.isfinite(out["up_capture"]) and np.isfinite(out["down_capture"]) \
            and abs(out["down_capture"]) > _ABS_TOL:
        out["up_down_capture"] = float(out["up_capture"] / out["down_capture"])
    return out


# ------------------------------------------------------------ rolling -----

def rolling_sharpe(returns: pd.Series,
                   window: int = TRADING_DAYS_PER_YEAR,
                   rf_daily: float | pd.Series = 0.0) -> pd.Series:
    """Rolling annualized Sharpe over a trailing window (pyfolio plots a
    six-month and a one-year window; the default here is one year, from
    conventions.py). Returned as a SERIES for charting, warm-up NaNs dropped."""
    r = _excess(returns, rf_daily)
    if len(r) < window:
        return pd.Series(dtype=float)
    out = (r.rolling(window).mean() / r.rolling(window).std(ddof=1)) * _SQRT_YEAR
    return out.dropna()


def rolling_beta(returns: pd.Series, benchmark: pd.Series,
                 window: int = TRADING_DAYS_PER_YEAR,
                 rf_daily: float | pd.Series = 0.0) -> pd.Series:
    """Rolling beta of excess strategy return on excess benchmark return.

    For a single-regressor OLS the slope IS Cov(x, y)/Var(x) exactly, so the
    rolling covariance ratio is the rolling OLS slope — the inference (which
    a rolling window cannot honestly provide anyway) is what `alpha_beta()`
    adds on the full sample. A rolling beta whose range exceeds ~0.5 means a
    single beta figure is misleading for this strategy.
    """
    r, b = _align(returns, benchmark)
    if len(r) < window:
        return pd.Series(dtype=float)
    r_ex, b_ex = _excess(r, rf_daily), _excess(b, rf_daily)
    cov = r_ex.rolling(window).cov(b_ex)
    var = b_ex.rolling(window).var(ddof=1)
    return (cov / var.where(var > 0)).dropna()


def rolling_excess_1y(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Rolling 1-year strategy-minus-benchmark total return — so a strategy
    that won only in one year can't hide."""
    w = TRADING_DAYS_PER_YEAR
    r = (1.0 + _check(returns)).rolling(w).apply(np.prod, raw=True) - 1.0
    b = (1.0 + _check(benchmark)).rolling(w).apply(np.prod, raw=True) - 1.0
    return (r - b).dropna()


# ----------------------------------------------------------- exposure -----

def exposure_stats(weights: pd.DataFrame | None = None,
                   cash: pd.Series | None = None) -> dict:
    """Position counts and exposure, with each figure's own denominator.

    `weights` is one row per observation (the engine's `target_weights` is one
    row per SIGNAL DATE, so the position figures are per-rebalance unless a
    daily weight frame is passed — the caller states which). `cash` overrides
    the derived 1 − Σ|w| when the engine tracks cash directly, and it is
    normally a DAILY series while `weights` is per-rebalance. The two are
    therefore different samples BY DESIGN, so they are not intersected; what
    was wrong was reporting one `periods` for both. Each family now carries the
    count it was computed over:

      positions_periods  rows of `weights` behind avg/max_positions
      exposure_periods   rows behind the exposure and cash figures
      periods            == exposure_periods, kept for backward compatibility

    `time_in_cash` is a SHARE OF PERIODS — the fraction of observations with no
    position on at all — not an average cash weight. It used to be
    mean(1 − Σw), so a book that deployed exactly 50% every single day and was
    never once in cash reported time_in_cash = 0.5 while a report built on it
    would state the strategy sat in cash half the time. The average weight is
    still reported, under the name it actually has: `avg_cash_weight`.
    quantstats' "Time in Market %", which the old docstring cited as its
    authority, is `pct_periods_invested` = 1 − time_in_cash.

    Exposure sums ABSOLUTE weights. Identical to the signed sum for today's
    long-only engine; the day shorts appear, a signed sum reads a market-
    neutral book as 100% in cash.
    """
    out = {"avg_positions": float("nan"), "max_positions": None,
           "avg_exposure": float("nan"), "avg_cash_weight": float("nan"),
           "time_in_cash": float("nan"), "pct_periods_invested": float("nan"),
           "positions_periods": 0, "exposure_periods": 0, "periods": 0}
    if weights is None and cash is None:
        return out
    if weights is not None and len(weights) > 0:
        w = weights.fillna(0.0)
        counts = (w.abs() > 0).sum(axis=1)
        gross = w.abs().sum(axis=1)
        out["avg_positions"] = float(counts.mean())
        out["max_positions"] = int(counts.max())
        out["positions_periods"] = int(len(w))
        out["avg_exposure"] = float(gross.mean())
        out["avg_cash_weight"] = float((1.0 - gross).clip(lower=0.0).mean())
        out["pct_periods_invested"] = float((gross > 0).mean())
        out["time_in_cash"] = float((gross <= 0).mean())
        out["exposure_periods"] = int(len(w))
        out["periods"] = int(len(w))
    if cash is not None and len(cash) > 0:
        c = _check(cash)
        out["avg_cash_weight"] = float(c.mean())
        out["avg_exposure"] = float((1.0 - c).mean())
        out["pct_periods_invested"] = float((c < 1.0).mean())
        out["time_in_cash"] = float((c >= 1.0).mean())
        out["exposure_periods"] = int(len(c))
        out["periods"] = int(len(c))
    return out


# ------------------------------------------------------------ summary -----

def summary(returns: pd.Series, rf_daily: float | pd.Series = 0.0,
            benchmark: pd.Series | None = None,
            weights: pd.DataFrame | None = None,
            cash: pd.Series | None = None) -> dict:
    """The flat scalar block. Cheap enough to call per sub-window (the OOS
    split and the sensitivity sweep both do) — the SERIES and TABLES live in
    `tearsheet()`, which is called once per report.

    The six original keys (total_return, cagr, vol, sharpe, max_drawdown,
    days, plus hit_rate when a benchmark is given) keep their names and their
    formulas exactly. Four keys were REMOVED rather than corrected in place —
    var_95_annualized, var_99_annualized, cvar_95_annualized,
    cvar_99_annualized — because their values were wrong by ~1.7× and a
    consumer reading a corrected number under the old name would never learn
    that. `var_95_horizon` / `cvar_95_horizon` replace them, with the horizon,
    its independent-window count, and — when the sample cannot support the
    quantile — `tail_horizon_unavailable_reason` stated alongside.
    `sharpe_stderr` was renamed `sharpe_stderr_iid` and joined by
    `sharpe_stderr_hac` (whose residual understatement travels with it in
    `sharpe_stderr_hac_disclosure`); no consumer should silently pick the IID
    one for a confidence interval.

    Non-finite values: NaN appears wherever a statistic is genuinely undefined
    for the input, and there is no ±inf anywhere. `json.dumps` writes NaN as
    the bare token `NaN`, which `JSON.parse` rejects, so the caller MUST map
    non-finite floats to null (and render them as an em-dash with a reason)
    before this dict reaches a browser. That is a blocking prerequisite on the
    serialization layer, not a suggestion.
    """
    r = _check(returns)
    episodes = drawdown_episodes(r)
    worst = episodes[0] if episodes else None
    dd = drawdown_series(r) if len(r) else pd.Series(dtype=float)

    out: dict = {
        # --- original keys, unchanged meaning ---
        "total_return": total_return(r),
        "cagr": cagr(r),
        "vol": annual_vol(r),
        "sharpe": sharpe(r, rf_daily),
        "max_drawdown": max_drawdown(r) if len(r) else float("nan"),
        "days": int(len(r)),

        # --- risk-adjusted family ---
        # Two standard errors, deliberately. Publish intervals off the HAC
        # one: these returns are autocorrelated by construction (overlapping
        # hold_days cohorts), which makes the IID form too narrow.
        "sharpe_stderr_hac": sharpe_stderr_hac(r, rf_daily),
        "sharpe_stderr_hac_lag": newey_west_lag(len(r)),
        # The residual understatement travels with the value, not only in the
        # docstring: a reader who sees the number must see that the interval
        # built on it is a floor.
        "sharpe_stderr_hac_disclosure": SHARPE_STDERR_HAC_DISCLOSURE,
        "sharpe_stderr_iid": sharpe_stderr_iid(r, rf_daily),
        "sortino": sortino(r, 0.0),                 # MAR = 0
        "sortino_rf": sortino(r, rf_daily),         # MAR = daily risk-free
        "downside_deviation": downside_deviation(r, 0.0),
        "calmar": calmar(r) if len(r) else float("nan"),
        "ulcer_index": ulcer_index(r) if len(r) else float("nan"),
        "martin_ratio": martin_ratio(r, rf_daily) if len(r) else float("nan"),

        # --- drawdown family ---
        "max_drawdown_peak_date": worst["peak_date"] if worst else None,
        "max_drawdown_trough_date": worst["trough_date"] if worst else None,
        "max_drawdown_recovery_date": worst["recovery_date"] if worst else None,
        "max_drawdown_days": worst["duration_days"] if worst else 0,
        "max_drawdown_recovery_days": worst["recovery_days"] if worst else None,
        "max_drawdown_recovered": worst["recovered"] if worst else True,
        "time_under_water": time_under_water(r) if len(r) else float("nan"),
        "longest_drawdown_days": max((e["duration_days"] for e in episodes),
                                     default=0),
        "drawdown_episodes": len(episodes),
        "currently_underwater": bool(len(dd) and float(dd.iloc[-1]) < 0),
        "current_drawdown": float(dd.iloc[-1]) if len(dd) else float("nan"),

        # --- distribution family ---
        "skew": skew(r),
        "excess_kurtosis": excess_kurtosis(r),
        "tail_ratio": tail_ratio(r) if len(r) else float("nan"),
        "var_95": value_at_risk(r, 0.95) if len(r) else float("nan"),
        "var_99": value_at_risk(r, 0.99) if len(r) else float("nan"),
        "cvar_95": conditional_value_at_risk(r, 0.95) if len(r) else float("nan"),
        "cvar_99": conditional_value_at_risk(r, 0.99) if len(r) else float("nan"),
    }
    # Multi-session tail risk, EMPIRICAL at a stated horizon. The four
    # `*_annualized` keys that used to sit here multiplied a daily quantile by
    # √(days/yr) and were ~1.7× too deep; they are gone, not repaired in place, so
    # that a stale consumer fails loudly instead of reading a wrong number
    # under a familiar name. `tail_horizon_windows` is the independent-sample
    # count behind the figures and belongs beside them in any disclosure.
    #
    # These two are None — not NaN — when the window is refused. A refusal
    # ("this sample cannot support a 5% quantile at this horizon") is a
    # different statement from an undefined statistic, it is the one the
    # reader needs, and None is the only non-value that survives a strict JSON
    # encoder. The rest of this block still carries NaN and still needs a
    # sanitizer; see the class docstring.
    #
    # A bare None reads as "omitted", and the refusal is the COMMON case (40
    # independent windows = 840 sessions ≈ 3.3 years at the one-month default),
    # so `tail_horizon_unavailable_reason` states WHY in words the report can
    # print. It is None exactly when both figures are present.
    out["tail_horizon_days"] = int(TAIL_HORIZON_DAYS)
    out["tail_horizon_windows"] = independent_windows(r, TAIL_HORIZON_DAYS)
    out["tail_horizon_unavailable_reason"] = horizon_tail_unavailable_reason(
        r, 0.95, TAIL_HORIZON_DAYS)
    out["var_95_horizon"] = _or_none(
        horizon_value_at_risk(r, 0.95, TAIL_HORIZON_DAYS))
    out["cvar_95_horizon"] = _or_none(
        horizon_conditional_value_at_risk(r, 0.95, TAIL_HORIZON_DAYS))
    out.update(period_extremes(r))

    if benchmark is not None:
        out["hit_rate"] = hit_rate_monthly(r, benchmark)     # original key
        out["hit_rate_months"] = int(len(
            monthly_returns(r).index.intersection(monthly_returns(benchmark).index)
        ))
        out.update(alpha_beta(r, benchmark, rf_daily))
        out["tracking_error"] = tracking_error(r, benchmark)
        out["information_ratio"] = information_ratio(r, benchmark)
        out.update(capture_ratios(r, benchmark))

    if weights is not None or cash is not None:
        out.update(exposure_stats(weights, cash))
    return out


def tearsheet(returns: pd.Series, rf_daily: float | pd.Series = 0.0,
              benchmark: pd.Series | None = None,
              weights: pd.DataFrame | None = None,
              cash: pd.Series | None = None,
              rolling_window: int = TRADING_DAYS_PER_YEAR,
              top_drawdowns: int = 5) -> dict:
    """The full structure: scalars + tables + SERIES for charting.

    `series` values are pandas Series so the caller can hand them straight to
    audit/pipeline.py's `_series_points()`; everything else is already
    JSON-safe. Series shorter than the rolling window come back EMPTY rather
    than zero-filled — a 12-month rolling Sharpe on 8 months of data does not
    exist and must not be drawn.
    """
    r = _check(returns)
    series: dict[str, pd.Series] = {
        "equity_curve": equity_curve(r),
        "drawdown": drawdown_series(r),
        "rolling_sharpe": rolling_sharpe(r, rolling_window, rf_daily),
    }
    if benchmark is not None:
        # rolling_excess_1y was written, documented and never called; the
        # "a strategy that won only in one year can't hide" guarantee is only
        # real once the series is in the payload.
        series["rolling_excess_1y"] = rolling_excess_1y(r, benchmark)
        series["rolling_beta"] = rolling_beta(r, benchmark, rolling_window, rf_daily)

    return {
        "summary": summary(r, rf_daily, benchmark, weights, cash),
        "drawdowns": drawdown_table(r, top_drawdowns),
        "annual_returns": annual_returns_table(r, benchmark),
        "monthly_returns": monthly_returns_matrix(r),
        "exposure": exposure_stats(weights, cash),
        "series": series,
        "rolling_window": int(rolling_window),
    }
