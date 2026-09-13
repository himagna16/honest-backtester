"""Day-count and calendar conventions — THE single source of truth (Gap #6).

The engine imports these constants, the LLM system prompt embeds the rendered
table, and the glossary page serves it via /api/conventions. If a convention
changes, it changes here and nowhere else.
"""
from __future__ import annotations

from numbers import Integral, Real

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5

# ---------------------------------------------------------------- execution
# Trading days between the close that PRODUCES a signal and the close that
# FILLS it. 1 is the conservative/honest default: you cannot see the close of
# t and also trade at it. This is a RUN-LEVEL convention, deliberately NOT a
# DSL field — a spec must not be able to buy itself a cheaper fill. The engine
# takes it as an argument so the audit can re-run at lag+1 and report the
# delta; a strategy whose edge dies under one extra day of lag is a
# microstructure artifact, which is exactly what this product exists to catch.
DEFAULT_EXECUTION_LAG_DAYS = 1

# Execution lag applied to BENCHMARKS: zero, always. A buy-and-hold benchmark
# has no signal to be late on — nobody reads a close, decides, and then trades.
# It is the passive alternative the strategy is measured against, so it is
# priced the way an index is priced: close-to-close, from the first accrual
# day. The SPY/QQQ benchmarks are raw pct_change (structurally lag 0), so the
# engine-computed 60/40 must be run at lag 0 too or the three benchmarks are
# mutually inconsistent and the comparison on every report is meaningless.
# Giving the benchmark the strategy's lag would also be a thumb on the scale in
# the strategy's favour: it would hand the benchmark a fabricated 0.0% (or
# rf-only) first day that the investor could in fact have earned.
BENCHMARK_EXECUTION_LAG_DAYS = 0


def coerce_execution_lag_days(execution_lag_days: int | None) -> int:
    """What counts as a legal run-level execution lag → the lag as an int.

    Raises ValueError (callers wrap it in their own refusal type) for anything
    that is not a non-negative WHOLE number of trading days. `int(...)` on its
    own turns True into 1 and 1.9 into 1, and a fill convention is not
    something to round off silently.

    This lives here, not in the engine, because the engine and the walk-forward
    purge in validation/splits.py both size themselves off the same lag: if
    they disagree about which values are legal, a lag the engine refuses can
    still produce a purge length, or vice versa. Numbers are matched through
    `numbers.Integral` / `numbers.Real` so numpy scalars pass without this
    module importing numpy (np.bool_ is neither, so it lands in the refusal).
    """
    if execution_lag_days is None:
        return DEFAULT_EXECUTION_LAG_DAYS
    if isinstance(execution_lag_days, bool):
        raise ValueError("execution_lag_days must be a whole number of "
                         "trading days, not a boolean")
    if isinstance(execution_lag_days, Integral):
        lag = int(execution_lag_days)
    elif isinstance(execution_lag_days, Real):
        if not float(execution_lag_days).is_integer():
            raise ValueError(
                f"execution_lag_days must be a whole number of trading days; "
                f"got {execution_lag_days}"
            )
        lag = int(execution_lag_days)
    else:
        raise ValueError(
            "execution_lag_days must be a whole number of trading days, not "
            f"{type(execution_lag_days).__name__}"
        )
    if lag < 0:
        raise ValueError("execution_lag_days cannot be negative — that is a "
                         "fill in the past")
    return lag


# ---------------------------------------------------------------- staleness
# A halted/stale vendor feed prints the same close every day. Such a name has
# (near-)zero measured volatility, zero drawdown and RSI 100 — i.e. it wins
# low-vol, min-drawdown and overbought rankings by being broken. The screen is
# a direct test on RAW (un-ffilled) closes, applied to EVERY selection path.
STALENESS_LOOKBACK_DAYS = 21
# Minimum number of OBSERVED close-to-close moves inside the lookback before
# the screen is allowed to judge a ticker at all. Counting rows instead (the
# first version of this screen) disqualified freshly-listed names on a sample
# of ONE observation: one printed diff that happened to be zero read as a
# 100%-unchanged feed. Fewer than this many observations is "unknown", never
# "stale" — we refuse to disqualify a ticker we cannot yet judge.
STALENESS_MIN_OBSERVED_DAYS = 15
# Share of unchanged closes inside the lookback that marks a feed dead. 0.60,
# not the original 0.50: a 21-sample proportion is noisy, and a name whose TRUE
# zero-change share is 0.45 trips a 0.5 cut-off on 32.1% of days (exact
# binomial) versus 9.1% at 0.60 — a 3.5× cut in false positives, each of which
# costs the user a forced sell plus a repurchase at 2 × bps.
#
# The threshold is deliberately kept SENSITIVE rather than pushed out to, say,
# 0.80. A mostly-frozen feed still wins a "rank by lowest volatility" or
# "smallest drawdown" contest, and the inverse-vol floor below does not protect
# those SELECTION paths — only the weighting one. Measured cost of that choice
# on fixtures/sp500_prices.csv: total stale flags fall from 2920 days to 2407,
# but borderline names flip state more often, not less (SW 76 → 116 flips,
# AMCR 14 → 22), because SW's true unchanged-share happens to sit near 0.57.
#
# NOT YET FIXED, deliberately: a name whose true share sits ON the threshold
# still oscillates, and the textbook answer (hysteresis — flag above 0.60,
# release below 0.40) is BLOCKED here. The screen must stay memoryless: the
# stale state at t may depend on closes in (t − STALENESS_LOOKBACK_DAYS, t] and
# nothing earlier, because validation/splits.py derives its purge length from
# STALENESS_LOOKBACK_DAYS. Hysteresis carries state forward indefinitely, so it
# lengthens how far a training-set price reaches into the test set — it turns
# turnover churn into LOOKAHEAD LEAKAGE, which is strictly worse. Measured:
# with hysteresis the walk-forward module's own perturbation probe finds test
# bars depending on a training close. Shipping it needs a
# STALENESS_REACH_DAYS = lookback + memory constant that purge_length consumes,
# which is a change to a file this engine does not own.
STALENESS_FLAG_SHARE = 0.60

# ------------------------------------------------------------ position sizing
# Floor on the annualized volatility used by inverse-vol weighting. The
# staleness screen only catches EXACTLY frozen feeds; a near-frozen one (a
# thinly-traded name that moves a cent a day, or a fund NAV printed to two
# decimals) measures a few basis points of vol, is never flagged, and 1/σ then
# hands it essentially the whole book. Floor it and the arithmetic can no
# longer reward a broken feed with a 98% weight.
#
# 0.005, not the 0.05 this shipped as first. The floor has to sit between the
# threat and the quietest REAL asset, and 5% was on the wrong side of the
# second one. Measured (63-day annualized vol, fixtures/):
#   threat  — a feed ticking one cent a day on a ~50 price: 0.001%; the noisier
#             version of the same thing (a NAV that moves 0 or 1 cent at
#             random): 0.16%.
#   quietest real asset — AGG, this project's own 60/40 bond leg: median 3.50%,
#             MINIMUM 1.96%, and 72.7% of its days sat BELOW a 5% floor. On a
#             SPY/AGG risk-parity spec that pushed AGG's mean target weight to
#             0.7038 against the textbook 1/σ 0.7582 (max error 18.5pp) and
#             emitted a "check the feed" data note about a healthy bond ETF —
#             a false data finding, which is worse than no finding.
#   equities — only 4 of 504 S&P names EVER dip below 5% (181 ticker-days of
#             1,364,436), so at 5% the floor was inert where it was aimed and
#             decisive where it was not.
# 0.005 is ~3× above the noisiest near-frozen measurement and ~4× below AGG's
# minimum, so both ends keep a margin. What it buys is a BOUND on the weight
# ratio (σ_other/0.005, e.g. 26:1 against a 13%-vol name) rather than the
# 10,000:1 an unfloored 1/σ hands a broken feed. It is not by itself a small
# weight: on a two-name book a floored name still takes ~97%, and against
# normal equity volatilities MAX_POSITION_WEIGHT below is what stops the
# concentration.
INVERSE_VOL_FLOOR_ANNUAL = 0.005
# Cap on any single name's weight, relaxed when there are too few names for the
# cap to be satisfiable (k names cannot all sit below 1/k).
MAX_POSITION_WEIGHT = 0.25

# Natural-language period → trading days (used by the LLM prompt and validator)
PERIOD_TO_TRADING_DAYS: dict[str, int] = {
    "1 week": 5,
    "2 weeks": 10,
    "1 month": 21,
    "2 months": 42,
    "3 months": 63,
    "6 months": 126,
    "9 months": 189,
    "1 year": 252,
    "12 months": 252,
    "2 years": 504,
    "3 years": 756,
}

CONVENTIONS_TABLE: list[dict[str, str]] = [
    {
        "convention": "Trading calendar",
        "rule": "SPY's own trading days (NYSE sessions). Every panel includes SPY; the date index of a run is exactly the days SPY traded.",
    },
    {
        "convention": "Period lengths",
        "rule": "1 week = 5 trading days; 1 month = 21; 3 months = 63; 6 months = 126; 12 months = 252. '12-1 momentum' = lookback 252 trading days, skip 21.",
    },
    {
        "convention": "Signal timing",
        "rule": "Signals at date t use closing data through t only — no value at t depends on any close after t, and a property test proves it by perturbing the future.",
    },
    {
        "convention": "Execution lag (fill timing)",
        "rule": (
            f"Default {DEFAULT_EXECUTION_LAG_DAYS} trading day: a signal read "
            f"at the close of t is filled at the close of "
            f"t+{DEFAULT_EXECUTION_LAG_DAYS}, and the position earns returns "
            f"from t+{DEFAULT_EXECUTION_LAG_DAYS + 1} onward. Until the fill "
            "the capital sits in cash. Execution lag is a run-level "
            "convention, not a spec field — no strategy can request a same-day "
            "fill. The lag a run actually used is carried on the result "
            "(EngineResult.execution_lag_days)."
        ),
    },
    {
        "convention": "Benchmark timing",
        "rule": (
            f"Benchmarks are priced at execution lag "
            f"{BENCHMARK_EXECUTION_LAG_DAYS} — close-to-close from the first "
            "accrual day — for all three of SPY, QQQ and 60/40. A passive "
            "buy-and-hold benchmark has no signal to be late on, so the "
            "strategy's fill lag does not apply to it; applying it would both "
            "make the 60/40 series inconsistent with the single-ticker "
            "benchmarks and hand the benchmark a fabricated flat first day."
        ),
    },
    {
        "convention": "Rebalance dates",
        "rule": "Signal date = last trading day of each calendar period (week / month / quarter / year); daily = every trading day. Positions earn returns starting the next trading day.",
    },
    {
        "convention": "Return over N days (signal 'return')",
        "rule": "P(t − skip) / P(t − skip − lookback) − 1, dividend-adjusted closes. skip > 0 excludes the most recent skip days (e.g. 12-1 momentum).",
    },
    {
        "convention": "Daily return (signal 'daily_return')",
        "rule": "P(t) / P(t − lookback) − 1 with lookback in trading days; lookback 1 = today's close over yesterday's close.",
    },
    {
        "convention": "Volatility (signal + metrics)",
        "rule": "Standard deviation of daily returns over the lookback window, annualized by √252.",
    },
    {
        "convention": "SMA cross (signal 'sma_cross')",
        "rule": "SMA(fast) / SMA(slow) − 1 at t. Positive = fast above slow. 'Price above its 200-day average' = fast 1, slow 200, op gt, value 0.",
    },
    {
        "convention": "Drawdown from high (signal 'drawdown_from_high')",
        "rule": "P(t) / max(P over trailing lookback days) − 1. Always ≤ 0.",
    },
    {
        "convention": "RSI (signal 'rsi')",
        "rule": "Wilder's RSI over lookback days (14 if unstated), 0–100 scale.",
    },
    {
        "convention": "hold_days (overlapping cohorts)",
        "rule": "Each rebalance date starts a new cohort held for hold_days trading days. Capital is split equally across all active cohorts each day; within a cohort the spec's weighting applies. The portfolio's daily return is the equal-weighted average of active-cohort returns. A day with no qualifying names spawns no cohort — capital concentrates in the remaining cohorts, or sits in cash if none are active. The daily re-equalization across cohorts is a real trade flow: pre-trade weights are NAV-share aggregated, so it is charged in turnover.",
    },
    {
        "convention": "Partial eligibility",
        "rule": "If only k < n names qualify, the portfolio holds k positions sized on the full-n baseline (equal weighting: 1/n each) with the remainder in cash — the conservative reading, disclosed via eligible counts.",
    },
    {
        "convention": "Turnover (the reported statistic)",
        "rule": "One-way turnover = 0.5 × Σ|w_target − w_drifted| at each execution, where w_drifted is the prior weights drifted by returns. This is the reporting statistic only — it is NOT the multiplier on the cost rate. Annualized = sum of one-way turnover ÷ years.",
    },
    {
        "convention": "Traded notional (the cost base)",
        "rule": "Traded notional = Σ|w_target − w_drifted| = 2 × one-way turnover, i.e. dollars of stock that actually change hands per dollar of NAV. Selling A and buying B trades 2 × NAV of notional; deploying 100% of NAV from cash trades 1 × NAV. The cash leg itself is not charged — cash does not cross a spread.",
    },
    {
        "convention": "Costs",
        "rule": "net_ret = gross_ret − traded_notional × cost_bps / 10,000, charged on the day the new weights start earning. cost_bps is PER SIDE, PER DOLLAR TRADED (half-spread + impact on one side of one trade) — not per round trip and not per unit of one-way turnover. Every run reports 0 / 5 / 20 bps per side, side by side.",
    },
    {
        "convention": "Cash",
        "rule": "Uninvested capital (partial eligibility, pre-fill days, no qualifying names) is carried as an explicit cash weight so weights sum to 1.0. The engine credits it the daily risk-free rate WHEN THE CALLER SUPPLIES ONE (run(..., rf_daily=...)); with no rate supplied cash earns 0%. The audit pipeline supplies the ^IRX path whenever the panel carries it, so the cash yield and the Sharpe hurdle are the same rate on the same days. Only when ^IRX is unavailable does cash earn 0% against a 0% hurdle — still consistent. Every run states which regime it was in (EngineResult.rf_used: None = 0% cash).",
    },
    {
        "convention": "Annualization",
        "rule": "252 trading days per year. CAGR = (1 + total_return)^(252/days) − 1. Sharpe = mean(daily − rf_daily)/std(daily) × √252.",
    },
    {
        "convention": "Risk-free rate",
        "rule": "13-week T-bill (^IRX) when available; otherwise 0, disclosed as an audit note. That rate is the Sharpe hurdle, and the same daily PATH is credited to uninvested cash (see Cash) — not a window average, so a rate that moves is subtracted on the days it actually moved. ^IRX keeps its own holiday calendar; interior gaps carry forward, and a gap at the window START takes 0% rather than carrying a later rate backward onto an earlier day, which is disclosed as a finding.",
    },
    {
        "convention": "Out-of-sample split",
        "rule": "First 70% of trading days = train, last 30% = holdout. The verdict privileges the holdout segment.",
    },
    {
        "convention": "Missing data",
        "rule": "Tickers with no data in the window are excluded and disclosed as an audit finding. Signals require a COMPLETE gap-free lookback — a halt or vendor hole inside the window makes the ticker ineligible until a clean lookback exists again. A NaN price mid-hold freezes at the last close, disclosed; standard mode exits the position at the next rebalance, cohort mode carries it to the cohort's expiry.",
    },
    {
        "convention": "Stale price feeds",
        "rule": (
            f"A ticker is treated as halted/stale on any day its RAW closes "
            f"are unchanged on more than {STALENESS_FLAG_SHARE:.0%} of the "
            f"trailing {STALENESS_LOOKBACK_DAYS} trading days, and is eligible "
            "again the moment that share drops back. The screen needs at least "
            f"{STALENESS_MIN_OBSERVED_DAYS} OBSERVED closes in the window to "
            "judge at all — a freshly-listed name with a handful of prints is "
            "unknown, never stale. It is memoryless: today's verdict depends "
            f"on the last {STALENESS_LOOKBACK_DAYS} sessions and nothing "
            "earlier, which is what lets the walk-forward purge length bound "
            "it. While flagged the ticker is ineligible for ranking, filtering "
            "and weighting alike, and the exclusion is disclosed as a data "
            "note. Two known limits: the screen tests REPEATED closes, so it "
            "catches a feed that has stopped printing new prices but NOT one "
            "that ticks by a cent every session (see Position size limits); "
            "and a name whose true unchanged-share sits on the threshold can "
            "still flip in and out day to day, because the hysteresis that "
            "would stop it would give the screen memory the purge length does "
            "not currently account for."
        ),
    },
    {
        "convention": "Position size limits",
        "rule": (
            f"Inverse-vol weighting floors each name's annualized volatility "
            f"at {INVERSE_VOL_FLOOR_ANNUAL:.2%} before inverting it, and then "
            "caps any single name at the greater of "
            f"{MAX_POSITION_WEIGHT:.0%} and twice the equal weight (2/k) of "
            "the invested book, redistributing the excess pro-rata to the "
            "uncapped names. A name whose measured volatility had to be "
            "floored is disclosed as a data note — a silent floor is still a "
            "silent screen. Both apply to inverse-vol weighting only: equal "
            "and rank weighting have fixed, spec-chosen concentration. Without "
            "the floor a near-frozen feed measuring a few bps of volatility "
            "takes ~99% of the book through 1/σ while never tripping the "
            f"staleness screen. The floor is set at {INVERSE_VOL_FLOOR_ANNUAL:.2%} "
            "so that it sits below the quietest genuine asset measured (a bond "
            "ETF's 63-day volatility bottoms out near 2%) and still above a "
            "cent-a-day feed. What it gives is a BOUND on how far 1/σ can tilt "
            "toward a suspiciously calm name, not a small weight: on a "
            "two-name book a floored name can still hold most of the capital, "
            "and against normal equity volatilities it is the position cap "
            "that stops the concentration."
        ),
    },
]


def conventions_markdown() -> str:
    lines = ["| Convention | Rule |", "|---|---|"]
    for row in CONVENTIONS_TABLE:
        lines.append(f"| {row['convention']} | {row['rule']} |")
    return "\n".join(lines)
