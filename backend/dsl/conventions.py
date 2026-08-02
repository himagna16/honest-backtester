"""Day-count and calendar conventions — THE single source of truth (Gap #6).

The engine imports these constants, the LLM system prompt embeds the rendered
table, and the glossary page serves it via /api/conventions. If a convention
changes, it changes here and nowhere else.
"""
from __future__ import annotations

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5

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
        "rule": "Signals at date t use closing data through t only. Weights chosen at t earn returns from t+1 onward (portfolio_ret[t+1] = Σ w[t]·r[t+1]). The shift lives inside the engine; no spec can request a same-day fill.",
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
        "convention": "Turnover",
        "rule": "0.5 × Σ|w_target − w_drifted| at each rebalance execution, where w_drifted is the prior weights drifted by returns. Annualized = mean per-rebalance turnover × rebalances/year.",
    },
    {
        "convention": "Costs",
        "rule": "net_ret = gross_ret − turnover × cost_bps / 10,000, charged on the execution day. Every run reports 0 / 5 / 20 bps side by side.",
    },
    {
        "convention": "Annualization",
        "rule": "252 trading days per year. CAGR = (1 + total_return)^(252/days) − 1. Sharpe = mean(daily − rf_daily)/std(daily) × √252.",
    },
    {
        "convention": "Risk-free rate",
        "rule": "13-week T-bill (^IRX) when available; otherwise 0, disclosed as an audit note.",
    },
    {
        "convention": "Out-of-sample split",
        "rule": "First 70% of trading days = train, last 30% = holdout. The verdict privileges the holdout segment.",
    },
    {
        "convention": "Missing data",
        "rule": "Tickers with no data in the window are excluded and disclosed as an audit finding. Signals require a COMPLETE gap-free lookback — a halt or vendor hole inside the window makes the ticker ineligible until a clean lookback exists again. A NaN price mid-hold freezes at the last close, disclosed; standard mode exits the position at the next rebalance, cohort mode carries it to the cohort's expiry. Zero-volatility (stale-feed) tickers are excluded from inverse-vol selection, disclosed.",
    },
]


def conventions_markdown() -> str:
    lines = ["| Convention | Rule |", "|---|---|"]
    for row in CONVENTIONS_TABLE:
        lines.append(f"| {row['convention']} | {row['rule']} |")
    return "\n".join(lines)
