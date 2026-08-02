"""Performance metrics. Formulas match the in-product glossary exactly
(dsl/conventions.py is the single source of truth for annualization).

- total_return = Π(1 + r) − 1
- CAGR         = (1 + total_return)^(252/days) − 1
- vol          = std(daily) × √252
- Sharpe       = mean(daily − rf_daily)/std(daily) × √252
- max_drawdown = min(equity/cummax(equity) − 1)
- hit_rate     = share of calendar months beating the benchmark
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


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
    return float(total ** (252.0 / days) - 1.0)


def annual_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(_check(returns).std(ddof=1) * math.sqrt(252.0))


def sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = _check(returns) - rf_daily
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(252.0))


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    # The peak includes the initial capital: an investor who funds at
    # inception and immediately loses 30% IS 30% underwater — measuring
    # only from later highs hides that (confirmed audit finding).
    eq = equity_curve(returns)
    peak = eq.cummax().clip(lower=1.0)
    return eq / peak - 1.0


def hit_rate_monthly(returns: pd.Series, benchmark: pd.Series) -> float:
    """Share of calendar months in which the strategy beat the benchmark."""
    if len(returns) == 0:
        return float("nan")
    r_m = (1.0 + returns).groupby(returns.index.to_period("M")).prod() - 1.0
    b_m = (1.0 + benchmark).groupby(benchmark.index.to_period("M")).prod() - 1.0
    common = r_m.index.intersection(b_m.index)
    if len(common) == 0:
        return float("nan")
    return float((r_m.loc[common] > b_m.loc[common]).mean())


def rolling_excess_1y(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Rolling 1-year (252d) strategy-minus-benchmark total return — so a
    strategy that won only in one year can't hide."""
    r = (1.0 + returns).rolling(252).apply(np.prod, raw=True) - 1.0
    b = (1.0 + benchmark).rolling(252).apply(np.prod, raw=True) - 1.0
    return (r - b).dropna()


def summary(returns: pd.Series, rf_daily: float = 0.0,
            benchmark: pd.Series | None = None) -> dict:
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "vol": annual_vol(returns),
        "sharpe": sharpe(returns, rf_daily),
        "max_drawdown": max_drawdown(returns),
        "days": int(len(returns)),
    }
    if benchmark is not None:
        out["hit_rate"] = hit_rate_monthly(returns, benchmark)
    return out
