"""Benchmark return series over the exact accrual dates of a run.

sixty_forty is computed by the engine itself (a static 60/40 SPY/AGG spec,
monthly rebalanced, zero cost) so its math is identical to any other backtest
— no separate code path to drift out of sync.
"""
from __future__ import annotations

import pandas as pd

from dsl.schema import (
    Costs, Selection, StaticPosition, StrategySpec, Universe, Window,
)

BENCHMARK_TICKERS: dict[str, list[str]] = {
    "SPY": ["SPY"],
    "QQQ": ["QQQ"],
    "sixty_forty": ["SPY", "AGG"],
}

BENCHMARK_LABELS: dict[str, str] = {
    "SPY": "SPY buy-and-hold",
    "QQQ": "QQQ buy-and-hold",
    "sixty_forty": "60/40 SPY/AGG (monthly rebalance)",
}


def benchmark_returns(
    benchmark: str, prices: pd.DataFrame, accrual_index: pd.DatetimeIndex
) -> pd.Series:
    """Daily benchmark returns aligned to the strategy's accrual days."""
    from engine.backtest import EngineError, run  # local import to avoid cycle

    for tk in BENCHMARK_TICKERS[benchmark]:
        if tk not in prices.columns or prices[tk].dropna().empty:
            raise EngineError(
                f"benchmark {benchmark} needs {tk} price data, which is missing "
                f"from the panel"
            )

    if benchmark in ("SPY", "QQQ"):
        px = prices[benchmark]
        rets = px.pct_change(fill_method=None)
        out = rets.reindex(accrual_index)
        if out.isna().any():
            raise EngineError(
                f"benchmark {benchmark} is missing returns on "
                f"{int(out.isna().sum())} accrual day(s)"
            )
        return out

    if benchmark == "sixty_forty":
        spec = StrategySpec(
            name="60/40 benchmark",
            universe=Universe(preset="user_watchlist", resolved_tickers=["SPY", "AGG"]),
            selection=Selection(static_allocation=[
                StaticPosition(ticker="SPY", weight=0.60),
                StaticPosition(ticker="AGG", weight=0.40),
            ]),
            weighting="equal",
            rebalance="monthly",
            costs=Costs(sweep_bps=[0]),
            benchmark="SPY",
            window=Window(
                start=str(accrual_index[0].date()),
                end=str(accrual_index[-1].date()),
            ),
        )
        # Static specs buy on the window's first day and accrue from day 2, so
        # feed a window starting one accrual day early to align exactly.
        pos = prices.index.get_loc(accrual_index[0])
        if pos == 0:
            raise EngineError("not enough lead-in data for the 60/40 benchmark")
        spec.window.start = str(prices.index[pos - 1].date())
        result = run(spec, prices, ["SPY", "AGG"])
        out = result.gross_returns.reindex(accrual_index)
        if out.isna().any():
            raise EngineError("60/40 benchmark could not cover the full window")
        return out

    raise EngineError(f"unknown benchmark: {benchmark}")
