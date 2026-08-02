"""Property tests (Section 3, 'Testing the Engine')."""
import numpy as np
import pandas as pd
import pytest

from dsl.schema import (
    Costs, Selection, Signal, StrategySpec, Universe, Window,
)
from engine import metrics
from engine.backtest import _deterministic_top, run
from engine.calendar import rebalance_signal_dates


def war_spec(**kw) -> StrategySpec:
    defaults = dict(
        name="12mo momentum top-5",
        universe=Universe(preset="user_watchlist"),
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=252), direction="top", n=5
        ),
        weighting="equal",
        rebalance="monthly",
        costs=Costs(sweep_bps=[0, 5, 10, 20]),
        benchmark="SPY",
        window=Window(start="2023-06-01", end="2026-05-29"),
    )
    defaults.update(kw)
    return StrategySpec(**defaults)


@pytest.fixture(scope="module")
def war_run(war_panel):
    universe = [c for c in war_panel.columns if c != "SPY"]
    return run(war_spec(), war_panel, universe)


def test_costs_never_help(war_run):
    gross = war_run.gross_returns
    for bps, net in war_run.net_returns.items():
        assert (net <= gross + 1e-15).all(), f"net > gross at {bps} bps"
        if bps > 0:
            assert metrics.total_return(net) < metrics.total_return(gross)


def test_weights_never_exceed_one(war_run):
    sums = war_run.target_weights.sum(axis=1)
    assert (sums <= 1.0 + 1e-9).all()
    assert (war_run.target_weights >= -1e-15).min().min()  # long-only


def test_doubling_costs_doubles_drag(war_run):
    drag5 = war_run.gross_returns - war_run.net_returns[5.0]
    drag10 = war_run.gross_returns - war_run.net_returns[10.0]
    assert np.allclose(drag10.values, 2.0 * drag5.values, atol=1e-15)


def test_deterministic_tie_breaking():
    scores = pd.Series({"ZZZ": 0.5, "AAA": 0.5, "MMM": 0.5, "BBB": 0.2})
    top2 = _deterministic_top(scores, 2, "top")
    assert list(top2.index) == ["AAA", "MMM"]  # alphabetical among ties
    bottom1 = _deterministic_top(scores, 1, "bottom")
    assert list(bottom1.index) == ["BBB"]


def test_identical_reruns_are_identical(war_panel):
    universe = [c for c in war_panel.columns if c != "SPY"]
    a = run(war_spec(), war_panel, universe)
    b = run(war_spec(), war_panel, universe)
    pd.testing.assert_series_equal(a.gross_returns, b.gross_returns)
    pd.testing.assert_frame_equal(a.target_weights, b.target_weights)


def test_monthly_signal_dates_are_month_ends(war_panel):
    sig = rebalance_signal_dates(war_panel.index, "monthly")
    by_month = pd.Series(war_panel.index, index=war_panel.index.to_period("M"))
    for d in sig:
        assert d == by_month.loc[pd.Timestamp(d).to_period("M")].max()


def test_partial_eligibility_leaves_rest_in_cash(three_ticker_panel):
    """Only 3 tickers exist but n=10 → each gets 1/10, 70% stays in cash."""
    spec = war_spec(
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=5), direction="top", n=10
        ),
        window=Window(start="2024-02-26", end="2024-04-03"),
    )
    res = run(spec, three_ticker_panel, ["A", "B", "C"])
    w = res.target_weights.iloc[0]
    assert w.sum() == pytest.approx(0.3, abs=1e-12)
    assert w[w > 0].tolist() == pytest.approx([0.1, 0.1, 0.1], abs=1e-12)


def test_metrics_formulas():
    r = pd.Series([0.01, -0.02, 0.03],
                  index=pd.bdate_range("2024-01-02", periods=3))
    eq = (1.01) * (0.98) * (1.03)
    assert metrics.total_return(r) == pytest.approx(eq - 1, abs=1e-12)
    assert metrics.cagr(r) == pytest.approx(eq ** (252 / 3) - 1, abs=1e-9)
    # max drawdown: peak after day1 = 1.01, trough 1.01*0.98 = 0.9898
    assert metrics.max_drawdown(r) == pytest.approx(0.98 - 1.0, abs=1e-12)


def test_inverse_vol_weighting(three_ticker_panel):
    spec = war_spec(
        weighting="inverse_vol",
        rebalance="daily",
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=5), direction="top", n=2
        ),
        window=Window(start="2024-04-01", end="2024-04-03"),
    )
    # 63-day vol lookback needs more history than this panel has → nobody
    # eligible → all cash, disclosed via eligible_counts.
    res = run(spec, three_ticker_panel, ["A", "B", "C"])
    assert (res.eligible_counts == 0).all()
    assert res.gross_returns.abs().sum() == 0.0
