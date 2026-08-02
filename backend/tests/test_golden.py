"""Golden tests (M1 acceptance) — the auditor audited.

(a) zero-cost buy-and-hold SPY reproduces SPY's total return exactly
(b) 3-ticker momentum fixture matches hand-computed values to 6 decimals
(c) overlapping-cohort dip-buy fixture matches hand-computed values
(d) lookahead bias is impossible through the public API

The literal expected values below were derived by hand with exact fraction
arithmetic (see derivations in comments) — NOT by running the engine.
"""
import numpy as np
import pandas as pd
import pytest

from dsl.schema import (
    Costs, EntryExit, Filter, Selection, Signal, StaticPosition, StrategySpec,
    Universe, Window,
)
from engine import metrics
from engine.backtest import run


def spy_spec(start: str, end: str) -> StrategySpec:
    return StrategySpec(
        name="SPY buy and hold",
        universe=Universe(preset="user_watchlist", resolved_tickers=["SPY"]),
        selection=Selection(static_allocation=[StaticPosition(ticker="SPY", weight=1.0)]),
        weighting="equal",
        rebalance="monthly",
        costs=Costs(sweep_bps=[0, 5, 20]),
        benchmark="SPY",
        window=Window(start=start, end=end),
    )


def momentum_spec(**kw) -> StrategySpec:
    defaults = dict(
        name="top-2 5d momentum",
        universe=Universe(preset="user_watchlist", resolved_tickers=["A", "B", "C"]),
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=5), direction="top", n=2
        ),
        weighting="equal",
        rebalance="monthly",
        costs=Costs(sweep_bps=[0, 5, 20]),
        benchmark="SPY",
        window=Window(start="2024-02-26", end="2024-04-03"),
    )
    defaults.update(kw)
    return StrategySpec(**defaults)


# ---------------------------------------------------------------- (a) SPY B&H

def test_spy_buy_and_hold_exact(sp500_panel):
    panel = sp500_panel[["SPY"]].dropna()
    start, end = "2016-01-04", "2025-12-31"
    res = run(spy_spec(start, end), panel, ["SPY"])

    # Strategy accrues from the second trading day of the window; SPY's own
    # total return over the identical accrual days:
    px = panel["SPY"]
    accrual = res.accrual_index
    expected = float(px.loc[accrual[-1]] / px.loc[px.index[px.index.get_loc(accrual[0]) - 1]] - 1)
    got = metrics.total_return(res.gross_returns)
    assert got == pytest.approx(expected, abs=1e-9)

    # A one-asset static allocation never trades after the initial buy.
    assert float(res.turnover.iloc[1:].sum()) == pytest.approx(0.0, abs=1e-12)
    # At zero cost, net == gross exactly.
    assert (res.net_returns[0.0] - res.gross_returns).abs().max() == 0.0
    # At 20 bps the ONLY cost is the initial deployment: 0.5 × 20bps once.
    drag = (res.gross_returns - res.net_returns[20.0]).sum()
    assert drag == pytest.approx(0.5 * 20 / 10_000, abs=1e-12)


# ------------------------------------------------- (b) 3-ticker momentum

# Derivation (hand-computed with exact fractions, independent of the engine):
#   Panel: A=100+i, B=50 (flat) then +0.5/day after i9, C=200−i then +2/day,
#   for i = 0..33 over weekdays 2024-02-16..2024-04-03. i9=Feb29, i30=Mar29.
#   Signal Feb 29: 5-day returns A=5/104, B=0, C=−5/196 → top-2 {A, B} @ 0.5.
#   Segment 1 (i10..i30): V = 0.5·(130/109) + 0.5·(60.5/50) = 1.2013302752293578
#   Drifted wA at Mar 29 = (130/109)/((130/109)+1.21)     = 0.4963916148001069
#   Signal Mar 29: A=0.04, B=2.5/58=0.0431034, C=10/223=0.0448430 → {C, B} @ 0.5
#   Turnover at Apr 1 = 0.5·(wA + (wB−0.5) + 0.5) = 0.5 exactly (wA+wB = 1).
#   Segment 2 (i31..i33): V = 0.5·(62/60.5) + 0.5·(239/233) = 1.0252722306955628
#   Total gross = V1·V2 − 1 = 0.231690571086518
V1_EXPECTED = 1.2013302752293578
V2_EXPECTED = 1.0252722306955628
TOTAL_GROSS_EXPECTED = 0.231690571086518
FIRST_DAY_GROSS = 0.009587155963302752   # 0.5·(110/109) + 0.5·(50.5/50) − 1
FIRST_DAY_NET5 = 0.009337155963302752    # minus initial deployment 0.5·5bps


def test_three_ticker_momentum_hand_computed(three_ticker_panel):
    res = run(momentum_spec(), three_ticker_panel, ["A", "B", "C"])

    sig_dates = [str(d.date()) for d in res.target_weights.index]
    assert sig_dates == ["2024-02-29", "2024-03-29"]

    w0 = res.target_weights.iloc[0]
    assert w0.get("A") == pytest.approx(0.5, abs=1e-12)
    assert w0.get("B") == pytest.approx(0.5, abs=1e-12)
    assert w0.get("C", 0.0) == 0.0
    w1 = res.target_weights.iloc[1]
    assert w1.get("B") == pytest.approx(0.5, abs=1e-12)
    assert w1.get("C") == pytest.approx(0.5, abs=1e-12)
    assert w1.get("A", 0.0) == 0.0

    total = metrics.total_return(res.gross_returns)
    assert total == pytest.approx(TOTAL_GROSS_EXPECTED, abs=1e-6)

    assert res.gross_returns.iloc[0] == pytest.approx(FIRST_DAY_GROSS, abs=1e-9)
    assert res.net_returns[5.0].iloc[0] == pytest.approx(FIRST_DAY_NET5, abs=1e-9)

    # Turnover: initial deployment 0.5, then exactly 0.5 at the Apr 1 roll.
    apr1 = pd.Timestamp("2024-04-01")
    assert res.turnover.loc[apr1] == pytest.approx(0.5, abs=1e-9)
    assert res.turnover.iloc[0] == pytest.approx(0.5, abs=1e-12)
    assert res.turnover.sum() == pytest.approx(1.0, abs=1e-9)

    # Segment-level equity checkpoints
    eq = metrics.equity_curve(res.gross_returns)
    assert eq.loc[pd.Timestamp("2024-03-29")] == pytest.approx(V1_EXPECTED, abs=1e-9)
    assert eq.iloc[-1] == pytest.approx(V1_EXPECTED * V2_EXPECTED, abs=1e-9)

    # Independent day-by-day reference loop (no pandas, no engine code):
    px = three_ticker_panel
    dates = list(px.index)
    i9, i30 = dates[9], dates[30]
    assert str(i9.date()) == "2024-02-29" and str(i30.date()) == "2024-03-29"
    expected_daily = {}
    prev_v, basis = 1.0, {"A": float(px.loc[i9, "A"]), "B": float(px.loc[i9, "B"])}
    for d in dates[10:31]:
        v = 0.5 * float(px.loc[d, "A"]) / basis["A"] + 0.5 * float(px.loc[d, "B"]) / basis["B"]
        expected_daily[d] = v / prev_v - 1.0
        prev_v = v
    prev_v, basis = 1.0, {"B": float(px.loc[i30, "B"]), "C": float(px.loc[i30, "C"])}
    for d in dates[31:]:
        v = 0.5 * float(px.loc[d, "B"]) / basis["B"] + 0.5 * float(px.loc[d, "C"]) / basis["C"]
        expected_daily[d] = v / prev_v - 1.0
        prev_v = v
    for d, exp in expected_daily.items():
        assert res.gross_returns.loc[d] == pytest.approx(exp, abs=1e-12), d


# ------------------------------------------------- (c) overlapping cohorts

# Derivation (exact fractions, independent of the engine):
#   Dip rule: buy names down ≥2% on the day, hold 2 days, ≤2 names sized 1/n,
#   daily rebalance. Fixture: A dips 3% on Jun 4 and Jun 11; B dips 5% Jun 5.
#   Cohort1 {A:0.5} spawned Jun 4 close: earns 1/194 on Jun 5, 2/195 on Jun 6.
#   Cohort2 {B:0.5} spawned Jun 5 close: earns 1/95 on Jun 6, 1/96 on Jun 7.
#   Cohort3 {A:0.5} spawned Jun 11 close: earns 1/194 Jun 12, 2/195 Jun 13.
#   Portfolio (equal capital across active cohorts):
#     Jun 5: 1/194; Jun 6: (2/195 + 1/95)/2; Jun 7: 1/96; Jun 12: 1/194;
#     Jun 13: 2/195; all other days 0.
#   Turnover: 0.25 @Jun5 (deploy C1); 0.25064102564 @Jun6 (C2 in, drift);
#     0.25320563663 @Jun7 (C1 out); 0.25515463918 @Jun10 (C2 out);
#     0.25 @Jun12 (C3 in); 0 @Jun13 (pure drift); 0.25380710660 @Jun14 (C3 out).
COHORT_EXPECTED_GROSS = {
    "2024-06-04": 0.0,
    "2024-06-05": 0.005154639175257732,
    "2024-06-06": 0.01039136302294197,
    "2024-06-07": 0.010416666666666666,
    "2024-06-10": 0.0,
    "2024-06-11": 0.0,
    "2024-06-12": 0.005154639175257732,
    "2024-06-13": 0.010256410256410256,
    "2024-06-14": 0.0,
}
COHORT_EXPECTED_TURNOVER = {
    "2024-06-04": 0.0,
    "2024-06-05": 0.25,
    "2024-06-06": 0.25064102564102564,
    "2024-06-07": 0.2532056366328257,
    "2024-06-10": 0.2551546391752577,
    "2024-06-11": 0.0,
    "2024-06-12": 0.25,
    "2024-06-13": 0.0,
    "2024-06-14": 0.25380710659898476,
}


def dip_spec() -> StrategySpec:
    return StrategySpec(
        name="dip buyer 2-day hold",
        universe=Universe(preset="user_watchlist", resolved_tickers=["A", "B", "C"]),
        filters=[Filter(signal=Signal(type="daily_return", lookback_days=1),
                        op="lte", value=-0.02)],
        selection=Selection(
            rank_by=Signal(type="daily_return", lookback_days=1),
            direction="bottom", n=2,
        ),
        weighting="equal",
        rebalance="daily",
        entry_exit=EntryExit(hold_days=2),
        costs=Costs(sweep_bps=[0, 5, 20]),
        benchmark="SPY",
        window=Window(start="2024-06-03", end="2024-06-14"),
    )


def test_cohort_dip_buy_hand_computed(cohort_panel):
    res = run(dip_spec(), cohort_panel, ["A", "B", "C"])

    assert [str(d.date()) for d in res.accrual_index] == list(COHORT_EXPECTED_GROSS)
    for d, exp in COHORT_EXPECTED_GROSS.items():
        assert res.gross_returns.loc[pd.Timestamp(d)] == pytest.approx(exp, abs=1e-12), d
    for d, exp in COHORT_EXPECTED_TURNOVER.items():
        assert res.turnover.loc[pd.Timestamp(d)] == pytest.approx(exp, abs=1e-12), d

    # Net at 20 bps: gross − turnover × 0.002, day by day.
    for d in COHORT_EXPECTED_GROSS:
        ts = pd.Timestamp(d)
        exp_net = COHORT_EXPECTED_GROSS[d] - COHORT_EXPECTED_TURNOVER[d] * 20 / 10_000
        assert res.net_returns[20.0].loc[ts] == pytest.approx(exp_net, abs=1e-12), d


# ------------------------------------------------- (d) lookahead impossibility

def test_future_perturbation_cannot_change_past(three_ticker_panel):
    """Scaling every price after a cutoff must leave weights and returns
    before the cutoff bit-identical (audit check #2's impossibility proof)."""
    res_a = run(momentum_spec(), three_ticker_panel, ["A", "B", "C"])

    tampered = three_ticker_panel.copy()
    cutoff = pd.Timestamp("2024-03-15")
    rng = np.random.default_rng(7)
    factors = rng.uniform(0.5, 2.0, size=(int((tampered.index > cutoff).sum()),
                                          tampered.shape[1]))
    tampered.loc[tampered.index > cutoff] *= factors
    res_b = run(momentum_spec(), tampered, ["A", "B", "C"])

    # Weights at the Feb 29 signal (≤ cutoff) are identical
    pd.testing.assert_series_equal(
        res_a.target_weights.iloc[0], res_b.target_weights.iloc[0]
    )
    # Returns strictly before the cutoff are identical
    before = res_a.accrual_index[res_a.accrual_index <= cutoff]
    assert (res_a.gross_returns.loc[before] == res_b.gross_returns.loc[before]).all()


def test_same_day_fill_is_inexpressible():
    """The DSL has no execution-timing field; trying to smuggle one in is a
    validation error, not a silent ignore."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        StrategySpec(
            name="cheater",
            universe=Universe(preset="sp500"),
            selection=Selection(
                rank_by=Signal(type="return", lookback_days=126),
                direction="top", n=5,
            ),
            weighting="equal",
            rebalance="monthly",
            execute_at="signal_close",  # ← no such field exists
        )

    with pytest.raises(pydantic.ValidationError):
        # stop-loss was cut from V0 (Gap #1) — must reject, not ignore
        EntryExit(hold_days=5, stop_loss_pct=0.1)


def test_signals_ignore_future_rows(three_ticker_panel):
    from engine.signals import compute_signal

    t = pd.Timestamp("2024-03-15")
    specs = [
        Signal(type="return", lookback_days=5, skip_days=2),
        Signal(type="daily_return", lookback_days=1),
        Signal(type="volatility", lookback_days=10),
        Signal(type="sma_cross", fast=3, slow=10),
        Signal(type="drawdown_from_high", lookback_days=10),
        Signal(type="rsi", lookback_days=5),
    ]
    tampered = three_ticker_panel.copy()
    tampered.loc[tampered.index > t] *= 3.7
    for sig in specs:
        a = compute_signal(three_ticker_panel[["A", "B", "C"]], sig).loc[:t]
        b = compute_signal(tampered[["A", "B", "C"]], sig).loc[:t]
        pd.testing.assert_frame_equal(a, b), sig.type
