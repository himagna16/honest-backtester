"""Execution timing and cost accounting — the two conventions a quant checks
first, and the two this engine used to get wrong in the flattering direction.

Each test pins one of the fixes:

1. EXECUTION LAG. A signal read at the close of t cannot also be filled at
   that close. `execution_lag_days` (default 1) fills at close t+lag, and the
   capital sits in cash until then. Audited measurement: a 1-day reversal
   strategy swung ~47% of Sharpe (0.55 → 0.81) purely on this convention, with
   identical picks — so the convention, not the alpha, was the result.
2. COST BASE. Costs are charged on TRADED NOTIONAL (Σ|Δw| — dollars of stock
   that change hands per dollar of NAV), not on the one-way turnover statistic
   (0.5·Σ|Δw|). The old code charged the latter, i.e. exactly half the real
   notional, which made every "20 bps" column really 10 bps per side.
3. CASH. Uninvested capital accrues the risk-free rate WHEN THE CALLER PASSES
   ONE, and `rf_used` says which regime a run was in. conventions.py must
   describe that, not the intended end-state where the pipeline passes it.
4. STALENESS. A frozen vendor feed is screened out of EVERY selection path, on
   raw prices, with an observation-count gate (so a freshly-listed name is
   never disqualified on a sample of one) and separate entry/exit thresholds
   (so a borderline name cannot flicker and churn turnover). The exclusion is
   disclosed, and the disclosure does not overstate what was measured.
5. BENCHMARK TIMING. All three benchmarks are priced at lag 0, so they are
   mutually consistent and the 60/40 cannot drift from SPY/QQQ.
6. BAD DATA. A non-positive close is refused, in both engine paths, instead of
   becoming an infinite return with a −100% drawdown.

Expected values here are hand-computed from the fixture definitions (shown in
the comments), never read back from the engine. In particular nothing here is
allowed to assert `turnover == 0.5 × traded_notional` or
`net == gross − notional × bps`: those are the engine's own definitions, so an
assertion of them can never fail.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsl.conventions import (
    BENCHMARK_EXECUTION_LAG_DAYS,
    CONVENTIONS_TABLE,
    DEFAULT_EXECUTION_LAG_DAYS,
    INVERSE_VOL_FLOOR_ANNUAL,
    STALENESS_FLAG_SHARE,
    STALENESS_LOOKBACK_DAYS,
    STALENESS_MIN_OBSERVED_DAYS,
)
from dsl.schema import (
    Costs, EntryExit, Filter, Selection, Signal, StaticPosition, StrategySpec,
    Universe, Window,
)
from engine import metrics
from engine.backtest import EngineError, run


def _spec(tickers: list[str], **kw) -> StrategySpec:
    base = dict(
        name="execution fixture",
        universe=Universe(preset="user_watchlist", resolved_tickers=tickers),
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=5), direction="top", n=1
        ),
        weighting="equal",
        rebalance="daily",
        costs=Costs(sweep_bps=[0, 5, 20]),
        benchmark="SPY",
        window=Window(start="2024-01-02", end="2024-02-29"),
    )
    base.update(kw)
    return StrategySpec(**base)


def _ramp(anchors: list[tuple[int, float]], n: int) -> np.ndarray:
    """Piecewise-linear price path through (index, price) anchors."""
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return np.interp(np.arange(n), xs, ys)


# ------------------------------------------------------- (1) the cost base

# Fixture (exact by construction):
#   44 business days, 2024-01-01 .. 2024-02-29. Jan 31 is day 22 (the last
#   business day of January), Feb 29 is day 43.
#   A: 100 → 120 over January, then 120 → 126 over February.
#   B: 100 →  80 over January, then  80 →  84 over February.
#   Static 50/50, monthly rebalance, execution_lag_days=0 (fills at the signal
#   close), so the signal dates are day 0 (the static day-one buy) and Jan 31.
#
#   Deployment at day 0:  Δw = (+0.5, +0.5)          → notional 1.0
#   Drift to Jan 31:      value = 0.5·1.2 + 0.5·0.8 = 1.0 exactly
#                         w_drifted = (0.60, 0.40)
#   Rebalance at Jan 31:  Δw = (−0.10, +0.10)        → notional 0.2
#   Total notional 1.2 of NAV traded; one-way turnover statistic 0.6.
#   Total cost at 20 bps = 1.2 × 0.0020 = 0.0024 of NAV.
#   (The pre-fix engine charged 0.6 × 0.0020 = 0.0012 — half the truth.)
DEPLOY_NOTIONAL = 1.0
REBALANCE_NOTIONAL = 0.2
TOTAL_NOTIONAL = 1.2


@pytest.fixture()
def two_asset_panel() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    assert len(idx) == 44 and str(idx[22].date()) == "2024-01-31"
    return pd.DataFrame(
        {
            "A": _ramp([(0, 100.0), (22, 120.0), (43, 126.0)], len(idx)),
            "B": _ramp([(0, 100.0), (22, 80.0), (43, 84.0)], len(idx)),
        },
        index=idx,
    )


def _static_5050(**kw) -> StrategySpec:
    return _spec(
        ["A", "B"],
        selection=Selection(static_allocation=[
            StaticPosition(ticker="A", weight=0.5),
            StaticPosition(ticker="B", weight=0.5),
        ]),
        rebalance="monthly",
        window=Window(start="2024-01-01", end="2024-02-29"),
        **kw,
    )


def test_cost_is_charged_on_traded_notional_not_on_turnover(two_asset_panel):
    """Hand-computed: 1.0 of NAV deployed, then a 0.2 rebalance. The bps rate
    is per side, per dollar traded, so the drag is 1.2 × bps — twice what the
    old turnover-as-multiplier formula charged."""
    res = run(_static_5050(), two_asset_panel, ["A", "B"], execution_lag_days=0)

    first_accrual = res.accrual_index[0]
    feb1 = pd.Timestamp("2024-02-01")
    assert res.traded_notional.loc[first_accrual] == pytest.approx(
        DEPLOY_NOTIONAL, abs=1e-12)
    assert res.traded_notional.loc[feb1] == pytest.approx(
        REBALANCE_NOTIONAL, abs=1e-12)
    assert res.traded_notional.sum() == pytest.approx(TOTAL_NOTIONAL, abs=1e-12)

    # The turnover STATISTIC stays one-way — it is reported, never charged.
    # Pinned to the hand-derived halves of the two executions above, NOT to
    # `2 × traded_notional`: turnover is DEFINED as 0.5 × notional in the
    # engine, so comparing the two series to each other can never fail.
    assert res.turnover.loc[first_accrual] == pytest.approx(0.5, abs=1e-12)
    assert res.turnover.loc[feb1] == pytest.approx(0.1, abs=1e-12)
    assert res.turnover.sum() == pytest.approx(0.6, abs=1e-12)
    # Nothing trades on any other day of the fixture.
    assert int((res.turnover > 1e-15).sum()) == 2

    for bps in (5.0, 20.0):
        drag = (res.gross_returns - res.net_returns[bps]).sum()
        assert drag == pytest.approx(TOTAL_NOTIONAL * bps / 10_000.0, abs=1e-12)
    # The specific regression: 20 bps costs 24 bps of NAV here, not 12.
    drag20 = (res.gross_returns - res.net_returns[20.0]).sum()
    assert drag20 == pytest.approx(0.0024, abs=1e-12)


def test_deployment_from_cash_is_charged_once_not_twice(two_asset_panel):
    """Cash is carried explicitly so the weights sum to 1, but the cash LEG is
    not charged: moving 100% of NAV out of cash is one buy of 1 × NAV (1 × bps),
    while rotating a fully-invested book is a sell plus a buy (2 × bps)."""
    panel = two_asset_panel.copy()
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start="2024-01-01", end="2024-02-29"),
    )
    res = run(spec, panel, ["A"], execution_lag_days=0)
    assert res.traded_notional.sum() == pytest.approx(1.0, abs=1e-12)
    drag = (res.gross_returns - res.net_returns[20.0]).sum()
    assert drag == pytest.approx(1.0 * 20 / 10_000.0, abs=1e-12)


def test_partial_eligibility_charges_only_the_dollars_that_move():
    """k < n leaves cash. Going from 1-of-4 filled to 2-of-4 filled buys 0.25
    of NAV — the charge must be 0.25 × bps, not the full-book 1.0 × bps."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    # A always qualifies; B only qualifies in the second half (its 5-day
    # return turns positive there). The filter keeps everything else out.
    a = 100 * np.cumprod(np.r_[1.0, np.full(len(idx) - 1, 1.002)])
    b = np.r_[np.full(20, 50.0) * np.cumprod(np.r_[1.0, np.full(19, 0.999)]),
              np.full(20, 50.0) * np.cumprod(np.full(20, 1.004))]
    panel = pd.DataFrame({"A": a, "B": b}, index=idx)
    spec = _spec(
        ["A", "B"],
        filters=[Filter(signal=Signal(type="return", lookback_days=5),
                        op="gt", value=0.0)],
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=4),
        rebalance="weekly",
        window=Window(start="2024-01-15", end="2024-02-23"),
    )
    res = run(spec, panel, ["A", "B"], execution_lag_days=0)

    counts = res.eligible_counts
    assert counts.min() == 1 and counts.max() == 2  # the fixture does its job
    # Each name is sized 1/4 of NAV, so the book is never more than half
    # invested and at least half the capital is cash throughout.
    assert res.mean_cash_weight > 0.5
    # No single execution can trade more than the 0.5 of NAV that is ever
    # invested; the first is exactly the 0.25 initial deployment.
    assert res.traded_notional.iloc[0] == pytest.approx(0.25, abs=1e-12)
    assert res.traded_notional.max() <= 0.5 + 1e-12


# --------------------------------------------------- (2) the execution lag

def test_default_execution_lag_is_one_day_and_is_reported():
    assert DEFAULT_EXECUTION_LAG_DAYS == 1


def test_lag_shifts_the_entry_price_to_the_next_close():
    """Same picks, same accrual days — only the fill price moves. The fixture
    jumps 20% on the day after the buy signal, so a same-day fill captures the
    jump and an honest 1-day-lagged fill does not."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    px = np.full(30, 100.0)
    px[16:] = 120.0          # +20% on day 16, then flat at the new level
    panel = pd.DataFrame({"A": px, "B": np.linspace(50.0, 51.0, 30)}, index=idx)

    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start=str(idx[15].date()), end=str(idx[-1].date())),
    )
    same_day = run(spec, panel, ["A"], execution_lag_days=0)
    lagged = run(spec, panel, ["A"], execution_lag_days=1)

    # Identical accrual window: the lag changes the fill, not the calendar.
    assert list(lagged.accrual_index) == list(same_day.accrual_index)
    assert same_day.execution_lag_days == 0 and lagged.execution_lag_days == 1

    # Same-day fill buys at 100 (the close that produced the signal) and books
    # the +20% jump; the 1-day fill buys at 120 and books nothing.
    assert metrics.total_return(same_day.gross_returns) == pytest.approx(0.2, abs=1e-12)
    assert metrics.total_return(lagged.gross_returns) == pytest.approx(0.0, abs=1e-12)
    # The lagged run's first accrual day is pre-fill: cash, not the asset.
    assert lagged.gross_returns.iloc[0] == 0.0
    assert lagged.fill_dates[0] == same_day.fill_dates[0] + pd.Timedelta(days=1)


def test_same_day_fill_flatters_a_one_day_reversal_strategy():
    """The audited credibility-killer, reproduced in miniature: a 1-day
    reversal on a bouncing series is hugely profitable if you may trade at the
    close you measured, and worthless one day later — identical picks."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    # Deterministic zig-zag: down 4%, up 5%, repeat. Buying the down day at
    # its own close captures every bounce.
    steps = np.where(np.arange(59) % 2 == 0, 0.96, 1.05)
    a = 100 * np.cumprod(np.r_[1.0, steps])
    b = np.linspace(200.0, 210.0, 60)   # a calm alternative, never the dipper
    panel = pd.DataFrame({"A": a, "B": b}, index=idx)
    spec = _spec(
        ["A", "B"],
        filters=[Filter(signal=Signal(type="daily_return", lookback_days=1),
                        op="lt", value=-0.02)],
        selection=Selection(rank_by=Signal(type="daily_return", lookback_days=1),
                            direction="bottom", n=1),
        rebalance="daily",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    same_day = run(spec, panel, ["A", "B"], execution_lag_days=0)
    lagged = run(spec, panel, ["A", "B"], execution_lag_days=1)

    picks_same = [p["ticker"] for h in same_day.holdings for p in h["picks"]]
    picks_lag = [p["ticker"] for h in lagged.holdings for p in h["picks"]]
    assert picks_same == picks_lag and picks_same  # identical decisions

    r_same = metrics.total_return(same_day.gross_returns)
    r_lag = metrics.total_return(lagged.gross_returns)
    assert r_same > 0.20          # "alpha" that is really a fill assumption
    assert r_lag < 0.0            # the same decisions, executably timed
    assert metrics.sharpe(same_day.gross_returns, 0.0) > \
        metrics.sharpe(lagged.gross_returns, 0.0)


def test_cohort_mode_respects_the_lag_too():
    """The cohort path had the same same-day fill. A cohort is established at
    its FILL close, so with lag 1 every cohort's entry price is one day later
    and the first accrual day is cash."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    # A ticks ±0.1% on alternate days (a live feed — a literally flat series
    # here would be screened out as frozen, which is a different test), then
    # jumps to 130 on day 20 and holds.
    a = 100.0 + 0.1 * np.where(np.arange(40) % 2 == 0, 1.0, -1.0)
    a[20:] = 130.0
    panel = pd.DataFrame({"A": a, "B": np.linspace(10.0, 11.0, 40)}, index=idx)
    # Day 19's close is 99.9, so a same-day fill books 130/99.9 − 1 = 30.13%.
    assert a[19] == pytest.approx(99.9, abs=1e-12)
    spec = _spec(
        ["A"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=1),
        entry_exit=EntryExit(hold_days=3),
        rebalance="daily",
        # The window opens on day 19 — the first cohort of the run is the one
        # whose fill date the lag moves, with no earlier cohort to mask it.
        window=Window(start=str(idx[19].date()), end=str(idx[-1].date())),
    )
    same_day = run(spec, panel, ["A"], execution_lag_days=0)
    lagged = run(spec, panel, ["A"], execution_lag_days=1)

    day20 = idx[20]
    # Signal at day 19's close (99.9) → same-day fill at 99.9 books the jump.
    assert same_day.gross_returns.loc[day20] == pytest.approx(
        130.0 / 99.9 - 1.0, abs=1e-12)
    # With a 1-day lag the first cohort is bought at day 20's close of 130
    # instead, so day 20 is spent in cash and the jump is earned by nobody.
    assert lagged.gross_returns.loc[day20] == pytest.approx(0.0, abs=1e-12)
    assert lagged.execution_lag_days == 1


def test_execution_lag_is_not_a_spec_field():
    """The lag is a run-level convention. The DSL stays a closed vocabulary —
    a spec that tries to buy itself a better fill is a validation error, not a
    silently-ignored key."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        StrategySpec(
            name="cheater",
            universe=Universe(preset="sp500"),
            selection=Selection(rank_by=Signal(type="return", lookback_days=21),
                                direction="top", n=5),
            weighting="equal",
            rebalance="monthly",
            execution_lag_days=0,  # ← no such field exists
        )


def test_negative_lag_is_refused(two_asset_panel):
    with pytest.raises(EngineError, match="in the past"):
        run(_static_5050(), two_asset_panel, ["A", "B"], execution_lag_days=-1)


# ------------------------------------------------------------- (3) cash rf

RF_DAILY = 0.0002   # ≈5% annualized, roughly ^IRX over the 2023-24 window


def test_all_cash_book_earns_the_risk_free_rate():
    """Nothing qualifies → 100% cash. Cash at 0% against an rf-based Sharpe
    hurdle would penalise the strategy on the numerator AND the hurdle."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    panel = pd.DataFrame(
        {"A": np.linspace(100.0, 90.0, 40), "B": np.linspace(50.0, 45.0, 40)},
        index=idx,
    )
    spec = _spec(
        ["A", "B"],
        filters=[Filter(signal=Signal(type="return", lookback_days=5),
                        op="gt", value=5.0)],   # nobody ever returns +500%
        rebalance="weekly",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    res = run(spec, panel, ["A", "B"], rf_daily=RF_DAILY)

    assert (res.eligible_counts == 0).all()
    assert res.mean_cash_weight == pytest.approx(1.0, abs=1e-12)
    assert res.traded_notional.sum() == 0.0          # nothing was ever traded
    assert (res.gross_returns - RF_DAILY).abs().max() < 1e-15
    n = len(res.accrual_index)
    assert metrics.total_return(res.gross_returns) == pytest.approx(
        (1.0 + RF_DAILY) ** n - 1.0, abs=1e-12)

    # Without an rf the same book reports a flat zero — the old behaviour,
    # kept as the explicit opt-out rather than the silent default.
    zero = run(spec, panel, ["A", "B"])
    assert zero.gross_returns.abs().sum() == 0.0


def test_pre_fill_days_earn_cash_not_zero():
    """The days between the first signal and its fill are uninvested, so they
    earn rf — the lag must not silently cost the strategy those days."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    panel = pd.DataFrame(
        {"A": np.linspace(100.0, 130.0, 30), "B": np.linspace(20.0, 21.0, 30)},
        index=idx,
    )
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    res = run(spec, panel, ["A"], rf_daily=RF_DAILY)
    assert res.gross_returns.iloc[0] == pytest.approx(RF_DAILY, abs=1e-15)


def test_partial_cash_sleeve_accrues_rf():
    """A k<n book holds real cash. That sleeve earns rf every day, so the run
    with an rf strictly beats the identical run at 0%."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    panel = pd.DataFrame(
        {"A": np.linspace(100.0, 110.0, 60), "B": np.linspace(40.0, 39.0, 60)},
        index=idx,
    )
    spec = _spec(
        ["A", "B"],
        filters=[Filter(signal=Signal(type="return", lookback_days=5),
                        op="gt", value=0.0)],       # only A ever qualifies
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=4),  # sized as 1-of-4 → 75% cash
        rebalance="weekly",
        window=Window(start=str(idx[20].date()), end=str(idx[-1].date())),
    )
    with_rf = run(spec, panel, ["A", "B"], rf_daily=RF_DAILY)
    without = run(spec, panel, ["A", "B"])

    assert with_rf.mean_cash_weight == pytest.approx(0.75, abs=0.02)
    n = len(with_rf.accrual_index)
    gap = metrics.total_return(with_rf.gross_returns) - \
        metrics.total_return(without.gross_returns)
    # ≈ 75% of the capital compounding at rf for n days.
    assert gap == pytest.approx(0.75 * RF_DAILY * n, rel=0.05)


def test_cohort_mode_accrues_rf_on_idle_capital():
    idx = pd.bdate_range("2024-01-01", periods=40)
    panel = pd.DataFrame(
        {"A": np.linspace(100.0, 95.0, 40), "B": np.linspace(10.0, 9.5, 40)},
        index=idx,
    )
    spec = _spec(
        ["A", "B"],
        filters=[Filter(signal=Signal(type="daily_return", lookback_days=1),
                        op="lt", value=-0.5)],   # nothing ever halves in a day
        entry_exit=EntryExit(hold_days=2),
        rebalance="daily",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    res = run(spec, panel, ["A", "B"], rf_daily=RF_DAILY)
    assert (res.gross_returns - RF_DAILY).abs().max() < 1e-15


# ------------------------------------------------------- (4) stale feeds

def _stale_panel(freeze_from: int | None = None, n: int = 200) -> pd.DataFrame:
    """A 3-name panel where FLAT's feed is frozen (optionally only from a
    given day), which is what a halt or a dead vendor symbol looks like."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(11)
    live = {tk: 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, n))
            for tk in ("A", "B")}
    flat = 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, n))
    if freeze_from is None:
        flat[:] = 42.0
    else:
        flat[freeze_from:] = flat[freeze_from - 1]
    return pd.DataFrame({**live, "FLAT": flat}, index=idx)


def test_frozen_feed_is_screened_out_of_a_volatility_ranking():
    """The old guard lived inside the inverse_vol branch and thresholded a
    rolling std at 1e-10 — a frozen series measures ~1e-8, so it never fired
    and 'rank by lowest volatility' bought the broken feed every time."""
    panel = _stale_panel()
    spec = _spec(
        ["A", "B", "FLAT"],
        selection=Selection(rank_by=Signal(type="volatility", lookback_days=63),
                            direction="bottom", n=1),
        rebalance="monthly",
        window=Window(start="2023-07-03", end="2023-10-13"),
    )
    res = run(spec, panel, ["A", "B", "FLAT"])

    # FLAT genuinely is the lowest-volatility name — that is the trap.
    from engine.signals import compute_signal
    vol = compute_signal(panel, Signal(type="volatility", lookback_days=63))
    assert vol.iloc[-1].idxmin() == "FLAT"

    held = {p["ticker"] for h in res.holdings for p in h["picks"]}
    assert "FLAT" not in held
    # FLAT is frozen outright, so the disclosure is allowed to say "zero
    # measured volatility" — and must name the ticker.
    assert any("frozen" in note for note in res.data_notes)
    assert any("FLAT" in note for note in res.data_notes)


def test_frozen_feed_is_screened_out_of_a_drawdown_ranking():
    """Same screen, a different ranking path: a frozen price never draws down,
    so it also tops 'smallest drawdown from high'."""
    panel = _stale_panel()
    spec = _spec(
        ["A", "B", "FLAT"],
        selection=Selection(
            rank_by=Signal(type="drawdown_from_high", lookback_days=63),
            direction="top", n=1),
        rebalance="monthly",
        window=Window(start="2023-07-03", end="2023-10-13"),
    )
    res = run(spec, panel, ["A", "B", "FLAT"])
    held = {p["ticker"] for h in res.holdings for p in h["picks"]}
    assert "FLAT" not in held


def test_staleness_screen_releases_a_ticker_when_its_feed_recovers():
    """The screen is a rolling state, not a permanent ban: a name frozen for
    part of the sample is ineligible only while the freeze is inside its
    trailing window."""
    n = 200
    freeze_from = 60
    panel = _stale_panel(freeze_from=freeze_from, n=n)
    # …then the feed resumes for the rest of the sample.
    rng = np.random.default_rng(3)
    idx = panel.index
    resume = 120
    path = panel["FLAT"].to_numpy().copy()
    path[resume:] = path[resume - 1] * np.cumprod(
        1 + rng.normal(0.004, 0.011, n - resume))
    panel["FLAT"] = path

    from engine.backtest import _staleness_panel
    stale = _staleness_panel(panel)["FLAT"]
    # Frozen once the whole trailing window sits inside the freeze…
    assert bool(stale.iloc[freeze_from + STALENESS_LOOKBACK_DAYS])
    # …and clean again once the trailing window is entirely post-recovery.
    assert not bool(stale.iloc[resume + STALENESS_LOOKBACK_DAYS])
    # A window with less history than the lookback is "unknown", never stale.
    assert not bool(stale.iloc[STALENESS_LOOKBACK_DAYS - 2])


def test_stale_screen_does_not_disqualify_a_normal_ticker():
    """The screen must not fire on ordinary names — a false positive here is a
    silent selection bias in the other direction."""
    from engine.backtest import _staleness_panel

    panel = _stale_panel()
    stale = _staleness_panel(panel)
    assert not stale[["A", "B"]].to_numpy().any()


# ------------------------------------------- (5) previously-free trades

def test_trade_decided_on_the_final_day_is_charged():
    """Cohort mode guarded the turnover write with `if i + 1 < len(accrual)`,
    so the expiries and spawn decided at the last close traded for free.

    Fixture: A and B zig-zag in exact anti-phase, so ranking on yesterday's
    return flips the pick EVERY day, including the last one. With hold_days=1
    that means a full 1.0-out / 1.0-in rotation daily → notional 2.0 a day."""
    n = 40
    idx = pd.bdate_range("2024-01-01", periods=n)
    up_first = np.where(np.arange(n - 1) % 2 == 0, 1.02, 0.98)
    a = 100 * np.cumprod(np.r_[1.0, up_first])
    b = 100 * np.cumprod(np.r_[1.0, 1.0 / up_first])
    panel = pd.DataFrame({"A": a, "B": b}, index=idx)
    spec = _spec(
        ["A", "B"],
        selection=Selection(rank_by=Signal(type="daily_return", lookback_days=1),
                            direction="top", n=1),
        entry_exit=EntryExit(hold_days=1),
        rebalance="daily",
        window=Window(start=str(idx[20].date()), end=str(idx[-1].date())),
    )
    res = run(spec, panel, ["A", "B"])

    # A mid-run day rotates the whole book: sell 1.0 of NAV, buy 1.0 of NAV.
    assert res.traded_notional.iloc[-2] == pytest.approx(2.0, abs=1e-12)
    # The final day carries that same 2.0 rotation (decided at the previous
    # close) PLUS the 1.0 liquidation of the cohort that expires at the final
    # close — the trade the old `if i + 1 < len(accrual)` guard threw away.
    assert res.traded_notional.iloc[-1] == pytest.approx(3.0, abs=1e-12)
    assert res.turnover.iloc[-1] == pytest.approx(1.5, abs=1e-12)
    assert res.net_returns[20.0].iloc[-1] == pytest.approx(
        res.gross_returns.iloc[-1] - 3.0 * 20 / 10_000.0, abs=1e-12)


def test_every_bps_column_is_a_pure_function_of_the_notional(two_asset_panel):
    """The relationship the report legend claims: net = gross − notional ×
    bps/1e4, for every column, including 0.

    Pinned to the fixture's HAND-DERIVED numbers, not to a recomputation of
    `res.gross_returns − res.traded_notional × bps/1e4` — that expression is
    the engine's own line 459, so comparing the result to it could never fail.

    Fixture arithmetic (see DEPLOY_NOTIONAL above): A and B are exactly
    anti-symmetric ramps around 100, so on the first accrual day
    value = 0.5·(100 + 20/22)/100 + 0.5·(100 − 20/22)/100 = 1.0 exactly and the
    gross return is 0. Notional that day is the 1.0 deployment, so the net
    return is −1.0 × bps/1e4 with NO gross term to cancel against.
    """
    res = run(_static_5050(), two_asset_panel, ["A", "B"], execution_lag_days=0)
    first = res.accrual_index[0]
    feb1 = pd.Timestamp("2024-02-01")

    assert res.gross_returns.loc[first] == pytest.approx(0.0, abs=1e-15)
    for bps in (0.0, 5.0, 20.0):
        assert res.net_returns[bps].loc[first] == pytest.approx(
            -DEPLOY_NOTIONAL * bps / 10_000.0, abs=1e-15), bps
        # The Feb 1 rebalance moves 0.2 of NAV, so it costs 0.2 × bps.
        assert res.net_returns[bps].loc[feb1] - res.gross_returns.loc[feb1] == \
            pytest.approx(-REBALANCE_NOTIONAL * bps / 10_000.0, abs=1e-15), bps

    # On every day with no execution the net columns are the gross column
    # exactly — a cost that appears on a non-trading day is a leak.
    quiet = res.accrual_index.drop([first, feb1])
    for bps, series in res.net_returns.items():
        assert (series.loc[quiet] - res.gross_returns.loc[quiet]).abs().max() == 0.0
    assert res.net_returns[0.0].equals(res.gross_returns)


# ----------------------------------------------------- (6) benchmark timing

def _bench_spec_free_panel(spy: list[float], agg: list[float],
                           qqq: list[float] | None = None) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(spy))
    data = {"SPY": spy, "AGG": agg}
    data["QQQ"] = qqq if qqq is not None else spy
    return pd.DataFrame(data, index=idx)


def test_sixty_forty_benchmark_first_day_is_the_real_close_to_close_move():
    """THE regression. benchmark.py computes 60/40 through run(), so when the
    engine's default lag became 1 the 60/40 silently inherited it: its first
    accrual day became a fabricated 0.0 (cash, pre-fill) and the whole series
    shifted, while the SPY/QQQ benchmarks beside it stayed raw pct_change. On
    the project's own 2016-2025 fixture that was +0.95pp of total return on the
    one number printed on every report.

    Hand arithmetic: SPY compounds at exactly ×1.10 a day, AGG at exactly
    ×1.02. A 60/40 book bought at day 0's close returns
    0.6 × 0.10 + 0.4 × 0.02 = 0.068 on day 1, exactly.
    On day 2 the weights have drifted: the book is worth
    0.6 × 1.10 + 0.4 × 1.02 = 1.068 and then 0.6 × 1.21 + 0.4 × 1.0404
    = 1.14216, so the return is 1.14216/1.068 − 1 = 0.07416/1.068.
    """
    spy = [100.0 * 1.10 ** i for i in range(10)]
    agg = [50.0 * 1.02 ** i for i in range(10)]
    prices = _bench_spec_free_panel(spy, agg)
    accrual = prices.index[1:]

    from engine.benchmark import benchmark_returns
    sf = benchmark_returns("sixty_forty", prices, accrual)

    assert sf.iloc[0] == pytest.approx(0.068, abs=1e-12)
    assert sf.iloc[1] == pytest.approx(0.07416 / 1.068, abs=1e-12)
    # The specific symptom of the inherited lag: a flat first day.
    assert sf.iloc[0] != 0.0


def test_all_three_benchmarks_are_timed_identically():
    """Benchmarks must be mutually consistent. Give SPY, QQQ and AGG the SAME
    price path: then 0.6·SPY + 0.4·AGG is arithmetically that same path, so all
    three benchmark series must be equal day for day. This is an identity, not
    a recomputation of the engine — any timing difference between the
    pct_change branch and the run() branch breaks it, which is exactly what a
    lag on only one of them is.
    """
    path = [100.0, 103.0, 99.0, 104.5, 101.0, 107.0, 110.0, 108.0, 112.0, 115.0]
    prices = _bench_spec_free_panel(path, path, path)
    accrual = prices.index[1:]

    from engine.benchmark import benchmark_returns
    spy = benchmark_returns("SPY", prices, accrual)
    qqq = benchmark_returns("QQQ", prices, accrual)
    sf = benchmark_returns("sixty_forty", prices, accrual)

    # Independent reference: plain close-to-close on the shared path.
    expected = [path[i + 1] / path[i] - 1.0 for i in range(len(path) - 1)]
    assert spy.round(12).tolist() == [round(e, 12) for e in expected]
    assert (spy - qqq).abs().max() == 0.0
    assert (sf - spy).abs().max() < 1e-12, (
        "the 60/40 benchmark is timed differently from SPY/QQQ — it inherited "
        "the strategy execution lag"
    )


def test_benchmark_lag_is_zero_and_is_not_the_strategy_default():
    """If the two constants were ever equal the test above would pass by
    accident, so pin that they differ and that the benchmark one is zero."""
    assert BENCHMARK_EXECUTION_LAG_DAYS == 0
    assert BENCHMARK_EXECUTION_LAG_DAYS != DEFAULT_EXECUTION_LAG_DAYS
    import inspect

    from engine import benchmark as bench_mod
    src = inspect.getsource(bench_mod)
    assert "execution_lag_days=BENCHMARK_EXECUTION_LAG_DAYS" in src, (
        "benchmark.py must pass the lag explicitly; omitting it silently "
        "inherits DEFAULT_EXECUTION_LAG_DAYS"
    )


# ------------------------------- (7) conventions.py must describe the code

def _row(name: str) -> str:
    return next(r["rule"] for r in CONVENTIONS_TABLE if r["convention"] == name)


def test_cash_convention_describes_what_the_engine_actually_does():
    """conventions.py is served at /api/conventions and embedded in the LLM
    system prompt. It used to assert flatly that cash 'accrues the daily
    risk-free rate', while the pipeline calls run() with no rf at all — the
    same defect class (a docstring promising a guarantee the code does not
    honour) that the audit called a credibility-killer, one file over.
    """
    idx = pd.bdate_range("2024-01-01", periods=40)
    panel = pd.DataFrame(
        {"A": np.linspace(100.0, 90.0, 40), "B": np.linspace(50.0, 45.0, 40)},
        index=idx,
    )
    spec = _spec(
        ["A", "B"],
        filters=[Filter(signal=Signal(type="return", lookback_days=5),
                        op="gt", value=5.0)],       # nobody ever qualifies
        rebalance="weekly",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    # SHIPPED behaviour, both regimes:
    default = run(spec, panel, ["A", "B"])
    supplied = run(spec, panel, ["A", "B"], rf_daily=RF_DAILY)
    assert default.gross_returns.abs().max() == 0.0     # cash earns 0%
    assert default.rf_used is None
    assert (supplied.gross_returns - RF_DAILY).abs().max() < 1e-15
    assert supplied.rf_used == pytest.approx(RF_DAILY, abs=1e-15)

    # …therefore the convention text must be conditional and must say which
    # regime the shipped pipeline runs in.
    rule = _row("Cash").lower()
    assert "when the caller supplies one" in rule
    assert "0%" in rule and "rf_used" in rule


def test_execution_lag_convention_describes_what_the_engine_actually_does():
    idx = pd.bdate_range("2024-01-01", periods=20)
    panel = pd.DataFrame({"A": np.linspace(100.0, 120.0, 20),
                          "B": np.linspace(50.0, 55.0, 20)}, index=idx)
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start=str(idx[0].date()), end=str(idx[-1].date())),
    )
    res = run(spec, panel, ["A"])
    rule = _row("Execution lag (fill timing)")

    assert res.execution_lag_days == DEFAULT_EXECUTION_LAG_DAYS
    assert f"Default {DEFAULT_EXECUTION_LAG_DAYS} trading day" in rule
    # The lag is carried on the result — which is what the row may claim.
    assert "EngineResult.execution_lag_days" in rule
    # It must NOT claim the reports print it: no report payload carries it.
    assert "every report states the lag" not in rule


def test_benchmark_timing_convention_exists_and_matches_the_constant():
    rule = _row("Benchmark timing")
    assert f"execution lag {BENCHMARK_EXECUTION_LAG_DAYS}" in rule
    assert "60/40" in rule


# ------------------------------------------- (8) the staleness screen, fixed

def _diff_pattern_prices(zero_runs: list[tuple[int, int]], n: int) -> pd.Series:
    """Prices whose day-over-day diff is exactly 0 on the given [lo, hi] row
    ranges and +1 everywhere else. Row 0 has no diff."""
    steps = np.ones(n)
    steps[0] = 0.0
    for lo, hi in zero_runs:
        steps[lo:hi + 1] = 0.0
    return pd.Series(100.0 + np.cumsum(steps),
                     index=pd.bdate_range("2023-01-02", periods=n))


def test_a_ticker_with_one_observation_is_unknown_not_stale():
    """The guard used to count ROWS, so a name with 21 rows of history but a
    single OBSERVED close-to-close move could be disqualified on a sample of
    one — VRT on 2018-08-03 and LITE on 2015-07-24 in this project's own S&P
    fixture, both freshly-listed real names.

    Fixture: NEW has no prices until row 40, then prints the same close twice.
    Its one observed diff is zero: 1 of 1 unchanged, a 100% 'stale' share on a
    sample of one.
    """
    from engine.backtest import _staleness_panel, _staleness_shares

    n = 60
    live = 100 + np.arange(n, dtype=float)
    new = np.full(n, np.nan)
    new[40] = 20.0
    new[41] = 20.0          # the single observed move, and it is zero
    panel = pd.DataFrame({"LIVE": live, "NEW": new},
                         index=pd.bdate_range("2023-01-02", periods=n))

    diffs = panel["NEW"].diff()
    observed = int(diffs.iloc[:42].notna().sum())
    assert observed == 1 and float(diffs.iloc[41]) == 0.0   # the fixture bites
    assert observed < STALENESS_MIN_OBSERVED_DAYS

    shares = _staleness_shares(panel)
    stale = _staleness_panel(panel)
    assert np.isnan(shares["NEW"].iloc[41])       # unknown, not 1.0
    assert not bool(stale["NEW"].iloc[41])
    assert not bool(stale["NEW"].iloc[42])


def test_a_thin_but_adequate_sample_is_still_judged():
    """The gate is on the DENOMINATOR, not a blanket refusal: once the window
    holds enough observed moves the screen judges normally. Fixture: a name
    with exactly STALENESS_MIN_OBSERVED_DAYS observed diffs, every one of them
    zero — a genuinely frozen feed, flagged."""
    from engine.backtest import _staleness_panel, _staleness_shares

    n = 60
    live = 100 + np.arange(n, dtype=float)
    thin = np.full(n, np.nan)
    m = STALENESS_MIN_OBSERVED_DAYS
    thin[30:31 + m] = 20.0          # m + 1 prints, all identical → m zero diffs
    panel = pd.DataFrame({"LIVE": live, "THIN": thin},
                         index=pd.bdate_range("2023-01-02", periods=n))

    row = 30 + m
    assert int(panel["THIN"].diff().iloc[:row + 1].notna().sum()) == m
    assert _staleness_shares(panel)["THIN"].iloc[row] == pytest.approx(1.0)
    assert bool(_staleness_panel(panel)["THIN"].iloc[row])


# The threshold fixture, hand-derived. One ticker, diffs zero on rows 30..43
# (14 of them), live on rows 44..54, zero again on rows 55..68 (14), live after.
# Trailing-21 zero count at row t = |[30,43] n [t-20,t]| + |[55,68] n [t-20,t]|:
#   rising into block 1:  count = t - 29           -> 13 at t=42, 14 at t=43
#   between the blocks:   count = (64 - t) + (t - 54) = 10, flat for t=54..63
#   rising into block 2:  count = t - 54           -> 13 at t=67, 14 at t=68
#   decaying after:       count = 89 - t           -> 8 at t=81
# Shares are those counts over 21, so the 0.60 flag level (count >= 13) holds
# rows 42..50 and 67..75, and the old 0.50 level (count >= 11) held 40..53 and
# 65..77. The 10/21 = 0.476 stretch is flagged by NEITHER: it is the noise band
# a 21-sample proportion cannot resolve.
THRESH_FIRST_FLAG_ROW = 42


def test_the_flag_threshold_is_where_the_arithmetic_says_it_is():
    from engine.backtest import _staleness_panel, _staleness_shares

    n = 100
    px = _diff_pattern_prices([(30, 43), (55, 68)], n)
    panel = pd.DataFrame({"X": px, "LIVE": 100 + 2.0 * np.arange(n)},
                         index=px.index)
    share = _staleness_shares(panel)["X"]

    # The hand-derived counts above, confirmed on the fixture.
    assert share.iloc[43] == pytest.approx(14 / 21)
    assert share.iloc[54] == pytest.approx(10 / 21)
    assert share.iloc[63] == pytest.approx(10 / 21)
    assert share.iloc[67] == pytest.approx(13 / 21)
    assert share.iloc[81] == pytest.approx(8 / 21)

    state = _staleness_panel(panel)["X"]
    flagged = np.flatnonzero(state.to_numpy())
    assert flagged[0] == THRESH_FIRST_FLAG_ROW
    # 12/21 = 0.571 was flagged by the old 0.5 cut-off and is not flagged now.
    assert share.iloc[41] == pytest.approx(12 / 21)
    assert not bool(state.iloc[41])
    assert bool(_staleness_shares(panel)["X"].iloc[41] > 0.5)   # the old rule fired


def test_the_flag_threshold_cuts_the_false_positive_rate_on_a_21_day_sample():
    """The audit's own statistic. A name whose TRUE zero-change share is 0.45
    is a live, thinly-traded feed, not a dead one; disqualifying it costs a
    forced sell plus a repurchase at 2 x bps. Exact binomial with n = 21:

        P(flagged) = P(X/21 > cut) = sum_{i>=k} C(21,i) p^i (1-p)^(21-i)

    at p = 0.45 the old cut of 0.50 needs X >= 11 and fires on 32.1% of days;
    the shipped cut of 0.60 needs X >= 13 and fires on 9.1%.
    """
    from math import comb

    def tail(k: int, n: int, p: float) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    n = STALENESS_LOOKBACK_DAYS
    k_old = int(np.floor(0.50 * n)) + 1
    k_new = int(np.floor(STALENESS_FLAG_SHARE * n)) + 1
    assert (k_old, k_new) == (11, 13)

    old_fp = tail(k_old, n, 0.45)
    new_fp = tail(k_new, n, 0.45)
    assert old_fp == pytest.approx(0.3210, abs=5e-4)
    assert new_fp == pytest.approx(0.0908, abs=5e-4)
    assert new_fp < old_fp / 3.0
    # …and the screen still fires on the thing it is for: a genuinely dead
    # feed has a true share of 1.0, flagged with certainty.
    assert tail(k_new, n, 1.0) == pytest.approx(1.0, abs=1e-12)


def test_the_staleness_screen_has_no_memory_beyond_its_lookback():
    """A CROSS-MODULE CONTRACT, not a style preference.

    validation/splits.py computes the walk-forward purge length from
    STALENESS_LOOKBACK_DAYS. That is only a valid bound if the stale verdict at
    t depends on closes in (t - STALENESS_LOOKBACK_DAYS, t] and on NOTHING
    earlier. Add hysteresis (or any carried state) and a training-set close
    starts reaching arbitrarily far into the test set — turnover churn becomes
    LOOKAHEAD LEAKAGE. This test fails the moment that happens, in this file,
    instead of showing up as an unexplained failure in the validation suite.

    Derivation of the bound: moving the close at row r changes diff[r] and
    diff[r+1]. diff[j] enters the windows ending at rows j..j+20, so the only
    verdicts that may move are rows r..r+21 = r..r+STALENESS_LOOKBACK_DAYS.
    """
    from engine.backtest import _staleness_panel

    n = 160
    r = 60
    # A feed sitting right in the noise band, where a nudge really can flip
    # the verdict — the hardest case for the bound.
    px = _diff_pattern_prices([(30, 41), (50, 61), (70, 81), (90, 101)], n)
    panel = pd.DataFrame({"X": px}, index=px.index)
    bumped = panel.copy()
    bumped.iloc[r, 0] = float(bumped.iloc[r, 0]) + 0.5   # splits one zero diff

    base = _staleness_panel(panel)["X"].to_numpy()
    moved = _staleness_panel(bumped)["X"].to_numpy()
    differs = np.flatnonzero(base != moved)

    assert len(differs) > 0, "the probe is inert — it proves nothing"
    assert differs.min() >= r
    assert differs.max() <= r + STALENESS_LOOKBACK_DAYS, (
        "a close at row %d changed the stale verdict %d rows later; the purge "
        "length validation/splits.py derives from STALENESS_LOOKBACK_DAYS no "
        "longer bounds this screen" % (r, int(differs.max()) - r)
    )


def test_stale_disclosure_does_not_claim_zero_volatility_for_a_name_that_moved():
    """The note used to tell the user every excluded name had 'zero measured
    volatility'. A name flagged at 13 of 21 unchanged closes moved on the other
    8 sessions; saying otherwise is a false statement in the disclosure the
    product exists to be trusted for."""
    from engine.backtest import _SelectionEngine, _stale_notes

    n = 100
    partial = _diff_pattern_prices([(30, 43), (55, 68)], n)
    panel = pd.DataFrame(
        {"PARTIAL": partial.to_numpy(),
         "FROZEN": np.full(n, 42.0),
         "LIVE": 100 + 2.0 * np.arange(n)},
        index=partial.index,
    )
    spec = _spec(
        ["PARTIAL", "FROZEN", "LIVE"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=3),
        rebalance="weekly",
        window=Window(start=str(panel.index[40].date()),
                      end=str(panel.index[-1].date())),
    )
    sel = _SelectionEngine(spec, panel, ["PARTIAL", "FROZEN", "LIVE"])
    for t in panel.index[40:]:
        sel.weights_at(t)

    assert sel.stale_excluded["FROZEN"] == pytest.approx(1.0)
    assert sel.stale_excluded["PARTIAL"] < 1.0
    assert "LIVE" not in sel.stale_excluded

    notes = _stale_notes(sel)
    frozen_note = [n_ for n_ in notes if "FROZEN" in n_]
    partial_note = [n_ for n_ in notes if "PARTIAL" in n_]
    assert len(frozen_note) == 1 and len(partial_note) == 1
    # FROZEN really never moved, so the strong claim is allowed there…
    assert "zero measured volatility" in frozen_note[0]
    assert "FROZEN" in frozen_note[0] and "PARTIAL" not in frozen_note[0]
    # …and is forbidden for a name that did move.
    assert "zero measured volatility" not in partial_note[0]
    assert "did still move" in partial_note[0]


# --------------------------- (9) the inverse-vol capture the screen misses

def _near_frozen_panel(cents_per_day: float, n: int = 200) -> pd.DataFrame:
    """Two live names plus Z, a feed that ticks `cents_per_day` every session:
    never exactly unchanged, so the staleness screen never sees it."""
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, n)),
         "B": 100 * np.cumprod(1 + rng.normal(0.0003, 0.013, n)),
         "Z": 50.0 + cents_per_day * np.arange(n)},
        index=pd.bdate_range("2023-01-02", periods=n),
    )


def test_a_near_frozen_feed_is_sized_at_the_floor_not_at_its_measured_vol():
    """The staleness screen only catches EXACTLY frozen feeds. A name that
    moves one cent every single session has one zero diff in the whole sample,
    is never flagged, measures ~0.001% annualized vol, and unfloored 1/σ then
    hands it 99.99% of the book with no disclosure at all.

    This test is deliberately run on a TWO-name book, because the version that
    preceded it did not test the floor at all: on its three-name book the
    position cap max(25%, 2/3) = 66.7% carried the only assertion, so deleting
    the floor entirely left it green. With k = 2 the cap is max(25%, 2/2) = 1.0
    and is therefore inert by construction, which leaves the floor as the only
    thing acting on the weights — so the number below is the floor's doing and
    it is pinned exactly, against 1/σ arithmetic computed here by hand.

    Note what is NOT claimed: at a 0.5% floor a near-frozen name still takes
    97% of a two-name book. The floor bounds the TILT (σ_A/floor ≈ 34:1 here
    instead of 15,000:1); on books of three or more it is the cap, asserted
    below, that stops the capture.
    """
    from engine.signals import INVERSE_VOL_LOOKBACK, compute_signal

    panel = _near_frozen_panel(0.01)              # one cent a day, forever
    idx = panel.index
    pair = _spec(
        ["A", "Z"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=21),
                            direction="top", n=2),
        weighting="inverse_vol",
        rebalance="monthly",
        window=Window(start=str(idx[100].date()), end=str(idx[-1].date())),
    )
    res = run(pair, panel[["A", "Z"]], ["A", "Z"])

    # Z is never flagged stale — that is the whole point of this finding.
    from engine.backtest import _staleness_panel
    assert not _staleness_panel(panel)["Z"].any()

    vol = compute_signal(panel, Signal(type="volatility",
                                       lookback_days=INVERSE_VOL_LOOKBACK))
    sig_date = res.target_weights.index[0]
    v = vol.loc[sig_date, ["A", "Z"]]
    assert v["Z"] < 0.0001                   # ~0.001% annualized
    assert v["A"] > INVERSE_VOL_FLOOR_ANNUAL      # the live name is untouched
    unfloored = (1.0 / v) / (1.0 / v).sum()
    assert unfloored["Z"] > 0.999            # the bug, quantified

    # Hand arithmetic: Z is sized as if its vol were exactly the floor.
    inv_z = 1.0 / INVERSE_VOL_FLOOR_ANNUAL
    inv_a = 1.0 / float(v["A"])
    w = res.target_weights.loc[sig_date]
    assert w["Z"] == pytest.approx(inv_z / (inv_z + inv_a), abs=1e-12)
    assert w["A"] == pytest.approx(inv_a / (inv_z + inv_a), abs=1e-12)
    assert w["Z"] < unfloored["Z"] - 0.02, "the floor did not bind at all"
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)
    # And the floor is disclosed rather than applied silently — with the
    # measured number, not a value rounded away to 0.00%.
    note = next(n_ for n_ in res.data_notes if "floored volatility" in n_)
    assert "Z (" in note and "(0.00%)" not in note

    # INVARIANCE: every name below the floor is sized AS IF it had exactly the
    # floor's volatility, so two sub-floor feeds that differ from each other
    # must get identical weights.
    panel3 = _near_frozen_panel(0.03)             # three cents a day
    res3 = run(pair, panel3[["A", "Z"]], ["A", "Z"])
    v3 = compute_signal(panel3, Signal(type="volatility",
                                       lookback_days=INVERSE_VOL_LOOKBACK))
    assert v3.loc[sig_date, "Z"] < INVERSE_VOL_FLOOR_ANNUAL
    assert v3.loc[sig_date, "Z"] != pytest.approx(v["Z"], rel=1e-6)  # genuinely different
    pd.testing.assert_series_equal(
        res3.target_weights.loc[sig_date], w, check_names=False,
        obj="sub-floor feeds must be sized identically",
    )

    # On a three-name book the CAP, not the floor, is what holds the line.
    trio = _spec(
        ["A", "B", "Z"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=21),
                            direction="top", n=3),
        weighting="inverse_vol",
        rebalance="monthly",
        window=Window(start=str(idx[100].date()), end=str(idx[-1].date())),
    )
    w3 = run(trio, panel, ["A", "B", "Z"]).target_weights.loc[sig_date]
    assert w3["Z"] == pytest.approx(max(0.25, 2.0 / 3.0), abs=1e-12)


def test_the_volatility_floor_does_not_bind_on_a_real_bond_fund(sp500_panel):
    """The calibration end that the 5% floor this shipped with got wrong.

    AGG is this project's own 60/40 bond leg (fixtures/extras_prices.csv). Its
    63-day annualized volatility runs a median 3.6% with a MINIMUM of 1.96%, so
    a 5% floor bound on 72.7% of its days: it pushed AGG's mean target weight in
    a SPY/AGG risk-parity book to 0.7038 against the textbook 1/σ 0.7582 (max
    error 18.5 percentage points) and emitted a 'check the feed' data note about
    a perfectly healthy bond ETF — a false data finding, which the house rules
    rank as worse than no finding.

    k = 2 here, so the position cap (max(25%, 2/2) = 1.0) is inert and the
    engine's weights must equal textbook 1/σ EXACTLY, not approximately.
    """
    from engine.signals import INVERSE_VOL_LOOKBACK, compute_signal

    extras = pd.read_csv(
        Path(__file__).resolve().parents[2] / "fixtures" / "extras_prices.csv",
        index_col=0, parse_dates=True,
    )
    panel = (sp500_panel[["SPY"]]
             .join(extras[["AGG"]], how="inner").dropna().astype(float))
    vol = compute_signal(panel, Signal(type="volatility",
                                       lookback_days=INVERSE_VOL_LOOKBACK))
    agg_vol = vol["AGG"].dropna()
    # Measured margin: the quietest genuine asset in the fixtures sits a factor
    # of ~4 above the floor. If this ever fails the floor is mis-calibrated
    # again, not the bond fund broken.
    assert agg_vol.min() > 3.0 * INVERSE_VOL_FLOOR_ANNUAL, (
        f"AGG's minimum 63-day vol is {agg_vol.min():.2%}, too close to the "
        f"{INVERSE_VOL_FLOOR_ANNUAL:.2%} floor"
    )

    spec = _spec(
        ["SPY", "AGG"],
        selection=Selection(rank_by=Signal(type="volatility", lookback_days=63),
                            direction="bottom", n=2),
        weighting="inverse_vol",
        rebalance="monthly",
        window=Window(start="2017-01-03", end="2025-12-31"),
    )
    res = run(spec, panel, ["SPY", "AGG"])
    w = res.target_weights[["SPY", "AGG"]]

    inv = 1.0 / vol.loc[w.index, ["SPY", "AGG"]]
    textbook = inv.div(inv.sum(axis=1), axis=0)
    assert float((w - textbook).abs().to_numpy().max()) == 0.0, (
        "the volatility floor is distorting an ordinary bond/equity risk-parity "
        "book away from 1/σ"
    )
    assert float(w["AGG"].mean()) == pytest.approx(0.7582, abs=5e-5)
    assert not any("floored volatility" in n_ for n_ in res.data_notes), (
        "a healthy bond ETF was accused of a broken feed"
    )


def test_the_volatility_floor_does_not_touch_ordinary_names():
    """A floor that binds on normal equities would be its own distortion: two
    names at 15% and 30% vol must still be weighted 2:1."""
    from engine.backtest import _position_weights

    picks = pd.Series({"A": 1.0, "B": 2.0})
    vol = pd.Series({"A": 0.15, "B": 0.30})
    assert vol.min() > INVERSE_VOL_FLOOR_ANNUAL
    w = _position_weights(picks, "inverse_vol", 2, vol)
    # 1/0.15 : 1/0.30 = 2 : 1 → 2/3 and 1/3 exactly.
    assert w["A"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert w["B"] == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_a_sub_floor_volatility_is_sized_as_if_it_were_the_floor():
    """The floor's exact arithmetic on a MULTI-name book, hand-derived and with
    the book chosen so the cap cannot mask it.

    The floor is 0.5%, so for a floored name to stay under the 25% cap on a
    12-name book the other names have to be calm too — hence 1.5% vol each,
    a very short-duration bond book:
        1/σ  →  Z: 1/0.005 = 200 ;  each other: 1/0.015 = 66.67 (11 × = 733.3)
        total 933.3  →  w_Z = 200/933.3 = 3/14, each other = 1/14.
    The cap for k = 12 is max(25%, 2/12) = 25%, and 3/14 = 21.4% is below it, so
    the numbers below are the FLOOR's doing and nothing else. Unfloored, Z's
    1/0.001 = 1000 would give it 1000/1733 = 58% of the book, which the cap
    would then clip to exactly 25% — a different number either way.
    """
    from engine.backtest import _position_weights

    k = 12
    picks = pd.Series({f"T{i}": float(k - i) for i in range(k)})
    vol = pd.Series({f"T{i}": (0.001 if i == 0 else 0.015) for i in range(k)})
    assert vol["T0"] < INVERSE_VOL_FLOOR_ANNUAL < vol["T1"]
    w = _position_weights(picks, "inverse_vol", k, vol)

    assert w["T0"] == pytest.approx(3.0 / 14.0, abs=1e-12)
    for i in range(1, k):
        assert w[f"T{i}"] == pytest.approx(1.0 / 14.0, abs=1e-12), i
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)
    assert w["T0"] < max(0.25, 2.0 / k)      # the cap is not what produced this


def test_the_position_cap_is_feasible_for_every_book_size():
    """max(25%, 2/k) always leaves room: a flat 25% cap is unsatisfiable for
    k ≤ 4 and would silently collapse inverse-vol into equal weighting."""
    from engine.backtest import _position_weights

    for k in (1, 2, 3, 4, 8, 12):
        picks = pd.Series({f"T{i}": float(i) for i in range(k)})
        vol = pd.Series({f"T{i}": 0.05 if i == 0 else 0.40 for i in range(k)})
        w = _position_weights(picks, "inverse_vol", k, vol)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-12), k
        assert w.max() <= max(0.25, 2.0 / k) + 1e-12, k
        assert (w > 0).all(), k
        if k >= 8:
            # the calm name is capped at 25%, not at its raw 1/σ share
            assert w.iloc[0] == pytest.approx(0.25, abs=1e-12), k


# ------------------------------------ (10) non-finite data is refused, once

def _one_name_buy_and_hold(closes: list[float]) -> tuple[StrategySpec, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    panel = pd.DataFrame({"A": closes, "B": np.linspace(10.0, 12.0, len(closes))},
                         index=idx)
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start=str(idx[0].date()), end=str(idx[-1].date())),
    )
    return spec, panel


def test_a_zero_close_is_refused_not_turned_into_an_infinite_return():
    """`gross.isna()` let ±inf straight through: metrics then reported
    total_return NaN, sharpe NaN and max_drawdown −100% — a confident wrong
    number for a drawdown that never happened."""
    spec, panel = _one_name_buy_and_hold(
        [100.0, 101.0, 0.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    with pytest.raises(EngineError, match="non-positive or infinite"):
        run(spec, panel, ["A"], execution_lag_days=0)


def test_the_cohort_path_refuses_the_same_data_the_standard_path_refuses():
    """One engine, one answer. The cohort path used to substitute a 1.0 return
    for a zero previous close while the standard path emitted inf for the same
    panel, so identical bad data gave two different results."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    a = np.linspace(100.0, 130.0, 30)
    a[15] = 0.0
    panel = pd.DataFrame({"A": a, "B": np.linspace(10.0, 11.0, 30)}, index=idx)
    spec = _spec(
        ["A", "B"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=1),
        entry_exit=EntryExit(hold_days=3),
        rebalance="daily",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    with pytest.raises(EngineError, match="non-positive or infinite"):
        run(spec, panel, ["A", "B"])

    standard = _spec(
        ["A", "B"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=1),
        rebalance="daily",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    with pytest.raises(EngineError, match="non-positive or infinite"):
        run(standard, panel, ["A", "B"])


def test_a_negative_close_is_refused_too():
    spec, panel = _one_name_buy_and_hold(
        [100.0, 101.0, -5.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    with pytest.raises(EngineError, match="non-positive or infinite"):
        run(spec, panel, ["A"], execution_lag_days=0)


def test_a_non_finite_return_is_refused_even_when_every_close_is_positive():
    """The last line of defence. `_check_prices_usable` cannot catch a ratio
    that overflows between two legal positive closes, so the return guard
    itself must test finiteness, not just isna() — that is the exact hole inf
    walked through on its way to a −100% max drawdown."""
    idx = pd.bdate_range("2024-01-01", periods=8)
    closes = np.array([1.0, 5e-324, 1e300, 1e300, 1e300, 1e300, 1e300, 1e300])
    assert (closes > 0).all()               # nothing here is a zero or a NaN
    with np.errstate(over="ignore"):
        assert not np.isfinite(closes[2] / closes[1])
    panel = pd.DataFrame({"A": closes}, index=idx)
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start=str(idx[0].date()), end=str(idx[-1].date())),
    )
    with pytest.raises(EngineError, match="non-finite"):
        run(spec, panel, ["A"], execution_lag_days=0)


def _scope_panel(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {"A": np.linspace(100.0, 150.0, n), "B": np.linspace(50.0, 60.0, n)},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def test_the_price_check_is_scoped_to_the_data_the_run_actually_reads():
    """The refusal above is right; its blast radius was not. The check scanned
    the WHOLE fetched frame — ~500 columns across the full fetched history —
    so one vendor zero dated after the window ended killed a run whose own
    window was spotless. Every panel in this engine is backward-looking, so a
    close after the last window day cannot reach a single reported number.

    Both ends are pinned here: the post-window zero must NOT refuse and must
    leave the numbers bit-identical to the clean panel, and a zero inside the
    window must still refuse.
    """
    panel = _scope_panel()
    idx = panel.index
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="monthly",
        window=Window(start=str(idx[10].date()), end=str(idx[60].date())),
    )
    clean = run(spec, panel, ["A"])

    after = panel.copy()
    after.loc[idx[110], "A"] = 0.0           # ~50 sessions past the window end
    ran = run(spec, after, ["A"])
    pd.testing.assert_series_equal(ran.gross_returns, clean.gross_returns)
    assert float((1 + ran.gross_returns).prod() - 1) == pytest.approx(
        float((1 + clean.gross_returns).prod() - 1), abs=0.0)

    inside = panel.copy()
    inside.loc[idx[30], "A"] = 0.0           # squarely inside the window
    with pytest.raises(EngineError, match="non-positive or infinite"):
        run(spec, inside, ["A"])


def test_the_price_check_still_covers_the_lead_in_the_signals_read():
    """The lead-in is not 'outside the window' in any sense that matters: a zero
    dated before window.start is inside the lookback the signals consume, where
    it becomes an infinite return in a ranking. Narrowing the check to the
    window rows alone would let that through."""
    panel = _scope_panel()
    idx = panel.index
    panel.loc[idx[5], "A"] = 0.0            # lead-in, 5 sessions before start
    spec = _spec(
        ["A", "B"],
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=1),
        rebalance="monthly",
        window=Window(start=str(idx[10].date()), end=str(idx[60].date())),
    )
    with pytest.raises(EngineError, match="non-positive or infinite"):
        run(spec, panel, ["A", "B"])


def test_a_missing_close_is_still_a_gap_not_a_refusal():
    """NaN and zero are different things: a hole is carried forward and
    disclosed, a zero is bad data and stops the run."""
    spec, panel = _one_name_buy_and_hold(
        [100.0, 101.0, np.nan, 103.0, 104.0, 105.0, 106.0, 107.0])
    res = run(spec, panel, ["A"], execution_lag_days=0)
    assert np.isfinite(res.gross_returns.to_numpy()).all()
    assert any("price missing" in note for note in res.data_notes)


# ------------------------------ (11) run() arguments are validated, not cast

@pytest.mark.parametrize("bad", [True, False, 1.9, -0.5, "1", None.__class__])
def test_a_non_integral_execution_lag_is_refused(two_asset_panel, bad):
    """int(True) is 1 and int(1.9) is 1. A fill convention is not something to
    round off silently."""
    with pytest.raises(EngineError):
        run(_static_5050(), two_asset_panel, ["A", "B"], execution_lag_days=bad)


def test_an_integral_float_lag_is_accepted(two_asset_panel):
    res = run(_static_5050(), two_asset_panel, ["A", "B"], execution_lag_days=2.0)
    assert res.execution_lag_days == 2


def test_the_engine_takes_its_lag_rule_from_conventions_not_its_own_copy():
    """What counts as a legal execution lag has to be ONE rule. The engine's
    hardening lived only in the engine, so validation/splits.py — which sizes
    the walk-forward purge off the same lag — was free to accept a value the
    engine refuses. conventions.py owns the rule; the engine delegates and only
    re-types the error.
    """
    from dsl.conventions import coerce_execution_lag_days
    from engine.backtest import _coerce_lag

    for good in (None, 0, 1, 5, 2.0, np.int64(3)):
        assert _coerce_lag(good) == coerce_execution_lag_days(good)
    assert coerce_execution_lag_days(None) == DEFAULT_EXECUTION_LAG_DAYS

    for bad in (True, False, 1.9, -0.5, -1, "1", None.__class__, np.True_):
        with pytest.raises(ValueError):       # the shared rule refuses it…
            coerce_execution_lag_days(bad)
        with pytest.raises(EngineError):      # …and the engine's is the same rule
            _coerce_lag(bad)


def test_a_broken_rf_series_raises_an_engine_error_not_a_pandas_error(two_asset_panel):
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03"])
    dup = pd.Series([0.0001, 0.0002, 0.0003], index=idx)
    with pytest.raises(EngineError, match="duplicate dates"):
        run(_static_5050(), two_asset_panel, ["A", "B"], rf_daily=dup)

    with pytest.raises(EngineError, match="finite"):
        run(_static_5050(), two_asset_panel, ["A", "B"], rf_daily=float("inf"))


def test_an_rf_series_that_misses_the_window_is_disclosed_not_absorbed():
    """An all-NaN / out-of-window rf silently became 0.0. A data problem is a
    finding, never a silent substitution."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    panel = pd.DataFrame(
        {"A": np.linspace(100.0, 130.0, 30), "B": np.linspace(20.0, 21.0, 30)},
        index=idx,
    )
    spec = _spec(
        ["A"],
        selection=Selection(static_allocation=[StaticPosition(ticker="A", weight=1.0)]),
        rebalance="annual",
        window=Window(start=str(idx[10].date()), end=str(idx[-1].date())),
    )
    # A series that only begins AFTER the window: nothing to carry forward, so
    # every accrual day would silently have fallen back to 0%.
    later = pd.Series(
        [RF_DAILY] * 5, index=pd.bdate_range("2029-01-01", periods=5))
    res = run(spec, panel, ["A"], rf_daily=later)
    assert res.rf_used == 0.0
    assert any("covers none of the simulated window" in n for n in res.data_notes)

    # Partial coverage is disclosed too: here the rate starts mid-window, so
    # the days before it earn 0% and the run says so.
    start = panel.index[15]
    half = pd.Series(RF_DAILY, index=idx[idx >= start])
    res2 = run(spec, panel, ["A"], rf_daily=half)
    assert 0.0 < res2.rf_used < RF_DAILY
    assert any("fewer day(s) than the simulated window" in n
               for n in res2.data_notes)
