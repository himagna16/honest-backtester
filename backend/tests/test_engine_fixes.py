"""Regression tests for the confirmed adversarial-audit findings (M1
verification workflow). Each test pins a bug that would have produced wrong
or silently-absorbed numbers."""
import numpy as np
import pandas as pd
import pytest

from dsl.schema import Costs, Selection, Signal, StrategySpec, Universe, Window
from dsl.validator import validate_spec
from engine import metrics
from engine.backtest import run
from engine.signals import compute_signal


def make_spec(**kw) -> StrategySpec:
    base = dict(
        name="fix-regression",
        universe=Universe(preset="user_watchlist"),
        selection=Selection(rank_by=Signal(type="return", lookback_days=21),
                            direction="top", n=2),
        weighting="equal", rebalance="monthly",
        costs=Costs(sweep_bps=[0, 5, 20]), benchmark="SPY",
        window=Window(start="2020-03-02", end="2020-12-31"),
    )
    base.update(kw)
    return StrategySpec(**base)


def walk_panel(seed: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range("2019-06-03", "2020-12-31")
    rng = np.random.default_rng(seed)
    data = {
        tk: 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(idx)))
        for tk in ("A", "B", "SPY")
    }
    return pd.DataFrame(data, index=idx)


# ---- CRITICAL: zero-vol ticker must not zero the portfolio ----

def test_zero_vol_ticker_excluded_and_disclosed():
    panel = walk_panel()
    panel["FLAT"] = 42.0  # stale feed: constant closes
    spec = make_spec(weighting="inverse_vol")
    res = run(spec, panel, ["A", "B", "FLAT"])
    # portfolio actually invests (the old bug: all-cash, all-zero returns)
    assert res.gross_returns.abs().sum() > 0
    assert res.turnover.sum() > 0
    # FLAT never held; exclusion is disclosed, not silent
    held = {p["ticker"] for h in res.holdings for p in h["picks"]}
    assert "FLAT" not in held
    assert any("zero measured volatility" in n for n in res.data_notes)
    # weights are finite everywhere
    assert np.isfinite(res.target_weights.to_numpy()).all()


# ---- CRITICAL: signals must not bridge data holes ----

def test_return_signal_requires_gap_free_lookback():
    panel = walk_panel()
    holed = panel.copy()
    holed.loc["2020-05-01":"2020-08-31", "B"] = np.nan  # 4-month halt
    sig = Signal(type="return", lookback_days=126)
    values = compute_signal(holed[["A", "B"]], sig)
    # B's signal right after relisting must be NaN (no clean 126d window yet)
    assert np.isnan(values.loc["2020-09-30", "B"])
    assert not np.isnan(values.loc["2020-09-30", "A"])
    # …and stays NaN until a full clean lookback exists again
    assert np.isnan(values.loc["2020-12-01", "B"])

    rsi_values = compute_signal(holed[["A", "B"]], Signal(type="rsi", lookback_days=14))
    assert np.isnan(rsi_values.loc["2020-09-04", "B"])


def test_engine_never_buys_across_hole():
    panel = walk_panel()
    panel.loc["2020-05-01":"2020-08-31", "B"] = np.nan
    panel.loc["2020-09-01":, "B"] *= 3.0  # relists gapped up: fake momentum
    spec = make_spec(selection=Selection(
        rank_by=Signal(type="return", lookback_days=63), direction="top", n=1))
    res = run(spec, panel, ["A", "B"])
    sep_oct = [h for h in res.holdings if h["date"] >= "2020-09-01"
               and h["date"] <= "2020-10-31"]
    held = {p["ticker"] for h in sep_oct for p in h["picks"]}
    assert "B" not in held  # the hole-bridging pick is impossible now


# ---- MAJOR: drawdown measured from initial capital ----

def test_drawdown_includes_initial_capital():
    idx = pd.bdate_range("2024-01-02", periods=4)
    r = pd.Series([-0.30, 0.10, 0.10, 0.10], index=idx)
    assert metrics.max_drawdown(r) == pytest.approx(-0.30, abs=1e-12)
    dd = metrics.drawdown_series(r)
    assert dd.iloc[0] == pytest.approx(-0.30, abs=1e-12)


# ---- MAJOR: metrics refuse NaN instead of silently skipping ----

def test_metrics_refuse_nan_series():
    idx = pd.bdate_range("2024-01-02", periods=5)
    r = pd.Series([0.01, np.nan, 0.02, 0.01, 0.0], index=idx)
    for fn in (metrics.total_return, metrics.cagr, metrics.annual_vol,
               metrics.max_drawdown):
        with pytest.raises(ValueError, match="NaN"):
            fn(r)


# ---- MAJOR: spurious non-SPY-session rows are dropped from the calendar ----

def test_panel_calendar_is_spy_sessions(tmp_path):
    from engine.data import PriceCache, get_panel

    cache = PriceCache(tmp_path)
    idx = pd.bdate_range("2024-01-01", "2024-06-28")
    rng = np.random.default_rng(1)
    spy = pd.Series(400 * np.cumprod(1 + rng.normal(0, 0.008, len(idx))), index=idx)
    g = pd.Series(50.0, index=idx.append(pd.DatetimeIndex([pd.Timestamp("2024-05-25")])))
    g = g.sort_index()  # 2024-05-25 is a Saturday — spurious row
    cache.put_series("SPY", spy, source="fixture")
    cache.put_series("G", g, source="fixture")

    panel = get_panel(["G", "SPY"], start="2024-01-01", end="2024-06-28", cache=cache)
    assert pd.Timestamp("2024-05-25") not in panel.prices.index
    assert any("SPY's trading days" in n for n in panel.notes)


# ---- MINOR: inapplicable signal fields are rejected, not ignored ----

@pytest.mark.parametrize("signal,needle", [
    ({"type": "volatility", "lookback_days": 63, "skip_days": 21}, "skip_days"),
    ({"type": "return", "lookback_days": 126, "fast": 10, "slow": 50}, "sma_cross"),
    ({"type": "sma_cross", "fast": 50, "slow": 200, "lookback_days": 100},
     "lookback_days"),
])
def test_validator_rejects_inapplicable_signal_fields(signal, needle):
    res = validate_spec({
        "name": "x", "universe": {"preset": "sp500"},
        "selection": {"rank_by": signal, "direction": "top", "n": 5},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
    })
    assert not res.valid
    assert any(needle in e for e in res.errors)
