"""M4 acceptance: the audit pipeline on the real frozen fixtures.

- The rigged-universe (War list) fixture must trigger the selection-bias
  audit and can never earn green.
- The dip-buying reproduction: beats SPY gross, dies at 20 bps.
- The FATAL >2× branch, regime/capacity/benchmark-fit checks unit-tested.
"""
import numpy as np
import pandas as pd
import pytest

from audit import extra, regimes
from audit.pipeline import build_report
from audit.universe import check_selection_bias
from dsl.validator import validate_spec
from engine.benchmark import BENCHMARK_TICKERS
from engine.data import get_panel

WAR_TICKERS = None


def war_tickers():
    global WAR_TICKERS
    if WAR_TICKERS is None:
        from config import settings
        path = settings.fixtures_dir / "war_list_tickers.txt"
        WAR_TICKERS = [t.strip() for t in path.read_text().splitlines() if t.strip()]
    return WAR_TICKERS


def build(spec_raw: dict, attempts: int = 1) -> dict:
    res = validate_spec(spec_raw)
    assert res.valid, res.errors
    tickers = res.resolved_universe + BENCHMARK_TICKERS[res.spec.benchmark] + ["^IRX"]
    panel = get_panel(tickers, start=res.spec.window.start, end=res.spec.window.end)
    universe = [t for t in res.resolved_universe if t in panel.prices.columns]
    return build_report(res.spec, panel, universe, attempts=attempts)


@pytest.fixture(scope="module")
def rigged_report() -> dict:
    return build({
        "name": "War list 12mo momentum top-5",
        "universe": {"preset": "user_watchlist", "resolved_tickers": war_tickers()},
        "selection": {"rank_by": {"type": "return", "lookback_days": 252},
                      "direction": "top", "n": 5},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
        "window": {"start": "2023-01-02", "end": "2026-05-29"},
    }, attempts=1)


def test_rigged_universe_triggers_selection_bias(rigged_report):
    r = rigged_report
    # The fake huge number is on screen…
    assert r["metrics"]["total_return"] > 5.0  # > +500%, the trap reproduced
    # …but the verdict can never be green on a hand-picked ranked universe
    assert r["verdict"]["color"] in ("red", "amber")
    bias = [f for f in r["audit"] if f["check"] == "selection_bias"]
    assert bias and bias[0]["severity"] in ("fatal", "warning")
    assert "hand-picked" in bias[0]["message"].lower() or "hindsight" in bias[0]["message"].lower()
    # …with a working neutral re-run action
    assert "S&P 500" in (bias[0]["action"] or "")
    assert r["neutral_rerun"] is not None
    assert r["neutral_rerun"]["universe"] == "sp500"


def test_report_shape_and_invariants(rigged_report):
    r = rigged_report
    # benchmark on every result; disclaimer on every report (non-negotiables)
    assert r["benchmark"]["total_return"] is not None
    assert "not investment advice" in r["disclaimer"].lower()
    # cost sweep has 0/5/20 + a benchmark row
    bps = [row["bps"] for row in r["cost_sweep"]]
    assert {0.0, 5.0, 20.0}.issubset({b for b in bps if b is not None})
    assert any(row["bps"] is None for row in r["cost_sweep"])
    # audit cards ordered by severity (fatal < warning < note < pass)
    order = {"fatal": 0, "warning": 1, "note": 2, "pass": 3}
    sevs = [order[f["severity"]] for f in r["audit"]]
    assert sevs == sorted(sevs)
    # lookahead impossibility statement always present
    assert any(f["check"] == "lookahead" for f in r["audit"])
    # verdict has icon + text (never color alone — Gap #12)
    assert r["verdict"]["icon"] and r["verdict"]["headline"]
    # equity curve rows: [date, net5, gross, benchmark]
    assert len(r["equity_curve"]) > 200 and len(r["equity_curve"][0]) == 4
    assert r["oos"]["out_of_sample"]["days"] > 0


def test_dip_buy_cost_collapse_reproduction():
    """The session's real result: War-list dip buying beats SPY gross but
    flips to losing at 20 bps (Definition of Done)."""
    r = build({
        "name": "Buy the losers, hold 1 day",
        "universe": {"preset": "user_watchlist", "resolved_tickers": war_tickers()},
        "filters": [{"signal": {"type": "daily_return", "lookback_days": 1},
                     "op": "lt", "value": 0}],
        "selection": {"rank_by": {"type": "daily_return", "lookback_days": 1},
                      "direction": "bottom", "n": 44},
        "weighting": "equal", "rebalance": "daily",
        "entry_exit": {"hold_days": 1},
        "benchmark": "SPY",
        "window": {"start": "2022-01-03", "end": "2026-05-29"},
    })
    sweep = {row["bps"]: row for row in r["cost_sweep"] if row["bps"] is not None}
    assert sweep[0.0]["beats_benchmark"] is True      # gross: beats SPY
    assert sweep[20.0]["beats_benchmark"] is False    # 20 bps: loses
    cost_f = [f for f in r["audit"] if f["check"] == "cost_sweep"][0]
    assert cost_f["severity"] == "warning"
    assert r["metrics"]["annual_turnover"] > 20  # churn is the killer
    assert r["verdict"]["color"] in ("red", "amber")


def test_honest_sp500_momentum_no_fatal():
    r = build({
        "name": "12-1 momentum top 25",
        "universe": {"preset": "sp500"},
        "selection": {"rank_by": {"type": "return", "lookback_days": 252,
                                  "skip_days": 21}, "direction": "top", "n": 25},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
        "window": {"start": "2016-01-04", "end": "2020-12-31"},
    })
    assert not any(f["severity"] == "fatal" for f in r["audit"])
    assert r["neutral_rerun"] is None
    # current-members caveat is disclosed on presets
    assert any(f["check"] == "survivorship_universe" for f in r["audit"])
    # window includes COVID → bear-market coverage
    assert r["regimes"]["has_bear_market"] is True


def test_fatal_branch_when_user_universe_outperforms_2x(monkeypatch):
    """Unit-test the >2× FATAL branch with a controlled neutral re-run."""
    import audit.universe as um

    idx = pd.bdate_range("2023-01-02", periods=300)
    big = pd.Series(0.01, index=idx)      # user run: huge
    small = pd.Series(0.0002, index=idx)  # neutral: modest

    class FakeResult:
        net_returns = {5.0: small}

    from types import SimpleNamespace

    monkeypatch.setattr(um, "preset_tickers", lambda p: ["AAA", "BBB"])
    monkeypatch.setattr(
        um, "get_panel",
        lambda *a, **k: SimpleNamespace(prices=pd.DataFrame(index=idx)),
    )
    monkeypatch.setattr(um, "run", lambda *a, **k: FakeResult())

    res = validate_spec({
        "name": "x", "universe": {"preset": "user_watchlist",
                                  "resolved_tickers": ["AAPL", "MSFT", "NVDA"]},
        "selection": {"rank_by": {"type": "return", "lookback_days": 21},
                      "direction": "top", "n": 2},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
    })
    finding, rerun = check_selection_bias(res.spec, big)
    assert finding.severity == "fatal"
    assert rerun["wealth_ratio"] > 2.0
    assert "+1650%" in finding.message or "hindsight" in finding.message


def test_regime_warning_without_bear():
    idx = pd.bdate_range("2016-06-01", "2017-06-01")
    calm_bull = pd.Series(0.0006, index=idx)
    finding, segments, has_bear = regimes.check_regimes(calm_bull, "SPY")
    assert has_bear is False
    assert finding.severity == "warning"
    assert "no sustained bear market" in finding.message.lower() or "bear" in finding.message.lower()
    assert segments


def test_capacity_flags_sub_share_positions():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"BRKA": 620000.0, "AAPL": 180.0}, index=idx)
    holdings = [{"date": "2024-01-08", "eligible": 2,
                 "picks": [{"ticker": "BRKA", "weight": 0.5, "score": 1.0},
                           {"ticker": "AAPL", "weight": 0.5, "score": 0.9}]}]
    finding = extra.check_capacity(holdings, prices)
    assert finding.severity == "note"
    assert "BRKA" in finding.message


def test_benchmark_fit_flags_low_beta():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    strat = 0.3 * bench  # realized beta 0.3
    finding = extra.check_benchmark_fit(strat, bench, "SPY", "SPY buy-and-hold")
    assert finding.severity == "warning"
    assert "60/40" in (finding.action or "")


def test_index_preset_extreme_outperformance_capped_at_amber():
    """A ranked strategy harvesting today's index membership for >2x wealth
    outperformance can never be called a survivor (green)."""
    r = build({
        "name": "sp500 momentum melt-up",
        "universe": {"preset": "sp500"},
        "selection": {"rank_by": {"type": "return", "lookback_days": 252},
                      "direction": "top", "n": 5},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
        "window": {"start": "2023-01-02", "end": "2026-05-29"},
    }, attempts=3)
    assert r["verdict"]["color"] in ("red", "amber")
    if r["verdict"]["color"] == "amber":
        assert "index membership" in r["verdict"]["headline"] or "hand-picked" in r["verdict"]["headline"]
