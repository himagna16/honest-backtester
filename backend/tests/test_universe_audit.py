"""The selection/survivorship audit against point-in-time membership.

Pins three things:
1. The neutral comparison for a hand-picked list is the S&P 500 as it stood at
   each rebalance, and when that is not available the check says so and
   refuses to clear the list (warning), or reports a >2× gap as FATAL with the
   comparison's own flattery disclosed.
2. Index presets disclose whether they ran point-in-time, and a ranked run
   without trustworthy point-in-time data cannot earn green.
3. On the real fixtures, the bias being fixed actually existed: today's-list
   momentum bought APP/CVNA/VRT/PLTR before S&P added them, point-in-time
   momentum never does.
"""
import pandas as pd
import pytest

import audit.universe as um
from audit.universe import (
    MembershipUse,
    check_selection_bias,
    index_universe_finding,
    membership_for,
    membership_verdict,
)
from dsl.membership import MembershipCoverage, load_membership
from dsl.universes import preset_tickers
from dsl.validator import validate_spec
from engine.backtest import run
from engine.benchmark import BENCHMARK_TICKERS
from engine.data import get_panel


def ranked(preset="user_watchlist", start="2023-01-02", end="2026-05-29", **universe):
    raw = {
        "name": "x", "universe": {"preset": preset, **universe},
        "selection": {"rank_by": {"type": "return", "lookback_days": 252},
                      "direction": "top", "n": 5},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
        "window": {"start": start, "end": end},
    }
    res = validate_spec(raw)
    assert res.valid, res.errors
    return res.spec


def coverage(share: float) -> MembershipCoverage:
    return MembershipCoverage(dates=40, mean_priced_share=share, min_priced_share=share - 0.02,
                              min_date="2023-01-31", mean_members=503.0,
                              mean_unpriced=503 * (1 - share))


def pit_use(share: float) -> MembershipUse:
    return MembershipUse("sp500", True, load_membership("sp500"), None, coverage(share))


def today_use(reason="dated S&P 500 membership starts 2015-01-01, and this window starts 2012-01-03"):
    return MembershipUse("sp500", True, None, reason)


WATCHLIST = ["AAPL", "MSFT", "NVDA"]
IDX = pd.bdate_range("2023-01-02", periods=300)
BIG = pd.Series(0.01, index=IDX)
SMALL = pd.Series(0.0002, index=IDX)


# ------------------------------------------------ hand-picked vs neutral

def test_a_2x_gap_is_fatal_against_the_point_in_time_index(monkeypatch):
    monkeypatch.setattr(um, "_neutral_run", lambda spec: (SMALL, pit_use(0.97)))
    finding, rerun = check_selection_bias(ranked(resolved_tickers=WATCHLIST), BIG)
    assert finding.severity == "fatal"
    assert rerun["point_in_time"] and rerun["trustworthy"]
    assert "as it stood at each rebalance" in finding.message
    assert "flattered" not in finding.message


def test_a_2x_gap_stays_fatal_and_says_the_comparison_is_flattered(monkeypatch):
    monkeypatch.setattr(um, "_neutral_run", lambda spec: (SMALL, today_use()))
    finding, rerun = check_selection_bias(ranked(resolved_tickers=WATCHLIST), BIG)
    assert finding.severity == "fatal"
    assert rerun["point_in_time"] is False and rerun["trustworthy"] is False
    assert "today's S&P 500 members" in finding.message
    assert "real gap is likely wider" in finding.message


@pytest.mark.parametrize("use", [today_use(), pit_use(0.82)], ids=["today's list", "thin PIT"])
def test_within_2x_against_an_untrustworthy_index_is_not_a_pass(monkeypatch, use):
    """The old failure: a hindsight list that is 'only' 1.8× today's S&P 500
    got the ordinary warning, as if the comparison had cleared it."""
    near = pd.Series(0.0095, index=IDX)
    monkeypatch.setattr(um, "_neutral_run", lambda spec: (near, use))
    finding, rerun = check_selection_bias(ranked(resolved_tickers=WATCHLIST), BIG)
    assert rerun["wealth_ratio"] < 2.0
    assert finding.severity == "warning"
    assert finding.name == "Hand-picked universe (neutral check unverified)"
    assert "can't clear your list" in finding.message
    assert rerun["membership"]["fallback_reason"]


def test_within_2x_against_the_point_in_time_index_is_an_ordinary_warning(monkeypatch):
    near = pd.Series(0.0095, index=IDX)
    monkeypatch.setattr(um, "_neutral_run", lambda spec: (near, pit_use(0.96)))
    finding, rerun = check_selection_bias(ranked(resolved_tickers=WATCHLIST), BIG)
    assert finding.severity == "warning" and finding.name == "Hand-picked universe"


# ------------------------------------------------ index presets

def test_sp500_before_coverage_falls_back_and_is_capped():
    spec = ranked("sp500", start="2012-01-03", end="2014-12-31")
    use = membership_for(spec)
    assert not use.point_in_time and "2015-01-01" in use.fallback_reason
    f = index_universe_finding(spec, use)
    assert f.severity == "warning" and f.name == "Index list is today's members"
    v = membership_verdict(use)
    assert v["color"] == "amber" and "today's index membership" in v["headline"]


def test_nasdaq100_has_no_dated_membership_and_is_capped():
    spec = ranked("nasdaq100", start="2016-01-04", end="2020-12-31")
    use = membership_for(spec)
    assert not use.point_in_time
    assert membership_verdict(use)["color"] == "amber"


@pytest.mark.parametrize("share,severity,capped", [
    (0.97, "note", False), (0.92, "warning", False), (0.85, "warning", True),
])
def test_point_in_time_coverage_sets_severity_and_the_cap(share, severity, capped):
    spec = ranked("sp500", start="2016-01-04", end="2020-12-31")
    use = pit_use(share)
    f = index_universe_finding(spec, use)
    assert f.severity == severity
    assert f.evidence["point_in_time"] is True
    assert (membership_verdict(use) is not None) is capped


def test_static_allocations_and_sector_etfs_are_not_capped():
    static = validate_spec({
        "name": "s", "universe": {"preset": "sp500"},
        "selection": {"static_allocation": [{"ticker": "SPY", "weight": 1.0}]},
        "weighting": "equal", "rebalance": "quarterly",
        "window": {"start": "2012-01-03", "end": "2014-12-31"},
    }).spec
    assert membership_verdict(membership_for(static)) is None
    etfs = ranked("sector_etfs", start="2016-01-04", end="2020-12-31")
    assert membership_verdict(membership_for(etfs)) is None
    assert index_universe_finding(etfs, membership_for(etfs)).severity == "note"


# ------------------------------------------------ the diagnosis, on real data

JOINED = {"PLTR": "2024-09-23", "APP": "2025-09-22", "CVNA": "2025-12-22", "VRT": "2026-03-23"}


@pytest.fixture(scope="module")
def momentum_2023_2026():
    """12-month momentum, top 5, monthly — the run that reported +3553% on
    today's S&P 500 list — on one panel, with and without membership."""
    spec = ranked("sp500", start="2023-01-02", end="2026-05-29")
    pit = load_membership("sp500")
    current = preset_tickers("sp500")
    tickers = sorted(set(pit.symbols_between("2023-01-02", "2026-05-29")) | set(current))
    panel = get_panel(tickers + BENCHMARK_TICKERS["SPY"], start="2023-01-02", end="2026-05-29")
    cols = set(panel.prices.columns)
    universe = [t for t in tickers if t in cols]
    biased = run(spec, panel.prices, [t for t in current if t in cols])
    honest = run(spec, panel.prices, universe,
                 membership=pit.eligibility(panel.prices.index, universe))
    return biased, honest


def first_pick_dates(res) -> dict[str, str]:
    out = {}
    for h in res.holdings:
        for p in h["picks"]:
            out.setdefault(p["ticker"], h["date"])
    return out


def test_todays_list_bought_the_winners_before_they_joined(momentum_2023_2026):
    biased, _ = momentum_2023_2026
    early = {t: d for t, d in first_pick_dates(biased).items() if t in JOINED and d < JOINED[t]}
    assert early, "the survivorship trap should be visible in the fixture data"


def test_point_in_time_momentum_never_does(momentum_2023_2026):
    biased, honest = momentum_2023_2026
    assert honest.membership_applied
    for h in honest.holdings:
        for p in h["picks"]:
            if p["ticker"] in JOINED:
                assert h["date"] >= JOINED[p["ticker"]], (p["ticker"], h["date"])
    assert biased.net_returns[5.0].add(1).prod() > honest.net_returns[5.0].add(1).prod()
