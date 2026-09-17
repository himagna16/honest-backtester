"""Point-in-time index membership (dsl/membership.py) and the engine's use of it.

The bug this pins: a ranked strategy on TODAY's S&P 500 list could buy a
company years before S&P added it — exactly the companies that later grew
into the index. With membership applied, a name is only eligible on signal
dates inside one of its membership spells.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dsl.membership import PointInTimeMembership, _load, load_membership
from dsl.schema import Costs, Selection, Signal, StaticPosition, StrategySpec, Universe, Window
from engine.backtest import run


def make_membership(rows, coverage_start="2019-01-01", through="2021-06-30"):
    spells = pd.DataFrame(rows, columns=["label", "start", "end", "price_symbol"])
    meta = {"coverage_start": coverage_start, "changes_through": through}
    return PointInTimeMembership("test", spells, meta)


def spec(**kw) -> StrategySpec:
    base = dict(
        name="pit",
        universe=Universe(preset="sp500"),
        selection=Selection(rank_by=Signal(type="return", lookback_days=21),
                            direction="top", n=1),
        weighting="equal", rebalance="monthly",
        costs=Costs(sweep_bps=[0, 5, 20]), benchmark="SPY",
        window=Window(start="2020-01-02", end="2020-12-31"),
    )
    base.update(kw)
    return StrategySpec(**base)


def trend_panel() -> pd.DataFrame:
    """LATE is the runaway winner all year; GONE is second best; A and B drift.
    Deterministic trends, so the top-1 momentum pick is known in advance."""
    idx = pd.bdate_range("2019-09-02", "2020-12-31")
    days = np.arange(len(idx))
    return pd.DataFrame({
        "LATE": 50 * 1.004 ** days,
        "GONE": 50 * 1.002 ** days,
        "A": 50 * 1.0005 ** days,
        "B": 50 * 1.0002 ** days,
        "SPY": 100 * 1.0003 ** days,
    }, index=idx)


def picks_by_date(res) -> dict[str, list[str]]:
    return {h["date"]: [p["ticker"] for p in h["picks"]] for h in res.holdings}


# ---------------------------------------------------------------- the table

def test_spells_are_start_inclusive_end_exclusive_and_union_across_spells():
    m = make_membership([
        ["X", None, "2020-03-02", "X"],
        ["X", "2020-06-01", None, "X"],       # removed, then re-added
        ["Y", "2020-03-02", None, "Y"],
    ])
    idx = pd.DatetimeIndex(["2020-02-28", "2020-03-02", "2020-05-29", "2020-06-01"])
    e = m.eligibility(idx, ["X", "Y"])
    assert e["X"].tolist() == [True, False, False, True]
    assert e["Y"].tolist() == [False, True, True, True]


def test_coverage_start_is_a_hard_line():
    m = make_membership([["X", None, None, "X"]], coverage_start="2015-01-01")
    assert not m.covers("2014-12-31")
    assert m.covers("2015-01-01")
    assert not m.covers(None)


def test_members_without_usable_prices_are_counted_but_not_fetched():
    m = make_membership([
        ["LIVE", None, None, "LIVE"],
        ["DEAD", None, "2020-06-01", ""],          # acquired; no free history
        ["LATE", "2020-09-01", None, "LATE"],
    ])
    assert m.symbols_between("2020-01-01", "2020-12-31") == ["LATE", "LIVE"]
    assert m.symbols_between("2020-01-01", "2020-05-01") == ["LIVE"]

    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    prices = pd.DataFrame({"LIVE": 1.0, "LATE": 1.0}, index=idx)
    prices.loc[:"2020-03-31", "LIVE"] = np.nan          # a priced member with a hole
    cov = m.coverage(pd.DatetimeIndex(["2020-02-03", "2020-05-01", "2020-07-01", "2020-10-01"]), prices)
    # Feb: LIVE (no close) + DEAD → 0/2; May: LIVE + DEAD → 1/2; Jul: LIVE → 1/1;
    # Oct: LIVE + LATE → 2/2
    assert cov.dates == 4
    assert cov.min_priced_share == 0.0 and cov.min_date == "2020-02-03"
    assert cov.mean_priced_share == pytest.approx((0 + 0.5 + 1 + 1) / 4)


def test_a_hand_edited_membership_file_is_refused(tmp_path):
    csv = tmp_path / "membership.csv"
    csv.write_text("label,security,start,end,price_symbol,price_check,price_note\nX,X,,,X,ok,\n")
    meta = {"coverage_start": "2015-01-01", "changes_through": "2026-01-01",
            "membership_sha256": hashlib.sha256(csv.read_bytes()).hexdigest()}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert _load(tmp_path, "t") is not None
    csv.write_text(csv.read_text() + "Y,Y,,,Y,ok,\n")
    with pytest.raises(RuntimeError, match="does not match the hash"):
        _load(tmp_path, "t")


# ---------------------------------------------------------------- the engine

def test_a_name_cannot_be_bought_before_it_joins_the_index():
    panel = trend_panel()
    m = make_membership([
        ["LATE", "2020-07-01", None, "LATE"],
        ["GONE", None, None, "GONE"],
        ["A", None, None, "A"],
        ["B", None, None, "B"],
    ])
    universe = ["LATE", "GONE", "A", "B"]

    today = run(spec(), panel, universe)
    assert not today.membership_applied
    assert all(p == ["LATE"] for p in picks_by_date(today).values())  # the survivor trap

    pit = run(spec(), panel, universe, membership=m.eligibility(panel.index, universe))
    assert pit.membership_applied
    for day, picks in picks_by_date(pit).items():
        assert picks == (["LATE"] if day >= "2020-07-01" else ["GONE"]), day
    assert pit.eligible_counts.loc["2020-01-31"] == 3
    assert pit.eligible_counts.loc["2020-07-31"] == 4
    # the flattering version really was flattered
    assert today.gross_returns.add(1).prod() > pit.gross_returns.add(1).prod()


def test_a_removed_and_delisted_name_stops_being_eligible():
    """GONE leaves the index on 2020-06-15 (acquired) and its quote stops a
    week later. It may be picked while a member, is held to the next rebalance
    with the gap disclosed, and is never picked again — even though its last
    close is still the best trailing return on the book."""
    panel = trend_panel()
    panel.loc["2020-06-22":, "GONE"] = np.nan
    m = make_membership([
        ["GONE", None, "2020-06-15", "GONE"],
        ["A", None, None, "A"],
        ["B", None, None, "B"],
    ])
    universe = ["GONE", "A", "B"]
    res = run(spec(), panel, universe, membership=m.eligibility(panel.index, universe))

    picks = picks_by_date(res)
    assert picks["2020-05-29"] == ["GONE"]
    assert all(p == ["A"] for d, p in picks.items() if d >= "2020-06-30")
    assert np.isfinite(res.gross_returns.to_numpy()).all()
    assert any(n.startswith("GONE: price missing") for n in res.data_notes)


def test_membership_gates_eligibility_even_when_prices_continue():
    """Removal is not the same as a missing quote: a demoted company keeps
    trading, and the mask alone must keep it out."""
    panel = trend_panel()
    m = make_membership([
        ["LATE", None, "2020-04-01", "LATE"],
        ["A", None, None, "A"],
        ["B", None, None, "B"],
    ])
    universe = ["LATE", "A", "B"]
    res = run(spec(), panel, universe, membership=m.eligibility(panel.index, universe))
    picks = picks_by_date(res)
    assert picks["2020-03-31"] == ["LATE"]
    assert all(p == ["A"] for d, p in picks.items() if d >= "2020-04-01")


def test_a_ticker_missing_from_the_mask_is_never_eligible():
    panel = trend_panel()
    mask = pd.DataFrame(True, index=panel.index, columns=["A", "B"])
    res = run(spec(), panel, ["LATE", "A", "B"], membership=mask)
    assert all("LATE" not in p for p in picks_by_date(res).values())


def test_static_allocations_ignore_membership():
    panel = trend_panel()
    s = spec(selection=Selection(static_allocation=[
        StaticPosition(ticker="LATE", weight=0.5), StaticPosition(ticker="A", weight=0.5)]),
        rebalance="quarterly")
    mask = pd.DataFrame(False, index=panel.index, columns=["LATE", "A"])
    base = run(s, panel, ["LATE", "A"])
    gated = run(s, panel, ["LATE", "A"], membership=mask)
    pd.testing.assert_series_equal(base.gross_returns, gated.gross_returns)
    assert not gated.membership_applied


# ---------------------------------------------------------------- the real table

@pytest.fixture(scope="module")
def sp500():
    m = load_membership("sp500")
    assert m is not None, "fixtures/sp500_membership is missing"
    return m


@pytest.mark.parametrize("ticker,joined", [
    ("PLTR", "2024-09-23"), ("APP", "2025-09-22"), ("CVNA", "2025-12-22"), ("VRT", "2026-03-23"),
])
def test_the_2023_2026_winners_join_on_their_real_dates(sp500, ticker, joined):
    before = (pd.Timestamp(joined) - pd.offsets.BDay(1))
    e = sp500.eligibility(pd.DatetimeIndex([before, pd.Timestamp(joined)]), [ticker])
    assert e[ticker].tolist() == [False, True]


def test_removed_names_are_members_until_their_removal(sp500):
    # American Airlines: removed 2024-09-23, still trading, priced from the PIT fixture
    e = sp500.eligibility(pd.DatetimeIndex(["2024-09-20", "2024-09-23"]), ["AAL"])
    assert e["AAL"].tolist() == [True, False]
    assert "AAL" in sp500.symbols_between("2016-01-01", "2020-12-31")


def test_a_failed_bank_is_a_member_with_no_usable_history(sp500):
    on = sp500.members_on("2022-06-30")
    svb = on[on["label"] == "SIVB"]
    assert len(svb) == 1 and svb["price_symbol"].iloc[0] == ""
    assert "SIVB" not in sp500.symbols_between("2022-01-01", "2022-12-31")


def test_a_reused_ticker_does_not_price_the_old_company(sp500):
    # FOXA was 21st Century Fox until 2019-03-19; Yahoo's FOXA is Fox Corp from
    # 2019-03-12, so the old spell must not borrow the new company's prices.
    old = sp500.spells[(sp500.spells["label"] == "FOXA") & sp500.spells["end"].notna()]
    assert (old["price_symbol"] == "").all()
    e = sp500.eligibility(pd.DatetimeIndex(["2019-03-18", "2019-03-19"]), ["FOXA"])
    assert e["FOXA"].tolist() == [False, True]


def test_member_counts_stay_index_sized_across_coverage(sp500):
    month_ends = pd.date_range(sp500.coverage_start, sp500.known_through, freq="ME")
    counts = [len(sp500.members_on(d)) for d in month_ends]
    assert min(counts) >= 495 and max(counts) <= 510
    spells = sp500.spells
    bounded = spells[spells["start"].notna() & spells["end"].notna()]
    assert (bounded["start"] < bounded["end"]).all()
