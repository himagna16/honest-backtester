"""Tests for validation/splits.py — the purge/embargo primitives.

TWO KINDS OF TEST LIVE HERE, AND ONLY ONE OF THEM CAN CATCH A LEAK.

`assert_no_leak` below is a STRUCTURAL check: given the purge length L the
constructor was handed, it verifies the constructor actually applied it. It
shares its constant with the implementation, so it can only ever prove that
`_build_split` did as it was told — it says nothing about whether L is big
enough, and a mutation that shrinks the carry table tenfold leaves every one
of these assertions green. Treat it as a shape test, never as evidence.

The evidence is in "engine-level leak" at the bottom of this file. Those tests
run the real engine over a real price panel, perturb the closes on a TRAINING
bar, re-run, and assert the net-return series on the TEST bars is bit-for-bit
unchanged — and, at a deliberately crafted boundary, that the last bar the
perturbation does reach is exactly a+L-1, so L is neither one bar short nor one
bar long. Nothing in them re-states the purge formula; the engine is the
independent authority. Shrink MAX_REBALANCE_CARRY_DAYS, drop the execution lag,
drop the staleness reach or drop the cost-charge bar and they fail. The
staleness term hangs on ONE parametrisation, the `staleness` case: every other
boundary uses a 21-bar ranking signal, whose reach equals the screen's and
masks it, so the engine cannot see that term there at all.

There is no `hypothesis` in requirements.txt, so the property tests sweep a
seeded pseudo-random grid of shapes instead — deterministic, and reproducible
from the seed printed in the parametrisation.
"""
import random

import numpy as np
import pandas as pd
import pytest

from dsl.conventions import (
    DEFAULT_EXECUTION_LAG_DAYS, STALENESS_LOOKBACK_DAYS,
)
from dsl.schema import (
    Costs, EntryExit, Filter, Selection, Signal, StrategySpec, Universe, Window,
)
from validation.splits import (
    CPCVDesign,
    MAX_REBALANCE_CARRY_DAYS,
    Split,
    SplitError,
    calendar_rebalance_carry,
    combinatorial_purged_cv,
    default_embargo,
    n_cpcv_paths,
    n_cpcv_splits,
    purge_components,
    purge_length,
    purged_split,
    rsi_reach,
    signal_reach,
    walk_forward,
)


def cal(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    """A synthetic trading calendar of n business days."""
    return pd.bdate_range(start, periods=n, name="Date")


def positions(index: pd.DatetimeIndex, sub: pd.DatetimeIndex) -> np.ndarray:
    return index.get_indexer(sub)


def train_start_pos(n: int, fraction: float) -> int:
    """Where `purged_split` puts the boundary — integer arithmetic, so the
    rule is exact rather than whatever binary floats round to."""
    return (n * round(fraction * 10_000)) // 10_000


def assert_no_leak(split: Split, index: pd.DatetimeIndex) -> None:
    """STRUCTURAL check, not a leak proof — see the module docstring.

    Verifies that the L bars immediately preceding any kept test bar are not
    training data, where L is the purge length the constructor was GIVEN. It
    cannot tell whether L is large enough; only the engine-level tests can.

    Also checks the sets are disjoint and that the embargo took bars on the
    correct (later) side of each test block.
    """
    L, h = split.lookback, split.embargo
    train_pos = set(positions(index, split.train).tolist())
    test_pos = positions(index, split.test)
    purged_pos = set(positions(index, split.purged).tolist())
    embargo_pos = set(positions(index, split.embargoed).tolist())

    # Disjointness of every reported set.
    assert not (train_pos & set(test_pos.tolist()))
    assert not (train_pos & purged_pos)
    assert not (train_pos & embargo_pos)
    assert not (set(test_pos.tolist()) & purged_pos)
    assert not (set(test_pos.tolist()) & embargo_pos)

    # The leak check.
    for p in test_pos:
        window = range(max(0, p - L), p + 1)
        leaked = train_pos.intersection(window)
        assert not leaked, (
            f"{split.label}: training bars {sorted(leaked)} lie inside the "
            f"{L}-bar lookback window of test bar {p}"
        )

    # Embargo is one-sided: it removes training bars AFTER a block, never
    # before it (data that ended before the test began was genuinely known).
    for lo, hi in split.test_blocks:
        a, b = index.get_loc(lo), index.get_loc(hi) + 1
        assert all(not (a <= q < b) for q in train_pos)
        for q in range(b, min(b + h, len(index))):
            assert q not in train_pos, (
                f"{split.label}: bar {q} is only {q - b} bars past the test "
                f"block but is being trained on (embargo is {h})"
            )


# ---------------------------------------------------------------------------
# purge_length / signal_reach
# ---------------------------------------------------------------------------

def spec_of(**kw) -> StrategySpec:
    defaults = dict(
        name="12mo momentum top-5",
        universe=Universe(preset="sp500"),
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=252), direction="top", n=5
        ),
        weighting="equal",
        rebalance="monthly",
        costs=Costs(),
        benchmark="SPY",
        window=Window(start="2015-01-01", end="2024-12-31"),
    )
    defaults.update(kw)
    return StrategySpec(**defaults)


def test_signal_reach_matches_each_formula():
    assert signal_reach(Signal(type="return", lookback_days=252)) == 252
    assert signal_reach(
        Signal(type="return", lookback_days=252, skip_days=21)) == 273
    assert signal_reach(Signal(type="volatility", lookback_days=60)) == 60
    assert signal_reach(Signal(type="sma_cross", fast=50, slow=200)) == 200
    assert signal_reach(Signal(type="drawdown_from_high", lookback_days=90)) == 90


def test_skip_days_is_added_only_where_the_engine_reads_it():
    """engine/signals.compute_signal consumes skip_days in the return /
    daily_return branch and NOWHERE else, so adding it elsewhere would make
    the 'mirrors the engine formula by formula' claim false."""
    assert signal_reach(Signal(type="volatility", lookback_days=60,
                               skip_days=10)) == 60
    assert signal_reach(Signal(type="drawdown_from_high", lookback_days=90,
                               skip_days=10)) == 90
    assert signal_reach(Signal(type="sma_cross", fast=10, slow=50,
                               skip_days=10)) == 50
    assert signal_reach(Signal(type="daily_return", lookback_days=5,
                               skip_days=3)) == 8


def test_rsi_reach_is_a_decay_bound_not_the_lookback():
    """Wilder smoothing is ewm(alpha=1/lb) — an IIR filter. The reach is the
    bar at which the EWMA's remaining weight mass falls below the truncation
    threshold, derived here from the geometric series by hand rather than by
    calling the implementation.

    Tail mass beyond n bars = (1 - 1/lb)^n. At lb=14, (13/14)^186 = 1.08e-6 is
    still above 1e-6 and (13/14)^187 = 1.00e-6 is below it, so n = 187, plus
    one bar for the prices.diff() the filter is fed => 188.
    """
    assert (13 / 14) ** 186 > 1e-6 > (13 / 14) ** 187
    assert rsi_reach(14) == 188
    assert signal_reach(Signal(type="rsi")) == 188          # engine default lb
    assert signal_reach(Signal(type="rsi", lookback_days=14)) == 188
    # …and the old answer (the lookback itself) is nowhere near it.
    assert rsi_reach(14) > 10 * 14
    assert (28 / 29) ** 393 > 1e-6 > (28 / 29) ** 394
    assert rsi_reach(29) == 395
    assert rsi_reach(1) == 2                                 # alpha=1, no memory


def test_rsi_at_the_declared_reach_actually_moves_less_than_the_threshold():
    """An end-to-end check on the engine's own RSI: a price perturbation well
    INSIDE the declared reach still moves RSI measurably, and one AT the
    declared reach moves it by 9.1e-06 points on this series — four orders of
    magnitude less, but NOT below RSI_TAIL_MASS_EPS, which bounds weight mass
    rather than RSI points. The bound asserted here is the measured residual
    (3.1e-05 is the worst case over 30 seeds of this generator), not the eps.
    """
    from engine.signals import compute_signal

    rng = np.random.default_rng(11)
    n = 900
    px = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})
    sig = Signal(type="rsi", lookback_days=14)
    base = compute_signal(px, sig)
    t = n - 1

    def moved(offset: int) -> float:
        q = px.copy()
        q.iloc[t - offset, 0] *= 1.05
        return abs(float(compute_signal(q, sig).iloc[t, 0] - base.iloc[t, 0]))

    # The old declared reach was the lookback itself (14). Three times that far
    # back a single close still moves RSI by half a point — more than enough to
    # reorder a top-n ranking, so 14 was not conservative, it was wrong.
    assert moved(3 * 14) > 0.4
    assert moved(5 * 14) > 1e-2
    assert moved(10 * 14) > 1e-4
    # …and the decay is geometric, so AT the declared reach it is negligible —
    # against the measured residual, which is the honest bound here.
    at_reach = moved(rsi_reach(14))
    assert 1e-6 < at_reach < 1e-4
    assert at_reach < moved(rsi_reach(14) // 2) / 100
    # Further out still, and monotonically smaller.
    assert moved(rsi_reach(14) + 40) < at_reach


def test_purge_length_is_reach_plus_carry_plus_lag_plus_cost_bar():
    c = purge_components(spec_of())
    assert c == {
        "signal_reach": 252, "staleness_reach": 21, "extra_reach": 0,
        "reach": 252, "carry": MAX_REBALANCE_CARRY_DAYS["monthly"],
        "execution_lag": DEFAULT_EXECUTION_LAG_DAYS, "cost_charge": 1,
        "total": 252 + 23 + 1 + 1,
    }
    assert purge_length(spec_of()) == 277


def test_execution_lag_changes_the_purge_because_the_engine_fills_late():
    """The audit re-runs the fill-sensitivity check at lag+1; the purge has to
    follow it, or that re-run is scored on a leaky split."""
    assert purge_length(spec_of(), execution_lag_days=0) == 276
    assert purge_length(spec_of(), execution_lag_days=1) == 277
    assert purge_length(spec_of(), execution_lag_days=2) == 278
    with pytest.raises(SplitError) as e:
        purge_length(spec_of(), execution_lag_days=-1)
    assert e.value.reason == "bad_execution_lag"


def test_the_staleness_screen_is_a_floor_under_every_non_static_spec():
    """engine/backtest gates eligibility on a rolling raw-close window on EVERY
    selection path, so a 5-day signal still depends on 21 bars of prices."""
    spec = spec_of(
        rebalance="daily",
        selection=Selection(rank_by=Signal(type="daily_return", lookback_days=1),
                            direction="bottom", n=5),
    )
    c = purge_components(spec)
    assert c["signal_reach"] == 1
    assert c["staleness_reach"] == STALENESS_LOOKBACK_DAYS == 21
    assert c["reach"] == 21                 # the screen, not the signal, binds
    assert c["total"] == 21 + 1 + 1 + 1


def test_the_staleness_screen_is_memoryless_and_reaches_exactly_21_bars():
    """Independent measurement of the floor, AND the contract it rests on.

    The screen's ELIGIBILITY VERDICT — not just the share behind it — must be a
    function of the trailing STALENESS_LOOKBACK_DAYS closes and nothing
    earlier. dsl/conventions.py states that requirement and names this module
    as the reason: give the screen memory (hysteresis with a forward-filled
    state) and its reach becomes unbounded, so no finite purge could cover it.

    Construction: 13 of every 21 closes repeat, a 61.9% unchanged share, just
    over the 60% flag level. Bumping one close makes two diffs non-zero, drops
    the share to 52.4%, and un-flags the ticker — for exactly as many bars as
    the window can still see that bump.
    """
    from engine.backtest import _staleness_panel

    pattern = [0.0] * 13 + [0.5] * 8
    v = [100.0]
    for i in range(1, 200):
        v.append(v[-1] + pattern[(i - 1) % 21])
    raw = pd.DataFrame({"A": v})
    before = _staleness_panel(raw)
    assert before["A"].iloc[60:].all(), "the fixture is supposed to read stale"

    # bars where the bumped close sits between two REPEATED closes, so the bump
    # actually destroys two zero-diffs (bar % 21 in 1..12 for this pattern)
    for bar in (106, 111, 148):
        bumped = raw.copy()
        bumped.iloc[bar, 0] += 0.37
        flipped = np.flatnonzero((before != _staleness_panel(bumped)).values[:, 0])
        assert len(flipped), "the bump did not change the verdict — inert probe"
        assert flipped.min() == bar          # never backwards: no lookahead
        assert flipped.max() - bar == STALENESS_LOOKBACK_DAYS == 21, (
            "the staleness verdict outlived its own window — the screen has "
            "grown memory, and purge_components' staleness floor is now wrong"
        )

    assert purge_components(spec_of())["staleness_reach"] == 21


def test_purge_length_covers_skip_days_filters_and_cohorts():
    spec = spec_of(
        selection=Selection(
            rank_by=Signal(type="return", lookback_days=252, skip_days=21),
            direction="top", n=5,
        ),
        filters=[Filter(signal=Signal(type="volatility", lookback_days=500),
                        op="lt", value=0.4)],
        entry_exit=EntryExit(hold_days=20),
    )
    # max(273, 500, 21) + 20 hold days + 1 lag + 1 cost bar; hold_days replaces
    # the rebalance carry because cohorts, not weights, define the holding
    # period.
    assert purge_length(spec) == 500 + 20 + 1 + 1


def test_inverse_vol_weighting_adds_its_implicit_63_day_lookback():
    spec = spec_of(
        weighting="inverse_vol",
        selection=Selection(rank_by=Signal(type="return", lookback_days=5),
                            direction="top", n=2),
        rebalance="daily",
    )
    assert purge_components(spec)["signal_reach"] == 63


def test_static_allocation_has_no_signal_and_no_staleness_reach():
    spec = spec_of(
        selection=Selection(static_allocation=[{"ticker": "SPY", "weight": 0.6},
                                               {"ticker": "TLT", "weight": 0.4}]),
        rebalance="quarterly",
    )
    assert purge_components(spec) == {
        "signal_reach": 0, "staleness_reach": 0, "extra_reach": 0, "reach": 0,
        "carry": 66, "execution_lag": 1, "cost_charge": 1, "total": 68,
    }


def test_extra_reach_is_disclosed_separately_from_what_the_module_derived():
    """If the engine ever grows a dependency this module cannot bound (a
    staleness screen with memory is the live example — see the note in
    dsl/conventions.py), a caller can add an allowance. The report must be able
    to tell that allowance apart from the reach this module derived itself."""
    c = purge_components(spec_of(rebalance="daily"), extra_reach=70)
    assert c["signal_reach"] == 252 and c["staleness_reach"] == 21
    assert c["extra_reach"] == 70
    assert c["reach"] == 252                       # max, not a sum
    c2 = purge_components(
        spec_of(rebalance="daily",
                selection=Selection(rank_by=Signal(type="daily_return",
                                                   lookback_days=1),
                                    direction="bottom", n=5)),
        extra_reach=70,
    )
    assert c2["reach"] == 70 and c2["total"] == 73
    with pytest.raises(SplitError) as e:
        purge_components(spec_of(), extra_reach=-1)
    assert e.value.reason == "bad_extra_reach"


def test_rebalance_carry_can_be_switched_off_but_is_on_by_default():
    assert purge_length(spec_of(rebalance="annual")) == 252 + 262 + 1 + 1
    assert purge_length(spec_of(rebalance="annual"),
                        include_rebalance_carry=False) == 252 + 0 + 1 + 1


def test_the_constant_carry_table_is_a_worst_case_bound_not_a_mean():
    """Signal dates are period ENDS, so the gap between two of them is exactly
    the number of index bars inside the later period. The bound therefore has
    to be the largest number of weekdays such a period can hold — derived here
    by counting a 200-year weekday calendar, with no reference to the module's
    own table. Holidays only ever shrink these.
    """
    days = pd.bdate_range("1900-01-01", "2100-12-31")
    counts = pd.Series(1, index=days)
    observed = {
        "weekly": int(counts.groupby(days.to_period("W")).size().max()),
        "monthly": int(counts.groupby(days.to_period("M")).size().max()),
        "quarterly": int(counts.groupby(days.to_period("Q")).size().max()),
        "annual": int(counts.groupby(days.to_period("Y")).size().max()),
    }
    assert observed == {"weekly": 5, "monthly": 23, "quarterly": 66,
                        "annual": 262}
    for freq, bound in observed.items():
        assert MAX_REBALANCE_CARRY_DAYS[freq] == bound, (
            f"{freq}: table says {MAX_REBALANCE_CARRY_DAYS[freq]} but a period "
            f"can hold {bound} trading days — the purge would be short"
        )
    assert MAX_REBALANCE_CARRY_DAYS["daily"] == 1


def test_calendar_carry_reads_the_true_maximum_off_a_real_calendar(sp500_panel):
    """On the project's own SPY calendar the mean month is 20.9 bars and the
    longest is 23. Counting bars per calendar month directly (a different route
    from the production groupby-max on period ends) must agree."""
    index = sp500_panel.index
    per_month = index.to_series().groupby(
        [index.year, index.month]).size().tolist()
    # Gap between consecutive month-end signal dates == bars in the later month.
    assert max(per_month[1:]) == 23
    assert calendar_rebalance_carry(index, "monthly") == 23
    assert calendar_rebalance_carry(index, "daily") == 1

    per_q = index.to_series().groupby([index.year, index.quarter]).size().tolist()
    assert calendar_rebalance_carry(index, "quarterly") == max(per_q[1:]) == 64
    per_y = index.to_series().groupby(index.year).size().tolist()
    assert calendar_rebalance_carry(index, "annual") == max(per_y[1:]) == 253
    assert calendar_rebalance_carry(index, "weekly") == 5

    # …and every observed maximum is inside the constant bound, so the fallback
    # can never under-purge relative to the calendar-derived number.
    for freq in ("weekly", "monthly", "quarterly", "annual"):
        assert (calendar_rebalance_carry(index, freq)
                <= MAX_REBALANCE_CARRY_DAYS[freq])


def test_calendar_carry_refuses_when_a_signal_date_is_not_a_calendar_bar():
    """A signal date the index does not contain makes every measured gap
    fiction, and the fiction is SHORT, not long.

    engine.calendar.rebalance_signal_dates ends with
    `pd.DatetimeIndex(last_days.values)`, which drops the tzinfo. tz-aware
    calendars are inside this module's contract (see
    test_cpcv_path_index_survives_a_tz_aware_calendar), and on one of them every
    signal date then misses: get_indexer returns -1 throughout, np.diff of
    those -1s is all zeros, and max([1, *zeros]) == 1 where the true monthly
    carry is 23 — a 22-bar-short purge produced silently. Refuse instead.
    """
    from engine.calendar import rebalance_signal_dates

    aware = pd.bdate_range("2015-01-05", periods=1500, tz="America/New_York")
    naive = aware.tz_localize(None)
    assert list(naive) == list(pd.bdate_range("2015-01-05", periods=1500))

    # The cause, measured here rather than asserted from memory.
    sig = pd.DatetimeIndex(rebalance_signal_dates(aware, "monthly"))
    pos = aware.get_indexer(sig)
    assert aware.tz is not None and sig.tz is None
    assert len(pos) == 70 and int((pos == -1).sum()) == 70
    assert np.diff(pos).tolist() == [0] * 69     # → an unguarded carry of 1

    with pytest.raises(SplitError) as e:
        calendar_rebalance_carry(aware, "monthly")
    assert e.value.reason == "signal_dates_not_on_calendar"
    assert e.value.details["n_missing"] == 70
    assert e.value.details["n_signal_dates"] == 70

    # The refusal reaches the caller instead of a purge that is 22 bars short.
    with pytest.raises(SplitError) as e:
        purge_components(spec_of(), index=aware)
    assert e.value.reason == "signal_dates_not_on_calendar"

    # The tz-naive twin of the very same calendar is untouched, so the guard
    # fires on the missing dates and not merely on the presence of a tz.
    assert calendar_rebalance_carry(naive, "monthly") == 23
    assert purge_components(spec_of(), index=naive)["carry"] == 23


def test_the_calendar_carry_is_what_the_purge_uses_when_an_index_is_given(
    sp500_panel,
):
    index = sp500_panel.index
    assert purge_components(spec_of(), index=index)["carry"] == 23
    assert purge_components(spec_of(rebalance="annual"),
                           index=index)["carry"] == 253
    # the constant fallback is never smaller than the measured number
    assert (purge_components(spec_of(rebalance="annual"))["carry"]
            >= purge_components(spec_of(rebalance="annual"),
                                index=index)["carry"])


def test_default_embargo_is_one_percent_floored_at_one_bar():
    assert default_embargo(2520) == 25
    assert default_embargo(1000) == 10
    assert default_embargo(50) == 1  # never silently zero
    with pytest.raises(SplitError) as e:
        default_embargo(0)
    assert e.value.reason == "empty_index"


# ---------------------------------------------------------------------------
# purged_split
# ---------------------------------------------------------------------------

def test_purged_split_leaves_exactly_L_bars_between_train_and_test():
    index = cal(1000)
    s = purged_split(index, 0.70, lookback=274)
    assert len(s.purged) == 274
    train_end = index.get_loc(s.train[-1])
    test_start = index.get_loc(s.test[0])
    assert train_end == 699                      # int(1000 * .70) - 1
    assert test_start - train_end - 1 == 274     # the purge, bar for bar
    assert len(s.train) + len(s.test) + len(s.purged) == 1000
    assert len(s.embargoed) == 0                 # nothing trains after a tail test
    assert_no_leak(s, index)


def test_purged_split_train_and_test_never_overlap_across_a_grid():
    rng = random.Random(20240712)
    for _ in range(200):
        n = rng.randint(120, 4000)
        frac = rng.uniform(0.4, 0.9)
        lookback = rng.randint(0, 200)
        index = cal(n)
        a = train_start_pos(n, frac)
        if a < 1 or n - a <= lookback:
            with pytest.raises(SplitError):
                purged_split(index, frac, lookback=lookback)
            continue
        s = purged_split(index, frac, lookback=lookback)
        assert s.train.intersection(s.test).empty
        assert len(s.purged) == lookback
        assert_no_leak(s, index)


def test_purged_split_refuses_when_the_holdout_is_shorter_than_the_purge():
    index = cal(500)  # holdout = 150 bars, purge = 274
    with pytest.raises(SplitError) as e:
        purged_split(index, 0.70, lookback=274)
    assert e.value.reason == "fold_shorter_than_purge"
    assert e.value.details["required_bars"] == 275
    assert "purged" in str(e.value)


def test_purged_split_rejects_a_degenerate_train_fraction():
    index = cal(500)
    for frac in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(SplitError) as e:
            purged_split(index, frac, lookback=5)
        assert e.value.reason in ("bad_train_fraction", "empty_train")


def test_purged_split_is_deterministic():
    index = cal(1500)
    a = purged_split(index, 0.7, lookback=100)
    b = purged_split(index, 0.7, lookback=100)
    pd.testing.assert_index_equal(a.train, b.train)
    pd.testing.assert_index_equal(a.test, b.test)
    assert a.as_dict() == b.as_dict()


def test_split_as_dict_is_json_safe():
    import json

    s = purged_split(cal(1000), 0.7, lookback=50)
    payload = json.dumps(s.as_dict())
    assert '"purged_days": 50' in payload
    assert '"purge_bars": 50' in payload


def test_as_dict_separates_the_raw_block_from_the_bars_actually_scored():
    """A report that prints the raw block bounds next to test_start/test_end is
    claiming L more holdout bars per block than it scored. Both are emitted,
    each under a name that says which one it is."""
    index = cal(1000)
    s = purged_split(index, 0.70, lookback=50)
    d = s.as_dict()
    assert d["block_bounds"] == [[str(index[700].date()), str(index[999].date())]]
    assert d["scored_bounds"] == [[str(index[750].date()), str(index[999].date())]]
    assert d["scored_bounds"][0][0] == d["test_start"]
    assert d["scored_bounds"][0][1] == d["test_end"]
    assert "test_blocks" not in d          # the ambiguous key is gone


def test_the_train_test_boundary_is_integer_arithmetic_not_a_float_round():
    """int(n * 0.70) lands one bar early for 143 of the index lengths in
    [10, 6000) — n=90 gives 62 where 70% of 90 is 63. A split boundary has to
    be a rule, not a rounding accident."""
    offenders = [n for n in range(10, 6000)
                 if int(n * 0.70) != (n * 70) // 100]
    assert len(offenders) == 143 and offenders[0] == 90

    for n in (90, 170, 180, 330, 340, 350, 360, 650, 660, 670):
        s = purged_split(cal(n), 0.70, lookback=1)
        boundary = cal(n).get_loc(s.train[-1]) + 1
        assert boundary == (n * 70) // 100 == int(n * 0.70) + 1

    # …and the documented 70/30 shape still holds exactly where it always did.
    s = purged_split(cal(1000), 0.70, lookback=1)
    assert cal(1000).get_loc(s.train[-1]) == 699


def test_contaminated_train_bars_is_reported_for_interior_blocks():
    """A trailing test block contaminates no training bar (nothing trains after
    it). An interior CPCV block does, because the embargo (h ≈ .01T) is far
    shorter than the purge — that residual is disclosed, not hidden."""
    index = cal(3000)
    assert purged_split(index, 0.70, lookback=274).contaminated_train_bars == 0

    design = combinatorial_purged_cv(index, 6, 1, lookback=100, embargo=30)
    interior = design.splits[2]
    # training resumes 30 bars after the block; a bar p is contaminated while
    # p - 100 is still inside it, i.e. for 100 - 30 = 70 bars.
    assert interior.contaminated_train_bars == 70
    assert interior.as_dict()["contaminated_train_bars"] == 70
    # …and with an embargo at least as long as the purge, nothing is left.
    clean = combinatorial_purged_cv(index, 6, 1, lookback=100, embargo=100)
    assert clean.splits[2].contaminated_train_bars == 0


# ---------------------------------------------------------------------------
# walk_forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchored", [True, False])
def test_walk_forward_folds_are_ordered_disjoint_and_purged(anchored):
    index = cal(3000)
    folds = walk_forward(index, 6, lookback=120, embargo=30, anchored=anchored)
    assert len(folds) == 6
    last_end = -1
    seen = pd.DatetimeIndex([])
    for f in folds:
        assert_no_leak(f, index)
        assert len(f.purged) == 120
        start = index.get_loc(f.test[0])
        assert start > last_end
        last_end = index.get_loc(f.test[-1])
        assert seen.intersection(f.test).empty
        seen = seen.union(f.test)


def test_walk_forward_training_is_always_strictly_in_the_past():
    index = cal(3000)
    for anchored in (True, False):
        for f in walk_forward(index, 5, lookback=90, anchored=anchored):
            assert f.train[-1] < f.test[0]
            block_start = f.test_blocks[0][0]
            assert f.train[-1] < block_start  # not even into the purge zone


def test_anchored_training_grows_and_always_starts_at_bar_zero():
    index = cal(3000)
    folds = walk_forward(index, 6, lookback=60, anchored=True)
    lengths = [len(f.train) for f in folds]
    assert all(f.train[0] == index[0] for f in folds)
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_rolling_training_window_is_a_constant_length_that_moves():
    index = cal(3000)
    folds = walk_forward(index, 6, lookback=60, anchored=False)
    lengths = {len(f.train) for f in folds}
    assert lengths == {3000 // 7}
    starts = [f.train[0] for f in folds]
    assert starts == sorted(starts) and starts[0] < starts[-1]


def test_walk_forward_last_fold_absorbs_the_remainder():
    index = cal(1003)
    folds = walk_forward(index, 4, lookback=10)
    assert folds[-1].test[-1] == index[-1]
    union = folds[0].test
    for f in folds[1:]:
        union = union.union(f.test)
    # every bar after the initial training block is either tested or purged
    start = 1003 // 5
    assert len(union) + 4 * 10 == 1003 - start


def test_walk_forward_refuses_folds_shorter_than_the_purge():
    index = cal(1000)
    with pytest.raises(SplitError) as e:
        walk_forward(index, 8, lookback=200)
    assert e.value.reason == "fold_shorter_than_purge"
    assert e.value.details["fold_bars"] < e.value.details["required_bars"]
    assert "fewer folds" in str(e.value)


def test_walk_forward_rejects_degenerate_shapes():
    with pytest.raises(SplitError) as e:
        walk_forward(cal(1000), 0, lookback=10)
    assert e.value.reason == "bad_n_folds"
    with pytest.raises(SplitError) as e:
        walk_forward(cal(5), 6, lookback=1)
    assert e.value.reason == "empty_train"
    with pytest.raises(SplitError) as e:
        walk_forward(cal(500), 3, lookback=10, train_bars=500)
    assert e.value.reason == "no_test_room"


def test_walk_forward_never_leaks_across_a_grid():
    rng = random.Random(31415)
    for _ in range(120):
        n = rng.randint(400, 5000)
        n_folds = rng.randint(2, 10)
        lookback = rng.randint(0, 120)
        anchored = rng.random() < 0.5
        index = cal(n)
        try:
            folds = walk_forward(index, n_folds, lookback=lookback,
                                 anchored=anchored)
        except SplitError as e:
            assert e.reason in ("fold_shorter_than_purge", "empty_train",
                                "no_test_room")
            continue
        assert len(folds) == n_folds
        for f in folds:
            assert_no_leak(f, index)


# ---------------------------------------------------------------------------
# combinatorial_purged_cv
# ---------------------------------------------------------------------------

def test_cpcv_worked_example_from_the_paper():
    """AFML ch.12: N=6, k=2 must give exactly 15 splits and 5 paths."""
    design = combinatorial_purged_cv(cal(3000), 6, 2, lookback=60)
    assert isinstance(design, CPCVDesign)
    assert design.n_splits == 15
    assert design.n_paths == 5
    assert n_cpcv_splits(6, 2) == 15
    assert n_cpcv_paths(6, 2) == 5


@pytest.mark.parametrize("n_groups,n_test_groups", [
    (2, 1), (4, 1), (5, 2), (6, 2), (6, 3), (8, 2), (8, 4), (10, 3),
])
def test_cpcv_combinatorics(n_groups, n_test_groups):
    from math import comb

    index = cal(6000)
    design = combinatorial_purged_cv(index, n_groups, n_test_groups,
                                     lookback=30)
    assert design.n_splits == comb(n_groups, n_test_groups)
    assert design.n_paths == comb(n_groups - 1, n_test_groups - 1)
    assert design.n_paths == n_test_groups * design.n_splits // n_groups

    # every split tests exactly k groups
    assert all(len(s.test_blocks) == n_test_groups for s in design.splits)
    # every group is tested in exactly φ splits
    counts = {g: 0 for g in range(n_groups)}
    for path in design.paths:
        for seg in path.segments:
            counts[seg.group] += 1
    assert set(counts.values()) == {design.n_paths}


def test_cpcv_paths_each_cover_the_whole_timeline_once():
    index = cal(3000)
    design = combinatorial_purged_cv(index, 6, 2, lookback=60)
    for path in design.paths:
        assert [s.group for s in path.segments] == list(range(6))
        idx = path.index
        assert idx.is_monotonic_increasing and not idx.has_duplicates
        # each group contributes its bars minus the 60 purged at its head
        assert len(idx) == 3000 - 6 * 60
        # a path draws from φ distinct splits' worth of provenance
        assert len({s.split_id for s in path.segments}) > 1


def test_cpcv_path_index_survives_a_tz_aware_calendar():
    """The documented integration call is metrics.summary(net.loc[path.index]).
    Round-tripping the segments through their raw numpy values turned a
    tz-aware DatetimeIndex into a UTC-naive one, so that .loc raised KeyError
    on any real exchange calendar."""
    index = pd.bdate_range("2015-01-05", periods=3000, tz="America/New_York",
                           name="Date")
    design = combinatorial_purged_cv(index, 6, 2, lookback=60)
    net = pd.Series(1.0, index=index)
    for path in design.paths:
        idx = path.index
        assert idx.tz is not None and str(idx.tz) == "America/New_York"
        assert idx.name == "Date"
        assert idx.is_monotonic_increasing and not idx.has_duplicates
        assert len(idx) == 3000 - 6 * 60
        assert idx.isin(index).all()
        assert float(net.loc[idx].sum()) == len(idx)   # the integration call


def test_cpcv_groups_partition_the_index_with_the_remainder_last():
    index = cal(1007)
    design = combinatorial_purged_cv(index, 5, 2, lookback=10)
    bounds = design.group_bounds
    assert len(bounds) == 5
    assert bounds[0][0] == index[0] and bounds[-1][1] == index[-1]
    sizes = [index.get_loc(hi) - index.get_loc(lo) + 1 for lo, hi in bounds]
    assert sizes[:-1] == [201] * 4          # floor(1007 / 5)
    assert sizes[-1] == 1007 - 201 * 4      # the remainder rides last
    assert sum(sizes) == 1007


def test_cpcv_every_split_is_purged_and_embargoed_without_leaking():
    index = cal(3000)
    design = combinatorial_purged_cv(index, 6, 2, lookback=60, embargo=30)
    for s in design.splits:
        assert_no_leak(s, index)
        assert len(s.purged) == 2 * 60
        assert len(s.train) > 0


def test_cpcv_embargo_bites_on_the_later_side_only():
    """An interior test block must cost training bars after it and none
    before it — that asymmetry IS the embargo rule (AFML §7.4.1)."""
    index = cal(3000)
    design = combinatorial_purged_cv(index, 6, 1, lookback=0, embargo=25)
    interior = design.splits[2]                   # group 3 of 6 → interior
    lo, hi = interior.test_blocks[0]
    a, b = index.get_loc(lo), index.get_loc(hi) + 1
    train_pos = set(positions(index, interior.train).tolist())

    assert len(interior.embargoed) == 25
    assert set(positions(index, interior.embargoed).tolist()) == set(range(b, b + 25))
    for q in range(b, b + 25):
        assert q not in train_pos
    assert b + 25 in train_pos                    # training resumes right after
    assert a - 1 in train_pos                     # …and was never cut before


def test_cpcv_embargo_is_not_double_counted_against_an_adjacent_block():
    index = cal(600)
    design = combinatorial_purged_cv(index, 6, 2, lookback=0, embargo=20)
    adjacent = next(s for s in design.splits
                    if s.label == "cpcv_g1+g2")
    # the embargo after group 1 falls entirely inside group 2, which is
    # already a test block — it is excluded, but not reported as embargoed
    assert len(adjacent.embargoed) == 20   # only the one after group 2
    assert_no_leak(adjacent, index)


def test_cpcv_refuses_groups_shorter_than_the_purge():
    with pytest.raises(SplitError) as e:
        combinatorial_purged_cv(cal(600), 6, 2, lookback=120)
    assert e.value.reason == "fold_shorter_than_purge"
    assert e.value.details["fold_bars"] == 100
    assert e.value.details["required_bars"] == 121
    assert "fewer groups" in str(e.value)


def test_cpcv_rejects_degenerate_group_counts():
    index = cal(3000)
    with pytest.raises(SplitError) as e:
        combinatorial_purged_cv(index, 1, 1, lookback=10)
    assert e.value.reason == "bad_n_groups"
    for k in (0, 6, 7):
        with pytest.raises(SplitError) as e:
            combinatorial_purged_cv(index, 6, k, lookback=10)
        assert e.value.reason == "bad_n_test_groups"


def test_cpcv_is_deterministic_and_json_safe():
    import json

    index = cal(2000)
    a = combinatorial_purged_cv(index, 6, 2, lookback=40)
    b = combinatorial_purged_cv(index, 6, 2, lookback=40)
    assert a.as_dict() == b.as_dict()
    payload = json.loads(json.dumps(a.as_dict()))
    assert payload["n_splits"] == 15 and payload["n_paths"] == 5


def test_cpcv_never_leaks_across_a_grid():
    rng = random.Random(2718281)
    for _ in range(40):
        n = rng.randint(800, 6000)
        n_groups = rng.randint(2, 9)
        n_test_groups = rng.randint(1, n_groups - 1)
        lookback = rng.randint(0, 150)
        index = cal(n)
        try:
            design = combinatorial_purged_cv(index, n_groups, n_test_groups,
                                             lookback=lookback)
        except SplitError as e:
            assert e.reason in ("fold_shorter_than_purge", "empty_train")
            continue
        assert design.n_splits == n_cpcv_splits(n_groups, n_test_groups)
        assert design.n_paths == n_cpcv_paths(n_groups, n_test_groups)
        for s in design.splits:
            assert_no_leak(s, index)


# ---------------------------------------------------------------------------
# Calendar hygiene — data problems surface, they are never absorbed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [
    lambda: pd.DatetimeIndex([]),
    lambda: pd.DatetimeIndex(["2020-01-02", "2020-01-02", "2020-01-03"]),
    lambda: pd.DatetimeIndex(["2020-01-03", "2020-01-02", "2020-01-06"]),
])
def test_a_broken_calendar_is_refused_not_absorbed(builder):
    index = builder()
    for call in (
        lambda: purged_split(index, 0.7, lookback=1),
        lambda: walk_forward(index, 2, lookback=1),
        lambda: combinatorial_purged_cv(index, 3, 1, lookback=1),
    ):
        with pytest.raises(SplitError) as e:
            call()
        assert e.value.reason in ("empty_index", "duplicate_dates",
                                  "unsorted_index")


def test_a_non_datetime_index_is_refused():
    with pytest.raises(SplitError) as e:
        purged_split(pd.RangeIndex(100), 0.7, lookback=1)
    assert e.value.reason == "not_a_datetime_index"


def test_negative_gaps_are_refused():
    index = cal(1000)
    with pytest.raises(SplitError) as e:
        purged_split(index, 0.7, lookback=-1)
    assert e.value.reason == "bad_lookback"
    with pytest.raises(SplitError) as e:
        purged_split(index, 0.7, lookback=1, embargo=-5)
    assert e.value.reason == "bad_embargo"


def test_real_spec_on_a_real_calendar_end_to_end(sp500_panel):
    """The defect that started this: a 252-day momentum spec on the audit's
    own calendar must leave a purge gap, and the 70/30 cut must lose bars."""
    index = sp500_panel.index
    L = purge_length(spec_of(), index=index)
    s = purged_split(index, 0.70, lookback=L)
    assert len(s.purged) == L == 252 + 23 + 1 + 1
    assert (s.test[0] - s.train[-1]).days >= L      # calendar days ≥ bar count
    assert len(s.test) < len(index) - train_start_pos(len(index), 0.70)
    assert_no_leak(s, index)


# ---------------------------------------------------------------------------
# Engine-level leak — the only tests in this file that can catch a short purge
# ---------------------------------------------------------------------------
#
# Everything above asserts that the constructors applied the purge length they
# were handed. These run the real engine and ask the opposite, load-bearing
# question: is that length ENOUGH? The method is a perturbation experiment, so
# nothing here re-derives the formula:
#
#   1. run the spec on a real price panel                      -> base
#   2. multiply the closes on ONE TRAINING BAR by ±8%          -> perturbed
#   3. re-run, and diff the two net-return series
#
# Any test bar whose net return moved is a bar that depends on a training
# price. Under a correct purge there are none, exactly.
#
# The boundary is not arbitrary: it is placed where the bound BINDS, so the
# last bar the perturbation reaches is exactly a+L-1. That pins every term of
# L at once — shrink MAX_REBALANCE_CARRY_DAYS, drop the execution lag or drop
# the cost-charge bar and a test bar moves.
#
# The STALENESS floor is the one term these boundaries CANNOT see, and that is
# structural, not an oversight: REACH below is 21 = STALENESS_LOOKBACK_DAYS, so
# `signal_reach` masks `staleness_reach` in every one of those parametrisations
# and zeroing the floor leaves L unchanged. The `staleness` case is the one that
# pins it — a 1-bar ranking signal, where the screen is the entire reach (L = 24
# with it, 4 without).

REACH = 21          # rank_by return(21) — chosen to equal STALENESS_LOOKBACK_DAYS


def leak_spec(index: pd.DatetimeIndex, **kw) -> StrategySpec:
    defaults = dict(
        name="leak probe",
        universe=Universe(preset="sp500"),
        selection=Selection(rank_by=Signal(type="return", lookback_days=REACH),
                            direction="top", n=3),
        weighting="equal",
        rebalance="monthly",
        costs=Costs(),
        benchmark="SPY",
        window=Window(start=str(index[0].date()), end=str(index[-1].date())),
    )
    defaults.update(kw)
    return StrategySpec(**defaults)


def signal_positions(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    from engine.calendar import rebalance_signal_dates

    return index.get_indexer(pd.DatetimeIndex(
        rebalance_signal_dates(index, freq)))


def worst_case_boundary(index: pd.DatetimeIndex, reach: int) -> int:
    """Position the train/test cut where the purge bound is TIGHT.

    A training bar at `a-1` is the oldest close a signal formed at `a+reach-1`
    can see, and that signal keeps earning until the NEXT signal date fills. So
    the bound binds when a signal date sits at `a + reach - 1` and the gap to
    the following one is the calendar's maximum. Anchor on the calendar's
    longest MONTH: its opening month-end is a signal date for the monthly,
    weekly (it is a Friday) and daily schedules alike, and the month that
    follows it is the 23-bar worst case.
    """
    pos = signal_positions(index, "monthly")
    k = int(np.argmax(np.diff(pos)))
    return int(pos[k]) - (reach - 1)


def local_carry(index: pd.DatetimeIndex, freq: str, anchor: int) -> int:
    """Bars from the signal date at `anchor` to the next one — the carry that
    actually binds at this boundary, measured rather than assumed."""
    pos = signal_positions(index, freq)
    assert anchor in pos, f"bar {anchor} is not a {freq} signal date"
    nxt = pos[pos > anchor]
    return int(nxt[0] - anchor)


def leak_window(sp500_panel, reach=REACH, n_tickers=12, before=300, after=220):
    a_full = worst_case_boundary(sp500_panel.index, reach)
    lo = max(0, a_full - before)
    hi = min(len(sp500_panel.index), a_full + after)
    index = sp500_panel.index[lo:hi]
    panel = sp500_panel.iloc[lo:hi, :n_tickers]
    return panel, index, a_full - lo


def staleness_leak_window(sp500_panel):
    """The boundary where the STALENESS screen is the WHOLE reach.

    `SW` is a stale feed in the fixture (218 of the 374 day-on-day changes in
    this window are exactly zero), so the rolling 21-bar screen in
    engine/backtest.py really does flip its eligibility here — it flags SW on
    167 of the 375 bars and no other ticker on any of them. The ranking signal is daily_return(1):
    signal_reach 1 against the screen's 21, so nothing masks the floor. The
    universe is SW plus seven more tickers so the bottom-3 selection has
    somewhere to go on the bars where SW is screened out.

    The cut is the bar after 2018-05-18, where the bound binds: measured, a ±8%
    perturbation of that close moves the net return at offsets
    -1, 0, 1, 2, 3, 17, 18, 22 and 23 from the cut — the furthest exactly
    a+L-1 for L = 21 + 1 + 1 + 1 = 24. Without the staleness floor L would be
    1 + 1 + 1 + 1 = 4, and the four bars at +17, +18, +22, +23 would be scored
    as holdout.
    """
    tickers = ["SW"] + [c for c in sp500_panel.columns[:7] if c != "SW"]
    panel = sp500_panel.loc["2018-01-01":"2019-06-30", tickers]
    index = panel.index
    return panel, index, index.get_loc(pd.Timestamp("2018-05-18")) + 1


def staleness_leak_spec(index: pd.DatetimeIndex) -> StrategySpec:
    return leak_spec(
        index, rebalance="daily",
        selection=Selection(
            rank_by=Signal(type="daily_return", lookback_days=1),
            direction="bottom", n=3),
    )


def net_returns(panel, spec, lag):
    from engine.backtest import run

    return run(spec, panel, list(panel.columns),
               execution_lag_days=lag).net_returns[5.0]


def perturbed_bars(panel, spec, index, bar_pos, lag):
    """Positions whose NET return moves when the closes at `bar_pos` move.

    The factors alternate ±8% across tickers so the perturbation can actually
    reorder a ranking; scaling every ticker by the same factor would leave a
    ratio-based signal's ordering untouched and the probe would prove nothing.
    """
    factors = pd.Series(
        [1.08 if i % 2 == 0 else 0.92 for i in range(panel.shape[1])],
        index=panel.columns,
    )
    base = net_returns(panel, spec, lag)
    bumped = panel.copy()
    bumped.loc[index[bar_pos]] = bumped.loc[index[bar_pos]] * factors
    moved = net_returns(bumped, spec, lag)
    assert moved.index.equals(base.index)
    delta = (moved - base).abs()
    return base, moved, [index.get_loc(t) for t in delta.index[delta.values > 0]]


@pytest.mark.parametrize("case,freq,lag", [
    ("signal", "monthly", 0), ("signal", "monthly", 1),
    ("signal", "monthly", 2), ("signal", "weekly", 1), ("signal", "daily", 1),
    # The only case whose reach comes from the staleness screen rather than
    # from the ranking signal — see staleness_leak_window.
    ("staleness", "daily", 1),
])
def test_engine_says_no_test_bar_depends_on_a_training_price(sp500_panel, case,
                                                             freq, lag):
    if case == "staleness":
        panel, index, a = staleness_leak_window(sp500_panel)
        spec = staleness_leak_spec(index)
        # Daily: every bar is a signal date, so the carry is 1 by construction
        # and the boundary is chosen for the staleness channel instead.
        assert calendar_rebalance_carry(index, freq) == \
            MAX_REBALANCE_CARRY_DAYS[freq] == 1
        assert purge_components(spec, index=index)["signal_reach"] == 1
    else:
        panel, index, a = leak_window(sp500_panel)
        spec = leak_spec(index, rebalance=freq)

        # The boundary really is the worst case for this schedule: the signal
        # date at a+REACH-1 is followed by the longest gap the constant table
        # allows, so the purge bound BINDS here and no slack hides a missing
        # term.
        assert local_carry(index, freq, a + REACH - 1) == \
            MAX_REBALANCE_CARRY_DAYS[freq] == calendar_rebalance_carry(index,
                                                                      freq)

    # The purge length production would hand the audit, taken from the CONSTANT
    # table (no index=) so a mutation of MAX_REBALANCE_CARRY_DAYS reaches here.
    L = purge_length(spec, execution_lag_days=lag)

    split = walk_forward(index, 1, lookback=L, train_bars=a)[0]
    assert split.train[-1] == index[a - 1]
    assert split.test[0] == index[a + L]

    base, moved, changed = perturbed_bars(panel, spec, index, a - 1, lag)

    # 1. The probe has bite: perturbing that bar DOES change the simulation.
    assert changed, "the perturbation changed nothing — the probe is inert"
    # 2. THE claim this module exists to make.
    kept = split.test.intersection(base.index)
    assert len(kept) > 50
    assert (moved.loc[kept] == base.loc[kept]).all(), (
        f"{freq} lag={lag}: net returns on "
        f"{int((moved.loc[kept] != base.loc[kept]).sum())} test bar(s) depend "
        f"on the close of training bar {index[a - 1].date()}"
    )
    # 3. …and L is no more than a bar longer than it has to be: the
    #    perturbation reaches the very edge of the purge zone. Drop any term
    #    from L and assertion 2 fires; inflate any term and this one does.
    #    (The single bar of slack is the cost-charge term, whose Σ|Δw| happens
    #    to come out identical on some boundaries — it is demonstrably needed
    #    on others, see test_the_purge_one_bar_shorter_actually_leaks.)
    assert a + L - 2 <= max(changed) <= a + L - 1, (
        f"{freq} lag={lag}: purge is {L}, but the training bar only reaches "
        f"bar a+{max(changed) - a} — {L - 1 - (max(changed) - a)} bars of the "
        f"holdout are being thrown away for nothing"
    )


def test_the_purge_one_bar_shorter_actually_leaks(sp500_panel):
    """The other half of the proof. If L-1 were also clean, the test above
    would be satisfied by any large enough number and could not pin anything.
    """
    panel, index, a = leak_window(sp500_panel)
    spec = leak_spec(index, rebalance="daily")
    lag = 1
    L = purge_length(spec, index=index, execution_lag_days=lag)

    base, moved, changed = perturbed_bars(panel, spec, index, a - 1, lag)
    short = walk_forward(index, 1, lookback=L - 1, train_bars=a)[0]
    kept = short.test.intersection(base.index)
    leaked = kept[(moved.loc[kept] != base.loc[kept]).values]
    assert len(leaked) == 1 and index.get_loc(leaked[0]) == a + L - 1
    assert float((moved.loc[leaked] - base.loc[leaked]).abs().max()) > 1e-5


def test_the_leak_probe_catches_the_old_purge_formula(sp500_panel):
    """Regression lock on the three terms the old formula got wrong. It used
    signal_reach + 1 + MEAN period length: no staleness floor, no execution
    lag, a 21-bar month and a label span that was really the cost-charge bar.
    On this boundary that is 21 + 1 + 21 = 43 against the true 46, and the
    engine says so."""
    panel, index, a = leak_window(sp500_panel)
    spec = leak_spec(index, rebalance="monthly")
    lag = DEFAULT_EXECUTION_LAG_DAYS
    old_L = signal_reach(spec.selection.rank_by) + 1 + 21
    new_L = purge_length(spec, index=index, execution_lag_days=lag)
    assert old_L == 43 < new_L == 46

    base, moved, changed = perturbed_bars(panel, spec, index, a - 1, lag)
    old_split = walk_forward(index, 1, lookback=old_L, train_bars=a)[0]
    kept = old_split.test.intersection(base.index)
    assert (moved.loc[kept] != base.loc[kept]).any(), (
        "the old purge length is supposed to leak on this boundary"
    )


def test_cohort_mode_carries_its_hold_days_without_leaking(sp500_panel):
    """Cohort mode replaces the rebalance carry with hold_days: a cohort opened
    on one signal date keeps accruing for hold_days bars whatever the rebalance
    frequency."""
    panel, index, a = leak_window(sp500_panel)
    lag = 1
    spec = leak_spec(index, rebalance="monthly",
                     entry_exit=EntryExit(hold_days=30))
    c = purge_components(spec, index=index, execution_lag_days=lag)
    assert c["carry"] == 30                    # hold_days, not the calendar
    L = c["total"]

    split = walk_forward(index, 1, lookback=L, train_bars=a)[0]
    base, moved, changed = perturbed_bars(panel, spec, index, a - 1, lag)
    assert changed
    kept = split.test.intersection(base.index)
    assert len(kept) > 50
    assert (moved.loc[kept] == base.loc[kept]).all()


def test_a_static_allocation_has_no_signal_to_leak(sp500_panel):
    """The degenerate end of the range: no ranking, no filters, no staleness
    screen. The purge is then just the carry, the lag and the cost bar — but it
    is never zero, because the first scored return still needs yesterday's
    close."""
    panel, index, a = leak_window(sp500_panel)
    tickers = list(panel.columns[:2])
    spec = leak_spec(
        index, rebalance="monthly",
        selection=Selection(static_allocation=[
            {"ticker": tickers[0], "weight": 0.6},
            {"ticker": tickers[1], "weight": 0.4},
        ]),
    )
    c = purge_components(spec, index=index, execution_lag_days=1)
    assert c["signal_reach"] == 0 and c["staleness_reach"] == 0
    assert c["total"] == 0 + 23 + 1 + 1

    split = walk_forward(index, 1, lookback=c["total"], train_bars=a)[0]
    base, moved, changed = perturbed_bars(panel, spec, index, a - 1, 1)
    kept = split.test.intersection(base.index)
    assert (moved.loc[kept] == base.loc[kept]).all()
