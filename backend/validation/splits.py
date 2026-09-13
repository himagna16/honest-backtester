"""Purged, embargoed train/test splits — López de Prado, AFML ch. 7 & 12.

WHY this module exists
----------------------
The audit's only out-of-sample evidence was ONE 70/30 chronological cut with
no gap between the segments (audit/oos.py). With signal lookbacks running to
756 days that cut is not out-of-sample at all: the signal that produces the
first "holdout" return is computed almost entirely from prices inside the
training window, and in cohort mode positions opened before the cut keep
accruing into the holdout for the whole holding period. A leaky split does not
merely add noise — it manufactures the very continuity that makes an overfit
strategy look stable, which is the one thing this product exists to detect.

The fix is López de Prado's pair of rules:

  PURGE   — remove the observations whose label span overlaps the other
            segment. Here the test block's first L bars are removed, because
            those bars' returns were produced by information that lives in the
            training segment. (AFML §7.4, "Purged K-Fold CV".)
  EMBARGO — additionally remove training observations that begin within h bars
            AFTER a test block, because serially-correlated features leak
            forward. López de Prado: "A small value h ≈ .01T, where T is the
            number of bars, often suffices to prevent all leakage." (§7.4.1.)
            The embargo is ONE-SIDED: training data that ends before the test
            block begins was genuinely available at test time and is kept.

Why purging the head of the test block is sufficient on the left-hand side
------------------------------------------------------------------------
Let the test block start at bar `a` and training be `[0, a)`. Trace the actual
engine (engine/backtest.py `_run_standard`): the signal read at the close of
signal date `s` is filled at `f = s + lag`, and the resulting weights earn on
every accrual bar `d` with `f < d <= s' + lag`, where `s'` is the NEXT signal
date. So the oldest signal that can still be earning at bar `d` was formed at

    s >= d - carry - lag,        carry = max(s' - s) over the calendar

and that signal's own window reaches back `reach` bars further, to
`d - carry - lag - reach`. NET returns reach one bar further still, because the
cost Σ|w_target − w_drifted| is charged on the first accrual day of the new
segment and `w_drifted` is the PREVIOUS signal's weights. The bar `d` also
needs the close at `d - 1` for its own price relative. Hence

    L = max(1, reach + carry + lag + 1)     (see `purge_length`)

and the first kept test bar `a + L` touches no price earlier than bar `a` —
the first bar that is NOT training data. The purge zone `[a, a+L)`
simultaneously absorbs the training side's forward label span (a train bar at
`a-1` earns into the purge zone, never into the kept test set), which is why
no separate tail purge of the training block is needed.

Four terms in that formula were wrong or missing before, and dropping any one
of them now makes a training close move a test bar in the engine-level tests:

  * `carry` was a table of MEAN period lengths (21 for monthly). The real
    maximum gap between month-end signal dates on the project's own SPY
    calendar is 23 bars. The carry is now read off the calendar being split
    (`calendar_rebalance_carry`), and the constant fallback is the worst-case
    number of weekdays in the period, never the mean.
  * `lag` was hard-coded as a 1-bar "label span". It is
    `dsl.conventions.DEFAULT_EXECUTION_LAG_DAYS`, and the audit deliberately
    re-runs at lag+1 — which leaked, because the formula only happened to be
    right at lag 1.
  * `reach` ignored the staleness screen. `_staleness_panel` is a rolling
    21-bar test on RAW closes gating eligibility on EVERY selection path, so
    every non-static spec carries an unconditional 21-bar reach whatever its
    signal lookback is. The engine-level evidence for THIS term is the
    `staleness` parametrisation of
    `test_engine_says_no_test_bar_depends_on_a_training_price`, whose ranking
    signal reaches a single bar: on the other boundaries the 21-bar signal
    reach equals the screen's, masks it, and dropping the floor there changes
    nothing the engine can see.
  * the trailing `+ 1` was documented as a "label span" — the bar between a
    signal and the return it earns. That bar is already inside `lag`. What the
    term really buys is the COST bar: engine/backtest.py charges
    Σ|w_target − w_drifted| on the first accrual day of a new segment, and
    `w_drifted` is the previous signal's weights, so a net return reaches one
    bar further back than the gross return under it.

The test suite does NOT verify this arithmetically — an assertion that
re-states the formula cannot fail when the formula is wrong. It verifies it by
running the engine, perturbing the closes on training bars, and asserting the
test-segment returns are bit-identical (`test_validation_splits.py`,
"engine-level leak" section).

Scope
-----
These are index-manipulation functions over a `pd.DatetimeIndex`: no I/O, no
randomness, and no backtest is ever run from here. They do read three things
from the engine layer rather than re-typing them — `engine.calendar
.rebalance_signal_dates` (so the carry is measured on the real schedule
instead of guessed), `engine.signals.INVERSE_VOL_LOOKBACK`, and the staleness
and execution-lag constants from `dsl.conventions`. A purge sized from a
private copy of those numbers is a purge that silently goes stale the day the
engine changes. The caller runs the engine and re-SCORES
the daily return series on the index sets handed back. That is legitimate here
only because the strategy is rule-based rather than fitted — which also means
a caller that merely re-scores a fixed rule will get identical CPCV paths (see
`combinatorial_purged_cv`); path DISPERSION only becomes meaningful once the
caller makes a per-split decision (a parameter choice, a selection) on the
split's training set. That caveat belongs in the report, not hidden here.

Sources
-------
- M. López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018),
  ch. 7 (purging, embargo, purged K-fold) and ch. 12 (CPCV).
- "The 10 Reasons Most Machine Learning Funds Fail" / AFML ch.11-12 summary:
  https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import ceil, comb, log

import numpy as np
import pandas as pd

from dsl.conventions import DEFAULT_EXECUTION_LAG_DAYS, STALENESS_LOOKBACK_DAYS
from dsl.schema import Rebalance, Signal, StrategySpec
from engine.calendar import rebalance_signal_dates
from engine.signals import INVERSE_VOL_LOOKBACK

DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_N_GROUPS = 6
DEFAULT_N_TEST_GROUPS = 2
DEFAULT_N_FOLDS = 6

# López de Prado's rule of thumb for the embargo: h ≈ .01 T (AFML §7.4.1).
EMBARGO_FRACTION = 0.01

# Trading days a position is carried between rebalances when hold_days is not
# set. In standard mode weights persist from one signal date to the next, so
# the return at bar t can still be the product of a signal formed a whole
# rebalance period earlier — the same leak channel as cohort hold_days.
#
# These are WORST-CASE bounds, used only when the caller cannot hand over the
# calendar being split. Because `rebalance_signal_dates` takes the LAST trading
# day of each period, the gap between consecutive signal dates is exactly the
# number of index bars inside the later period — so the bound is the largest
# number of weekdays any such calendar period can contain (holidays only ever
# shrink it): 5 in an ISO week, 23 in a calendar month, 66 in a quarter, 262 in
# a year. The previous table held 21 / 63 / 252, which are the MEAN period
# lengths; on the project's own SPY calendar the observed maxima are 5 / 23 /
# 64 / 253, so that table under-purged by up to two bars and demonstrably
# leaked. Prefer `calendar_rebalance_carry(index, rebalance)`, which reads the
# true maximum off the calendar in hand.
MAX_REBALANCE_CARRY_DAYS: dict[str, int] = {
    "daily": 1, "weekly": 5, "monthly": 23, "quarterly": 66, "annual": 262,
}

# Back-compat alias for the old name; same object, so a mutation of either is
# a mutation of both.
REBALANCE_CARRY_DAYS = MAX_REBALANCE_CARRY_DAYS

# RSI falls back to 14 when lookback_days is omitted (engine/signals.py).
_RSI_DEFAULT_LOOKBACK = 14

# Wilder smoothing (engine/signals.py: ewm(alpha=1/lb, adjust=False)) is an
# INFINITE-impulse-response filter: RSI at t carries a non-zero weight on every
# prior close, so there is no finite reach that makes the dependency exactly
# zero. The share of the EWMA's weight mass lying further back than n bars is
# (1 - 1/lb)^n, so we purge the smallest n with (1 - 1/lb)^n <= eps and state
# the truncation rather than pretending it is not there. eps = 1e-6 → n = 187
# at the default lb = 14, and `rsi_reach` adds the diff bar for 188.
#
# eps IS A BOUND ON EWMA WEIGHT MASS, NOT ON RSI POINTS. It says only that the
# weight sitting further back than the reach is <= 1e-6 of the total; the
# residual MOVE in RSI points at that reach is a different, larger number.
# Measured on the seed-11 random walk in the tests, the +5% one-bar perturbation
# that test applies at rsi_reach(14) = 188 still moves RSI(14) by 9.1e-06 points
# (3.1e-05 worst case over 30 seeds of that generator). For scale, the same
# perturbation moves RSI(14) by 2.58 points at t-15, 0.49 at t-40, 0.014 at
# t-80, 7.9e-08 at t-250. (The sign matters at this resolution: the -5% twin
# gives 1.96 / 0.40 / 0.019 / 6.8e-06 / 7.6e-08, so a "±5%" figure would be two
# different series printed as one.)
RSI_TAIL_MASS_EPS = 1e-6


class SplitError(ValueError):
    """A requested split cannot be built without leaking.

    Carries a machine-readable `reason` and `details` so a caller can turn the
    refusal into an audit Finding instead of silently falling back to a leaky
    split. Never caught-and-ignored inside this module.
    """

    def __init__(self, message: str, *, reason: str, **details) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details


# ---------------------------------------------------------------------------
# Purge length from a spec
# ---------------------------------------------------------------------------

def rsi_reach(lookback: int, eps: float = RSI_TAIL_MASS_EPS) -> int:
    """Effective reach of Wilder-smoothed RSI — an IIR filter, so TRUNCATED.

    engine/signals.py computes RSI from `ewm(alpha=1/lb, adjust=False)`, whose
    value at t depends on EVERY prior close; `lookback` is not a reach at all
    (three lookbacks back, one perturbed close still moves RSI(14) by half a
    point — plenty to reorder a top-n ranking). The weight mass sitting further
    back than n bars is (1 - 1/lb)^n, so we return the smallest n with
    (1 - 1/lb)^n <= eps, plus one bar because the filter is fed
    `prices.diff()` and the value at t-n consumes the close at t-n-1.

    This is the one branch where the purge does not make the dependency
    exactly zero; `RSI_TAIL_MASS_EPS` is the stated truncation threshold.
    """
    lb = int(lookback)
    if lb <= 1:
        # alpha = 1: no smoothing at all, the value at t is just today's diff.
        return 2
    n = ceil(log(eps) / log(1.0 - 1.0 / lb))
    return int(n) + 1


def signal_reach(signal: Signal) -> int:
    """How many bars BEFORE date t the signal value at t depends on.

    Mirrors engine/signals.py formula-by-formula and rounds up where the two
    disagree by a bar (drawdown_from_high truly reaches back lookback-1, we
    say lookback). Over-stating the reach purges one extra bar; under-stating
    it leaks, so the asymmetry is deliberate.

    `skip_days` is a field of the return/daily_return formulas only — the
    engine ignores it everywhere else, so it is added only there.
    """
    t = signal.type
    if t in ("return", "daily_return"):
        # prices at t-skip and t-skip-lookback.
        return int(signal.lookback_days or 0) + int(signal.skip_days)
    if t == "sma_cross":
        # fast/slow rolling means: value at t uses prices t-max+1..t.
        return max(int(signal.fast or 0), int(signal.slow or 0))
    if t == "rsi":
        return rsi_reach(signal.lookback_days or _RSI_DEFAULT_LOOKBACK)
    # volatility: pct_change over a rolling window → prices t-lookback..t.
    # drawdown_from_high: rolling max → t-lookback+1..t, rounded up a bar.
    return int(signal.lookback_days or 0)


def calendar_rebalance_carry(
    index: pd.DatetimeIndex, rebalance: Rebalance | str
) -> int:
    """The LARGEST gap, in bars, between consecutive signal dates on `index`.

    This is the number the purge actually needs, and it is a property of the
    calendar, not of an average month. `engine.calendar.rebalance_signal_dates`
    picks the last trading day of each period, so the gap between consecutive
    signal dates equals the bar count of the later period — 23 for a long
    month, 64 for a long quarter, 253 for a long year on the project's SPY
    calendar, against the 21 / 63 / 252 the old constant table claimed.

    Raises `SplitError(reason="signal_dates_not_on_calendar")` if the schedule
    returns a date that is not a bar of `index` — see the guard below.
    """
    idx = _validated_index(index)
    freq = str(rebalance)
    if freq not in MAX_REBALANCE_CARRY_DAYS:  # pragma: no cover - closed Literal
        raise SplitError(
            f"unknown rebalance frequency '{rebalance}' — cannot size the "
            f"purge, and guessing would risk a leaky split",
            reason="unknown_rebalance", rebalance=freq,
        )
    if freq == "daily":
        return 1
    sig = pd.DatetimeIndex(rebalance_signal_dates(idx, freq))
    pos = idx.get_indexer(sig)
    # A signal date the calendar does not contain means the schedule and the
    # index disagree, and every gap measured from such a position is fiction.
    # This is NOT hypothetical: engine.calendar.rebalance_signal_dates ends with
    # pd.DatetimeIndex(last_days.values), which drops the tzinfo, so on a
    # tz-aware calendar EVERY signal date misses and `gaps` collapses to zeros —
    # a carry of 1 where the true answer is 23, i.e. a silently short purge.
    # Refuse; a wrong carry is exactly the leak this module exists to stop.
    missing = int((pos == -1).sum())
    if missing:
        raise SplitError(
            f"{missing} of {len(pos)} '{freq}' rebalance signal dates are not "
            f"bars of the trading calendar — the rebalance carry cannot be "
            f"measured, and guessing it would under-purge the split",
            reason="signal_dates_not_on_calendar", rebalance=freq,
            n_bars=len(idx), n_signal_dates=len(pos), n_missing=missing,
            index_tz=str(idx.tz), signal_tz=str(sig.tz),
        )
    gaps = np.diff(pos).tolist() if len(pos) > 1 else []
    return int(max([1, *gaps]))


def purge_components(
    spec: StrategySpec,
    *,
    index: pd.DatetimeIndex | None = None,
    execution_lag_days: int | None = None,
    include_rebalance_carry: bool = True,
    extra_reach: int = 0,
) -> dict[str, int]:
    """Breakdown of the purge length L, so the report can disclose it.

    L = max(1, reach + carry + execution_lag + cost_charge)
    reach = max(signal_reach, staleness_reach, extra_reach)

    - signal_reach: the longest lookback in the spec (ranking signal, every
      filter signal, and the implicit 63-day volatility signal that
      inverse_vol weighting computes in engine/backtest.py).
    - staleness_reach: the stale-feed screen in engine/backtest.py is a rolling
      STALENESS_LOOKBACK_DAYS window on RAW closes applied to eligibility on
      EVERY selection path, so it is an unconditional floor on the reach of any
      non-static spec — including the short-horizon specs whose own signal
      reach is far below it. `raw.diff()` inside a 21-bar window touches the
      close 21 bars back, hence the reach is STALENESS_LOOKBACK_DAYS, not 20.
    - carry: how long a position formed on one signal date keeps earning —
      hold_days in cohort mode, otherwise the largest gap between rebalance
      signal dates (read off `index` when given, else the worst-case bound in
      MAX_REBALANCE_CARRY_DAYS).
    - execution_lag: the signal read at t is filled at t+lag and earns from
      t+lag+1, so the lag adds directly to the maximum age of a live signal.
      Defaults to dsl.conventions.DEFAULT_EXECUTION_LAG_DAYS; a caller re-running
      the fill-sensitivity check at lag+1 MUST pass the lag it actually ran at.
    - cost_charge: one further bar, because NET returns reach a bar beyond
      gross ones. engine/backtest.py charges Σ|w_target − w_drifted| on the
      FIRST accrual day of the new segment, and `w_drifted` is the PREVIOUS
      signal's weights — so the cost printed at bar s+lag+1 still carries the
      signal formed at s−carry. Measured: with a 21-bar signal, daily
      rebalance and lag 1, perturbing the last training bar leaves gross
      returns untouched from bar a+23 on but still moves the NET return at
      a+23 by 3.3e-04; at a+24 (= a+L) it is exactly zero.
    - extra_reach: caller-supplied allowance for an engine dependency this
      module cannot bound on its own. See the KNOWN RESIDUAL note below.

    The floor of 1 covers the price relative itself: the return at the first
    kept test bar always consumes the close of the bar before it.

    THE STALENESS FLOOR RESTS ON A CONTRACT. `staleness_reach` is only equal to
    STALENESS_LOOKBACK_DAYS while the engine's screen is MEMORYLESS — while the
    verdict at t is a function of the closes in (t - 21, t] and nothing
    earlier. dsl/conventions.py states that requirement and names this module
    as the reason (it is why the flicker fix there is a single raised threshold
    rather than textbook hysteresis: a flag/release state machine with an ffill
    has no finite reach at all, so a training close could flip a flag that
    survives arbitrarily far into the test set). If the screen ever grows
    memory, this function's floor becomes wrong and the engine must publish a
    STALENESS_REACH_DAYS for it to consume. `extra_reach` is the stopgap for
    that day: a caller-supplied allowance, reported as its own term so the
    disclosure can never pass it off as something this module measured.
    """
    lag = (DEFAULT_EXECUTION_LAG_DAYS if execution_lag_days is None
           else int(execution_lag_days))
    if lag < 0:
        raise SplitError(
            f"execution_lag_days must be ≥ 0, got {lag} — a fill in the past "
            f"is not a thing the engine can run",
            reason="bad_execution_lag", execution_lag_days=lag,
        )

    reaches = [0]
    sel = spec.selection
    if sel.rank_by is not None:
        reaches.append(signal_reach(sel.rank_by))
    for f in spec.filters:
        reaches.append(signal_reach(f.signal))
    if not spec.is_static and spec.weighting == "inverse_vol":
        reaches.append(
            signal_reach(Signal(type="volatility",
                                lookback_days=INVERSE_VOL_LOOKBACK))
        )
    signal_part = max(reaches)
    stale_part = 0 if spec.is_static else int(STALENESS_LOOKBACK_DAYS)

    hold_days = spec.entry_exit.hold_days
    if hold_days is not None:
        carry = int(hold_days)
    elif include_rebalance_carry:
        carry = (_rebalance_carry(spec.rebalance) if index is None
                 else calendar_rebalance_carry(index, spec.rebalance))
    else:
        carry = 0

    extra = int(extra_reach)
    if extra < 0:
        raise SplitError(f"extra_reach must be ≥ 0, got {extra}",
                         reason="bad_extra_reach", extra_reach=extra)
    reach = max(signal_part, stale_part, extra)
    return {
        "signal_reach": signal_part,
        "staleness_reach": stale_part,
        "extra_reach": extra,
        "reach": reach,
        "carry": carry,
        "execution_lag": lag,
        "cost_charge": 1,
        "total": max(1, reach + carry + lag + 1),
    }


def purge_length(
    spec: StrategySpec,
    *,
    index: pd.DatetimeIndex | None = None,
    execution_lag_days: int | None = None,
    include_rebalance_carry: bool = True,
    extra_reach: int = 0,
) -> int:
    """Bars that must be purged at the head of every test block for `spec`.

    AFML §7.4: the purge must cover the whole label span of the observations
    that straddle the boundary. See `purge_components` for the terms. Pass the
    `index` the engine will actually run over whenever you have it — the
    constant carry table is a worst-case fallback, not the preferred path.
    """
    return purge_components(
        spec, index=index, execution_lag_days=execution_lag_days,
        include_rebalance_carry=include_rebalance_carry,
        extra_reach=extra_reach,
    )["total"]


def _rebalance_carry(rebalance: Rebalance | str) -> int:
    try:
        return MAX_REBALANCE_CARRY_DAYS[str(rebalance)]
    except KeyError:  # pragma: no cover - schema is a closed Literal
        raise SplitError(
            f"unknown rebalance frequency '{rebalance}' — cannot size the "
            f"purge, and guessing would risk a leaky split",
            reason="unknown_rebalance", rebalance=str(rebalance),
        ) from None


def default_embargo(n_bars: int) -> int:
    """h = max(1, int(.01 T)) — AFML §7.4.1's rule of thumb, floored at one
    bar so an embargo is never silently switched off on a short history."""
    if n_bars <= 0:
        raise SplitError("cannot size an embargo on an empty index",
                         reason="empty_index", n_bars=int(n_bars))
    return max(1, int(EMBARGO_FRACTION * n_bars))


# ---------------------------------------------------------------------------
# Split containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """One purged, embargoed train/test partition of a trading calendar.

    `train`, `test`, `purged` and `embargoed` are disjoint by construction;
    `purged` and `embargoed` are the bars deliberately thrown away, kept so
    the report can say how much history the honesty cost.
    """

    label: str
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purged: pd.DatetimeIndex
    embargoed: pd.DatetimeIndex
    # RAW block bounds, purged head included — the slice of calendar that was
    # taken out of training, NOT the slice that gets scored.
    test_blocks: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    # The same blocks after the purge: the bars that are actually scored.
    scored_blocks: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    lookback: int
    embargo: int
    # Training bars whose own `lookback`-bar dependency window reaches into a
    # test block. Zero for a trailing test block (nothing trains after it);
    # positive in CPCV, where training resumes `embargo` bars after an interior
    # block and the embargo (h ≈ .01T) is far shorter than the purge. Disclosed
    # rather than hidden, because a caller that FITS on split.train is exposed
    # to exactly these bars.
    contaminated_train_bars: int = 0

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def as_dict(self) -> dict:
        """JSON-safe summary for the report payload (no Timestamps)."""
        return {
            "label": self.label,
            "train_days": len(self.train),
            "test_days": len(self.test),
            "purged_days": len(self.purged),
            "embargoed_days": len(self.embargoed),
            "purge_bars": self.lookback,
            "embargo_bars": self.embargo,
            "contaminated_train_bars": self.contaminated_train_bars,
            "train_start": _d(self.train[0]) if len(self.train) else None,
            "train_end": _d(self.train[-1]) if len(self.train) else None,
            "test_start": _d(self.test[0]) if len(self.test) else None,
            "test_end": _d(self.test[-1]) if len(self.test) else None,
            # Named unambiguously: "block_bounds" still includes the purged
            # head, "scored_bounds" is the holdout that actually counts. The
            # old single "test_blocks" key reported the former while
            # test_start/test_end reported the latter, which overstated the
            # holdout by `lookback` bars per block.
            "block_bounds": [[_d(a), _d(b)] for a, b in self.test_blocks],
            "scored_bounds": [[_d(a), _d(b)] for a, b in self.scored_blocks],
        }


@dataclass(frozen=True)
class PathSegment:
    """One group's contribution to a CPCV path, with its provenance: the
    split whose TRAINING set a caller must use when it makes a fitted or
    selected decision for these dates."""

    group: int
    split_id: int
    index: pd.DatetimeIndex


@dataclass(frozen=True)
class CPCVPath:
    """One complete pass over the timeline, assembled from k-group test
    blocks drawn from φ different splits."""

    path_id: int
    segments: tuple[PathSegment, ...]

    @property
    def index(self) -> pd.DatetimeIndex:
        """All dates on the path, in calendar order. Note the gaps: each
        group's first `lookback` bars were purged.

        Built by appending the segment indexes rather than concatenating their
        `.values`: the raw-values route drops the timezone (a tz-aware index
        silently becomes UTC-naive) and the index name, which made the
        documented `net.loc[path.index]` integration call raise KeyError on any
        tz-aware calendar.
        """
        parts = [s.index for s in self.segments if len(s.index)]
        if not parts:
            return pd.DatetimeIndex([], name="Date")
        if len(parts) == 1:
            return parts[0].sort_values()
        return parts[0].append(parts[1:]).sort_values()

    def as_dict(self) -> dict:
        idx = self.index
        return {
            "path_id": self.path_id,
            "days": len(idx),
            "start": _d(idx[0]) if len(idx) else None,
            "end": _d(idx[-1]) if len(idx) else None,
            "splits": [s.split_id for s in self.segments],
        }


@dataclass(frozen=True)
class CPCVDesign:
    """The full combinatorial purged CV design: C(N,k) splits recombined into
    φ[N,k] = C(N-1,k-1) backtest paths."""

    n_groups: int
    n_test_groups: int
    group_bounds: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    splits: tuple[Split, ...]
    paths: tuple[CPCVPath, ...]
    lookback: int
    embargo: int

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def as_dict(self) -> dict:
        return {
            "n_groups": self.n_groups,
            "n_test_groups": self.n_test_groups,
            "n_splits": self.n_splits,
            "n_paths": self.n_paths,
            "purge_bars": self.lookback,
            "embargo_bars": self.embargo,
            "group_bounds": [[_d(a), _d(b)] for a, b in self.group_bounds],
            "paths": [p.as_dict() for p in self.paths],
        }


def n_cpcv_splits(n_groups: int, n_test_groups: int) -> int:
    """C(N, k) — AFML ch.12 writes it C(N, N-k), which is the same number."""
    return comb(n_groups, n_test_groups)


def n_cpcv_paths(n_groups: int, n_test_groups: int) -> int:
    """φ[N,k] = (k/N)·C(N,k) = C(N-1, k-1) — AFML ch.12. N=6,k=2 → 5."""
    return comb(n_groups - 1, n_test_groups - 1)


# ---------------------------------------------------------------------------
# Public split constructors
# ---------------------------------------------------------------------------

def purged_split(
    index: pd.DatetimeIndex,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    *,
    lookback: int,
    embargo: int | None = None,
) -> Split:
    """One chronological train/test cut with the boundary purged.

    Train is `[0, a)` where `a = int(T·train_fraction)`; the test block is
    `[a, T)` with its first `lookback` bars purged. The embargo has nothing to
    protect here (no training data follows a trailing test block), so
    `embargoed` comes back empty — that is correct, not a bug: the embargo is
    one-sided (AFML §7.4.1) and only bites when training resumes after a test
    block, as it does in `combinatorial_purged_cv`.

    `lookback` is keyword-only and required on purpose: a forgotten purge is
    exactly the defect this module exists to remove.
    """
    idx = _validated_index(index)
    n = len(idx)
    if not 0.0 < train_fraction < 1.0:
        raise SplitError(
            f"train_fraction must be strictly between 0 and 1, got "
            f"{train_fraction}", reason="bad_train_fraction",
            train_fraction=float(train_fraction),
        )
    # Integer arithmetic, not int(n * frac): binary floats put the boundary one
    # bar early for 143 of the 5990 index lengths in [10, 6000) at frac=0.70
    # (n=90 → 62 instead of 63, likewise 170, 180, 330, 340, …). A split
    # boundary must be a deterministic rule, not a rounding accident.
    a = (n * round(train_fraction * 10_000)) // 10_000
    if a < 1:
        raise SplitError(
            f"a {train_fraction:.0%} training share of {n} bars leaves no "
            f"training data", reason="empty_train", n_bars=n,
        )
    lookback, embargo = _validated_gaps(lookback, embargo, n)
    return _build_split(idx, [(a, n)], lookback, embargo,
                        label=f"purged_{int(train_fraction * 100)}_"
                              f"{100 - int(train_fraction * 100)}")


def walk_forward(
    index: pd.DatetimeIndex,
    n_folds: int = DEFAULT_N_FOLDS,
    *,
    lookback: int,
    embargo: int | None = None,
    anchored: bool = True,
    train_bars: int | None = None,
) -> list[Split]:
    """Sequential OOS folds, each purged, with training strictly in the past.

    AFML ch.7 credits walk-forward with "a clear historical interpretation"
    and warns that it tests a single historical path; run it alongside
    `combinatorial_purged_cv`, not instead of it.

    The first `train_bars` bars (default T // (n_folds + 1)) are the initial
    training block; the rest is partitioned into `n_folds` contiguous test
    folds, the last absorbing the remainder. ANCHORED keeps training at
    `[0, fold_start)` so history grows; ROLLING keeps it at the fixed
    `train_bars` immediately before the fold. Because training always precedes
    the fold, the embargo is inert here (nothing to protect after the fold) —
    the purge at the fold head is what does the work.

    Returns the folds in chronological order.
    """
    idx = _validated_index(index)
    n = len(idx)
    if n_folds < 1:
        raise SplitError(f"n_folds must be ≥ 1, got {n_folds}",
                         reason="bad_n_folds", n_folds=int(n_folds))
    lookback, embargo = _validated_gaps(lookback, embargo, n)

    start = n // (n_folds + 1) if train_bars is None else int(train_bars)
    if start < 1:
        raise SplitError(
            f"{n} bars split into {n_folds + 1} blocks leaves no initial "
            f"training block", reason="empty_train",
            n_bars=n, n_folds=int(n_folds),
        )
    if start >= n:
        raise SplitError(
            f"train_bars={start} leaves no room for {n_folds} test fold(s) in "
            f"{n} bars", reason="no_test_room", n_bars=n, train_bars=start,
        )
    fold_len = (n - start) // n_folds
    if fold_len <= lookback:
        raise SplitError(
            f"{n_folds} walk-forward folds over {n - start} test bars give "
            f"{fold_len} bars per fold, but {lookback} bars must be purged "
            f"from each fold's head — use fewer folds, a longer window, or a "
            f"shorter lookback",
            reason="fold_shorter_than_purge", n_bars=n, n_folds=int(n_folds),
            fold_bars=int(fold_len), required_bars=int(lookback) + 1,
        )

    splits: list[Split] = []
    for i in range(n_folds):
        a = start + i * fold_len
        b = n if i == n_folds - 1 else a + fold_len
        lo = 0 if anchored else max(0, a - start)
        allowed = np.zeros(n, dtype=bool)
        allowed[lo:a] = True
        splits.append(_build_split(
            idx, [(a, b)], lookback, embargo,
            label=f"{'anchored' if anchored else 'rolling'}_fold_{i + 1}",
            allowed_train=allowed,
        ))
    return splits


def combinatorial_purged_cv(
    index: pd.DatetimeIndex,
    n_groups: int = DEFAULT_N_GROUPS,
    n_test_groups: int = DEFAULT_N_TEST_GROUPS,
    *,
    lookback: int,
    embargo: int | None = None,
) -> CPCVDesign:
    """CPCV (AFML ch.12): C(N,k) purged splits recombined into φ[N,k] paths.

    The index is cut into N contiguous groups — groups 1..N-1 of size
    ⌊T/N⌋, group N taking the remainder — and every combination of k groups is
    tested in turn, each block purged at the head and embargoed after. Each
    group is tested in exactly φ = C(N-1,k-1) splits, so the splits recombine
    into φ complete passes over the timeline: "a key advantage of CPCV is that
    it allows us to derive a distribution of Sharpe ratios, as opposed to a
    single (likely overfit) Sharpe ratio estimate" (AFML ch.12).

    Worked case from the source, asserted in the tests: N=6, k=2 → 15 splits
    and 5 paths.

    Path assembly: group g's j-th testing split contributes g to path j. A
    caller that only re-scores a fixed rule will find every path identical —
    the paths differ only to the extent the caller lets each split's TRAINING
    set change what it does on that split's test groups. `PathSegment` carries
    the split id for exactly that purpose.
    """
    idx = _validated_index(index)
    n = len(idx)
    if n_groups < 2:
        raise SplitError(f"n_groups must be ≥ 2, got {n_groups}",
                         reason="bad_n_groups", n_groups=int(n_groups))
    if not 1 <= n_test_groups < n_groups:
        raise SplitError(
            f"n_test_groups must be ≥ 1 and < n_groups ({n_groups}), got "
            f"{n_test_groups} — every split needs at least one training group",
            reason="bad_n_test_groups", n_groups=int(n_groups),
            n_test_groups=int(n_test_groups),
        )
    lookback, embargo = _validated_gaps(lookback, embargo, n)

    size = n // n_groups
    if size <= lookback:
        raise SplitError(
            f"{n} bars split into {n_groups} groups give {size} bars per "
            f"group, but {lookback} bars must be purged from each group's "
            f"head — use fewer groups, a longer window, or a shorter lookback",
            reason="fold_shorter_than_purge", n_bars=n,
            n_groups=int(n_groups), fold_bars=int(size),
            required_bars=int(lookback) + 1,
        )
    bounds = [(i * size, (i + 1) * size) for i in range(n_groups - 1)]
    bounds.append(((n_groups - 1) * size, n))

    combos = list(itertools.combinations(range(n_groups), n_test_groups))
    splits: list[Split] = []
    testing_splits: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for split_id, groups in enumerate(combos):
        blocks = [bounds[g] for g in groups]
        splits.append(_build_split(
            idx, blocks, lookback, embargo,
            label="cpcv_" + "+".join(f"g{g + 1}" for g in groups),
        ))
        for g in groups:
            testing_splits[g].append(split_id)

    n_paths = n_cpcv_paths(n_groups, n_test_groups)
    paths: list[CPCVPath] = []
    for p in range(n_paths):
        segments = []
        for g in range(n_groups):
            split_id = testing_splits[g][p]
            a, b = bounds[g]
            segments.append(PathSegment(group=g, split_id=split_id,
                                        index=idx[a + lookback:b]))
        paths.append(CPCVPath(path_id=p, segments=tuple(segments)))

    return CPCVDesign(
        n_groups=n_groups, n_test_groups=n_test_groups,
        group_bounds=tuple((idx[a], idx[b - 1]) for a, b in bounds),
        splits=tuple(splits), paths=tuple(paths),
        lookback=lookback, embargo=embargo,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_split(
    index: pd.DatetimeIndex,
    blocks: list[tuple[int, int]],
    lookback: int,
    embargo: int,
    *,
    label: str,
    allowed_train: np.ndarray | None = None,
) -> Split:
    """Assemble one Split from test blocks given as [start, stop) positions.

    Applies the purge to each block's head and the embargo to the bars that
    follow it, then hands everything left over (intersected with
    `allowed_train`, which walk-forward uses to keep training in the past) to
    the training set. Raises rather than returning a leaky or empty split.
    """
    n = len(index)
    in_block = np.zeros(n, dtype=bool)
    purged_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    prev_stop = 0
    for a, b in blocks:
        if not 0 <= a < b <= n:
            raise SplitError(
                f"test block [{a}, {b}) is outside the {n}-bar index",
                reason="block_out_of_range", start=int(a), stop=int(b),
                n_bars=n,
            )
        if a < prev_stop:
            raise SplitError(
                f"test blocks must be sorted and disjoint; [{a}, {b}) "
                f"overlaps the previous block",
                reason="overlapping_blocks", start=int(a), stop=int(b),
            )
        prev_stop = b
        if b - a <= lookback:
            raise SplitError(
                f"test block {label} holds {b - a} bars but {lookback} must "
                f"be purged from its head — the fold would be empty or leaky",
                reason="fold_shorter_than_purge", fold_bars=int(b - a),
                required_bars=int(lookback) + 1, label=label,
            )
        in_block[a:b] = True
        purged_mask[a:a + lookback] = True
        test_mask[a + lookback:b] = True

    embargo_mask = np.zeros(n, dtype=bool)
    for _, b in blocks:
        embargo_mask[b:min(b + embargo, n)] = True
    # A bar inside another test block is already excluded; it is not an
    # embargo casualty, so don't double-count it in the disclosure.
    embargo_mask &= ~in_block

    train_mask = ~(in_block | embargo_mask)
    if allowed_train is not None:
        train_mask &= allowed_train
    if not train_mask.any():
        raise SplitError(
            f"split {label} leaves no training bars after purging "
            f"{int(purged_mask.sum())} and embargoing "
            f"{int(embargo_mask.sum())}",
            reason="empty_train", label=label, n_bars=n,
        )

    # How many training bars have a dependency window reaching into a test
    # block. On the pre-test side this is zero by construction (the purge is
    # at the block head); it is the POST-block side that the embargo only
    # partially covers, so this number is the residual the report must state.
    train_pos = np.flatnonzero(train_mask)
    contaminated = 0
    if lookback > 0 and len(train_pos):
        block_cum = np.cumsum(in_block.astype(np.int64))
        lo = np.maximum(train_pos - lookback, 0)
        # bars in [lo, p] that sit inside any test block
        inside = block_cum[train_pos] - block_cum[lo] + in_block[lo]
        contaminated = int(np.count_nonzero(inside > 0))

    return Split(
        label=label,
        train=index[train_mask],
        test=index[test_mask],
        purged=index[purged_mask],
        embargoed=index[embargo_mask],
        test_blocks=tuple((index[a], index[b - 1]) for a, b in blocks),
        scored_blocks=tuple(
            (index[a + lookback], index[b - 1]) for a, b in blocks
        ),
        lookback=lookback,
        embargo=embargo,
        contaminated_train_bars=contaminated,
    )


def _validated_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """A calendar that is unsorted, duplicated or empty is a data problem, and
    data problems surface here rather than producing a quietly wrong split."""
    if not isinstance(index, pd.DatetimeIndex):
        raise SplitError(
            f"expected a pd.DatetimeIndex of trading days, got "
            f"{type(index).__name__}", reason="not_a_datetime_index",
        )
    if len(index) == 0:
        raise SplitError("cannot split an empty trading calendar",
                         reason="empty_index", n_bars=0)
    if index.has_duplicates:
        raise SplitError(
            "trading calendar contains duplicate dates — the split would "
            "score some bars twice", reason="duplicate_dates",
            n_bars=len(index),
        )
    if not index.is_monotonic_increasing:
        raise SplitError(
            "trading calendar is not sorted ascending — chronological splits "
            "are meaningless on a shuffled index", reason="unsorted_index",
            n_bars=len(index),
        )
    return index


def _validated_gaps(
    lookback: int, embargo: int | None, n_bars: int
) -> tuple[int, int]:
    lookback = int(lookback)
    if lookback < 0:
        raise SplitError(f"lookback must be ≥ 0, got {lookback}",
                         reason="bad_lookback", lookback=lookback)
    embargo = default_embargo(n_bars) if embargo is None else int(embargo)
    if embargo < 0:
        raise SplitError(f"embargo must be ≥ 0, got {embargo}",
                         reason="bad_embargo", embargo=embargo)
    return lookback, embargo


def _d(ts: pd.Timestamp) -> str:
    return str(pd.Timestamp(ts).date())
