"""The vectorized backtest engine.

Point-in-time discipline has two separate halves, and conflating them is the
classic way a backtest flatters itself:

1. SIGNAL timing — weights for signal date t are computed from closes ≤ t.
   This is structural: every signal primitive is backward-looking and a
   property test proves perturbing the future cannot change the past.
2. EXECUTION timing — WHEN you get to trade on that signal. Reading the close
   of t and filling at that same close is a fill you could never have got.
   `execution_lag_days` (default 1, dsl/conventions.DEFAULT_EXECUTION_LAG_DAYS)
   makes this explicit: the signal read at close t is FILLED at close t+lag and
   the position earns from t+lag+1; the capital sits in cash until the fill.
   An earlier version of this docstring claimed a same-day fill was
   structurally impossible while the code hard-coded exactly that. The
   convention is worth tens of percent of Sharpe on short-horizon strategies
   (measured on a 1-day reversal: ~47% of Sharpe in the audit; 0.02 → 0.29 on
   an S&P 500 reproduction here) — i.e. for that class of strategy the fill
   assumption IS the result. The lag is a run-level argument, NOT a DSL
   field, so no spec can buy itself a better fill, but the audit can re-run
   at lag+1 and report the delta.

Two modes:
- standard: the whole portfolio is re-targeted at each rebalance signal date
  and drifts in between (segment-vectorized; no per-day Python loop).
- cohort (entry_exit.hold_days set): each rebalance date spawns a cohort held
  for hold_days trading days; capital is split equally across active cohorts
  each day (see dsl/conventions.py). This is daily bookkeeping by definition,
  so it uses the one permitted per-day loop with numpy row operations.

Costs. Two numbers, deliberately named apart (they used to be the same number,
which under-charged every run by exactly 2×):
- TURNOVER (the statistic the literature means) = 0.5 × Σ|Δw| per execution.
- TRADED NOTIONAL (the cost base) = Σ|Δw| = dollars of stock that actually
  change hands per dollar of NAV. net_ret = gross_ret − traded_notional ×
  bps/10⁴, charged on the first day the new weights earn. `bps` is therefore
  PER SIDE, PER DOLLAR TRADED. Cash is carried as an explicit weight so the
  weight vector sums to 1 and the deltas are exact, but the cash leg is not
  itself charged: deploying 100% of NAV from cash is one buy of 1 × NAV (cost
  1 × bps), while rotating a fully-invested book is a sell plus a buy (cost
  2 × bps). Every weight change is charged, including the cohort expiries and
  spawn decided on the FINAL accrual day, which the old `i + 1 < len(accrual)`
  guard threw away. (The standard path never had an equivalent hole: its
  segments are provably non-empty, so the "empty segment" accumulator that
  used to sit here was unreachable code and has been removed.)

Uninvested cash (partial eligibility, pre-fill days, no qualifying names)
accrues the daily risk-free rate WHEN THE CALLER PASSES ONE (`rf_daily=`);
with no rate it earns 0%, which is what the audit pipeline currently does, and
`EngineResult.rf_used` says which of the two a given run was. Cash at 0%
against an rf-based Sharpe hurdle penalises a strategy on the numerator and
again on the hurdle — see the Cash row of dsl/conventions.py, which documents
the shipped state rather than the intended one.

Benchmarks are NOT run at this lag: engine/benchmark.py passes
BENCHMARK_EXECUTION_LAG_DAYS (0) explicitly, so the engine-computed 60/40 is
timed like the raw-pct_change SPY and QQQ series it sits beside.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dsl.conventions import (
    DEFAULT_EXECUTION_LAG_DAYS,
    INVERSE_VOL_FLOOR_ANNUAL,
    MAX_POSITION_WEIGHT,
    STALENESS_FLAG_SHARE,
    STALENESS_LOOKBACK_DAYS,
    STALENESS_MIN_OBSERVED_DAYS,
    coerce_execution_lag_days,
)
from dsl.schema import Selection, Signal, StrategySpec
from engine.calendar import clip_window, rebalance_signal_dates
from engine.signals import INVERSE_VOL_LOOKBACK, apply_filter_op, compute_signal


class EngineError(Exception):
    """A run-blocking problem, stated in plain English."""


@dataclass
class EngineResult:
    accrual_index: pd.DatetimeIndex        # days on which returns accrue
    gross_returns: pd.Series               # daily portfolio gross returns
    net_returns: dict[float, pd.Series]    # cost bps/side -> daily net returns
    turnover: pd.Series                    # ONE-WAY statistic (0.5·Σ|Δw|), 0 elsewhere
    target_weights: pd.DataFrame           # rows = signal dates, cols = tickers held
    eligible_counts: pd.Series             # per signal date
    holdings: list[dict]                   # per signal date: picks with weights/scores
    data_notes: list[str] = field(default_factory=list)
    annual_turnover: float = 0.0
    signal_dates: pd.DatetimeIndex | None = None
    # Dollars traded per dollar of NAV (Σ|Δw|) — the base the bps rate is
    # charged on. Always exactly 2 × `turnover`; kept as its own series so a
    # reader of the report can never mistake one for the other.
    traded_notional: pd.Series | None = None
    # Trading days between the signal close and the fill close for this run.
    execution_lag_days: int = DEFAULT_EXECUTION_LAG_DAYS
    # Mean weight sitting in cash across the accrual window (0..1). Makes the
    # cash drag visible instead of hiding inside the return series.
    mean_cash_weight: float = 0.0
    # Fill dates actually executed (signal date + execution_lag_days).
    fill_dates: pd.DatetimeIndex | None = None
    # Mean daily risk-free rate actually credited to cash over the accrual
    # window, or None when the caller supplied no rate (cash earned 0%).
    # Without this nothing downstream can state which regime a run was in —
    # and conventions.py must not promise an rf the caller never passed.
    rf_used: float | None = None

    @property
    def effective_start(self) -> pd.Timestamp:
        return self.accrual_index[0]

    @property
    def effective_end(self) -> pd.Timestamp:
        return self.accrual_index[-1]


def _deterministic_top(
    scores: pd.Series, n: int, direction: str
) -> pd.Series:
    """Top/bottom-n with deterministic tie-breaking (Gap #2): sort by score,
    then ticker alphabetically. Identical inputs → identical picks anywhere."""
    df = pd.DataFrame({"score": scores})
    df["ticker"] = df.index
    ascending = direction == "bottom"
    df = df.sort_values(
        ["score", "ticker"], ascending=[ascending, True], kind="mergesort"
    )
    return df["score"].head(n)


def _cap_weights(w: pd.Series) -> pd.Series:
    """Cap any single name at the greater of MAX_POSITION_WEIGHT and twice the
    equal weight, redistributing the excess pro-rata across the uncapped names.

    The relaxation matters: k names cannot all sit below 1/k, so a flat 25% cap
    is infeasible for k ≤ 4 and would silently turn inverse-vol into equal
    weighting for small books. `max(cap, 2/k)` always leaves room and only
    binds once a single name would take more than double its fair share.
    """
    k = len(w)
    cap = max(MAX_POSITION_WEIGHT, 2.0 / k)
    if cap >= 1.0 or w.max() <= cap + 1e-15:
        return w
    w = w.copy()
    capped = pd.Series(False, index=w.index)
    for _ in range(k):
        over = (w > cap + 1e-15) & ~capped
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        capped |= over
        free = ~capped
        if not free.any():
            break                       # cap ≥ 1/k guarantees this is exact
        base = float(w[free].sum())
        if base > 0:
            w[free] += excess * w[free] / base
        else:
            w[free] += excess / int(free.sum())
    return w


def _position_weights(
    picks: pd.Series, weighting: str, n_target: int, vol: pd.Series | None
) -> pd.Series:
    """Weights for the k ≤ n selected names. Sizing baseline is n_target: with
    only k eligible names the portfolio holds k positions sized as if there
    were n (remainder in cash) — the conservative reading, disclosed via
    eligible_counts."""
    k = len(picks)
    if k == 0:
        return pd.Series(dtype=float)
    scale = k / n_target
    if weighting == "equal":
        w = pd.Series(1.0 / k, index=picks.index)
    elif weighting == "rank":
        # Best pick gets weight ∝ k, worst ∝ 1 (picks arrive best-first)
        ranks = np.arange(k, 0, -1, dtype=float)
        w = pd.Series(ranks / ranks.sum(), index=picks.index)
    elif weighting == "inverse_vol":
        assert vol is not None
        # Volatility FLOOR before inverting. The staleness screen only catches
        # feeds that are exactly frozen; a near-frozen one (a name that moves
        # one cent a day) prints ~2 bps of annualized vol, is never flagged,
        # and 1/σ then hands it ~99% of the book with no disclosure at all.
        # Flooring σ caps how much the arithmetic can reward a broken feed.
        inv = 1.0 / vol.loc[picks.index].clip(lower=INVERSE_VOL_FLOOR_ANNUAL)
        w = inv / inv.sum()
        if not np.isfinite(w).all():
            # zero-vol picks are excluded at eligibility AND floored above;
            # this is a backstop for a NaN that got past both
            raise EngineError(
                "inverse-vol weighting hit a zero-volatility ticker — "
                "stale price data in the universe"
            )
        # …and a hard cap, so no single name can take the book even if it
        # legitimately is the calmest thing in the universe.
        w = _cap_weights(w)
    else:
        raise EngineError(f"unknown weighting: {weighting}")
    return w * scale


def _staleness_shares(raw: pd.DataFrame) -> pd.DataFrame:
    """Share of exactly-unchanged closes in the trailing window, or NaN where
    the window holds too few OBSERVED moves to judge.

    A dead feed reprints the same close every day. Such a name has (near-)zero
    measured volatility, (near-)zero drawdown from its high and RSI 100 — it
    wins low-vol, smallest-drawdown, inverse-vol and overbought contests by
    being broken, not by being good. The old guard thresholded a *rolling std*
    at 1e-10; pandas' online variance algorithm leaves ~1e-8 of residue on a
    frozen window, so that guard never once fired. This test is direct, on RAW
    (un-ffilled) closes.

    The adequacy gate is on the DENOMINATOR, not on the row count. Requiring
    STALENESS_LOOKBACK_DAYS *rows* let a freshly-listed name be disqualified on
    a sample of ONE: one observed diff that happened to be zero read as a
    100%-unchanged feed (VRT on 2018-08-03 and LITE on 2015-07-24 in this
    project's own S&P fixture, both real listings with a handful of prints).
    Fewer than STALENESS_MIN_OBSERVED_DAYS observations is "unknown" — NaN —
    and we refuse to disqualify a ticker we cannot yet judge.
    """
    lb = STALENESS_LOOKBACK_DAYS
    diff = raw.diff()
    observed = diff.notna().rolling(lb, min_periods=1).sum()
    unchanged = (diff == 0).rolling(lb, min_periods=1).sum()
    return unchanged / observed.where(observed >= STALENESS_MIN_OBSERVED_DAYS)


def _staleness_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """True where a ticker looks like a halted / stale vendor feed at date t.

    MEMORYLESS BY CONSTRUCTION, and that is a hard constraint, not an accident:
    the state at t may depend on closes in (t − STALENESS_LOOKBACK_DAYS, t] and
    on nothing earlier. validation/splits.py derives its purge length from
    STALENESS_LOOKBACK_DAYS, so any extra memory here silently lengthens how
    far a training-set price reaches into the test set — i.e. it turns a
    turnover problem into a LEAKAGE problem. `test_the_staleness_screen_has_no
    _memory_beyond_its_lookback` pins the bound; the walk-forward module's own
    perturbation probe fails the moment it is broken.

    That constraint is why the false-positive fix here is the THRESHOLD and not
    hysteresis. The audit's measurement was right: with 21 observed diffs, a
    name whose true zero-change share is 0.45 trips a 0.5 cut-off on 32.1% of
    days (exact binomial), which flickers, and each flip is a forced sell plus
    a repurchase charged at 2 × bps. Raising the bar to STALENESS_FLAG_SHARE
    cuts that to 9.1%, and the observation gate above removes the
    sample-of-one disqualifications entirely. What it does NOT remove is
    oscillation for a name whose true share sits ON the threshold — that needs
    hysteresis, which needs the purge length to know about it. See the
    "Stale price feeds" row of dsl/conventions.py.
    """
    share = _staleness_shares(raw)
    return (share > STALENESS_FLAG_SHARE).fillna(False)


class _SelectionEngine:
    """Computes target weights at any signal date from data ≤ that date."""

    def __init__(self, spec: StrategySpec, prices: pd.DataFrame, universe: list[str]):
        self.spec = spec
        self.universe = [t for t in dict.fromkeys(universe) if t in prices.columns]
        self.prices = prices
        self.zero_vol_excluded: set[str] = set()
        # ticker -> lowest annualized vol seen where the inverse-vol floor had
        # to bind. A silent floor is still a silent screen.
        self.vol_floored: dict[str, float] = {}
        # ticker -> the highest unchanged-share seen on a day it was excluded.
        # Kept so the disclosure can tell a FROZEN feed (share 1.0, genuinely
        # zero measured volatility) apart from a merely stale one, instead of
        # telling every excluded name it had "zero measured volatility" when
        # it in fact moved on 10 of 21 sessions.
        self.stale_excluded: dict[str, float] = {}
        sel = spec.selection
        self.static = sel.static_allocation is not None
        if not self.static:
            # Staleness is a UNIVERSE-construction screen, so it is computed
            # once here and applied to the eligibility mask on every path —
            # ranking, filtering and weighting alike. It used to live inside
            # the inverse_vol branch, which left rank-by-volatility and
            # low-vol filters completely unscreened.
            self.stale_share_panel = _staleness_shares(prices[self.universe])
            self.stale_panel = _staleness_panel(prices[self.universe])
            self.rank_panel = compute_signal(prices[self.universe], sel.rank_by)
            self.filter_panels = [
                (compute_signal(prices[self.universe], f.signal), f.op, f.value)
                for f in spec.filters
            ]
            self.vol_panel = None
            if spec.weighting == "inverse_vol":
                self.vol_panel = compute_signal(
                    prices[self.universe],
                    Signal(type="volatility", lookback_days=INVERSE_VOL_LOOKBACK),
                )

    def weights_at(self, t: pd.Timestamp) -> tuple[pd.Series, int, list[dict]]:
        """(weights, eligible_count, picks_detail) using data through t only."""
        if self.static:
            alloc = self.spec.selection.static_allocation or []
            w = pd.Series({p.ticker: p.weight for p in alloc})
            missing = [tk for tk in w.index if pd.isna(self.prices.loc[t].get(tk))]
            if missing:
                raise EngineError(
                    f"static allocation tickers missing price data on "
                    f"{t.date()}: {', '.join(missing)}"
                )
            detail = [
                {"ticker": tk, "weight": float(wt), "score": None}
                for tk, wt in w.items()
            ]
            return w, len(w), detail

        sel = self.spec.selection
        px_row = self.prices.loc[t, self.universe]
        eligible = px_row.notna()
        # Stale feeds are screened BEFORE any ranking or filtering, whatever
        # the weighting scheme (see _staleness_panel). Exclusions are tracked
        # and disclosed — a silent screen is its own bias.
        stale_row = self.stale_panel.loc[t]
        share_row = self.stale_share_panel.loc[t]
        for tk in stale_row[eligible & stale_row].index:
            share = float(share_row.get(tk, float("nan")))
            prev = self.stale_excluded.get(str(tk), 0.0)
            self.stale_excluded[str(tk)] = (
                prev if np.isnan(share) else max(prev, share)
            )
        eligible &= ~stale_row
        scores_row = self.rank_panel.loc[t]
        eligible &= scores_row.notna()
        for panel, op, value in self.filter_panels:
            eligible &= apply_filter_op(panel.loc[t], op, value).fillna(False)
        if self.vol_panel is not None:
            vol_row = self.vol_panel.loc[t]
            # a zero-vol ticker is a stale/halted feed — excluding it silently
            # would be its own bias, so exclusions are tracked and disclosed
            zero_vol = eligible & vol_row.notna() & (vol_row <= 1e-10)
            for tk in zero_vol[zero_vol].index:
                self.zero_vol_excluded.add(str(tk))
            eligible &= vol_row.notna() & (vol_row > 1e-10)

        elig_scores = scores_row[eligible]
        k_eligible = int(len(elig_scores))
        if k_eligible == 0:
            return pd.Series(dtype=float), 0, []

        picks = _deterministic_top(elig_scores.dropna(), sel.n or 1, sel.direction or "top")
        vol_row = self.vol_panel.loc[t] if self.vol_panel is not None else None
        if vol_row is not None:
            picked_vol = vol_row.loc[picks.index]
            low = picked_vol[picked_vol.notna() & (picked_vol < INVERSE_VOL_FLOOR_ANNUAL)]
            for tk, v in low.items():
                prev = self.vol_floored.get(str(tk), float("inf"))
                self.vol_floored[str(tk)] = min(prev, float(v))
        w = _position_weights(picks, self.spec.weighting, sel.n or 1, vol_row)
        detail = [
            {"ticker": tk, "weight": float(w[tk]), "score": float(picks[tk])}
            for tk in picks.index
        ]
        return w, k_eligible, detail


def _rf_vector(
    rf_daily: pd.Series | float | None, index: pd.DatetimeIndex
) -> tuple[pd.Series, str | None]:
    """(daily risk-free rate aligned to `index`, disclosure note or None).

    Accepts a scalar (a flat rate, what the pipeline computes today from the
    mean of ^IRX) or a dated series (the honest time-varying version). Missing
    days carry forward, then fall back to 0 — never NaN, because a NaN here
    would silently poison the whole return series. A supplied series that
    covers none (or only part) of the window is a DATA PROBLEM, so it comes
    back as a note rather than being absorbed into a silent 0%.
    """
    if rf_daily is None:
        return pd.Series(0.0, index=index), None
    if isinstance(rf_daily, bool):
        raise EngineError("rf_daily must be a rate, not a boolean")
    if isinstance(rf_daily, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(rf_daily)):
            raise EngineError("rf_daily must be a finite daily rate")
        return pd.Series(float(rf_daily), index=index), None
    if not isinstance(rf_daily, pd.Series):
        raise EngineError(
            "rf_daily must be a number or a date-indexed pandas Series, not "
            f"{type(rf_daily).__name__}"
        )
    if rf_daily.index.has_duplicates:
        raise EngineError("rf_daily has duplicate dates in its index")
    if not rf_daily.index.is_monotonic_increasing:
        raise EngineError("rf_daily must be sorted by date")
    if not np.isfinite(rf_daily.dropna().to_numpy(dtype=float)).all():
        raise EngineError("rf_daily contains non-finite values")
    out = rf_daily.reindex(index.union(rf_daily.index)).ffill().reindex(index)
    out = out.astype(float)
    uncovered = int(out.isna().sum())
    note = None
    if uncovered == len(index):
        note = ("the supplied risk-free series covers none of the simulated "
                "window — cash earned 0% for every day of this run")
    elif uncovered > 0:
        note = (f"the supplied risk-free series covers {uncovered} fewer day(s) "
                f"than the simulated window — cash earned 0% on those days")
    return out.fillna(0.0), note


def _coerce_lag(execution_lag_days: int | None) -> int:
    """Validate the run-level execution lag, as the conventions module defines
    it. The rule lives in dsl/conventions.py because validation/splits.py sizes
    the walk-forward purge off the same lag — two files deciding separately
    what a legal lag is means the engine can refuse a value the purge accepts.
    """
    try:
        return coerce_execution_lag_days(execution_lag_days)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc


def _check_prices_usable(prices: pd.DataFrame, universe: list[str]) -> None:
    """Refuse a panel with a non-positive or infinite close in the universe.

    A zero close makes P(t)/P(t−1) infinite. The standard path's only guard was
    `gross.isna()`, which inf passes straight through: metrics then reported
    total_return NaN, sharpe NaN and max_drawdown −100% — a confident wrong
    number for a drawdown that never happened. The cohort path silently
    substituted a 1.0 return for the same bad data, so one engine gave two
    different answers to one vendor zero. Both now refuse here, up front, where
    the house NaN-discipline rule says a data problem belongs: surfaced, never
    absorbed. (NaN closes are a different thing — a gap, handled and disclosed.)
    """
    if not universe:
        return
    sub = prices[universe]
    bad = (sub <= 0) | np.isinf(sub)
    if not bad.to_numpy().any():
        return
    cols = [str(c) for c in sub.columns[bad.any(axis=0)]]
    first_col = cols[0]
    first_date = sub.index[bad[first_col]][0]
    raise EngineError(
        f"price panel contains non-positive or infinite closes for "
        f"{', '.join(cols[:8])} (first: {first_col} on {first_date.date()}). "
        "A zero or infinite close makes the daily return infinite, so the run "
        "is refused rather than reported."
    )


def _fill_dates(
    index: pd.DatetimeIndex, sig: pd.DatetimeIndex, lag: int
) -> pd.Series:
    """Signal date → fill date (the close `lag` trading days later).

    Signals whose fill would land on or after the last day of the window are
    dropped: the order would be placed but never earn a single day, so it is
    not executed at all (rather than executed for free). With lag 0 this
    reduces exactly to the old `sig < window_index[-1]` rule.
    """
    pos = index.get_indexer(sig)
    fill_pos = pos + lag
    ok = (pos >= 0) & (fill_pos < len(index) - 1)
    return pd.Series(index[fill_pos[ok]], index=sig[ok])


def run(
    spec: StrategySpec,
    prices: pd.DataFrame,
    universe: list[str],
    *,
    execution_lag_days: int | None = None,
    rf_daily: pd.Series | float | None = None,
) -> EngineResult:
    """Pure function: (spec, price panel, universe tickers) → results.

    `prices` must include lead-in history before window.start for signal
    lookbacks; simulation accrues returns only inside the window.

    execution_lag_days: trading days between the close that produces a signal
        and the close that fills it. None → DEFAULT_EXECUTION_LAG_DAYS (1).
        It is a run-level convention and NOT part of the spec, so no strategy
        can request a cheaper fill; pass it explicitly to re-run a strategy at
        a different lag (the audit's fill-sensitivity check).
    rf_daily: daily risk-free rate credited to uninvested cash — a scalar or a
        dated series. None → 0%, which is only honest when the report also
        uses a 0% hurdle.
    """
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise EngineError("price panel index must be sorted, unique dates")

    lag = _coerce_lag(execution_lag_days)

    window_index = clip_window(prices.index, spec.window.start, spec.window.end)
    if len(window_index) < 3:
        raise EngineError(
            "the requested window contains fewer than 3 trading days of data"
        )
    # Whether the lag leaves room to execute is decided per signal date below,
    # not by a blanket length rule — a 3-day window with a 1-day lag is still
    # a (very short) runnable backtest.

    # Refuse unusable prices BEFORE any signal is computed on them — over the
    # rows this run actually reads, which is the lead-in history the lookbacks
    # consume plus the window itself. Scanning the WHOLE fetched frame (what
    # this did first) let one vendor zero dated after the window end refuse a
    # run whose own data is clean: every panel here is backward-looking, so a
    # close after window_index[-1] cannot reach a single reported number.
    # Refusal semantics are unchanged — only the rows in scope are.
    _check_prices_usable(
        prices.loc[: window_index[-1]],
        [t for t in dict.fromkeys(universe) if t in prices.columns],
    )
    selector = _SelectionEngine(spec, prices, universe)

    # Signal dates. Ranked strategies enter at period ends; static allocations
    # buy on day one and then rebalance at period ends (so buy-and-hold over
    # the window means exactly the window's return — golden test a).
    sig = rebalance_signal_dates(window_index, spec.rebalance)
    if selector.static:
        sig = pd.DatetimeIndex(sorted({window_index[0], *sig}))
    fills = _fill_dates(window_index, sig, lag)
    if len(fills) == 0:
        raise EngineError(
            "no rebalance date in the window leaves room to execute — with a "
            f"{lag}-day execution lag, every signal fills at or after the last "
            "day of the window"
        )

    hold_days = spec.entry_exit.hold_days
    rf, rf_note = _rf_vector(rf_daily, window_index)
    rf_supplied = rf_daily is not None
    if hold_days is not None:
        return _run_cohorts(spec, prices, selector, window_index, fills, hold_days,
                            lag, rf, rf_supplied, rf_note)
    return _run_standard(spec, prices, selector, window_index, fills, lag, rf,
                         rf_supplied, rf_note)


def _data_gap_notes(
    raw: pd.DataFrame, held: list[str], seg_days: pd.DatetimeIndex
) -> list[str]:
    notes = []
    for tk in held:
        gaps = raw.loc[seg_days, tk].isna()
        if gaps.any():
            notes.append(
                f"{tk}: price missing for {int(gaps.sum())} day(s) while held "
                f"(around {gaps.idxmax().date()}); last close carried forward, "
                f"position exited at the next rebalance"
            )
    return notes


def _run_standard(
    spec: StrategySpec,
    prices: pd.DataFrame,
    selector: _SelectionEngine,
    window_index: pd.DatetimeIndex,
    fills: pd.Series,
    lag: int,
    rf: pd.Series,
    rf_supplied: bool,
    rf_note: str | None,
) -> EngineResult:
    prices_ff = prices.ffill()
    sig = pd.DatetimeIndex(fills.index)
    s0 = sig[0]
    accrual = window_index[window_index > s0]
    if len(accrual) == 0:
        raise EngineError("window ends before the first rebalance can execute")

    rf_acc = rf.reindex(accrual).fillna(0.0)
    gross = pd.Series(0.0, index=accrual)
    # traded_notional is Σ|Δw| (the cost base); turnover = half of it (the
    # reported statistic). Both are kept so neither can be mistaken for the
    # other downstream.
    notional = pd.Series(0.0, index=accrual)
    cash_weight = pd.Series(1.0, index=accrual)   # 1.0 until the first fill
    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    eligible_counts: dict[pd.Timestamp, int] = {}
    holdings: list[dict] = []
    data_notes: list[str] = []

    # Days between the first signal and its fill: the capital is not yet
    # invested, so it earns the cash rate. With lag 0 this set is empty.
    pre_fill = accrual[accrual <= fills.iloc[0]]
    if len(pre_fill) > 0:
        gross.loc[pre_fill] = rf_acc.loc[pre_fill].values

    if rf_note:
        data_notes.append(rf_note)

    prev_drifted = pd.Series(dtype=float)  # asset weights before first buy: none
    sig_list = list(sig)
    for k, t in enumerate(sig_list):
        w, n_elig, detail = selector.weights_at(t)
        weights_rows[t] = w
        eligible_counts[t] = n_elig
        holdings.append({"date": str(t.date()), "eligible": n_elig, "picks": detail})

        # The signal read at close t is filled at close f = t + lag; the
        # position earns from the next session onward.
        f = fills.iloc[k]
        seg_end = fills.iloc[k + 1] if k + 1 < len(sig_list) else accrual[-1]
        seg_days = accrual[(accrual > f) & (accrual <= seg_end)]

        # Cost base: dollars traded per dollar of NAV, charged on the first
        # day the new weights earn. The cash leg funds the trade and is not
        # itself charged (see module docstring).
        union = w.index.union(prev_drifted.index)
        delta = w.reindex(union, fill_value=0.0) - prev_drifted.reindex(union, fill_value=0.0)
        traded = float(delta.abs().sum())
        if len(seg_days) == 0:
            # _fill_dates drops any signal whose fill lands on or after the
            # last accrual day, and fill dates strictly increase, so a segment
            # is never empty (replayed across 25 rebalance × lag combinations
            # on the real S&P calendar: 0 occurrences). The previous version
            # carried a `pending_notional` accumulator here and the module
            # docstring advertised it as a fixed finding — it was unreachable
            # code. If this ever fires it is an engine bug, not a free trade.
            raise EngineError(
                f"internal error: the trade filled {f.date()} has no day left "
                "to earn on — execution segment construction is inconsistent"
            )
        notional.loc[seg_days[0]] += traded

        cash = 1.0 - float(w.sum())
        # Cash compounds at the daily risk-free rate alongside the positions.
        cash_path = (1.0 + rf_acc.loc[seg_days]).cumprod() * cash
        if len(w) > 0:
            held = list(w.index)
            base = prices_ff.loc[f, held]          # FILL price, not signal price
            rel = prices_ff.loc[seg_days, held].div(base, axis=1)
            value = rel.mul(w, axis=1).sum(axis=1) + cash_path
            data_notes.extend(_data_gap_notes(prices, held, seg_days))
            end_rel = rel.iloc[-1]
            end_value = float(value.iloc[-1])
            prev_drifted = (w * end_rel) / end_value
        else:
            value = cash_path                       # all cash, earning rf
            prev_drifted = pd.Series(dtype=float)
        cash_weight.loc[seg_days] = (cash_path / value).values
        prev_vals = value.shift(1)
        prev_vals.iloc[0] = 1.0
        gross.loc[seg_days] = (value / prev_vals - 1.0).values

    turnover = 0.5 * notional

    # isna() alone let ±inf through: a zero close produced inf daily returns
    # that reached the report as total_return NaN / max_drawdown −100%.
    # Non-finite is non-finite — refuse, never fabricate.
    if not np.isfinite(gross.to_numpy(dtype=float)).all():
        raise EngineError(
            "engine produced non-finite returns — data alignment bug or an "
            "unusable close in the panel"
        )
    net = {
        float(bps): gross - notional * (bps / 10_000.0)
        for bps in spec.costs.sweep_bps
    }
    data_notes.extend(_zero_vol_notes(selector))
    data_notes.extend(_vol_floor_notes(selector))
    data_notes.extend(_stale_notes(selector))
    years = len(accrual) / 252.0
    result = EngineResult(
        accrual_index=accrual,
        gross_returns=gross,
        net_returns=net,
        turnover=turnover,
        target_weights=pd.DataFrame(weights_rows).T.fillna(0.0),
        eligible_counts=pd.Series(eligible_counts),
        holdings=holdings,
        data_notes=sorted(set(data_notes)),
        annual_turnover=float(turnover.sum() / years) if years > 0 else 0.0,
        signal_dates=sig,
        traded_notional=notional,
        execution_lag_days=lag,
        mean_cash_weight=float(cash_weight.mean()),
        fill_dates=pd.DatetimeIndex(fills.values),
        rf_used=float(rf_acc.mean()) if rf_supplied else None,
    )
    return result


def _zero_vol_notes(selector: _SelectionEngine) -> list[str]:
    if not selector.zero_vol_excluded:
        return []
    names = ", ".join(sorted(selector.zero_vol_excluded)[:8])
    return [
        f"excluded from inverse-vol selection for zero measured volatility "
        f"(stale/halted price feed): {names}"
    ]


def _vol_floor_notes(selector: _SelectionEngine) -> list[str]:
    if not selector.vol_floored:
        return []
    items = sorted(selector.vol_floored.items(), key=lambda kv: kv[1])[:8]
    # 3 significant figures, not 2 decimal places: the feeds this note is about
    # measure thousandths of a percent, and "0.00%" would report a measurement
    # the run did not make.
    names = ", ".join(f"{tk} ({v * 100:.3g}%)" for tk, v in items)
    return [
        f"held with a floored volatility: measured annualized volatility below "
        f"{INVERSE_VOL_FLOOR_ANNUAL:.2%}, which inverse-vol weighting would "
        f"otherwise turn into most of the book. Sized at the "
        f"{INVERSE_VOL_FLOOR_ANNUAL:.2%} floor instead — check the feed: "
        f"{names}"
    ]


def _stale_notes(selector: _SelectionEngine) -> list[str]:
    """Disclose stale-feed exclusions, in two honest groups.

    The single note this replaced told the user every excluded name had "zero
    measured volatility". That is true only of a feed unchanged on 100% of the
    window; a name excluded at 11 of 21 unchanged closes moved on the other 10
    sessions and has perfectly real volatility. A disclosure that overstates
    what was measured is its own credibility problem.
    """
    if not selector.stale_excluded:
        return []

    def _fmt(names: list[str]) -> tuple[str, str]:
        more = len(names) - 8
        return ", ".join(sorted(names)[:8]), (f" and {more} more" if more > 0 else "")

    frozen = [tk for tk, s in selector.stale_excluded.items() if s >= 1.0]
    partial = [tk for tk, s in selector.stale_excluded.items() if s < 1.0]
    notes: list[str] = []
    if frozen:
        names, tail = _fmt(frozen)
        notes.append(
            f"excluded from selection while their price feed was frozen — the "
            f"close did not move once in {STALENESS_LOOKBACK_DAYS} sessions, "
            f"i.e. zero measured volatility (halt, delisting or dead vendor "
            f"feed): {names}{tail}"
        )
    if partial:
        names, tail = _fmt(partial)
        notes.append(
            f"excluded from selection while their price feed looked stale — "
            f"closes unchanged on more than {STALENESS_FLAG_SHARE:.0%} of the "
            f"trailing {STALENESS_LOOKBACK_DAYS} sessions. These names did "
            f"still move on some sessions, so this is a judgement about the "
            f"feed, not a measurement of zero volatility: {names}{tail}"
        )
    return notes


def _run_cohorts(
    spec: StrategySpec,
    prices: pd.DataFrame,
    selector: _SelectionEngine,
    window_index: pd.DatetimeIndex,
    fills: pd.Series,
    hold_days: int,
    lag: int,
    rf: pd.Series,
    rf_supplied: bool,
    rf_note: str | None,
) -> EngineResult:
    """Overlapping cohorts: each signal date spawns a cohort that is FILLED
    `lag` trading days later and then earns returns for hold_days trading days
    from the fill. Capital is split equally across active cohorts each day; the
    portfolio's daily gross return is the mean of active-cohort returns, and
    with no active cohort the whole book is cash earning the risk-free rate.
    Traded notional is measured on the combined weight vector, so overlapping
    names net out (honest cost accounting)."""
    if selector.static:
        raise EngineError("hold_days does not apply to static allocations")

    tickers = selector.universe
    col_ix = {tk: i for i, tk in enumerate(tickers)}
    prices_ff = prices[tickers].ffill()
    px = prices_ff.to_numpy()
    raw_px = prices[tickers].to_numpy()  # UN-filled — for honest gap detection
    date_pos = {d: i for i, d in enumerate(prices.index)}
    sig = pd.DatetimeIndex(fills.index)
    # fill date → the signal date whose weights get bought at that close
    signal_of_fill: dict[pd.Timestamp, pd.Timestamp] = {
        f: t for t, f in fills.items()
    }

    s0 = sig[0]
    accrual = window_index[window_index > s0]
    if len(accrual) == 0:
        raise EngineError("window ends before the first cohort can be established")

    rf_acc = rf.reindex(accrual).fillna(0.0).to_numpy()
    n_assets = len(tickers)
    gross = np.zeros(len(accrual))
    notional = np.zeros(len(accrual))   # Σ|Δw| — the cost base
    cash_weight = np.ones(len(accrual))
    eligible_counts: dict[pd.Timestamp, int] = {}
    holdings: list[dict] = []
    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    data_notes: set[str] = set()
    if rf_note:
        data_notes.add(rf_note)

    # Each cohort: dict(v=np.array asset values, cash=float, age=int)
    cohorts: list[dict] = []

    def spawn(t: pd.Timestamp) -> None:
        w, n_elig, detail = selector.weights_at(t)
        eligible_counts[t] = n_elig
        holdings.append({"date": str(t.date()), "eligible": n_elig, "picks": detail})
        weights_rows[t] = w
        if len(w) == 0:
            # No qualifying names today → no cohort. Capital stays with the
            # remaining active cohorts (or cash if none) — an empty cohort
            # would silently dilute returns.
            return
        v = np.zeros(n_assets)
        for tk, wt in w.items():
            v[col_ix[tk]] = wt
        cohorts.append({"v": v, "cash": 1.0 - float(w.sum()), "age": 0})

    def aggregate_weights() -> np.ndarray:
        if not cohorts:
            return np.zeros(n_assets)
        acc = np.zeros(n_assets)
        for c in cohorts:
            total = c["v"].sum() + c["cash"]
            acc += c["v"] / total
        return acc / len(cohorts)

    # A cohort is established at its FILL close, not at the signal close. With
    # lag 0 the first fill is s0 itself, so the book is in force entering the
    # first accrual day (the pre-fix behaviour); with the default lag of 1 the
    # first fill lands on an accrual day and the book starts empty — those
    # days are cash, earning rf.
    for f in fills.values:
        if f <= s0:
            spawn(signal_of_fill[f])
    w_post = aggregate_weights()  # weights in force entering the first day
    notional[0] = float(np.abs(w_post).sum())  # initial deployment, if any
    prev_pos = date_pos[s0]

    for i, d in enumerate(accrual):
        pos = date_pos[d]
        row_now = px[pos]
        row_prev = px[prev_pos]
        # A missing close (not yet listed, or a gap) marks flat for the day and
        # is disclosed below. A ZERO close is not handled here: it is refused
        # up front by _check_prices_usable, because substituting 1.0 for it
        # here while the standard path emitted inf for the same data made one
        # engine give two different answers to one vendor zero.
        with np.errstate(invalid="ignore", divide="ignore"):
            day_rel = np.where(
                np.isnan(row_now) | np.isnan(row_prev),
                1.0,
                row_now / row_prev,
            )

        # 1. Earn day d in every active cohort. Gap detection uses the RAW
        # (un-filled) prices — checking the ffilled array can never fire
        # (confirmed audit finding: silent absorption).
        rets = []
        nan_now = np.isnan(raw_px[pos])
        for c in cohorts:
            before = c["v"].sum() + c["cash"]
            held_gaps = nan_now & (c["v"] > 0)
            if held_gaps.any():
                for j in np.flatnonzero(held_gaps):
                    data_notes.add(
                        f"{tickers[j]}: price missing while held in a cohort "
                        f"(around {d.date()}); marked at the last close until "
                        f"the cohort expires"
                    )
            c["v"] = c["v"] * day_rel
            # A cohort's own uninvested sleeve (partial eligibility) earns the
            # risk-free rate, like any other idle balance.
            c["cash"] = c["cash"] * (1.0 + rf_acc[i])
            after = c["v"].sum() + c["cash"]
            rets.append(after / before - 1.0)
            c["age"] += 1
        # No active cohort → the entire book is cash, which earns rf, not 0%.
        gross[i] = float(np.mean(rets)) if rets else float(rf_acc[i])
        if cohorts:
            invested = sum(float(c["v"].sum()) / (c["v"].sum() + c["cash"])
                           for c in cohorts) / len(cohorts)
            cash_weight[i] = 1.0 - invested

        # (The trade decided at the previous close was already written into
        # notional[i] at the end of the previous iteration; notional[0]
        # carries the initial deployment in the lag-0 case.)

        # 2. Close-of-day d bookkeeping. Pre-trade weights are NAV-SHARE
        # aggregated: during day d each cohort's capital grew by its own
        # return, so before the daily re-equalization the portfolio is a
        # NAV-weighted mixture of cohort books. Charging |post − pre| on the
        # equal-mean for both sides would let the re-split flow trade for
        # free (confirmed audit finding: uncharged turnover).
        if cohorts:
            growth = np.array([1.0 + r for r in rets])
            shares = growth / growth.sum()
            w_pre = np.zeros(n_assets)
            for share, c in zip(shares, cohorts):
                total = c["v"].sum() + c["cash"]
                w_pre += share * (c["v"] / total)
        else:
            w_pre = np.zeros(n_assets)
        cohorts[:] = [c for c in cohorts if c["age"] < hold_days]
        if d in signal_of_fill:
            spawn(signal_of_fill[d])
        w_post = aggregate_weights()
        traded = float(np.abs(w_post - w_pre).sum())
        if i + 1 < len(accrual):
            notional[i + 1] = traded
        else:
            # Expiries and spawns decided on the final close are real trades;
            # charging them on the same day is the only day left (the old
            # guard dropped them, so the last rebalance was free).
            notional[i] += traded

        prev_pos = pos

    gross_s = pd.Series(gross, index=accrual)
    notional_s = pd.Series(notional, index=accrual)
    turnover_s = 0.5 * notional_s
    if not np.isfinite(gross).all():
        raise EngineError(
            "engine produced non-finite returns — data alignment bug or an "
            "unusable close in the panel"
        )
    net = {
        float(bps): gross_s - notional_s * (bps / 10_000.0)
        for bps in spec.costs.sweep_bps
    }
    data_notes.update(_zero_vol_notes(selector))
    data_notes.update(_vol_floor_notes(selector))
    data_notes.update(_stale_notes(selector))
    years = len(accrual) / 252.0
    return EngineResult(
        accrual_index=accrual,
        gross_returns=gross_s,
        net_returns=net,
        turnover=turnover_s,
        target_weights=pd.DataFrame(weights_rows).T.fillna(0.0),
        eligible_counts=pd.Series(eligible_counts),
        holdings=holdings,
        data_notes=sorted(data_notes),
        annual_turnover=float(turnover_s.sum() / years) if years > 0 else 0.0,
        signal_dates=sig,
        traded_notional=notional_s,
        execution_lag_days=lag,
        mean_cash_weight=float(np.mean(cash_weight)),
        fill_dates=pd.DatetimeIndex(fills.values),
        rf_used=float(np.mean(rf_acc)) if rf_supplied else None,
    )
