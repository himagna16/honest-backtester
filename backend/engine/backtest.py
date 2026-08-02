"""The vectorized backtest engine.

Point-in-time discipline is structural, not a convention the caller follows:
weights chosen at signal date t are computed from closes ≤ t and earn returns
from t+1 onward. The shift lives HERE — the DSL has no "execute at signal
price" primitive, so no spec can request a same-day fill.

Two modes:
- standard: the whole portfolio is re-targeted at each rebalance signal date
  and drifts in between (segment-vectorized; no per-day Python loop).
- cohort (entry_exit.hold_days set): each rebalance date spawns a cohort held
  for hold_days trading days; capital is split equally across active cohorts
  each day (see dsl/conventions.py). This is daily bookkeeping by definition,
  so it uses the one permitted per-day loop with numpy row operations.

Costs (Section 3, verbatim): turnover = 0.5 × Σ|w_target − w_drifted| at each
execution, net_ret = gross_ret − turnover × bps/10⁴ charged on the execution
day. Initial deployment from cash is charged the same way (you pay to enter —
the honest choice; documented in DECISIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dsl.schema import Selection, Signal, StrategySpec
from engine.calendar import clip_window, rebalance_signal_dates
from engine.signals import INVERSE_VOL_LOOKBACK, apply_filter_op, compute_signal


class EngineError(Exception):
    """A run-blocking problem, stated in plain English."""


@dataclass
class EngineResult:
    accrual_index: pd.DatetimeIndex        # days on which returns accrue
    gross_returns: pd.Series               # daily portfolio gross returns
    net_returns: dict[float, pd.Series]    # cost bps -> daily net returns
    turnover: pd.Series                    # charged on execution days, 0 elsewhere
    target_weights: pd.DataFrame           # rows = signal dates, cols = tickers held
    eligible_counts: pd.Series             # per signal date
    holdings: list[dict]                   # per signal date: picks with weights/scores
    data_notes: list[str] = field(default_factory=list)
    annual_turnover: float = 0.0
    signal_dates: pd.DatetimeIndex | None = None

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
        inv = 1.0 / vol.loc[picks.index]
        w = inv / inv.sum()
    else:
        raise EngineError(f"unknown weighting: {weighting}")
    return w * scale


class _SelectionEngine:
    """Computes target weights at any signal date from data ≤ that date."""

    def __init__(self, spec: StrategySpec, prices: pd.DataFrame, universe: list[str]):
        self.spec = spec
        self.universe = [t for t in dict.fromkeys(universe) if t in prices.columns]
        self.prices = prices
        sel = spec.selection
        self.static = sel.static_allocation is not None
        if not self.static:
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
        scores_row = self.rank_panel.loc[t]
        eligible &= scores_row.notna()
        for panel, op, value in self.filter_panels:
            eligible &= apply_filter_op(panel.loc[t], op, value).fillna(False)
        if self.vol_panel is not None:
            eligible &= self.vol_panel.loc[t].notna()

        elig_scores = scores_row[eligible]
        k_eligible = int(len(elig_scores))
        if k_eligible == 0:
            return pd.Series(dtype=float), 0, []

        picks = _deterministic_top(elig_scores.dropna(), sel.n or 1, sel.direction or "top")
        vol_row = self.vol_panel.loc[t] if self.vol_panel is not None else None
        w = _position_weights(picks, self.spec.weighting, sel.n or 1, vol_row)
        detail = [
            {"ticker": tk, "weight": float(w[tk]), "score": float(picks[tk])}
            for tk in picks.index
        ]
        return w, k_eligible, detail


def run(spec: StrategySpec, prices: pd.DataFrame, universe: list[str]) -> EngineResult:
    """Pure function: (spec, price panel, universe tickers) → results.

    `prices` must include lead-in history before window.start for signal
    lookbacks; simulation accrues returns only inside the window.
    """
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise EngineError("price panel index must be sorted, unique dates")

    window_index = clip_window(prices.index, spec.window.start, spec.window.end)
    if len(window_index) < 3:
        raise EngineError(
            "the requested window contains fewer than 3 trading days of data"
        )

    selector = _SelectionEngine(spec, prices, universe)

    # Signal dates. Ranked strategies enter at period ends; static allocations
    # buy on day one and then rebalance at period ends (so buy-and-hold over
    # the window means exactly the window's return — golden test a).
    sig = rebalance_signal_dates(window_index, spec.rebalance)
    if selector.static:
        sig = pd.DatetimeIndex(sorted({window_index[0], *sig}))
    sig = sig[sig < window_index[-1]]  # a signal on the last day has no next day
    if len(sig) == 0:
        raise EngineError("no rebalance dates fall inside the requested window")

    hold_days = spec.entry_exit.hold_days
    if hold_days is not None:
        return _run_cohorts(spec, prices, selector, window_index, sig, hold_days)
    return _run_standard(spec, prices, selector, window_index, sig)


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
    sig: pd.DatetimeIndex,
) -> EngineResult:
    prices_ff = prices.ffill()
    s0 = sig[0]
    accrual = window_index[window_index > s0]
    if len(accrual) == 0:
        raise EngineError("window ends before the first rebalance can execute")

    gross = pd.Series(0.0, index=accrual)
    turnover = pd.Series(0.0, index=accrual)
    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    eligible_counts: dict[pd.Timestamp, int] = {}
    holdings: list[dict] = []
    data_notes: list[str] = []

    prev_drifted = pd.Series(dtype=float)  # asset weights before first buy: none
    sig_list = list(sig)
    for k, t in enumerate(sig_list):
        w, n_elig, detail = selector.weights_at(t)
        weights_rows[t] = w
        eligible_counts[t] = n_elig
        holdings.append({"date": str(t.date()), "eligible": n_elig, "picks": detail})

        seg_end = sig_list[k + 1] if k + 1 < len(sig_list) else accrual[-1]
        seg_days = accrual[(accrual > t) & (accrual <= seg_end)]
        if len(seg_days) == 0:
            prev_drifted = w
            continue

        # Turnover: trade from drifted prior weights into w, charged on the
        # first accrual day of this segment.
        union = w.index.union(prev_drifted.index)
        delta = w.reindex(union, fill_value=0.0) - prev_drifted.reindex(union, fill_value=0.0)
        turnover.loc[seg_days[0]] += 0.5 * float(delta.abs().sum())

        cash = 1.0 - float(w.sum())
        if len(w) > 0:
            held = list(w.index)
            base = prices_ff.loc[t, held]
            rel = prices_ff.loc[seg_days, held].div(base, axis=1)
            value = rel.mul(w, axis=1).sum(axis=1) + cash
            data_notes.extend(_data_gap_notes(prices, held, seg_days))
            end_rel = rel.iloc[-1]
            end_value = float(value.iloc[-1])
            prev_drifted = (w * end_rel) / end_value
        else:
            value = pd.Series(1.0, index=seg_days)  # all cash
            prev_drifted = pd.Series(dtype=float)
        prev_vals = value.shift(1)
        prev_vals.iloc[0] = 1.0
        gross.loc[seg_days] = (value / prev_vals - 1.0).values

    net = {
        float(bps): gross - turnover * (bps / 10_000.0)
        for bps in spec.costs.sweep_bps
    }
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
    )
    return result


def _run_cohorts(
    spec: StrategySpec,
    prices: pd.DataFrame,
    selector: _SelectionEngine,
    window_index: pd.DatetimeIndex,
    sig: pd.DatetimeIndex,
    hold_days: int,
) -> EngineResult:
    """Overlapping cohorts: each signal date spawns a cohort that earns returns
    for the next hold_days trading days. Capital is split equally across
    active cohorts each day; the portfolio's daily gross return is the mean of
    active-cohort returns. Aggregate turnover is measured on the combined
    weight vector, so overlapping names net out (honest cost accounting)."""
    if selector.static:
        raise EngineError("hold_days does not apply to static allocations")

    tickers = selector.universe
    col_ix = {tk: i for i, tk in enumerate(tickers)}
    prices_ff = prices[tickers].ffill()
    px = prices_ff.to_numpy()
    date_pos = {d: i for i, d in enumerate(prices.index)}
    sig_set = set(sig)

    s0 = sig[0]
    accrual = window_index[window_index > s0]
    if len(accrual) == 0:
        raise EngineError("window ends before the first cohort can be established")

    n_assets = len(tickers)
    gross = np.zeros(len(accrual))
    turnover = np.zeros(len(accrual))
    eligible_counts: dict[pd.Timestamp, int] = {}
    holdings: list[dict] = []
    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    data_notes: set[str] = set()

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

    spawn(s0)
    w_post = aggregate_weights()  # weights in force entering the first day
    prev_pos = date_pos[s0]

    for i, d in enumerate(accrual):
        pos = date_pos[d]
        row_now = px[pos]
        row_prev = px[prev_pos]
        with np.errstate(invalid="ignore", divide="ignore"):
            day_rel = np.where(
                np.isnan(row_now) | np.isnan(row_prev) | (row_prev == 0),
                1.0,
                row_now / row_prev,
            )

        # 1. Earn day d in every active cohort
        rets = []
        nan_now = np.isnan(row_now)
        for c in cohorts:
            before = c["v"].sum() + c["cash"]
            held_gaps = nan_now & (c["v"] > 0)
            if held_gaps.any():
                for j in np.flatnonzero(held_gaps):
                    data_notes.add(
                        f"{tickers[j]}: price missing while held (around "
                        f"{d.date()}); last close carried forward"
                    )
            c["v"] = c["v"] * day_rel
            after = c["v"].sum() + c["cash"]
            rets.append(after / before - 1.0)
            c["age"] += 1
        gross[i] = float(np.mean(rets)) if rets else 0.0

        # Turnover decided at the previous close is charged on day d
        # (w_post vs w_pre computed at the end of the previous iteration).
        # For day 0 it is the initial deployment from cash.
        if i == 0:
            turnover[i] = 0.5 * float(np.abs(w_post).sum())

        # 2. Close-of-day d bookkeeping: kill expired cohorts, spawn new one
        w_pre = aggregate_weights()
        cohorts[:] = [c for c in cohorts if c["age"] < hold_days]
        if d in sig_set:
            spawn(d)
        w_post = aggregate_weights()
        if i + 1 < len(accrual):
            turnover[i + 1] = 0.5 * float(np.abs(w_post - w_pre).sum())

        prev_pos = pos

    gross_s = pd.Series(gross, index=accrual)
    turnover_s = pd.Series(turnover, index=accrual)
    net = {
        float(bps): gross_s - turnover_s * (bps / 10_000.0)
        for bps in spec.costs.sweep_bps
    }
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
    )
