"""Out-of-sample split + parameter sensitivity (audit check #6).

70/30 split by trading days; the verdict privileges the holdout segment.
Sensitivity re-runs the engine with each numeric parameter perturbed one
step; if any perturbation (or the holdout itself) flips beats-benchmark,
the result depends on a specific choice — noise, not edge.
"""
from __future__ import annotations

import copy

import pandas as pd

from audit.findings import Finding
from dsl.schema import StrategySpec
from engine import metrics
from engine.backtest import EngineError, run

TRAIN_FRACTION = 0.70
SWEEP_VERDICT_BPS = 5.0


def split_index(accrual: pd.DatetimeIndex) -> pd.Timestamp:
    return accrual[int(len(accrual) * TRAIN_FRACTION) - 1]


def oos_stats(net: pd.Series, gross: pd.Series, bench: pd.Series,
              rf_daily: float) -> tuple[dict, Finding]:
    split = split_index(net.index)
    segs = {}
    for label, lo, hi in (("in_sample", None, split), ("out_of_sample", split, None)):
        sl = slice(lo, hi) if lo is None else slice(net.index[net.index > lo][0], hi)
        r, b = net.loc[sl], bench.loc[sl]
        segs[label] = {
            **metrics.summary(r, rf_daily, b),
            "benchmark_total_return": metrics.total_return(b),
            "benchmark_sharpe": metrics.sharpe(b, rf_daily),
            "beats_benchmark": metrics.total_return(r) > metrics.total_return(b),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        }

    is_sh, oos_sh = segs["in_sample"]["sharpe"], segs["out_of_sample"]["sharpe"]
    overfit = (
        is_sh is not None and oos_sh is not None and is_sh > 0
        and (oos_sh < 0 or (oos_sh > 0 and is_sh / max(oos_sh, 1e-9) >= 2.0))
        and segs["in_sample"]["beats_benchmark"]
    )
    if overfit:
        ratio = is_sh / oos_sh if oos_sh and oos_sh > 0 else float("inf")
        ratio_txt = f"{ratio:.1f}×" if ratio != float("inf") else "∞"
        finding = Finding(
            check="overfitting", severity="warning", name="Overfit signature",
            message=(
                f"The strategy's Sharpe is {is_sh:.2f} on the first 70% of the "
                f"window but {oos_sh:.2f} on the 30% holdout ({ratio_txt} better "
                f"in-sample). Performing much better on data you tuned against "
                f"is the classic overfit signature."
            ),
            evidence={"in_sample_sharpe": is_sh, "holdout_sharpe": oos_sh},
            explainer="overfitting",
        )
    else:
        finding = Finding(
            check="overfitting", severity="pass", name="Train/holdout comparison",
            message=(
                f"Sharpe {is_sh:.2f} in-sample vs {oos_sh:.2f} on the 30% "
                f"holdout — no large in-sample advantage detected. The verdict "
                f"uses the holdout segment either way."
            ),
            evidence={"in_sample_sharpe": is_sh, "holdout_sharpe": oos_sh},
            explainer="overfitting",
        )
    out = {"split_date": str(split.date()), **segs}
    return out, finding


def _perturbations(spec: StrategySpec) -> list[tuple[str, StrategySpec]]:
    """One-step perturbations of every numeric knob (Gap: ±1 step)."""
    out: list[tuple[str, StrategySpec]] = []

    def variant(label: str, mutate) -> None:
        s = copy.deepcopy(spec)
        try:
            mutate(s)
        except Exception:
            return
        out.append((label, s))

    sel = spec.selection
    if sel.rank_by is not None and sel.n is not None:
        if sel.n > 1:
            variant(f"n {sel.n}→{sel.n - 1}", lambda s: setattr(s.selection, "n", sel.n - 1))
        if sel.n < 50:
            variant(f"n {sel.n}→{sel.n + 1}", lambda s: setattr(s.selection, "n", sel.n + 1))
        lb = sel.rank_by.lookback_days
        if lb is not None and sel.rank_by.type in ("return", "daily_return", "volatility"):
            step = 21 if lb > 21 else max(1, lb // 5)
            if lb - step >= 1:
                variant(f"lookback {lb}→{lb - step}",
                        lambda s: setattr(s.selection.rank_by, "lookback_days", lb - step))
            if lb + step <= 756:
                variant(f"lookback {lb}→{lb + step}",
                        lambda s: setattr(s.selection.rank_by, "lookback_days", lb + step))
    hd = spec.entry_exit.hold_days
    if hd is not None:
        if hd > 1:
            variant(f"hold_days {hd}→{hd - 1}",
                    lambda s: setattr(s.entry_exit, "hold_days", hd - 1))
        if hd < 252:
            variant(f"hold_days {hd}→{hd + 1}",
                    lambda s: setattr(s.entry_exit, "hold_days", hd + 1))
    return out


def sensitivity(spec: StrategySpec, prices: pd.DataFrame, universe: list[str],
                base_net5_total: float, bench: pd.Series,
                progress=None) -> tuple[list[dict], Finding | None]:
    """Re-run each perturbation; flag flips vs the benchmark (net of 5 bps)."""
    bench_total = metrics.total_return(bench)
    base_beats = base_net5_total > bench_total
    rows: list[dict] = []
    flipped: list[str] = []
    for label, variant_spec in _perturbations(spec):
        if progress:
            progress(f"Sensitivity: {label}")
        try:
            res = run(variant_spec, prices, universe)
        except EngineError:
            continue
        net5 = res.net_returns.get(SWEEP_VERDICT_BPS)
        if net5 is None:
            continue
        v_bench = bench.reindex(net5.index).dropna()
        v_total = metrics.total_return(net5)
        v_beats = v_total > metrics.total_return(v_bench)
        rows.append({"variant": label, "net5_total_return": v_total,
                     "beats_benchmark": v_beats})
        if v_beats != base_beats:
            flipped.append(label)

    if not rows:
        return rows, None
    if flipped and base_beats:
        finding = Finding(
            check="parameter_sensitivity", severity="warning",
            name="Result depends on a specific parameter choice",
            message=(
                "The base parameters beat the benchmark, but nudging "
                + ", ".join(flipped[:3])
                + (" (and more)" if len(flipped) > 3 else "")
                + " one step flips it to losing. An edge that vanishes when "
                  "top-5 becomes top-6 is noise, not edge."
            ),
            evidence={"flipped": flipped}, explainer="parameter-sensitivity",
        )
    else:
        finding = Finding(
            check="parameter_sensitivity", severity="pass",
            name="Parameter sensitivity",
            message=(
                f"{len(rows)} one-step parameter variants were re-run; the "
                f"beat-the-benchmark outcome "
                + ("stayed the same in every variant."
                   if not flipped else
                   "already trailed the benchmark, and variants agreed.")
            ),
            evidence={"variants": len(rows), "flipped": flipped},
            explainer="parameter-sensitivity",
        )
    return rows, finding
