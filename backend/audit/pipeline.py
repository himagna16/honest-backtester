"""The audit pipeline: spec → engine → every check → verdict → report JSON.

This is THE only path to a backtest result. There is no route, function, or
flag that runs the engine for a user without the full audit — the pipeline is
one function and the audit is not a parameter (non-negotiable #1).

Verdict rollup (Section 3, determinized — documented in DECISIONS.md):
1. any FATAL                        → RED   "Result not trustworthy"
2. holdout net-5bps trails bench    → RED   "Loses to <bench> after costs"
3. net-20bps (full window) trails   → AMBER "Beats gross, dies at 20 bps"
4. no bear market in window         → AMBER "…but your window has no bear market"
5. sensitivity flipped              → AMBER "…but depends on a parameter choice"
6. otherwise                        → GREEN "Survived the gauntlet" (rare)
The verdict always quotes the pessimistic cell: net-of-costs, holdout.
"""
from __future__ import annotations

import pandas as pd

from audit import costs as costs_mod
from audit import extra, oos, regimes, universe as universe_mod
from audit.findings import Finding, sort_findings
from audit.lookahead import lookahead_statement
from dsl.readback import readback
from dsl.schema import StrategySpec
from dsl.universes import preset_tickers
from engine import metrics
from engine.backtest import run
from engine.benchmark import BENCHMARK_LABELS, BENCHMARK_TICKERS, benchmark_returns
from engine.data import PricePanel

DISCLAIMER = (
    "This is a historical simulation for educational purposes only. It is not "
    "investment advice, and The Honest Backtester is not a registered "
    "investment adviser or broker-dealer. Simulated performance does not "
    "reflect actual trading and has inherent limitations. Past performance "
    "does not guarantee future results. You can lose money trading securities."
)

VERDICT_BPS = 5.0
GREEN_BPS = 20.0


def _rf_daily(panel: PricePanel, index: pd.DatetimeIndex) -> tuple[float, str]:
    if "^IRX" in panel.prices.columns:
        irx = panel.prices["^IRX"].reindex(index).ffill().dropna()
        if len(irx) > 0:
            annual = float(irx.mean()) / 100.0
            return annual / 252.0, "^IRX"
    return 0.0, "none"


def _series_points(series_map: dict[str, pd.Series]) -> list[list]:
    """[[date, v1, v2, ...], ...] over the union index, JSON-safe."""
    df = pd.DataFrame(series_map)
    out = []
    for ts, row in df.iterrows():
        out.append([str(ts.date())] + [None if pd.isna(v) else round(float(v), 6)
                                       for v in row.values])
    return out


def build_report(
    spec: StrategySpec,
    panel: PricePanel,
    universe_tickers: list[str],
    attempts: int,
    progress=None,
) -> dict:
    """Pure function over (spec, panel) — replaying the same snapshot
    reproduces the same report (Gap #2). `attempts` comes from the caller so
    the counter bump stays outside the pure path."""
    def report_progress(stage: str) -> None:
        if progress:
            progress(stage)

    report_progress("Simulating with costs…")
    result = run(spec, panel.prices, universe_tickers)
    gross = result.gross_returns
    nets = result.net_returns
    net5 = nets[VERDICT_BPS]
    net20 = nets[GREEN_BPS]

    report_progress(f"Comparing against {spec.benchmark}…")
    bench = benchmark_returns(spec.benchmark, panel.prices, result.accrual_index)
    bench_label = BENCHMARK_LABELS[spec.benchmark]
    rf_daily, rf_source = _rf_daily(panel, result.accrual_index)

    findings: list[Finding] = [lookahead_statement()]

    # ---- selection bias (may trigger a neutral-universe re-run) ----
    bias_finding, neutral_rerun = universe_mod.check_selection_bias(
        spec, net5, progress=report_progress
    )
    findings.append(bias_finding)

    # ---- cost sweep ----
    sweep = costs_mod.cost_sweep_table(gross, nets, bench, rf_daily, bench_label)
    findings.append(costs_mod.check_costs(sweep, result.annual_turnover, bench_label))
    known_large = set(preset_tickers("sp500")) | set(preset_tickers("nasdaq100")) \
        | set(preset_tickers("dow30")) | set(BENCHMARK_TICKERS["sixty_forty"]) \
        | {"QQQ", "DIA"}
    spread_note = costs_mod.small_cap_note(universe_tickers, known_large)
    if spread_note:
        findings.append(spread_note)

    # ---- regimes ----
    report_progress("Stress-testing regimes…")
    regime_finding, regime_segments, has_bear = regimes.check_regimes(bench, bench_label)
    findings.append(regime_finding)

    # ---- out-of-sample + sensitivity ----
    report_progress("Splitting train vs holdout…")
    oos_data, overfit_finding = oos.oos_stats(net5, gross, bench, rf_daily)
    findings.append(overfit_finding)
    sens_rows, sens_finding = oos.sensitivity(
        spec, panel.prices, universe_tickers,
        metrics.total_return(net5), bench, progress=report_progress,
    )
    if sens_finding:
        findings.append(sens_finding)

    # ---- benchmark fit, capacity, multiple testing, data quality ----
    findings.append(extra.check_benchmark_fit(net5, bench, spec.benchmark, bench_label))
    findings.append(extra.check_capacity(result.holdings, panel.prices))
    findings.append(extra.check_multiple_testing(spec.universe_key(), attempts))
    findings.extend(extra.data_findings(
        panel, result.data_notes, spec.window.start or "",
        str(result.effective_start.date()), panel.first_dates,
    ))
    findings.append(extra.rf_finding(rf_source, rf_daily * 252))

    # ---- verdict rollup ----
    fatals = [f for f in findings if f.severity == "fatal"]
    holdout = oos_data["out_of_sample"]
    holdout_beats = holdout["beats_benchmark"]
    net20_beats = metrics.total_return(net20) > metrics.total_return(bench)
    sens_flipped = bool(sens_finding and sens_finding.severity == "warning")

    if fatals:
        verdict = {
            "color": "red", "icon": "✕",
            "headline": "Result not trustworthy.",
            "sub": fatals[0].name + " — " + (fatals[0].action or "see the findings below."),
        }
    elif not holdout_beats:
        verdict = {
            "color": "red", "icon": "✕",
            "headline": f"Loses to {spec.benchmark.replace('sixty_forty', '60/40')} after costs.",
            "sub": (
                f"On the 30% holdout, net of 5 bps: "
                f"{holdout['total_return']:+.1%} vs benchmark "
                f"{holdout['benchmark_total_return']:+.1%}. The honest verdict "
                f"uses the pessimistic cell."
            ),
        }
    elif not net20_beats:
        verdict = {
            "color": "amber", "icon": "▲",
            "headline": "Beats the benchmark gross — but dies at realistic costs.",
            "sub": (
                f"Survives 5 bps but loses at 20 bps "
                f"({metrics.total_return(net20):+.1%} vs "
                f"{metrics.total_return(bench):+.1%}). Your edge is smaller "
                f"than frequent-trading costs."
            ),
        }
    elif not has_bear:
        verdict = {
            "color": "amber", "icon": "▲",
            "headline": f"Beats {spec.benchmark.replace('sixty_forty', '60/40')} — but your test window has no bear market.",
            "sub": "Results from an all-bull window may not survive the first real drawdown.",
        }
    elif sens_flipped:
        verdict = {
            "color": "amber", "icon": "▲",
            "headline": "Beats the benchmark — but only at these exact parameters.",
            "sub": "One-step parameter changes flip the result. That pattern is noise, not edge.",
        }
    elif (spec.universe.preset == "user_watchlist"
          and spec.selection.static_allocation is None):
        # A ranked strategy on a hindsight universe can never earn green —
        # even when the neutral re-run happens to do as well (a CURRENT index
        # list is itself survivor-loaded). Honesty > flattery.
        verdict = {
            "color": "amber", "icon": "▲",
            "headline": "Beats the benchmark — but your universe is hand-picked.",
            "sub": (
                "You chose these tickers knowing how they performed. A ranked "
                "strategy on a hindsight universe cannot earn a green verdict — "
                "re-run it on the full S&P 500."
            ),
        }
    else:
        verdict = {
            "color": "green", "icon": "✓",
            "headline": "Survived the gauntlet. Beats the benchmark net of 20 bps across regimes.",
            "sub": ("Surviving a backtest is necessary, not sufficient. "
                    "This is not investment advice."),
        }

    # ---- assemble ----
    strat_stats = {
        **metrics.summary(net5, rf_daily, bench),
        "annual_turnover": result.annual_turnover,
    }
    report = {
        "verdict": verdict,
        "metrics": strat_stats,                                   # net of 5 bps
        "gross_metrics": metrics.summary(gross, rf_daily, bench),
        "net20_metrics": metrics.summary(net20, rf_daily, bench),
        "benchmark": {
            "name": spec.benchmark, "label": bench_label,
            **metrics.summary(bench, rf_daily),
        },
        "cost_sweep": sweep,
        "audit": [f.to_dict() for f in sort_findings(findings)],
        "oos": oos_data,
        "sensitivity": sens_rows,
        "regimes": {"segments": regime_segments, "has_bear_market": has_bear},
        "neutral_rerun": neutral_rerun,
        "multiple_testing": {"universe_key": spec.universe_key(), "attempts": attempts},
        "equity_curve": _series_points({
            "net5": metrics.equity_curve(net5),
            "gross": metrics.equity_curve(gross),
            "benchmark": metrics.equity_curve(bench),
        }),
        "drawdown_curve": _series_points({
            "strategy": metrics.drawdown_series(net5),
            "benchmark": metrics.drawdown_series(bench),
        }),
        "data": {
            "as_of": panel.as_of,
            "stale": panel.stale,
            "missing": panel.missing,
            "notes": panel.notes,
            "universe_size": len(universe_tickers),
            "effective_start": str(result.effective_start.date()),
            "effective_end": str(result.effective_end.date()),
            "requested_start": spec.window.start,
            "requested_end": spec.window.end,
            "rf_source": rf_source,
        },
        "spec": spec.model_dump(mode="json"),
        "readback": readback(spec),
        "holdings_recent": result.holdings[-3:],
        "disclaimer": DISCLAIMER,
    }
    return report
