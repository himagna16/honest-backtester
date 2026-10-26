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
6. ranked on a hand-picked list     → AMBER "…but your universe is hand-picked"
7. ranked on an index without
   trustworthy point-in-time data   → AMBER "…but on today's index membership"
8. otherwise                        → GREEN "Survived the gauntlet" (rare)
The verdict always quotes the pessimistic cell: net-of-costs, holdout.
"""
from __future__ import annotations

import pandas as pd

import hashlib
import json

from audit import attribution, significance
from audit import costs as costs_mod
from audit import extra, oos, regimes, universe as universe_mod
from audit.findings import Finding, sort_findings
from audit.lookahead import lookahead_statement
from dsl.readback import readback
from dsl.schema import StrategySpec
from dsl.membership import load_membership
from dsl.universes import preset_tickers
from engine import metrics
from engine.backtest import run
from engine.benchmark import BENCHMARK_LABELS, BENCHMARK_TICKERS, benchmark_returns
from engine.data import PricePanel
from validation import SplitError, purge_components

DISCLAIMER = (
    "This is a historical simulation for educational purposes only. It is not "
    "investment advice, and The Honest Backtester is not a registered "
    "investment adviser or broker-dealer. Simulated performance does not "
    "reflect actual trading and has inherent limitations. Past performance "
    "does not guarantee future results. You can lose money trading securities."
)

VERDICT_BPS = 5.0
GREEN_BPS = 20.0

#: The bar a PSR/DSR probability must clear to count as "not luck". Matches the
#: convention inside audit/significance.py rather than inventing a second one.
SIGNIFICANCE_LEVEL = 0.95


def _rf_daily(
    panel: PricePanel, index: pd.DatetimeIndex
) -> tuple[pd.Series | float, float, str, str | None]:
    """→ (rf path or 0.0, mean daily rate, source, disclosure or None).

    The PATH, not a window average. It is used three ways that must agree: the
    engine credits it to uninvested cash, metrics subtract it to form excess
    returns, and the Sharpe hurdle quotes it. Averaging for one and pathing for
    another is how a hurdle and a cash yield drift apart — the disagreement
    `dsl/conventions.py` used to have to disclose.

    ^IRX quotes an ANNUALIZED discount rate in percent, so the daily rate is
    irx / 100 / 252.

    `engine.metrics._excess` REFUSES a rate path that does not cover every
    session of the return window, so partial coverage cannot be passed along
    silently. ^IRX keeps its own holiday calendar, so holes inside the window
    are normal and carried forward. A gap at the START has no earlier rate to
    carry, and back-filling one would let a later rate set an earlier day's
    hurdle — so those sessions take 0.0 and the shortfall is DISCLOSED, never
    absorbed.
    """
    if "^IRX" not in panel.prices.columns:
        return 0.0, 0.0, "none", None

    irx = panel.prices["^IRX"].reindex(index).ffill()
    if irx.isna().all():
        return 0.0, 0.0, "none", None

    lead_gap = int(irx.isna().sum())
    daily = (irx / 100.0 / 252.0).fillna(0.0)
    note = None
    if lead_gap:
        note = (
            f"^IRX has no quote for the first {lead_gap} session(s) of the "
            f"accrual window; those days use a 0% risk-free rate rather than "
            f"carrying a later rate backward onto an earlier day."
        )
    return daily, float(daily.mean()), "^IRX", note


def _seed(spec: StrategySpec) -> int:
    """A deterministic bootstrap seed derived from the spec.

    The resampling underneath the confidence intervals must not make a report
    irreproducible: the same spec has to yield the same interval on every
    replay (Gap #2). Derived from the spec, never from the clock.
    """
    blob = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(blob.encode()).digest()[:4], "big")


def _purge_terms(purge: dict) -> str:
    """"signal reach 252 + carry 23 + lag 1" — the terms, not just the total."""
    parts = [(k, int(v)) for k, v in purge.items()
             if k != "total" and isinstance(v, (int, float)) and int(v) > 0]
    return " + ".join(f"{k.replace('_', ' ')} {v}" for k, v in parts) or "total"


def _significance_findings(
    sharpe_stats, excess_stats, bench_label: str
) -> list[Finding]:
    """Turn the two assessments into findings — TEMPLATE CODE, not prose from a
    model. Every number here is read straight off the assessment objects.

    Two separate questions, so two separate findings:
      * is this Sharpe distinguishable from luck, after N attempts (DSR)?
      * does it actually beat the benchmark, or is the gap inside the noise?
    A refusal ("available: False") is reported AS a refusal — never quietly
    dropped, and never rendered as a pass.
    """
    out: list[Finding] = []

    if not sharpe_stats.available:
        out.append(Finding(
            check="sharpe_significance", severity="note",
            name="Sharpe significance not computable",
            message=(sharpe_stats.unavailable_reason
                     or "this return series cannot support Sharpe inference."),
            evidence={"n_obs": sharpe_stats.n_obs},
            explainer="sharpe",
        ))
    else:
        # DSR is the PROBABILITY that the true Sharpe beats what n_trials of
        # pure luck would have produced — not a p-value. The module's own
        # convention for "significant" is >= 0.95 (see PSR at significance.py
        # :560 and :699), so the same bar is used here rather than a new one.
        dsr = sharpe_stats.dsr
        psr = sharpe_stats.psr
        ev = {"n_obs": sharpe_stats.n_obs}
        if psr is not None:
            ev["psr"] = psr.psr
        dsr_value = dsr.dsr if (dsr is not None and dsr.available) else None
        if dsr is not None:
            ev["dsr"] = dsr_value
            ev["trials"] = dsr.n_trials
            ev["luck_threshold_sharpe_annual"] = dsr.sr0_annual

        if dsr_value is None:
            out.append(Finding(
                check="sharpe_significance", severity="note",
                name="Deflated Sharpe not computable",
                message=(dsr.interpretation if dsr is not None else
                         "the deflated Sharpe ratio could not be computed."),
                evidence=ev, explainer="sharpe",
            ))
        elif dsr_value < SIGNIFICANCE_LEVEL:
            out.append(Finding(
                check="sharpe_significance", severity="warning",
                name="Sharpe is not distinguishable from luck",
                message=(
                    f"After {dsr.n_trials} recorded attempt(s) on this "
                    f"universe, the deflated Sharpe ratio is {dsr_value:.2f} — "
                    f"below the {SIGNIFICANCE_LEVEL:.2f} bar. {dsr.n_trials} "
                    f"trials of pure luck would be expected to produce a "
                    f"Sharpe near {dsr.sr0_annual:.2f} annualized on their own."
                ),
                action="Re-test on data you have not already searched over.",
                evidence=ev, explainer="sharpe",
            ))
        else:
            out.append(Finding(
                check="sharpe_significance", severity="pass",
                name="Sharpe survives the multiple-testing haircut",
                message=dsr.interpretation,
                evidence=ev, explainer="sharpe",
            ))

    if not excess_stats.available:
        out.append(Finding(
            check="excess_significance", severity="note",
            name=f"Margin over {bench_label} not computable",
            message=(excess_stats.unavailable_reason
                     or "the sample cannot support an interval on the margin."),
            evidence={"n_obs": excess_stats.n_obs},
            explainer="benchmark",
        ))
    else:
        # The caveat is the honest half of a three-state verdict: the interval
        # behind it covers less than nominal, and the reader is entitled to
        # know that next to the word "beats".
        caveat = excess_stats.ci_coverage_caveat
        msg = excess_stats.interpretation
        if caveat:
            msg = f"{msg} {caveat}"
        sev = {"beats": "pass", "loses": "warning"}.get(
            excess_stats.verdict, "note")
        out.append(Finding(
            check="excess_significance", severity=sev,
            name={
                "beats": f"Margin over {bench_label} is outside the noise",
                "loses": f"Trails {bench_label} by more than noise",
            }.get(excess_stats.verdict,
                  f"Margin over {bench_label} is inside the noise"),
            message=msg,
            evidence={"n_obs": excess_stats.n_obs,
                      "verdict": excess_stats.verdict,
                      "verdict_source": excess_stats.verdict_source},
            explainer="benchmark",
        ))
    return out


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
    trial_sharpes_annual: list[float] | None = None,
) -> dict:
    """Pure function over (spec, panel) — replaying the same snapshot
    reproduces the same report (Gap #2). `attempts` comes from the caller so
    the counter bump stays outside the pure path.

    `trial_sharpes_annual` is the same deal one level further: the ledger of
    Sharpes already recorded against this universe. The Deflated Sharpe Ratio
    needs their SPREAD, not just the count in `attempts`; passing it in keeps
    the DB read out of the pure path so a replay stays reproducible. None or
    too few entries simply means the DSR reports itself as not computable — it
    is never silently replaced by an undeflated number."""
    def report_progress(stage: str) -> None:
        if progress:
            progress(stage)

    report_progress("Simulating with costs…")
    # The rate is built over the WHOLE panel index rather than the accrual
    # window, so the engine can be run once: engine._rf_vector reindexes to the
    # window itself (and discloses its own shortfalls). Cash earns this rate,
    # closing the hurdle-vs-cash-yield disagreement conventions.py documented.
    rf_panel, _, rf_source, _ = _rf_daily(panel, panel.prices.index)
    # Index presets pick only from the index as it stood on each signal date
    # when dated membership covers the window (audit/universe.py).
    membership_use = universe_mod.membership_for(spec)
    membership_mask = membership_use.mask(panel.prices.index, universe_tickers)
    result = run(spec, panel.prices, universe_tickers,
                 rf_daily=None if rf_source == "none" else rf_panel,
                 membership=membership_mask)
    membership_use.measure(result.signal_dates, panel.prices)

    # Metrics need the same rate on the ACCRUAL calendar; metrics._excess
    # refuses partial coverage, which is why this is derived, not resliced.
    rf_daily, rf_mean_daily, _, rf_note = _rf_daily(panel, result.accrual_index)

    gross = result.gross_returns
    nets = result.net_returns
    net5 = nets[VERDICT_BPS]
    net20 = nets[GREEN_BPS]

    report_progress(f"Comparing against {spec.benchmark}…")
    bench = benchmark_returns(spec.benchmark, panel.prices, result.accrual_index)
    bench_label = BENCHMARK_LABELS[spec.benchmark]

    findings: list[Finding] = [lookahead_statement()]
    if rf_note:
        findings.append(Finding(
            check="risk_free", severity="note",
            name="Risk-free rate incomplete at the window start",
            message=rf_note, explainer="sharpe",
        ))

    # ---- selection bias (may trigger a neutral-universe re-run) ----
    bias_finding, neutral_rerun = universe_mod.check_selection_bias(
        spec, net5, membership_use, progress=report_progress
    )
    findings.append(bias_finding)

    # ---- cost sweep ----
    sweep = costs_mod.cost_sweep_table(gross, nets, bench, rf_daily, bench_label)
    findings.append(costs_mod.check_costs(sweep, result.annual_turnover, bench_label))
    known_large = set(preset_tickers("sp500")) | set(preset_tickers("nasdaq100")) \
        | set(preset_tickers("dow30")) | set(BENCHMARK_TICKERS["sixty_forty"]) \
        | {"QQQ", "DIA"}
    sp500_history = load_membership("sp500")
    if sp500_history is not None:
        # Former members were large caps while they were in the index.
        known_large |= set(sp500_history.symbols_between(sp500_history.coverage_start, None))
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
        membership=membership_mask,
    )
    if sens_finding:
        findings.append(sens_finding)

    # ---- how leaky is that holdout, in bars? ----
    # The 70/30 cut above is a plain date split: no purge, no embargo. With
    # overlapping holds and multi-month lookbacks the training side can reach
    # across the boundary, so the split is not as out-of-sample as the word
    # "holdout" suggests. We do not silently move the verdict's split here —
    # that would change every historical number — but the size of the overlap
    # is a data problem, and data problems surface.
    purge = None
    try:
        purge = purge_components(
            spec, index=result.accrual_index,
            execution_lag_days=result.execution_lag_days,
        )
    except SplitError as exc:
        # SplitError carries a machine-readable reason precisely so a caller can
        # turn the refusal into a Finding "instead of silently falling back to a
        # leaky split" (validation/splits.py). Swallowing it would make the
        # disclosure below VANISH on exactly the runs that cannot size their own
        # overlap — the most dangerous case, reported as the quietest.
        findings.append(Finding(
            check="holdout_purge", severity="warning",
            name="Holdout overlap could not be measured",
            message=(
                f"The purge length for this run could not be computed "
                f"({exc}). The train/holdout split is a plain date cut, so how "
                f"far a training bar reaches into the holdout is unknown here "
                f"rather than zero."
            ),
            evidence={"reason": exc.reason, **exc.details},
            explainer="overfitting",
        ))
    except (ValueError, KeyError, TypeError) as exc:
        findings.append(Finding(
            check="holdout_purge", severity="warning",
            name="Holdout overlap could not be measured",
            message=(
                f"The purge length for this run could not be computed "
                f"({type(exc).__name__}: {exc}). How far a training bar reaches "
                f"into the holdout is unknown here rather than zero."
            ),
            evidence={"reason": type(exc).__name__},
            explainer="overfitting",
        ))
    if purge:
        bars = int(purge.get("total", 0))
        if bars > 0:
            findings.append(Finding(
                check="holdout_purge", severity="note",
                name="The holdout boundary is not purged",
                message=(
                    f"A training bar can influence a holdout bar up to {bars} "
                    f"sessions past the split ({_purge_terms(purge)}). The "
                    f"train/holdout numbers above use a plain date cut, so the "
                    f"first {bars} holdout sessions are not strictly "
                    f"out-of-sample."
                ),
                evidence=purge,
                explainer="overfitting",
            ))

    # ---- is the Sharpe real, or is it luck across many attempts? ----
    report_progress("Testing statistical significance…")
    seed = _seed(spec)
    hold_days = spec.entry_exit.hold_days or 1
    # A variance needs at least two points. One recorded trial is not a
    # "spread of zero" — that would claim the search was free and hand the
    # strategy an undeserved pass. Below two, hand over nothing and let the DSR
    # report itself as not computable, which is the truth.
    ledger = list(trial_sharpes_annual or ())
    sharpe_stats = significance.assess_sharpe(
        net5, seed=seed, rf_daily=rf_daily, n_trials=max(1, attempts),
        trial_sharpes_annual=ledger if len(ledger) >= 2 else None,
        holding_period_days=hold_days,
    )
    excess_stats = significance.assess_excess(
        net5, bench, seed=seed, holding_period_days=hold_days, label=bench_label,
    )
    findings.extend(_significance_findings(sharpe_stats, excess_stats, bench_label))

    # ---- is the "alpha" just known factors? ----
    report_progress("Decomposing against the factor model…")
    attrib, attrib_findings = attribution.attribution_report(
        net5, bench, bench_label, attempts=max(1, attempts),
    )
    findings.extend(attrib_findings)

    # ---- benchmark fit, capacity, multiple testing, data quality ----
    findings.append(extra.check_benchmark_fit(net5, bench, spec.benchmark, bench_label))
    findings.append(extra.check_capacity(result.holdings, panel.prices))
    findings.append(extra.check_multiple_testing(spec.universe_key(), attempts))
    findings.extend(extra.data_findings(
        panel, result.data_notes, spec.window.start or "",
        str(result.effective_start.date()), panel.first_dates,
    ))
    findings.append(extra.rf_finding(rf_source, rf_mean_daily * 252))

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
    elif (index_verdict := universe_mod.membership_verdict(membership_use)) is not None:
        # A ranked strategy on an index without trustworthy point-in-time
        # membership is picking from a survivor-loaded list. This used to fire
        # only above a 2× wealth gap, and a +3553% S&P 500 momentum run earned
        # GREEN under it: today's constituents include exactly the winners
        # that grew into the index, so no size of margin clears that list.
        verdict = index_verdict
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
        # Inference and attribution. Each carries its own `available` flag and
        # reason — a refusal is part of the payload, not an omission.
        "significance": {
            "sharpe": sharpe_stats.to_dict(),
            "excess": excess_stats.to_dict(),
            "seed": seed,
        },
        "attribution": attrib,
        "holdout_purge": purge,
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
            "universe_membership": membership_use.to_dict(),
        },
        "spec": spec.model_dump(mode="json"),
        "readback": readback(spec),
        # Rebalance log for the report's holdings table (most recent first).
        "holdings_recent": list(reversed(result.holdings[-24:])),
        "disclaimer": DISCLAIMER,
    }
    return report
