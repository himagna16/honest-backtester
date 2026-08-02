"""Remaining audit checks: benchmark appropriateness (Gap #5), capacity (#8),
multiple testing (#7), and data-quality disclosures (Gap #4 / non-neg #7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text as sql

from audit.findings import Finding
from config import settings
from engine.data import PricePanel
from store.db import get_engine

LOW_BETA_BAR = 0.65


# ---------------- benchmark appropriateness (Gap #5) ----------------

def check_benchmark_fit(strategy: pd.Series, bench: pd.Series,
                        benchmark_name: str, bench_label: str) -> Finding:
    common = strategy.index.intersection(bench.index)
    s, b = strategy.loc[common], bench.loc[common]
    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(s, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")

    if benchmark_name in ("SPY", "QQQ") and not np.isnan(beta) and abs(beta) < LOW_BETA_BAR:
        return Finding(
            check="benchmark_fit", severity="warning",
            name=f"{benchmark_name} may be the wrong yardstick",
            message=(
                f"The strategy's realized beta to {bench_label} is {beta:.2f} — "
                f"it moves much less than the benchmark it's judged against. "
                f"A low-beta portfolio looks bad in bull markets and fake-good "
                f"in bears when compared to pure stocks. Re-run against the "
                f"60/40 benchmark for a fairer yardstick."
            ),
            evidence={"realized_beta": beta},
            action="Re-run with benchmark = 60/40",
            explainer="benchmark-fit",
        )
    return Finding(
        check="benchmark_fit", severity="pass",
        name="Benchmark appropriateness",
        message=f"Realized beta to {bench_label} is {beta:.2f} — the chosen "
                f"benchmark is a reasonable yardstick.",
        evidence={"realized_beta": beta},
        explainer="benchmark-fit",
    )


# ---------------- capacity / realism (#8) ----------------

def check_capacity(holdings: list[dict], prices: pd.DataFrame) -> Finding:
    account = settings.account_size
    sub_share: list[str] = []
    for record in holdings[-3:]:  # recent rebalances are the relevant ones
        t = pd.Timestamp(record["date"])
        for pick in record["picks"]:
            tk, w = pick["ticker"], pick["weight"]
            px = prices[tk].asof(t) if tk in prices.columns else np.nan
            if not np.isnan(px) and w * account < px:
                sub_share.append(f"{tk} (~${w * account:,.0f} target vs ${px:,.0f}/share)")
    if sub_share:
        uniq = list(dict.fromkeys(sub_share))
        return Finding(
            check="capacity", severity="note",
            name=f"Position sizes assume fractional shares (at ${account:,.0f})",
            message=(
                f"At a ${account:,.0f} account, some positions are under one "
                f"share: {'; '.join(uniq[:4])}"
                + ("…" if len(uniq) > 4 else "")
                + ". Without fractional shares these weights are untradeable "
                  "as specified."
            ),
            evidence={"examples": uniq[:8], "account_size": account},
            explainer="capacity",
        )
    return Finding(
        check="capacity", severity="pass", name="Capacity / realism",
        message=f"All recent positions are at least one share at a "
                f"${account:,.0f} account.",
        explainer="capacity",
    )


# ---------------- multiple testing (#7) ----------------

def bump_attempts(universe_key: str) -> int:
    with get_engine().begin() as conn:
        conn.execute(sql(
            "INSERT INTO attempt_counters (universe_key, attempts) VALUES (:k, 1) "
            "ON CONFLICT(universe_key) DO UPDATE SET attempts = attempts + 1"
        ), {"k": universe_key})
        row = conn.execute(sql(
            "SELECT attempts FROM attempt_counters WHERE universe_key = :k"
        ), {"k": universe_key}).fetchone()
    return int(row[0])


def check_multiple_testing(universe_key: str, attempts: int) -> Finding:
    if attempts <= 3:
        return Finding(
            check="multiple_testing", severity="pass",
            name="Multiple-testing counter",
            message=f"This is run #{attempts} against this universe. The "
                    f"counter never resets — with enough tries, something "
                    f"beats the benchmark by luck.",
            evidence={"attempts": attempts, "universe_key": universe_key},
            explainer="multiple-testing",
        )
    return Finding(
        check="multiple_testing", severity="note",
        name=f"This is strategy variant #{attempts} on this data",
        message=(
            f"You have now tested {attempts} strategies against this universe. "
            f"With enough tries, something beats the benchmark by luck alone — "
            f"a Bonferroni-adjusted significance bar is roughly {attempts}× "
            f"higher than for a single test. This never blocks a run; it "
            f"informs how much to trust a winner."
        ),
        evidence={"attempts": attempts, "universe_key": universe_key},
        explainer="multiple-testing",
    )


# ---------------- data quality (Gap #4 / non-negotiable #7) ----------------

def data_findings(panel: PricePanel, engine_notes: list[str],
                  requested_start: str, effective_start: str,
                  short_history: dict[str, str]) -> list[Finding]:
    out: list[Finding] = []
    if panel.missing:
        out.append(Finding(
            check="data_missing", severity="warning",
            name=f"{len(panel.missing)} ticker(s) excluded — no data",
            message=(
                ", ".join(sorted(panel.missing)[:10])
                + ("…" if len(panel.missing) > 10 else "")
                + " had no price data and were EXCLUDED from the universe. "
                  "Excluding losers-with-no-data is itself a survivorship "
                  "tilt; results lean toward the names that still exist."
            ),
            evidence={"missing": panel.missing},
            explainer="missing-data",
        ))
    late = {t: d for t, d in short_history.items()
            if pd.Timestamp(d) > pd.Timestamp(requested_start) + pd.Timedelta(days=370)}
    if late:
        sample = ", ".join(f"{t} (from {d})" for t, d in sorted(late.items())[:6])
        out.append(Finding(
            check="data_short_history", severity="note",
            name=f"{len(late)} ticker(s) join the window late",
            message=(
                f"{sample}" + ("…" if len(late) > 6 else "") +
                f". They become eligible only once their signal lookback "
                f"exists — early results draw from a smaller universe."
            ),
            evidence={"late": late},
            explainer="missing-data",
        ))
    if pd.Timestamp(effective_start) > pd.Timestamp(requested_start) + pd.Timedelta(days=45):
        out.append(Finding(
            check="data_window_shortened", severity="warning",
            name="Backtest window shortened",
            message=(
                f"You asked for {requested_start} but simulation starts "
                f"{effective_start} — the earlier span lacks data (or a full "
                f"signal lookback). The report covers the shorter window only."
            ),
            explainer="missing-data",
        ))
    if panel.stale:
        out.append(Finding(
            check="data_stale", severity="note",
            name=f"Data as of {panel.as_of}",
            message=(
                f"Prices come from the local cache stamped {panel.as_of} — a "
                f"fresh fetch was skipped or failed. Numbers are exact for the "
                f"data shown but may not include recent sessions."
            ),
            explainer="data-freshness",
        ))
    for note in engine_notes[:5]:
        out.append(Finding(
            check="data_gap", severity="note", name="Price gap while held",
            message=note, explainer="missing-data",
        ))
    return out


def rf_finding(rf_source: str, rf_annual: float) -> Finding:
    if rf_source == "^IRX":
        return Finding(
            check="risk_free", severity="pass", name="Risk-free rate",
            message=f"Sharpe uses the 13-week T-bill (^IRX), averaging "
                    f"{rf_annual:.1%} annualized over this window.",
            explainer="sharpe",
        )
    return Finding(
        check="risk_free", severity="note",
        name="Risk-free rate assumed 0",
        message="^IRX data was unavailable, so Sharpe ratios use a 0% "
                "risk-free rate — they look slightly better than reality.",
        explainer="sharpe",
    )
