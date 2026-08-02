"""Regime coverage (audit check #5).

A hardcoded table of known market events + drawdown detection on the
benchmark inside the actual window. If the window contains no ≥15% benchmark
drawdown, the strategy has never been tested in a bear market — WARNING.
"""
from __future__ import annotations

import pandas as pd

from audit.findings import Finding
from engine import metrics

# Known stress events (label, start, end, kind). Dates are approximate
# peak-to-trough; the bar clips them to the tested window.
KNOWN_EVENTS = [
    ("China devaluation selloff", "2015-08-10", "2015-09-29", "correction"),
    ("2018 Q4 rate scare", "2018-10-01", "2018-12-24", "bear"),
    ("COVID crash", "2020-02-19", "2020-03-23", "bear"),
    ("2022 rate shock", "2022-01-03", "2022-10-12", "bear"),
    ("2023 autumn pullback", "2023-08-01", "2023-10-27", "correction"),
    ("2025 tariff shock", "2025-02-19", "2025-04-08", "bear"),
]

BEAR_DD_THRESHOLD = -0.15


def regime_segments(bench_returns: pd.Series) -> list[dict]:
    """Timeline segments for the regime bar: known events inside the window,
    gaps labeled bull/sideways by the benchmark's own return over the gap."""
    start, end = bench_returns.index[0], bench_returns.index[-1]
    events = []
    for label, s, e, kind in KNOWN_EVENTS:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if e_ts < start or s_ts > end:
            continue
        events.append((max(s_ts, start), min(e_ts, end), label, kind))
    events.sort()

    segments: list[dict] = []
    cursor = start

    def add_gap(a: pd.Timestamp, b: pd.Timestamp) -> None:
        if b <= a:
            return
        chunk = bench_returns.loc[a:b]
        if len(chunk) < 2:
            return
        total = metrics.total_return(chunk)
        years = max(len(chunk) / 252.0, 1e-9)
        annual = (1 + total) ** (1 / years) - 1
        kind = "bull" if annual > 0.05 else ("sideways" if annual > -0.05 else "down")
        segments.append({"label": kind, "start": str(a.date()), "end": str(b.date()),
                         "kind": kind})

    for s_ts, e_ts, label, kind in events:
        add_gap(cursor, s_ts)
        segments.append({"label": label, "start": str(s_ts.date()),
                         "end": str(e_ts.date()), "kind": kind})
        cursor = e_ts
    add_gap(cursor, end)
    return segments


def check_regimes(bench_returns: pd.Series, bench_label: str) -> tuple[Finding, list[dict], bool]:
    """→ (finding, timeline segments, has_bear_market)."""
    segments = regime_segments(bench_returns)
    window_dd = metrics.max_drawdown(bench_returns)
    has_bear = window_dd <= BEAR_DD_THRESHOLD
    covered = sorted({s["kind"] for s in segments})

    if has_bear:
        finding = Finding(
            check="regime_coverage", severity="pass", name="Bear-market coverage",
            message=(
                f"The test window includes at least one real drawdown — "
                f"{bench_label} fell {window_dd:.0%} peak-to-trough inside it. "
                f"Regimes covered: {', '.join(covered)}."
            ),
            evidence={"benchmark_max_drawdown": window_dd, "covered": covered},
            explainer="regime-coverage",
        )
    else:
        finding = Finding(
            check="regime_coverage", severity="warning",
            name="No bear market in the test window",
            message=(
                f"The deepest {bench_label} drawdown inside this window is "
                f"{window_dd:.0%} — there is no sustained bear market here. "
                f"Strategies that look great in bull markets often die in the "
                f"first real drawdown; these results may not survive one."
            ),
            evidence={"benchmark_max_drawdown": window_dd, "covered": covered},
            explainer="regime-coverage",
        )
    return finding, segments, has_bear
