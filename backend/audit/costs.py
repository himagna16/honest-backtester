"""Transaction-cost sweep (audit check #3) — the buy-the-dip killer."""
from __future__ import annotations

import pandas as pd

from audit.findings import Finding
from engine import metrics


def cost_sweep_table(
    gross: pd.Series, nets: dict[float, pd.Series], bench: pd.Series,
    rf_daily: float, bench_label: str,
) -> list[dict]:
    bench_total = metrics.total_return(bench)
    rows = []
    for bps in sorted(nets):
        r = nets[bps]
        rows.append({
            "bps": bps,
            "total_return": metrics.total_return(r),
            "cagr": metrics.cagr(r),
            "sharpe": metrics.sharpe(r, rf_daily),
            "max_drawdown": metrics.max_drawdown(r),
            "beats_benchmark": metrics.total_return(r) > bench_total,
        })
    rows.append({
        "bps": None, "label": bench_label,
        "total_return": bench_total,
        "cagr": metrics.cagr(bench),
        "sharpe": metrics.sharpe(bench, rf_daily),
        "max_drawdown": metrics.max_drawdown(bench),
        "beats_benchmark": None,
    })
    return rows


def check_costs(sweep: list[dict], annual_turnover: float,
                bench_label: str) -> Finding:
    by_bps = {row["bps"]: row for row in sweep if row["bps"] is not None}
    at0, at5, at20 = by_bps.get(0.0), by_bps.get(5.0), by_bps.get(20.0)
    if not (at0 and at5 and at20):
        return Finding(check="cost_sweep", severity="note", name="Cost sweep",
                       message="cost sweep incomplete", explainer="costs")

    if at0["beats_benchmark"] and not at5["beats_benchmark"]:
        return Finding(
            check="cost_sweep", severity="warning",
            name="Your edge is smaller than your costs",
            message=(
                f"Gross of costs this beats {bench_label} "
                f"({at0['total_return']:+.0%} vs benchmark), but at just 5 bps "
                f"per unit of turnover it loses "
                f"({at5['total_return']:+.0%}). With {annual_turnover:.1f}× "
                f"annual turnover, trading costs eat the entire edge — 5 bps "
                f"is OPTIMISTIC for retail."
            ),
            evidence={"annual_turnover": annual_turnover},
            explainer="costs",
        )
    if at5["beats_benchmark"] and not at20["beats_benchmark"]:
        return Finding(
            check="cost_sweep", severity="warning",
            name="Edge survives 5 bps, dies at 20",
            message=(
                f"The strategy beats {bench_label} at 5 bps but loses at 20 bps "
                f"({at20['total_return']:+.0%}). 20 bps is realistic for "
                f"frequent trading; assume the pessimistic column."
            ),
            evidence={"annual_turnover": annual_turnover},
            explainer="costs",
        )
    if not at0["beats_benchmark"]:
        return Finding(
            check="cost_sweep", severity="note",
            name="Costs are not the problem",
            message=(
                f"This strategy trails {bench_label} even before any trading "
                f"costs — the cost sweep only makes it worse."
            ),
            explainer="costs",
        )
    return Finding(
        check="cost_sweep", severity="pass",
        name="Survives the cost sweep",
        message=(
            f"Beats {bench_label} at 0, 5 AND 20 bps per unit of turnover "
            f"({annual_turnover:.1f}× annually). Rare — and still only "
            f"necessary, not sufficient."
        ),
        evidence={"annual_turnover": annual_turnover},
        explainer="costs",
    )


def small_cap_note(universe: list[str], known_large: set[str]) -> Finding | None:
    """V0 has no market-cap data; heuristic disclosure only (documented)."""
    unknown = [t for t in universe if t not in known_large]
    if not unknown:
        return None
    sample = ", ".join(sorted(unknown)[:6]) + ("…" if len(unknown) > 6 else "")
    return Finding(
        check="spread_realism", severity="note",
        name="Some names may trade with wide spreads",
        message=(
            f"{len(unknown)} universe ticker(s) are not in the major large-cap "
            f"indexes ({sample}). Small-cap spreads often exceed 20 bps — for "
            f"these names even the pessimistic cost column may be optimistic. "
            f"(V0 has no market-cap data; this is a heuristic disclosure.)"
        ),
        explainer="costs",
    )
