"""Selection/survivorship bias (audit check #1) — the +1650% trap.

A ranked strategy on a hand-picked watchlist is re-run on the full S&P 500
(neutral-ish; still today's members, which is disclosed). If the hand-picked
universe outperforms the neutral one by more than 2× in wealth-multiple
terms, the headline number is hindsight, not skill → FATAL.
"""
from __future__ import annotations

import copy

import pandas as pd

from audit.findings import Finding
from dsl.schema import StrategySpec
from dsl.universes import preset_tickers
from engine import metrics
from engine.backtest import EngineError, run
from engine.benchmark import BENCHMARK_TICKERS
from engine.data import get_panel

FATAL_RATIO = 2.0
NEUTRAL_BPS = 5.0


def current_members_note() -> Finding:
    return Finding(
        check="survivorship_universe", severity="note",
        name="Index list is today's members",
        message=(
            "This preset uses the index's CURRENT constituent list. Companies "
            "that collapsed and left the index are missing, so past windows "
            "are tilted toward survivors — less biased than a hand-picked "
            "list, but not unbiased. True point-in-time membership needs "
            "licensed data (V1)."
        ),
        explainer="survivorship-bias",
    )


def check_selection_bias(
    spec: StrategySpec, net5: pd.Series, progress=None,
) -> tuple[Finding, dict | None]:
    """→ (finding, neutral_rerun_summary | None)."""
    if spec.universe.preset != "user_watchlist":
        return current_members_note(), None
    if spec.selection.static_allocation is not None:
        return Finding(
            check="selection_bias", severity="note",
            name="Fixed allocation on chosen tickers",
            message=(
                "This is a fixed allocation, so ranking bias doesn't apply — "
                "but remember you chose these tickers knowing their history."
            ),
            explainer="selection-bias",
        ), None

    tickers = spec.universe.resolved_tickers or []
    if progress:
        progress("Checking your universe for selection bias (neutral re-run)…")

    neutral_spec = copy.deepcopy(spec)
    neutral_spec.universe.preset = "sp500"
    neutral_spec.universe.resolved_tickers = None
    neutral_spec.universe.watchlist_file = None
    try:
        neutral_universe = preset_tickers("sp500")
        panel = get_panel(
            neutral_universe + BENCHMARK_TICKERS[spec.benchmark] + ["SPY"],
            start=spec.window.start, end=spec.window.end,
        )
        neutral_res = run(neutral_spec, panel.prices, neutral_universe)
        neutral_net5 = neutral_res.net_returns[NEUTRAL_BPS]
    except (EngineError, RuntimeError) as exc:
        return Finding(
            check="selection_bias", severity="warning",
            name="Hand-picked universe (neutral re-run unavailable)",
            message=(
                f"Your universe is {len(tickers)} hand-picked tickers — chosen "
                f"knowing how they performed. The neutral-universe comparison "
                f"could not run ({exc}), so treat the headline number as "
                f"unverified."
            ),
            evidence={"tickers": len(tickers)},
            explainer="selection-bias",
        ), None

    # Compare over COMMON accrual dates only — the two runs can start accruing
    # at different times (lookback history differs per universe), and totals
    # over different periods are not comparable.
    common = net5.index.intersection(neutral_net5.index)
    if len(common) < 30:
        return Finding(
            check="selection_bias", severity="warning",
            name="Hand-picked universe (windows barely overlap)",
            message=(
                f"Your universe is {len(tickers)} hand-picked tickers, and the "
                f"neutral-universe comparison shares under 30 trading days with "
                f"your run — treat the headline number as unverified hindsight."
            ),
            evidence={"tickers": len(tickers)}, explainer="selection-bias",
        ), None
    user_total = metrics.total_return(net5.loc[common])
    neutral_total = metrics.total_return(neutral_net5.loc[common])
    ratio = (1 + user_total) / max(1 + neutral_total, 1e-9)
    rerun = {
        "universe": "sp500",
        "net5_total_return": neutral_total,
        "user_net5_total_return": user_total,
        "wealth_ratio": ratio,
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
    }

    if ratio > FATAL_RATIO:
        finding = Finding(
            check="selection_bias", severity="fatal",
            name="Selection bias detected",
            message=(
                f"Your universe is {len(tickers)} hand-picked tickers, and you "
                f"picked them knowing how they performed. On this list the "
                f"strategy returned {user_total:+.0%} (net of 5 bps); on the "
                f"full S&P 500 the identical rules returned {neutral_total:+.0%} "
                f"— a {ratio:.1f}× wealth gap. The difference is hindsight, not "
                f"skill. This is the same trap that produces fake +1650% "
                f"backtests."
            ),
            evidence=rerun,
            action="Re-run this spec on the full S&P 500",
            explainer="selection-bias",
        )
    else:
        tail = (
            "Over these exact dates the S&P list did even better — a reminder "
            "that a CURRENT constituent list is itself loaded with survivors, "
            "not proof your list is clean."
            if neutral_total >= user_total else
            "Within the 2× bar, but a chosen list still bakes in hindsight — "
            "prefer the neutral number."
        )
        finding = Finding(
            check="selection_bias", severity="warning",
            name="Hand-picked universe",
            message=(
                f"Your universe is {len(tickers)} hand-picked tickers — chosen "
                f"knowing how they performed. Identical rules, same dates: "
                f"your list {user_total:+.0%} vs full S&P 500 "
                f"{neutral_total:+.0%} (net of 5 bps). {tail}"
            ),
            evidence=rerun,
            action="Re-run this spec on the full S&P 500",
            explainer="selection-bias",
        )
    return finding, rerun
