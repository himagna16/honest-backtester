"""Selection and survivorship bias (audit check #1) — the +1650% trap.

Two related questions, both about WHICH stocks a strategy was allowed to see:

1. Hand-picked universe (user_watchlist, ranked): re-run the identical rules on
   the neutral universe, the S&P 500 as it stood at each rebalance, and compare
   over common dates. More than a 2× wealth gap means the headline is hindsight,
   not skill → FATAL.

2. Index preset: was the index point-in-time, or today's members standing in
   for the past? Today's list is loaded with companies that joined AFTER their
   run-ups (AppLovin, Carvana, Vertiv, Palantir in 2023-2026), which is exactly
   what a ranked strategy harvests. A ranked run without point-in-time data
   cannot earn a green verdict, and a hand-picked comparison against today's
   list cannot clear a hand-picked universe.

The neutral comparison USED to be today's S&P 500 list, and that biased
benchmark let a 44-ticker hindsight list pass as "within 2×" of it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from audit.findings import Finding
from dsl.membership import MembershipCoverage, PointInTimeMembership
from dsl.schema import StrategySpec
from dsl.universes import INDEX_NAMES, INDEX_PRESETS, resolve_preset
from engine import metrics
from engine.backtest import EngineError, run
from engine.benchmark import BENCHMARK_TICKERS
from engine.data import get_panel

FATAL_RATIO = 2.0
NEUTRAL_BPS = 5.0
NEUTRAL_PRESET = "sp500"

# Share of index members, averaged over signal dates, that had a usable close.
# Below PIT_MIN the point-in-time universe is too thin to trust (treated like
# no point-in-time data); below PIT_CLEAN the gap is disclosed as a warning.
PIT_MIN_PRICED_SHARE = 0.90
PIT_CLEAN_PRICED_SHARE = 0.95


@dataclass
class MembershipUse:
    """How a run's universe relates to index membership over its window."""
    preset: str
    ranked: bool
    membership: PointInTimeMembership | None = None
    fallback_reason: str | None = None
    coverage: MembershipCoverage | None = field(default=None)

    @property
    def index_preset(self) -> bool:
        return self.preset in INDEX_PRESETS

    @property
    def point_in_time(self) -> bool:
        return self.membership is not None

    @property
    def trustworthy(self) -> bool:
        """Point-in-time AND enough of the index priced to mean it."""
        return (self.point_in_time and self.coverage is not None
                and self.coverage.mean_priced_share >= PIT_MIN_PRICED_SHARE)

    def mask(self, index: pd.DatetimeIndex, universe: list[str]) -> pd.DataFrame | None:
        if self.membership is None or not self.ranked:
            return None
        return self.membership.eligibility(index, universe)

    def measure(self, signal_dates, prices: pd.DataFrame) -> None:
        if self.membership is not None and signal_dates is not None:
            self.coverage = self.membership.coverage(signal_dates, prices)

    def why_not_trustworthy(self) -> str | None:
        if not self.point_in_time:
            return self.fallback_reason
        if self.coverage is None or self.coverage.dates == 0:
            return "no rebalance date could be checked against the membership history"
        if self.coverage.mean_priced_share < PIT_MIN_PRICED_SHARE:
            return (f"only {self.coverage.mean_priced_share:.0%} of members at the "
                    f"average rebalance have usable free price history")
        return None

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "point_in_time": self.point_in_time,
            "trustworthy": self.trustworthy,
            "fallback_reason": self.why_not_trustworthy(),
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "source": self.membership.provenance() if self.membership else None,
        }


def membership_for(spec: StrategySpec) -> MembershipUse:
    ranked = spec.selection.static_allocation is None
    if spec.universe.preset not in INDEX_PRESETS:
        return MembershipUse(spec.universe.preset, ranked)
    resolved = resolve_preset(spec.universe.preset, spec.window.start, spec.window.end)
    return MembershipUse(spec.universe.preset, ranked, resolved.membership,
                         resolved.fallback_reason)


def index_universe_finding(spec: StrategySpec, use: MembershipUse) -> Finding:
    """The survivorship disclosure for preset universes (check id kept as
    survivorship_universe so existing reports and UI keep working)."""
    if spec.universe.preset == "sector_etfs":
        return Finding(
            check="survivorship_universe", severity="note",
            name="Fixed ETF list",
            message=(
                "The 11 SPDR sector ETFs are a fixed list with no index membership "
                "to reconstruct. Any fund that launched mid-window is flagged in "
                "the data findings."
            ),
            explainer="survivorship-bias",
        )
    name = INDEX_NAMES.get(spec.universe.preset, spec.universe.preset)
    if not use.ranked:
        return Finding(
            check="survivorship_universe", severity="note",
            name="Fixed allocation",
            message=(
                f"This is a fixed allocation, so the {name} list never picks a "
                f"stock and survivorship in that list does not apply."
            ),
            explainer="survivorship-bias",
        )
    if not use.point_in_time:
        return Finding(
            check="survivorship_universe", severity="warning",
            name="Index list is today's members",
            message=(
                f"Point-in-time membership isn't available for this run: "
                f"{use.fallback_reason}. So the universe is TODAY's {name}, which "
                f"includes companies that only joined after their run-ups, and a "
                f"ranked strategy buys exactly those. The result leans optimistic, "
                f"and it can't earn a green verdict."
            ),
            evidence=use.to_dict(),
            explainer="survivorship-bias",
        )
    cov = use.coverage
    share = cov.mean_priced_share if cov else 0.0
    unpriced = cov.mean_unpriced if cov else 0.0
    source = use.membership.provenance()
    if not use.trustworthy:
        detail = (
            f" (about {unpriced:.0f} missing per rebalance, lowest "
            f"{cov.min_priced_share:.0%} on {cov.min_date})" if cov and cov.dates else ""
        )
        return Finding(
            check="survivorship_universe", severity="warning",
            name="Point-in-time index, too thin to trust",
            message=(
                f"Each rebalance picked only from that day's {name} members, but "
                f"{use.why_not_trustworthy()}{detail}. The missing ones are mostly "
                f"acquired or delisted companies, and leaving out the failures "
                f"flatters the result. It can't earn a green verdict on this coverage."
            ),
            evidence=use.to_dict(),
            explainer="survivorship-bias",
        )
    if share < PIT_CLEAN_PRICED_SHARE:
        return Finding(
            check="survivorship_universe", severity="warning",
            name="Point-in-time index, with gaps",
            message=(
                f"Each rebalance picked only from that day's {name} members (dated "
                f"membership from {source['coverage_start']}). {share:.0%} of those "
                f"members had usable free price history; about {unpriced:.0f} per "
                f"rebalance did not (lowest {cov.min_priced_share:.0%} on "
                f"{cov.min_date}), mostly acquired or delisted companies. Leaving "
                f"out the failures flatters early years a little."
            ),
            evidence=use.to_dict(),
            explainer="survivorship-bias",
        )
    return Finding(
        check="survivorship_universe", severity="note",
        name="Point-in-time index membership",
        message=(
            f"Each rebalance picked only from that day's {name} members (dated "
            f"membership from {source['coverage_start']}), and {share:.0%} of "
            f"them had usable price history. Companies that joined the index "
            f"later were not available before they joined."
        ),
        evidence=use.to_dict(),
        explainer="survivorship-bias",
    )


def membership_verdict(use: MembershipUse) -> dict | None:
    """The amber cap for a ranked index run without trustworthy point-in-time
    membership, or None when the universe is clean (or not an index)."""
    if not (use.index_preset and use.ranked) or use.trustworthy:
        return None
    if use.point_in_time:
        return {
            "color": "amber", "icon": "▲",
            "headline": "Beats the benchmark — but too much of the index is missing.",
            "sub": (
                f"The universe is point-in-time, but {use.why_not_trustworthy()} "
                f"(mostly acquired or delisted companies). Leaving out the "
                f"failures flatters the result."
            ),
        }
    return {
        "color": "amber", "icon": "▲",
        "headline": "Beats the benchmark — but on today's index membership.",
        "sub": (
            f"Point-in-time membership isn't available here "
            f"({use.why_not_trustworthy()}), so the universe is today's "
            f"constituents, including the winners that only joined after their "
            f"run-ups. A ranked strategy on that list can't earn a green verdict."
        ),
    }


def _neutral_run(spec: StrategySpec) -> tuple[pd.Series, MembershipUse]:
    """The identical rules on the S&P 500 as it stood at each rebalance —
    point-in-time when the window allows, today's list (flagged) when not."""
    neutral_spec = copy.deepcopy(spec)
    neutral_spec.universe.preset = NEUTRAL_PRESET
    neutral_spec.universe.resolved_tickers = None
    neutral_spec.universe.watchlist_file = None
    use = membership_for(neutral_spec)
    resolved = resolve_preset(NEUTRAL_PRESET, spec.window.start, spec.window.end)
    panel = get_panel(
        resolved.tickers + BENCHMARK_TICKERS[spec.benchmark] + ["SPY"],
        start=spec.window.start, end=spec.window.end,
    )
    universe = [t for t in resolved.tickers if t in panel.prices.columns]
    res = run(neutral_spec, panel.prices, universe,
              membership=use.mask(panel.prices.index, universe))
    use.measure(res.signal_dates, panel.prices)
    return res.net_returns[NEUTRAL_BPS], use


def check_selection_bias(
    spec: StrategySpec, net5: pd.Series, use: MembershipUse | None = None,
    progress=None,
) -> tuple[Finding, dict | None]:
    """→ (finding, neutral_rerun_summary | None)."""
    if spec.universe.preset != "user_watchlist":
        return index_universe_finding(spec, use or membership_for(spec)), None
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

    try:
        neutral_net5, neutral = _neutral_run(spec)
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
    clean = neutral.trustworthy
    rerun = {
        "universe": NEUTRAL_PRESET,
        "net5_total_return": neutral_total,
        "user_net5_total_return": user_total,
        "wealth_ratio": ratio,
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "point_in_time": neutral.point_in_time,
        "trustworthy": clean,
        "membership": neutral.to_dict(),
    }
    neutral_label = ("the S&P 500 as it stood at each rebalance" if clean
                     else "today's S&P 500 members")

    if ratio > FATAL_RATIO:
        tail = (
            "" if clean else
            f" And that comparison is itself flattered — {neutral.why_not_trustworthy()}, "
            f"so the S&P number is inflated and the real gap is likely wider."
        )
        finding = Finding(
            check="selection_bias", severity="fatal",
            name="Selection bias detected",
            message=(
                f"Your universe is {len(tickers)} hand-picked tickers, and you "
                f"picked them knowing how they performed. On this list the "
                f"strategy returned {user_total:+.0%} (net of 5 bps); on "
                f"{neutral_label} the identical rules returned "
                f"{neutral_total:+.0%} — a {ratio:.1f}× wealth gap. The "
                f"difference is hindsight, not skill. This is the same trap that "
                f"produces fake +1650% backtests.{tail}"
            ),
            evidence=rerun,
            action="Re-run this spec on the full S&P 500",
            explainer="selection-bias",
        )
    elif not clean:
        finding = Finding(
            check="selection_bias", severity="warning",
            name="Hand-picked universe (neutral check unverified)",
            message=(
                f"Your universe is {len(tickers)} hand-picked tickers — chosen "
                f"knowing how they performed. Identical rules, same dates: your "
                f"list {user_total:+.0%} vs today's S&P 500 members "
                f"{neutral_total:+.0%} (net of 5 bps). That comparison can't clear "
                f"your list: {neutral.why_not_trustworthy()}, so it is TODAY's "
                f"index, including companies that joined after their run-ups, and "
                f"its number is inflated. Treat the headline as hindsight."
            ),
            evidence=rerun,
            action="Re-run this spec on the full S&P 500",
            explainer="selection-bias",
        )
    else:
        tail = (
            "Over these exact dates the point-in-time S&P 500 did as well or "
            "better — hand-picking added nothing you could bank on."
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
                f"your list {user_total:+.0%} vs the S&P 500 as it stood at "
                f"each rebalance {neutral_total:+.0%} (net of 5 bps). {tail}"
            ),
            evidence=rerun,
            action="Re-run this spec on the full S&P 500",
            explainer="selection-bias",
        )
    return finding, rerun
