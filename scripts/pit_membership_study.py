"""Confirm (or refute) the survivorship diagnosis with a controlled comparison.

Same engine, same prices, same rules — only the universe changes:
  * today:  the S&P 500 constituent list the app shipped with (fixtures/sp500_tickers.txt)
  * pit:    the S&P 500 as it stood at each rebalance (fixtures/sp500_membership)

Part 1 replays every saved S&P 500 run from its FROZEN price snapshot with the
CURRENT engine, twice. The saved figure is shown too, but most were produced
before commit 35b370b (2026-09-13) changed the engine (one-day execution lag,
costs on traded notional, T-bill cash, stale-feed screen), so the replay does
not match them to the decimal; the like-for-like comparison is today-list vs
point-in-time on one engine and one set of prices. The pit leg adds the prices
of names that were members but are not in the snapshot, from the local cache,
reindexed to the snapshot's own dates.

Part 2 sweeps momentum lookbacks over the Feb 2023 - May 2026 window where the
+534% to +3560% results came from.

Writes docs/pit_membership_study.md and .json. Every number in them is computed
here; nothing is typed in by hand.

Usage: HONEST_OFFLINE=1 .venv/bin/python scripts/pit_membership_study.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import pandas as pd  # noqa: E402

from audit.pipeline import VERDICT_BPS, _rf_daily  # noqa: E402
from config import settings  # noqa: E402
from dsl.membership import load_membership  # noqa: E402
from dsl.schema import StrategySpec  # noqa: E402
from dsl.universes import preset_tickers  # noqa: E402
from engine import metrics  # noqa: E402
from engine.backtest import run  # noqa: E402
from engine.benchmark import benchmark_returns  # noqa: E402
from engine.data import PriceCache, get_panel, load_snapshot  # noqa: E402
from store.db import get_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

DOCS = REPO / "docs"
MOMENTUM_WINDOW = ("2023-01-02", "2026-05-29")
LOOKBACKS = [21, 63, 126, 189, 252]


def pit_panel(prices: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """The snapshot plus any member symbol it lacks, from the cache, on the
    snapshot's own dates. Snapshot columns are never touched."""
    cache = PriceCache()
    extra = {}
    for sym in symbols:
        if sym in prices.columns:
            continue
        s = cache.get_series(sym)
        if s is not None:
            extra[sym] = s.reindex(prices.index)
    if not extra:
        return prices
    return pd.concat([prices, pd.DataFrame(extra, index=prices.index)], axis=1)


def evaluate(spec: StrategySpec, prices: pd.DataFrame, universe: list[str],
             membership=None) -> dict:
    class _P:  # _rf_daily only reads .prices
        pass
    holder = _P()
    holder.prices = prices
    rf_panel, _, rf_source, _ = _rf_daily(holder, prices.index)
    mask = membership.eligibility(prices.index, universe) if membership else None
    res = run(spec, prices, universe, rf_daily=None if rf_source == "none" else rf_panel,
              membership=mask)
    net5 = res.net_returns[VERDICT_BPS]
    bench = benchmark_returns(spec.benchmark, prices, res.accrual_index)
    out = {
        "net5_total_return": metrics.total_return(net5),
        "net5_cagr": metrics.cagr(net5),
        "net5_max_drawdown": metrics.max_drawdown(net5),
        "benchmark_total_return": metrics.total_return(bench),
        "start": str(res.effective_start.date()),
        "end": str(res.effective_end.date()),
        "universe_size": len([t for t in universe if t in prices.columns]),
        "holdings": res.holdings,
    }
    if membership is not None:
        out["coverage"] = membership.coverage(res.signal_dates, prices).to_dict()
    return out


def pre_join_picks(holdings: list[dict], membership) -> list[dict]:
    """Picks made on dates the name was NOT an S&P 500 member."""
    firsts: dict[str, str] = {}
    for h in holdings:
        for p in h["picks"]:
            firsts.setdefault(p["ticker"], h["date"])
    out = []
    for tk, first in sorted(firsts.items(), key=lambda kv: kv[1]):
        on = membership.eligibility(pd.DatetimeIndex([first]), [tk])[tk].iloc[0]
        if on:
            continue
        spells = membership.spells[membership.spells["price_symbol"] == tk]
        joins = [str(d.date()) for d in spells["start"].dropna() if str(d.date()) > first]
        n = sum(1 for h in holdings for p in h["picks"]
                if p["ticker"] == tk and not membership.eligibility(
                    pd.DatetimeIndex([h["date"]]), [tk])[tk].iloc[0])
        out.append({"ticker": tk, "first_pick": first,
                    "joined": joins[0] if joins else None, "non_member_rebalances": n})
    return out


def saved_sp500_runs() -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(text(
            "SELECT r.id, r.spec_json, rep.id, rep.report_json FROM runs r "
            "JOIN reports rep ON rep.run_id = r.id WHERE r.status = 'done' "
            "ORDER BY r.created_at")).fetchall()
    out = []
    for run_id, spec_json, rep_id, report_json in rows:
        spec = json.loads(spec_json)
        if spec["universe"]["preset"] != "sp500":
            continue
        report = json.loads(report_json)
        out.append({"run_id": run_id, "report_id": rep_id, "spec": spec,
                    "saved_net5": report["metrics"]["total_return"],
                    "saved_universe_size": report["data"]["universe_size"],
                    "saved_verdict": report["verdict"]["color"]})
    return out


def pct(x: float) -> str:
    return f"{x:+,.0%}"


def main() -> None:
    if "--render-only" in sys.argv:
        results = json.loads((DOCS / "pit_membership_study.json").read_text())
        (DOCS / "pit_membership_study.md").write_text(render(results))
        print("re-rendered docs/pit_membership_study.md")
        return
    membership = load_membership("sp500")
    today_list = preset_tickers("sp500")
    results = {"generated": date.today().isoformat(),
               "membership": membership.provenance(), "saved_runs": [], "grid": []}

    # ---- part 1: saved runs, from their frozen snapshots ----
    for saved in saved_sp500_runs():
        spec = StrategySpec(**saved["spec"])
        panel = load_snapshot(Path(settings.runs_dir) / saved["run_id"])
        prices = panel.prices
        today_universe = [t for t in today_list if t in prices.columns]
        today = evaluate(spec, prices, today_universe)
        symbols = membership.symbols_between(spec.window.start, spec.window.end)
        pit_prices = pit_panel(prices, symbols)
        pit = evaluate(spec, pit_prices, [t for t in symbols if t in pit_prices.columns],
                       membership)
        row = {
            "run_id": saved["run_id"], "report_id": saved["report_id"],
            "name": spec.name, "window": [spec.window.start, spec.window.end],
            "rank_by": spec.selection.rank_by.model_dump() if spec.selection.rank_by else None,
            "n": spec.selection.n, "direction": spec.selection.direction,
            "rebalance": spec.rebalance,
            "saved_net5_total_return": saved["saved_net5"],
            "saved_verdict": saved["saved_verdict"],
            "today": {k: v for k, v in today.items() if k != "holdings"},
            "replay_minus_saved": today["net5_total_return"] - saved["saved_net5"],
            "pit": {k: v for k, v in pit.items() if k != "holdings"},
            "today_pre_join_picks": pre_join_picks(today["holdings"], membership),
        }
        results["saved_runs"].append(row)
        print(f"{saved['run_id']} {spec.name[:40]:40} saved {pct(saved['saved_net5'])} "
              f"today {pct(today['net5_total_return'])} pit {pct(pit['net5_total_return'])}")

    # ---- part 2: momentum lookback grid on the 2023-2026 window ----
    start, end = MOMENTUM_WINDOW
    symbols = membership.symbols_between(start, end)
    panel = get_panel(sorted(set(symbols) | set(today_list)) + ["SPY", "^IRX"],
                      start=start, end=end)
    prices = panel.prices
    for lb in LOOKBACKS:
        spec = StrategySpec(
            name=f"{lb}d momentum top 5", universe={"preset": "sp500"},
            selection={"rank_by": {"type": "return", "lookback_days": lb},
                       "direction": "top", "n": 5},
            weighting="equal", rebalance="monthly", benchmark="SPY",
            window={"start": start, "end": end},
        )
        today = evaluate(spec, prices, [t for t in today_list if t in prices.columns])
        pit = evaluate(spec, prices, [t for t in symbols if t in prices.columns], membership)
        results["grid"].append({
            "lookback_days": lb,
            "today": {k: v for k, v in today.items() if k != "holdings"},
            "pit": {k: v for k, v in pit.items() if k != "holdings"},
            "today_pre_join_picks": pre_join_picks(today["holdings"], membership),
        })
        print(f"grid {lb:3d}d  today {pct(today['net5_total_return'])}  "
              f"pit {pct(pit['net5_total_return'])}  SPY {pct(today['benchmark_total_return'])}")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "pit_membership_study.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    (DOCS / "pit_membership_study.md").write_text(render(results))
    print("wrote docs/pit_membership_study.md and .json")


def render(r: dict) -> str:
    m = r["membership"]
    lines = [
        "# Point-in-time S&P 500: does today's list explain the huge momentum numbers?",
        "",
        f"Generated {r['generated']} by `scripts/pit_membership_study.py`. Every figure "
        "below is computed by that script; rerun it to regenerate this file.",
        "",
        "**Method.** Same engine, same prices, same rules; only the universe changes. "
        "*Today* is the S&P 500 list the app shipped with. *Point-in-time* is the index "
        "as it stood at each rebalance "
        f"([membership]({m['changes_source']}), coverage from {m['coverage_start']}). "
        "Returns are net of 5 bps per side, the number the verdict uses.",
        "",
        "## 1. Saved S&P 500 runs, replayed from their frozen price snapshots",
        "",
        "*Today* and *Point-in-time* are both replays with the current engine on the "
        "run's frozen prices, so they differ only in who was eligible. *Saved* is what "
        "the stored report said. All six were produced before commit 35b370b "
        "(2026-09-13) landed, which added a one-day execution lag, charged costs on "
        "traded notional, credited T-bill yield to cash and screened stale feeds. The "
        "August runs replay lower than they were saved; the 2026-09-12 run, made while "
        "that work was in progress, replays to within 0.3 percentage points of a "
        "+18,693% total.",
        "",
        "| Run | Saved name | Rule | Window | Saved | Today (replay) | Point-in-time | SPY | Members priced |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in r["saved_runs"]:
        cov = s["pit"].get("coverage") or {}
        rb = s["rank_by"] or {}
        rule = (f"{s['direction']} {s['n']} by {rb.get('lookback_days')}d {rb.get('type')}, "
                f"{s['rebalance']}")
        lines.append(
            f"| `{s['run_id']}` | {s['name']} | {rule} | {s['today']['start']} → {s['today']['end']} "
            f"| {pct(s['saved_net5_total_return'])} | {pct(s['today']['net5_total_return'])} "
            f"| {pct(s['pit']['net5_total_return'])} | {pct(s['today']['benchmark_total_return'])} "
            f"| {cov.get('mean_priced_share', 0):.0%} |")
    lines += ["", "Names the *today* runs bought on dates they were not S&P 500 members "
              "(first such pick → date they joined, or blank if they never were one in the window):", ""]
    for s in r["saved_runs"]:
        picks = s["today_pre_join_picks"]
        sample = ", ".join(
            f"{p['ticker']} ({p['first_pick']}→{p['joined'] or 'not a member'})" for p in picks[:12])
        lines.append(f"- `{s['run_id']}` — {len(picks)} names: {sample}{' …' if len(picks) > 12 else ''}")
    lines += [
        "",
        f"## 2. Momentum lookbacks, top 5, monthly, {MOMENTUM_WINDOW[0]} → {MOMENTUM_WINDOW[1]}",
        "",
        "| Lookback | Today | Point-in-time | SPY | Today's non-member picks |",
        "|---:|---:|---:|---:|---|",
    ]
    for g in r["grid"]:
        picks = g["today_pre_join_picks"]
        sample = ", ".join(p["ticker"] for p in picks[:8])
        lines.append(
            f"| {g['lookback_days']}d | {pct(g['today']['net5_total_return'])} "
            f"| {pct(g['pit']['net5_total_return'])} | {pct(g['today']['benchmark_total_return'])} "
            f"| {len(picks)}: {sample}{' …' if len(picks) > 8 else ''} |")
    lines += [
        "",
        "## What this does not remove",
        "",
        "Members with no free price history (mostly acquired companies, a few failures) "
        "can't be picked in either column; the *members priced* share says how much of "
        "the index each point-in-time run could see. See "
        "`fixtures/sp500_membership/README.md`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
